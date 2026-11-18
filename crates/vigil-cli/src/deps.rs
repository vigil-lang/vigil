//! this module resolves the libraries of a project: finding the
//! directories of `[dependencies]`, fetching the git ones and loading their
//! modules, under the dependency name (`parse` of `netx` is `netx.parse`).
//!
//! A library with a `vigil.toml` has its sources in `src/` and its own
//! dependencies. Without one, every `.vg` file under it is a module. All
//! dependency names share one namespace, so two different libraries with
//! the same name are an error. Only the project may choose `std`.
//!
//! The commits of git dependencies are kept in `vigil.lock`, so builds are
//! the same and work offline, until the entry changes or `vigil update`.
//! Checkouts go in a shared cache, see [`cache_root`].

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use vigil_diagnostics::{Diagnostic, Diagnostics, FileId, SourceMap, Span, codes};
use vigil_target::{Dependency, GitRef, MANIFEST_NAME, Source};

use crate::project::{self, SOURCE_DIR, SOURCE_EXTENSION};

/// the file with the commits of the git dependencies, next to the manifest
pub const LOCK_NAME: &str = "vigil.lock";

/// a module: its file and its module path
pub type Module = (FileId, Vec<String>);

/// the loaded libraries of a project
#[derive(Debug, Default)]
pub struct Libraries {
    /// the standard library, when a dependency replaces the built-in one
    pub std: Option<Vec<Module>>,
    /// the modules of the other libraries
    pub modules: Vec<Module>,
    /// each library, sorted by name, for `vigil check -v` and `vigil update`
    pub resolved: Vec<Resolved>,
}

/// a resolved library
#[derive(Debug)]
pub struct Resolved {
    pub name: String,
    /// the directory as written, or the repository and the commit
    pub origin: String,
}

/// what a dependency is, to compare two declarations of a name
#[derive(Clone, PartialEq, Eq)]
enum Identity {
    Path(PathBuf),
    Git { url: String, commit: String },
}

/// the main function of the module: resolving the `dependencies` of the
/// project at `root`. `None` after an error, which is in `diags`
///
/// A dependency may not have the name of one of `project_modules`. With
/// `update`, git dependencies are resolved again instead of read from
/// `vigil.lock`.
pub fn resolve(
    root: &Path,
    dependencies: &[Dependency],
    project_modules: &[Module],
    update: bool,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<Libraries> {
    let errors_before = diags.error_count();
    let lock_path = root.join(LOCK_NAME);
    let lock = read_lock(&lock_path, diags);
    let mut locked: Vec<LockEntry> = Vec::new();
    let mut libraries = Libraries::default();
    let mut seen: HashMap<String, (Identity, Span)> = HashMap::new();
    let mut queue: VecDeque<(Dependency, PathBuf, bool)> = dependencies
        .iter()
        .map(|d| (d.clone(), root.to_path_buf(), true))
        .collect();

    while let Some((dep, base, of_project)) = queue.pop_front() {
        if dep.name == "std" && !of_project {
            diags.push(
                Diagnostic::warning("a library cannot choose the standard library")
                    .with_code(codes::IGNORED_DEPENDENCY)
                    .with_primary(dep.span, "ignored")
                    .with_note("only the project being built has a `std` dependency"),
            );
            continue;
        }
        let (dir, identity, origin) = match &dep.source {
            Source::Path(path) => {
                let dir = base.join(path);
                match dir.canonicalize() {
                    Ok(canonical) if canonical.is_dir() => {
                        (canonical.clone(), Identity::Path(canonical), path.clone())
                    }
                    _ => {
                        diags.push(
                            Diagnostic::error(format!(
                                "the path of dependency `{}` is not a directory",
                                dep.name
                            ))
                            .with_code(codes::INVALID_DEPENDENCY)
                            .with_primary(dep.span, format!("no directory at `{}`", dir.display()))
                            .with_note("a `path` is relative to the directory of its `vigil.toml`"),
                        );
                        continue;
                    }
                }
            }
            Source::Git { url, reference } => {
                let spec = reference.spec();
                let from_lock = (!update)
                    .then(|| lock.iter().find(|e| e.url == *url && e.reference == spec))
                    .flatten()
                    .map(|e| e.commit.clone());
                let commit = match from_lock.map_or_else(|| git::resolve(url, reference), Ok) {
                    Ok(commit) => commit,
                    Err(err) => {
                        diags.push(unavailable(&dep, url, err));
                        continue;
                    }
                };
                let dir = match git::checkout(url, &commit) {
                    Ok(dir) => dir,
                    Err(err) => {
                        diags.push(unavailable(&dep, url, err));
                        continue;
                    }
                };
                locked.push(LockEntry {
                    url: url.clone(),
                    reference: spec,
                    commit: commit.clone(),
                });
                let origin = format!("{url} @ {}", &commit[..12]);
                (
                    dir,
                    Identity::Git {
                        url: url.clone(),
                        commit,
                    },
                    origin,
                )
            }
        };

        if let Some((first, first_span)) = seen.get(&dep.name) {
            if *first != identity {
                diags.push(
                    Diagnostic::error(format!("two different libraries are named `{}`", dep.name))
                        .with_code(codes::INVALID_DEPENDENCY)
                        .with_primary(dep.span, "this one")
                        .with_secondary(*first_span, "and this one")
                        .with_note("the modules of both would be imported by the same name"),
                );
            }
            continue;
        }
        seen.insert(dep.name.clone(), (identity, dep.span));

        let source_dir = if dir.join(MANIFEST_NAME).is_file() {
            let manifest = project::load_manifest(&dir.join(MANIFEST_NAME), sources, diags)?;
            queue.extend(
                manifest
                    .dependencies
                    .into_iter()
                    .map(|d| (d, dir.clone(), false)),
            );
            dir.join(SOURCE_DIR)
        } else {
            dir.clone()
        };
        let modules = load_library(&dep, &source_dir, sources, diags)?;
        if dep.name == "std" {
            libraries.std = Some(modules);
        } else {
            libraries.modules.extend(modules);
        }
        libraries.resolved.push(Resolved {
            name: dep.name.clone(),
            origin,
        });
    }

    // a project module with the name of a library would share its
    // namespace, and the analysis would silently mix them up
    let project_names: BTreeSet<&str> = project_modules
        .iter()
        .filter_map(|(_, path)| path.first().map(String::as_str))
        .collect();
    for (name, (_, span)) in &seen {
        if project_names.contains(name.as_str()) {
            diags.push(
                Diagnostic::error(format!(
                    "the project has a module `{name}`, which is also the name of a dependency"
                ))
                .with_code(codes::INVALID_DEPENDENCY)
                .with_primary(*span, "the dependency")
                .with_help(format!(
                    "rename `src/{name}.{SOURCE_EXTENSION}` or the dependency"
                )),
            );
        }
    }

    if diags.error_count() > errors_before {
        return None;
    }
    write_lock(&lock_path, &lock, locked, diags);
    libraries.resolved.sort_by(|a, b| a.name.cmp(&b.name));
    Some(libraries)
}

fn unavailable(dep: &Dependency, url: &str, err: String) -> Diagnostic {
    Diagnostic::error(format!("could not fetch `{}` from `{url}`", dep.name))
        .with_code(codes::DEPENDENCY_UNAVAILABLE)
        .with_primary(dep.span, "this dependency")
        .with_note(err)
        .with_help(
            "check the URL and the reference; a private repository needs credentials git can use",
        )
}

/// loading every `.vg` file under `dir`, under the dependency name. A
/// library without sources is an error here, otherwise only its imports
/// would fail.
fn load_library(
    dep: &Dependency,
    dir: &Path,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<Vec<Module>> {
    let mut paths = Vec::new();
    if dir.is_dir()
        && let Err(err) = project::collect_sources(dir, &mut paths)
    {
        diags.push(project::io_error(dir, &err));
        return None;
    }
    if paths.is_empty() {
        diags.push(
            Diagnostic::error(format!(
                "the library `{}` has no `.{SOURCE_EXTENSION}` files in `{}`",
                dep.name,
                dir.display()
            ))
            .with_code(codes::INVALID_DEPENDENCY)
            .with_primary(dep.span, "this dependency"),
        );
        return None;
    }
    // sorting so that two runs give the same file ids
    paths.sort();
    let mut modules = Vec::with_capacity(paths.len());
    for path in &paths {
        let module = std::iter::once(dep.name.clone())
            .chain(
                path.strip_prefix(dir)
                    .unwrap_or(path)
                    .with_extension("")
                    .iter()
                    .map(|s| s.to_string_lossy().into_owned()),
            )
            .collect();
        modules.push((project::load_file(path, sources, diags)?, module));
    }
    Some(modules)
}

// vigil.lock
// -------------------------

#[derive(Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct LockFile {
    version: u32,
    #[serde(default)]
    git: Vec<LockEntry>,
}

/// the commit a git dependency resolved to
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct LockEntry {
    url: String,
    /// the reference as in the manifest ([`GitRef::spec`])
    reference: String,
    commit: String,
}

const LOCK_VERSION: u32 = 1;
const LOCK_HEADER: &str = "# The commits of the git dependencies, written by `vigil`; \
                           `vigil update` resolves them again.\n";

/// reading the entries of `vigil.lock`, none if there is no file. An
/// invalid file is only a warning and is taken as empty, so everything is
/// resolved again. An entry whose commit is not a full hash (a hand-edited
/// lock) is dropped with the same warning, so only its dependency is
/// resolved again.
fn read_lock(path: &Path, diags: &mut Diagnostics) -> Vec<LockEntry> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let invalid = |diags: &mut Diagnostics| {
        diags.push(
            Diagnostic::warning(format!("ignoring `{}`, which is not valid", path.display()))
                .with_code(codes::IGNORED_DEPENDENCY)
                .with_note("the git dependencies are resolved again"),
        );
    };
    match toml::from_str::<LockFile>(&text) {
        Ok(lock) if lock.version == LOCK_VERSION => {
            let total = lock.git.len();
            let entries: Vec<LockEntry> = lock
                .git
                .into_iter()
                .filter(|e| is_commit(&e.commit))
                .collect();
            if entries.len() < total {
                invalid(diags);
            }
            entries
        }
        Ok(_) | Err(_) => {
            invalid(diags);
            Vec::new()
        }
    }
}

/// whether `commit` is a full commit hash, 40 hexadecimal digits, as git
/// gives and as `vigil.toml` requires of `rev`
fn is_commit(commit: &str) -> bool {
    commit.len() == 40 && commit.chars().all(|c| c.is_ascii_hexdigit())
}

/// writing `vigil.lock` with the entries of this resolution, if they
/// changed. Unused entries are dropped. We do not create a lock file without
/// git dependencies, but an existing one is emptied.
fn write_lock(path: &Path, old: &[LockEntry], mut new: Vec<LockEntry>, diags: &mut Diagnostics) {
    new.sort();
    new.dedup();
    let mut sorted_old = old.to_vec();
    sorted_old.sort();
    if new == sorted_old && (path.exists() || new.is_empty()) {
        return;
    }
    let lock = LockFile {
        version: LOCK_VERSION,
        git: new,
    };
    let text = match toml::to_string(&lock) {
        Ok(text) => format!("{LOCK_HEADER}{text}"),
        Err(_) => return,
    };
    if let Err(err) = std::fs::write(path, text) {
        diags.push(
            Diagnostic::warning(format!("could not write `{}`", path.display()))
                .with_code(codes::IO_ERROR)
                .with_note(err.to_string()),
        );
    }
}

// git
// -------------------------

/// the cache of git checkouts: `$VIGIL_HOME`, or `vigil` in
/// `$XDG_CACHE_HOME`, or on Windows `vigil` in `%LOCALAPPDATA%`, or
/// `~/.cache/vigil`
pub fn cache_root() -> Result<PathBuf, String> {
    cache_root_in(|name| std::env::var_os(name), cfg!(windows))
}

/// [`cache_root`] for the environment `var` (empty values count as unset)
fn cache_root_in(
    var: impl Fn(&str) -> Option<std::ffi::OsString>,
    windows: bool,
) -> Result<PathBuf, String> {
    let var = |name| var(name).filter(|v| !v.is_empty());
    if let Some(home) = var("VIGIL_HOME") {
        return Ok(PathBuf::from(home));
    }
    if let Some(cache) = var("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(cache).join("vigil"));
    }
    // Windows has no `HOME` by default, and `%LOCALAPPDATA%` is where its
    // programs keep caches
    if windows && let Some(local) = var("LOCALAPPDATA") {
        return Ok(PathBuf::from(local).join("vigil"));
    }
    var("HOME")
        .map(|home| PathBuf::from(home).join(".cache").join("vigil"))
        .ok_or_else(|| "no cache directory: set VIGIL_HOME".into())
}

mod git {
    use super::*;

    /// running `git args` in `dir`, giving its output. We tell git never to
    /// prompt, so a private or missing repository is an error and not a wait
    /// for a password. Stored credentials still work.
    fn run(args: &[&str], dir: Option<&Path>) -> Result<String, String> {
        let mut command = Command::new("git");
        if let Some(dir) = dir {
            command.arg("-C").arg(dir);
        }
        let output = command
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_ASKPASS", "true")
            .env("SSH_ASKPASS", "true")
            .env("GCM_INTERACTIVE", "never")
            .output()
            .map_err(|e| format!("could not run `git`: {e}"))?;
        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).into_owned())
        } else {
            Err(format!(
                "`git {}` failed: {}",
                args.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            ))
        }
    }

    /// asking the remote for the commit of `reference`. A `rev` is already
    /// a commit.
    pub(super) fn resolve(url: &str, reference: &GitRef) -> Result<String, String> {
        let (patterns, what) = match reference {
            GitRef::Rev(rev) => return Ok(rev.clone()),
            GitRef::DefaultBranch => (vec!["HEAD".to_string()], "no default branch".to_string()),
            GitRef::Branch(b) => (vec![format!("refs/heads/{b}")], format!("no branch `{b}`")),
            // an annotated tag is an object, and `^{}` is its commit, which
            // is what we check out
            GitRef::Tag(t) => (
                vec![format!("refs/tags/{t}"), format!("refs/tags/{t}^{{}}")],
                format!("no tag `{t}`"),
            ),
        };
        let mut args = vec!["ls-remote", url];
        args.extend(patterns.iter().map(String::as_str));
        let out = run(&args, None)?;
        let lines: Vec<(&str, &str)> = out.lines().filter_map(|l| l.split_once('\t')).collect();
        lines
            .iter()
            .find(|(_, r)| r.ends_with("^{}"))
            .or_else(|| lines.first())
            .map(|(commit, _)| commit.to_string())
            .ok_or(what)
    }

    /// the cache directory of `commit` of `url`, fetching it if needed
    ///
    /// We check out in a temporary directory and rename it, so a directory
    /// in the cache is always complete. An interrupted fetch only leaves the
    /// temporary one, which the next fetch replaces.
    pub(super) fn checkout(url: &str, commit: &str) -> Result<PathBuf, String> {
        let repo =
            super::cache_root()?
                .join("git")
                .join(format!("{}-{:016x}", slug(url), fnv1a(url)));
        let dir = repo.join(commit);
        if dir.is_dir() {
            return Ok(dir);
        }
        let tmp = repo.join(format!("{commit}.tmp{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let fetched = (|| {
            run(&["init", "-q"], Some(&tmp))?;
            // fetching only the commit is enough (GitHub allows it). Other
            // servers are asked for every branch and tag
            if run(&["fetch", "-q", "--depth", "1", url, commit], Some(&tmp)).is_err() {
                run(
                    &[
                        "fetch",
                        "-q",
                        url,
                        "+refs/heads/*:refs/remotes/origin/*",
                        "+refs/tags/*:refs/tags/*",
                    ],
                    Some(&tmp),
                )?;
            }
            run(
                &["-c", "advice.detachedHead=false", "checkout", "-q", commit],
                Some(&tmp),
            )
            .map_err(|_| format!("no commit {commit} in {url}"))
        })();
        if let Err(err) = fetched {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(err);
        }
        let _ = std::fs::remove_dir_all(tmp.join(".git"));
        if std::fs::rename(&tmp, &dir).is_err() {
            // another `vigil` put the same commit here meanwhile
            let _ = std::fs::remove_dir_all(&tmp);
            if !dir.is_dir() {
                return Err(format!("could not create `{}`", dir.display()));
            }
        }
        Ok(dir)
    }

    /// the last segment of a URL without `.git`, a readable part of its
    /// cache directory name
    fn slug(url: &str) -> String {
        let last = url
            .trim_end_matches('/')
            .rsplit(['/', ':'])
            .next()
            .unwrap_or("repo");
        let name: String = last
            .trim_end_matches(".git")
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        if name.is_empty() { "repo".into() } else { name }
    }

    /// FNV-1a. We do not use the std hasher because it may change between
    /// versions, and the cache names must not
    fn fnv1a(text: &str) -> u64 {
        text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vigil_target::parse_manifest;

    /// a fresh directory for one test
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vigil-deps-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    /// resolving the project at `root`, with the diagnostics as text
    fn resolve_at(root: &Path, update: bool) -> (Option<Libraries>, String) {
        let mut sources = SourceMap::new();
        let mut diags = Diagnostics::new();
        let text = std::fs::read_to_string(root.join(MANIFEST_NAME)).unwrap();
        let file = sources.add(MANIFEST_NAME, text.clone());
        let (manifest, manifest_diags) = parse_manifest(file, &text);
        diags.extend(manifest_diags);
        let manifest = manifest.expect("a valid manifest");
        let libraries = resolve(
            root,
            &manifest.dependencies,
            &[],
            update,
            &mut sources,
            &mut diags,
        );
        let mut out = Vec::new();
        vigil_diagnostics::Renderer::new(&sources, false)
            .render_all(&diags, &mut out)
            .unwrap();
        (libraries, String::from_utf8(out).unwrap())
    }

    fn paths(modules: &[Module]) -> Vec<String> {
        let mut paths: Vec<_> = modules.iter().map(|(_, p)| p.join(".")).collect();
        paths.sort();
        paths
    }

    #[test]
    fn resolves_libraries_on_disk() {
        let dir = scratch("disk");
        write(&dir.join("mystd/net.vg"), "def f() -> u32:\n    return 1\n");
        write(
            &dir.join("mystd/ml/tensor.vg"),
            "def g() -> u32:\n    return 1\n",
        );
        write(&dir.join("flat/ops.vg"), "def h() -> u32:\n    return 1\n");
        // a library with a manifest, sources in `src/`, the same `flat` as
        // the project, and a `std` that is ignored
        write(
            &dir.join("geo/vigil.toml"),
            "[package]\nname = \"geo\"\n\n[dependencies]\nflat = { path = \"../flat\" }\n\
             std = { path = \"../flat\" }\n",
        );
        write(
            &dir.join("geo/src/square.vg"),
            "def s() -> u32:\n    return 1\n",
        );
        write(&dir.join("geo/README.vg.txt"), "not a module");
        write(
            &dir.join("app/vigil.toml"),
            "[package]\nname = \"app\"\n\n[dependencies]\nstd = { path = \"../mystd\" }\n\
             geo = { path = \"../geo\" }\nflat = { path = \"../flat\" }\n",
        );
        let (libraries, out) = resolve_at(&dir.join("app"), false);
        let libraries = libraries.expect(&out);
        assert!(
            out.contains("a library cannot choose the standard library"),
            "{out}"
        );
        assert_eq!(
            paths(libraries.std.as_ref().unwrap()),
            ["std.ml.tensor", "std.net"]
        );
        assert_eq!(paths(&libraries.modules), ["flat.ops", "geo.square"]);
        assert!(!dir.join("app").join(LOCK_NAME).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reports_two_libraries_under_one_name() {
        let dir = scratch("clash");
        write(&dir.join("a/x.vg"), "def f() -> u32:\n    return 1\n");
        write(&dir.join("b/x.vg"), "def f() -> u32:\n    return 1\n");
        write(
            &dir.join("lib/vigil.toml"),
            "[package]\nname = \"lib\"\n\n[dependencies]\nshared = { path = \"../b\" }\n",
        );
        write(&dir.join("lib/src/m.vg"), "def m() -> u32:\n    return 1\n");
        write(
            &dir.join("app/vigil.toml"),
            "[package]\nname = \"app\"\n\n[dependencies]\nshared = { path = \"../a\" }\n\
             lib = { path = \"../lib\" }\nnone = { path = \"../missing\" }\n",
        );
        let (libraries, out) = resolve_at(&dir.join("app"), false);
        assert!(libraries.is_none());
        assert!(
            out.contains("two different libraries are named `shared`"),
            "{out}"
        );
        assert!(
            out.contains("the path of dependency `none` is not a directory"),
            "{out}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// a hand-edited lock with a short commit is ignored with a warning,
    /// and never sliced
    #[test]
    fn ignores_invalid_lock_entries() {
        let dir = scratch("badlock");
        let good = "0123456789abcdef0123456789abcdef01234567";
        write(
            &dir.join(LOCK_NAME),
            &format!(
                "version = 1\n\n[[git]]\nurl = \"a\"\nreference = \"tag=v1\"\ncommit = \"abc\"\n\n\
                 [[git]]\nurl = \"b\"\nreference = \"tag=v1\"\ncommit = \"{good}\"\n\n\
                 [[git]]\nurl = \"c\"\nreference = \"tag=v1\"\ncommit = \"{}\"\n",
                "z".repeat(40)
            ),
        );
        let mut diags = Diagnostics::new();
        let entries = read_lock(&dir.join(LOCK_NAME), &mut diags);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].commit, good);
        assert_eq!(diags.warning_count(), 1);
        assert!(!diags.has_errors());

        // a valid lock gives no warning
        write(
            &dir.join(LOCK_NAME),
            &format!(
                "version = 1\n\n[[git]]\nurl = \"b\"\nreference = \"tag=v1\"\ncommit = \"{good}\"\n"
            ),
        );
        let mut diags = Diagnostics::new();
        assert_eq!(read_lock(&dir.join(LOCK_NAME), &mut diags).len(), 1);
        assert!(diags.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    }

    #[test]
    fn finds_the_cache_directory() {
        let root = |vars: &[(&str, &str)], windows| {
            let vars: Vec<(String, String)> = vars
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            cache_root_in(
                |name| vars.iter().find(|(k, _)| k == name).map(|(_, v)| v.into()),
                windows,
            )
        };
        let path = |p: &str| Ok(PathBuf::from(p));
        let all = [
            ("VIGIL_HOME", "/v"),
            ("XDG_CACHE_HOME", "/x"),
            ("LOCALAPPDATA", "/l"),
            ("HOME", "/h"),
        ];
        assert_eq!(root(&all, false), path("/v"));
        assert_eq!(root(&all[1..], false), path("/x/vigil"));
        assert_eq!(root(&all[2..], false), path("/h/.cache/vigil"));
        assert_eq!(root(&all[2..], true), path("/l/vigil"));
        assert_eq!(
            root(&[("VIGIL_HOME", ""), ("HOME", "/h")], true),
            path("/h/.cache/vigil")
        );
        assert!(root(&[], true).is_err());
    }

    /// the commit of a tag is locked, so moving the tag changes nothing
    /// until an update
    #[test]
    fn locks_git_dependencies() {
        let dir = scratch("git");
        // the only test that reads `VIGIL_HOME`, so no other thread races it
        unsafe { std::env::set_var("VIGIL_HOME", dir.join("cache")) };
        let repo = dir.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-q", "-b", "main"]);
        write(&repo.join("one.vg"), "def one() -> u32:\n    return 1\n");
        git(&repo, &["add", "."]);
        git(&repo, &["commit", "-qm", "one"]);
        git(&repo, &["tag", "v1"]);
        let first = git(&repo, &["rev-parse", "HEAD"]);
        // `file:///C:/...` on Windows, `file:///tmp/...` elsewhere
        let url = format!(
            "file:///{}",
            repo.display()
                .to_string()
                .replace('\\', "/")
                .trim_start_matches('/')
        );
        write(
            &dir.join("app/vigil.toml"),
            &format!(
                "[package]\nname = \"app\"\n\n[dependencies]\nlib = {{ git = \"{url}\", tag = \"v1\" }}\n"
            ),
        );
        let app = dir.join("app");
        let lock = || std::fs::read_to_string(app.join(LOCK_NAME)).unwrap();

        let (libraries, out) = resolve_at(&app, false);
        assert_eq!(paths(&libraries.expect(&out).modules), ["lib.one"]);
        assert!(lock().contains(&first), "{}", lock());

        write(&repo.join("two.vg"), "def two() -> u32:\n    return 2\n");
        git(&repo, &["add", "."]);
        git(&repo, &["commit", "-qm", "two"]);
        git(&repo, &["tag", "-f", "v1"]);
        let second = git(&repo, &["rev-parse", "HEAD"]);

        let (libraries, out) = resolve_at(&app, false);
        assert_eq!(paths(&libraries.expect(&out).modules), ["lib.one"]);
        assert!(lock().contains(&first));

        let (libraries, out) = resolve_at(&app, true);
        assert_eq!(
            paths(&libraries.expect(&out).modules),
            ["lib.one", "lib.two"]
        );
        assert!(
            lock().contains(&second) && !lock().contains(&first),
            "{}",
            lock()
        );

        // a hand-edited short commit is resolved again, not sliced
        let edited = lock().replace(&second, &second[..7]);
        write(&app.join(LOCK_NAME), &edited);
        let (libraries, out) = resolve_at(&app, false);
        assert_eq!(
            paths(&libraries.expect(&out).modules),
            ["lib.one", "lib.two"]
        );
        assert!(out.contains("which is not valid"), "{out}");
        assert!(lock().contains(&second), "{}", lock());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
