//! this module reads the project manifest, `vigil.toml`, with its tables
//! `[package]`, `[target]` and `[dependencies]`. Without `[target]` we use
//! [`Profile::default_profile`].
//!
//! We check it in two steps: `serde` checks the shape into the `Raw*`
//! structs (unknown keys are errors, so typos are not ignored), then we
//! check the values by hand. The strings are `Spanned` so diagnostics point
//! at the bad entry. A dependency named `std` replaces the built-in
//! standard library, and `kernel` is reserved for the kernel BTF types.

use serde::Deserialize;
use toml::Spanned;
use vigil_diagnostics::{Diagnostic, Diagnostics, FileId, Span, codes};

use crate::arch::Arch;
use crate::profile::{MIN_SUPPORTED, Profile};
use crate::version::KernelVersion;

/// the manifest file name, at the root of a project
pub const MANIFEST_NAME: &str = "vigil.toml";

/// a manifest that passed every check
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub name: String,
    pub version: Option<String>,
    pub profile: Profile,
    /// the BTF of the target kernel, used instead of the running one
    pub btf: Option<String>,
    /// the libraries of `[dependencies]`, sorted by name
    pub dependencies: Vec<Dependency>,
}

/// a library from `[dependencies]`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dependency {
    /// the name its modules are imported under, as in `netx.parse`
    pub name: String,
    pub source: Source,
    /// the manifest entry, for diagnostics
    pub span: Span,
}

/// where a library is
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// a directory as written, relative to the manifest unless absolute
    Path(String),
    /// a git repository and the commit to take
    Git { url: String, reference: GitRef },
}

/// the commit of a git repository a dependency takes
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GitRef {
    /// the commit of the remote `HEAD`
    DefaultBranch,
    Branch(String),
    Tag(String),
    /// a full commit hash, 40 hex digits
    Rev(String),
}

impl GitRef {
    /// the reference as text (`branch=main`, `tag=v1`, `rev=...`, `HEAD`).
    /// `vigil.lock` keeps it next to the resolved commit, so a changed
    /// entry is resolved again.
    pub fn spec(&self) -> String {
        match self {
            GitRef::DefaultBranch => "HEAD".into(),
            GitRef::Branch(b) => format!("branch={b}"),
            GitRef::Tag(t) => format!("tag={t}"),
            GitRef::Rev(r) => format!("rev={r}"),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    package: RawPackage,
    target: Option<RawTarget>,
    #[serde(default)]
    dependencies: std::collections::BTreeMap<Spanned<String>, Spanned<RawDependency>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDependency {
    path: Option<String>,
    git: Option<String>,
    github: Option<String>,
    branch: Option<String>,
    tag: Option<String>,
    rev: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPackage {
    name: Spanned<String>,
    version: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTarget {
    kernels: Spanned<Vec<Spanned<String>>>,
    #[serde(default = "default_privileged")]
    privileged: bool,
    btf: Option<String>,
    /// a name or alias that [`Arch::parse`] accepts, the host by default
    arch: Option<Spanned<String>>,
}

fn default_privileged() -> bool {
    true
}

/// parsing a `vigil.toml`, reporting all the problems found
///
/// We return the manifest only if there is no error (a duplicate kernel is
/// only a warning). Invalid TOML or a wrong shape is reported alone, because
/// we cannot check anything else.
pub fn parse_manifest(file: FileId, text: &str) -> (Option<Manifest>, Diagnostics) {
    let mut diags = Diagnostics::new();
    let raw: RawManifest = match toml::from_str(text) {
        Ok(raw) => raw,
        Err(err) => {
            let mut diag = Diagnostic::error(format!("invalid `{MANIFEST_NAME}`"))
                .with_code(codes::INVALID_MANIFEST);
            let message = toml_message(err.message());
            diag = match err.span() {
                Some(range) => diag.with_primary(Span::new(file, range), message),
                None => diag.with_note(message),
            };
            if err.message() == "missing field `package`" {
                diag = diag.with_help("start the file with `[package]` and `name = \"...\"`");
            }
            diags.push(diag);
            return (None, diags);
        }
    };

    let name = check_name(file, &raw.package.name, &mut diags);
    let dependencies: Vec<Dependency> = raw
        .dependencies
        .iter()
        .filter_map(|(name, entry)| check_dependency(file, name, entry, &mut diags))
        .collect();
    let btf = raw.target.as_ref().and_then(|t| t.btf.clone());
    let profile = match raw.target {
        Some(target) => check_target(file, target, &mut diags),
        None => Some(Profile::default_profile()),
    };

    let manifest = match (name, profile) {
        (Some(name), Some(profile)) if !diags.has_errors() => Some(Manifest {
            name,
            version: raw.package.version,
            profile,
            btf,
            dependencies,
        }),
        _ => None,
    };
    (manifest, diags)
}

/// a message of `toml` and `serde` in the terms of the file: tables, keys
/// and arrays, not structs, fields and sequences
fn toml_message(message: &str) -> String {
    if let Some(rest) = message.strip_prefix("invalid type: ")
        && let Some((found, expected)) = rest.split_once(", expected ")
    {
        let expected = match expected {
            "struct RawDependency" => "a table such as `{ path = \"../lib\" }`",
            e if e.starts_with("struct ") => "a table",
            "a sequence" => "an array",
            e => e,
        };
        let found = found
            .replace("floating point", "float")
            .replace("sequence", "array")
            .replace("map", "table");
        return format!("expected {expected}, found {found}");
    }
    if let Some(field) = message.strip_prefix("missing field ") {
        return match field {
            "`package`" | "`target`" | "`dependencies`" => {
                format!("missing table `[{}]`", field.trim_matches('`'))
            }
            _ => format!("missing key {field}"),
        };
    }
    if let Some(rest) = message.strip_prefix("unknown field ") {
        return format!("unknown key {rest}");
    }
    message.to_string()
}

/// checking the package name: a lowercase ASCII letter followed by
/// lowercase letters, digits, `_` and `-`
fn check_name(file: FileId, name: &Spanned<String>, diags: &mut Diagnostics) -> Option<String> {
    let value = name.get_ref();
    let mut chars = value.chars();
    let valid = chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
    if valid {
        return Some(value.clone());
    }
    diags.push(
        Diagnostic::error(format!("invalid package name `{value}`"))
            .with_code(codes::INVALID_PACKAGE_NAME)
            .with_primary(Span::new(file, name.span()), "not a valid package name")
            .with_help(
                "package names start with a lowercase letter and contain only \
                 lowercase letters, digits, `_` and `-`",
            ),
    );
    None
}

/// checking a `[dependencies]` entry: a name that can start a module path,
/// and one source with at most one reference
fn check_dependency(
    file: FileId,
    name: &Spanned<String>,
    entry: &Spanned<RawDependency>,
    diags: &mut Diagnostics,
) -> Option<Dependency> {
    let span = Span::new(file, name.span().start..entry.span().end);
    let mut error = |message: String, label: &str, help: Option<&str>| {
        let mut diag = Diagnostic::error(message)
            .with_code(codes::INVALID_DEPENDENCY)
            .with_primary(span, label);
        if let Some(help) = help {
            diag = diag.with_help(help);
        }
        diags.push(diag);
        None
    };
    let text = name.get_ref();
    let mut chars = text.chars();
    let identifier = chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if !identifier {
        return error(
            format!("invalid dependency name `{text}`"),
            "not a valid module name",
            Some(
                "a dependency is imported by its name, which starts with a lowercase letter \
                 and contains only lowercase letters, digits and `_`",
            ),
        );
    }
    if text == "kernel" {
        return error(
            "`kernel` cannot be a dependency".into(),
            "a reserved name",
            Some("`kernel` names the types of the kernel's BTF"),
        );
    }
    let raw = entry.get_ref();
    let source = match (&raw.path, &raw.git, &raw.github) {
        (Some(path), None, None) => {
            if raw.branch.is_some() || raw.tag.is_some() || raw.rev.is_some() {
                return error(
                    format!("dependency `{text}` is a path and cannot have a branch, tag or rev"),
                    "a path with a git reference",
                    Some("`branch`, `tag` and `rev` go with `git` or `github`"),
                );
            }
            Source::Path(path.clone())
        }
        (None, Some(_), None) | (None, None, Some(_)) => {
            let url = match (&raw.git, &raw.github) {
                (Some(url), _) => url.clone(),
                (_, Some(repo)) => {
                    let parts: Vec<&str> = repo.split('/').collect();
                    if parts.len() != 2 || parts.iter().any(|p| p.is_empty()) {
                        return error(
                            format!("invalid GitHub repository `{repo}`"),
                            "not `owner/repo`",
                            Some("write `github = \"owner/repo\"`"),
                        );
                    }
                    format!("https://github.com/{repo}")
                }
                _ => unreachable!(),
            };
            let reference = match (&raw.branch, &raw.tag, &raw.rev) {
                (None, None, None) => GitRef::DefaultBranch,
                (Some(b), None, None) => GitRef::Branch(b.clone()),
                (None, Some(t), None) => GitRef::Tag(t.clone()),
                (None, None, Some(r)) => {
                    if r.len() != 40 || !r.chars().all(|c| c.is_ascii_hexdigit()) {
                        return error(
                            format!("invalid rev `{r}` for `{text}`"),
                            "not a full commit hash",
                            Some("`rev` is the full hash of a commit, 40 hexadecimal digits"),
                        );
                    }
                    GitRef::Rev(r.to_ascii_lowercase())
                }
                _ => {
                    return error(
                        format!(
                            "dependency `{text}` has more than one of `branch`, `tag` and `rev`"
                        ),
                        "more than one reference",
                        None,
                    );
                }
            };
            Source::Git { url, reference }
        }
        (None, None, None) => {
            return error(
                format!("dependency `{text}` has no source"),
                "no `path`, `git` or `github`",
                Some("give one of `path`, `git` or `github`"),
            );
        }
        _ => {
            return error(
                format!("dependency `{text}` has more than one source"),
                "more than one of `path`, `git` and `github`",
                None,
            );
        }
    };
    Some(Dependency {
        name: text.clone(),
        source,
        span,
    })
}

/// checking a `[target]` table
///
/// We check every kernel entry, so one bad entry does not hide the next,
/// and any error gives `None`. An unknown architecture also gives `None`.
fn check_target(file: FileId, target: RawTarget, diags: &mut Diagnostics) -> Option<Profile> {
    let list_span = Span::new(file, target.kernels.span());
    let entries = target.kernels.into_inner();
    if entries.is_empty() {
        diags.push(
            Diagnostic::error("no target kernel")
                .with_code(codes::EMPTY_KERNEL_LIST)
                .with_primary(list_span, "this list is empty")
                .with_help("list at least one kernel, for example `kernels = [\"linux-6.1\"]`"),
        );
        return None;
    }

    let mut kernels: Vec<(KernelVersion, Span)> = Vec::new();
    let mut failed = false;
    for entry in &entries {
        let span = Span::new(file, entry.span());
        let text = entry.get_ref();
        let version: KernelVersion = match text.parse() {
            Ok(version) => version,
            Err(err) => {
                diags.push(
                    Diagnostic::error(format!("invalid kernel version `{text}`"))
                        .with_code(codes::INVALID_KERNEL_VERSION)
                        .with_primary(span, err.to_string()),
                );
                failed = true;
                continue;
            }
        };
        if version < MIN_SUPPORTED {
            diags.push(
                Diagnostic::error(format!("kernel {version} is not supported"))
                    .with_code(codes::UNSUPPORTED_KERNEL)
                    .with_primary(span, format!("older than {MIN_SUPPORTED}"))
                    .with_note(format!(
                        "Vigil needs bounded loops, available since {MIN_SUPPORTED}"
                    )),
            );
            failed = true;
            continue;
        }
        if let Some(&(_, first)) = kernels.iter().find(|(v, _)| *v == version) {
            diags.push(
                Diagnostic::warning(format!("kernel {version} is listed twice"))
                    .with_code(codes::DUPLICATE_KERNEL)
                    .with_primary(span, "duplicate entry")
                    .with_secondary(first, "first listed here"),
            );
            continue;
        }
        kernels.push((version, span));
    }

    let arch = match &target.arch {
        None => Arch::host(),
        Some(name) => match Arch::parse(name.get_ref()) {
            Some(arch) => arch,
            None => {
                let known: Vec<_> = Arch::ALL.iter().map(|a| format!("`{a}`")).collect();
                diags.push(
                    Diagnostic::error(format!("unknown architecture `{}`", name.get_ref()))
                        .with_code(codes::INVALID_MANIFEST)
                        .with_primary(Span::new(file, name.span()), "not a target architecture")
                        .with_help(format!("Vigil targets {}", known.join(" and "))),
                );
                return None;
            }
        },
    };
    if failed {
        return None;
    }
    Profile::new(
        kernels.into_iter().map(|(v, _)| v).collect(),
        target.privileged,
    )
    .map(|p| p.with_arch(arch))
}

#[cfg(test)]
mod tests {
    use vigil_diagnostics::{Renderer, SourceMap};

    use super::*;

    /// parsing a text, with the diagnostics rendered without colour
    fn parse(text: &str) -> (Option<Manifest>, String) {
        let mut sources = SourceMap::new();
        let file = sources.add(MANIFEST_NAME, text);
        let (manifest, diags) = parse_manifest(file, text);
        let mut out = Vec::new();
        Renderer::new(&sources, false)
            .render_all(&diags, &mut out)
            .unwrap();
        (manifest, String::from_utf8(out).unwrap())
    }

    #[test]
    fn parses_the_architecture() {
        let (manifest, out) = parse(
            r#"
[package]
name = "probe"

[target]
kernels = ["linux-6.1"]
arch = "arm64"
"#,
        );
        assert_eq!(out, "");
        assert_eq!(manifest.unwrap().profile.arch(), Arch::Aarch64);
        let (manifest, _) = parse("[package]\nname = \"p\"\n\n[target]\nkernels = [\"6.1\"]\n");
        assert_eq!(manifest.unwrap().profile.arch(), Arch::host());
        let (manifest, out) =
            parse("[package]\nname = \"p\"\n\n[target]\nkernels = [\"6.1\"]\narch = \"riscv64\"\n");
        assert!(manifest.is_none());
        assert!(out.contains("unknown architecture `riscv64`"), "{out}");
        assert!(out.contains("`x86_64` and `aarch64`"), "{out}");
    }

    #[test]
    fn parses_a_complete_manifest() {
        let (manifest, out) = parse(
            r#"
[package]
name = "flowmon"
version = "0.1.0"

[target]
kernels = ["linux-6.1", "5.15"]
privileged = false
"#,
        );
        assert_eq!(out, "");
        let manifest = manifest.unwrap();
        assert_eq!(manifest.name, "flowmon");
        assert_eq!(manifest.version.as_deref(), Some("0.1.0"));
        assert_eq!(manifest.profile.baseline(), KernelVersion::new(5, 15));
        assert!(!manifest.profile.privileged());
    }

    #[test]
    fn target_defaults_to_the_default_profile() {
        let (manifest, out) = parse("[package]\nname = \"hello\"\n");
        assert_eq!(out, "");
        assert_eq!(manifest.unwrap().profile, Profile::default_profile());
    }

    #[test]
    fn parses_dependencies() {
        let rev = "0123456789abcdef0123456789abcdef01234567";
        let (manifest, out) = parse(&format!(
            r#"[package]
name = "app"

[dependencies]
std = {{ path = "../std" }}
netx = {{ github = "acme/netx", tag = "v0.2.0" }}
flows = {{ git = "https://example.org/flows.git", rev = "{rev}" }}
plain = {{ git = "file:///tmp/plain" }}
"#
        ));
        assert_eq!(out, "");
        let deps: Vec<_> = manifest
            .unwrap()
            .dependencies
            .into_iter()
            .map(|d| (d.name, d.source))
            .collect();
        assert_eq!(
            deps,
            [
                (
                    "flows".into(),
                    Source::Git {
                        url: "https://example.org/flows.git".into(),
                        reference: GitRef::Rev(rev.into())
                    }
                ),
                (
                    "netx".into(),
                    Source::Git {
                        url: "https://github.com/acme/netx".into(),
                        reference: GitRef::Tag("v0.2.0".into())
                    }
                ),
                (
                    "plain".into(),
                    Source::Git {
                        url: "file:///tmp/plain".into(),
                        reference: GitRef::DefaultBranch
                    }
                ),
                ("std".into(), Source::Path("../std".into())),
            ]
        );
    }

    #[test]
    fn reports_bad_dependencies() {
        let (manifest, out) = parse(
            r#"[package]
name = "app"

[dependencies]
Net-X = { path = "x" }
kernel = { path = "k" }
both = { path = "a", github = "o/r" }
none = {}
gh = { github = "just-a-name" }
refs = { github = "o/r", tag = "v1", branch = "main" }
short = { github = "o/r", rev = "abc123" }
pathref = { path = "p", tag = "v1" }
"#,
        );
        assert!(manifest.is_none());
        for expected in [
            "invalid dependency name `Net-X`",
            "`kernel` cannot be a dependency",
            "dependency `both` has more than one source",
            "dependency `none` has no source",
            "invalid GitHub repository `just-a-name`",
            "dependency `refs` has more than one of `branch`, `tag` and `rev`",
            "invalid rev `abc123` for `short`",
            "dependency `pathref` is a path and cannot have a branch, tag or rev",
        ] {
            assert!(out.contains(expected), "missing {expected:?} in:\n{out}");
        }
    }

    #[test]
    fn reports_unknown_fields() {
        let (manifest, out) = parse("[package]\nname = \"hello\"\nauthor = \"me\"\n");
        assert!(manifest.is_none());
        insta::assert_snapshot!(out);
    }

    /// the messages of `serde` speak of tables, keys and arrays
    #[test]
    fn reports_bad_shapes_in_toml_terms() {
        for (text, expected) in [
            (
                "[package]\nname = \"p\"\n[target]\nkernels = \"6.1\"\n",
                "expected an array, found string \"6.1\"",
            ),
            (
                "[package]\nname = \"p\"\n[dependencies]\nnet = \"1.0\"\n",
                "expected a table such as `{ path = \"../lib\" }`, found string \"1.0\"",
            ),
            ("[package]\nversion = \"1\"\n", "missing key `name`"),
            (
                "[target]\nkernels = [\"6.1\"]\n",
                "missing table `[package]`",
            ),
        ] {
            let (manifest, out) = parse(text);
            assert!(manifest.is_none());
            assert!(out.contains(expected), "missing {expected:?} in:\n{out}");
        }
    }

    #[test]
    fn reports_every_bad_kernel() {
        let (manifest, out) = parse(
            r#"[package]
name = "flowmon"

[target]
kernels = ["linux-6.1", "linux-4.19", "six.one"]
"#,
        );
        assert!(manifest.is_none());
        insta::assert_snapshot!(out);
    }

    #[test]
    fn duplicate_kernels_only_warn() {
        let (manifest, out) = parse(
            r#"[package]
name = "flowmon"

[target]
kernels = ["linux-6.1", "6.1"]
"#,
        );
        assert_eq!(
            manifest.unwrap().profile.kernels(),
            [KernelVersion::new(6, 1)]
        );
        insta::assert_snapshot!(out);
    }

    #[test]
    fn reports_empty_kernel_list_and_bad_name() {
        let (manifest, out) = parse(
            r#"[package]
name = "Flow Mon"

[target]
kernels = []
"#,
        );
        assert!(manifest.is_none());
        insta::assert_snapshot!(out);
    }
}
