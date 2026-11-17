//! this module finds a project and loads its manifest and sources.
//! [`load`] turns a directory or a `.vg` file into a [`Project`], and every
//! problem becomes a diagnostic. The libraries are loaded by
//! [`crate::deps`].

use std::io;
use std::path::{Path, PathBuf};

use vigil_diagnostics::{Diagnostic, Diagnostics, FileId, SourceMap, codes};
use vigil_target::{MANIFEST_NAME, Manifest, Profile, parse_manifest};

use crate::deps::{self, Libraries};

/// the extension of source files
pub const SOURCE_EXTENSION: &str = "vg";
/// the sources directory, under the project root
pub const SOURCE_DIR: &str = "src";

/// a loaded project: what the manifest says, and the source files
#[derive(Debug)]
pub struct Project {
    /// the manifest name, or the stem of a lone file
    pub name: String,
    /// the kernel profile we check and compile for
    pub profile: Profile,
    /// the directory with `vigil.toml`, `None` for a lone file
    pub root: Option<PathBuf>,
    /// the BTF of the manifest, joined to the root
    pub btf: Option<PathBuf>,
    /// the source files with their module paths, sorted by path so that two
    /// runs give the same file ids
    pub files: Vec<(FileId, Vec<String>)>,
    /// the libraries of `[dependencies]`, none for a lone file
    pub libraries: Libraries,
}

/// loading the project of `path`, adding its files to `sources`. `None`
/// after an error, which is in `diags`
///
/// - a directory must have a `vigil.toml`. Its sources are the `.vg` files
///   under `src/`, and `src/net/parse.vg` is the module `net.parse`
/// - a `.vg` file means the project of the nearest ancestor with a
///   `vigil.toml`, so its imports resolve. Without one, it is a lone module
///   with the default profile and the built-in standard library
///
/// With `update`, git libraries are resolved again instead of read from
/// `vigil.lock`.
pub fn load(
    path: &Path,
    update: bool,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<Project> {
    if !path.exists() {
        diags.push(
            Diagnostic::error(format!("`{}` does not exist", path.display()))
                .with_code(codes::PROJECT_NOT_FOUND),
        );
        return None;
    }

    if path.is_dir() {
        let manifest_path = path.join(MANIFEST_NAME);
        if !manifest_path.is_file() {
            diags.push(
                Diagnostic::error(format!(
                    "could not find `{MANIFEST_NAME}` in `{}`",
                    path.display()
                ))
                .with_code(codes::PROJECT_NOT_FOUND)
                .with_help("create a `vigil.toml` or pass a `.vg` file"),
            );
            return None;
        }
        let manifest = load_manifest(&manifest_path, sources, diags)?;
        let files = load_sources(&path.join(SOURCE_DIR), sources, diags)?;
        let libraries =
            deps::resolve(path, &manifest.dependencies, &files, update, sources, diags)?;
        return Some(project(manifest, path, files, libraries));
    }

    if path.extension().is_some_and(|e| e == SOURCE_EXTENSION) {
        if let Some(root) = path.parent().and_then(find_root) {
            return load(&root, update, sources, diags);
        }
        let file = load_file(path, sources, diags)?;
        return Some(Project {
            name: stem(path),
            profile: Profile::default_profile(),
            root: None,
            btf: None,
            files: vec![(file, vec![stem(path)])],
            libraries: Libraries::default(),
        });
    }

    diags.push(
        Diagnostic::error(format!(
            "`{}` is neither a directory nor a `.{SOURCE_EXTENSION}` file",
            path.display()
        ))
        .with_code(codes::PROJECT_NOT_FOUND),
    );
    None
}

fn project(
    manifest: Manifest,
    root: &Path,
    files: Vec<(FileId, Vec<String>)>,
    libraries: Libraries,
) -> Project {
    Project {
        name: manifest.name,
        profile: manifest.profile,
        root: Some(root.to_path_buf()),
        btf: manifest.btf.map(|b| root.join(b)),
        files,
        libraries,
    }
}

/// the files that [`load`] would read (manifest and sources), sorted, but
/// without reading them or reporting errors. A lone `.vg` file is the only
/// file. The name comes from `vigil check --watch` in the full Vigil, here
/// `vigil fmt` uses it.
pub fn watched_files(path: &Path) -> Vec<PathBuf> {
    let root = if path.is_dir() {
        Some(path.to_path_buf())
    } else {
        path.parent().and_then(find_root)
    };
    let Some(root) = root else {
        return vec![path.to_path_buf()];
    };
    let mut files = vec![root.join(MANIFEST_NAME)];
    let _ = collect_sources(&root.join(SOURCE_DIR), &mut files);
    files.sort();
    files
}

/// the nearest directory, from `dir` upwards, with a `vigil.toml`. An empty
/// `dir` (the parent of a bare file name) means the current directory.
///
/// We walk the ancestors of the canonical `dir`, because the ancestors of a
/// relative path stop at the current directory and end in an empty path.
/// For a relative `dir` the root is given relative to the current
/// directory (`.`, `..`, `src/..` is `.`), so the diagnostics stay short.
fn find_root(dir: &Path) -> Option<PathBuf> {
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    let canonical = dir.canonicalize().ok()?;
    let root = canonical
        .ancestors()
        .find(|d| d.join(MANIFEST_NAME).is_file())?;
    if dir.is_absolute() {
        return Some(root.to_path_buf());
    }
    let Ok(cwd) = std::env::current_dir().and_then(|d| d.canonicalize()) else {
        return Some(root.to_path_buf());
    };
    if let Ok(below) = root.strip_prefix(&cwd) {
        let below = below.to_path_buf();
        return Some(if below.as_os_str().is_empty() {
            ".".into()
        } else {
            below
        });
    }
    if let Ok(above) = cwd.strip_prefix(root) {
        return Some(above.iter().map(|_| "..").collect());
    }
    Some(root.to_path_buf())
}

fn stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

pub(crate) fn load_manifest(
    path: &Path,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<Manifest> {
    let file = load_file(path, sources, diags)?;
    let (manifest, manifest_diags) = parse_manifest(file, sources.get(file).text());
    diags.extend(manifest_diags);
    manifest
}

pub(crate) fn load_file(
    path: &Path,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<FileId> {
    match sources.load(path) {
        Ok(file) => Some(file),
        Err(err) => {
            diags.push(io_error(path, &err));
            None
        }
    }
}

/// loading the `.vg` files under `dir`, with their module paths (the path
/// from `dir` without the extension). No sources is only a warning.
fn load_sources(
    dir: &Path,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<Vec<(FileId, Vec<String>)>> {
    let mut paths = Vec::new();
    if dir.is_dir()
        && let Err(err) = collect_sources(dir, &mut paths)
    {
        diags.push(io_error(dir, &err));
        return None;
    }
    paths.sort();
    if paths.is_empty() {
        diags.push(
            Diagnostic::warning(format!(
                "no `.{SOURCE_EXTENSION}` files in `{}`",
                dir.display()
            ))
            .with_code(codes::NO_SOURCES)
            .with_help(format!(
                "the sources of a project are the `.{SOURCE_EXTENSION}` files under `{SOURCE_DIR}/`, \
                 such as `{SOURCE_DIR}/main.{SOURCE_EXTENSION}`"
            )),
        );
    }
    let mut files = Vec::with_capacity(paths.len());
    for path in &paths {
        let module = path
            .strip_prefix(dir)
            .unwrap_or(path)
            .with_extension("")
            .iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        files.push((load_file(path, sources, diags)?, module));
    }
    Some(files)
}

pub(crate) fn collect_sources(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_sources(&path, out)?;
        } else if path.extension().is_some_and(|e| e == SOURCE_EXTENSION) {
            out.push(path);
        }
    }
    Ok(())
}

pub(crate) fn io_error(path: &Path, err: &io::Error) -> Diagnostic {
    Diagnostic::error(format!("could not read `{}`", path.display()))
        .with_code(codes::IO_ERROR)
        .with_note(err.to_string())
}
