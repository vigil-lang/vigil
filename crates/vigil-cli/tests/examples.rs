//! every example of `compiler/examples` checks, and builds to an object
//! when there is a clang that can (`VIGIL_CLANG` or `clang`, 15 or newer)

use std::path::{Path, PathBuf};
use std::process::Command;

use vigil_codegen_llvm::clang::Clang;

fn examples() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "vg"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no examples in {}", dir.display());
    paths
}

/// running `vigil args`, giving the failure as text
fn vigil(args: &[&std::ffi::OsStr]) -> Result<(), String> {
    let out = Command::new(env!("CARGO_BIN_EXE_vigil"))
        .args(["--color", "never"])
        .args(args)
        .output()
        .unwrap();
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

/// running every example before failing, so one report lists them all

fn assert_all(what: &str, run: impl Fn(&Path) -> Result<(), String>) {
    let failures: Vec<String> = examples()
        .iter()
        .filter_map(|p| run(p).err().map(|e| format!("{}:\n{e}", p.display())))
        .collect();
    assert!(
        failures.is_empty(),
        "{} examples fail to {what}:\n\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn examples_check() {
    assert_all("check", |p| vigil(&["check".as_ref(), p.as_os_str()]));
}

#[test]
fn examples_build() {
    if let Err(err) = Clang::discover() {
        println!("skipping the build of the examples: {err}");
        return;
    }
    let out = std::env::temp_dir().join(format!("vigil-examples-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    assert_all("build", |p| {
        let object = out.join(p.file_stem().unwrap()).with_extension("o");
        vigil(&[
            "build".as_ref(),
            p.as_os_str(),
            "-o".as_ref(),
            object.as_os_str(),
        ])
    });
    let _ = std::fs::remove_dir_all(&out);
}
