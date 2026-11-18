//! the project of a `.vg` file is found from any form of its path, run from
//! the project root, from `src/` or from elsewhere

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use vigil_codegen_llvm::clang::Clang;

/// a fresh project whose `main` imports `util`, so a check of `main.vg`
/// alone (as a lone file) fails
fn project(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vigil-root-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("vigil.toml"), "[package]\nname = \"app\"\n").unwrap();
    std::fs::write(dir.join("src/util.vg"), "def one() -> u32:\n    return 1\n").unwrap();
    std::fs::write(
        dir.join("src/main.vg"),
        "from util import one\n\ndef two() -> u32:\n    return one() + 1\n",
    )
    .unwrap();
    dir
}

fn check(dir: &Path, path: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vigil"))
        .args(["--color", "never", "check", path])
        .current_dir(dir)
        .output()
        .unwrap()
}

fn assert_ok(out: &Output, what: &str) {
    assert!(
        out.status.success(),
        "{what}:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn finds_the_project_of_a_relative_file() {
    let dir = project("relative");
    assert_ok(&check(&dir, "src/main.vg"), "from the root");
    assert_ok(&check(&dir, "./src/main.vg"), "from the root, with `./`");
    assert_ok(&check(&dir.join("src"), "main.vg"), "from `src/`");
    assert_ok(
        &check(&dir.join("src"), "../src/main.vg"),
        "from `src/`, with `..`",
    );
    let abs = dir.join("src/main.vg");
    assert_ok(
        &check(&std::env::temp_dir(), abs.to_str().unwrap()),
        "absolute, from elsewhere",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// a file outside any project is still a lone module, and its missing
/// import is an error
#[test]
fn a_file_outside_a_project_is_alone() {
    let dir = project("lone");
    std::fs::remove_file(dir.join("vigil.toml")).unwrap();
    std::fs::write(dir.join("solo.vg"), "def f() -> u32:\n    return 1\n").unwrap();
    assert_ok(&check(&dir, "solo.vg"), "a lone file");
    let out = check(&dir.join("src"), "main.vg");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("does not exist"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// a missing BTF is named once, in the message: the note gives only the
/// reason
#[test]
fn a_missing_btf_is_named_once() {
    let dir = project("btf");
    std::fs::write(
        dir.join("vigil.toml"),
        "[package]\nname = \"app\"\n\n[target]\nkernels = [\"linux-6.1\"]\nbtf = \"missing.btf\"\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/util.vg"),
        "from kernel import task_struct\n\ndef one() -> u32:\n    return 1\n",
    )
    .unwrap();
    let out = check(&dir, ".");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("could not read the BTF"), "{stderr}");
    assert_eq!(stderr.matches("missing.btf").count(), 1, "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `build` names a project as `check` does, and a lone file by its path
#[test]
fn build_names_the_project() {
    if Clang::discover().is_err() {
        println!("skipping: no clang");
        return;
    }
    let dir = project("build");
    std::fs::write(
        dir.join("src/main.vg"),
        "from std.bpf import KprobeContext\n\n@kprobe(\"do_sys_openat2\")\ndef on_open(ctx: KprobeContext) -> i32:\n    return 0\n",
    )
    .unwrap();
    let build = |dir: &Path, path: &str| {
        Command::new(env!("CARGO_BIN_EXE_vigil"))
            .args(["--color", "never", "build", path])
            .current_dir(dir)
            .output()
            .unwrap()
    };
    let out = build(&dir, ".");
    assert_ok(&out, "the project");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.starts_with("built `app` → "), "{stdout}");
    std::fs::remove_file(dir.join("vigil.toml")).unwrap();
    let out = build(&dir.join("src"), "main.vg");
    assert_ok(&out, "a lone file");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.starts_with("built `main.vg` → "), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}
