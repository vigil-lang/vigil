//! `vigil --version` prints the name of the compiler and its version, and
//! `vigil` without a command is an error, not a version or a help

use std::process::{Command, Output};

fn vigil(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vigil"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn prints_the_version() {
    let expected = format!(
        "Vigil Programming Language Compiler - version {}\n",
        env!("CARGO_PKG_VERSION")
    );
    for flag in ["--version", "-V"] {
        let out = vigil(&[flag]);
        assert!(out.status.success(), "{flag}");
        assert_eq!(String::from_utf8_lossy(&out.stdout), expected, "{flag}");
    }
}

#[test]
fn requires_a_command() {
    for args in [&[][..], &["-O0"][..]] {
        let out = vigil(args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("a command is required"), "{err}");
    }
}
