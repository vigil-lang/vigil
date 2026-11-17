//! `vigil build` reports the kernel limits at the program: length, stack,
//! frames of calls, and what clang cannot compile or takes too long on.
//! The tests that need clang skip when there is none.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use vigil_codegen_llvm::clang::Clang;

/// a fresh directory for one test
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vigil-limits-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// building `source` as a lone file, with these environment variables
fn build(name: &str, source: &str, env: &[(&str, &Path)]) -> (Output, String) {
    let dir = scratch(name);
    let file = dir.join(format!("{name}.vg"));
    std::fs::write(&file, source).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_vigil"))
        .args(["--color", "never", "build"])
        .arg(&file)
        .arg("-o")
        .arg(dir.join("out.o"))
        .envs(env.iter().map(|&(k, v)| (k, v.as_os_str())))
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    (out, text)
}

fn has_clang() -> bool {
    match Clang::discover() {
        Ok(_) => true,
        Err(err) => {
            println!("skipping: {err}");
            false
        }
    }
}

/// an XDP program whose body is `body`, indented by four spaces
fn xdp(body: &str) -> String {
    let body: String = body.lines().map(|l| format!("    {l}\n")).collect();
    format!(
        "from std.net import XdpContext, XdpAction, Byte\n\n@xdp\ndef p(ctx: XdpContext) -> XdpAction:\n{body}    return XdpAction.PASS\n"
    )
}

#[test]
fn the_stack_shown_is_the_one_checked() {
    if !has_clang() {
        return;
    }
    // the program touches less than 32 bytes of stack, but the verifier
    // rounds a frame up to 32
    let (out, text) = build(
        "shown",
        &xdp("x = ctx.parse[Byte](0)?.value\nif x > 1:\n    return XdpAction.DROP"),
        &[],
    );
    assert!(out.status.success(), "{text}");
    assert!(text.contains(" B of stack"), "{text}");
    let stack: usize = text
        .split(" B of stack")
        .next()
        .and_then(|s| s.rsplit(", ").next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no stack in {text}"));
    assert!(stack >= 32 && stack % 32 == 0, "{text}");
}

#[test]
fn nested_callbacks_exceed_the_frames() {
    if !has_clang() {
        return;
    }
    // eight nested long loops are eight `bpf_loop` callbacks below the
    // program: nine frames
    let mut body = String::from("total: u32 = 0\n");
    for depth in 0..8 {
        let _ = std::fmt::Write::write_fmt(
            &mut body,
            format_args!("{}for i{depth} in range(100):\n", "    ".repeat(depth)),
        );
    }
    body.push_str(&format!("{}total += 1\n", "    ".repeat(8)));
    body.push_str("if total > 1:\n    return XdpAction.DROP\n");
    let (out, text) = build("frames", &xdp(&body), &[]);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("[E0501] Error: `p` nests calls too deeply"),
        "{text}"
    );
    assert!(
        text.contains("9 frames in a chain of calls, at most 8"),
        "{text}"
    );
}

#[test]
fn an_unrolled_loop_that_overflows_the_stack_is_a_limit() {
    if !has_clang() {
        return;
    }
    let body = "total: u32 = 0\n@unroll\nfor i in range(300):\n    total += u32(ctx.parse[Byte](u32(i))?.value)\nif total > 1:\n    return XdpAction.DROP";
    let (out, text) = build("unrolled", &xdp(body), &[]);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("[E0501] Error: `p` uses too much stack"),
        "{text}"
    );
    assert!(text.contains("`@no_unroll`"), "{text}");
    assert!(!text.contains("E0500"), "{text}");
}

#[test]
fn a_local_larger_than_scratch_never_reaches_clang() {
    if !has_clang() {
        return;
    }
    // clang took minutes on this local before
    let body = "seen = [u8(0)] * 40000\nfirst = ctx.parse[Byte](0)?.value\nseen[first] = 1\nif seen[3] > 0:\n    return XdpAction.DROP";
    let started = std::time::Instant::now();
    let (out, text) = build("huge", &xdp(body), &[]);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("[E0501] Error: `p` needs too much memory"),
        "{text}"
    );
    assert!(started.elapsed().as_secs() < 30, "{text}");
}

#[cfg(unix)]
#[test]
fn a_clang_that_hangs_is_killed() {
    use std::os::unix::fs::PermissionsExt;
    let dir = scratch("fake-clang");
    let clang = dir.join("clang");
    std::fs::write(
        &clang,
        "#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'clang version 20.1.0'; exit 0; fi\nexec sleep 30\n",
    )
    .unwrap();
    std::fs::set_permissions(&clang, std::fs::Permissions::from_mode(0o755)).unwrap();
    let started = std::time::Instant::now();
    let (out, text) = build(
        "hang",
        &xdp(""),
        &[
            ("VIGIL_CLANG", &clang),
            ("VIGIL_CLANG_TIMEOUT", Path::new("1")),
        ],
    );
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("[E0500] Error: clang did not finish within 1 s"),
        "{text}"
    );
    assert!(text.contains("VIGIL_CLANG_TIMEOUT"), "{text}");
    assert!(started.elapsed().as_secs() < 20, "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_length_limit_says_what_it_checks() {
    if !has_clang() {
        return;
    }
    let dir = scratch("length");
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(
        dir.join("vigil.toml"),
        "[package]\nname = \"big\"\n\n[target]\nkernels = [\"6.1\"]\nprivileged = false\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/main.vg"),
        "from std.net import XdpContext, XdpAction\nfrom std.bpf import prandom_u32\n\n@xdp\n\
         def big(ctx: XdpContext) -> XdpAction:\n    total: u32 = 0\n    @unroll\n    \
         for i in range(2100):\n        total += prandom_u32()\n    if total > 1:\n        \
         return XdpAction.DROP\n    return XdpAction.PASS\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_vigil"))
        .args(["--color", "never", "build", "src/main.vg", "-o", "big.o"])
        .current_dir(&dir)
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stderr).into_owned();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("instructions; the kernel loads programs of at most 4096 instructions"),
        "{text}"
    );
}
