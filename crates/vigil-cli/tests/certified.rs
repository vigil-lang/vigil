//! `vigil core` and `vigil simulate` on single files: they agree with the
//! certified backend on what is in Vigil Core, and name the verdicts of the
//! program kind

use std::path::PathBuf;
use std::process::{Command, Output};

/// writing `source` to a fresh `.vg` file
fn file(name: &str, source: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vigil-certified-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.vg"));
    std::fs::write(&path, source).unwrap();
    path
}

fn vigil(args: &[&str]) -> (Output, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_vigil"))
        .args(["--color", "never"])
        .args(args)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out, text)
}

// an abort is `TC_ACT_OK` (0) for TC, and a drop `TC_ACT_SHOT` (2), not
// the XDP names
#[test]
fn simulate_names_tc_verdicts() {
    let path = file(
        "tc",
        "\
from std.net import TcContext, TcAction, Byte

@tc
def pick(ctx: TcContext) -> TcAction:
    a = [u8(1), 2]
    if a[ctx.parse[Byte](0)?.value] == 2:
        return TcAction.SHOT
    return TcAction.SHOT
",
    );
    let path = path.to_str().unwrap();
    let (out, text) = vigil(&["simulate", path, "--packet", "01"]);
    assert!(out.status.success(), "{text}");
    assert!(text.contains("returned 2 (TC_ACT_SHOT)"), "{text}");
    let (out, text) = vigil(&["simulate", path, "--packet", "05"]);
    assert!(out.status.success(), "{text}");
    assert!(text.contains("returned 0 (TC_ACT_OK)"), "{text}");
}

// a whole array assigned to another is copied element by element
#[test]
fn core_exports_array_copies() {
    let path = file(
        "copy",
        "\
from std.net import XdpContext, XdpAction, Byte

@xdp
def copies(ctx: XdpContext) -> XdpAction:
    n = ctx.parse[Byte](0)?.value
    a = [n, n]
    b = a
    b[0] = 9
    if a[0] == 5 and b[0] == 9:
        return XdpAction.DROP
    return XdpAction.PASS
",
    );
    let path = path.to_str().unwrap();
    let (out, text) = vigil(&["core", path]);
    assert!(out.status.success(), "{text}");
    assert!(text.contains("v2[u32(1)] = v1[u32(1)]"), "{text}");
    let (out, text) = vigil(&["simulate", path, "--packet", "05"]);
    assert!(out.status.success(), "{text}");
    assert!(text.contains("(XDP_DROP)"), "{text}");
}

// `core` rejects a package with a map as `build --certified` does, with
// the same code
#[test]
fn core_rejects_maps() {
    let path = file(
        "map",
        "\
from std.net import XdpContext, XdpAction

map hits: Array[u64, 1]

@xdp
def pass_all(ctx: XdpContext) -> XdpAction:
    return XdpAction.PASS
",
    );
    let (out, text) = vigil(&["core", path.to_str().unwrap()]);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("[E0501] Error: Vigil Core has no maps"),
        "{text}"
    );
}

// a construct outside Vigil Core is `E0501`, as in `build --certified`
#[test]
fn core_rejects_with_the_certified_code() {
    let path = file(
        "range",
        "\
from std.net import XdpContext, XdpAction

@xdp
def count(ctx: XdpContext) -> XdpAction:
    n: u8 = 0
    for i in range(0, 4):
        n += 1
    return XdpAction.PASS
",
    );
    let (out, text) = vigil(&["core", path.to_str().unwrap()]);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("[E0501] Error: `count` is not in Vigil Core"),
        "{text}"
    );
}

// `--certified --emit ir` is rejected before the analysis, so the type
// error of the file is not reported
#[test]
fn certified_has_no_ir() {
    let path = file("noir", "def f() -> u32:\n    return true\n");
    let (out, text) = vigil(&[
        "build",
        "--certified",
        "--emit",
        "ir",
        path.to_str().unwrap(),
    ]);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("[E0010] Error: `--certified` cannot emit LLVM IR"),
        "{text}"
    );
    assert!(text.contains("--emit asm"), "{text}");
    assert!(!text.contains("E0400"), "{text}");
}

const TWO_PROGRAMS: &str = "\
from std.net import XdpContext, XdpAction

@xdp
def first(ctx: XdpContext) -> XdpAction:
    return XdpAction.PASS

@xdp
def second(ctx: XdpContext) -> XdpAction:
    return XdpAction.DROP
";

// a bad packet has its own code and points at the bad digit
#[test]
fn simulate_rejects_bad_packets() {
    let path = file("packet", TWO_PROGRAMS);
    let path = path.to_str().unwrap();
    let (out, text) = vigil(&["simulate", path, "--packet", "aa bq", "--program", "first"]);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains(
            "[E0011] Error: invalid `--packet`: `q` at position 5 is not a hexadecimal digit"
        ),
        "{text}"
    );
    let (out, text) = vigil(&["simulate", path, "--packet", "abc", "--program", "first"]);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("[E0011]") && text.contains("odd number"),
        "{text}"
    );
}

// a missing or unknown `--program` is a wrong option, not an I/O error
#[test]
fn chooses_programs_by_name() {
    let path = file("choose", TWO_PROGRAMS);
    let path = path.to_str().unwrap();
    let (out, text) = vigil(&["simulate", path, "--packet", "00"]);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("[E0010] Error: this project has several programs: `first`, `second`"),
        "{text}"
    );
    let (out, text) = vigil(&["core", path, "--program", "third"]);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("[E0010] Error: there is no program `third`"),
        "{text}"
    );
    assert!(
        text.contains("the programs are `first`, `second`"),
        "{text}"
    );
    let (out, text) = vigil(&["simulate", path, "--packet", "00", "--program", "second"]);
    assert!(out.status.success(), "{text}");
    assert!(text.contains("returned 1 (XDP_DROP)"), "{text}");
}
