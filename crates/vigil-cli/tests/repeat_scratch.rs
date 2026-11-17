//! a local initialized with a large repeat, `[x] * n`, is filled in place,
//! so no temporary takes as much scratch again, and in a loop, because eBPF
//! has no `memset` to call

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use vigil_codegen_llvm::clang::Clang;

/// a program with a table of `n` copies of `element` (a Vigil expression),
/// which reads back one of them
fn program(element: &str, n: u32) -> String {
    format!(
        "\
from std.net import XdpContext, XdpAction, Byte

@xdp
def p(ctx: XdpContext) -> XdpAction:
    first = ctx.parse[Byte](0)?.value
    table = [{element}] * {n}
    table[u32(first) % {n}] = 1
    if table[u32(first) % 7] == 1:
        return XdpAction.DROP
    return XdpAction.PASS
"
    )
}

/// `vigil build` of `code`, with `extra` arguments: whether it succeeded,
/// the output file and stderr
fn build(code: &str, extra: &[&str]) -> (bool, String, String) {
    static BUILDS: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "vigil-repeat-scratch-{}-{}",
        std::process::id(),
        BUILDS.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("table.vg");
    std::fs::write(&source, code).unwrap();
    let output = dir.join("table.out");
    let out = Command::new(env!("CARGO_BIN_EXE_vigil"))
        .args(["--color", "never", "build"])
        .args(extra)
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    let text = std::fs::read_to_string(&output).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.success(),
        text,
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn a_repeat_is_counted_once() {
    let (ok, _, stderr) = build(&program("u8(0)", 40000), &[]);
    assert!(!ok);
    assert!(
        stderr.contains("40000 bytes of locals"),
        "the table should be counted once:\n{stderr}"
    );
}

#[test]
fn large_fills_build() {
    if Clang::discover().is_err() {
        println!("skipping: no clang");
        return;
    }
    for (element, n) in [
        ("u8(0)", 30000),
        ("u8(3)", 3000),
        ("u64(0)", 3000),
        ("u32(0x01020304)", 1001),
        ("u16(first)", 3001),
    ] {
        let (ok, _, stderr) = build(&program(element, n), &[]);
        assert!(ok, "`[{element}] * {n}` should build:\n{stderr}");
    }
}

#[test]
fn ir_of_large_fills() {
    if Clang::discover().is_err() {
        println!("skipping: no clang");
        return;
    }
    // 3 in every byte, 8 bytes at a time, in a loop
    let (ok, ir, stderr) = build(&program("u8(3)", 3000), &["--emit", "ir"]);
    assert!(ok, "{stderr}");
    assert!(
        ir.contains("store volatile i64 217020518514230019"),
        "the bytes should be written by words:\n{ir}"
    );
    assert!(!ir.contains("call void @llvm.memset"), "no memset:\n{ir}");
    // 1001 words of 4 bytes: 500 stores of 8 bytes, then one of 4
    let (ok, ir, stderr) = build(&program("u32(0x01020304)", 1001), &["--emit", "ir"]);
    assert!(ok, "{stderr}");
    assert!(
        ir.contains("store volatile i64 72623859723010820")
            && ir.contains("store volatile i32 16909060"),
        "the words and the rest should be written:\n{ir}"
    );
}

/// a value over 1 KiB stored in an array map (its key and value slots are
/// zeroed first), returned as an option (a `None` is zeroed) and copied:
/// loops too, not one instruction per byte
const LARGE_VALUES: &str = "\
from std.net import XdpContext, XdpAction, Byte

map big: Array[u8[2000], 4]

def pick(x: u8) -> Option[u8[2000]]:
    if x == 3:
        return None
    return Some([x] * 2000)

@xdp
def p(ctx: XdpContext) -> XdpAction:
    first = ctx.parse[Byte](0)?.value
    table = [u8(2)] * 2000
    table[u32(first) % 2000] = first
    big.insert(1, table)
    got = pick(first)?
    grid = [[first] * 300] * 20
    if got[0] == 9 or grid[3][7] == 1:
        return XdpAction.DROP
    return XdpAction.PASS
";

#[test]
fn large_values_build() {
    if Clang::discover().is_err() {
        println!("skipping: no clang");
        return;
    }
    let (ok, _, stderr) = build(LARGE_VALUES, &[]);
    assert!(ok, "{stderr}");
    let (ok, ir, stderr) = build(LARGE_VALUES, &["--emit", "ir"]);
    assert!(ok, "{stderr}");
    // the 4-byte key may still be zeroed with a `memset`
    let large_memset = ir.lines().any(|l| {
        l.contains("call void @llvm.memset")
            && l.split("i8 0, i64 ")
                .nth(1)
                .and_then(|rest| rest.split(',').next())
                .and_then(|size| size.parse::<u64>().ok())
                .is_some_and(|size| size > 256)
    });
    assert!(!large_memset, "no large memset:\n{ir}");
    let (ok, asm, stderr) = build(LARGE_VALUES, &["--emit", "asm"]);
    assert!(ok, "{stderr}");
    let insns = asm.lines().filter(|l| l.contains(": ")).count();
    assert!(
        insns < 1000,
        "the copies should be loops, not {insns} instructions"
    );
}
