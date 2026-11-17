//! what `vigil build --emit ir` gives: with clang, the IR the object is
//! built from, so a program that needs more stack shows its locals in
//! scratch

use std::process::Command;

use vigil_codegen_llvm::clang::Clang;

/// a program whose 640 bytes of locals do not fit in the 512 bytes of stack
const BIG: &str = "\
from std.net import XdpContext, XdpAction, Byte

@xdp
def big(ctx: XdpContext) -> XdpAction:
    first = ctx.parse[Byte](0)?.value
    table = [u64(0)] * 80
    for i in range(80):
        table[i] = u64(i) * u64(first)
    if table[u32(first) % 80] > 1000:
        return XdpAction.DROP
    return XdpAction.PASS
";

#[test]
fn ir_of_a_program_in_scratch() {
    if Clang::discover().is_err() {
        println!("skipping: no clang, so `--emit ir` gives the first pass");
        return;
    }
    let dir = std::env::temp_dir().join(format!("vigil-emit-ir-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("big.vg");
    std::fs::write(&source, BIG).unwrap();
    let ir = dir.join("big.ll");
    let out = Command::new(env!("CARGO_BIN_EXE_vigil"))
        .args(["--color", "never", "build", "--emit", "ir"])
        .arg(&source)
        .arg("-o")
        .arg(&ir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let ir = std::fs::read_to_string(&ir).unwrap();
    assert!(
        ir.contains("_vigil_scratch_big"),
        "the IR should be the second pass, with the locals in scratch"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
