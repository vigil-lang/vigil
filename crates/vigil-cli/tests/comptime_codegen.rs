//! what the code generator keeps of comptime code: no comptime function is
//! compiled, a `comptime:` block leaves constants, and only the tables the
//! programs read go in `.rodata`

use std::path::{Path, PathBuf};
use std::process::Command;

use vigil_codegen_llvm::clang::Clang;

/// a fresh `.vg` file with this source
fn file(name: &str, source: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vigil-comptime-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.vg"));
    std::fs::write(&path, source).unwrap();
    path
}

/// running `vigil build` on a file, to an output next to it with this
/// extension; it gives the output, `None` when there is no clang for an
/// object
fn build(path: &Path, emit: &str, extension: &str) -> Option<Vec<u8>> {
    if emit == "object" && Clang::discover().is_err() {
        println!("skipping the object of {}: no clang", path.display());
        return None;
    }
    let output = path.with_extension(extension);
    let out = Command::new(env!("CARGO_BIN_EXE_vigil"))
        .args(["--color", "never", "build", "--emit", emit])
        .arg(path)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "vigil build --emit {emit} {}:\n{}{}",
        path.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    Some(std::fs::read(output).unwrap())
}

fn ir(path: &Path) -> String {
    String::from_utf8(build(path, "ir", "ll").unwrap()).unwrap()
}

#[test]
fn comptime_function_of_a_block_is_not_compiled() {
    let path = file(
        "block",
        "from std.net import XdpContext, XdpAction\n\
         \n\
         comptime def half(x: f64) -> i64:\n    return i64(x / 2.0)\n\
         \n\
         @xdp\n\
         def prog(ctx: XdpContext) -> XdpAction:\n\
         \x20   comptime:\n        h = half(9.0)\n\
         \x20   if i64(ctx.len()) < h:\n        return XdpAction.DROP\n\
         \x20   return XdpAction.PASS\n",
    );
    let ir = ir(&path);
    assert!(!ir.contains("@\"half\""), "`half` is compiled:\n{ir}");
    assert!(!ir.contains("double"), "floats are compiled:\n{ir}");
    assert!(
        ir.contains("store i64 4,"),
        "`h` is not the constant 4:\n{ir}"
    );
    build(&path, "object", "o");
}

#[test]
fn rodata_has_only_the_tables_read() {
    // std.sketch has tables of its own, which this program does not read
    let path = file(
        "tables",
        "from std.net import XdpContext, XdpAction\n\
         from std.sketch import CountMin\n\
         \n\
         comptime def squares() -> u32[8]:\n\
         \x20   t: u32[8] = [0, 0, 0, 0, 0, 0, 0, 0]\n\
         \x20   for i in range(8):\n        t[i] = u32(i * i + 1000)\n\
         \x20   return t\n\
         \n\
         const SQUARES: u32[8] = squares()\n\
         \n\
         @xdp\n\
         def prog(ctx: XdpContext) -> XdpAction:\n\
         \x20   i = usize(ctx.len()) % 8\n\
         \x20   if SQUARES[i] > 1040:\n        return XdpAction.DROP\n\
         \x20   return XdpAction.PASS\n",
    );
    let ir = ir(&path);
    let tables: Vec<&str> = ir
        .lines()
        .filter(|l| l.contains("section \".rodata\""))
        .collect();
    assert_eq!(
        tables.len(),
        1,
        "tables in `.rodata`:\n{}",
        tables.join("\n")
    );
    assert!(tables[0].contains("[8 x i32] [i32 1000, i32 1001, i32 1004,"));
    build(&path, "object", "o");

    let hello = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/hello_kprobe.vg");
    let ir = ir_of_example(&hello);
    assert!(!ir.contains(".rodata"), "`.rodata` in hello_kprobe:\n{ir}");
}

/// the IR of an example, written to a fresh directory
fn ir_of_example(example: &Path) -> String {
    let source = std::fs::read_to_string(example).unwrap();
    ir(&file("example", &source))
}
