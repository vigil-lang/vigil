//! the layout of `@network_order` structs: a header is byte-exact, so its
//! fields follow each other without padding, in the sizes the analysis
//! checks, the offsets userspace decodes, and the code that reads packets

use std::process::Command;

use vigil_codegen_llvm::clang::Clang;
use vigil_codegen_llvm::desc::TypeDesc;
use vigil_sema::ModuleInput;
use vigil_sema::ty::TyKind;

/// a header where C would add padding: `b` at 2 and 8 bytes in all
const ODD: &str = "\
from std.net import XdpContext, XdpAction

@network_order
struct H:
    a: u8
    b: u16
    c: u32

struct Holder:
    tag: u8
    header: H
    count: u64

map seen: Hash[u32, Holder](max_entries=8)

@xdp
def prog(ctx: XdpContext) -> XdpAction:
    h = ctx.parse[H](0)?
    if h.b == 0x0203 and h.c == 0x04050607:
        return XdpAction.DROP
    return XdpAction.PASS
";

/// the analysis of a program with the standard library
fn analyze(code: &str) -> vigil_sema::Analysis {
    let mut sources = vigil_diagnostics::SourceMap::new();
    let mut inputs = vec![ModuleInput {
        file: sources.add("main.vg", code.to_string()),
        path: vec!["main".into()],
        tree: vigil_syntax::parse(code).tree(),
        std: false,
    }];
    for (path, name, code) in vigil_sema::stdlib::MODULES {
        inputs.push(ModuleInput {
            file: sources.add(*name, *code),
            path: path.split('.').map(String::from).collect(),
            tree: vigil_syntax::parse(code).tree(),
            std: true,
        });
    }
    let (analysis, diags) = vigil_sema::analyze(&inputs);
    assert!(!diags.has_errors(), "the program should check");
    analysis
}

/// the size and the field offsets of the struct `name`, as compiled code
/// lays it out
fn layout(analysis: &vigil_sema::Analysis, name: &str) -> (u64, Vec<u64>) {
    let (id, _) = analysis
        .package
        .structs
        .iter()
        .find(|(_, s)| s.name.text == name)
        .unwrap_or_else(|| panic!("no struct {name}"));
    let ty = analysis.types.lookup(&TyKind::Struct(id)).unwrap();
    let (desc, size) = vigil_codegen_llvm::describe(analysis, ty);
    let TypeDesc::Struct { fields, .. } = desc else {
        panic!("{name} is not described as a struct");
    };
    (size, fields.iter().map(|(_, offset, _)| *offset).collect())
}

#[test]
fn headers_have_no_padding() {
    let analysis = analyze(ODD);
    assert_eq!(layout(&analysis, "H"), (7, vec![0, 1, 3]));
    // inside a C struct, a header is aligned to 1
    assert_eq!(layout(&analysis, "Holder"), (16, vec![0, 1, 8]));
}

#[test]
fn std_headers_keep_their_offsets() {
    let analysis = analyze(ODD);
    assert_eq!(layout(&analysis, "EthHdr"), (14, vec![0, 6, 12]));
    assert_eq!(
        layout(&analysis, "Ipv4Hdr"),
        (20, vec![0, 1, 2, 4, 6, 8, 9, 10, 12, 16])
    );
    assert_eq!(layout(&analysis, "Ipv6Hdr"), (40, vec![0, 4, 6, 7, 8, 24]));
    assert_eq!(
        layout(&analysis, "TcpHdr"),
        (20, vec![0, 2, 4, 8, 12, 14, 16, 18])
    );
    assert_eq!(layout(&analysis, "UdpHdr"), (8, vec![0, 2, 4, 6]));
}

/// the output of `vigil build --emit <what>` for `code`
fn emit(code: &str, what: &str) -> String {
    let dir =
        std::env::temp_dir().join(format!("vigil-network-order-{what}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("odd.vg");
    std::fs::write(&source, code).unwrap();
    let output = dir.join("odd.out");
    let out = Command::new(env!("CARGO_BIN_EXE_vigil"))
        .args(["--color", "never", "build", "--emit", what])
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&output).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    text
}

#[test]
fn ir_of_a_packed_header() {
    let ir = emit(ODD, "ir");
    assert!(
        ir.contains("= type <{ i8, i16, i32 }>"),
        "the header should be a packed type:\n{ir}"
    );
    // `parse` checks 7 bytes, and `b` is read from an odd address
    assert!(
        ir.contains("i64 7"),
        "the bounds check should be of 7 bytes:\n{ir}"
    );
    assert!(
        ir.lines()
            .any(|l| l.contains("load i16") && l.contains("align 1")),
        "`b` should be loaded with alignment 1:\n{ir}"
    );
}

#[test]
fn asm_of_a_packed_header() {
    if Clang::discover().is_err() {
        println!("skipping: no clang");
        return;
    }
    let asm = emit(ODD, "asm");
    // `b` is bytes 1 and 2, `c` bytes 3 to 6, read a byte at a time
    for offset in 1..=6 {
        assert!(
            asm.lines()
                .any(|l| l.contains("= *(u8 *)(r") && l.ends_with(&format!(" +{offset})"))),
            "byte {offset} should be read:\n{asm}"
        );
    }
    assert!(
        !asm.lines()
            .any(|l| l.contains("= *(u16 *)(r") && !l.contains("(r10")),
        "no misaligned 16-bit read:\n{asm}"
    );
    assert!(
        asm.contains("+= 7"),
        "the header should end 7 bytes in:\n{asm}"
    );
}
