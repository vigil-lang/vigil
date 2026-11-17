//! this example prints the listing of a section of an eBPF object, as
//! [`vigil_codegen_llvm::disasm::listing`] writes it:
//!
//! ```text
//! cargo run -p vigil-codegen-llvm --example disasm -- file.o section
//! ```

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("object file");
    let file = object::File::parse(&*bytes).expect("ELF");
    use object::{Object, ObjectSection};
    let section = file.section_by_name(&args[2]).expect("section");
    print!(
        "{}",
        vigil_codegen_llvm::disasm::listing(section.data().unwrap())
    );
}
