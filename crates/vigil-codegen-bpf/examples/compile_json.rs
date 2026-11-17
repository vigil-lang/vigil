//! compiling a Core program from JSON and printing the code in hex, for the
//! model or for XDP in Linux with `kernel`:
//!
//! `cargo run -p vigil-codegen-bpf --example compile_json FILE [kernel]`
//!
//! the output has the format of the Lean compiler of the full Vigil, to
//! compare the two byte for byte

fn main() {
    let path = std::env::args().nth(1).expect("a JSON file");
    let kernel = std::env::args().nth(2).as_deref() == Some("kernel");
    let program: vigil_core::Program =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    match vigil_codegen_bpf::compile(&program, kernel) {
        Ok(code) => println!(
            "{}",
            code.iter().map(|b| format!("{b:02x}")).collect::<String>()
        ),
        Err(e) => println!("error: {e}"),
    }
}
