//! this crate exports the compiler to WebAssembly for the web playground.
//! Each export returns the JSON of the matching operation of
//! `vigil_driver::playground`. The operations live in the driver so they
//! are built and tested natively.
//!
//! `compiler/build-wasm.sh` builds it into `page/pkg/`.

use vigil_driver::playground::{self, Settings};
use wasm_bindgen::prelude::*;

/// building the [`Settings`] from the strings the page passes
fn settings(kernel: &str, arch: &str, optimize: bool) -> Settings {
    Settings {
        kernel: kernel.to_string(),
        arch: arch.to_string(),
        optimize,
    }
}

/// the compiler version, the one of this crate
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// checking and compiling a program: diagnostics, programs, maps, LLVM IR
/// and the eBPF code of the certified backend
#[wasm_bindgen]
pub fn compile(source: &str, kernel: &str, arch: &str, optimize: bool) -> String {
    playground::compile(source, &settings(kernel, arch, optimize)).to_string()
}

/// running a Vigil Core program, compiled by the certified backend, on a
/// packet given in hex
#[wasm_bindgen]
pub fn run(source: &str, kernel: &str, arch: &str, program: &str, packet: &str) -> String {
    playground::run(source, &settings(kernel, arch, true), program, packet).to_string()
}

/// formatting a program, as `vigil fmt` does
#[wasm_bindgen]
pub fn format(source: &str) -> String {
    playground::format(source).to_string()
}

/// the sources of the standard library, for the playground docs
#[wasm_bindgen]
pub fn stdlib() -> String {
    playground::stdlib().to_string()
}
