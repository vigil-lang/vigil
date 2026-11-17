#!/usr/bin/env sh
# Builds the compiler for the browser and writes it to ../page/pkg/:
# vigil_wasm.js (ES module) and vigil_wasm_bg.wasm.
#
# Needs the wasm32-unknown-unknown target (`rustup target add
# wasm32-unknown-unknown`) and wasm-bindgen-cli of the same version as the
# wasm-bindgen crate in Cargo.toml (`cargo install wasm-bindgen-cli
# --version 0.2.126`). wasm-opt (binaryen), when found, shrinks the module.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out="$here/../page/pkg"
cd "$here"
cargo build --release --target wasm32-unknown-unknown -p vigil-wasm
wasm-bindgen --target web --no-typescript --out-dir "$out" \
    target/wasm32-unknown-unknown/release/vigil_wasm.wasm
if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -Os --enable-bulk-memory --enable-nontrapping-float-to-int \
        "$out/vigil_wasm_bg.wasm" -o "$out/vigil_wasm_bg.wasm"
fi
ls -l "$out"
