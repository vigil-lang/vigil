# The Vigil compiler

Vigil is a statically typed language that compiles to eBPF. Its syntax is close
to Python's. Its type checker also enforces the rules of the kernel verifier:
bounded loops, checked packet accesses, and stack and program-size limits. This
repository holds the compiler: the `vigil` command and a WebAssembly build for
the playground of the website.

```python
from std.net import XdpContext, XdpAction, Byte

@xdp
def dns_drop(ctx: XdpContext) -> XdpAction:
    if ctx.parse[Byte](12)?.value != 8 or ctx.parse[Byte](13)?.value != 0:
        return XdpAction.PASS
    if ctx.parse[Byte](14)?.value != 0x45 or ctx.parse[Byte](23)?.value != 17:
        return XdpAction.PASS
    port = u16(ctx.parse[Byte](36)?.value) << 8 | u16(ctx.parse[Byte](37)?.value)
    if port == 53:
        return XdpAction.DROP
    return XdpAction.PASS
```

- Website, tutorial and playground: <https://vigil-lang.github.io>
- Standard library: <https://github.com/vigil-lang/std>
- Issues: <https://github.com/vigil-lang/vigil/issues>

## Installing

```sh
curl -fsSL https://vigil-lang.github.io/install.sh | sh
```

The script downloads the binary of the latest release for the machine: Linux or
macOS, on x86_64 or aarch64. It checks the binary's SHA-256 and puts it in
`~/.local/bin`.

### From source

The compiler embeds the standard library from `../std`, so the two repositories
are cloned side by side:

```sh
git clone https://github.com/vigil-lang/vigil compiler
git clone https://github.com/vigil-lang/std std
cd compiler
cargo build --release          # target/release/vigil
```

You need Rust 1.85 or newer (edition 2024). Building eBPF objects
(`--emit object` and `--emit asm`) also needs clang 15 or newer, found on the
`PATH` or in `$VIGIL_CLANG`. The `check`, `--emit ir`, `--certified` and
`simulate` paths do not need clang.

## Quick start

```sh
vigil check examples/hello_kprobe.vg            # types and the verifier's rules
vigil build examples/hello_kprobe.vg            # an eBPF object, through clang
vigil build examples/xdp_certified.vg --certified   # without LLVM or clang
vigil simulate examples/xdp_certified.vg \
  --packet 0000000000000000000000000800450000000000000000110000000000000000000000000035
# `dns_drop` returned 1 (XDP_DROP) after 65 instructions
```

The `examples/` directory has kprobes, uprobes, tracepoints, LSM, XDP and TC
programs, multi-stage pipelines with tail calls, and an integer neural network
for flow classification.

## The `vigil` command

```sh
vigil check    [PATH] [-v]                       # check without generating code
vigil build    [PATH] [-o OUT] [--emit object|ir|asm] [--certified]
vigil core     [PATH] [--program P] [--json]     # an XDP/TC program as Vigil Core
vigil simulate PATH --packet HEX [--program P]   # run Vigil Core on a packet, no kernel
vigil update   [PATH]                            # resolve the git dependencies again
vigil fmt      [PATHS] [--check | --diff]
vigil parse    FILE [--tree [--trivia]]
```

- `PATH` is a `.vg` file or a project directory with a `vigil.toml`, and
  defaults to the current directory.
- `-O0` turns off the optimizations of the Vigil code generator. LLVM still runs
  with `-O2`.
- `--color auto|always|never` controls the colors of the diagnostics.
- A build fails when it would exceed a limit of the target kernels:
  - program length;
  - stack;
  - call depth;
  - tail calls.

  It also fails when clang runs longer than `$VIGIL_CLANG_TIMEOUT` seconds (120
  by default).

## Projects and dependencies

A project is a directory with a `vigil.toml` and its sources in `src/`:

```toml
[package]
name = "flowmon"

[target]
kernels = ["linux-6.1"]

[dependencies]
std    = { path = "../std" }                                  # instead of the built-in std
sketch = { path = "libs/sketch" }                             # a directory
netx   = { github = "acme/netx", tag = "v0.2.0" }             # a GitHub repository
flows  = { git = "https://example.org/flows.git", branch = "main" }
pinned = { github = "acme/pinned", rev = "<40-digit commit hash>" }
```

- **Module names.** A dependency's modules are imported under its name, so
  `netx/src/parse.vg` is imported with `from netx.parse import ...`.
- **Replacing `std`.** The `std` dependency replaces the standard library
  entirely. It must have every module the programs import, so start from a copy
  of `vigil-lang/std`. `kernel` cannot be a dependency.
- **Library layout.** A library with a `vigil.toml` takes its sources from
  `src/` and may have dependencies of its own. In a library without one, every
  `.vg` file under its directory is a module.
- **One namespace.** Names are shared by the whole build: two different
  libraries under one name are an error. Only the project chooses `std`.
- **Git sources.** A git dependency takes `branch`, `tag` or `rev` (a full
  commit hash); without one, it uses the default branch.
- **The lock file.** The commits are recorded in `vigil.lock`, which should be
  committed. Later builds use those commits without contacting the remote, and
  `vigil update` resolves them again.
- **Cache.** Checkouts are cached in `$VIGIL_HOME`, else
  `$XDG_CACHE_HOME/vigil`, else `~/.cache/vigil`. Fetching needs `git`.

## The certified backend

XDP and TC programs written in Vigil Core, the subset formalized in Lean, can be
compiled without LLVM by `vigil-codegen-bpf`. This backend produces, instruction
for instruction, the code of the formalized compiler. Vigil Core covers:

- scalars;
- local arrays;
- bounded loops;
- reads of the packet at checked offsets.

`vigil core` prints a program as Vigil Core, or explains why it is outside the
subset. `vigil simulate` runs the certified code on a packet in an eBPF
interpreter, with no kernel and no root.

## Repository layout

| Crate                |                                                                                                                 |
| -------------------- | --------------------------------------------------------------------------------------------------------------- |
| `vigil-syntax`       | lexer, parser, lossless syntax trees, formatter                                                                 |
| `vigil-sema`         | lowering to HIR, name resolution, type checking, generics, compile-time evaluation                              |
| `vigil-codegen-llvm` | code generation to LLVM IR, compiled to eBPF by clang                                                           |
| `vigil-core`         | Vigil Core, exported from typed programs                                                                        |
| `vigil-codegen-bpf`  | the certified backend: Vigil Core to eBPF instructions and ELF                                                  |
| `vigil-driver`       | the pipeline without a file system, shared by the CLI and WASM; the playground operations; the eBPF interpreter |
| `vigil-diagnostics`  | source maps, spans, diagnostics and their codes                                                                 |
| `vigil-target`       | kernel profiles, verifier limits, `vigil.toml`                                                                  |
| `vigil-btf`          | the BTF reader, for `from kernel import ...`                                                                    |
| `vigil-cli`          | the `vigil` command, projects and dependencies                                                                  |
| `vigil-wasm`         | the compiler for the browser (`wasm-bindgen`)                                                                   |

| File            |                                                                       |
| --------------- | --------------------------------------------------------------------- |
| `std.ref`       | the branch or tag of `vigil-lang/std` that CI and releases build with |
| `build-wasm.sh` | builds `vigil-wasm` into `../page/pkg/`                               |
| `examples/`     | example programs, checked and built by the tests                      |

## Development

```sh
cargo test --workspace         # unit, snapshot and command tests
cargo fmt --all
cargo doc --no-deps --document-private-items
```

Some command tests build objects and are skipped when clang is not found.

### WebAssembly

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.126
./build-wasm.sh
```

The script builds `vigil-wasm` and runs `wasm-bindgen --target web`. It writes
`vigil_wasm.js` and `vigil_wasm_bg.wasm` to `../page/pkg/`. The version of
`wasm-bindgen-cli` must match the one pinned in `Cargo.toml`.

The module exports `compile`, `run`, `format`, `stdlib` and `version`. Each
returns JSON (see `crates/vigil-driver/src/playground.rs`).

The browser has no clang and no kernel BTF, so the playground shows:

- diagnostics, maps and LLVM IR for every program;
- the eBPF code of the programs in Vigil Core, which it can also run on a
  packet.

It reports `from kernel import ...` as an error.

## Releases

`.github/workflows/ci.yml` runs the tests on every push and pull request, with
`vigil-lang/std` checked out in `../std` at the ref named in `std.ref`.

On a tag `v*`, `release.yml` builds `vigil` for Linux (musl) and macOS, on
x86_64 and aarch64. It publishes each build as `vigil-<target>.tar.gz` with its
`.sha256`, which the install script downloads.

For a release:

1. Tag `vigil-lang/std` and name that tag in `std.ref`.
2. Tag the compiler with the `version` of `Cargo.toml` (`v0.1.0`).

If the repository has a `PAGES_TOKEN` secret, a release also rebuilds the
website with the new compiler.
