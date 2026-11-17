//! this module defines the operations of the web playground on one source
//! file, with JSON results: [`compile`], [`run`], [`format`](format()) and
//! [`stdlib`]. All but [`stdlib`] return an object with a boolean `ok`, and
//! an `error` message when it is false. They live here, not in `vigil-wasm`,
//! so `cargo test` covers them.
//!
//! There is no clang in the browser, so we show LLVM IR for every program
//! and eBPF code only for XDP and TC programs in Vigil Core, which can also
//! run on a packet with [`crate::interp`].

use serde_json::{Value, json};
use vigil_diagnostics::{Diagnostics, SourceMap};
use vigil_target::{Arch, KernelVersion, Profile};

use vigil_sema::program::ProgramKind;

use crate::{Module, interp};

/// the file name of the playground in diagnostics
pub const FILE_NAME: &str = "playground.vg";

/// the settings of a compilation
#[derive(Clone, Debug)]
pub struct Settings {
    /// the target kernel, as `6.1` or `linux-6.1`
    pub kernel: String,
    /// the architecture, `x86_64` or `aarch64`
    pub arch: String,
    /// whether to run the optimizations of the code generator
    pub optimize: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            kernel: "6.1".into(),
            arch: "x86_64".into(),
            optimize: true,
        }
    }
}

impl Settings {
    /// the kernel profile of the settings, or a message saying which one is
    /// wrong
    fn profile(&self) -> Result<Profile, String> {
        let kernel: KernelVersion = self
            .kernel
            .parse()
            .map_err(|_| format!("`{}` is not a kernel version", self.kernel))?;
        let arch = Arch::parse(&self.arch)
            .ok_or_else(|| format!("unknown architecture `{}`", self.arch))?;
        Profile::new(vec![kernel], true)
            .map(|p| p.with_arch(arch))
            .ok_or_else(|| format!("Vigil supports {} or newer", vigil_target::MIN_SUPPORTED))
    }
}

/// an analyzed source, the start of every operation
struct Session {
    sources: SourceMap,
    diags: Diagnostics,
    file: vigil_diagnostics::FileId,
    /// `None` when the source has a syntax error
    analysis: Option<vigil_sema::Analysis>,
    profile: Profile,
}

/// creating the [`Session`] of a source, as the module `playground`, with
/// the built-in standard library and no kernel BTF. Fails if the settings
/// are wrong.
fn session(source: &str, settings: &Settings) -> Result<Session, String> {
    let profile = settings.profile()?;
    let mut sources = SourceMap::new();
    let mut diags = Diagnostics::new();
    let file = sources.add(FILE_NAME, source);
    let modules: Vec<Module> = vec![(file, vec!["playground".into()])];
    let std = crate::embedded_std(&mut sources);
    let analysis = crate::analyze(&modules, &std, &profile, None, &mut sources, &mut diags);
    Ok(Session {
        sources,
        diags,
        file,
        analysis,
        profile,
    })
}

/// the line, from 1, where a span starts
fn line_of(s: &Session, span: vigil_diagnostics::Span) -> usize {
    crate::line_col(s.sources.get(span.file).text(), span.start as usize).0
}

/// the result of an operation that failed as a whole
fn error(message: impl Into<String>) -> Value {
    json!({ "ok": false, "error": message.into() })
}

/// compiling `source`: diagnostics, programs, maps, LLVM IR and the
/// certified backend result of each XDP and TC program
///
/// Programs, maps and code are there only when the analysis has no errors.
/// A program the certified backend rejects is not an error of the source:
/// its entry gets `ok` false and the rendered diagnostic.
pub fn compile(source: &str, settings: &Settings) -> Value {
    let mut s = match session(source, settings) {
        Ok(s) => s,
        Err(e) => return error(e),
    };
    let mut programs = Vec::new();
    let mut maps = Vec::new();
    let mut ir = Value::Null;
    let mut certified = Vec::new();
    if let Some(analysis) = s.analysis.as_ref().filter(|_| !s.diags.has_errors()) {
        for p in crate::programs(analysis) {
            programs.push(json!({
                "name": p.name,
                "kind": p.info.kind.name(),
                "section": p.info.section(),
                "line": line_of(&s, p.span),
            }));
        }
        let options = crate::codegen_options(&s.profile, settings.optimize);
        match vigil_codegen_llvm::generate_with(analysis, &s.sources, options) {
            Ok(output) => {
                ir = output.ir.into();
                for m in &output.maps {
                    maps.push(json!({
                        "name": m.name,
                        "kind": m.kind.name(),
                        "keySize": m.key_size,
                        "valueSize": m.value_size,
                        "maxEntries": m.max_entries,
                    }));
                }
            }
            Err(errors) => s.diags.extend(errors),
        }
        // we skip non packet programs, because they cannot be in Vigil Core.
        // `certify` and `programs` use the same order, so the zip is safe
        let kinds = crate::programs(analysis).into_iter().map(|p| p.info.kind);
        for (result, kind) in crate::certify(analysis).into_iter().zip(kinds) {
            if !matches!(kind, ProgramKind::Xdp | ProgramKind::Tc) {
                continue;
            }
            certified.push(match result {
                Ok(c) => {
                    let stats = crate::disasm::stats(&c.code);
                    json!({
                        "name": c.desc.name,
                        "ok": true,
                        "core": vigil_core::print::print(&c.core),
                        "asm": crate::disasm::listing(&c.code),
                        "insns": stats.insns,
                        "stack": stats.stack,
                        "bytes": c.code.len(),
                    })
                }
                Err(diag) => {
                    let mut one = Diagnostics::new();
                    one.push(diag);
                    json!({
                        "ok": false,
                        "error": crate::render(&s.sources, &one, false),
                    })
                }
            });
        }
    }
    json!({
        "ok": !s.diags.has_errors(),
        "profile": s.profile.to_string(),
        "diagnostics": crate::diagnostics_json(&s.sources, &s.diags),
        "rendered": crate::render(&s.sources, &s.diags, false),
        "errors": s.diags.error_count(),
        "warnings": s.diags.warning_count(),
        "programs": programs,
        "maps": maps,
        "ir": ir,
        "certified": certified,
        "file": s.sources.get(s.file).name(),
    })
}

/// running the program `name` of `source`, compiled by the certified
/// backend, on a packet in hex. It gives the returned value, the verdict,
/// the step count and the packet after the run. We run the code for the
/// model of the formalization, which is what [`interp`] executes.
pub fn run(source: &str, settings: &Settings, name: &str, packet_hex: &str) -> Value {
    let s = match session(source, settings) {
        Ok(s) => s,
        Err(e) => return error(e),
    };
    let Some(analysis) = s.analysis.as_ref().filter(|_| !s.diags.has_errors()) else {
        return error(format!(
            "the program does not compile:\n{}",
            crate::render(&s.sources, &s.diags, false)
        ));
    };
    let mut packet = match interp::parse_hex(packet_hex) {
        Ok(p) => p,
        Err(e) => return error(format!("invalid packet: {e}; {}", interp::HEX_FORMAT)),
    };
    // `certify` uses the order of `programs`, so we report the diagnostic
    // of `name`, not of the first program outside Vigil Core
    let found = crate::programs(analysis)
        .into_iter()
        .zip(crate::certify(analysis))
        .find(|(p, _)| p.name == name)
        .map(|(_, r)| r);
    let program = match found {
        Some(Ok(c)) => c,
        Some(Err(d)) => {
            let mut one = Diagnostics::new();
            one.push(d);
            return error(crate::render(&s.sources, &one, false));
        }
        _ => return error(format!("there is no program `{name}`")),
    };
    let kind = program.desc.info.kind;
    match interp::run(&program.model_code, &mut packet) {
        Ok(outcome) => {
            // the model code always returns XDP verdicts, so for TC we
            // give the value and name of the TC one
            let (value, verdict) = interp::verdict(kind, outcome.value);
            json!({
                "ok": true,
                "value": value,
                "verdict": verdict,
                "steps": outcome.steps,
                "packet": packet.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "),
            })
        }
        Err(e) => error(format!("the program faulted in the eBPF interpreter: {e}")),
    }
}

/// formatting `source`, unless it has syntax errors. If the output means
/// another program, it is a formatter bug and we report it as one.
pub fn format(source: &str) -> Value {
    match vigil_syntax::fmt::format(source) {
        Ok(text) => json!({ "ok": true, "text": text }),
        Err(vigil_syntax::fmt::Error::Syntax) => error("the program has syntax errors"),
        Err(vigil_syntax::fmt::Error::Changed) => error(format!(
            "formatting would change the meaning of the program, so it is left as is; \
             this is a bug in the formatter, please report it at {} with the program",
            vigil_diagnostics::ISSUES
        )),
    }
}

/// the standard library modules as `[{path, source}]`, for the playground
/// docs
pub fn stdlib() -> Value {
    vigil_sema::stdlib::MODULES
        .iter()
        .map(|(path, _, text)| json!({ "path": path, "source": text }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILTER: &str = "\
from std.net import XdpContext, XdpAction, Byte

@xdp
def first_byte(ctx: XdpContext) -> XdpAction:
    b = ctx.parse[Byte](0)?.value
    if b == 0xff:
        return XdpAction.DROP
    return XdpAction.PASS
";

    #[test]
    fn compiles_and_certifies_a_filter() {
        let out = compile(FILTER, &Settings::default());
        assert_eq!(out["ok"], true, "{}", out["rendered"]);
        assert_eq!(out["programs"][0]["name"], "first_byte");
        assert!(out["ir"].as_str().unwrap().contains("define"));
        assert_eq!(out["certified"][0]["ok"], true, "{}", out["certified"][0]);
    }

    /// a program outside Vigil Core defined first must not block running a
    /// Core program defined after it
    #[test]
    fn runs_the_program_asked_for() {
        let source = "\
from std.bpf import KprobeContext
from std.net import XdpContext, XdpAction, Byte

@kprobe(\"vfs_read\")
def early(ctx: KprobeContext) -> i32:
    return 0

@xdp
def first_byte(ctx: XdpContext) -> XdpAction:
    if ctx.parse[Byte](0)?.value == 0xff:
        return XdpAction.DROP
    return XdpAction.PASS
";
        let out = run(source, &Settings::default(), "first_byte", "ff");
        assert_eq!(out["verdict"], "XDP_DROP", "{out}");
    }

    #[test]
    fn runs_a_filter_on_packets() {
        let s = Settings::default();
        let drop = run(FILTER, &s, "first_byte", "ff 00");
        assert_eq!(drop["verdict"], "XDP_DROP", "{drop}");
        let pass = run(FILTER, &s, "first_byte", "01");
        assert_eq!(pass["verdict"], "XDP_PASS", "{pass}");
        // too short for the byte at offset 0, so `?` returns and the XDP
        // program passes the packet
        let empty = run(FILTER, &s, "first_byte", "");
        assert_eq!(empty["verdict"], "XDP_PASS", "{empty}");
    }

    // a bad packet names the bad character and gives the format
    #[test]
    fn reports_bad_packets() {
        let out = run(FILTER, &Settings::default(), "first_byte", "ff zz");
        assert_eq!(out["ok"], false, "{out}");
        let error = out["error"].as_str().unwrap();
        assert!(
            error.starts_with("invalid packet: `z` at position 4 is not a hexadecimal digit; "),
            "{error}"
        );
    }

    // Core has no array values, so `b = a` is exported as a copy of each
    // element, and `b[0] = 9` leaves `a` alone
    #[test]
    fn copies_arrays() {
        let source = "\
from std.net import XdpContext, XdpAction, Byte

@xdp
def copies(ctx: XdpContext) -> XdpAction:
    n = ctx.parse[Byte](0)?.value
    a = [n, n, n]
    b = a
    b[0] = 9
    if a[0] == 5 and b[0] == 9 and b[2] == 5:
        return XdpAction.DROP
    return XdpAction.PASS
";
        let out = run(source, &Settings::default(), "copies", "05");
        assert_eq!(out["verdict"], "XDP_DROP", "{out}");
    }

    // an abort (here an index out of bounds) returns 0 for every kind, so
    // a TC program gives `TC_ACT_OK`; a drop gives `TC_ACT_SHOT`, 2
    #[test]
    fn names_tc_verdicts() {
        let source = "\
from std.net import TcContext, TcAction, Byte

@tc
def pick(ctx: TcContext) -> TcAction:
    a = [u8(1), 2]
    if a[ctx.parse[Byte](0)?.value] == 2:
        return TcAction.SHOT
    return TcAction.SHOT
";
        let s = Settings::default();
        let abort = run(source, &s, "pick", "05");
        assert_eq!(abort["verdict"], "TC_ACT_OK", "{abort}");
        assert_eq!(abort["value"], 0, "{abort}");
        let drop = run(source, &s, "pick", "01");
        assert_eq!(drop["verdict"], "TC_ACT_SHOT", "{drop}");
        assert_eq!(drop["value"], 2, "{drop}");
    }

    // the error points at `range(0, 4)`, not at the loop and what follows
    #[test]
    fn range_errors_point_at_the_range() {
        let source = "\
from std.net import XdpContext, XdpAction

@xdp
def count(ctx: XdpContext) -> XdpAction:
    n: u8 = 0
    for i in range(0, 4):
        n += 1
    return XdpAction.PASS
";
        let out = compile(source, &Settings::default());
        let error = out["certified"][0]["error"].as_str().unwrap();
        assert!(
            error.contains("`range(start, end)` is not in Vigil Core"),
            "{error}"
        );
        assert!(!error.contains("return XdpAction.PASS"), "{error}");
    }

    #[test]
    fn reports_type_errors_with_positions() {
        let out = compile("def f() -> u32:\n    return true\n", &Settings::default());
        assert_eq!(out["ok"], false);
        let d = &out["diagnostics"][0];
        assert_eq!(d["severity"], "error");
        assert_eq!(d["labels"][0]["line"], 2, "{d}");
    }

    // a program with a map is outside Vigil Core, but we still get the
    // LLVM IR and the maps
    #[test]
    fn programs_with_maps_still_get_ir() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/xdp_firewall.vg"
        ))
        .unwrap();
        let out = compile(&source, &Settings::default());
        assert_eq!(out["ok"], true, "{}", out["rendered"]);
        assert!(!out["maps"].as_array().unwrap().is_empty());
        assert_eq!(out["certified"][0]["ok"], false);
    }
}
