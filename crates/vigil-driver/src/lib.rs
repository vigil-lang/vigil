//! this crate defines the compiler pipeline without a file system or
//! external programs, shared by the `vigil` command and `vigil-wasm` (the
//! browser has neither), so the two cannot drift apart.
//!
//! - [`analyze`]: parsing and analysis, with the standard library from
//!   [`embedded_std`] or one loaded by the caller
//! - [`codegen_options`]: the LLVM codegen options of a kernel profile
//! - [`certify`] and [`certified_object`]: the certified backend, through
//!   Vigil Core, and its ELF object
//! - [`render`] and [`diagnostics_json`]: diagnostics as text or JSON

pub mod interp;
pub mod playground;

use std::sync::Arc;

use vigil_codegen_llvm::ProgramDesc;
use vigil_diagnostics::{Diagnostic, Diagnostics, FileId, Renderer, SourceMap, Span, codes};
use vigil_sema::program::ProgramKind;
use vigil_sema::{Analysis, ModuleInput, stdlib};
use vigil_target::{Feature, Profile};

pub use vigil_codegen_llvm::{cost, disasm};

/// a module being compiled: its file and its module path
/// (`src/net/parse.vg` is `["net", "parse"]`)
pub type Module = (FileId, Vec<String>);

/// adding the built-in standard library to `sources`, with module paths
/// starting with `std`, ready for [`analyze`]. Its files are named
/// `<std>/net.vg` and so on, because they are not on disk.
pub fn embedded_std(sources: &mut SourceMap) -> Vec<Module> {
    stdlib::MODULES
        .iter()
        .map(|(path, name, text)| {
            let file = sources.add(*name, *text);
            (file, path.split('.').map(String::from).collect())
        })
        .collect()
}

/// the analysis of `modules` together with the standard library `std`, or
/// `None` if there is a syntax error
///
/// `std` is usually [`embedded_std`], but a caller may load another one,
/// with paths starting with `std`. Only these modules have the privileges
/// of the standard library (`@helper`, `@context`), and a module of
/// `modules` whose path starts with `std` is an error. A module the
/// compiler needs and `std` lacks is reported as an import error. With
/// type errors we still return the analysis. All errors go in `diags`,
/// sorted by location.

pub fn analyze(
    modules: &[Module],
    std: &[Module],
    profile: &Profile,
    btf: Option<Arc<vigil_btf::Btf>>,
    sources: &mut SourceMap,
    diags: &mut Diagnostics,
) -> Option<Analysis> {
    // only the modules of `std` have the privileges of the standard library,
    // whatever the paths of the others
    let all = modules
        .iter()
        .map(|m| (m, false))
        .chain(std.iter().map(|m| (m, true)));
    let mut inputs = Vec::new();
    for ((file, module), is_std) in all {
        let (file, module) = (*file, module.clone());
        let parse = vigil_syntax::parse(sources.get(file).text());
        for error in parse.errors() {
            diags.push(error.to_diagnostic(file));
        }
        inputs.push(ModuleInput {
            file,
            path: module,
            tree: parse.tree(),
            std: is_std,
        });
    }
    // we stop on syntax errors, because the analysis would mostly report
    // their consequences and hide the real error
    if diags.has_errors() {
        diags.sort_by_location();
        return None;
    }
    let options = vigil_sema::Options {
        kernel: profile.baseline(),
        btf,
        arch: profile.arch(),
    };
    let (analysis, sema_diags) = vigil_sema::analyze_with(&inputs, options);
    diags.extend(sema_diags);
    diags.sort_by_location();
    Some(analysis)
}

/// the LLVM codegen options for the kernels of `profile`
pub fn codegen_options(profile: &Profile, optimize: bool) -> vigil_codegen_llvm::Options {
    vigil_codegen_llvm::Options {
        probe_read_kernel: profile.supports(Feature::ProbeReadKernel),
        bpf_loop: profile.supports(Feature::BpfLoop),
        cpu_v4: profile.supports(Feature::CpuV4),
        arch: profile.arch(),
        optimize,
        ..Default::default()
    }
}

/// a program compiled by the certified backend, through Vigil Core
#[derive(Debug)]
pub struct Certified {
    /// name, kind and location of the program
    pub desc: ProgramDesc,
    /// its export to Vigil Core
    pub core: vigil_core::Program,
    /// the code for Linux, with the XDP or TC context
    pub code: Vec<u8>,
    /// the code for the eBPF model of the Lean formalization, where the
    /// packet bounds are 64-bit words at offsets 0 and 8 of the context.
    /// This is what [`interp`] runs.
    pub model_code: Vec<u8>,
}

/// what Vigil Core has, as a note for the certified backend diagnostics
pub const CORE_SUBSET: &str = "Vigil Core has XDP and TC programs over integers, `bool` and local \
     arrays of integers: `if`, `for i in range(N)`, the bytes of the packet \
     (`ctx.parse[Byte](offset)?.value`) and its length, and verdicts; no \
     maps, helpers, structs, enums, `Option`, function calls or tail calls";

/// the programs of `analysis` in definition order, without tests
pub fn programs(analysis: &Analysis) -> Vec<ProgramDesc> {
    let mut programs: Vec<_> = analysis
        .programs
        .iter()
        .filter(|(_, info)| info.kind != ProgramKind::Test)
        .collect();
    programs.sort_by_key(|(f, _)| f.into_raw().into_u32());
    programs
        .into_iter()
        .map(|(f, info)| {
            let data = &analysis.package.fns[f];
            ProgramDesc {
                name: data.name.text.clone(),
                info: info.clone(),
                span: data.name.span,
            }
        })
        .collect()
}

/// compiling each program of `analysis` with the certified backend,
/// `vigil-codegen-bpf`, without LLVM. A program outside Vigil Core gives a
/// diagnostic.
///
/// If the package defines a map, every program fails with the same error at
/// the first map, even if it does not use it. Each program is compiled
/// twice, for its kernel target and for the model of the formalization.
pub fn certify(analysis: &Analysis) -> Vec<Result<Certified, Diagnostic>> {
    let mut functions: Vec<_> = analysis
        .programs
        .iter()
        .filter(|(_, info)| info.kind != ProgramKind::Test)
        .collect();
    functions.sort_by_key(|(f, _)| f.into_raw().into_u32());
    let first_map = analysis
        .package
        .maps
        .iter()
        .next()
        .map(|(_, m)| m.name.span);
    functions
        .into_iter()
        .map(|(function, info)| {
            let data = &analysis.package.fns[function];
            let span = data.name.span;
            let name = &data.name.text;
            if let Some(map) = first_map {
                return Err(Diagnostic::error("Vigil Core has no maps")
                    .with_code(codes::CODEGEN_LIMIT)
                    .with_primary(map, "a map keeps every program out of Vigil Core")
                    .with_note(CORE_SUBSET));
            }
            let core = vigil_core::export::export(analysis, function).map_err(|e| {
                Diagnostic::error(format!("`{name}` is not in Vigil Core"))
                    .with_code(codes::CODEGEN_LIMIT)
                    .with_primary(e.span, e.message)
                    .with_note(CORE_SUBSET)
            })?;
            let target = match info.kind {
                ProgramKind::Tc => vigil_codegen_bpf::Target::Tc,
                _ => vigil_codegen_bpf::Target::Xdp,
            };
            let outside = |e: String| {
                Diagnostic::error(format!("`{name}` is outside the certified backend"))
                    .with_code(codes::CODEGEN_LIMIT)
                    .with_primary(span, e)
            };
            let model_code =
                vigil_codegen_bpf::compile_for(&core, vigil_codegen_bpf::Target::Model)
                    .map_err(outside)?;
            let code = vigil_codegen_bpf::compile_for(&core, target).map_err(outside)?;
            Ok(Certified {
                desc: ProgramDesc {
                    name: name.clone(),
                    info: info.clone(),
                    span,
                },
                core,
                code,
                model_code,
            })
        })
        .collect()
}

/// the ELF object of certified programs, each in the section of its kind
pub fn certified_object(programs: &[Certified]) -> Vec<u8> {
    let sections: Vec<String> = programs.iter().map(|p| p.desc.info.section()).collect();
    let parts: Vec<(&str, &str, &[u8])> = programs
        .iter()
        .zip(&sections)
        .map(|(p, section)| (section.as_str(), p.desc.name.as_str(), p.code.as_slice()))
        .collect();
    vigil_codegen_bpf::elf::objects(&parts)
}

/// the diagnostics as text, as the command line shows them, with or
/// without colours
pub fn render(sources: &SourceMap, diags: &Diagnostics, color: bool) -> String {
    let mut out = Vec::new();
    // writing to a vector does not fail
    let _ = Renderer::new(sources, color).render_all(diags, &mut out);
    String::from_utf8_lossy(&out).into_owned()
}

/// line and column of a byte offset, both from 1, the column in characters.
/// An offset past the end or inside a character is moved back, so a stale
/// span never panics.
pub fn line_col(text: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(text.len());
    let before = &text[..floor_char_boundary(text, offset)];
    let line = before.matches('\n').count() + 1;
    let start = before.rfind('\n').map_or(0, |i| i + 1);
    (line, before[start..].chars().count() + 1)
}

/// the last char boundary at or before `offset`, which must be at most the
/// length of `text`
fn floor_char_boundary(text: &str, mut offset: usize) -> usize {
    while !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

/// a span as JSON: the file, and start and end as lines and columns
pub fn span_json(sources: &SourceMap, span: Span) -> serde_json::Value {
    let file = sources.get(span.file);
    let (line, col) = line_col(file.text(), span.start as usize);
    let (end_line, end_col) = line_col(file.text(), span.end as usize);
    serde_json::json!({
        "file": file.name(),
        "line": line,
        "col": col,
        "endLine": end_line,
        "endCol": end_col,
    })
}

/// the diagnostics as JSON, for editors and the playground, with labels as
/// in [`span_json`]
pub fn diagnostics_json(sources: &SourceMap, diags: &Diagnostics) -> serde_json::Value {
    let list: Vec<_> = diags
        .iter()
        .map(|d| {
            let severity = match d.severity {
                vigil_diagnostics::Severity::Error => "error",
                vigil_diagnostics::Severity::Warning => "warning",
                vigil_diagnostics::Severity::Note => "note",
            };
            let labels: Vec<_> = d
                .labels
                .iter()
                .map(|l| {
                    let mut span = span_json(sources, l.span);
                    span["message"] = l.message.clone().into();
                    span["primary"] = l.primary.into();
                    span
                })
                .collect();
            serde_json::json!({
                "severity": severity,
                "code": d.code,
                "message": d.message,
                "labels": labels,
                "notes": d.notes,
                "help": d.help,
            })
        })
        .collect();
    serde_json::Value::Array(list)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a module the caller adds to the standard library is imported like a
    /// built-in one, as a `std` dependency needs
    #[test]
    fn analyzes_with_a_library_of_the_caller() {
        let mut sources = SourceMap::new();
        let mut diags = Diagnostics::new();
        let main = sources.add(
            "main.vg",
            "from std.extra import double\n\ndef f(x: u32) -> u32:\n    return double(x)\n",
        );
        let mut std = embedded_std(&mut sources);
        let extra = sources.add("extra.vg", "def double(x: u32) -> u32:\n    return x * 2\n");
        std.push((extra, vec!["std".into(), "extra".into()]));
        let modules = vec![(main, vec!["main".into()])];
        let analysis = analyze(
            &modules,
            &std,
            &Profile::default_profile(),
            None,
            &mut sources,
            &mut diags,
        );
        assert!(analysis.is_some());
        assert!(!diags.has_errors(), "{}", render(&sources, &diags, false));

        // without it, the import is an error
        let mut sources = SourceMap::new();
        let mut diags = Diagnostics::new();
        let main = sources.add("main.vg", "from std.extra import double\n");
        let std = embedded_std(&mut sources);
        let modules = vec![(main, vec!["main".into()])];
        analyze(
            &modules,
            &std,
            &Profile::default_profile(),
            None,
            &mut sources,
            &mut diags,
        );
        assert!(diags.has_errors());
    }

    /// the privileges of the standard library (`@helper` here) come from
    /// the `std` argument, not from a path starting with `std`
    #[test]
    fn only_the_std_argument_is_the_standard_library() {
        let helper = "@helper(14)\ndef my_pid() -> u64: ...\n";
        let check = |as_std: bool| {
            let mut sources = SourceMap::new();
            let mut diags = Diagnostics::new();
            let mut std = embedded_std(&mut sources);
            let mine = (
                sources.add("mine.vg", helper),
                vec!["std".into(), "mine".into()],
            );
            let mut modules = Vec::new();
            if as_std {
                std.push(mine);
            } else {
                modules.push(mine);
            }
            analyze(
                &modules,
                &std,
                &Profile::default_profile(),
                None,
                &mut sources,
                &mut diags,
            );
            render(&sources, &diags, false)
        };
        assert_eq!(check(true), "");
        let errors = check(false);
        assert!(
            errors.contains("has the name of the standard library"),
            "{errors}"
        );
        assert!(errors.contains("`@helper` is reserved"), "{errors}");
    }
}
