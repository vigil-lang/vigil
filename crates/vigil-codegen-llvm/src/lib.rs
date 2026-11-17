//! this module is the code generator: from a checked package it writes LLVM
//! IR as text and compiles it with `clang -target bpfel` ([`clang`]), so we
//! do not depend on the LLVM libraries.
//!
//! Entry points: [`build`] gives the object ([`Built`]), [`generate`] and
//! [`generate_with`] stop at the IR ([`Output`]). They expect an
//! [`Analysis`] without errors.
//!
//! Each program goes in an ELF section named after its attach point
//! (`kprobe/do_sys_openat2`), maps go in `.maps` with BTF, and the debug
//! info gives the BTF line info that maps the verifier log back to Vigil
//! lines. When a program needs more than [`STACK_LIMIT`] bytes of stack,
//! [`build`] generates the package again with the large locals of every
//! program in scratch, a per-CPU array.

pub mod clang;
pub mod cost;
pub mod desc;
pub mod disasm;
mod function;
mod layout;
pub mod link;
mod meta;
mod range;

use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;

use vigil_diagnostics::{Diagnostic, Diagnostics, SourceMap, Span, codes};
use vigil_sema::Analysis;
use vigil_sema::hir::FnId;
use vigil_sema::infer::Res;
use vigil_sema::maps::MapKind;
use vigil_sema::program::{ProgramInfo, ProgramKind};
use vigil_sema::ty::TyKind;

use crate::clang::{Clang, ClangError};
use crate::desc::MapDesc;
use crate::function::{FnGen, ScratchUse, Stage, carry_map, scratch_map, stages_map};
use crate::layout::{DATA_LAYOUT, Layout, TRIPLE};
use crate::meta::Meta;

/// the LLVM module of a package, and its programs and maps
#[derive(Debug)]
pub struct Output {
    pub ir: String,
    pub programs: Vec<ProgramDesc>,
    pub maps: Vec<MapDesc>,
}

/// a program of the object: what a loader needs to find and attach it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramDesc {
    /// the function name, also the ELF symbol of the program
    pub name: String,
    pub info: ProgramInfo,
    pub span: Span,
}

/// the helper called by `run(program, packet)` in a `@test` function:
/// `(program, packet, damaged)`. The program is the raw index of its
/// function, the packet the address of a `std.testing.Packet` (a `Damaged`
/// one when `damaged` is 1). It returns the verdict of the program.
///
/// This helper and the three below are not kernel helpers. Only a userspace
/// simulator that runs `@test` functions ([`Options::tests`]) answers them,
/// and this repository does not have one.
pub const HELPER_RUN: u32 = 0x7F00_0001;
/// the helper called by `set_time(ns)`: it sets what `bpf_ktime_get_ns`
/// returns
pub const HELPER_SET_TIME: u32 = 0x7F00_0002;
/// the helper called by `count()` of a ring buffer: `(map)`, giving the
/// number of events emitted to it
pub const HELPER_EVENT_COUNT: u32 = 0x7F00_0003;
/// the helper called by `last()` of a ring buffer: `(map, buffer, size)`.
/// It copies the last event into the buffer and returns 1, or 0 when there
/// is none.
pub const HELPER_LAST_EVENT: u32 = 0x7F00_0004;

/// the description of the values of type `ty` as compiled code lays them
/// out, and their size in bytes. Options that only change code are off.
pub fn describe(analysis: &Analysis, ty: vigil_sema::ty::TyId) -> (desc::TypeDesc, u64) {
    let layout = Layout {
        analysis,
        test_context: false,
        probe_read_kernel: false,
        scratch: None,
        bpf_loop: false,
        arch: Default::default(),
        optimize: false,
        packet_types: std::collections::HashSet::new(),
    };
    (layout.describe(ty), layout.size_align(ty).0)
}

/// options of the code generator. The default needs Linux 5.17
/// (`bpf_probe_read_kernel` and `bpf_loop`), does not use CPU version 4,
/// and optimizes.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// packet bounds read as 64-bit words at offsets 0 and 8 of the context,
    /// as userspace eBPF interpreters do. Only for tests: the kernel rejects
    /// such programs.
    pub test_context: bool,
    /// reading kernel memory with `bpf_probe_read_kernel` (Linux 5.5)
    /// instead of `bpf_probe_read`
    pub probe_read_kernel: bool,
    /// running long loops with `bpf_loop` (Linux 5.17)
    pub bpf_loop: bool,
    /// the size from which a local goes in scratch instead of the stack;
    /// `None` keeps every local on the stack. [`build`] sets it when needed.
    pub scratch: Option<u64>,
    /// the architecture of the target kernels, which says where
    /// `struct pt_regs` holds the arguments of probed functions
    pub arch: vigil_target::Arch,
    /// building the `@test` functions too, as XDP programs for a simulator
    /// (see [`HELPER_RUN`])
    pub tests: bool,
    /// using CPU version 4 (Linux 6.6, clang 18) when clang has it; version
    /// 3 otherwise
    pub cpu_v4: bool,
    /// our own optimizations before LLVM: unrolling loops with a constant
    /// variable, and reading constant tables at constant indices. Off
    /// (`-O0`) gives the reference code to compare with.
    pub optimize: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            test_context: false,
            probe_read_kernel: true,
            bpf_loop: true,
            scratch: None,
            arch: vigil_target::Arch::host(),
            tests: false,
            cpu_v4: false,
            optimize: true,
        }
    }
}

/// the size in bytes from which a local moves to scratch when the stack is
/// too small
pub const SCRATCH_MIN: u64 = 32;
/// the stack of an eBPF program, in bytes
pub const STACK_LIMIT: usize = 512;
/// the largest value of a per-CPU array, so the most scratch a program can
/// have
const MAX_SCRATCH: u64 = 32 * 1024;

/// an eBPF object built from a package
#[derive(Debug)]
pub struct Built {
    /// the IR of the object, and its programs and maps
    pub output: Output,
    /// the ELF object
    pub object: Vec<u8>,
}

/// why a package could not be built
#[derive(Debug)]
pub enum BuildError {
    /// errors while generating the IR: no program, two programs or maps
    /// with the same name, a program that needs too much scratch, or code
    /// the generator cannot compile. Also a program that clang could not fit
    /// in the stack even with its large locals in scratch.
    Diagnostics(Diagnostics),
    /// clang failed: a bug of Vigil, or a limit of eBPF
    Clang(String),
    /// clang ran longer than this many seconds (see [`clang::timeout`])
    ClangTimeout(u64),
}

/// building the object of a package
///
/// When a program needs more than [`STACK_LIMIT`] bytes of stack (or clang
/// says it does), we generate the package again with its large locals in
/// scratch. If the options already ask for scratch, we compile only once.
/// When the locals of a reachable function add up to more than the largest
/// scratch, we go straight to scratch, where generation rejects them,
/// because clang can take minutes on such locals.
pub fn build(
    analysis: &Analysis,
    sources: &SourceMap,
    options: Options,
    clang: &Clang,
) -> Result<Built, BuildError> {
    let options = match options.scratch {
        None if locals_exceed(analysis, options, MAX_SCRATCH) => Options {
            scratch: Some(SCRATCH_MIN),
            ..options
        },
        _ => options,
    };
    let output = generate_with(analysis, sources, options).map_err(BuildError::Diagnostics)?;
    let cpu = cpu(options, clang);
    let first = clang.compile(&output.ir, cpu);
    let overflows = match &first {
        Ok(object) => output
            .programs
            .iter()
            .any(|p| stack_depth(object, &p.info.section(), &p.name) > STACK_LIMIT),
        Err(ClangError::Failed(message)) => message.contains("stack limit"),
        Err(ClangError::TimedOut(_)) => false,
    };
    let (output, result) = if !overflows || options.scratch.is_some() {
        (output, first)
    } else {
        let options = Options {
            scratch: Some(SCRATCH_MIN),
            ..options
        };
        let output = generate_with(analysis, sources, options).map_err(BuildError::Diagnostics)?;
        let result = clang.compile(&output.ir, cpu);
        (output, result)
    };
    match result {
        Ok(object) => Ok(Built { output, object }),
        Err(ClangError::TimedOut(seconds)) => Err(BuildError::ClangTimeout(seconds)),
        Err(ClangError::Failed(message)) if message.contains("stack limit") => {
            Err(BuildError::Diagnostics(stack_exceeded(&output, &message)))
        }
        Err(ClangError::Failed(message)) => Err(BuildError::Clang(message)),
    }
}

/// the error for a program that clang could not fit in the stack, even
/// with its large locals in scratch. Clang names the LLVM function: a
/// program, a stage, or the callback of a loop, which we cannot map back,
/// so then we name no program.
fn stack_exceeded(output: &Output, message: &str) -> Diagnostics {
    let first = message
        .lines()
        .find(|l| l.contains("stack limit"))
        .unwrap_or(message)
        .trim();
    let function = first
        .split(" in function ")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next());
    let program = output
        .programs
        .iter()
        .find(|p| function == Some(p.name.as_str()));
    let error = match program {
        Some(p) => Diagnostic::error(format!("`{}` uses too much stack", p.name)).with_primary(
            p.span,
            format!("clang could not fit its values in {STACK_LIMIT} bytes"),
        ),
        None => Diagnostic::error("a program uses too much stack"),
    };
    let mut diags = Diagnostics::new();
    diags.push(
        error
            .with_code(codes::CODEGEN_LIMIT)
            .with_note(format!("clang: {first}"))
            .with_note(format!(
                "the eBPF stack has {STACK_LIMIT} bytes, and locals of {SCRATCH_MIN} bytes or more \
                 already go in a scratch map; the rest usually comes from an `@unroll` loop, \
                 whose copies of the body each keep their values"
            ))
            .with_help("use `@no_unroll` or a shorter loop, or make the locals smaller"),
    );
    diags
}

/// checking if a function reachable from a program has more than `limit`
/// bytes of locals
fn locals_exceed(analysis: &Analysis, options: Options, limit: u64) -> bool {
    let layout = Layout {
        analysis,
        test_context: options.test_context,
        probe_read_kernel: options.probe_read_kernel,
        scratch: options.scratch,
        bpf_loop: options.bpf_loop,
        arch: options.arch,
        optimize: options.optimize,
        packet_types: std::collections::HashSet::new(),
    };
    reachable(analysis, options).into_iter().any(|f| {
        let total: u64 = analysis.bodies[f]
            .locals
            .iter()
            .map(|(_, local)| layout.size_align(local.ty).0)
            .sum();
        total > limit
    })
}

/// the functions reachable from the programs. Helpers are kernel calls, so
/// the walk stops at them, and comptime functions run only in the compiler.
fn reachable(analysis: &Analysis, options: Options) -> Vec<FnId> {
    let mut seen = BTreeSet::new();
    let mut found = Vec::new();
    let mut stack: Vec<FnId> = analysis
        .programs
        .iter()
        .filter(|(_, info)| options.tests || info.kind != ProgramKind::Test)
        .map(|(f, _)| f)
        .collect();
    while let Some(f) = stack.pop() {
        if analysis.helpers.get(f).is_some()
            || analysis.package.fns[f].is_comptime
            || !seen.insert(f.into_raw().into_u32())
        {
            continue;
        }
        found.push(f);
        stack.extend(runtime_callees(analysis, f));
    }
    found
}

/// the CPU version of the instructions: 4 when the kernels and clang have
/// it and we optimize, 3 otherwise. `-O0`, `test_context` and tests keep
/// version 3, because `-O0` is the reference code and version 3 is what the
/// interpreters run.
fn cpu(options: Options, clang: &Clang) -> &'static str {
    let v4 = options.cpu_v4
        && options.optimize
        && !options.test_context
        && !options.tests
        && clang.version >= 18;
    if v4 { "v4" } else { "v3" }
}

/// the stack a program uses, as the verifier counts it: the deepest chain
/// of frames of the program and the functions and `bpf_loop` callbacks it
/// calls, each frame rounded up to 32 bytes, at least 32 (see [`frames`])
pub fn stack_depth(object: &[u8], section: &str, name: &str) -> usize {
    frames(object, section, name).stack
}

/// the chains of frames of a program, as the verifier counts them
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Frames {
    /// the bytes of stack of the deepest chain, each frame rounded up to 32
    /// bytes, at least 32. The kernel allows 512 over the whole chain.
    pub stack: usize,
    /// the most frames in a chain, the program's own included. The kernel
    /// allows 8.
    pub depth: usize,
}

/// the frames of a program in an object: the program, then the `.text`
/// functions it calls (BPF-to-BPF calls, `call` with source 1) or passes as
/// `bpf_loop` callbacks (an `lddw` relocated to `.text`), and so on. We read
/// the final object because clang may have inlined calls.
///
/// It gives zeros when the object cannot be parsed or has no such program.
pub fn frames(object: &[u8], section: &str, name: &str) -> Frames {
    use object::{Object, ObjectSection, ObjectSymbol, RelocationTarget};
    let Ok(file) = object::File::parse(object) else {
        return Frames::default();
    };
    let frame = |code: &[u8]| disasm::stats(code).stack.max(1).next_multiple_of(32);
    // the functions of `.text`: start, end and frame
    let text = file.section_by_name(".text");
    let functions: Vec<(u64, u64, usize)> = match &text {
        Some(text) => {
            let data = text.data().unwrap_or_default();
            file.symbols()
                .filter(|s| s.section_index() == Some(text.index()) && s.size() > 0)
                .map(|s| {
                    let (start, end) = (s.address(), s.address() + s.size());
                    let code = data.get(start as usize..end as usize).unwrap_or_default();
                    (start, end, frame(code))
                })
                .collect()
        }
        None => Vec::new(),
    };
    // the functions the code of a range of a section calls, as indices in
    // `functions`. A relocation names a symbol, maybe the section itself.
    // For a call the target is the symbol plus `imm + 1` instructions, as
    // libbpf reads it, and a call without relocation is relative to itself
    // (only inside `.text`). For an `lddw` the immediate is an offset in
    // bytes from the symbol.
    let callees = |section: &object::Section, start: u64, end: u64| -> Vec<usize> {
        let in_text = text.as_ref().map(|t| t.index()) == Some(section.index());
        let data = section.data().unwrap_or_default();
        let bases: HashMap<u64, i64> = section
            .relocations()
            .filter_map(|(offset, relocation)| {
                let RelocationTarget::Symbol(symbol) = relocation.target() else {
                    return None;
                };
                let symbol = file.symbol_by_index(symbol).ok()?;
                (text.as_ref().map(|t| t.index()) == symbol.section_index())
                    .then_some((offset, symbol.address() as i64))
            })
            .collect();
        let code = data.get(start as usize..end as usize).unwrap_or_default();
        disasm::decode(code)
            .iter()
            .filter_map(|insn| {
                let at = start + insn.slot as u64 * 8;
                let base = bases.get(&at).copied();
                let target = if insn.opcode == 0x85 && insn.src == 1 {
                    let base = base.or(in_text.then_some(at as i64))?;
                    base + (insn.imm + 1) * 8
                } else if insn.is_lddw() {
                    base? + insn.imm
                } else {
                    return None;
                };
                functions
                    .iter()
                    .position(|&(a, e, _)| (a as i64..e as i64).contains(&target))
            })
            .collect()
    };
    let calls: Vec<Vec<usize>> = match &text {
        Some(text) => functions
            .iter()
            .map(|&(a, e, _)| callees(text, a, e))
            .collect(),
        None => Vec::new(),
    };
    // the frames below a function, memoized. Here we first mark the function
    // with its own frame, so that a recursive call (which the verifier
    // rejects anyway) does not loop forever.
    fn below(
        f: usize,
        functions: &[(u64, u64, usize)],
        calls: &[Vec<usize>],
        memo: &mut [Option<Frames>],
    ) -> Frames {
        if let Some(frames) = memo[f] {
            return frames;
        }
        let own = Frames {
            stack: functions[f].2,
            depth: 1,
        };
        memo[f] = Some(own);
        let frames = chain(
            own,
            calls[f]
                .iter()
                .map(|&g| below(g, functions, calls, memo))
                .collect(),
        );
        memo[f] = Some(frames);
        frames
    }
    fn chain(own: Frames, callees: Vec<Frames>) -> Frames {
        Frames {
            stack: own.stack + callees.iter().map(|c| c.stack).max().unwrap_or(0),
            depth: own.depth + callees.iter().map(|c| c.depth).max().unwrap_or(0),
        }
    }
    let Some(program) = file.section_by_name(section) else {
        return Frames::default();
    };
    let Some(symbol) = file
        .symbols()
        .find(|s| s.section_index() == Some(program.index()) && s.name() == Ok(name))
    else {
        return Frames::default();
    };
    let start = symbol.address();
    let end = if symbol.size() > 0 {
        start + symbol.size()
    } else {
        program.size()
    };
    let code = program_code(object, section, name).unwrap_or_default();
    let mut memo = vec![None; functions.len()];
    let own = Frames {
        stack: frame(&code),
        depth: 1,
    };
    let callees = callees(&program, start, end)
        .into_iter()
        .map(|g| below(g, &functions, &calls, &mut memo))
        .collect();
    chain(own, callees)
}

/// the chains of tail calls of the programs split by `stage()`: for each
/// such program, the tail calls from its first stage to its last one. The
/// tail calls through the user's own program arrays are not here, because
/// they are chosen at run time.
pub fn stage_chains(programs: &[ProgramDesc]) -> Vec<(&ProgramDesc, u32)> {
    programs
        .iter()
        .filter_map(|p| {
            let map = stages_map(&p.name);
            programs
                .iter()
                .filter_map(|q| q.info.tail.as_ref().filter(|(m, _)| *m == map))
                .map(|&(_, index)| index)
                .max()
                .map(|tail_calls| (p, tail_calls))
        })
        .collect()
}

/// the instructions of a program, the bytes of its symbol in its section.
/// A symbol without a size runs to the end of the section.
pub fn program_code(object: &[u8], section: &str, name: &str) -> Option<Vec<u8>> {
    use object::{Object, ObjectSection, ObjectSymbol};
    let file = object::File::parse(object).ok()?;
    let section = file.section_by_name(section)?;
    let data = section.data().ok()?;
    let symbol = file
        .symbols()
        .find(|s| s.section_index() == Some(section.index()) && s.name() == Ok(name))?;
    let start = symbol.address() as usize;
    let end = if symbol.size() > 0 {
        start + symbol.size() as usize
    } else {
        data.len()
    };
    data.get(start..end).map(<[u8]>::to_vec)
}

/// the LLVM IR of a package without errors, with the default [`Options`]
pub fn generate(analysis: &Analysis, sources: &SourceMap) -> Result<Output, Diagnostics> {
    generate_with(analysis, sources, Options::default())
}

/// the LLVM IR of a package without errors
///
/// We generate only the functions reachable from a program. A program
/// split by `stage()` becomes one program per stage: the later stages are
/// reached by tail calls through a program array, and the variables that
/// live across stages go in a per-CPU array.
pub fn generate_with(
    analysis: &Analysis,
    sources: &SourceMap,
    options: Options,
) -> Result<Output, Diagnostics> {
    let mut diags = Diagnostics::new();
    let mut programs: Vec<(FnId, &ProgramInfo)> = analysis
        .programs
        .iter()
        .filter(|(_, info)| options.tests || info.kind != ProgramKind::Test)
        .collect();
    programs.sort_by_key(|(f, _)| f.into_raw().into_u32());
    if programs.is_empty() {
        diags.push(
            Diagnostic::error("no eBPF programs to build")
                .with_code(codes::CODEGEN_LIMIT)
                .with_help(
                    "mark a function with the decorator of its program type: `@kprobe`, `@kretprobe`, \
                     `@tracepoint`, `@uprobe`, `@uretprobe`, `@lsm`, `@xdp` or `@tc`",
                ),
        );
        return Err(diags);
    }

    // programs and maps share one symbol namespace in the object, so we
    // report two with the same name here instead of a clash in clang
    let mut names: HashMap<&str, Span> = HashMap::new();
    let program_names = programs.iter().map(|&(f, _)| &analysis.package.fns[f].name);
    let map_names = analysis
        .maps
        .iter()
        .map(|(m, _)| &analysis.package.maps[m].name);
    for name in program_names.chain(map_names) {
        if let Some(first) = names.insert(&name.text, name.span) {
            diags.push(
                Diagnostic::error(format!("two programs or maps are named `{}`", name.text))
                    .with_code(codes::DUPLICATE_DEFINITION)
                    .with_primary(name.span, "second definition")
                    .with_secondary(first, "first definition")
                    .with_note(
                        "programs and maps are symbols of the object file, shared by all modules",
                    ),
            );
        }
    }
    if diags.has_errors() {
        return Err(diags);
    }

    let reachable: BTreeSet<u32> = reachable(analysis, options)
        .into_iter()
        .map(|f| f.into_raw().into_u32())
        .collect();

    let layout = Layout {
        analysis,
        test_context: options.test_context,
        probe_read_kernel: options.probe_read_kernel,
        scratch: options.scratch,
        bpf_loop: options.bpf_loop,
        arch: options.arch,
        optimize: options.optimize,
        packet_types: layout::packet_types(analysis),
    };
    let mut meta = Meta::new(sources, analysis.btf.as_deref());
    let mut functions = String::new();
    let mut scratch: HashMap<FnId, ScratchUse> = HashMap::new();
    // programs split by `stage()`: the stages after the first, and for each
    // program its number of stages and the size it carries between stages
    let mut stage_programs: Vec<(String, FnId, usize, ScratchUse)> = Vec::new();
    let mut carries: Vec<(String, usize, u64)> = Vec::new();
    for (f, _) in analysis.package.fns.iter() {
        if !reachable.contains(&f.into_raw().into_u32()) {
            continue;
        }
        let Some(stages) = split(analysis, &layout, f) else {
            let (text, usage) = FnGen::generate(analysis, &layout, &mut meta, &mut diags, f);
            functions.push_str(&text);
            functions.push('\n');
            scratch.insert(f, usage);
            continue;
        };
        let name = analysis.package.fns[f].name.text.clone();
        let res = &analysis.bodies[f];
        let carried_size =
            struct_size(&layout, stages[0].carried.iter().map(|&l| res.locals[l].ty));
        carries.push((name.clone(), stages.len(), carried_size));
        for stage in stages {
            let (index, stage_name) = (stage.index, stage.name.clone());
            let (text, usage) =
                FnGen::generate_stage(analysis, &layout, &mut meta, &mut diags, f, Some(stage));
            functions.push_str(&text);
            functions.push('\n');
            if index == 0 {
                scratch.insert(f, usage);
            } else {
                stage_programs.push((stage_name, f, index, usage));
            }
        }
    }
    // the scratch of each program: its frame and the deepest chain of
    // frames below it. At least 8 bytes, so no map has an empty value.
    let mut scratch_maps = Vec::new();
    if options.scratch.is_some() {
        let mut totals = HashMap::new();
        for &(f, _) in &programs {
            let total = scratch_total(&scratch[&f], &scratch, &mut totals).max(8);
            let name = &analysis.package.fns[f].name;
            if total > MAX_SCRATCH {
                diags.push(
                    Diagnostic::error(format!("`{}` needs too much memory", name.text))
                        .with_code(codes::CODEGEN_LIMIT)
                        .with_primary(
                            name.span,
                            format!("{total} bytes of locals, at most {MAX_SCRATCH}"),
                        )
                        .with_note(format!(
                            "locals that do not fit in the {STACK_LIMIT}-byte stack go in a \
                             per-CPU array, whose value has at most {MAX_SCRATCH} bytes; the \
                             locals of the functions it calls count too"
                        ))
                        .with_help("make the large locals smaller, or keep the data in a map"),
                );
            }
            scratch_maps.push((scratch_map(&name.text), total));
        }
        for (stage, _, _, usage) in &stage_programs {
            let total = scratch_total(usage, &scratch, &mut totals).max(8);
            scratch_maps.push((scratch_map(stage), total));
        }
    }
    if diags.has_errors() {
        return Err(diags);
    }
    let (relocations, placeholders) = meta.finish_kernel();
    for (placeholder, name) in placeholders {
        functions = functions.replace(&placeholder, &name);
    }

    let mut ir = String::new();
    let _ = writeln!(ir, "; Generated by vigil.");
    let _ = writeln!(ir, "target datalayout = \"{DATA_LAYOUT}\"");
    let _ = writeln!(ir, "target triple = \"{TRIPLE}\"\n");
    ir.push_str(&layout.type_definitions());
    // GPL, because the printing helper is only for GPL-compatible programs
    let _ = writeln!(
        ir,
        "\n@_license = dso_local global [4 x i8] c\"GPL\\00\", section \"license\", align 1"
    );
    // maps, in `.maps` with BTF, as libbpf declares them (loaders reject the
    // legacy `maps` section). Every global goes in `@llvm.compiler.used`,
    // otherwise LLVM drops the ones no code refers to.
    let mut used = vec!["ptr @_license".to_string()];
    let mut maps = Vec::new();
    let mut define = |ir: &mut String,
                      name: &str,
                      kind: MapKind,
                      sizes: (u64, u64),
                      entries: u64| {
        let mut fields = vec![("type", u64::from(kind.bpf_type()))];
        if kind != MapKind::RingBuf {
            fields.extend([("key_size", sizes.0), ("value_size", sizes.1)]);
        }
        fields.push(("max_entries", entries));
        let dbg = meta
            .map_definition(name, &fields)
            .map_or(String::new(), |node| format!(", !dbg !{node}"));
        let pointers = vec!["ptr"; fields.len()].join(", ");
        let _ = writeln!(
            ir,
            "@\"{name}\" = dso_local global {{ {pointers} }} zeroinitializer, section \".maps\", align 8{dbg}"
        );
    };
    for (m, info) in analysis.maps.iter() {
        let name = &analysis.package.maps[m].name.text;
        let (key_size, value_size) = match info.kind {
            MapKind::RingBuf => (0, 0),
            _ => (
                layout.size_align(info.key).0,
                layout.size_align(info.value).0,
            ),
        };
        define(
            &mut ir,
            name,
            info.kind,
            (key_size, value_size),
            u64::from(info.max_entries),
        );
        used.push(format!("ptr @\"{name}\""));
        maps.push(MapDesc {
            name: name.clone(),
            kind: info.kind,
            key: layout.describe(info.key),
            value: layout.describe(info.value),
            key_size,
            value_size: layout.size_align(info.value).0,
            max_entries: info.max_entries,
        });
    }
    // for each split program, a program array for the tail calls between
    // its stages, and what one stage carries to the next
    for (program, count, size) in &carries {
        let (stages, carry) = (stages_map(program), carry_map(program));
        define(&mut ir, &stages, MapKind::ProgArray, (4, 4), *count as u64);
        define(
            &mut ir,
            &carry,
            MapKind::PerCpuArray,
            (4, (*size).max(8)),
            1,
        );
        used.push(format!("ptr @\"{stages}\""));
        used.push(format!("ptr @\"{carry}\""));
    }
    // the scratch of the programs, one per-CPU array element each
    for (name, size) in &scratch_maps {
        define(&mut ir, name, MapKind::PerCpuArray, (4, *size), 1);
        used.push(format!("ptr @\"{name}\""));
    }
    // tables computed by comptime functions, as read-only data. The linkage
    // is external so LLVM does not mark them `unnamed_addr`, because then
    // small constants go in a mergeable section also called `.rodata`, and
    // loaders take it for the same section. Only the tables the functions
    // refer to are written, because every module of the package has its
    // tables here and loaders make a map of `.rodata`.
    for (c, value) in analysis.comptime_consts.iter() {
        let ty = analysis.consts[c].0;
        let name = format!("@\"vigil.const.{}\"", c.into_raw());
        if let TyKind::Array(..) = analysis.types.kind(ty)
            && functions.contains(&name)
        {
            let _ = writeln!(
                ir,
                "{name} = dso_local constant {} {}, section \".rodata\", align 8",
                layout.ty(ty),
                layout.constant(ty, value)
            );
        }
    }
    let _ = writeln!(
        ir,
        "@llvm.compiler.used = appending global [{} x ptr] [{}], section \"llvm.metadata\"\n",
        used.len(),
        used.join(", ")
    );
    // CO-RE relocations of fields of kernel structs
    ir.push_str(&relocations);
    let _ = writeln!(ir, "declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)");
    let _ = writeln!(ir, "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)");
    let _ = writeln!(ir, "declare void @llvm.lifetime.start.p0(i64, ptr)");
    let _ = writeln!(ir, "declare void @llvm.lifetime.end.p0(i64, ptr)");
    let _ = writeln!(ir, "declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr)");
    for bits in [16, 32, 64] {
        let _ = writeln!(ir, "declare i{bits} @llvm.bswap.i{bits}(i{bits})");
    }
    for bits in [8, 16, 32, 64] {
        let _ = writeln!(
            ir,
            "declare i{bits} @llvm.bpf.passthrough.i{bits}.i{bits}(i32, i{bits})"
        );
    }
    ir.push('\n');
    ir.push_str(&functions);
    let _ = writeln!(ir, "attributes #0 = {{ alwaysinline nounwind }}");
    let _ = writeln!(ir, "attributes #1 = {{ nounwind }}");
    let _ = writeln!(ir, "attributes #2 = {{ \"btf_ama\" }}");
    let _ = writeln!(ir, "attributes #3 = {{ noinline nounwind }}");
    // the empty asm that bounds indices (`FnGen::element_ptr`) only has
    // effects on memory no access names, so loads and stores can move
    // across it. It still needs some effects: without them LLVM merged and
    // moved these calls, which kept values live across loop iterations and
    // made the verifier explore many more states.
    let _ = writeln!(ir, "attributes #4 = {{ nounwind inaccessiblememonly }}\n");
    meta.write(&mut ir);

    let mut programs: Vec<ProgramDesc> = programs
        .into_iter()
        .map(|(f, info)| {
            let name = &analysis.package.fns[f].name;
            ProgramDesc {
                name: name.text.clone(),
                info: info.clone(),
                span: name.span,
            }
        })
        .collect();
    // the stages after the first are programs too, reached by tail calls
    // instead of being attached: no interface, and their slot in the
    // program array
    for (stage, f, index, _) in stage_programs {
        let name = &analysis.package.fns[f].name;
        let mut info = analysis.programs[f].clone();
        info.iface = None;
        info.tail = Some((stages_map(&name.text), index as u32));
        programs.push(ProgramDesc {
            name: stage,
            info,
            span: name.span,
        });
    }
    Ok(Output { ir, programs, maps })
}

/// the functions a function calls at run time
///
/// The calls in its `comptime:` blocks are left out, because the
/// interpreter already ran them and only their values are compiled. An
/// expression is in a block when its span is inside the span of the block.
fn runtime_callees(analysis: &Analysis, f: FnId) -> Vec<FnId> {
    let data = &analysis.package.fns[f];
    let body = &analysis.bodies[f];
    let blocks: Vec<Span> = body
        .comptime_blocks
        .iter()
        .filter_map(|&s| data.spans.stmts.get(s).copied())
        .collect();
    let in_block = |span: Option<&Span>| {
        span.is_some_and(|span| {
            blocks
                .iter()
                .any(|b| b.file == span.file && b.start <= span.start && span.end <= b.end)
        })
    };
    let mut callees: Vec<FnId> = body
        .res
        .iter()
        .filter(|&(e, _)| !in_block(data.spans.exprs.get(e)))
        .filter_map(|(_, res)| match res {
            Res::Fn(g) | Res::Method(g) | Res::Operator(g) => Some(*g),
            _ => None,
        })
        .chain(
            body.aug_operators
                .iter()
                .filter(|&(s, _)| !in_block(data.spans.stmts.get(s)))
                .map(|(_, &g)| g),
        )
        .collect();
    callees.sort_by_key(|g| g.into_raw());
    callees.dedup();
    callees
}

/// the stages of a program split by `stage()`, or `None` for other functions
///
/// Each stage has its top-level statements and the variables carried
/// across stages: the locals (not parameters) that a stage uses and an
/// earlier stage already used. None of them holds a pointer, because the
/// type checker rejects that (the verifier forgets what a pointer stored in
/// a map points to). Locals of size zero are left out.
fn split(analysis: &Analysis, layout: &Layout, f: FnId) -> Option<Vec<Stage>> {
    let boundaries = analysis.stages.get(f)?;
    analysis.programs.get(f)?;
    let data = &analysis.package.fns[f];
    let res = &analysis.bodies[f];
    let mut segments: Vec<Vec<vigil_sema::hir::StmtId>> = vec![Vec::new()];
    for &stmt in &data.body.root {
        if boundaries.contains(&stmt) {
            segments.push(Vec::new());
        } else {
            segments.last_mut().unwrap().push(stmt);
        }
    }
    let params = data.params.len();
    let mut carried = std::collections::BTreeSet::new();
    for b in 0..boundaries.len() {
        let before: std::collections::HashSet<_> = segments[..=b]
            .iter()
            .flat_map(|s| vigil_sema::walk::locals(&data.body, res, s))
            .collect();
        for s in &segments[b + 1..] {
            for local in vigil_sema::walk::locals(&data.body, res, s) {
                let ty = res.locals[local].ty;
                if before.contains(&local)
                    && local.into_raw().into_u32() as usize >= params
                    && !layout.has_pointer(ty)
                    && layout.size_align(ty).0 > 0
                {
                    carried.insert(local.into_raw().into_u32());
                }
            }
        }
    }
    let carried: Vec<_> = carried
        .into_iter()
        .map(|i| la_arena::Idx::from_raw(la_arena::RawIdx::from_u32(i)))
        .collect();
    let count = segments.len();
    Some(
        segments
            .into_iter()
            .enumerate()
            .map(|(index, statements)| Stage {
                index,
                count,
                statements,
                carried: carried.clone(),
                name: if index == 0 {
                    data.name.text.clone()
                } else {
                    format!("{}_stage{index}", data.name.text)
                },
            })
            .collect(),
    )
}

/// the size of a struct with fields of these types, in order with padding
fn struct_size(layout: &Layout, fields: impl Iterator<Item = vigil_sema::ty::TyId>) -> u64 {
    let mut size: u64 = 0;
    let mut align: u64 = 1;
    for ty in fields {
        let (s, a) = layout.size_align(ty);
        size = size.next_multiple_of(a.max(1)) + s;
        align = align.max(a);
    }
    size.next_multiple_of(align)
}

/// the scratch a function needs: its frame, then the largest need of the
/// functions and callbacks it calls
///
/// Totals of functions are memoized in `totals`. A callback counts as part
/// of the function that passes it, so it is counted again each time.
fn scratch_total(
    usage: &ScratchUse,
    functions: &HashMap<FnId, ScratchUse>,
    totals: &mut HashMap<FnId, u64>,
) -> u64 {
    let mut deepest = 0;
    for &f in &usage.calls {
        let total = match totals.get(&f) {
            Some(&total) => total,
            None => {
                let total = functions
                    .get(&f)
                    .map_or(0, |u| scratch_total(u, functions, totals));
                totals.insert(f, total);
                total
            }
        };
        deepest = deepest.max(total);
    }
    for callback in &usage.callbacks {
        deepest = deepest.max(scratch_total(callback, functions, totals));
    }
    usage.frame + deepest
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the IR of an XDP program `prog` that calls `f0`, which calls `f1`,
    /// and so on until `f{n-1}`, none of them inlined
    fn chain_ir(n: usize, linkage: &str) -> String {
        let mut ir = String::from("target triple = \"bpfel\"\n");
        for i in 0..n {
            let body = if i + 1 < n {
                format!(
                    "  %r = call i64 @f{}(i64 %x)\n  %s = add i64 %r, 1\n  ret i64 %s\n",
                    i + 1
                )
            } else {
                "  ret i64 %x\n".to_string()
            };
            let _ = writeln!(ir, "define {linkage} i64 @f{i}(i64 %x) #0 {{\n{body}}}");
        }
        ir.push_str(
            "define dso_local i64 @prog(ptr %ctx) #0 section \"xdp\" {\n  \
             %r = call i64 @f0(i64 1)\n  ret i64 %r\n}\n\
             attributes #0 = { noinline optnone nounwind }\n",
        );
        ir
    }

    #[test]
    fn counts_the_frames_of_calls() {
        let Ok(clang) = Clang::discover() else {
            println!("skipping: no clang");
            return;
        };
        // global functions are called through relocations, internal ones
        // maybe without
        for linkage in ["dso_local", "internal"] {
            for n in [1, 3, 9] {
                let object = clang.compile(&chain_ir(n, linkage), "v3").unwrap();
                let frames = frames(&object, "xdp", "prog");
                assert_eq!(frames.depth, n + 1, "{n} {linkage} functions");
                assert_eq!(frames.stack, 32 * (n + 1), "{n} {linkage} functions");
            }
        }
    }

    #[test]
    fn finds_the_tail_calls_of_stages() {
        let file = SourceMap::new().add("a.vg", "");
        let program = |name: &str, tail: Option<(&str, u32)>| ProgramDesc {
            name: name.into(),
            info: ProgramInfo {
                kind: ProgramKind::Xdp,
                target: String::new(),
                iface: None,
                tail: tail.map(|(m, i)| (m.to_string(), i)),
            },
            span: Span::new(file, 0..0),
        };
        let programs = [
            program("split", None),
            program("split_stage1", Some(("_vigil_stages_split", 1))),
            program("split_stage2", Some(("_vigil_stages_split", 2))),
            // a program of the user's own program array is not a stage
            program("handler", Some(("jumps", 5))),
        ];
        let chains: Vec<_> = stage_chains(&programs)
            .into_iter()
            .map(|(p, n)| (p.name.as_str(), n))
            .collect();
        assert_eq!(chains, [("split", 2)]);
    }
}
