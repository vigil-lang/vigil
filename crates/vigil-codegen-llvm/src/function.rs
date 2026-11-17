//! this module generates the LLVM IR of one function: a program, an
//! internal function, or the body of a loop run by `bpf_loop`. Every local
//! lives in an `alloca`, which the optimizer turns into registers.
//!
//! Some facts the code relies on:
//! - an internal function returns `{ value, i1 }` (or just `i1`), the flag
//!   set when it aborts, and callers propagate it. A program that aborts
//!   returns `ProgramKind::abort_return`. We abort on an index out of
//!   bounds, a full map or a failed lookup of our own per-CPU memory.
//!   Internal functions are always inlined, so the flags go away.
//! - integer operations are those of eBPF: arithmetic wraps,
//!   `x // 0 == 0`, `x % 0 == x`, signed division truncates, shift amounts
//!   are modulo the width. [`FnGen::fold_arith`] folds with the same rules.
//! - with `Layout::scratch`, large values live in a per-CPU array, the
//!   scratch, used like a stack: each internal function gets its frame as a
//!   hidden first parameter. Those values are handled by address
//!   ([`FnGen::by_address`]), because a big SSA aggregate would spill to the
//!   stack anyway.
//! - with `Layout::bpf_loop`, a long loop body becomes a callback that the
//!   verifier checks once. Its locals live in a frame struct, with a flag
//!   saying how the body left and the value of a `return`. Short loops stay
//!   loops or are unrolled by us ([`FnGen::unrolled`]).

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use vigil_diagnostics::{Diagnostic, Diagnostics, codes};
use vigil_sema::Analysis;
use vigil_sema::comptime::CValue;
use vigil_sema::db::ConstValue;
use vigil_sema::hir::*;
use vigil_sema::infer::{InferenceResult, KernelAccess, KernelRead, LocalId, Res};
use vigil_sema::maps::MapMethod;
use vigil_sema::program::{ContextKind, ProgramInfo};
use vigil_sema::resolve::Builtin;
use vigil_sema::ty::{IntTy, TyId, TyKind};

use crate::layout::{Layout, int_const};
use crate::meta::Meta;
use crate::range::{Eval, Range, Ranges};
use vigil_sema::walk;

// helper numbers and limits
// -------------------------

/// the number of the helper `bpf_trace_printk`; the constants after it are
/// the other helpers they name, and the flags of `bpf_map_update_elem`
const TRACE_PRINTK: u32 = 6;
const MAP_LOOKUP: u32 = 1;
const MAP_UPDATE: u32 = 2;
const MAP_DELETE: u32 = 3;
const RINGBUF_OUTPUT: u32 = 130;
const TAIL_CALL: u32 = 12;
const BPF_LOOP: u32 = 181;
const BPF_ANY: u64 = 0;
const BPF_NOEXIST: u64 = 1;

/// the size in bytes above which [`FnGen::fill`] fills an array, and
/// [`FnGen::copy`] copies a value, in a loop. Up to it, the stores are
/// written out, which is shorter and faster.
const FILL_INLINE: u64 = 256;

/// the largest packet offset of the end of a header that `parse` accepts,
/// `MAX_PACKET_OFF` of Linux. The verifier only learns the range of a
/// packet pointer from a check when it is at most this far in, so a header
/// at a variable offset starts at most at `MAX_PACKET_OFFSET - size`.
const MAX_PACKET_OFFSET: u64 = 0xFFFF;
/// the most copies of an `@unroll` loop body that we make ourselves. A
/// longer loop is left to LLVM, with `llvm.loop.unroll.full`.
const MAX_UNROLLED: u64 = 1024;
/// the most iterations of a loop without `@unroll` that we unroll, on any
/// kernel, fewer than [`BPF_LOOP_MIN`]. It is small because unrolling loops
/// of 16 iterations on kernels without `bpf_loop` made the verifier go past
/// its limit on a model of `std.ml`.
const AUTO_UNROLL_ITERATIONS: u64 = 7;
/// the largest size, once unrolled, of a loop without `@unroll` that we
/// unroll, as [`FnGen::size_of`] counts it
const AUTO_UNROLL_BUDGET: u64 = 256;
/// the longest format string of `print`, newline and NUL included. It lives
/// on the 512-byte stack.
const MAX_FORMAT: usize = 128;
/// the fewest iterations of a loop that runs with `bpf_loop`, when the
/// kernels have it. A loop with an unknown number of iterations uses it too.
///
/// The verifier walks other loops one iteration at a time, and a branch on
/// what the iterations carry doubles the paths each time: 2^7 paths stays
/// below the limit.
const BPF_LOOP_MIN: i128 = 8;
/// the LLVM type of the context of a `bpf_loop` callback: the frame, the
/// scratch of the locals, the scratch of the callback, the first value of
/// the loop variable, and the array iterated over
const CALLBACK_CONTEXT: &str = "{ ptr, ptr, ptr, i64, ptr }";

/// how the body of a callback left the loop, in the flag of the frame: a
/// `return`, an early exit of the function (`?` on `None`), or an abort.
/// Zero means none of these.
const FLAG_RETURN: u8 = 1;
const FLAG_EXIT: u8 = 2;
const FLAG_ABORT: u8 = 3;

// the generator
// -------------------------

/// a value: its LLVM operand (or the address of a copy when handled by
/// address, see [`FnGen::by_address`]) and its Vigil type
#[derive(Clone, Debug)]
struct V {
    s: String,
    ty: TyId,
}

/// what we are generating: a program, which returns an `i64`, or an
/// internal function, which returns its value and the abort flag
#[derive(Clone)]
enum Mode {
    Program { info: ProgramInfo, ret: TyId },
    Internal { ret: TyId },
}

/// the LLVM symbol of a function. A program keeps its name; an internal
/// function also gets its id, so two functions with the same name do not
/// collide.
pub(crate) fn symbol(analysis: &Analysis, f: FnId) -> String {
    let data = &analysis.package.fns[f];
    if analysis.programs.get(f).is_some() {
        data.name.text.clone()
    } else {
        format!("vg.{}.{}", f.into_raw().into_u32(), data.name.text)
    }
}

/// the LLVM return type of an internal function: the value and the abort
/// flag
fn internal_ret(layout: &Layout, ret: TyId) -> String {
    if layout.is_unit(ret) {
        "i1".into()
    } else {
        format!("{{ {}, i1 }}", layout.ty(ret))
    }
}

/// where a local lives
#[derive(Clone, Copy, Debug)]
enum Slot {
    /// an `alloca` of the function
    Stack,
    /// a field of the frame struct
    Frame(u32),
    /// an offset in the scratch of the function
    Scratch(u64),
}

/// the scratch of a generated function: its frame, then the frame of the
/// deepest function or callback it calls. Only the frame is known here, so
/// the caller adds up the rest from `calls` and `callbacks` once every
/// function is generated.
#[derive(Debug, Default)]
pub(crate) struct ScratchUse {
    /// bytes of the function's own frame, a multiple of 8
    pub frame: u64,
    /// the internal functions it calls
    pub calls: Vec<FnId>,
    /// the callbacks of its loops
    pub callbacks: Vec<ScratchUse>,
}

/// what a loop run by `bpf_loop` iterates over
enum Iteration {
    /// a `range`: the loop variable is `start + index`
    Range { start: V, end: V },
    /// an array: the loop variable is the element at `index`
    Array { array: V },
    /// a `while`, run at most `bound` times
    While { cond: ExprId, bound: u64 },
}

/// a stage of a program split by `stage()`: a program of its own, reached
/// from the previous stage by a tail call. A tail call keeps neither the
/// stack nor the registers, so the variables that live across stages go in
/// a per-CPU array ([`carry_map`]).
#[derive(Clone, Debug)]
pub(crate) struct Stage {
    /// which stage this is, from 0
    pub index: usize,
    /// how many stages the program has
    pub count: usize,
    /// the statements of the body in this stage
    pub statements: Vec<StmtId>,
    /// the locals that live across stages
    pub carried: Vec<LocalId>,
    /// the symbol of the program of the stage
    pub name: String,
}

/// the name of the program array of the stages of a program
pub(crate) fn stages_map(program: &str) -> String {
    format!("_vigil_stages_{program}")
}

/// the name of the per-CPU array that carries variables from a stage to the
/// next
pub(crate) fn carry_map(program: &str) -> String {
    format!("_vigil_carry_{program}")
}

/// the generator of one function, or of one loop callback
///
/// It emits instructions into `code` while walking the body, keeping track
/// of the open block (`block`, `terminated`). The `alloca`s and the prologue
/// are kept apart and put first by [`FnGen::finish`].
pub(crate) struct FnGen<'a, 'm> {
    an: &'a Analysis,
    layout: &'a Layout<'a>,
    meta: &'m mut Meta<'a>,
    diags: &'m mut Diagnostics,
    function: FnId,
    data: &'a FnData,
    res: &'a InferenceResult,
    mode: Mode,
    sp: usize,
    allocas: Vec<String>,
    code: Vec<String>,
    tmp: u32,
    labels: u32,
    block: String,
    terminated: bool,
    loc: usize,
    /// the `continue` and `break` targets of each enclosing loop
    loops: Vec<(String, String)>,
    /// whether some instruction branches to the `abort` block
    aborts: bool,
    /// instructions after the allocas: the scratch of a program, the context
    /// of a callback, then the addresses of the locals
    prologue: Vec<String>,
    slots: Vec<(LocalId, Slot)>,
    /// the LLVM type of the frame struct, when a loop runs with `bpf_loop`
    frame: Option<String>,
    /// the fields of the frame with the flag and the returned value
    flag_field: u32,
    ret_field: Option<u32>,
    /// the offset of a large returned value in the scratch of the locals
    ret_scratch: Option<u64>,
    /// whether we generate a loop body as a `bpf_loop` callback
    callback: bool,
    /// bytes of scratch used, and the placeholder of their final count.
    /// Callees get the address past the frame before the frame is complete,
    /// so [`FnGen::finish`] writes the count in at the end.
    scratch_used: u64,
    scratch_mark: String,
    calls: Vec<FnId>,
    callbacks: Vec<ScratchUse>,
    /// callbacks generated for the loops, emitted after the function
    extra: Vec<String>,
    /// the symbol of the program (of the stage, for a split program)
    program_name: String,
    /// locals whose value is a known constant here: the variable of an
    /// unrolled loop, in each copy of the body
    const_locals: HashMap<LocalId, String>,
    /// the ranges of the locals of the function ([`crate::range`])
    ranges: Rc<Ranges>,
    /// the alignment of the addresses that may be less aligned than their
    /// type: parts of a packed `@network_order` struct, and what is inside
    /// them ([`FnGen::align_at`])
    aligns: HashMap<String, u64>,
    /// the addresses of large locals and temporaries, aligned to 8 by
    /// [`FnGen::slot_align`]
    wide: HashSet<String>,
    /// the loops of [`FnGen::counted_loop`] not closed yet, innermost last
    counted: Vec<Counted>,
}

/// a loop of [`FnGen::counted_loop`]: the line of the `phi` of its counter,
/// the counter and its next value, its first block and the one after it,
/// the end and the step
struct Counted {
    phi: usize,
    counter: String,
    next: String,
    body: String,
    done: String,
    end: u64,
    step: u64,
}

impl<'a, 'm> FnGen<'a, 'm> {
    /// the text of a function, followed by its callbacks, and the scratch it
    /// uses
    pub fn generate(
        an: &'a Analysis,
        layout: &'a Layout<'a>,
        meta: &'m mut Meta<'a>,
        diags: &'m mut Diagnostics,
        f: FnId,
    ) -> (String, ScratchUse) {
        Self::generate_stage(an, layout, meta, diags, f, None)
    }

    /// the text of a function, or of one stage of a program split by
    /// `stage()`, followed by its callbacks, and the scratch it uses
    pub fn generate_stage(
        an: &'a Analysis,
        layout: &'a Layout<'a>,
        meta: &'m mut Meta<'a>,
        diags: &'m mut Diagnostics,
        f: FnId,
        stage: Option<Stage>,
    ) -> (String, ScratchUse) {
        let data = &an.package.fns[f];
        let sig = &an.fn_sigs[f];
        let mode = match an.programs.get(f) {
            Some(info) => Mode::Program {
                info: info.clone(),
                ret: sig.ret,
            },
            None => Mode::Internal { ret: sig.ret },
        };
        let program = matches!(mode, Mode::Program { .. });
        let sp = meta.subprogram(&data.name.text, data.name.span, program);
        let loc = meta.location(sp, data.name.span);
        let mark = format!("@@scratch.{}@@", meta.unique());
        let mut g = FnGen {
            an,
            layout,
            meta,
            diags,
            function: f,
            data,
            res: &an.bodies[f],
            mode,
            sp,
            allocas: Vec::new(),
            code: Vec::new(),
            tmp: 0,
            labels: 0,
            block: "entry".into(),
            terminated: false,
            loc,
            loops: Vec::new(),
            aborts: false,
            prologue: Vec::new(),
            slots: Vec::new(),
            frame: None,
            flag_field: 0,
            ret_field: None,
            ret_scratch: None,
            callback: false,
            scratch_used: 0,
            aligns: HashMap::new(),
            wide: HashSet::new(),
            counted: Vec::new(),
            scratch_mark: mark,
            calls: Vec::new(),
            callbacks: Vec::new(),
            extra: Vec::new(),
            program_name: stage
                .as_ref()
                .map_or_else(|| data.name.text.clone(), |s| s.name.clone()),
            const_locals: HashMap::new(),
            ranges: Rc::new(if layout.optimize {
                Ranges::of_function(an, f, &data.body, &an.bodies[f])
            } else {
                Ranges::default()
            }),
        };

        let header = g.header(f);
        g.prologue();
        match stage {
            None => g.block(&data.body.root),
            Some(stage) => g.stage_body(&stage),
        }
        g.epilogue();
        g.finish(header)
    }

    /// the statements of a stage, between the variables received from the
    /// previous stage and those given to the next, which it then calls
    fn stage_body(&mut self, stage: &Stage) {
        if stage.index == 0 {
            // every carried variable is written before it is read, because
            // the verifier rejects reads of memory never written
            for &local in &stage.carried {
                let ty = self.res.locals[local].ty;
                let zero = self.zero(ty);
                let ptr = self.local_ptr(local);
                self.store(&zero, &ptr);
            }
        } else {
            let carry = self.carry();
            for (i, &local) in stage.carried.iter().enumerate() {
                let ty = self.res.locals[local].ty;
                let field = self.carry_field(stage, &carry, i);
                let value = self.load(&field, ty);
                let ptr = self.local_ptr(local);
                self.store(&value, &ptr);
            }
        }
        let statements = stage.statements.clone();
        for stmt in statements {
            self.stmt(stmt);
        }
        if stage.index + 1 < stage.count && !self.terminated {
            let carry = self.carry();
            for (i, &local) in stage.carried.iter().enumerate() {
                let ty = self.res.locals[local].ty;
                let ptr = self.local_ptr(local);
                let value = self.load(&ptr, ty);
                let field = self.carry_field(stage, &carry, i);
                self.store(&value, &field);
            }
            let base = &self.data.name.text;
            let map = stages_map(base);
            let result = self.tmp();
            self.emit(format!(
                "{result} = call i64 inttoptr (i64 {TAIL_CALL} to ptr)(ptr %a0, ptr @\"{map}\", i64 {})",
                stage.index + 1
            ));
            // the tail call only returns when the next stage was not loaded,
            // and then the program returns its default
            self.early_exit(false);
        }
    }

    /// the LLVM type of the variables carried between stages
    fn carry_type(&self, stage: &Stage) -> String {
        let fields: Vec<String> = stage
            .carried
            .iter()
            .map(|&l| self.ll(self.res.locals[l].ty))
            .collect();
        format!("{{ {} }}", fields.join(", "))
    }

    /// the address of the element of the carry array. The program aborts if
    /// the lookup fails.
    fn carry(&mut self) -> String {
        let key = self.stack_alloca("i32", 4);
        self.emit(format!("store i32 0, ptr {key}, align 4"));
        let map = carry_map(&self.data.name.text);
        let ptr = self.tmp();
        self.emit(format!(
            "{ptr} = call ptr inttoptr (i64 {MAP_LOOKUP} to ptr)(ptr @\"{map}\", ptr {key})"
        ));
        let missing = self.tmp();
        self.emit(format!("{missing} = icmp eq ptr {ptr}, null"));
        let ok = self.label("carry");
        self.aborts = true;
        self.cond_br(&missing, "abort", &ok);
        self.place(ok);
        ptr
    }

    /// the address of a carried variable in the element of the carry array
    fn carry_field(&mut self, stage: &Stage, carry: &str, field: usize) -> String {
        let ty = self.carry_type(stage);
        self.gep(&ty, carry, &format!("i32 0, i32 {field}"))
    }

    /// the text of the function, followed by its callbacks, and the scratch
    /// it uses
    fn finish(self, header: String) -> (String, ScratchUse) {
        let mut out = header;
        out.push_str("entry:\n");
        for line in self.allocas.iter().chain(&self.prologue).chain(&self.code) {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("}\n");
        let frame = self.scratch_used.next_multiple_of(8);
        let mut out = out.replace(&self.scratch_mark, &frame.to_string());
        for callback in &self.extra {
            out.push('\n');
            out.push_str(callback);
        }
        let usage = ScratchUse {
            frame,
            calls: self.calls,
            callbacks: self.callbacks,
        };
        (out, usage)
    }

    /// the `define` line of the function. A program takes its context; an
    /// internal function takes its scratch, then the address of its result
    /// when handled by address, then its parameters.
    fn header(&self, f: FnId) -> String {
        let name = match &self.mode {
            Mode::Program { .. } => self.program_name.clone(),
            Mode::Internal { .. } => symbol(self.an, f),
        };
        match &self.mode {
            Mode::Program { info, .. } => format!(
                "define dso_local i64 @\"{name}\"(ptr %a0) #1 section \"{}\" !dbg !{} {{\n",
                info.section(),
                self.sp
            ),
            Mode::Internal { ret } => {
                let mut params = vec!["ptr %sc".to_string()];
                if self.by_address(*ret) {
                    params.push("ptr %ret".into());
                }
                for (i, (_, ty)) in self.an.fn_sigs[f].params.iter().enumerate() {
                    params.push(format!("{} %a{i}", self.value_ll(*ty)));
                }
                format!(
                    "define internal {} @\"{name}\"({}) #0 !dbg !{} {{\n",
                    self.internal_ret(*ret),
                    params.join(", "),
                    self.sp
                )
            }
        }
    }

    /// placing the locals (stack, frame or scratch) and storing the
    /// parameters, which are the first locals declared
    fn prologue(&mut self) {
        // `result` exists only in contracts, which are not compiled, so it
        // gets no place
        let locals: Vec<_> = self
            .res
            .locals
            .iter()
            .filter(|&(id, _)| self.res.result_local != Some(id))
            .map(|(id, l)| (id, l.ty))
            .collect();
        if matches!(self.mode, Mode::Program { .. }) && self.layout.scratch.is_some() {
            self.scratch_lookup();
        }
        // the locals of loops run by `bpf_loop` go in the frame, because that
        // is all a callback can reach of the function
        let mut all = Vec::new();
        walk::stmts(&self.data.body, &self.data.body.root, &mut all);
        let loops: Vec<StmtId> = all
            .into_iter()
            .filter(|&s| self.runs_with_bpf_loop(s))
            .collect();
        let mut in_frame = HashSet::new();
        for &stmt in &loops {
            in_frame.extend(walk::locals(&self.data.body, self.res, &vec![stmt]));
        }
        // the fields of the frame, with sizes and alignments
        let mut fields = Vec::new();
        let mut field_layouts = Vec::new();
        let mut frame_pointers = false;
        for &(id, ty) in &locals {
            if !self.representable(ty) {
                continue;
            }
            let (size, _) = self.layout.size_align(ty);
            let align = self.slot_align(ty);
            if size > FILL_INLINE {
                self.wide.insert(self.local_ptr(id));
            }
            // a frame local is in memory anyway, so it may as well be in
            // scratch, unless it holds pointers, which stay on the stack
            let scratch_frame = self.layout.scratch.is_some()
                && in_frame.contains(&id)
                && !self.layout.has_pointer(ty);
            let slot = if self.layout.in_scratch(ty) || scratch_frame {
                Slot::Scratch(self.scratch_alloc(size, align))
            } else if in_frame.contains(&id) {
                self.wide.remove(&self.local_ptr(id));
                fields.push(self.layout.ty(ty));
                field_layouts.push(self.layout.size_align(ty));
                frame_pointers |= self.layout.has_pointer(ty);
                Slot::Frame(fields.len() as u32 - 1)
            } else {
                let ll = self.layout.ty(ty);
                self.allocas.push(format!(
                    "  %l{} = alloca {ll}, align {align}",
                    id.into_raw().into_u32()
                ));
                Slot::Stack
            };
            self.slots.push((id, slot));
        }
        if !loops.is_empty() {
            self.flag_field = fields.len() as u32;
            fields.push("i8".into());
            field_layouts.push((1, 1));
            // a large returned value has its own place in scratch, not a
            // field of the frame
            match self.ret_slot_type() {
                Some(ret) if self.by_address(ret) => {
                    let (size, align) = self.layout.size_align(ret);
                    self.ret_scratch = Some(self.scratch_alloc(size, align));
                }
                Some(ret) => {
                    self.ret_field = Some(fields.len() as u32);
                    fields.push(self.layout.ty(ret));
                    field_layouts.push(self.layout.size_align(ret));
                    frame_pointers |= self.layout.has_pointer(ret);
                }
                None => {}
            }
            let ty = format!("{{ {} }}", fields.join(", "));
            // a large frame goes in scratch too, unless it holds pointers.
            // Its size follows the C layout rules.
            let mut size: u64 = 0;
            for &(field_size, align) in &field_layouts {
                size = size.next_multiple_of(align.max(1)) + field_size;
            }
            if !frame_pointers && self.layout.scratch.is_some_and(|min| size >= min) {
                let offset = self.scratch_alloc(size.next_multiple_of(8), 8);
                self.prologue.push(format!(
                    "  %frame = getelementptr i8, ptr %sc, i64 {offset}"
                ));
            } else {
                self.allocas
                    .push(format!("  %frame = alloca {ty}, align 8"));
            }
            self.frame = Some(ty);
        }
        self.local_addresses("%sc");
        for (i, _) in self.data.params.iter().enumerate() {
            let (id, ty) = locals[i];
            let local = format!("%l{}", id.into_raw().into_u32());
            self.store(
                &V {
                    s: format!("%a{i}"),
                    ty,
                },
                &local,
            );
        }
    }

    /// looking up the scratch of a program, the only element of its per-CPU
    /// array. The program aborts if the lookup fails.
    fn scratch_lookup(&mut self) {
        let name = &self.program_name;
        let loc = self.loc;
        self.allocas
            .push("  %sckey = alloca i32, align 4".to_string());
        self.prologue.extend([
            format!("  store i32 0, ptr %sckey, align 4, !dbg !{loc}"),
            format!(
                "  %sc = call ptr inttoptr (i64 {MAP_LOOKUP} to ptr)(ptr @\"{}\", ptr %sckey), !dbg !{loc}",
                scratch_map(name)
            ),
            format!("  %scnull = icmp eq ptr %sc, null, !dbg !{loc}"),
            format!("  br i1 %scnull, label %abort, label %start, !dbg !{loc}"),
            "start:".to_string(),
        ]);
        self.block = "start".into();
        self.aborts = true;
    }

    /// computing the addresses of the locals not on the stack: fields of the
    /// frame, and offsets in `base`, the scratch of the locals
    fn local_addresses(&mut self, base: &str) {
        for &(id, slot) in &self.slots {
            let address = match slot {
                Slot::Stack => continue,
                Slot::Frame(field) => format!(
                    "getelementptr {}, ptr %frame, i32 0, i32 {field}",
                    self.frame.as_deref().unwrap_or("{}")
                ),
                Slot::Scratch(offset) => format!("getelementptr i8, ptr {base}, i64 {offset}"),
            };
            self.prologue
                .push(format!("  %l{} = {address}", id.into_raw().into_u32()));
        }
    }

    /// the type of the returned value kept for a callback: the `u64` of a
    /// program, the value of an internal function, `None` for unit
    fn ret_slot_type(&self) -> Option<TyId> {
        match &self.mode {
            Mode::Program { .. } => Some(self.an_int(IntTy::U64)),
            Mode::Internal { ret } if self.layout.is_unit(*ret) => None,
            Mode::Internal { ret } => Some(*ret),
        }
    }

    /// the offset of `size` newly reserved bytes of scratch
    fn scratch_alloc(&mut self, size: u64, align: u64) -> u64 {
        let offset = self.scratch_used.next_multiple_of(align.max(1));
        self.scratch_used = offset + size;
        offset
    }

    /// the scratch of the callees: just past our own, or `null` without
    /// scratch. The offset is a placeholder that [`FnGen::finish`] replaces.
    fn callee_scratch(&mut self) -> String {
        if self.layout.scratch.is_none() {
            return "null".into();
        }
        let name = self.tmp();
        let mark = self.scratch_mark.clone();
        self.emit(format!("{name} = getelementptr i8, ptr %sc, i64 {mark}"));
        name
    }

    /// where a callback leaves the value of a `return`, if there is one
    fn ret_slot(&mut self) -> Option<String> {
        if let Some(offset) = self.ret_scratch {
            let base = if self.callback { "%lsc" } else { "%sc" };
            return Some(self.gep("i8", base, &format!("i64 {offset}")));
        }
        self.ret_field.map(|field| self.frame_field(field))
    }

    /// the address of a field of the frame
    fn frame_field(&mut self, field: u32) -> String {
        let ty = self.frame.clone().unwrap_or_default();
        self.gep(&ty, "%frame", &format!("i32 0, i32 {field}"))
    }

    /// writing the flag of the frame (`FLAG_*`)
    fn set_flag(&mut self, flag: u8) {
        let ptr = self.frame_field(self.flag_field);
        self.emit(format!("store i8 {flag}, ptr {ptr}, align 1"));
    }

    /// ending the body: the implicit return at the end, then the `abort`
    /// block if anything branches to it
    fn epilogue(&mut self) {
        self.loc = self.meta.location(self.sp, self.data.name.span);
        if !self.terminated {
            match &self.mode {
                Mode::Internal { ret } if self.layout.is_unit(*ret) => {
                    self.terminate("ret i1 false")
                }
                // a function with a value cannot reach its end, the type
                // checker ensures it
                _ => self.terminate("unreachable"),
            }
        }
        if self.aborts {
            self.place("abort".into());
            self.early_exit(true);
        }
    }

    /// returning without a value: the default of a program (its abort value
    /// if `aborted`), or the result of an internal function that aborted or
    /// gave `None`. A callback sets the flag of the frame and stops the loop
    /// instead.
    fn early_exit(&mut self, aborted: bool) {
        if self.callback {
            self.set_flag(if aborted { FLAG_ABORT } else { FLAG_EXIT });
            self.terminate("ret i64 1");
            return;
        }
        match &self.mode {
            Mode::Program { info, .. } => {
                let value = if aborted {
                    info.kind.abort_return()
                } else {
                    info.kind.default_return()
                };
                self.terminate(format!("ret i64 {value}"));
            }
            Mode::Internal { ret } => {
                let ret = *ret;
                if self.by_address(ret) && !aborted {
                    // `None` is all zeros
                    self.memset_zero("%ret", ret);
                }
                if self.layout.is_unit(ret) || self.by_address(ret) {
                    self.terminate(format!("ret i1 {aborted}"));
                    return;
                }
                let ll = self.internal_ret(ret);
                // zero even when aborting. With `poison`, LLVM may copy the
                // value from any register, and the verifier rejects reading
                // a register never written or clobbered by a call.
                let value = "zeroinitializer";
                let inner = self.layout.ty(ret);
                let a = self.tmp();
                self.emit(format!("{a} = insertvalue {ll} poison, {inner} {value}, 0"));
                let b = self.tmp();
                self.emit(format!("{b} = insertvalue {ll} {a}, i1 {aborted}, 1"));
                self.terminate(format!("ret {ll} {b}"));
            }
        }
    }

    /// whether values of `ty` exist at run time: not unit, strings, floats,
    /// or the types of unresolved or wrong expressions
    fn representable(&self, ty: TyId) -> bool {
        !matches!(
            self.layout.kind(ty),
            TyKind::Unit | TyKind::Str | TyKind::Var(_) | TyKind::Error | TyKind::Float
        )
    }

    // emission
    // -------------------------

    /// a fresh temporary, `%tN`
    fn tmp(&mut self) -> String {
        self.tmp += 1;
        format!("%t{}", self.tmp)
    }

    /// a fresh label, made of `hint` and a number
    fn label(&mut self, hint: &str) -> String {
        self.labels += 1;
        format!("{hint}{}", self.labels)
    }

    /// appending an instruction, with the current debug location
    fn emit(&mut self, instr: impl AsRef<str>) {
        self.emit_md(instr, "");
    }

    /// appending an instruction with more metadata after its location. Code
    /// after a terminator (after a `return` or a `break`) goes in a fresh
    /// block nothing branches to, because LLVM accepts no instruction after
    /// a terminator.
    fn emit_md(&mut self, instr: impl AsRef<str>, extra: &str) {
        if self.terminated {
            let dead = self.label("dead");
            self.place(dead);
        }
        let line = format!("  {}, !dbg !{}{extra}", instr.as_ref(), self.loc);
        self.code.push(line);
    }

    /// appending the terminator of the current block
    fn terminate(&mut self, instr: impl AsRef<str>) {
        self.emit(instr);
        self.terminated = true;
    }

    fn br(&mut self, target: &str) {
        self.terminate(format!("br label %{target}"));
    }

    fn cond_br(&mut self, cond: &str, yes: &str, no: &str) {
        self.terminate(format!("br i1 {cond}, label %{yes}, label %{no}"));
    }

    /// starting a block, falling through from the current one if it is still
    /// open
    fn place(&mut self, label: String) {
        if !self.terminated {
            self.code
                .push(format!("  br label %{label}, !dbg !{}", self.loc));
        }
        self.code.push(format!("{label}:"));
        self.block = label;
        self.terminated = false;
    }

    /// the address of fresh memory for a value of type `ty`: on the stack,
    /// or in scratch when it is large
    fn alloca(&mut self, ty: TyId) -> String {
        let size = self.layout.size_align(ty).0;
        let align = self.slot_align(ty);
        let name = if self.layout.in_scratch(ty) {
            let offset = self.scratch_alloc(size, align);
            let name = self.tmp();
            self.prologue.push(format!(
                "  {name} = getelementptr i8, ptr %sc, i64 {offset}"
            ));
            name
        } else {
            let ll = self.layout.ty(ty);
            self.stack_alloca(&ll, align)
        };
        if size > FILL_INLINE {
            self.wide.insert(name.clone());
        }
        name
    }

    /// the alignment of fresh memory for a value of type `ty`: the one of
    /// the type, and 8 for a large value, so [`FnGen::fill`] writes it 8
    /// bytes at a time
    fn slot_align(&self, ty: TyId) -> u64 {
        let (size, align) = self.layout.size_align(ty);
        if size > FILL_INLINE {
            align.max(8)
        } else {
            align
        }
    }

    /// the address of a fresh `alloca` of LLVM type `ll`
    fn stack_alloca(&mut self, ll: &str, align: u64) -> String {
        let name = self.tmp();
        self.allocas
            .push(format!("  {name} = alloca {ll}, align {}", align.max(1)));
        name
    }

    fn set_loc(&mut self, span: vigil_diagnostics::Span) {
        self.loc = self.meta.location(self.sp, span);
    }

    fn ll(&self, ty: TyId) -> String {
        self.layout.ty(ty)
    }

    /// the integer type of `ty`; `i64` for anything else, like a kernel
    /// pointer
    fn int_ty(&self, ty: TyId) -> IntTy {
        match self.layout.kind(ty) {
            TyKind::Int(int) => *int,
            _ => IntTy::I64,
        }
    }

    /// whether values of `ty` are handled by address: with scratch, a large
    /// value is always the address of a copy only it uses
    fn by_address(&self, ty: TyId) -> bool {
        self.representable(ty) && !self.layout.is_unit(ty) && self.layout.in_scratch(ty)
    }

    /// the LLVM type of a value of `ty`, `ptr` when it is by address
    fn value_ll(&self, ty: TyId) -> String {
        if self.by_address(ty) {
            "ptr".into()
        } else {
            self.ll(ty)
        }
    }

    /// the LLVM return type of an internal function: the value and the abort
    /// flag, or only the flag when the value is unit or written to `%ret`
    fn internal_ret(&self, ret: TyId) -> String {
        if self.by_address(ret) {
            "i1".into()
        } else {
            internal_ret(self.layout, ret)
        }
    }

    /// copying a value of type `ty` word by word. A `memcpy` would load every
    /// word before storing any, taking a register for each, so we use
    /// volatile accesses to keep each load next to its store. A large value
    /// is copied in a loop, as [`FnGen::fill_words`] fills one.
    fn copy(&mut self, to: &str, from: &str, ty: TyId) {
        let (size, _) = self.layout.size_align(ty);
        let word = self.word_at(to, ty).min(self.word_at(from, ty));
        if size > FILL_INLINE {
            let end = size - size % word;
            let int = format!("i{}", word * 8);
            let i = self.counted_loop(end, word);
            let source = self.tmp();
            self.emit(format!("{source} = getelementptr i8, ptr {from}, i64 {i}"));
            let target = self.tmp();
            self.emit(format!("{target} = getelementptr i8, ptr {to}, i64 {i}"));
            let value = self.tmp();
            self.emit(format!(
                "{value} = load volatile {int}, ptr {source}, align {word}"
            ));
            self.emit(format!(
                "store volatile {int} {value}, ptr {target}, align {word}"
            ));
            self.end_loop();
            // the rest is shorter than a word
            let mut offset = end;
            let mut width = word / 2;
            while offset < size {
                if width <= size - offset {
                    let int = format!("i{}", width * 8);
                    let source = self.gep("i8", from, &format!("i64 {offset}"));
                    let target = self.gep("i8", to, &format!("i64 {offset}"));
                    let value = self.tmp();
                    self.emit(format!(
                        "{value} = load volatile {int}, ptr {source}, align {width}"
                    ));
                    self.emit(format!(
                        "store volatile {int} {value}, ptr {target}, align {width}"
                    ));
                    offset += width;
                }
                width /= 2;
            }
            return;
        }
        let int = format!("i{}", word * 8);
        for offset in (0..size).step_by(word as usize) {
            let (source, target) = if offset == 0 {
                (from.to_string(), to.to_string())
            } else {
                (
                    self.gep("i8", from, &format!("i64 {offset}")),
                    self.gep("i8", to, &format!("i64 {offset}")),
                )
            };
            let value = self.tmp();
            self.emit(format!(
                "{value} = load volatile {int}, ptr {source}, align {word}"
            ));
            self.emit(format!(
                "store volatile {int} {value}, ptr {target}, align {word}"
            ));
        }
    }

    /// the value of type `ty` at `ptr`: a load, or a copy when handled by
    /// address
    fn load(&mut self, ptr: &str, ty: TyId) -> V {
        self.load_tagged(ptr, ty, "")
    }

    /// [`FnGen::load`] with metadata after the instruction, like an alias
    /// tag from [`FnGen::access_tag`]
    fn load_tagged(&mut self, ptr: &str, ty: TyId, tag: &str) -> V {
        if self.by_address(ty) {
            let copy = self.alloca(ty);
            self.copy(&copy, ptr, ty);
            return V { s: copy, ty };
        }
        let name = self.tmp();
        let ll = self.ll(ty);
        let align = self.align_at(ptr, ty);
        self.emit_md(format!("{name} = load {ll}, ptr {ptr}, align {align}"), tag);
        V { s: name, ty }
    }

    /// storing a value at `ptr`, or copying it there when handled by address
    fn store(&mut self, value: &V, ptr: &str) {
        self.store_tagged(value, ptr, "");
    }

    /// [`FnGen::store`] with metadata after the instruction
    fn store_tagged(&mut self, value: &V, ptr: &str, tag: &str) {
        if self.by_address(value.ty) {
            self.copy(ptr, &value.s, value.ty);
            return;
        }
        let ll = self.ll(value.ty);
        let align = self.align_at(ptr, value.ty);
        self.emit_md(
            format!("store {ll} {}, ptr {ptr}, align {align}", value.s),
            tag,
        );
    }

    /// the alias tag of an access to the place `e` (a field or an element):
    /// memory that may be in the packet, or memory that never is. The type
    /// of the value the place is part of, its `root`, decides.
    ///
    /// With the tags, LLVM knows writing a map value does not change the
    /// packet, and reads a header field like `ip.saddr` only once. An access
    /// without a tag (root not an aggregate, or holding packet types among
    /// others) may touch any memory.
    fn access_tag(&mut self, e: ExprId) -> String {
        if !self.layout.optimize {
            return String::new();
        }
        let body = &self.data.body;
        let mut root = e;
        while let Expr::Field { base, .. } | Expr::Index { base, .. } = body.exprs[root] {
            root = base;
        }
        if root == e {
            return String::new();
        }
        let Some(&ty) = self.res.expr_types.get(root) else {
            return String::new();
        };
        let ty = match *self.layout.kind(ty) {
            TyKind::Ref(inner) => inner,
            _ => ty,
        };
        let packet = if self.layout.packet_types.contains(&ty) {
            true
        } else if self.layout.packet_free(ty) {
            false
        } else {
            return String::new();
        };
        format!(", !tbaa !{}", self.meta.tbaa(packet))
    }

    /// the address of the receiver of a method, which takes `self` by
    /// reference, and the place to copy it back to after the call, if any
    ///
    /// The method reads `self` as aligned as its type, so a receiver that is
    /// less aligned, inside a packed struct, is passed as an aligned copy. A
    /// `@mut` method changes the copy, which we then copy back.
    fn receiver(&mut self, method: FnId, base: ExprId) -> Option<(V, Option<String>)> {
        let self_ty = self.an.fn_sigs[method].params[0].1;
        let s = match *self.layout.kind(self.res.expr_types[base]) {
            TyKind::Ref(_) => self.expr_inner(base)?.s,
            _ if self.is_place(base) => self.place_ptr(base).0,
            _ => {
                let value = self.expr(base)?;
                self.spill(&value)
            }
        };
        let pointee = match *self.layout.kind(self_ty) {
            TyKind::Ref(inner) => inner,
            _ => self_ty,
        };
        if self.align_at(&s, pointee) < self.layout.size_align(pointee).1 {
            let copy = self.alloca(pointee);
            self.copy(&copy, &s, pointee);
            let mutating = self.an.package.fns[method]
                .decorators
                .iter()
                .any(|d| d.name.to_string() == "mut");
            let back = mutating.then_some(s);
            return Some((
                V {
                    s: copy,
                    ty: self_ty,
                },
                back,
            ));
        }
        Some((V { s, ty: self_ty }, None))
    }

    /// a call of the method `f`, with the receiver from
    /// [`FnGen::receiver`] first in `values`, and copying the receiver back
    /// to `back`
    fn method_call(&mut self, f: FnId, values: Vec<V>, back: Option<String>) -> Option<V> {
        let receiver = values.first().map(|v| v.s.clone());
        let result = self.internal_call(f, values);
        if let (Some(back), Some(receiver)) = (back, receiver) {
            let self_ty = self.an.fn_sigs[f].params[0].1;
            let pointee = match *self.layout.kind(self_ty) {
                TyKind::Ref(inner) => inner,
                _ => self_ty,
            };
            self.copy(&back, &receiver, pointee);
        }
        result
    }

    /// the address of a value: a fresh copy in memory, or its own address
    /// when handled by address
    fn spill(&mut self, value: &V) -> String {
        if self.by_address(value.ty) {
            return value.s.clone();
        }
        let ptr = self.alloca(value.ty);
        self.store(value, &ptr);
        ptr
    }

    /// element `index`, of type `ty`, of an aggregate value
    fn extract(&mut self, aggregate: &V, index: u32, ty: TyId) -> V {
        if self.by_address(aggregate.ty) {
            let ptr = self.part_ptr(&aggregate.s, aggregate.ty, index);
            if self.by_address(ty) {
                return V { s: ptr, ty };
            }
            return self.load(&ptr, ty);
        }
        let name = self.tmp();
        let ll = self.ll(aggregate.ty);
        self.emit(format!(
            "{name} = extractvalue {ll} {}, {index}",
            aggregate.s
        ));
        V { s: name, ty }
    }

    /// a value of the aggregate type `ty`, built from its elements
    fn aggregate_value(&mut self, ty: TyId, elements: &[V]) -> V {
        let ll = self.ll(ty);
        if !self.by_address(ty) {
            let s = self.aggregate(&ll, elements);
            return V { s, ty };
        }
        let ptr = self.alloca(ty);
        for (i, element) in elements.iter().enumerate() {
            let field = self.part_ptr(&ptr, ty, i as u32);
            self.store(element, &field);
        }
        V { s: ptr, ty }
    }

    /// the zero value of `ty`, which is `None` for an option
    fn zero(&mut self, ty: TyId) -> V {
        if self.by_address(ty) {
            let ptr = self.alloca(ty);
            self.memset_zero(&ptr, ty);
            return V { s: ptr, ty };
        }
        V {
            s: "zeroinitializer".into(),
            ty,
        }
    }

    /// writing zeros over the value of type `ty` at `ptr`
    fn memset_zero(&mut self, ptr: &str, ty: TyId) {
        let size = self.layout.size_align(ty).0;
        if size > FILL_INLINE {
            self.fill_words(ptr, ty, 0, 1);
            return;
        }
        let align = self.align_at(ptr, ty);
        self.emit(format!(
            "call void @llvm.memset.p0.i64(ptr align {align} {ptr}, i8 0, i64 {size}, i1 false)"
        ));
    }

    /// writing `pattern`, `bytes` long, over and over on the value of type
    /// `ty` at `ptr`, in a loop of the widest stores its alignment allows,
    /// then the rest with narrower ones
    ///
    /// LLVM would turn a large `memset` into a call of `memset`, which eBPF
    /// does not have, and turns a loop of plain stores back into one, so the
    /// stores are volatile. The loop has a constant trip count, which the
    /// verifier follows; 8-byte stores keep it short.
    fn fill_words(&mut self, ptr: &str, ty: TyId, pattern: u64, bytes: u64) {
        let size = self.layout.size_align(ty).0;
        let align = self.word_at(ptr, ty);
        let word = align.max(bytes);
        // the pattern repeated over a word, little-endian
        let mut splat: u64 = 0;
        for k in 0..word / bytes {
            splat |= pattern << (8 * bytes * k);
        }
        let int = |width: u64| match width {
            1 => IntTy::U8,
            2 => IntTy::U16,
            4 => IntTy::U32,
            _ => IntTy::U64,
        };
        let end = size - size % word;
        if end > 0 {
            let value = int_const(i128::from(splat), int(word));
            let i = self.counted_loop(end, word);
            let at = self.tmp();
            self.emit(format!("{at} = getelementptr i8, ptr {ptr}, i64 {i}"));
            self.emit(format!(
                "store volatile i{} {value}, ptr {at}, align {word}",
                word * 8
            ));
            self.end_loop();
        }
        // the rest is shorter than a word and a whole number of patterns
        let mut offset = end;
        let mut width = word / 2;
        while offset < size {
            if width <= size - offset {
                let value = int_const(i128::from(splat), int(width));
                let at = self.gep("i8", ptr, &format!("i64 {offset}"));
                self.emit(format!(
                    "store volatile i{} {value}, ptr {at}, align {}",
                    width * 8,
                    width.min(align)
                ));
                offset += width;
            }
            width /= 2;
        }
    }

    /// the element and the count of `e` if it is a repeat, `[x] * n`, to
    /// write in place: handled by address, or large. A large value in a
    /// register would be one huge store, on which clang takes minutes.
    fn repeat_in_place(&self, e: ExprId) -> Option<(ExprId, u64)> {
        let body = &self.data.body;
        let Expr::Binary { lhs, .. } = body.exprs[e] else {
            return None;
        };
        let (Some(&Res::Repeat(n)), Expr::Array(elements)) =
            (self.res.res.get(e), &body.exprs[lhs])
        else {
            return None;
        };
        let ty = *self.res.expr_types.get(e)?;
        let in_place = self.by_address(ty) || self.layout.size_align(ty).0 > FILL_INLINE;
        Some((*elements.first()?, n)).filter(|_| in_place)
    }

    /// checking if `value` is a constant zero
    fn is_zero(&self, value: &V) -> bool {
        matches!(value.s.as_str(), "false" | "zeroinitializer" | "null")
            || self.constant_int(value) == Some(0)
    }

    /// writing `n` copies of `element` in the array of type `ty` at `ptr`
    ///
    /// Zeros are a `memset`, other values one store each. A large array is
    /// filled in a loop instead: zeros, a constant integer or `true` by words
    /// ([`FnGen::fill_words`]), any other value by elements.
    fn fill(&mut self, ptr: &str, ty: TyId, element: &V, n: u64) {
        if self.is_zero(element) {
            self.memset_zero(ptr, ty);
            return;
        }
        let ll = self.ll(ty);
        if self.layout.size_align(ty).0 <= FILL_INLINE {
            for i in 0..n {
                let at = self.gep(&ll, ptr, &format!("i64 0, i64 {i}"));
                self.store(element, &at);
            }
            return;
        }
        let pattern = match (self.layout.kind(element.ty), element.s.as_str()) {
            (TyKind::Bool, "true") => Some((1, 1)),
            (TyKind::Int(int), _) => self.constant_int(element).map(|v| {
                let bytes = u64::from(int.bits() / 8);
                let mask = if bytes == 8 {
                    u64::MAX
                } else {
                    (1 << (8 * bytes)) - 1
                };
                ((v as u64) & mask, bytes)
            }),
            _ => None,
        };
        if let Some((pattern, bytes)) = pattern {
            self.fill_words(ptr, ty, pattern, bytes);
            return;
        }
        // here we hide the address of each element behind an empty asm,
        // because LLVM would turn the loop into a `memset` of a byte
        let i = self.counted_loop(n, 1);
        let at = self.gep(&ll, ptr, &format!("i64 0, i64 {i}"));
        let hidden = self.tmp();
        self.emit(format!("{hidden} = call ptr asm \"\", \"=r,0\"(ptr {at})"));
        if let Some(&align) = self.aligns.get(&at) {
            self.aligns.insert(hidden.clone(), align);
        }
        self.store(element, &hidden);
        self.end_loop();
    }

    /// the widest access, up to 8 bytes, that the alignment of `ptr`, the
    /// address of a value of type `ty`, allows
    fn word_at(&self, ptr: &str, ty: TyId) -> u64 {
        if self.wide.contains(ptr) {
            8
        } else {
            self.align_at(ptr, ty).clamp(1, 8)
        }
    }

    /// starting a loop of a counter from 0 to `end`, by `step`, which
    /// [`FnGen::end_loop`] closes. It gives the counter. Both bounds are
    /// constants, so the verifier knows the counter at each turn.
    fn counted_loop(&mut self, end: u64, step: u64) -> String {
        let body = self.label("fill");
        let done = self.label("filled");
        self.br(&body);
        let before = self.block.clone();
        self.place(body.clone());
        let (i, next) = (self.tmp(), self.tmp());
        // here the block of the back edge is left blank, because the body
        // may open new blocks; `end_loop` fills it in
        self.emit(format!(
            "{i} = phi i64 [ 0, %{before} ], [ {next}, %LATCH ]"
        ));
        self.counted.push(Counted {
            phi: self.code.len() - 1,
            counter: i.clone(),
            next,
            body,
            done,
            end,
            step,
        });
        i
    }

    /// closing the innermost loop of [`FnGen::counted_loop`]
    fn end_loop(&mut self) {
        let Some(l) = self.counted.pop() else {
            return;
        };
        self.emit(format!("{} = add i64 {}, {}", l.next, l.counter, l.step));
        let latch = format!("%{}", self.block);
        self.code[l.phi] = self.code[l.phi].replace("%LATCH", &latch);
        let more = self.tmp();
        self.emit(format!("{more} = icmp ult i64 {}, {}", l.next, l.end));
        self.cond_br(&more, &l.body, &l.done);
        self.place(l.done);
    }

    /// an SSA aggregate of LLVM type `ll`, built by a chain of `insertvalue`
    fn aggregate(&mut self, ll: &str, elements: &[V]) -> String {
        if elements.is_empty() {
            return "zeroinitializer".into();
        }
        let mut current = "poison".to_string();
        for (i, element) in elements.iter().enumerate() {
            let name = self.tmp();
            let element_ll = self.ll(element.ty);
            self.emit(format!(
                "{name} = insertvalue {ll} {current}, {element_ll} {}, {i}",
                element.s
            ));
            current = name;
        }
        current
    }

    /// the address given by `getelementptr inbounds` for `indices` into the
    /// `ll` at `ptr`
    ///
    /// An address inside a less aligned one is at most as aligned. Parts
    /// of a packed struct are less aligned still, [`FnGen::part_ptr`] says
    /// how much.
    fn gep(&mut self, ll: &str, ptr: &str, indices: &str) -> String {
        let name = self.tmp();
        self.emit(format!(
            "{name} = getelementptr inbounds {ll}, ptr {ptr}, {indices}"
        ));
        if let Some(&align) = self.aligns.get(ptr) {
            self.aligns.insert(name.clone(), align);
        }
        name
    }

    /// the alignment of `ptr`, the address of a value of type `ty`: the one
    /// of the type, unless we know it is less
    fn align_at(&self, ptr: &str, ty: TyId) -> u64 {
        let align = self.layout.size_align(ty).1;
        self.aligns
            .get(ptr)
            .map_or(align, |&known| known.min(align))
    }

    /// recording that `ptr`, at `offset` from `base` (the address of a value
    /// of type `base_ty`), is only aligned to the largest power of two
    /// dividing `offset`, and no more than `base`. Only the offsets inside
    /// packed structs need it, others are multiples of the alignment of what
    /// is there.
    fn offset_align(&mut self, ptr: &str, base: &str, base_ty: TyId, offset: u64) {
        let base_align = self.align_at(base, base_ty);
        let align = if offset == 0 {
            base_align
        } else {
            base_align.min(1 << offset.trailing_zeros())
        };
        if align < 8 {
            self.aligns.insert(ptr.to_string(), align);
        }
    }

    /// the address of part `index` (a field, or the value of an option) of
    /// the aggregate of type `ty` at `ptr`
    fn part_ptr(&mut self, ptr: &str, ty: TyId, index: u32) -> String {
        let ll = self.ll(ty);
        let name = self.gep(&ll, ptr, &format!("i32 0, i32 {index}"));
        if let TyKind::Struct(s) = *self.layout.kind(ty)
            && self.layout.is_packed(s)
        {
            let offset = self.layout.part_offset(ty, index);
            self.offset_align(&name, ptr, ty, offset);
        }
        name
    }

    /// the address of a local, set up by [`FnGen::prologue`]
    fn local_ptr(&self, local: vigil_sema::infer::LocalId) -> String {
        format!("%l{}", local.into_raw().into_u32())
    }

    // statements
    // -------------------------

    fn block(&mut self, block: &Block) {
        for &stmt in block {
            self.stmt(stmt);
        }
    }

    fn stmt(&mut self, id: StmtId) {
        let span = self.data.spans.stmts[id];
        self.set_loc(span);
        let data = self.data;
        let body = &data.body;
        match &body.stmts[id] {
            Stmt::Missing | Stmt::Pass => {}
            // a comptime block runs in the compiler, what is left are the
            // values it gave its variables
            Stmt::Comptime(_) => {
                let values = self
                    .an
                    .comptime_blocks
                    .get(self.function)
                    .and_then(|blocks| blocks.iter().find(|(s, _)| *s == id))
                    .map(|(_, values)| values.clone())
                    .unwrap_or_default();
                for (local, value) in values {
                    let ty = self.res.locals[local].ty;
                    if !self.representable(ty) {
                        continue;
                    }
                    let constant = self.layout.constant(ty, &value);
                    let ll = self.ll(ty);
                    let ptr = self.local_ptr(local);
                    self.emit(format!("store {ll} {constant}, ptr {ptr}"));
                }
            }
            Stmt::Expr(e) => {
                self.expr(*e);
            }
            Stmt::Assign {
                target,
                value: Some(value),
                ..
            } if let Some((element, n)) = self.repeat_in_place(*value) => {
                // here we fill the target in place, because a temporary
                // would take as much scratch again
                let ty = self.res.expr_types[*value];
                let Some(element) = self.expr(element) else {
                    return;
                };
                let ptr = match self.res.decl_locals.get(id) {
                    Some(&local) => self.local_ptr(local),
                    None => self.place_ptr(*target).0,
                };
                self.fill(&ptr, ty, &element, n);
            }
            Stmt::Assign { target, value, .. } => {
                let Some(value) = value.and_then(|v| self.expr(v)) else {
                    return;
                };
                // a declaration initializes the variable itself, even one
                // bound to a reference. A later assignment writes through the
                // reference.
                let ptr = match self.res.decl_locals.get(id) {
                    Some(&local) => self.local_ptr(local),
                    None => self.place_ptr(*target).0,
                };
                let value = if self.swaps(*target) {
                    self.byte_swap(&value)
                } else {
                    value
                };
                let tag = if self.res.decl_locals.get(id).is_some() {
                    String::new()
                } else {
                    self.access_tag(*target)
                };
                self.store_tagged(&value, &ptr, &tag);
            }
            Stmt::AugAssign { target, value, .. } if self.res.aug_operators.get(id).is_some() => {
                let method = self.res.aug_operators[id];
                let (ptr, _) = self.place_ptr(*target);
                let current = V {
                    s: ptr.clone(),
                    ty: self.an.fn_sigs[method].params[0].1,
                };
                let Some(rhs) = self.expr(*value) else {
                    return;
                };
                if let Some(result) = self.internal_call(method, vec![current, rhs]) {
                    self.store(&result, &ptr);
                }
            }
            Stmt::AugAssign { target, op, value } => {
                let swap = self.swaps(*target);
                let (ptr, ty) = self.place_ptr(*target);
                let tag = self.access_tag(*target);
                let mut current = self.load_tagged(&ptr, ty, &tag);
                if swap {
                    current = self.byte_swap(&current);
                }
                let Some(rhs) = self.expr(*value) else {
                    return;
                };
                let mut result = self.arith(*op, current, rhs);
                if swap {
                    result = self.byte_swap(&result);
                }
                self.store_tagged(&result, &ptr, &tag);
            }
            Stmt::Return(value) => {
                let value = value.and_then(|v| self.expr(v));
                self.ret(value);
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                let Some(cond) = self.expr(*cond) else {
                    return;
                };
                let then_label = self.label("then");
                let else_label = self.label("else");
                let end = self.label("endif");
                let no = if otherwise.is_some() {
                    &else_label
                } else {
                    &end
                };
                self.cond_br(&cond.s, &then_label, no);
                self.place(then_label);
                self.block(then);
                if !self.terminated {
                    self.br(&end);
                }
                if let Some(otherwise) = otherwise {
                    self.place(else_label);
                    self.block(otherwise);
                }
                self.place(end);
            }
            Stmt::For {
                decorators,
                iter,
                body: loop_body,
                ..
            } => self.for_loop(id, *iter, loop_body, Unroll::of(decorators)),
            Stmt::While {
                decorators,
                cond,
                body: loop_body,
                ..
            } => self.while_loop(id, *cond, loop_body, Unroll::of(decorators)),
            Stmt::Match { scrutinee, arms } => self.match_stmt(*scrutinee, arms),
            Stmt::Break => {
                if let Some((_, exit)) = self.loops.last().cloned() {
                    self.br(&exit);
                }
            }
            Stmt::Continue => {
                if let Some((latch, _)) = self.loops.last().cloned() {
                    self.br(&latch);
                }
            }
        }
    }

    /// returning from the function with the value of a `return`
    fn ret(&mut self, value: Option<V>) {
        let value = self.returned(value);
        self.return_value(value);
    }

    /// what a `return` gives back: the `i64` of a program, the value of an
    /// internal function (`None` for unit)
    fn returned(&mut self, value: Option<V>) -> Option<String> {
        match &self.mode {
            Mode::Program { ret, .. } => {
                let ret = *ret;
                Some(match (value, self.layout.kind(ret).clone()) {
                    (None, _) => "0".to_string(),
                    // an action enum returns the index of its variant
                    (Some(v), TyKind::Enum(_)) => {
                        let u32_ty = self.an_int(IntTy::U32);
                        let tag = self.extract(&v, 0, u32_ty);
                        self.widen(&tag, IntTy::U32)
                    }
                    (Some(v), TyKind::Int(int)) => self.widen(&v, int),
                    // a test returns 1 when it passes
                    (Some(v), TyKind::Bool) => {
                        let name = self.tmp();
                        self.emit(format!("{name} = zext i1 {} to i64", v.s));
                        name
                    }
                    (Some(v), _) => v.s,
                })
            }
            Mode::Internal { ret } if self.layout.is_unit(*ret) => None,
            Mode::Internal { .. } => Some(value.map_or("zeroinitializer".to_string(), |v| v.s)),
        }
    }

    /// returning a value from [`Self::returned`]. A callback cannot return
    /// for the function, so it leaves the value in the frame, sets the flag
    /// and stops the loop.
    fn return_value(&mut self, value: Option<String>) {
        if self.callback {
            if let (Some(value), Some(ty)) = (value, self.ret_slot_type())
                && let Some(ptr) = self.ret_slot()
            {
                self.store(&V { s: value, ty }, &ptr);
            }
            self.set_flag(FLAG_RETURN);
            self.terminate("ret i64 1");
            return;
        }
        match &self.mode {
            Mode::Program { .. } => {
                let value = value.unwrap_or_else(|| "0".into());
                self.terminate(format!("ret i64 {value}"));
            }
            Mode::Internal { ret } => {
                let ret = *ret;
                let Some(value) = value else {
                    self.terminate("ret i1 false");
                    return;
                };
                if self.by_address(ret) {
                    self.copy("%ret", &value, ret);
                    self.terminate("ret i1 false");
                    return;
                }
                let ll = internal_ret(self.layout, ret);
                let inner = self.ll(ret);
                let a = self.tmp();
                self.emit(format!("{a} = insertvalue {ll} poison, {inner} {value}, 0"));
                let b = self.tmp();
                self.emit(format!("{b} = insertvalue {ll} {a}, i1 false, 1"));
                self.terminate(format!("ret {ll} {b}"));
            }
        }
    }

    /// an integer of type `int` widened to an `i64`, by its signedness
    fn widen(&mut self, value: &V, int: IntTy) -> String {
        if int.bits() == 64 {
            return value.s.clone();
        }
        let ext = if int.signed() { "sext" } else { "zext" };
        let name = self.tmp();
        self.emit(format!("{name} = {ext} i{} {} to i64", int.bits(), value.s));
        name
    }

    /// emitting a loop. `header` computes the condition and branches to the
    /// body or out, `body` runs in the loop, and `latch` (the `continue`
    /// target) goes to the next iteration.
    fn loop_parts(
        &mut self,
        unroll: Unroll,
        header: impl FnOnce(&mut Self, &str, &str),
        body: impl FnOnce(&mut Self),
        latch: impl FnOnce(&mut Self),
    ) {
        let header_label = self.label("loop");
        let body_label = self.label("body");
        let latch_label = self.label("next");
        let exit = self.label("endloop");
        self.place(header_label.clone());
        header(self, &body_label, &exit);
        self.place(body_label);
        self.loops.push((latch_label.clone(), exit.clone()));
        body(self);
        self.loops.pop();
        self.place(latch_label);
        latch(self);
        // with large values in scratch LLVM must not unroll either, because
        // the copies would load many of them at once and spill to the stack
        let extra = if unroll == Unroll::Always {
            format!(", !llvm.loop !{}", self.meta.unroll())
        } else if unroll == Unroll::Never || self.layout.scratch.is_some() {
            format!(", !llvm.loop !{}", self.meta.no_unroll())
        } else {
            String::new()
        };
        self.emit_md(format!("br label %{header_label}"), &extra);
        self.terminated = true;
        self.place(exit);
    }

    /// emitting a `for` over a `range` or an array: with `bpf_loop`,
    /// unrolled by us, or as a plain loop
    fn for_loop(&mut self, id: StmtId, iter: ExprId, body: &'a Block, unroll: Unroll) {
        let binding = self.res.for_locals[id];
        let binding_ptr = self.local_ptr(binding);
        let element_ty = self.res.locals[binding].ty;

        if let Expr::Call { callee, args } = &self.data.body.exprs[iter]
            && self.res.res.get(*callee) == Some(&Res::Builtin(Builtin::Range))
        {
            let int = self.int_ty(element_ty);
            let start = match args.as_slice() {
                [start, _] => self.expr(start.value),
                _ => None,
            }
            .unwrap_or(V {
                s: "0".into(),
                ty: element_ty,
            });
            let Some(end) = args.last().and_then(|a| self.expr(a.value)) else {
                return;
            };
            if self.runs_with_bpf_loop(id) {
                self.callback_loop(id, body, Iteration::Range { start, end });
                return;
            }
            if let (Some(first), Some(last)) = (self.constant_int(&start), self.constant_int(&end))
                && let Some(count) = self.unrolled_count(unroll, last - first, body)
            {
                let constant = !self.writes_local(body, binding);
                self.unrolled(count, body, |g, k| {
                    let value = V {
                        s: int_const(first + i128::from(k), int),
                        ty: element_ty,
                    };
                    g.store(&value, &binding_ptr);
                    if constant {
                        g.const_locals.insert(binding, value.s);
                    }
                });
                self.const_locals.remove(&binding);
                return;
            }
            let counter = self.spill(&start);
            let lt = if int.signed() { "slt" } else { "ult" };
            let ll = self.ll(element_ty);
            self.loop_parts(
                unroll,
                |g, body_label, exit| {
                    let value = g.load(&counter, element_ty);
                    let cond = g.tmp();
                    g.emit(format!("{cond} = icmp {lt} {ll} {}, {}", value.s, end.s));
                    g.cond_br(&cond, body_label, exit);
                },
                |g| {
                    let mut value = g.load(&counter, element_ty);
                    if unroll != Unroll::Always {
                        // here we make the variable opaque to LLVM, otherwise
                        // it replaces sums over the loop variable by a closed
                        // form with 128-bit products, which eBPF lacks
                        let sequence = g.meta.passthrough();
                        let opaque = g.tmp();
                        g.emit(format!(
                            "{opaque} = call {ll} @llvm.bpf.passthrough.{ll}.{ll}(i32 {sequence}, {ll} {})",
                            value.s
                        ));
                        value.s = opaque;
                    }
                    g.store(&value, &binding_ptr);
                    g.block(body);
                },
                |g| {
                    let value = g.load(&counter, element_ty);
                    let next = g.tmp();
                    g.emit(format!("{next} = add {ll} {}, 1", value.s));
                    g.store(
                        &V {
                            s: next,
                            ty: element_ty,
                        },
                        &counter,
                    );
                },
            );
            return;
        }

        // an array, iterated by index
        let Some(array) = self.expr(iter) else {
            return;
        };
        let TyKind::Array(_, len) = *self.layout.kind(array.ty) else {
            return;
        };
        if self.runs_with_bpf_loop(id) {
            self.callback_loop(id, body, Iteration::Array { array });
            return;
        }
        let array_ll = self.ll(array.ty);
        let array_ptr = self.spill(&array);
        if let Some(count) = self.unrolled_count(unroll, i128::from(len), body) {
            self.unrolled(count, body, |g, k| {
                let ptr = g.gep(&array_ll, &array_ptr, &format!("i64 0, i64 {k}"));
                let element = g.load(&ptr, element_ty);
                g.store(&element, &binding_ptr);
            });
            return;
        }
        let index_ty = self.an_int(IntTy::U64);
        let counter = self.spill(&V {
            s: "0".into(),
            ty: index_ty,
        });
        self.loop_parts(
            unroll,
            |g, body_label, exit| {
                let index = g.load(&counter, index_ty);
                let cond = g.tmp();
                g.emit(format!("{cond} = icmp ult i64 {}, {len}", index.s));
                g.cond_br(&cond, body_label, exit);
            },
            |g| {
                let index = g.load(&counter, index_ty);
                let ptr = g.gep(&array_ll, &array_ptr, &format!("i64 0, i64 {}", index.s));
                let element = g.load(&ptr, element_ty);
                g.store(&element, &binding_ptr);
                g.block(body);
            },
            |g| {
                let index = g.load(&counter, index_ty);
                let next = g.tmp();
                g.emit(format!("{next} = add i64 {}, 1", index.s));
                g.store(
                    &V {
                        s: next,
                        ty: index_ty,
                    },
                    &counter,
                );
            },
        );
    }

    /// the number of copies of the body of a loop of `iterations`
    /// iterations, when we unroll it ourselves ([`FnGen::unrolled`])
    ///
    /// With `@unroll`, up to [`MAX_UNROLLED`] iterations. Without it, only a
    /// short loop with a small body ([`AUTO_UNROLL_ITERATIONS`],
    /// [`AUTO_UNROLL_BUDGET`]), because then the copies are straight code
    /// the verifier walks no more than the loop.
    fn unrolled_count(&self, unroll: Unroll, iterations: i128, body: &Block) -> Option<u64> {
        if !self.layout.optimize {
            return None;
        }
        let count = u64::try_from(iterations.max(0)).ok()?;
        match unroll {
            Unroll::Always => (count <= MAX_UNROLLED).then_some(count),
            Unroll::Never => None,
            Unroll::Auto => {
                // the copies would load large values in scratch at once and
                // spill them, as in `loop_parts`
                if self.layout.scratch.is_some() || count > AUTO_UNROLL_ITERATIONS {
                    return None;
                }
                let size = self.size_of(body)?;
                (count.saturating_mul(size) <= AUTO_UNROLL_BUDGET).then_some(count)
            }
        }
    }

    /// the size of a block once its loops are unrolled, in expressions and
    /// statements. `None` when it has a comptime block or a loop that would
    /// not be unrolled (a `while`, a `bpf_loop` loop, or unknown length).
    fn size_of(&self, block: &Block) -> Option<u64> {
        let body = &self.data.body;
        let exprs = |e: ExprId| walk::subexprs(body, e).len() as u64;
        let mut total: u64 = 0;
        for &stmt in block {
            let size = match &body.stmts[stmt] {
                Stmt::Expr(e) | Stmt::Return(Some(e)) => 1 + exprs(*e),
                Stmt::Assign { target, value, .. } => 1 + exprs(*target) + value.map_or(0, exprs),
                Stmt::AugAssign { target, value, .. } => 1 + exprs(*target) + exprs(*value),
                Stmt::If {
                    cond,
                    then,
                    otherwise,
                } => {
                    1 + exprs(*cond)
                        + self.size_of(then)?
                        + otherwise.as_ref().map_or(Some(0), |b| self.size_of(b))?
                }
                Stmt::Match { scrutinee, arms } => {
                    let mut size = 1 + exprs(*scrutinee);
                    for arm in arms {
                        size += arm.guard.map_or(0, exprs) + self.size_of(&arm.body)?;
                    }
                    size
                }
                Stmt::For {
                    decorators,
                    iter,
                    body: inner,
                    ..
                } => {
                    if Unroll::of(decorators) == Unroll::Never || self.runs_with_bpf_loop(stmt) {
                        return None;
                    }
                    let iterations = self.loop_iterations(*iter)?;
                    u64::try_from(iterations.max(0))
                        .ok()?
                        .checked_mul(self.size_of(inner)?)?
                }
                Stmt::While { .. } | Stmt::Comptime(_) => return None,
                _ => 1,
            };
            total = total.checked_add(size)?;
        }
        Some(total)
    }

    /// the number of iterations of a `for` over `iter`, when it is a constant
    fn loop_iterations(&self, iter: ExprId) -> Option<i128> {
        let body = &self.data.body;
        match &body.exprs[iter] {
            Expr::Call { callee, args }
                if self.res.res.get(*callee) == Some(&Res::Builtin(Builtin::Range)) =>
            {
                let start = match args.as_slice() {
                    [start, _] => self.const_int(start.value)?,
                    _ => 0,
                };
                Some(self.const_int(args.last()?.value)? - start)
            }
            _ => match self.res.expr_types.get(iter).map(|&t| self.layout.kind(t)) {
                Some(TyKind::Array(_, len)) => Some(i128::from(*len)),
                _ => None,
            },
        }
    }

    /// emitting an unrolled loop: `count` copies of the body, each after
    /// `bind` sets the loop variable for iteration `k`. `continue` goes to
    /// the next copy, `break` past the last.
    ///
    /// We unroll here instead of in LLVM so each copy sees the variable as a
    /// constant, and array and table indices fold with no bounds check.
    fn unrolled(&mut self, count: u64, body: &'a Block, mut bind: impl FnMut(&mut Self, u64)) {
        let exit = self.label("endloop");
        for k in 0..count {
            let next = self.label("next");
            bind(self, k);
            self.loops.push((next.clone(), exit.clone()));
            self.block(body);
            self.loops.pop();
            self.place(next);
        }
        self.place(exit);
    }

    /// whether a block may assign `local`, declared outside it
    fn writes_local(&self, block: &Block, local: LocalId) -> bool {
        let body = &self.data.body;
        let mut stmts = Vec::new();
        walk::stmts(body, block, &mut stmts);
        stmts.into_iter().any(|stmt| {
            if self.res.decl_locals.get(stmt) == Some(&local)
                || self.res.for_locals.get(stmt) == Some(&local)
            {
                return true;
            }
            match &body.stmts[stmt] {
                Stmt::Assign { target, .. } | Stmt::AugAssign { target, .. } => {
                    self.res.res.get(*target) == Some(&Res::Local(local))
                }
                _ => false,
            }
        })
    }

    /// whether a loop runs with `bpf_loop`: a long loop, or one whose length
    /// is only known at run time, unless it has `@unroll`. A `while` counts
    /// by its bound, `@bounded(N)`.
    fn runs_with_bpf_loop(&self, stmt: StmtId) -> bool {
        if !self.layout.bpf_loop {
            return false;
        }
        let body = &self.data.body;
        match &body.stmts[stmt] {
            Stmt::For {
                decorators, iter, ..
            } => {
                if Unroll::of(decorators) == Unroll::Always {
                    return false;
                }
                let iterations = match &body.exprs[*iter] {
                    Expr::Call { callee, args }
                        if self.res.res.get(*callee) == Some(&Res::Builtin(Builtin::Range)) =>
                    {
                        let start = match args.as_slice() {
                            [start, _] => self.const_int(start.value),
                            _ => Some(0),
                        };
                        let end = args.last().and_then(|a| self.const_int(a.value));
                        start.zip(end).map(|(start, end)| end - start)
                    }
                    _ => match self.res.expr_types.get(*iter).map(|&t| self.layout.kind(t)) {
                        Some(TyKind::Array(_, len)) => Some(i128::from(*len)),
                        _ => Some(0),
                    },
                };
                iterations.is_none_or(|n| n >= BPF_LOOP_MIN)
            }
            Stmt::While { decorators, .. } => {
                Unroll::of(decorators) != Unroll::Always
                    && i128::from(self.res.loop_bounds.get(stmt).copied().unwrap_or(1))
                        >= BPF_LOOP_MIN
            }
            _ => false,
        }
    }

    /// the value of an integer expression known at compile time: a literal,
    /// a constant, or a sum, difference or product of them
    fn const_int(&self, e: ExprId) -> Option<i128> {
        match self.res.res.get(e) {
            Some(Res::ConstValue(v)) => return Some(*v),
            Some(Res::Const(c)) => {
                return match self.an.consts[*c].1 {
                    Some(ConstValue::Int(v)) => Some(v),
                    _ => None,
                };
            }
            _ => {}
        }
        match &self.data.body.exprs[e] {
            Expr::Int(v) => i128::try_from(*v).ok(),
            Expr::Unary {
                op: UnaryOp::Neg,
                operand,
            } => self.const_int(*operand).map(|v| -v),
            Expr::Binary { op, lhs, rhs } => {
                let (a, b) = (self.const_int(*lhs)?, self.const_int(*rhs)?);
                match op {
                    BinaryOp::Add => a.checked_add(b),
                    BinaryOp::Sub => a.checked_sub(b),
                    BinaryOp::Mul => a.checked_mul(b),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// emitting a loop as a call to `bpf_loop` with the body in a callback,
    /// then leaving the function if the body did, as the frame flag says
    fn callback_loop(&mut self, id: StmtId, body: &'a Block, iteration: Iteration) {
        let (count, start, array) = match &iteration {
            Iteration::Range { start, end } => {
                let int = self.int_ty(start.ty);
                let (a, b) = (self.widen(start, int), self.widen(end, int));
                let difference = self.tmp();
                self.emit(format!("{difference} = sub i64 {b}, {a}"));
                let more = self.tmp();
                let gt = if int.signed() { "sgt" } else { "ugt" };
                self.emit(format!("{more} = icmp {gt} i64 {b}, {a}"));
                let count = self.tmp();
                self.emit(format!(
                    "{count} = select i1 {more}, i64 {difference}, i64 0"
                ));
                (count, a, "null".to_string())
            }
            Iteration::Array { array } => {
                let TyKind::Array(_, len) = *self.layout.kind(array.ty) else {
                    return;
                };
                let ptr = self.spill(array);
                (len.to_string(), "0".into(), ptr)
            }
            Iteration::While { bound, .. } => (bound.to_string(), "0".into(), "null".into()),
        };
        let name = format!("vg_loop_{}", self.meta.unique());
        let (text, usage) = self.callback(&name, id, body, &iteration);
        self.extra.push(text);
        self.callbacks.push(usage);

        // the context lives only during the call, so the contexts of the
        // loops of a function share their stack slot
        let context = self.stack_alloca(CALLBACK_CONTEXT, 8);
        self.emit(format!(
            "call void @llvm.lifetime.start.p0(i64 40, ptr {context})"
        ));
        let scratch = self.callee_scratch();
        let locals_scratch = match (self.layout.scratch, self.callback) {
            (None, _) => "null",
            (Some(_), false) => "%sc",
            (Some(_), true) => "%lsc",
        }
        .to_string();
        let fields = [
            ("ptr", "%frame".to_string()),
            ("ptr", locals_scratch),
            ("ptr", scratch),
            ("i64", start),
            ("ptr", array),
        ];
        for (i, (ty, value)) in fields.iter().enumerate() {
            let field = self.gep(CALLBACK_CONTEXT, &context, &format!("i32 0, i32 {i}"));
            self.emit(format!("store {ty} {value}, ptr {field}, align 8"));
        }
        self.set_flag(0);
        let result = self.tmp();
        self.emit(format!(
            "{result} = call i64 inttoptr (i64 {BPF_LOOP} to ptr)(i64 {count}, ptr @\"{name}\", ptr {context}, i64 0)"
        ));
        self.emit(format!(
            "call void @llvm.lifetime.end.p0(i64 40, ptr {context})"
        ));
        // `bpf_loop` refuses more than 2^23 iterations with a negative
        // result, and then the program aborts
        let failed = self.tmp();
        self.emit(format!("{failed} = icmp slt i64 {result}, 0"));
        let ran = self.label("ran");
        self.aborts = true;
        self.cond_br(&failed, "abort", &ran);
        self.place(ran);
        let flag_ptr = self.frame_field(self.flag_field);
        let flag = self.tmp();
        self.emit(format!("{flag} = load i8, ptr {flag_ptr}, align 1"));
        let (returned, exited, after) = (
            self.label("returned"),
            self.label("exited"),
            self.label("endloop"),
        );
        self.terminate(format!(
            "switch i8 {flag}, label %{after} [ i8 {FLAG_RETURN}, label %{returned} i8 {FLAG_EXIT}, label %{exited} i8 {FLAG_ABORT}, label %abort ]"
        ));
        self.place(returned);
        let value = match self.ret_slot_type() {
            Some(ty) => self.ret_slot().map(|ptr| self.load(&ptr, ty).s),
            None => None,
        };
        self.return_value(value);
        self.place(exited);
        self.early_exit(false);
        self.place(after);
    }

    /// the text of a loop body as a `bpf_loop` callback,
    /// `i64 (i64 index, ptr context)`, which returns 1 to stop the loop, and
    /// the scratch it uses
    ///
    /// The callback has its own generator, sharing the places of the locals
    /// with the function: it reaches the frame and the scratch through its
    /// context.
    fn callback(
        &mut self,
        name: &str,
        id: StmtId,
        body: &'a Block,
        iteration: &Iteration,
    ) -> (String, ScratchUse) {
        let span = self.data.spans.stmts[id];
        let sp = self.meta.callback_subprogram(name, span);
        let loc = self.meta.location(sp, span);
        let mark = format!("@@scratch.{}@@", self.meta.unique());
        let mut g = FnGen {
            an: self.an,
            layout: self.layout,
            meta: &mut *self.meta,
            diags: &mut *self.diags,
            function: self.function,
            data: self.data,
            res: self.res,
            mode: self.mode.clone(),
            sp,
            allocas: Vec::new(),
            code: Vec::new(),
            tmp: 0,
            labels: 0,
            block: "entry".into(),
            terminated: false,
            loc,
            loops: Vec::new(),
            aborts: false,
            prologue: Vec::new(),
            slots: self.slots.clone(),
            frame: self.frame.clone(),
            flag_field: self.flag_field,
            ret_field: self.ret_field,
            ret_scratch: self.ret_scratch,
            callback: true,
            scratch_used: 0,
            aligns: HashMap::new(),
            wide: HashSet::new(),
            counted: Vec::new(),
            scratch_mark: mark,
            calls: Vec::new(),
            callbacks: Vec::new(),
            extra: Vec::new(),
            program_name: self.program_name.clone(),
            const_locals: self.const_locals.clone(),
            ranges: Rc::clone(&self.ranges),
        };
        let header =
            format!("define internal i64 @\"{name}\"(i64 %index, ptr %context) #3 !dbg !{sp} {{\n");
        let names = ["%frame", "%lsc", "%sc", "%start", "%array"];
        let types = ["ptr", "ptr", "ptr", "i64", "ptr"];
        for (i, (name, ty)) in names.iter().zip(types).enumerate() {
            g.prologue.extend([
                format!(
                    "  %context.{i} = getelementptr {CALLBACK_CONTEXT}, ptr %context, i32 0, i32 {i}"
                ),
                format!("  {name} = load {ty}, ptr %context.{i}, align 8"),
            ]);
        }
        g.local_addresses("%lsc");
        let (next, stop) = ("next".to_string(), "stop".to_string());
        g.loops.push((next.clone(), stop.clone()));
        let index_ty = g.an_int(IntTy::U64);
        let index = V {
            s: "%index".into(),
            ty: index_ty,
        };
        match iteration {
            Iteration::Range { start, .. } => {
                let binding = g.res.for_locals[id];
                let ty = g.res.locals[binding].ty;
                let value = g.tmp();
                g.emit(format!("{value} = add i64 %start, %index"));
                let value = g.convert(
                    &V {
                        s: value,
                        ty: index_ty,
                    },
                    g.int_ty(start.ty),
                );
                let value = V { s: value.s, ty };
                let ptr = g.local_ptr(binding);
                g.store(&value, &ptr);
            }
            Iteration::Array { array } => {
                let binding = g.res.for_locals[id];
                let ty = g.res.locals[binding].ty;
                // `bpf_loop` runs the callback only for indices below the
                // length of the array, so no bounds check
                let range = match *g.layout.kind(array.ty) {
                    TyKind::Array(_, len) => Some(Range {
                        lo: 0,
                        hi: i128::from(len) - 1,
                    }),
                    _ => None,
                }
                .filter(|_| g.layout.optimize);
                let element = g.element_ptr("%array", array.ty, &index, range);
                let value = g.load(&element, ty);
                let ptr = g.local_ptr(binding);
                g.store(&value, &ptr);
            }
            Iteration::While { cond, .. } => {
                if let Some(value) = g.expr(*cond) {
                    let go = g.label("body");
                    g.cond_br(&value.s, &go, &stop);
                    g.place(go);
                }
            }
        }
        g.block(body);
        g.place(next);
        g.terminate("ret i64 0");
        g.place(stop);
        g.terminate("ret i64 1");
        if g.aborts {
            g.place("abort".into());
            g.early_exit(true);
        }
        g.finish(header)
    }

    /// the type of the integers `int`
    fn an_int(&self, int: IntTy) -> TyId {
        // cannot fail, every integer type is interned when the interner is
        // created
        self.layout
            .analysis
            .types
            .lookup(&TyKind::Int(int))
            .expect("integer types are interned")
    }

    /// emitting a `while`: with `bpf_loop`, or as a plain loop that runs at
    /// most its bound times
    fn while_loop(&mut self, id: StmtId, cond: ExprId, body: &'a Block, unroll: Unroll) {
        let bound = self.res.loop_bounds.get(id).copied().unwrap_or(1);
        if self.runs_with_bpf_loop(id) {
            self.callback_loop(id, body, Iteration::While { cond, bound });
            return;
        }
        let index_ty = self.an_int(IntTy::U64);
        let counter = self.spill(&V {
            s: "0".into(),
            ty: index_ty,
        });
        self.loop_parts(
            unroll,
            |g, body_label, exit| {
                // `@bounded(N)`: at most N iterations. The count goes through
                // an empty asm, because reasoning on it and the condition
                // together, LLVM left branches to loops without exit
                // (`goto -1`) that the verifier must prove dead.
                // `llvm.bpf.passthrough` is not enough, the backend sees
                // through it.
                let count = g.load(&counter, index_ty);
                let opaque = g.tmp();
                g.emit(format!(
                    "{opaque} = call i64 asm \"\", \"=r,0\"(i64 {})",
                    count.s
                ));
                let more = g.tmp();
                g.emit(format!("{more} = icmp ult i64 {opaque}, {bound}"));
                let check = g.label("cond");
                g.cond_br(&more, &check, exit);
                g.place(check);
                if let Some(value) = g.expr(cond) {
                    g.cond_br(&value.s, body_label, exit);
                }
            },
            |g| g.block(body),
            |g| {
                let count = g.load(&counter, index_ty);
                let next = g.tmp();
                g.emit(format!("{next} = add i64 {}, 1", count.s));
                g.store(
                    &V {
                        s: next,
                        ty: index_ty,
                    },
                    &counter,
                );
            },
        );
    }

    /// emitting a `match`: each arm in turn, its pattern then its guard, and
    /// on a failure the next arm
    fn match_stmt(&mut self, scrutinee: ExprId, arms: &'a [MatchArm]) {
        let Some(value) = self.expr(scrutinee) else {
            return;
        };
        let ptr = self.spill(&value);
        let end = self.label("endmatch");
        for arm in arms {
            let next = self.label("arm");
            self.pattern(arm.pat, &ptr, value.ty, &next);
            if let Some(guard) = arm.guard
                && let Some(cond) = self.expr(guard)
            {
                let body = self.label("case");
                self.cond_br(&cond.s, &body, &next);
                self.place(body);
            }
            self.block(&arm.body);
            if !self.terminated {
                self.br(&end);
            }
            self.place(next);
        }
        // not reachable, the type checker ensures the arms are exhaustive
        self.terminate("unreachable");
        self.place(end);
    }

    /// testing the value at `ptr` against a pattern: it branches to `fail`
    /// if it does not match, and binds the names of the pattern otherwise
    fn pattern(&mut self, pat: PatId, ptr: &str, ty: TyId, fail: &str) {
        let data = self.data;
        let body = &data.body;
        match &body.pats[pat] {
            Pat::Missing | Pat::Wild | Pat::Str | Pat::Float => {}
            Pat::Bind(_) => {
                let value = self.load(ptr, ty);
                let local = self.local_ptr(self.res.pat_locals[pat]);
                self.store(&value, &local);
            }
            Pat::Int { value, negative } => {
                let int = self.int_ty(ty);
                let value = i128::try_from(*value).unwrap_or(i128::MAX);
                let value = if *negative { -value } else { value };
                let loaded = self.load(ptr, ty);
                self.test_eq(&loaded, &int_const(value, int), fail);
            }
            Pat::Bool(b) => {
                let loaded = self.load(ptr, ty);
                self.test_eq(&loaded, &b.to_string(), fail);
            }
            Pat::None => {
                let tag = self.option_tag(ptr, ty);
                self.test_eq(&tag, "false", fail);
            }
            Pat::Path(_) => match self.res.pat_res.get(pat).copied() {
                Some(Res::Variant(e, i)) => {
                    let tag = self.enum_tag(ptr, e);
                    self.test_eq(&tag, &i.to_string(), fail);
                }
                Some(Res::Const(c)) => {
                    let loaded = self.load(ptr, ty);
                    let constant = self.const_value(c);
                    self.test_eq(&loaded, &constant.s, fail);
                }
                _ => {}
            },
            Pat::TupleStruct { fields, .. } => {
                let fields = fields.clone();
                match self.res.pat_res.get(pat).copied() {
                    Some(Res::Builtin(Builtin::Some)) => {
                        let tag = self.option_tag(ptr, ty);
                        self.test_eq(&tag, "true", fail);
                        let TyKind::Option(inner) = *self.layout.kind(ty) else {
                            return;
                        };
                        let payload = if self.layout.is_niche(ty) {
                            ptr.to_string()
                        } else {
                            let ll = self.ll(ty);
                            self.gep(&ll, ptr, "i32 0, i32 1")
                        };
                        if let [field] = fields.as_slice() {
                            self.pattern(*field, &payload, inner, fail);
                        }
                    }
                    Some(Res::Variant(e, i)) => {
                        let tag = self.enum_tag(ptr, e);
                        self.test_eq(&tag, &i.to_string(), fail);
                        let Some(payload_ll) = self.layout.payload_ty(e, i as usize) else {
                            return;
                        };
                        let types = self.an.variant_fields[e][i as usize]
                            .clone()
                            .unwrap_or_default();
                        let enum_ll = format!("%e.{}", e.into_raw());
                        let payload = self.gep(&enum_ll, ptr, "i32 0, i32 1");
                        for (j, (&field, &field_ty)) in fields.iter().zip(&types).enumerate() {
                            let field_ptr =
                                self.gep(&payload_ll, &payload, &format!("i32 0, i32 {j}"));
                            self.pattern(field, &field_ptr, field_ty, fail);
                        }
                    }
                    _ => {}
                }
            }
            Pat::Or(alternatives) => {
                let alternatives = alternatives.clone();
                let matched = self.label("matched");
                for (k, &alternative) in alternatives.iter().enumerate() {
                    let last = k + 1 == alternatives.len();
                    let next = if last {
                        fail.to_string()
                    } else {
                        self.label("alt")
                    };
                    self.pattern(alternative, ptr, ty, &next);
                    self.br(&matched);
                    if !last {
                        self.place(next);
                    }
                }
                self.place(matched);
            }
        }
    }

    /// branching to `fail` unless a value equals a constant
    fn test_eq(&mut self, value: &V, constant: &str, fail: &str) {
        let ll = self.ll(value.ty);
        let cond = self.tmp();
        self.emit(format!("{cond} = icmp eq {ll} {}, {constant}", value.s));
        let ok = self.label("ok");
        self.cond_br(&cond, &ok, fail);
        self.place(ok);
    }

    /// whether the option at `ptr` is `Some`. An option of a reference is a
    /// pointer, null for `None`; the others have a `bool` tag first.
    fn option_tag(&mut self, ptr: &str, ty: TyId) -> V {
        if self.layout.is_niche(ty) {
            let value = self.load(ptr, ty);
            let name = self.tmp();
            self.emit(format!("{name} = icmp ne ptr {}, null", value.s));
            return V {
                s: name,
                ty: self.an.types.bool,
            };
        }
        let ll = self.ll(ty);
        let tag_ptr = self.gep(&ll, ptr, "i32 0, i32 0");
        let bool_ty = self.an.types.bool;
        self.load(&tag_ptr, bool_ty)
    }

    /// the variant index of the enum at `ptr`, its first field
    fn enum_tag(&mut self, ptr: &str, e: EnumId) -> V {
        let ll = format!("%e.{}", e.into_raw());
        let tag_ptr = self.gep(&ll, ptr, "i32 0, i32 0");
        let u32_ty = self.an_int(IntTy::U32);
        self.load(&tag_ptr, u32_ty)
    }

    // places
    // -------------------------

    /// whether `e` designates memory of the function: a local, a table, or a
    /// field or element of one
    fn is_place(&self, e: ExprId) -> bool {
        match &self.data.body.exprs[e] {
            Expr::Name(_) => match self.res.res.get(e) {
                Some(Res::Local(_)) => true,
                Some(Res::Const(c)) => self.table(*c).is_some(),
                _ => false,
            },
            Expr::Field { base, .. } => {
                matches!(self.res.res.get(e), Some(Res::Field(_))) && self.is_place(*base)
            }
            Expr::Index { base, .. } => self.is_place(*base),
            _ => false,
        }
    }

    /// the address of the base of a field or element, and its type: a place,
    /// or the value of a reference (`ctx.parse[H](0)?`)
    fn base_ptr(&mut self, base: ExprId) -> (String, TyId) {
        if !self.is_place(base)
            && let TyKind::Ref(inner) = *self.layout.kind(self.res.expr_types[base])
            && let Some(pointer) = self.expr_inner(base)
        {
            return (pointer.s, inner);
        }
        self.place_ptr(base)
    }

    /// the address of an assignable expression, and its type
    fn place_ptr(&mut self, e: ExprId) -> (String, TyId) {
        let ty = self.res.expr_types[e];
        match &self.data.body.exprs[e] {
            Expr::Name(_) => match self.res.res.get(e).copied() {
                Some(Res::Local(local)) => {
                    let local_ty = self.res.locals[local].ty;
                    let ptr = self.local_ptr(local);
                    // a variable bound to a reference designates the
                    // referenced memory
                    match *self.layout.kind(local_ty) {
                        TyKind::Ref(inner) => (self.load(&ptr, local_ty).s, inner),
                        _ => (ptr, local_ty),
                    }
                }
                Some(Res::Const(c)) if let Some(table) = self.table(c) => (table, ty),
                _ => (self.spill_poison(ty), ty),
            },
            Expr::Field { base, .. } => {
                let base = *base;
                let (base_ptr, base_ty) = self.base_ptr(base);
                let Some(Res::Field(i)) = self.res.res.get(e).copied() else {
                    return (self.spill_poison(ty), ty);
                };
                (self.part_ptr(&base_ptr, base_ty, i), ty)
            }
            Expr::Index { base, indices } => {
                let (base, index) = (*base, indices[0]);
                let (base_ptr, base_ty) = self.base_ptr(base);
                let range = self.range_of(index);
                let Some(index) = self.expr(index) else {
                    return (self.spill_poison(ty), ty);
                };
                (self.element_ptr(&base_ptr, base_ty, &index, range), ty)
            }
            _ => (self.spill_poison(ty), ty),
        }
    }

    /// fresh memory, for an expression that is not a place
    fn spill_poison(&mut self, ty: TyId) -> String {
        self.alloca(ty)
    }

    /// the range of the values of an integer expression, when we optimize
    /// ([`crate::range`])
    fn range_of(&self, e: ExprId) -> Option<Range> {
        if !self.layout.optimize {
            return None;
        }
        Eval {
            an: self.an,
            body: &self.data.body,
            res: self.res,
            ranges: &self.ranges,
        }
        .expr(e)
    }

    /// the address of element `index` of the array at `ptr`. The program
    /// aborts if the index is out of bounds; we leave out the check when
    /// `range`, the range of the index, shows it is inside the array.
    fn element_ptr(
        &mut self,
        ptr: &str,
        array_ty: TyId,
        index: &V,
        range: Option<Range>,
    ) -> String {
        let TyKind::Array(_, len) = *self.layout.kind(array_ty) else {
            return ptr.to_string();
        };
        let int = self.int_ty(index.ty);
        let wide = if int.bits() == 64 {
            index.s.clone()
        } else {
            // widened by its signedness, a negative index is huge as an
            // unsigned number, so one unsigned comparison checks both bounds
            let name = self.tmp();
            let ext = if int.signed() { "sext" } else { "zext" };
            self.emit(format!("{name} = {ext} i{} {} to i64", int.bits(), index.s));
            name
        };
        let proven = range.is_some_and(|r| r.lo >= 0 && r.hi < i128::from(len));
        if !proven {
            let in_bounds = self.tmp();
            self.emit(format!("{in_bounds} = icmp ult i64 {wide}, {len}"));
            let ok = self.label("inbounds");
            self.aborts = true;
            self.cond_br(&in_bounds, &ok, "abort");
            self.place(ok);
        }
        let ll = self.ll(array_ty);
        // a constant index needs no more, the address is a constant offset
        if index.s.parse::<i128>().is_ok() {
            return self.gep(&ll, ptr, &format!("i64 0, i64 {wide}"));
        }
        // the verifier must know the bound of the register added to the
        // pointer. The check above may be on another register (a narrower
        // copy, or one LLVM reloads), and the range analysis may rely on
        // things the verifier does not follow, like a `clamp` without
        // branches. So we bound the index right before the address. The
        // empty asm keeps LLVM from removing the bound, which it may know is
        // a no-op, and is opaque to instruction selection too.
        let opaque = self.tmp();
        let effects = if self.layout.optimize { " #4" } else { "" };
        self.emit(format!(
            "{opaque} = call i64 asm \"\", \"=r,0\"(i64 {wide}){effects}"
        ));
        let last = len.saturating_sub(1);
        let safe = self.tmp();
        if proven && len.is_power_of_two() {
            // inside the bounds and a power of two: masking is a no-op, and
            // one instruction
            self.emit(format!("{safe} = and i64 {opaque}, {last}"));
        } else {
            let beyond = self.tmp();
            self.emit(format!("{beyond} = icmp ugt i64 {opaque}, {last}"));
            self.emit(format!(
                "{safe} = select i1 {beyond}, i64 {last}, i64 {opaque}"
            ));
        }
        // the bounded index is small and the stack array aligned, so
        // instruction selection would add it with an `or`, which the
        // verifier rejects on pointers. Behind another empty asm LLVM does
        // not know its bits, but the verifier does.
        let hidden = self.tmp();
        self.emit(format!(
            "{hidden} = call i64 asm \"\", \"=r,0\"(i64 {safe}){effects}"
        ));
        self.gep(&ll, ptr, &format!("i64 0, i64 {hidden}"))
    }

    // expressions
    // -------------------------

    /// the value of an expression; a reference is read when the type checker
    /// marked its value as used
    fn expr(&mut self, e: ExprId) -> Option<V> {
        let value = self.expr_inner(e)?;
        if self.res.derefs.contains(&e)
            && let TyKind::Ref(inner) = *self.layout.kind(value.ty)
        {
            return Some(self.load(&value.s, inner));
        }
        Some(value)
    }

    /// the value of an expression, leaving a reference as an address. `None`
    /// when it has no value at run time: unit, a string, a float, `...`, or
    /// an expression missing from the source.
    fn expr_inner(&mut self, e: ExprId) -> Option<V> {
        let ty = self
            .res
            .expr_types
            .get(e)
            .copied()
            .unwrap_or(self.an.types.error);
        let data = self.data;
        let body = &data.body;
        let value = match &body.exprs[e] {
            Expr::Int(value) => V {
                s: int_const(i128::try_from(*value).unwrap_or(i128::MAX), self.int_ty(ty)),
                ty,
            },
            Expr::Bool(b) => V {
                s: b.to_string(),
                ty,
            },
            Expr::None => self.zero(ty),
            Expr::Field { base, .. }
                if matches!(self.res.res.get(e), Some(Res::KernelField(_))) =>
            {
                let Some(Res::KernelField(index)) = self.res.res.get(e).copied() else {
                    return None;
                };
                // a field of an embedded struct, like `cred.uid.val`, is one
                // access from the outermost struct, as clang emits it. An
                // embedded struct may be anonymous, and loaders find the
                // relocation types by name.
                let mut base = *base;
                let mut access = self.res.kernel_accesses[index as usize].clone();
                while let Expr::Field { base: inner, .. } = body.exprs[base]
                    && let Some(Res::KernelField(i)) = self.res.res.get(base).copied()
                    && let outer = &self.res.kernel_accesses[i as usize]
                    && matches!(outer.read, KernelRead::Address)
                {
                    access = KernelAccess {
                        composite: outer.composite,
                        path: outer.path.iter().chain(&access.path).copied().collect(),
                        offset: outer.offset + access.offset,
                        read: access.read,
                    };
                    base = inner;
                }
                let base = self.expr(base)?;
                self.kernel_read(&base, &access, ty)
            }
            Expr::Name(_) | Expr::Field { .. } => match self.res.res.get(e).copied() {
                Some(Res::Local(local)) => {
                    if let Some(constant) = self.const_locals.get(&local) {
                        return Some(V {
                            s: constant.clone(),
                            ty,
                        });
                    }
                    let ptr = self.local_ptr(local);
                    self.load(&ptr, ty)
                }
                Some(Res::Const(c)) => self.const_value(c),
                Some(Res::ConstValue(v)) => V {
                    s: int_const(v, self.int_ty(ty)),
                    ty,
                },
                Some(Res::Variant(en, i)) => self.enum_value(en, i, &[]),
                Some(Res::Field(i)) => {
                    let swap = self.swaps(e);
                    let value = if self.is_place(e) {
                        let (ptr, _) = self.place_ptr(e);
                        let tag = self.access_tag(e);
                        self.load_tagged(&ptr, ty, &tag)
                    } else {
                        let Expr::Field { base, .. } = body.exprs[e] else {
                            return None;
                        };
                        let base = self.expr(base)?;
                        match *self.layout.kind(base.ty) {
                            // a field of a referenced struct, read in place
                            TyKind::Ref(inner) => {
                                let ptr = self.part_ptr(&base.s, inner, i);
                                self.load(&ptr, ty)
                            }
                            _ => self.extract(&base, i, ty),
                        }
                    };
                    if swap { self.byte_swap(&value) } else { value }
                }
                _ => return None,
            },
            Expr::Call { callee, args } => {
                let span = self.data.spans.exprs[e];
                let saved = self.loc;
                self.set_loc(span);
                let value = self.call(e, *callee, args);
                self.loc = saved;
                return value;
            }
            Expr::Index { base, indices } => {
                let (base, index) = (*base, indices[0]);
                if let Some(Res::Map(m)) = self.res.res.get(base).copied() {
                    return self.map_index(m, index, ty);
                }
                if self.table_base(base).is_some() {
                    // a table has no effects, so we can evaluate the index
                    // first, and an element at a constant index becomes an
                    // immediate
                    let range = self.range_of(index);
                    let index = self.expr(index)?;
                    if let Some(element) = self.table_element(base, &index) {
                        return Some(element);
                    }
                    let (ptr, base_ty) = self.place_ptr(base);
                    let element = self.element_ptr(&ptr, base_ty, &index, range);
                    return Some(self.load(&element, ty));
                }
                let (ptr, base_ty) = if self.is_place(base) {
                    self.place_ptr(base)
                } else {
                    let value = self.expr(base)?;
                    match *self.layout.kind(value.ty) {
                        TyKind::Ref(inner) => (value.s, inner),
                        _ => (self.spill(&value), value.ty),
                    }
                };
                let range = self.range_of(index);
                let index = self.expr(index)?;
                let element = self.element_ptr(&ptr, base_ty, &index, range);
                let tag = self.access_tag(e);
                self.load_tagged(&element, ty, &tag)
            }
            Expr::Unary { operand, .. } | Expr::Binary { lhs: operand, .. }
                if matches!(self.res.res.get(e), Some(Res::Operator(_))) =>
            {
                let Some(Res::Operator(method)) = self.res.res.get(e).copied() else {
                    return None;
                };
                let (receiver, back) = self.receiver(method, *operand)?;
                let mut values = vec![receiver];
                if let Expr::Binary { rhs, .. } = body.exprs[e] {
                    values.push(self.expr(rhs)?);
                }
                return self.method_call(method, values, back);
            }
            Expr::Binary { lhs, .. } if matches!(self.res.res.get(e), Some(Res::Repeat(_))) => {
                let (Some(&Res::Repeat(n)), Expr::Array(elements)) =
                    (self.res.res.get(e), &body.exprs[*lhs])
                else {
                    return None;
                };
                let element = self.expr(elements[0])?;
                if self.is_zero(&element) {
                    self.zero(ty)
                } else if self.by_address(ty) {
                    let ptr = self.alloca(ty);
                    self.fill(&ptr, ty, &element, n);
                    V { s: ptr, ty }
                } else {
                    self.aggregate_value(ty, &vec![element; n as usize])
                }
            }
            Expr::Unary { op, operand } => {
                let v = self.expr(*operand)?;
                let ll = self.ll(ty);
                let name = self.tmp();
                match op {
                    UnaryOp::Not => self.emit(format!("{name} = xor i1 {}, true", v.s)),
                    UnaryOp::Neg => self.emit(format!("{name} = sub {ll} 0, {}", v.s)),
                    UnaryOp::BitNot => self.emit(format!("{name} = xor {ll} {}, -1", v.s)),
                }
                V { s: name, ty }
            }
            Expr::Binary { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                match op {
                    BinaryOp::And | BinaryOp::Or => return self.short_circuit(op, lhs, rhs),
                    BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge => {
                        let a = self.expr(lhs)?;
                        let b = self.expr(rhs)?;
                        self.compare(op, &a, &b, ty)
                    }
                    _ => {
                        let a = self.expr(lhs)?;
                        let b = self.expr(rhs)?;
                        self.arith(op, a, b)
                    }
                }
            }
            Expr::Try(inner) => {
                // `ctx.parse[H](offset)?` branches straight to the early
                // exit. No null pointer is built, so the verifier sees only
                // the bounds checks.
                if let Expr::Call { callee, args } = &body.exprs[*inner]
                    && let Some(Res::PacketParse(header)) = self.res.res.get(*callee).copied()
                    && let Expr::Index { base, .. } = body.exprs[*callee]
                    && let Expr::Field { base: receiver, .. } = body.exprs[base]
                {
                    let context = self.expr(receiver)?;
                    let offset = self.expr(args.first()?.value)?;
                    let pointer = self.packet_checked(&context, header, &offset);
                    return Some(V { s: pointer, ty });
                }
                let option = self.expr(*inner)?;
                let niche = self.layout.is_niche(option.ty);
                let tag = if niche {
                    let name = self.tmp();
                    self.emit(format!("{name} = icmp ne ptr {}, null", option.s));
                    V {
                        s: name,
                        ty: self.an.types.bool,
                    }
                } else {
                    self.extract(&option, 0, self.an.types.bool)
                };
                let some = self.label("some");
                let none = self.label("none");
                self.cond_br(&tag.s, &some, &none);
                self.place(none);
                self.early_exit(false);
                self.place(some);
                if niche {
                    V { s: option.s, ty }
                } else {
                    self.extract(&option, 1, ty)
                }
            }
            Expr::Array(elements) => {
                let mut values = Vec::new();
                for &element in elements {
                    values.push(self.expr(element)?);
                }
                self.aggregate_value(ty, &values)
            }
            Expr::Missing | Expr::Float(_) | Expr::Str(_) | Expr::FStr(_) | Expr::Ellipsis => {
                return None;
            }
        };
        Some(value)
    }

    /// the constant of a table computed by a comptime function, when `base`
    /// names one
    fn table_base(&self, base: ExprId) -> Option<ConstId> {
        if !matches!(self.data.body.exprs[base], Expr::Name(_)) {
            return None;
        }
        let Some(Res::Const(c)) = self.res.res.get(base).copied() else {
            return None;
        };
        self.table(c).map(|_| c)
    }

    /// `TABLE[i]` at an index that is a constant here (after unrolling or
    /// folding): the element itself as an immediate, instead of a load from
    /// the global
    fn table_element(&mut self, base: ExprId, index: &V) -> Option<V> {
        let c = self.table_base(base)?;
        let ty = self.an.consts[c].0;
        let TyKind::Array(element_ty, _) = *self.layout.kind(ty) else {
            return None;
        };
        let Some(CValue::Array(elements)) = self.an.comptime_consts.get(c) else {
            return None;
        };
        if !matches!(self.layout.kind(element_ty), TyKind::Int(_) | TyKind::Bool) {
            return None;
        }
        let at = self.folded(index)?;
        let element = elements.get(usize::try_from(at).ok()?)?;
        Some(V {
            s: self.layout.constant(element_ty, element),
            ty: element_ty,
        })
    }

    /// the value of an integer known at compile time, from its LLVM constant
    /// (wrapped to the type and written signed by `int_const`)
    fn constant_int(&self, value: &V) -> Option<i128> {
        let raw = value.s.parse::<i128>().ok()?;
        let int = self.int_ty(value.ty);
        Some(if !int.signed() && raw < 0 {
            raw + (1i128 << int.bits())
        } else {
            raw
        })
    }

    /// the global of a table computed by a comptime function
    fn table(&self, c: ConstId) -> Option<String> {
        let ty = self.an.consts[c].0;
        let computed = self.an.comptime_consts.get(c).is_some();
        (computed && matches!(self.layout.kind(ty), TyKind::Array(..)))
            .then(|| format!("@\"vigil.const.{}\"", c.into_raw()))
    }

    /// the value of a constant: a load from its global for a table, an LLVM
    /// constant otherwise
    fn const_value(&mut self, c: ConstId) -> V {
        let (ty, value) = self.an.consts[c];
        if let Some(table) = self.table(c) {
            return self.load(&table, ty);
        }
        if let Some(computed) = self.an.comptime_consts.get(c) {
            let s = self.layout.constant(ty, computed);
            if self.by_address(ty) {
                let ptr = self.alloca(ty);
                let ll = self.ll(ty);
                self.emit(format!("store {ll} {s}, ptr {ptr}"));
                return V { s: ptr, ty };
            }
            return V { s, ty };
        }
        let s = match value {
            Some(ConstValue::Int(v)) => int_const(v, self.int_ty(ty)),
            Some(ConstValue::Bool(b)) => b.to_string(),
            None => "zeroinitializer".into(),
        };
        V { s, ty }
    }

    /// a value of an enum: the variant index, then the payload
    fn enum_value(&mut self, e: EnumId, variant: u32, payload: &[V]) -> V {
        let ty = self
            .an
            .types
            .lookup(&TyKind::Enum(e))
            .expect("enum types are interned");
        let ll = format!("%e.{}", e.into_raw());
        let ptr = self.alloca(ty);
        let tag_ptr = self.gep(&ll, &ptr, "i32 0, i32 0");
        self.emit(format!("store i32 {variant}, ptr {tag_ptr}, align 4"));
        if let Some(payload_ll) = self.layout.payload_ty(e, variant as usize)
            && !payload.is_empty()
        {
            let payload_ptr = self.gep(&ll, &ptr, "i32 0, i32 1");
            for (i, value) in payload.iter().enumerate() {
                let field = self.gep(&payload_ll, &payload_ptr, &format!("i32 0, i32 {i}"));
                self.store(value, &field);
            }
        }
        self.load(&ptr, ty)
    }

    /// `a and b` or `a or b`, evaluating `b` only when `a` does not decide
    fn short_circuit(&mut self, op: BinaryOp, lhs: ExprId, rhs: ExprId) -> Option<V> {
        let a = self.expr(lhs)?;
        let rhs_label = self.label("rhs");
        let end = self.label("sc");
        // the predecessors of `end` are read after branching, because an
        // operand that ends its block, as `?` does, moves the branch to a
        // new block
        if op == BinaryOp::And {
            self.cond_br(&a.s, &rhs_label, &end);
        } else {
            self.cond_br(&a.s, &end, &rhs_label);
        }
        let lhs_block = self.block.clone();
        self.place(rhs_label);
        let b = self.expr(rhs)?;
        self.br(&end);
        let rhs_block = self.block.clone();
        self.place(end);
        let short = op == BinaryOp::Or;
        let name = self.tmp();
        self.emit(format!(
            "{name} = phi i1 [ {short}, %{lhs_block} ], [ {}, %{rhs_block} ]",
            b.s
        ));
        Some(V {
            s: name,
            ty: self.an.types.bool,
        })
    }

    /// a comparison, signed or unsigned by the type of `a`
    fn compare(&mut self, op: BinaryOp, a: &V, b: &V, bool_ty: TyId) -> V {
        let (a, b) = match self.layout.kind(a.ty) {
            // variants without fields are equal when their indices are
            TyKind::Enum(_) => {
                let u32_ty = self.an_int(IntTy::U32);
                (self.extract(a, 0, u32_ty), self.extract(b, 0, u32_ty))
            }
            _ => (a.clone(), b.clone()),
        };
        let signed = matches!(self.layout.kind(a.ty), TyKind::Int(int) if int.signed());
        let predicate = match (op, signed) {
            (BinaryOp::Eq, _) => "eq",
            (BinaryOp::Ne, _) => "ne",
            (BinaryOp::Lt, true) => "slt",
            (BinaryOp::Lt, false) => "ult",
            (BinaryOp::Le, true) => "sle",
            (BinaryOp::Le, false) => "ule",
            (BinaryOp::Gt, true) => "sgt",
            (BinaryOp::Gt, false) => "ugt",
            (BinaryOp::Ge, true) => "sge",
            _ => "uge",
        };
        let ll = self.ll(a.ty);
        let name = self.tmp();
        self.emit(format!("{name} = icmp {predicate} {ll} {}, {}", a.s, b.s));
        V {
            s: name,
            ty: bool_ty,
        }
    }

    /// an arithmetic or bitwise operation on integers of the type of `a`, as
    /// the eBPF instructions do it (see the module header)
    fn arith(&mut self, op: BinaryOp, a: V, b: V) -> V {
        let ty = a.ty;
        let int = self.int_ty(ty);
        let ll = self.ll(ty);
        if let Some(value) = self.fold_arith(op, &a, &b) {
            return V {
                s: int_const(value, int),
                ty,
            };
        }
        let simple = match op {
            BinaryOp::Add => Some("add"),
            BinaryOp::Sub => Some("sub"),
            BinaryOp::Mul => Some("mul"),
            BinaryOp::BitAnd => Some("and"),
            BinaryOp::BitOr => Some("or"),
            BinaryOp::BitXor => Some("xor"),
            _ => None,
        };
        if let Some(instr) = simple {
            let name = self.tmp();
            self.emit(format!("{name} = {instr} {ll} {}, {}", a.s, b.s));
            return V { s: name, ty };
        }
        match op {
            BinaryOp::Shl | BinaryOp::Shr => {
                let amount = self.convert(&b, int);
                let masked = self.tmp();
                self.emit(format!(
                    "{masked} = and {ll} {}, {}",
                    amount.s,
                    int.bits() - 1
                ));
                let instr = match (op, int.signed()) {
                    (BinaryOp::Shl, _) => "shl",
                    (_, true) => "ashr",
                    (_, false) => "lshr",
                };
                let name = self.tmp();
                self.emit(format!("{name} = {instr} {ll} {}, {masked}", a.s));
                V { s: name, ty }
            }
            BinaryOp::FloorDiv | BinaryOp::Rem => {
                let rem = op == BinaryOp::Rem;
                if !int.signed() {
                    return self.unsigned_div(&a, &b, rem);
                }
                // signed: we divide the magnitudes and fix the sign, because
                // eBPF has signed division only from kernel 6.6
                let abs = |g: &mut Self, v: &V| -> (String, V) {
                    let negative = g.tmp();
                    g.emit(format!("{negative} = icmp slt {ll} {}, 0", v.s));
                    let negated = g.tmp();
                    g.emit(format!("{negated} = sub {ll} 0, {}", v.s));
                    let magnitude = g.tmp();
                    g.emit(format!(
                        "{magnitude} = select i1 {negative}, {ll} {negated}, {ll} {}",
                        v.s
                    ));
                    (negative, V { s: magnitude, ty })
                };
                let (a_negative, a_abs) = abs(self, &a);
                let (b_negative, b_abs) = abs(self, &b);
                let magnitude = self.unsigned_div(&a_abs, &b_abs, rem);
                let negative = if rem {
                    a_negative
                } else {
                    let name = self.tmp();
                    self.emit(format!("{name} = xor i1 {a_negative}, {b_negative}"));
                    name
                };
                let negated = self.tmp();
                self.emit(format!("{negated} = sub {ll} 0, {}", magnitude.s));
                let name = self.tmp();
                self.emit(format!(
                    "{name} = select i1 {negative}, {ll} {negated}, {ll} {}",
                    magnitude.s
                ));
                V { s: name, ty }
            }
            // `/` cannot be here, the type checker rejects it on integers
            _ => a,
        }
    }

    /// `a op b` when both are constants, as the instructions compute it
    /// ([`FnGen::arith`]): wrapping, shift amounts modulo the width. Division
    /// is left to LLVM. The result must still be wrapped to its type by
    /// `int_const`.
    fn fold_arith(&self, op: BinaryOp, a: &V, b: &V) -> Option<i128> {
        let (x, y) = (self.folded(a)?, self.folded(b)?);
        let int = self.int_ty(a.ty);
        Some(match op {
            BinaryOp::Add => x.wrapping_add(y),
            BinaryOp::Sub => x.wrapping_sub(y),
            BinaryOp::Mul => x.wrapping_mul(y),
            BinaryOp::BitAnd => x & y,
            BinaryOp::BitOr => x | y,
            BinaryOp::BitXor => x ^ y,
            BinaryOp::Shl | BinaryOp::Shr => {
                let amount = (y & i128::from(int.bits() - 1)) as u32;
                if op == BinaryOp::Shl {
                    x << amount
                } else {
                    // `x` is the value of its type, so this is an arithmetic
                    // shift for signed types and a logical one for unsigned
                    // ones, which are non-negative
                    x >> amount
                }
            }
            _ => return None,
        })
    }

    /// the value of a constant integer, when we fold constants
    /// (`Options::optimize`)
    fn folded(&self, value: &V) -> Option<i128> {
        if !self.layout.optimize || !matches!(self.layout.kind(value.ty), TyKind::Int(_)) {
            return None;
        }
        self.constant_int(value)
    }

    /// `a // b` or `a % b` on unsigned values, with the eBPF results for a
    /// zero divisor: 0 for the quotient, `a` for the remainder
    fn unsigned_div(&mut self, a: &V, b: &V, rem: bool) -> V {
        let ll = self.ll(a.ty);
        let zero = self.tmp();
        self.emit(format!("{zero} = icmp eq {ll} {}, 0", b.s));
        let divisor = self.tmp();
        self.emit(format!(
            "{divisor} = select i1 {zero}, {ll} 1, {ll} {}",
            b.s
        ));
        let result = self.tmp();
        let instr = if rem { "urem" } else { "udiv" };
        self.emit(format!("{result} = {instr} {ll} {}, {divisor}", a.s));
        let name = self.tmp();
        let on_zero = if rem { a.s.clone() } else { "0".into() };
        self.emit(format!(
            "{name} = select i1 {zero}, {ll} {on_zero}, {ll} {result}"
        ));
        V { s: name, ty: a.ty }
    }

    /// an integer, a `bool` or a kernel pointer converted to the integer type
    /// `int`: truncated, or extended by the signedness of its type
    fn convert(&mut self, value: &V, int: IntTy) -> V {
        let target = self.an_int(int);
        if let TyKind::KPtr(_) = self.layout.kind(value.ty) {
            let address = self.tmp();
            self.emit(format!("{address} = ptrtoint ptr {} to i64", value.s));
            return self.convert(
                &V {
                    s: address,
                    ty: self.an_int(IntTy::U64),
                },
                int,
            );
        }
        let (bits, signed) = match self.layout.kind(value.ty) {
            TyKind::Int(from) => (from.bits(), from.signed()),
            _ => (1, false),
        };
        if bits == int.bits() {
            return V {
                s: value.s.clone(),
                ty: target,
            };
        }
        if let TyKind::Int(_) = self.layout.kind(value.ty)
            && let Some(constant) = self.folded(value)
        {
            return V {
                s: int_const(constant, int),
                ty: target,
            };
        }
        let instr = if bits > int.bits() {
            "trunc"
        } else if signed {
            "sext"
        } else {
            "zext"
        };
        let name = self.tmp();
        self.emit(format!(
            "{name} = {instr} i{bits} {} to i{}",
            value.s,
            int.bits()
        ));
        V {
            s: name,
            ty: target,
        }
    }

    // calls
    // -------------------------

    /// the value of a call: a function, a method, a helper, a constructor, a
    /// builtin, a map method, or a context method
    fn call(&mut self, e: ExprId, callee: ExprId, args: &'a [Arg]) -> Option<V> {
        let ty = self
            .res
            .expr_types
            .get(e)
            .copied()
            .unwrap_or(self.an.types.error);
        let an = self.an;
        let package = &an.package;
        match self.res.res.get(callee).copied()? {
            Res::Fn(f) => {
                let values = self.call_arguments(f, args)?;
                match self.an.helpers.get(f).copied() {
                    Some(number) => self.helper_call(number, f, &values),
                    None => self.internal_call(f, values),
                }
            }
            Res::Method(f) => {
                let Expr::Field { base, .. } = self.data.body.exprs[callee] else {
                    return None;
                };
                let (receiver, back) = self.receiver(f, base)?;
                let mut values = vec![receiver];
                values.extend(self.call_arguments(f, args)?);
                self.method_call(f, values, back)
            }
            Res::StructCtor(s) => {
                let names: Vec<String> = package.structs[s]
                    .fields
                    .iter()
                    .map(|f| f.name.text.clone())
                    .collect();
                let mut values = self.arguments(&names, args)?;
                if self.an.network_order.get(s).is_some() {
                    for value in &mut values {
                        *value = self.byte_swap(value);
                    }
                }
                Some(self.aggregate_value(ty, &values))
            }
            Res::Variant(en, i) => {
                let mut values = Vec::new();
                for arg in args {
                    values.push(self.expr(arg.value)?);
                }
                Some(self.enum_value(en, i, &values))
            }
            Res::Builtin(builtin) => self.builtin(builtin, e, ty, args),
            Res::MapMethod(m, method) => self.map_call(m, method, args, ty),
            Res::PacketParse(header) => {
                let Expr::Index { base, .. } = self.data.body.exprs[callee] else {
                    return None;
                };
                let Expr::Field { base: receiver, .. } = self.data.body.exprs[base] else {
                    return None;
                };
                let context = self.expr(receiver)?;
                let offset = self.expr(args.first()?.value)?;
                Some(self.packet_parse(&context, header, &offset, ty))
            }
            Res::KPtrCast => {
                let address = self.expr(args.first()?.value)?;
                let name = self.tmp();
                self.emit(format!("{name} = inttoptr i64 {} to ptr", address.s));
                Some(V { s: name, ty })
            }
            Res::KPtrIsNull => {
                let Expr::Field { base, .. } = self.data.body.exprs[callee] else {
                    return None;
                };
                let pointer = self.expr(base)?;
                let name = self.tmp();
                self.emit(format!("{name} = icmp eq ptr {}, null", pointer.s));
                Some(V { s: name, ty })
            }
            Res::KprobeArg(_) | Res::KprobeRet => {
                let Expr::Field { base, .. } = self.data.body.exprs[callee] else {
                    return None;
                };
                let context = self.expr(base)?;
                let lsm = matches!(
                    self.layout.kind(context.ty),
                    TyKind::Struct(s) if self.an.struct_contexts.get(*s) == Some(&ContextKind::Lsm)
                );
                // the registers of the arguments and the result are at
                // offsets of `struct pt_regs` that depend on the
                // architecture. An LSM program gets the hook arguments as an
                // array of 64-bit words instead.
                let arch = self.layout.arch;
                let offset = match self.res.res.get(callee) {
                    Some(Res::KprobeArg(n)) if lsm => 8 * *n as usize,
                    Some(Res::KprobeArg(n)) => arch.arg_offsets()[*n as usize] as usize,
                    _ => arch.ret_offset() as usize,
                };
                let field = self.gep("i8", &context.s, &format!("i64 {offset}"));
                Some(self.load(&field, ty))
            }
            Res::PacketLen => {
                let Expr::Field { base, .. } = self.data.body.exprs[callee] else {
                    return None;
                };
                let context = self.expr(base)?;
                Some(self.packet_len(&context))
            }
            _ => None,
        }
    }

    /// the arguments of a call of `f` in parameter order, with the defaults
    /// of the parameters left out. They are evaluated in source order, and
    /// keyword arguments are matched by name.
    fn call_arguments(&mut self, f: FnId, args: &[Arg]) -> Option<Vec<V>> {
        let sig = &self.an.fn_sigs[f];
        let params: Vec<(String, TyId)> = sig.call_params().to_vec();
        let defaults: Vec<Option<ConstValue>> = sig.call_defaults().to_vec();
        let mut slots: Vec<Option<V>> = vec![None; params.len()];
        for (i, arg) in args.iter().enumerate() {
            let value = self.expr(arg.value)?;
            let slot = match &arg.name {
                Some(name) => params.iter().position(|(n, _)| *n == name.text)?,
                None => i,
            };
            *slots.get_mut(slot)? = Some(value);
        }
        slots
            .into_iter()
            .zip(params.iter().zip(&defaults))
            .map(|(slot, (&(_, ty), default))| match (slot, default) {
                (Some(value), _) => Some(value),
                (None, Some(ConstValue::Int(v))) => Some(V {
                    s: int_const(*v, self.int_ty(ty)),
                    ty,
                }),
                (None, Some(ConstValue::Bool(b))) => Some(V {
                    s: b.to_string(),
                    ty,
                }),
                (None, None) => None,
            })
            .collect()
    }

    /// the arguments of a call in the order of `names`, the fields of a
    /// struct, evaluated in source order. `None` if one is missing.
    fn arguments(&mut self, names: &[String], args: &[Arg]) -> Option<Vec<V>> {
        let mut slots: Vec<Option<V>> = vec![None; names.len()];
        for (i, arg) in args.iter().enumerate() {
            let value = self.expr(arg.value)?;
            let slot = match &arg.name {
                Some(name) => names.iter().position(|n| *n == name.text)?,
                None => i,
            };
            *slots.get_mut(slot)? = Some(value);
        }
        slots.into_iter().collect()
    }

    /// the value of a call of an internal function, `None` for unit. The
    /// caller aborts if the callee did.
    fn internal_call(&mut self, f: FnId, values: Vec<V>) -> Option<V> {
        let ret = self.an.fn_sigs[f].ret;
        let ll = self.internal_ret(ret);
        let scratch = self.callee_scratch();
        self.calls.push(f);
        let mut args = vec![format!("ptr {scratch}")];
        let destination = self.by_address(ret).then(|| self.alloca(ret));
        if let Some(destination) = &destination {
            args.push(format!("ptr {destination}"));
        }
        for v in &values {
            args.push(format!("{} {}", self.value_ll(v.ty), v.s));
        }
        let result = self.tmp();
        self.emit(format!(
            "{result} = call {ll} @\"{}\"({})",
            symbol(self.an, f),
            args.join(", ")
        ));
        let unit = self.layout.is_unit(ret) || destination.is_some();
        let aborted = if unit {
            result.clone()
        } else {
            let name = self.tmp();
            self.emit(format!("{name} = extractvalue {ll} {result}, 1"));
            name
        };
        let ok = self.label("ret");
        self.aborts = true;
        self.cond_br(&aborted, "abort", &ok);
        self.place(ok);
        if let Some(destination) = destination {
            return Some(V {
                s: destination,
                ty: ret,
            });
        }
        if unit {
            return None;
        }
        let name = self.tmp();
        self.emit(format!("{name} = extractvalue {ll} {result}, 0"));
        Some(V { s: name, ty: ret })
    }

    /// the value of a call of the kernel helper `number`. Every argument is
    /// passed as a 64-bit word, and the result is cut back to the type of
    /// `f`.
    fn helper_call(&mut self, number: u32, f: FnId, values: &[V]) -> Option<V> {
        let mut args = Vec::new();
        for value in values {
            let int = match self.layout.kind(value.ty) {
                TyKind::Int(int) => *int,
                _ => IntTy::U8,
            };
            let wide = if matches!(self.layout.kind(value.ty), TyKind::Bool) {
                let name = self.tmp();
                self.emit(format!("{name} = zext i1 {} to i64", value.s));
                name
            } else if matches!(self.layout.kind(value.ty), TyKind::KPtr(_)) {
                let name = self.tmp();
                self.emit(format!("{name} = ptrtoint ptr {} to i64", value.s));
                name
            } else {
                self.widen(value, int)
            };
            args.push(format!("i64 {wide}"));
        }
        let ret = self.an.fn_sigs[f].ret;
        // a helper giving an aggregate fills a buffer given with its size, as
        // `bpf_get_current_comm(buf, size)` does
        if matches!(self.layout.kind(ret), TyKind::Array(..) | TyKind::Struct(_)) {
            let buffer = self.alloca(ret);
            let (size, _) = self.layout.size_align(ret);
            self.memset_zero(&buffer, ret);
            let mut all = vec![format!("ptr {buffer}"), format!("i64 {size}")];
            all.extend(args);
            let result = self.tmp();
            self.emit(format!(
                "{result} = call i64 inttoptr (i64 {number} to ptr)({})",
                all.join(", ")
            ));
            return Some(self.load(&buffer, ret));
        }
        let result = self.tmp();
        self.emit(format!(
            "{result} = call i64 inttoptr (i64 {number} to ptr)({})",
            args.join(", ")
        ));
        match *self.layout.kind(ret) {
            TyKind::Int(int) if int.bits() < 64 => {
                let name = self.tmp();
                self.emit(format!("{name} = trunc i64 {result} to i{}", int.bits()));
                Some(V { s: name, ty: ret })
            }
            TyKind::Int(_) => Some(V { s: result, ty: ret }),
            TyKind::Bool => {
                let name = self.tmp();
                self.emit(format!("{name} = icmp ne i64 {result}, 0"));
                Some(V { s: name, ty: ret })
            }
            _ => None,
        }
    }

    /// the value of a call of a builtin function
    fn builtin(&mut self, builtin: Builtin, e: ExprId, ty: TyId, args: &'a [Arg]) -> Option<V> {
        match builtin {
            Builtin::Some => {
                let value = self.expr(args.first()?.value)?;
                if self.layout.is_niche(ty) {
                    return Some(V { s: value.s, ty });
                }
                let tag = V {
                    s: "true".into(),
                    ty: self.an.types.bool,
                };
                Some(self.aggregate_value(ty, &[tag, value]))
            }
            Builtin::Int(int) => {
                let value = self.expr(args.first()?.value)?;
                Some(self.convert(&value, int))
            }
            Builtin::Min | Builtin::Max => {
                let a = self.expr(args.first()?.value)?;
                let b = self.expr(args.get(1)?.value)?;
                let op = if builtin == Builtin::Min {
                    BinaryOp::Lt
                } else {
                    BinaryOp::Gt
                };
                let bool_ty = self.an.types.bool;
                let cond = self.compare(op, &a, &b, bool_ty);
                let ll = self.ll(a.ty);
                let name = self.tmp();
                self.emit(format!(
                    "{name} = select i1 {}, {ll} {}, {ll} {}",
                    cond.s, a.s, b.s
                ));
                Some(V { s: name, ty: a.ty })
            }
            Builtin::SaturatingAdd | Builtin::SaturatingSub => {
                let a = self.expr(args.first()?.value)?;
                let b = self.expr(args.get(1)?.value)?;
                Some(self.saturating(&a, &b, builtin == Builtin::SaturatingAdd))
            }
            Builtin::Clamp => {
                let x = self.expr(args.first()?.value)?;
                let lo = self.expr(args.get(1)?.value)?;
                let hi = self.expr(args.get(2)?.value)?;
                let at_least_lo = self.branchless(&x, &lo, false);
                Some(self.branchless(&at_least_lo, &hi, true))
            }
            Builtin::Print => {
                self.print(e, args.first()?.value);
                None
            }
            Builtin::Run => {
                // a helper of the userspace simulator of the full Vigil
                // (`HELPER_RUN`): it runs the program, named by its
                // id, on the packet bytes at the address given
                let target = args.first()?.value;
                let &(program, damaged) = self.res.runs.get(target)?;
                let packet = self.expr(args.get(1)?.value)?;
                let address = self.spill(&packet);
                let id = program.into_raw().into_u32();
                let verdict = self.tmp();
                self.emit(format!(
                    "{verdict} = call i64 inttoptr (i64 {} to ptr)(i64 {id}, ptr {address}, i64 {})",
                    crate::HELPER_RUN,
                    u8::from(damaged)
                ));
                // the verdict is the variant index of the action enum of the
                // program
                let ll = self.ll(ty);
                let slot = self.alloca(ty);
                let tag = self.gep(&ll, &slot, "i32 0, i32 0");
                let narrow = self.tmp();
                self.emit(format!("{narrow} = trunc i64 {verdict} to i32"));
                self.emit(format!("store i32 {narrow}, ptr {tag}, align 4"));
                Some(self.load(&slot, ty))
            }
            Builtin::SetTime => {
                let ns = self.expr(args.first()?.value)?;
                let result = self.tmp();
                self.emit(format!(
                    "{result} = call i64 inttoptr (i64 {} to ptr)(i64 {})",
                    crate::HELPER_SET_TIME,
                    ns.s
                ));
                None
            }
            _ => None,
        }
    }

    /// `min(a, b)`, or `max` when `min` is false, without branches and exact
    /// for every pair (Hacker's Delight, 2-12)
    ///
    /// A mask of ones when `a < b` selects with `xor` and `and`. The mask
    /// goes through `llvm.bpf.passthrough`, otherwise LLVM recognizes `smin`
    /// and lowers it to a branch.
    fn branchless(&mut self, a: &V, b: &V, min: bool) -> V {
        let int = self.int_ty(a.ty);
        let ll = self.ll(a.ty);
        if self.layout.optimize && int.bits() < 64 {
            // narrower than a register: the difference of the values widened
            // to 64 bits cannot overflow, and its sign says which is smaller
            let (a64, b64) = (self.widen64(a), self.widen64(b));
            let d = self.tmp();
            self.emit(format!("{d} = sub i64 {a64}, {b64}"));
            let mask = self.negative_mask(&d);
            let diff = self.tmp();
            self.emit(format!("{diff} = xor i64 {a64}, {b64}"));
            let picked = self.tmp();
            self.emit(format!("{picked} = and i64 {diff}, {mask}"));
            let base = if min { b64 } else { a64 };
            let wide = self.tmp();
            self.emit(format!("{wide} = xor i64 {base}, {picked}"));
            return self.narrow64(&wide, a.ty);
        }
        let (a_s, b_s) = (a.s.clone(), b.s.clone());
        let op = |this: &mut Self, instr: String| {
            let name = this.tmp();
            this.emit(format!("{name} = {instr}"));
            name
        };
        let d = op(self, format!("sub {ll} {a_s}, {b_s}"));
        let sign = if int.signed() {
            let x = op(self, format!("xor {ll} {a_s}, {b_s}"));
            let y = op(self, format!("xor {ll} {d}, {a_s}"));
            let z = op(self, format!("and {ll} {x}, {y}"));
            op(self, format!("xor {ll} {d}, {z}"))
        } else {
            let not_a = op(self, format!("xor {ll} {a_s}, -1"));
            let x = op(self, format!("and {ll} {not_a}, {b_s}"));
            let y = op(self, format!("or {ll} {not_a}, {b_s}"));
            let z = op(self, format!("and {ll} {y}, {d}"));
            op(self, format!("or {ll} {x}, {z}"))
        };
        let mask = op(self, format!("ashr {ll} {sign}, {}", int.bits() - 1));
        let sequence = self.meta.passthrough();
        let mask = op(
            self,
            format!("call {ll} @llvm.bpf.passthrough.{ll}.{ll}(i32 {sequence}, {ll} {mask})"),
        );
        let diff = op(self, format!("xor {ll} {a_s}, {b_s}"));
        let picked = op(self, format!("and {ll} {diff}, {mask}"));
        let base = if min { b_s } else { a_s };
        let name = op(self, format!("xor {ll} {base}, {picked}"));
        V { s: name, ty: a.ty }
    }

    /// a value widened to 64 bits, by the signedness of its type
    fn widen64(&mut self, v: &V) -> String {
        let int = self.int_ty(v.ty);
        if int.bits() == 64 {
            return v.s.clone();
        }
        let ext = if int.signed() { "sext" } else { "zext" };
        let name = self.tmp();
        self.emit(format!("{name} = {ext} i{} {} to i64", int.bits(), v.s));
        name
    }

    /// a 64-bit value cut to the width of `ty`
    fn narrow64(&mut self, wide: &str, ty: TyId) -> V {
        let bits = self.int_ty(ty).bits();
        if bits == 64 {
            return V { s: wide.into(), ty };
        }
        let name = self.tmp();
        self.emit(format!("{name} = trunc i64 {wide} to i{bits}"));
        V { s: name, ty }
    }

    /// all ones if the 64-bit `x` is negative, zero otherwise
    ///
    /// Both `x` and the mask go through `llvm.bpf.passthrough`, otherwise
    /// LLVM turns the use of the mask into a compare and a branch: knowing
    /// how `x` is computed (`MAX - (c + 1)`, say), it rewrites the mask as
    /// `c == MAX`.
    fn negative_mask(&mut self, x: &str) -> String {
        let passthrough = |this: &mut Self, value: &str| {
            let sequence = this.meta.passthrough();
            let name = this.tmp();
            this.emit(format!(
                "{name} = call i64 @llvm.bpf.passthrough.i64.i64(i32 {sequence}, i64 {value})"
            ));
            name
        };
        let opaque = passthrough(self, x);
        let shifted = self.tmp();
        self.emit(format!("{shifted} = ashr i64 {opaque}, 63"));
        passthrough(self, &shifted)
    }

    /// `saturating_add(a, b)` or `saturating_sub(a, b)` without branches:
    /// `a ± b` in the type of `a`, or the bound of the type the exact result
    /// goes past
    fn saturating(&mut self, a: &V, b: &V, add: bool) -> V {
        let int = self.int_ty(a.ty);
        let op = |this: &mut Self, instr: String| {
            let name = this.tmp();
            this.emit(format!("{name} = {instr}"));
            name
        };
        let arith = if add { "add" } else { "sub" };
        if int.bits() < 64 {
            // narrower than a register: the exact result fits in 64 bits,
            // and we clamp it to the type
            let (a64, b64) = (self.widen64(a), self.widen64(b));
            let mut r = op(self, format!("{arith} i64 {a64}, {b64}"));
            // above the maximum, which an unsigned subtraction cannot be
            if int.signed() || add {
                let max = int.max();
                let gap = op(self, format!("sub i64 {max}, {r}"));
                let mask = self.negative_mask(&gap);
                let diff = op(self, format!("xor i64 {r}, {max}"));
                let picked = op(self, format!("and i64 {diff}, {mask}"));
                r = op(self, format!("xor i64 {r}, {picked}"));
            }
            // below the minimum, which an unsigned addition cannot be
            if int.signed() || !add {
                let min = int.min();
                let gap = op(self, format!("sub i64 {r}, {min}"));
                let mask = self.negative_mask(&gap);
                let diff = op(self, format!("xor i64 {r}, {min}"));
                let picked = op(self, format!("and i64 {diff}, {mask}"));
                r = op(self, format!("xor i64 {r}, {picked}"));
            }
            return self.narrow64(&r, a.ty);
        }
        // 64 bits: the overflow comes from the signs (Hacker's Delight, 2-13)
        let (x, y) = (a.s.clone(), b.s.clone());
        let r = op(self, format!("{arith} i64 {x}, {y}"));
        let name = if !int.signed() {
            let not_x = op(self, format!("xor i64 {x}, -1"));
            // the carry of `+`, or the borrow of `-`, in the top bit
            let carry = if add {
                let both = op(self, format!("and i64 {x}, {y}"));
                let either = op(self, format!("or i64 {x}, {y}"));
                let not_r = op(self, format!("xor i64 {r}, -1"));
                let lost = op(self, format!("and i64 {either}, {not_r}"));
                op(self, format!("or i64 {both}, {lost}"))
            } else {
                let both = op(self, format!("and i64 {not_x}, {y}"));
                let either = op(self, format!("or i64 {not_x}, {y}"));
                let kept = op(self, format!("and i64 {either}, {r}"));
                op(self, format!("or i64 {both}, {kept}"))
            };
            let mask = self.negative_mask(&carry);
            if add {
                op(self, format!("or i64 {r}, {mask}"))
            } else {
                let keep = op(self, format!("xor i64 {mask}, -1"));
                op(self, format!("and i64 {r}, {keep}"))
            }
        } else {
            let overflow = if add {
                let p = op(self, format!("xor i64 {x}, {r}"));
                let q = op(self, format!("xor i64 {y}, {r}"));
                op(self, format!("and i64 {p}, {q}"))
            } else {
                let p = op(self, format!("xor i64 {x}, {y}"));
                let q = op(self, format!("xor i64 {x}, {r}"));
                op(self, format!("and i64 {p}, {q}"))
            };
            let mask = self.negative_mask(&overflow);
            // the bound passed: the minimum when `a` is negative, the maximum
            // otherwise
            let sign = op(self, format!("ashr i64 {x}, 63"));
            let bound = op(self, format!("xor i64 {sign}, {}", i64::MAX));
            let diff = op(self, format!("xor i64 {r}, {bound}"));
            let picked = op(self, format!("and i64 {diff}, {mask}"));
            op(self, format!("xor i64 {r}, {picked}"))
        };
        V { s: name, ty: a.ty }
    }

    /// `print(s)`: `bpf_trace_printk` with the format string on the stack
    ///
    /// `s` is a string literal or an f-string. Each value in it becomes a
    /// helper argument: integers and `bool`s as 64-bit words, `bytes[N]` as
    /// a string. A format longer than [`MAX_FORMAT`] is an error.
    fn print(&mut self, call: ExprId, arg: ExprId) {
        let parts: Vec<Result<String, ExprId>> = match &self.data.body.exprs[arg] {
            Expr::Str(s) => vec![Ok(s.clone())],
            Expr::FStr(parts) => parts
                .iter()
                .map(|p| match p {
                    FStrPart::Text(t) => Ok(t.clone()),
                    FStrPart::Expr(e) => Err(*e),
                })
                .collect(),
            _ => return,
        };
        let mut format = String::new();
        let mut values = Vec::new();
        // the stack buffers of this call and their sizes, whose lifetimes end
        // after it
        let mut strings: Vec<(String, u64)> = Vec::new();
        for part in parts {
            match part {
                Ok(text) => format.push_str(&text.replace('%', "%%")),
                Err(e) => {
                    let Some(value) = self.expr(e) else {
                        return;
                    };
                    let (specifier, wide) = match *self.layout.kind(value.ty) {
                        TyKind::Int(int) if int.signed() => ("%lld", self.widen(&value, int)),
                        TyKind::Int(int) => ("%llu", self.widen(&value, int)),
                        // `bytes[N]`: a copy ending in NUL, whose address the
                        // helper reads as a string
                        TyKind::Array(_, len) => {
                            let buffer = self.stack_alloca(&format!("[{} x i8]", len + 1), 8);
                            self.emit(format!(
                                "call void @llvm.lifetime.start.p0(i64 {}, ptr {buffer})",
                                len + 1
                            ));
                            strings.push((buffer.clone(), len + 1));
                            self.store(&value, &buffer);
                            let end = self.gep("i8", &buffer, &format!("i64 {len}"));
                            self.emit(format!("store i8 0, ptr {end}, align 1"));
                            let address = self.tmp();
                            self.emit(format!("{address} = ptrtoint ptr {buffer} to i64"));
                            ("%s", address)
                        }
                        _ => {
                            let name = self.tmp();
                            self.emit(format!("{name} = zext i1 {} to i64", value.s));
                            ("%llu", name)
                        }
                    };
                    format.push_str(specifier);
                    values.push(wide);
                }
            }
        }
        if !format.ends_with('\n') {
            format.push('\n');
        }
        let mut bytes = format.into_bytes();
        bytes.push(0);
        if bytes.len() > MAX_FORMAT {
            self.diags.push(
                Diagnostic::error("this message is too long for `print`")
                    .with_code(codes::CODEGEN_LIMIT)
                    .with_primary(
                        self.data.spans.exprs[call],
                        format!(
                            "a format string of {} bytes, at most {MAX_FORMAT}",
                            bytes.len()
                        ),
                    )
                    .with_note("the format string is built on the 512-byte stack")
                    .with_help("split the message into several `print` calls"),
            );
            return;
        }
        let size = bytes.len();
        let words = size.div_ceil(8);
        let buffer = self.tmp();
        self.allocas
            .push(format!("  {buffer} = alloca [{} x i8], align 8", words * 8));
        // the buffers live only around the call. Knowing their lifetimes,
        // LLVM gives the buffers of different `print`s the same stack.
        self.emit(format!(
            "call void @llvm.lifetime.start.p0(i64 {}, ptr {buffer})",
            words * 8
        ));
        strings.push((buffer.clone(), words as u64 * 8));
        bytes.resize(words * 8, 0);
        for (i, chunk) in bytes.chunks(8).enumerate() {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            let ptr = self.gep(
                &format!("[{} x i8]", words * 8),
                &buffer,
                &format!("i64 0, i64 {}", i * 8),
            );
            self.emit(format!("store i64 {}, ptr {ptr}, align 8", word as i64));
        }
        let mut args = vec![format!("ptr {buffer}"), format!("i32 {size}")];
        args.extend(values.iter().map(|v| format!("i64 {v}")));
        let result = self.tmp();
        self.emit(format!(
            "{result} = call i64 inttoptr (i64 {TRACE_PRINTK} to ptr)({})",
            args.join(", ")
        ));
        for (buffer, size) in strings {
            self.emit(format!(
                "call void @llvm.lifetime.end.p0(i64 {size}, ptr {buffer})"
            ));
        }
    }
}

// packets, kernel memory and maps
// -------------------------

impl FnGen<'_, '_> {
    /// whether `e` is an integer field, wider than a byte, of a
    /// `@network_order` struct, so stored big-endian
    fn swaps(&self, e: ExprId) -> bool {
        let Expr::Field { base, .. } = self.data.body.exprs[e] else {
            return false;
        };
        let Some(Res::Field(_)) = self.res.res.get(e) else {
            return false;
        };
        let base_ty = self.res.expr_types[base];
        let struct_ty = match *self.layout.kind(base_ty) {
            TyKind::Ref(inner) => inner,
            _ => base_ty,
        };
        let network = matches!(
            self.layout.kind(struct_ty),
            TyKind::Struct(s) if self.an.network_order.get(*s).is_some()
        );
        network
            && matches!(self.layout.kind(self.res.expr_types[e]), TyKind::Int(int) if int.bits() > 8)
    }

    /// an integer wider than a byte with its bytes swapped, between
    /// big-endian and host order; any other value as it is
    fn byte_swap(&mut self, value: &V) -> V {
        let TyKind::Int(int) = *self.layout.kind(value.ty) else {
            return value.clone();
        };
        if int.bits() == 8 {
            return value.clone();
        }
        let bits = int.bits();
        let name = self.tmp();
        self.emit(format!(
            "{name} = call i{bits} @llvm.bswap.i{bits}(i{bits} {})",
            value.s
        ));
        V {
            s: name,
            ty: value.ty,
        }
    }

    /// pointers to the start and end of the packet, read from the context:
    /// `xdp_md.data` and `data_end` (offsets 0 and 4), or `__sk_buff.data`
    /// and `data_end` (76 and 80), as 32-bit fields the verifier turns into
    /// packet pointers
    ///
    /// With `Layout::test_context` they are 64-bit words at offsets 0 and 8
    /// instead, as the userspace interpreters want. The kernel would reject
    /// that.
    fn packet_bounds(&mut self, context: &V) -> (String, String) {
        let kind = match self.layout.kind(context.ty) {
            TyKind::Struct(s) => self.an.struct_contexts.get(*s).copied(),
            _ => None,
        };
        let (data, end, width) = if self.layout.test_context {
            (0, 8, 64)
        } else if kind == Some(ContextKind::Tc) {
            (76, 80, 32)
        } else {
            (0, 4, 32)
        };
        let pointer = |g: &mut Self, offset: u32| {
            let field = g.gep("i8", &context.s, &format!("i64 {offset}"));
            let raw = g.tmp();
            g.emit(format!(
                "{raw} = load i{width}, ptr {field}, align {}",
                width / 8
            ));
            let wide = if width == 64 {
                raw
            } else {
                let name = g.tmp();
                g.emit(format!("{name} = zext i32 {raw} to i64"));
                name
            };
            let name = g.tmp();
            g.emit(format!("{name} = inttoptr i64 {wide} to ptr"));
            name
        };
        let start = pointer(self, data);
        let end = pointer(self, end);
        (start, end)
    }

    /// `ctx.parse[H](offset)`: the address of the header at `offset`, or
    /// null if the packet is too short
    ///
    /// We first bound the offset by [`MAX_PACKET_OFFSET`], because the
    /// verifier needs a known range for a variable offset, then compare the
    /// end of the header with the end of the packet.
    fn packet_parse(&mut self, context: &V, header: StructId, offset: &V, ty: TyId) -> V {
        let (data, data_end) = self.packet_bounds(context);
        let offset = self.convert(offset, IntTy::U64);
        let header_ty = self
            .an
            .types
            .lookup(&TyKind::Struct(header))
            .expect("struct types are interned");
        let (size, _) = self.layout.size_align(header_ty);
        let in_range = self.tmp();
        self.emit(format!(
            "{in_range} = icmp ule i64 {}, {}",
            offset.s,
            MAX_PACKET_OFFSET.saturating_sub(size)
        ));
        let check = self.label("bounds");
        let short = self.label("short");
        let done = self.label("parsed");
        self.cond_br(&in_range, &check, &done);
        // the predecessors of `done` are read after their branches, which
        // may have opened new blocks
        let start_block = self.block.clone();
        self.place(check);
        let at = self.gep("i8", &data, &format!("i64 {}", offset.s));
        let end = self.gep("i8", &at, &format!("i64 {size}"));
        let beyond = self.tmp();
        self.emit(format!("{beyond} = icmp ugt ptr {end}, {data_end}"));
        // the barrier keeps LLVM from turning the null test of the result
        // into a 32-bit test of `data`, which the verifier rejects
        let opaque = self.tmp();
        self.emit(format!(
            "{opaque} = call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr {at})"
        ));
        self.cond_br(&beyond, &short, &done);
        let check_block = self.block.clone();
        self.place(short);
        self.br(&done);
        let short_block = self.block.clone();
        self.place(done);
        let result = self.tmp();
        self.emit(format!(
            "{result} = phi ptr [ null, %{start_block} ], [ {opaque}, %{check_block} ], [ null, %{short_block} ]"
        ));
        V { s: result, ty }
    }

    /// `ctx.parse[H](offset)?`: the address of the header, with an early
    /// return of the program if the packet is too short. No null pointer is
    /// built, so the verifier sees only the bounds checks.
    fn packet_checked(&mut self, context: &V, header: StructId, offset: &V) -> String {
        let (data, data_end) = self.packet_bounds(context);
        let offset = self.convert(offset, IntTy::U64);
        let header_ty = self
            .an
            .types
            .lookup(&TyKind::Struct(header))
            .expect("struct types are interned");
        let (size, _) = self.layout.size_align(header_ty);
        let fail = self.label("short");
        let check = self.label("bounds");
        let ok = self.label("parsed");
        let in_range = self.tmp();
        self.emit(format!(
            "{in_range} = icmp ule i64 {}, {}",
            offset.s,
            MAX_PACKET_OFFSET.saturating_sub(size)
        ));
        self.cond_br(&in_range, &check, &fail);
        self.place(check);
        let at = self.gep("i8", &data, &format!("i64 {}", offset.s));
        let end = self.gep("i8", &at, &format!("i64 {size}"));
        let beyond = self.tmp();
        self.emit(format!("{beyond} = icmp ugt ptr {end}, {data_end}"));
        self.cond_br(&beyond, &fail, &ok);
        self.place(fail);
        self.early_exit(false);
        self.place(ok);
        at
    }

    /// the value of a field of a kernel struct at `base`, or its address
    ///
    /// The offset is loaded from a relocation global, which the backend
    /// turns into an immediate with a CO-RE record, so the loader adjusts it
    /// to the target kernel. The field is then read directly or copied with
    /// `bpf_probe_read_kernel` (`bpf_probe_read` without
    /// `Layout::probe_read_kernel`), as the access says.
    fn kernel_read(&mut self, base: &V, access: &KernelAccess, ty: TyId) -> V {
        let global = self
            .meta
            .kernel_access(access.composite, &access.path, access.offset);
        let offset = self.tmp();
        self.emit(format!("{offset} = load i64, ptr {global}, align 8"));
        let address = self.gep("i8", &base.s, &format!("i64 {offset}"));
        let opaque = self.tmp();
        let sequence = self.meta.passthrough();
        self.emit(format!(
            "{opaque} = call ptr @llvm.bpf.passthrough.p0.p0(i32 {sequence}, ptr {address})"
        ));
        match access.read {
            KernelRead::Address => V { s: opaque, ty },
            KernelRead::Direct => self.load(&opaque, ty),
            KernelRead::Probe => {
                let buffer = self.alloca(ty);
                let (size, _) = self.layout.size_align(ty);
                let helper = if self.layout.probe_read_kernel {
                    113
                } else {
                    4
                };
                let result = self.tmp();
                self.emit(format!(
                    "{result} = call i64 inttoptr (i64 {helper} to ptr)(ptr {buffer}, i32 {size}, ptr {opaque})"
                ));
                self.load(&buffer, ty)
            }
        }
    }

    /// `ctx.len()`: the length of the packet in bytes
    fn packet_len(&mut self, context: &V) -> V {
        let u32_ty = self.an_int(IntTy::U32);
        let is_tc = matches!(
            self.layout.kind(context.ty),
            TyKind::Struct(s) if self.an.struct_contexts.get(*s) == Some(&ContextKind::Tc)
        );
        if is_tc && !self.layout.test_context {
            // `__sk_buff.len`, which counts non-linear data too; the
            // difference of the bounds would not
            let field = self.gep("i8", &context.s, "i64 0");
            return self.load(&field, u32_ty);
        }
        let (data, end) = self.packet_bounds(context);
        let (a, b) = (self.tmp(), self.tmp());
        self.emit(format!("{a} = ptrtoint ptr {data} to i64"));
        self.emit(format!("{b} = ptrtoint ptr {end} to i64"));
        let difference = self.tmp();
        self.emit(format!("{difference} = sub i64 {b}, {a}"));
        // the barrier, an empty inline asm, keeps LLVM from narrowing the
        // subtraction to 32 bits, which the verifier forbids on pointers
        let opaque = self.tmp();
        self.emit(format!(
            "{opaque} = call i64 asm \"\", \"=r,0\"(i64 {difference})"
        ));
        let name = self.tmp();
        self.emit(format!("{name} = trunc i64 {opaque} to i32"));
        V {
            s: name,
            ty: u32_ty,
        }
    }

    /// a slot holding `value`, zeroed first so the padding of a key is
    /// deterministic, because the kernel hashes every byte
    fn zeroed_slot(&mut self, value: &V) -> String {
        let slot = self.alloca(value.ty);
        self.memset_zero(&slot, value.ty);
        self.store(value, &slot);
        slot
    }

    /// the LLVM symbol of a map
    fn map_symbol(&self, m: MapId) -> String {
        format!("@\"{}\"", self.an.package.maps[m].name.text)
    }

    /// `bpf_map_lookup_elem`: the address of the value of the key at
    /// `key_slot`, or null
    fn lookup(&mut self, m: MapId, key_slot: &str) -> String {
        let map = self.map_symbol(m);
        let result = self.tmp();
        self.emit(format!(
            "{result} = call ptr inttoptr (i64 {MAP_LOOKUP} to ptr)(ptr {map}, ptr {key_slot})"
        ));
        result
    }

    /// `bpf_map_update_elem`, and whether it succeeded
    fn update(&mut self, m: MapId, key_slot: &str, value_slot: &str, flags: u64) -> String {
        let map = self.map_symbol(m);
        let result = self.tmp();
        self.emit(format!(
            "{result} = call i64 inttoptr (i64 {MAP_UPDATE} to ptr)(ptr {map}, ptr {key_slot}, ptr {value_slot}, i64 {flags})"
        ));
        let ok = self.tmp();
        self.emit(format!("{ok} = icmp eq i64 {result}, 0"));
        ok
    }

    /// the value of a call of a map method
    fn map_call(&mut self, m: MapId, method: MapMethod, args: &[Arg], ty: TyId) -> Option<V> {
        let names: &[&str] = match method {
            MapMethod::GetOrInsert => &["key", "default"],
            MapMethod::Insert => &["key", "value"],
            MapMethod::Emit => &["event"],
            MapMethod::TailCall => &["ctx", "index"],
            MapMethod::Count | MapMethod::Last => &[],
            _ => &["key"],
        };
        let names: Vec<String> = names.iter().map(|n| n.to_string()).collect();
        let values = self.arguments(&names, args)?;
        let bool_ty = self.an.types.bool;
        let boolean = |s: String| Some(V { s, ty: bool_ty });
        match method {
            MapMethod::Get => {
                let key = self.zeroed_slot(&values[0]);
                let pointer = self.lookup(m, &key);
                Some(V { s: pointer, ty })
            }
            MapMethod::Contains => {
                let key = self.zeroed_slot(&values[0]);
                let pointer = self.lookup(m, &key);
                let name = self.tmp();
                self.emit(format!("{name} = icmp ne ptr {pointer}, null"));
                boolean(name)
            }
            MapMethod::Insert => {
                let key = self.zeroed_slot(&values[0]);
                let value = self.zeroed_slot(&values[1]);
                boolean(self.update(m, &key, &value, BPF_ANY))
            }
            MapMethod::Remove => {
                let key = self.zeroed_slot(&values[0]);
                let map = self.map_symbol(m);
                let result = self.tmp();
                self.emit(format!(
                    "{result} = call i64 inttoptr (i64 {MAP_DELETE} to ptr)(ptr {map}, ptr {key})"
                ));
                let name = self.tmp();
                self.emit(format!("{name} = icmp eq i64 {result}, 0"));
                boolean(name)
            }
            MapMethod::Emit => {
                let event = self.zeroed_slot(&values[0]);
                let (size, _) = self.layout.size_align(values[0].ty);
                let map = self.map_symbol(m);
                let result = self.tmp();
                self.emit(format!(
                    "{result} = call i64 inttoptr (i64 {RINGBUF_OUTPUT} to ptr)(ptr {map}, ptr {event}, i64 {size}, i64 0)"
                ));
                let name = self.tmp();
                self.emit(format!("{name} = icmp eq i64 {result}, 0"));
                boolean(name)
            }
            MapMethod::Count => {
                // a helper of the userspace simulator
                // (`HELPER_EVENT_COUNT`): the number of events of the
                // ring buffer
                let map = self.map_symbol(m);
                let result = self.tmp();
                self.emit(format!(
                    "{result} = call i64 inttoptr (i64 {} to ptr)(ptr {map})",
                    crate::HELPER_EVENT_COUNT
                ));
                Some(V { s: result, ty })
            }
            MapMethod::Last => {
                // a helper of the userspace simulator
                // (`HELPER_LAST_EVENT`): it copies the last event
                // into a buffer and says whether there is one
                let event_ty = self.an.maps[m].value;
                let buffer = self.alloca(event_ty);
                let (size, _) = self.layout.size_align(event_ty);
                self.memset_zero(&buffer, event_ty);
                let map = self.map_symbol(m);
                let result = self.tmp();
                self.emit(format!(
                    "{result} = call i64 inttoptr (i64 {} to ptr)(ptr {map}, ptr {buffer}, i64 {size})",
                    crate::HELPER_LAST_EVENT
                ));
                let found = self.tmp();
                self.emit(format!("{found} = icmp ne i64 {result}, 0"));
                let event = self.load(&buffer, event_ty);
                let tag = V {
                    s: found,
                    ty: bool_ty,
                };
                Some(self.aggregate_value(ty, &[tag, event]))
            }
            MapMethod::TailCall => {
                // the tail call only returns when there is no program at the
                // index, and then the program goes on
                let map = self.map_symbol(m);
                let index = self.widen(&values[1], IntTy::U32);
                let result = self.tmp();
                self.emit(format!(
                    "{result} = call i64 inttoptr (i64 {TAIL_CALL} to ptr)(ptr {}, ptr {map}, i64 {index})",
                    values[0].s
                ));
                None
            }
            MapMethod::GetOrInsert => {
                // look up; if absent insert the default, without overwriting
                // an entry another CPU added meanwhile; look up again. A full
                // map aborts the program.
                let key = self.zeroed_slot(&values[0]);
                let first = self.lookup(m, &key);
                let first_block = self.block.clone();
                let found = self.tmp();
                self.emit(format!("{found} = icmp ne ptr {first}, null"));
                let done = self.label("found");
                let insert = self.label("insert");
                self.cond_br(&found, &done, &insert);
                self.place(insert);
                let value = self.zeroed_slot(&values[1]);
                self.update(m, &key, &value, BPF_NOEXIST);
                let second = self.lookup(m, &key);
                let present = self.tmp();
                self.emit(format!("{present} = icmp ne ptr {second}, null"));
                let second_block = self.block.clone();
                self.aborts = true;
                self.cond_br(&present, &done, "abort");
                self.place(done);
                let pointer = self.tmp();
                self.emit(format!(
                    "{pointer} = phi ptr [ {first}, %{first_block} ], [ {second}, %{second_block} ]"
                ));
                Some(V { s: pointer, ty })
            }
        }
    }

    /// `map[index]` on an array map: a lookup with a `u32` key
    fn map_index(&mut self, m: MapId, index: ExprId, ty: TyId) -> Option<V> {
        let index = self.expr(index)?;
        let key = self.convert(&index, IntTy::U32);
        let slot = self.zeroed_slot(&key);
        let pointer = self.lookup(m, &slot);
        Some(V { s: pointer, ty })
    }
}

/// how a loop may be unrolled, as its decorators say
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unroll {
    /// `@unroll`: unrolled by us, or by LLVM when too long for us
    Always,
    /// `@no_unroll`: unrolled neither by us nor by LLVM
    Never,
    /// no decorator: unrolled when short and small
    /// ([`FnGen::unrolled_count`])
    Auto,
}

impl Unroll {
    /// what the decorators of a loop say; `@unroll` wins over `@no_unroll`
    fn of(decorators: &[Decorator]) -> Unroll {
        let has = |name: &str| decorators.iter().any(|d| d.name.to_string() == name);
        if has("unroll") {
            Unroll::Always
        } else if has("no_unroll") {
            Unroll::Never
        } else {
            Unroll::Auto
        }
    }
}

/// the name of the per-CPU array with the scratch of a program
pub(crate) fn scratch_map(program: &str) -> String {
    format!("_vigil_scratch_{program}")
}
