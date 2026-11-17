//! this module implements the type checking of function bodies, by local
//! inference. `check_fn` checks a body against its signature in the [`Db`]
//! and gives an [`InferenceResult`]. Integer variables left unbound default
//! to `i64`, and the error type ([`TyKind::Error`]) unifies with everything,
//! so an error is reported only once

use std::collections::HashSet;

use crate::maps::MapMethod;

use la_arena::{Arena, ArenaMap, Idx};
use vigil_diagnostics::{Diagnostic, Span, codes};

use crate::db::{ConstValue, Db, FnSig, Value, binary_error, eval_binary, out_of_range};
use crate::generics::{EnvArg, GenericArgValue};
use crate::hir::*;
use crate::program::{ContextKind, ProgramKind, unknown_decorator};
use crate::resolve::{Builtin, Def};
use crate::ty::{IntTy, TyId, TyKind};

/// a local, in [`InferenceResult::locals`]
pub type LocalId = Idx<Local>;

/// a variable of a body: a parameter, a name assigned or bound by a
/// pattern, a loop variable or the `result` of `@ensures`
#[derive(Debug)]
pub struct Local {
    pub name: String,
    pub ty: TyId,
    pub span: Span,
}

/// what a name, a path or a call refers to, for the later phases
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Res {
    /// a local of the body
    Local(LocalId),
    /// a constant of the package
    Const(ConstId),
    /// a function called by name
    Fn(FnId),
    /// a method called on a value
    Method(FnId),
    /// a variant of an enum, by index
    Variant(EnumId, u32),
    /// a call that constructs a struct
    StructCtor(StructId),
    /// a field of a struct, by index
    Field(u32),
    /// a builtin of the prelude
    Builtin(Builtin),
    /// the name of a map, as the base of `map[index]`
    Map(MapId),
    /// a method of a map, on the callee
    MapMethod(MapId, MapMethod),
    /// `ctx.parse[H](offset)` on a packet context, on the callee
    PacketParse(StructId),
    /// `ctx.len()` on a packet context, on the callee
    PacketLen,
    /// a field of a kernel struct, an index into `kernel_accesses`
    KernelField(u32),
    /// `kptr[T](address)`, on the callee
    KPtrCast,
    /// `pointer.is_null()` on a kernel pointer, on the callee
    KPtrIsNull,
    /// `ctx.arg(n)` on a kprobe context, on the callee
    KprobeArg(u32),
    /// `ctx.ret()` on a kretprobe context, on the callee
    KprobeRet,
    /// a constant generic parameter, with its value
    ConstValue(i128),
    /// an operator of a struct, on the unary or binary expression: the
    /// method that implements it (`__add__`)
    Operator(FnId),
    /// `[x] * N`, on the binary expression: an array of `N` copies
    Repeat(u64),
}

/// how a field of a kernel struct is reached
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelRead {
    /// copied with `bpf_probe_read_kernel`
    Probe,
    /// loaded directly, for a field of a tracepoint context
    Direct,
    /// not read: the address of an embedded struct, for the next accesses
    Address,
}

/// a field access in a kernel struct, relocated at load time (CO-RE).
/// The offset is the one of the kernel we build for, and the loader fixes
/// it for the running kernel
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelAccess {
    /// the struct or union accessed (BTF type id)
    pub composite: u32,
    /// member indices from `composite` to the field, through anonymous
    /// members
    pub path: Vec<u32>,
    /// offset of the field in the kernel we build for
    pub offset: u32,
    /// how the field is reached
    pub read: KernelRead,
}

/// what we learn from checking a body. Every type in it is resolved, no
/// inference variable is left
#[derive(Debug, Default)]
pub struct InferenceResult {
    /// the type of every expression. A reference keeps its `ref` type here,
    /// `derefs` says if it is read
    pub expr_types: ArenaMap<ExprId, TyId>,
    /// the locals, parameters first and in order
    pub locals: Arena<Local>,
    /// what names, paths, calls and operators refer to
    pub res: ArenaMap<ExprId, Res>,
    /// locals bound by `Bind` patterns
    pub pat_locals: ArenaMap<PatId, LocalId>,
    /// variables of `for` loops
    pub for_locals: ArenaMap<StmtId, LocalId>,
    /// locals declared by assignments
    pub decl_locals: ArenaMap<StmtId, LocalId>,
    /// what the paths of patterns refer to (variants, constants, `Some`)
    pub pat_res: ArenaMap<PatId, Res>,
    /// maximum number of iterations of `@bounded(N)` loops
    pub loop_bounds: ArenaMap<StmtId, u64>,
    /// expressions of reference type whose value is read, that is, copied
    /// out of the map or the packet
    pub derefs: HashSet<ExprId>,
    /// kernel field accesses, indexed by [`Res::KernelField`]
    pub kernel_accesses: Vec<KernelAccess>,
    /// `result` in `@ensures`: a local only the contracts see, left out by
    /// code generation
    pub result_local: Option<LocalId>,
    /// augmented assignments (`a += b`) done by an operator method
    pub aug_operators: ArenaMap<StmtId, FnId>,
    /// the `comptime:` blocks of a run-time function, in order
    pub comptime_blocks: Vec<StmtId>,
    /// where `stage()` is called
    pub stage_calls: Vec<Span>,
    /// the program run by each `run(program, packet)`, keyed by the first
    /// argument, and if the packet is a `Damaged` one
    pub runs: ArenaMap<ExprId, (FnId, bool)>,
    /// calls only tests may make (`run`, `set_time`, `count` and `last` on a
    /// ring buffer), with the name to report
    pub test_calls: Vec<(Span, &'static str)>,
}

/// the most values a `print` may format, because `bpf_printk` takes at
/// most three
const MAX_PRINT_ARGS: usize = 3;

/// the longest format string of a `print`, with its newline and NUL, as the
/// code generator builds it on the stack
const MAX_PRINT_FORMAT: usize = 128;

/// checking the body of a function and its contracts against its
/// signature. Here we also report an `f64` in the signature of a run-time
/// function, a parameter that hides `result` and a body that may end
/// without a value. A `@helper` gives an empty result
pub(crate) fn check_fn(db: &mut Db, id: FnId) -> InferenceResult {
    let package = db.package;
    let data = &package.fns[id];
    db.enter(data.instance.as_ref());
    let sig = db.fn_sigs[id].clone();
    let program = db.programs.get(id).is_some();
    let program_kind = db.programs.get(id).map(|p| p.kind);
    // a helper has no body, the kernel implements it. A misplaced `@helper`
    // is already reported
    if data
        .decorators
        .iter()
        .any(|d| d.name.to_string() == "helper")
    {
        return InferenceResult::default();
    }
    let mut cx = Infer {
        db,
        module: data.module,
        name: &data.name.text,
        body: &data.body,
        spans: &data.spans,
        ret: sig.ret,
        ret_span: data.ret.as_ref().and_then(TypeRef::span),
        program,
        program_kind,
        comptime: data.is_comptime,
        readonly_self: None,
        in_comptime_block: false,
        comptime_locals: HashSet::new(),
        scopes: vec![Vec::new()],
        result: InferenceResult::default(),
        vars: Vec::new(),
        int_literals: Vec::new(),
        negations: Vec::new(),
        loops: 0,
        in_print: false,
        in_contract: false,
        ended: Vec::new(),
        exhaustive: HashSet::new(),
        reported_vars: HashSet::new(),
    };
    let mutating = data.decorators.iter().any(|d| d.name.to_string() == "mut");
    for (i, param) in data.params.iter().enumerate() {
        let local = cx.declare(&param.name, sig.params[i].1);
        if i == 0 && sig.has_self && !mutating {
            cx.readonly_self = Some(local);
        }
    }
    // `@requires` is about the parameters, so we check it before the body
    // declares anything
    for contract in data
        .contracts
        .iter()
        .filter(|c| c.kind == ContractKind::Requires)
    {
        cx.in_contract = true;
        cx.check(contract.expr, cx.db.types.bool);
        cx.in_contract = false;
    }
    if !cx.comptime {
        // here we point at the first written type with an `f64`
        let written = data
            .params
            .iter()
            .map(|p| p.ty.as_ref())
            .chain([data.ret.as_ref()]);
        let floats = sig.params.iter().map(|p| p.1).chain([sig.ret]);
        let float = floats.zip(written).find(|(t, _)| cx.db.types.has_float(*t));
        if let Some((_, written)) = float {
            let span = written.and_then(TypeRef::span).unwrap_or(data.name.span);
            let mut diag = Diagnostic::error("`f64` is only allowed in comptime functions")
                .with_code(codes::NOT_SUPPORTED_YET)
                .with_primary(span, "the verifier rejects floating point")
                .with_help(
                    "mark the function `comptime def`, or use `fixed[I, F]` from `std.fixed`",
                );
            if span != data.name.span {
                diag = diag.with_secondary(data.name.span, "a run-time function");
            }
            cx.db.diags.push(diag);
        }
    }
    let diverges = cx.block(&data.body.root);
    let unit = cx.db.types.unit;
    // `@ensures` is checked after the body, in its own scope, so only it
    // sees `result`
    let ensures: Vec<_> = data
        .contracts
        .iter()
        .filter(|c| c.kind == ContractKind::Ensures)
        .collect();
    if !ensures.is_empty() {
        let shadowed = data.params.iter().find(|p| p.name.text == "result");
        if let Some(param) = shadowed {
            cx.db.diags.push(
                Diagnostic::error("a parameter named `result` hides the result in `@ensures`")
                    .with_code(codes::BAD_CONTRACT)
                    .with_primary(param.name.span, "rename this parameter"),
            );
        }
        cx.scoped(|cx| {
            if sig.ret != unit && shadowed.is_none() {
                let name = Ident {
                    text: "result".into(),
                    span: ensures[0].span,
                };
                let local = cx.declare(&name, sig.ret);
                cx.result.result_local = Some(local);
            }
            cx.in_contract = true;
            for contract in &ensures {
                cx.check(contract.expr, cx.db.types.bool);
            }
            cx.in_contract = false;
        });
    }
    if !diverges && sig.ret != unit && !cx.db.types.is_error(sig.ret) {
        let ret = cx.show(sig.ret);
        let span = data
            .ret
            .as_ref()
            .and_then(TypeRef::span)
            .unwrap_or(data.name.span);
        cx.db.diags.push(
            Diagnostic::error(format!("`{}` may end without returning", data.name.text))
                .with_code(codes::MISSING_RETURN)
                .with_primary(span, format!("returns `{ret}`"))
                .with_secondary(data.name.span, "some paths reach the end of this function")
                .with_help("end every path with `return value`"),
        );
    }
    cx.finish()
}

/// an inference variable, unbound or bound to a type (maybe a variable)
#[derive(Clone, Copy)]
enum Var {
    /// not solved yet. `int` is set when it can only be an integer type
    Unbound { int: bool },
    /// solved
    Bound(TyId),
}

/// what a path denotes when it is not a value: a namespace, a type or a
/// member of one
#[derive(Clone, Copy, Debug)]
enum Path {
    /// a module or directory
    Namespace,
    Struct(StructId),
    Enum(EnumId),
    Builtin(Builtin),
    Fn(FnId),
    Const(ConstId),
    Variant(EnumId, u32),
    /// a method used through its type (`Proto.of`)
    Method(FnId),
    Map(MapId),
    /// a kernel struct or union, by BTF type id
    Kernel(u32),
    Protocol,
    /// a constant generic parameter: its value and type
    Value(i128, TyId),
    /// a path whose error is already reported
    Error,
}

/// the state of checking one body
struct Infer<'a, 'p> {
    db: &'a mut Db<'p>,
    /// the module of the function, where names are looked up
    module: ModuleId,
    /// the name of the function, for messages
    name: &'p str,
    body: &'p Body,
    spans: &'p BodySpans,
    /// the return type of the function
    ret: TyId,
    /// where the return type is written, if it is
    ret_span: Option<Span>,
    /// if the function is an eBPF program
    program: bool,
    /// the program type, for a program
    program_kind: Option<crate::program::ProgramKind>,
    /// if the function runs in the compiler (`comptime def`), where `f64` is
    /// allowed
    comptime: bool,
    /// `self` in a method without `@mut`, which cannot change it
    readonly_self: Option<LocalId>,
    /// if we are checking a `comptime:` block of a run-time function
    in_comptime_block: bool,
    /// variables of the `comptime:` blocks, constants for the code after them
    comptime_locals: HashSet<LocalId>,
    /// names in scope, innermost block last, each block in declaration order
    scopes: Vec<Vec<(String, LocalId)>>,
    result: InferenceResult,
    /// the inference variables, by number
    vars: Vec<Var>,
    /// integer literals, range-checked once their type is known
    int_literals: Vec<(ExprId, i128)>,
    /// negated expressions, whose type must end up signed
    negations: Vec<ExprId>,
    /// number of loops around the current statement, for `break` and
    /// `continue`
    loops: u32,
    /// if we are in the argument of `print`, the only place for strings
    in_print: bool,
    /// if we are in a contract or an invariant, the only places for
    /// `is_some`, `is_none` and `value`
    in_contract: bool,
    /// variables of ended blocks, to explain why they are not found
    ended: Vec<(String, Span)>,
    /// `match` statements found exhaustive, for the return analysis
    exhaustive: HashSet<StmtId>,
    /// variables already reported as impossible to infer. A variable is
    /// shared by all expressions unified with it, so we report it once
    reported_vars: HashSet<u32>,
}

impl<'p> Infer<'_, 'p> {
    // inference variables
    // -------------------------

    /// a new inference variable, an integer one when `int` is set
    fn var(&mut self, int: bool) -> TyId {
        let n = self.vars.len() as u32;
        self.vars.push(Var::Unbound { int });
        self.db.types.intern(TyKind::Var(n))
    }

    /// following the bound variables at the top of a type. The result is a
    /// variable only when unbound, inner variables may remain
    fn shallow(&self, ty: TyId) -> TyId {
        let mut ty = ty;
        while let TyKind::Var(v) = self.db.types.kind(ty) {
            match self.vars[*v as usize] {
                Var::Bound(t) => ty = t,
                Var::Unbound { .. } => break,
            }
        }
        ty
    }

    /// the kind of a type, after following the variables at its top
    fn kind(&self, ty: TyId) -> TyKind {
        self.db.types.kind(self.shallow(ty)).clone()
    }

    /// unification of two types, returning if it succeeded. The caller
    /// reports a failure.
    ///
    /// The error type unifies with anything. An integer variable unifies only
    /// with an integer type or another variable, which becomes an integer one.
    /// We never bind a variable to a type that contains it
    fn unify(&mut self, a: TyId, b: TyId) -> bool {
        let (a, b) = (self.shallow(a), self.shallow(b));
        if a == b {
            return true;
        }
        match (self.db.types.kind(a).clone(), self.db.types.kind(b).clone()) {
            (TyKind::Error, _) | (_, TyKind::Error) => true,
            (TyKind::Var(x), TyKind::Var(y)) => {
                let int = matches!(self.vars[x as usize], Var::Unbound { int: true })
                    || matches!(self.vars[y as usize], Var::Unbound { int: true });
                self.vars[y as usize] = Var::Unbound { int };
                self.vars[x as usize] = Var::Bound(b);
                true
            }
            (TyKind::Var(x), other) | (other, TyKind::Var(x)) => {
                let target = if matches!(self.db.types.kind(a), TyKind::Var(_)) {
                    b
                } else {
                    a
                };
                if matches!(self.vars[x as usize], Var::Unbound { int: true })
                    && !matches!(other, TyKind::Int(_))
                {
                    return false;
                }
                if self.occurs(x, target) {
                    return false;
                }
                self.vars[x as usize] = Var::Bound(target);
                true
            }
            (TyKind::Option(x), TyKind::Option(y)) | (TyKind::Ref(x), TyKind::Ref(y)) => {
                self.unify(x, y)
            }
            (TyKind::Array(x, n), TyKind::Array(y, m)) => n == m && self.unify(x, y),
            _ => false,
        }
    }

    /// occurs check, because binding a variable to a type that contains it
    /// would make an infinite type
    fn occurs(&self, var: u32, ty: TyId) -> bool {
        match self.kind(ty) {
            TyKind::Var(v) => v == var,
            TyKind::Option(inner) | TyKind::Array(inner, _) | TyKind::Ref(inner) => {
                self.occurs(var, inner)
            }
            _ => false,
        }
    }

    /// replacing every variable by its binding. An unbound integer variable
    /// is bound to `i64` for good, other unbound variables become the error
    /// type
    fn zonk(&mut self, ty: TyId) -> TyId {
        let ty = self.shallow(ty);
        match self.db.types.kind(ty).clone() {
            TyKind::Var(v) => match self.vars[v as usize] {
                Var::Unbound { int: true } => {
                    let i64 = self.db.types.int(IntTy::I64);
                    self.vars[v as usize] = Var::Bound(i64);
                    i64
                }
                _ => self.db.types.error,
            },
            TyKind::Option(inner) => {
                let inner = self.zonk(inner);
                self.db.types.option(inner)
            }
            TyKind::Array(inner, len) => {
                let inner = self.zonk(inner);
                self.db.types.array(inner, len)
            }
            TyKind::Ref(inner) => {
                let inner = self.zonk(inner);
                self.db.types.intern(TyKind::Ref(inner))
            }
            _ => ty,
        }
    }

    /// an unbound, non-integer variable still in `ty`, if any
    fn unknown_var(&self, ty: TyId) -> Option<u32> {
        match self.kind(ty) {
            TyKind::Var(v) => match self.vars[v as usize] {
                Var::Unbound { int: false } => Some(v),
                _ => None,
            },
            TyKind::Option(inner) | TyKind::Array(inner, _) | TyKind::Ref(inner) => {
                self.unknown_var(inner)
            }
            _ => None,
        }
    }

    /// showing a type in a message. An unbound integer variable is shown as
    /// `{integer}` and any other as `_`
    fn show(&self, ty: TyId) -> String {
        let resolved = |v: u32| -> Option<String> {
            match self.vars[v as usize] {
                Var::Bound(t) => Some(self.show(t)),
                Var::Unbound { int: true } => Some("{integer}".into()),
                Var::Unbound { int: false } => None,
            }
        };
        self.db.types.display(self.db.package, ty, &resolved)
    }

    /// finishing the body: resolving the types and reporting what had to
    /// wait for them (variables that cannot be inferred, literals out of range,
    /// negations of unsigned values)
    fn finish(mut self) -> InferenceResult {
        let ids: Vec<ExprId> = self.result.expr_types.iter().map(|(id, _)| id).collect();
        for id in ids {
            let ty = self.result.expr_types[id];
            if let Some(v) = self.unknown_var(ty)
                && self.reported_vars.insert(v)
            {
                // a variable is named and can be annotated, other expressions
                // get their type from where they are used
                let diag = match &self.body.exprs[id] {
                    Expr::Name(name) => {
                        // an `Option` of unknown type is the usual case: `x = None`
                        let example = if self.show(ty) == "_?" { "u32?" } else { "T" };
                        Diagnostic::error(format!("cannot infer the type of `{name}`"))
                            .with_primary(self.span(id), "nothing here tells its type")
                            .with_help(format!("give it a type, as in `{name}: {example} = ...`"))
                    }
                    _ => Diagnostic::error("cannot infer the type of this expression")
                        .with_primary(self.span(id), "type annotations needed"),
                };
                self.error(diag.with_code(codes::CANNOT_INFER));
            }
            let zonked = self.zonk(ty);
            self.result.expr_types.insert(id, zonked);
        }
        let locals: Vec<LocalId> = self.result.locals.iter().map(|(id, _)| id).collect();
        for id in locals {
            let ty = self.result.locals[id].ty;
            let zonked = self.zonk(ty);
            self.result.locals[id].ty = zonked;
        }
        for &(expr, value) in &self.int_literals {
            // a negative literal of an unsigned type is reported as a
            // negation below
            if let TyKind::Int(int) = *self.db.types.kind(self.result.expr_types[expr])
                && !int.contains(value)
                && (value >= 0 || int.signed())
            {
                let span = self.spans.exprs[expr];
                // a literal above `i128::MAX` was kept as `i128::MAX`: show
                // the number as written
                let shown = match self.body.exprs[expr] {
                    Expr::Int(written) if value == i128::MAX => written.to_string(),
                    _ => value.to_string(),
                };
                self.db.diags.push(out_of_range(shown, int, span));
            }
        }
        for &expr in &self.negations {
            if let TyKind::Int(int) = *self.db.types.kind(self.result.expr_types[expr])
                && !int.signed()
            {
                let span = self.spans.exprs[expr];
                self.db.diags.push(
                    Diagnostic::error(format!("cannot negate a value of type `{}`", int.name()))
                        .with_code(codes::BAD_OPERAND)
                        .with_primary(span, "unsigned integers have no negative values"),
                );
            }
        }
        self.result
    }

    // helper functions
    // -------------------------

    /// the span of an expression
    fn span(&self, expr: ExprId) -> Span {
        self.spans.exprs[expr]
    }

    /// reporting a diagnostic, returning the error type for the expression
    fn error(&mut self, diagnostic: Diagnostic) -> TyId {
        self.db.diags.push(diagnostic);
        self.db.types.error
    }

    /// reporting a type mismatch
    fn mismatch(&mut self, span: Span, expected: TyId, found: TyId) {
        self.mismatch_with(span, expected, found, None);
    }

    /// reporting a type mismatch, with a label on what fixed the expected
    /// type (a return type, a declaration). Between integers, or from a
    /// `bool` to an integer, we suggest the conversion
    fn mismatch_with(
        &mut self,
        span: Span,
        expected: TyId,
        found: TyId,
        because: Option<(Span, String)>,
    ) {
        let convertible = matches!(self.kind(expected), TyKind::Int(_))
            && matches!(self.kind(found), TyKind::Int(_) | TyKind::Bool);
        let (expected, found) = (self.show(expected), self.show(found));
        let mut diag = Diagnostic::error("mismatched types")
            .with_code(codes::MISMATCHED_TYPES)
            .with_primary(span, format!("expected `{expected}`, found `{found}`"));
        if let Some((span, label)) = because {
            diag = diag.with_secondary(span, label);
        }
        if convertible {
            diag = diag.with_help(format!("convert the value with `{expected}(...)`"));
        }
        self.error(diag);
    }

    /// a new local in the innermost scope. It hides any local with the same
    /// name for the rest of the scope
    fn declare(&mut self, name: &Ident, ty: TyId) -> LocalId {
        let id = self.result.locals.alloc(Local {
            name: name.text.clone(),
            ty,
            span: name.span,
        });
        if self.in_comptime_block {
            self.comptime_locals.insert(id);
        }
        self.scopes
            .last_mut()
            .unwrap()
            .push((name.text.clone(), id));
        id
    }

    /// looking up a local: the latest one with that name, innermost scope
    /// first
    fn lookup_local(&self, name: &str) -> Option<LocalId> {
        self.scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().rev())
            .find(|(n, _)| n == name)
            .map(|&(_, id)| id)
    }

    /// running `f` in a new scope. Its names are remembered as ended when it
    /// returns
    fn scoped<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.scopes.push(Vec::new());
        let result = f(self);
        for (name, local) in self.scopes.pop().unwrap() {
            let span = self.result.locals[local].span;
            self.ended.push((name, span));
        }
        result
    }

    /// if a type is, or can only become, an integer type. The error type
    /// counts, so we do not report it again
    fn is_int(&self, ty: TyId) -> bool {
        match self.kind(ty) {
            TyKind::Int(_) | TyKind::Error => true,
            TyKind::Var(v) => matches!(self.vars[v as usize], Var::Unbound { int: true }),
            _ => false,
        }
    }

    // paths
    // -------------------------

    /// the segments of `a.b.c` when its root is not a local, so it may be a
    /// path to an item and not fields of a value
    fn expr_segments(&self, expr: ExprId) -> Option<Vec<Ident>> {
        match &self.body.exprs[expr] {
            Expr::Name(name) if self.lookup_local(name).is_none() => Some(vec![Ident {
                text: name.clone(),
                span: self.span(expr),
            }]),
            Expr::Field { base, name } => {
                let mut segments = self.expr_segments(*base)?;
                segments.push(name.clone());
                Some(segments)
            }
            _ => None,
        }
    }

    /// resolving a path that does not start with a local, reporting errors
    /// in its members. `None` when the root names nothing (the caller reports
    /// it) or the path goes into a value, as `CONST.x` (the caller checks it as
    /// field accesses)
    fn resolve_path(&mut self, segments: &[Ident]) -> Option<Path> {
        match self.lookup_path(segments)? {
            Ok(path) => Some(path),
            Err(diagnostic) => {
                self.db.diags.push(diagnostic);
                Some(Path::Error)
            }
        }
    }

    /// [`Self::resolve_path`] without reporting, used to probe a path before
    /// deciding how to check an expression
    fn lookup_path(&mut self, segments: &[Ident]) -> Option<Result<Path, Diagnostic>> {
        let package = self.db.package;
        // a generic parameter of the instance being checked hides any item with
        // the same name
        if let Some(first) = segments.first()
            && let Some(arg) = self.db.env_lookup(&first.text)
        {
            let path = match arg {
                EnvArg::Const(v, ty) => Path::Value(v, ty),
                EnvArg::Type(ty) => self.path_of_type(ty),
            };
            return self.member_path(path, &segments[..1], &segments[1..]);
        }
        let (def, used) = self
            .db
            .scopes
            .resolve_prefix(package, self.module, segments)?;
        let path = match def {
            Def::Namespace(_) => Path::Namespace,
            Def::Item(Item::Struct(s)) => Path::Struct(s),
            Def::Item(Item::Enum(e)) => Path::Enum(e),
            Def::Item(Item::Fn(f)) => Path::Fn(f),
            Def::Item(Item::Const(c)) => Path::Const(c),
            Def::Item(Item::Map(m)) => Path::Map(m),
            Def::Item(Item::Protocol(_)) => Path::Protocol,
            Def::Kernel(id) => Path::Kernel(id),
            Def::Item(Item::Alias(a)) => {
                let ty = self.db.alias_ty(a);
                match self.db.types.kind(ty) {
                    TyKind::Struct(s) => Path::Struct(*s),
                    TyKind::Enum(e) => Path::Enum(*e),
                    _ => Path::Error,
                }
            }
            Def::Builtin(b) => Path::Builtin(b),
            Def::Error => Path::Error,
        };
        self.member_path(path, &segments[..used], &segments[used..])
    }

    /// resolving the members `segments` of `path`: variants and methods of
    /// types. `prefix` is how `path` is written, for messages
    fn member_path(
        &mut self,
        mut path: Path,
        prefix: &[Ident],
        segments: &[Ident],
    ) -> Option<Result<Path, Diagnostic>> {
        let package = self.db.package;
        for segment in segments {
            let not_found = |message: String, label: &str| {
                Diagnostic::error(message)
                    .with_code(codes::UNRESOLVED_NAME)
                    .with_primary(segment.span, label.to_string())
            };
            path = match path {
                Path::Enum(e) => {
                    let data = &package.enums[e];
                    if let Some(i) = data
                        .variants
                        .iter()
                        .position(|v| v.name.text == segment.text)
                    {
                        Path::Variant(e, i as u32)
                    } else if let Some(&m) = data
                        .methods
                        .iter()
                        .find(|&&m| package.fns[m].name.text == segment.text)
                    {
                        Path::Method(m)
                    } else {
                        return Some(Err(not_found(
                            format!(
                                "no variant or method `{}` in `{}`",
                                segment.text, data.name.text
                            ),
                            "not found",
                        )));
                    }
                }
                Path::Struct(s) => {
                    let data = &package.structs[s];
                    match data
                        .methods
                        .iter()
                        .find(|&&m| package.fns[m].name.text == segment.text)
                    {
                        Some(&m) => Path::Method(m),
                        None => {
                            return Some(Err(not_found(
                                format!("no method `{}` in `{}`", segment.text, data.name.text),
                                "not found",
                            )
                            .with_note("fields are accessed on values, not on the type")));
                        }
                    }
                }
                // only a prefix of the path is a module
                Path::Namespace => {
                    let module: Vec<_> = prefix.iter().map(|s| s.text.as_str()).collect();
                    return Some(Err(not_found(
                        format!(
                            "`{}` not found in module `{}`",
                            segment.text,
                            module.join(".")
                        ),
                        "not found",
                    )));
                }
                Path::Error => Path::Error,
                // a member of a value (`CONST.x`) is a field access, not a path
                _ => return None,
            };
        }
        Some(Ok(path))
    }

    /// what a type stands for in an expression (`T(x)`, `T.zero()`)
    fn path_of_type(&self, ty: TyId) -> Path {
        match self.db.types.kind(ty) {
            TyKind::Int(int) => Path::Builtin(Builtin::Int(*int)),
            TyKind::Bool => Path::Builtin(Builtin::Bool),
            TyKind::Struct(s) => Path::Struct(*s),
            TyKind::Enum(e) => Path::Enum(*e),
            _ => Path::Error,
        }
    }

    /// resolving `Generic[args]` in an expression to its instance. `None`
    /// when the expression is not of that form or the item is not generic.
    ///
    /// An instance not in the package yet is requested from the `Db` and we
    /// give `Path::Error`. The analysis runs again once it is added, and until
    /// then its uses have the error type and report nothing
    fn instance_path(&mut self, expr: ExprId) -> Option<Path> {
        let Expr::Index { base, indices } = &self.body.exprs[expr] else {
            return None;
        };
        let generic = match self.expr_segments(*base).and_then(|s| self.lookup_path(&s)) {
            Some(Ok(Path::Struct(s))) if self.db.is_template_struct(s) => Item::Struct(s),
            Some(Ok(Path::Enum(e))) if self.db.is_template_enum(e) => Item::Enum(e),
            Some(Ok(Path::Fn(f))) if self.db.is_template_fn(f) => Item::Fn(f),
            _ => return None,
        };
        let indices = indices.clone();
        let mut args = Vec::new();
        for &index in &indices {
            let span = self.span(index);
            args.push((self.generic_arg(index), span));
        }
        let span = self.span(expr);
        let Some(args) = self.db.check_generic_args(generic, args, span) else {
            return Some(Path::Error);
        };
        Some(match self.db.instantiate(generic, args) {
            Some(Item::Struct(s)) => Path::Struct(s),
            Some(Item::Enum(e)) => Path::Enum(e),
            Some(Item::Fn(f)) => Path::Fn(f),
            _ => Path::Error,
        })
    }

    /// a generic argument in an expression: a type (named or an instance) or
    /// a constant expression
    fn generic_arg(&mut self, expr: ExprId) -> GenericArgValue {
        if let Some(path) = self.instance_path(expr) {
            return match path {
                Path::Struct(s) => GenericArgValue::Type(self.db.types.intern(TyKind::Struct(s))),
                Path::Enum(e) => GenericArgValue::Type(self.db.types.intern(TyKind::Enum(e))),
                _ => GenericArgValue::Error,
            };
        }
        if let Some(segments) = self.expr_segments(expr) {
            match self.lookup_path(&segments) {
                Some(Ok(Path::Struct(s))) => {
                    return GenericArgValue::Type(self.db.types.intern(TyKind::Struct(s)));
                }
                Some(Ok(Path::Enum(e))) => {
                    return GenericArgValue::Type(self.db.types.intern(TyKind::Enum(e)));
                }
                Some(Ok(Path::Builtin(Builtin::Int(int)))) => {
                    return GenericArgValue::Type(self.db.types.int(int));
                }
                Some(Ok(Path::Builtin(Builtin::Bool))) => {
                    return GenericArgValue::Type(self.db.types.bool);
                }
                Some(Ok(Path::Error)) => return GenericArgValue::Error,
                _ => {}
            }
            // an alias names a type too
            if let Some((Def::Item(Item::Alias(a)), used)) =
                self.db
                    .scopes
                    .resolve_prefix(self.db.package, self.module, &segments)
                && used == segments.len()
            {
                return GenericArgValue::Type(self.db.alias_ty(a));
            }
        }
        match self.const_of(expr) {
            Ok(v) => GenericArgValue::Const(v),
            Err(true) => GenericArgValue::Error,
            Err(false) => {
                self.error(
                    Diagnostic::error("expected a type or a constant")
                        .with_code(codes::WRONG_KIND)
                        .with_primary(self.span(expr), "not a generic argument"),
                );
                GenericArgValue::Error
            }
        }
    }

    /// evaluating a constant expression (integer literals, constants,
    /// constant generic parameters and operators) as `const` does. An
    /// overflow, a division by zero or a bad shift is reported here and
    /// gives `Err(true)`. `Err(false)` is an expression that is not a
    /// constant, which the caller reports
    fn const_of(&mut self, expr: ExprId) -> Result<i128, bool> {
        let invalid = |cx: &mut Self, message: &str| {
            cx.error(
                Diagnostic::error(message.to_string())
                    .with_code(codes::CONST_EVAL)
                    .with_primary(cx.span(expr), "in this constant expression"),
            );
            Err(true)
        };
        match &self.body.exprs[expr] {
            Expr::Int(v) => match i128::try_from(*v) {
                Ok(v) => Ok(v),
                Err(_) => invalid(self, "integer is too large"),
            },
            Expr::Unary {
                op: UnaryOp::Neg,
                operand,
            } => match self.const_of(*operand)?.checked_neg() {
                Some(v) => Ok(v),
                None => invalid(self, "arithmetic overflow"),
            },
            Expr::Binary { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                let (a, b) = (self.const_of(lhs)?, self.const_of(rhs)?);
                match eval_binary(op, Value::Int(a), Value::Int(b)) {
                    Ok(Value::Int(v)) => Ok(v),
                    Ok(_) => Err(false),
                    // the same error as in a `const`, as in "the divisor is 0"
                    Err(message) => {
                        let kinds = ("an integer", "an integer");
                        let span = self.span(expr);
                        self.error(binary_error(op, kinds, &Value::Int(b), &message, span));
                        Err(true)
                    }
                }
            }
            _ => {
                let segments = self.expr_segments(expr).ok_or(false)?;
                match self.lookup_path(&segments).ok_or(false)? {
                    Ok(Path::Value(v, _)) => Ok(v),
                    Ok(Path::Const(c)) => match self.db.const_info(c).1 {
                        Some(ConstValue::Int(v)) => Ok(v),
                        _ => Err(false),
                    },
                    _ => Err(false),
                }
            }
        }
    }

    /// reporting a name that refers to nothing, saying when it was declared
    /// in a block that has ended
    fn unresolved(&mut self, name: &str, span: Span) -> TyId {
        let mut diag = Diagnostic::error(format!("cannot find `{name}` in this scope"))
            .with_code(codes::UNRESOLVED_NAME)
            .with_primary(span, "not found");
        if let Some((_, declared)) = self.ended.iter().rev().find(|(n, _)| n == name) {
            diag = diag
                .with_secondary(*declared, "declared here, in a block that has ended")
                .with_help(format!(
                    "variables live until the end of their block; assign `{name}` before the block to use it after"
                ));
        }
        self.error(diag)
    }

    /// reporting a path used as a value when it is something else
    fn not_a_value(&mut self, path: Path, span: Span) -> TyId {
        let (what, help) = match path {
            Path::Namespace => ("a module", "use one of its items"),
            Path::Struct(_) | Path::Builtin(_) => ("a type", "construct a value by calling it"),
            Path::Enum(_) => ("a type", "use one of its variants"),
            Path::Fn(_) | Path::Method(_) => ("a function", "call it"),
            Path::Map(_) => ("a map", "use its methods, as in `name.get(key)`"),
            Path::Protocol => ("a protocol", "protocols bound generic parameters"),
            Path::Kernel(_) => (
                "a kernel type",
                "reach kernel structs through `kptr[T](address)`",
            ),
            _ => ("something else", ""),
        };
        let mut diag = Diagnostic::error(format!("expected a value, found {what}"))
            .with_code(codes::WRONG_KIND)
            .with_primary(span, "not a value");
        if !help.is_empty() {
            diag = diag.with_help(help);
        }
        self.error(diag)
    }

    // statements
    // -------------------------

    /// checking a block in a new scope. Returns if it always diverges
    /// (returns, breaks or continues)
    fn block(&mut self, block: &Block) -> bool {
        self.scoped(|cx| cx.stmts(block))
    }

    /// checking the statements of a block in the current scope. Returns if
    /// one of them always diverges
    fn stmts(&mut self, block: &Block) -> bool {
        let mut diverges = false;
        for &stmt in block {
            diverges |= self.stmt(stmt);
        }
        diverges
    }

    /// checking a statement, returning if it always diverges. A loop never
    /// does, because its body may not run at all
    fn stmt(&mut self, id: StmtId) -> bool {
        let span = self.spans.stmts[id];
        match &self.body.stmts[id] {
            Stmt::Missing | Stmt::Pass => false,
            Stmt::Expr(expr) => {
                if let Expr::Ellipsis = self.body.exprs[*expr] {
                    self.error(
                        Diagnostic::error("`...` is only allowed in protocol methods")
                            .with_code(codes::NOT_SUPPORTED_YET)
                            .with_primary(span, "use `pass` for an empty block"),
                    );
                } else {
                    self.infer(*expr, None);
                }
                false
            }
            Stmt::Assign { target, ty, value } => {
                self.assign(id, *target, ty.as_ref(), *value, span);
                false
            }
            Stmt::AugAssign { target, op, value } => {
                let target_ty = self.place(*target, None);
                if let Some(method) = self.operator_method(target_ty, binary_method(*op)) {
                    let ret = self.operator_call(method, Some(*value), span);
                    if !self.unify(ret, target_ty) {
                        self.mismatch(span, target_ty, ret);
                    }
                    self.result.aug_operators.insert(id, method);
                    return false;
                }
                // same rules as `target op value`: the value has the type of the
                // target, except the amount of a shift, which is any integer
                if let BinaryOp::Shl | BinaryOp::Shr = op {
                    let amount = self.infer(*value, None);
                    if !self.is_int(amount) {
                        self.bad_operand(op.symbol(), amount, self.span(*value));
                    }
                } else if self.operator_applies(*op, target_ty) {
                    self.check(*value, target_ty);
                } else {
                    // the target is the error, the value is not checked against it,
                    // only unified to infer what it can
                    let value_ty = self.infer(*value, Some(target_ty));
                    self.unify(value_ty, target_ty);
                }
                self.binary_result(*op, target_ty, span, self.span(*target));
                false
            }
            Stmt::Return(_) if self.in_comptime_block => {
                self.error(
                    Diagnostic::error("`return` cannot leave a comptime block")
                        .with_code(codes::CONST_EVAL)
                        .with_primary(span, "the block runs in the compiler"),
                );
                true
            }
            Stmt::Return(value) => {
                let unit = self.db.types.unit;
                match value {
                    Some(value) => {
                        if self.ret == unit {
                            self.infer(*value, None);
                            let name = self.name;
                            self.error(
                                Diagnostic::error(format!("`{name}` returns nothing"))
                                    .with_code(codes::MISMATCHED_TYPES)
                                    .with_primary(self.span(*value), "unexpected value")
                                    .with_help(format!(
                                        "declare its return type, as in `def {name}(...) -> u32:`"
                                    )),
                            );
                        } else {
                            let because = self
                                .ret_span
                                .map(|span| (span, format!("the return type of `{}`", self.name)));
                            self.check_with(*value, self.ret, because);
                        }
                    }
                    None if self.ret != unit && !self.db.types.is_error(self.ret) => {
                        let ret = self.show(self.ret);
                        let mut diag = Diagnostic::error("missing return value")
                            .with_code(codes::MISMATCHED_TYPES)
                            .with_primary(span, format!("expected a value of type `{ret}`"));
                        if let Some(ret_span) = self.ret_span {
                            diag = diag.with_secondary(
                                ret_span,
                                format!("the return type of `{}`", self.name),
                            );
                        }
                        self.error(diag);
                    }
                    None => {}
                }
                true
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                self.check(*cond, self.db.types.bool);
                let then = self.block(then);
                let otherwise = otherwise.as_ref().is_some_and(|b| self.block(b));
                then && otherwise
            }
            Stmt::For {
                decorators,
                invariants,
                binding,
                iter,
                body,
            } => {
                self.loop_decorators(id, decorators, false, span);
                let element = self.iterable(*iter);
                self.loops += 1;
                self.scoped(|cx| {
                    let local = cx.declare(binding, element);
                    cx.result.for_locals.insert(id, local);
                    // an invariant holds at the start of each iteration, so it sees the
                    // loop variable and what is before the loop, but nothing of the body
                    cx.in_contract = true;
                    for &invariant in invariants {
                        cx.check(invariant, cx.db.types.bool);
                    }
                    cx.in_contract = false;
                    cx.stmts(body);
                });
                self.loops -= 1;
                false
            }
            Stmt::While {
                decorators,
                invariants,
                cond,
                body,
            } => {
                let cond_span = self.span(*cond);
                self.loop_decorators(id, decorators, true, cond_span);
                self.in_contract = true;
                for &invariant in invariants {
                    self.check(invariant, self.db.types.bool);
                }
                self.in_contract = false;
                self.check(*cond, self.db.types.bool);
                self.loops += 1;
                self.block(body);
                self.loops -= 1;
                false
            }
            Stmt::Match { scrutinee, arms } => {
                let ty = self.infer(*scrutinee, None);
                let mut all_diverge = true;
                for arm in arms {
                    all_diverge &= self.scoped(|cx| {
                        cx.pat(arm.pat, ty, false);
                        if let Some(guard) = arm.guard {
                            cx.check(guard, cx.db.types.bool);
                        }
                        cx.stmts(&arm.body)
                    });
                }
                let exhaustive = self.exhaustiveness(ty, arms, self.span(*scrutinee));
                if exhaustive {
                    self.exhaustive.insert(id);
                }
                // exhaustiveness does not count here: a non-exhaustive match is already
                // reported, and this avoids a second error about a missing return
                all_diverge && !arms.is_empty()
            }
            Stmt::Break | Stmt::Continue => {
                if self.loops == 0 {
                    let keyword = match self.body.stmts[id] {
                        Stmt::Break => "break",
                        _ => "continue",
                    };
                    self.error(
                        Diagnostic::error(format!("`{keyword}` outside of a loop"))
                            .with_code(codes::OUTSIDE_LOOP)
                            .with_primary(span, "not in a `for` or `while` loop"),
                    );
                }
                true
            }
            // a `comptime:` block runs in the compiler. It has no scope of its own,
            // because its variables are constants of the code after it. In a comptime
            // function it is plain code
            Stmt::Comptime(block) if self.comptime => self.stmts(block),
            Stmt::Comptime(block) => {
                let (comptime, inside, loops) = (self.comptime, self.in_comptime_block, self.loops);
                self.comptime = true;
                self.in_comptime_block = true;
                self.loops = 0;
                self.stmts(block);
                self.comptime = comptime;
                self.in_comptime_block = inside;
                self.loops = loops;
                self.result.comptime_blocks.push(id);
                false
            }
        }
    }

    /// checking an assignment. To a name not in scope it declares a local,
    /// typed by the annotation or by the value. Otherwise it is a store,
    /// checked against the type of the place
    fn assign(
        &mut self,
        stmt: StmtId,
        target: ExprId,
        ty: Option<&TypeRef>,
        value: Option<ExprId>,
        span: Span,
    ) {
        let Some(value) = value else {
            self.error(
                Diagnostic::error("a declaration needs an initial value")
                    .with_code(codes::BAD_ASSIGNMENT)
                    .with_primary(span, "add `= value`"),
            );
            // the variable is still declared, with its written type, so its
            // uses report nothing more
            if let Expr::Name(name) = &self.body.exprs[target]
                && self.lookup_local(name).is_none()
            {
                let declared = match ty {
                    Some(ty) => {
                        let declared = self.db.lower_ty(self.module, ty);
                        self.db.value_type(declared, ty.span())
                    }
                    None => self.db.types.error,
                };
                let ident = Ident {
                    text: name.clone(),
                    span: self.span(target),
                };
                let local = self.declare(&ident, declared);
                self.result.res.insert(target, Res::Local(local));
                self.result.expr_types.insert(target, declared);
            }
            return;
        };
        let name = match &self.body.exprs[target] {
            Expr::Name(name) => Some(name.clone()),
            _ => None,
        };
        let Some(name) = name else {
            if let Some(ty) = ty {
                self.error(
                    Diagnostic::error("only variables can have type annotations")
                        .with_code(codes::BAD_ASSIGNMENT)
                        .with_primary(ty.span().unwrap_or(span), "unexpected annotation"),
                );
            }
            let target_ty = self.place(target, Some(value));
            self.check(value, target_ty);
            return;
        };
        let ident = Ident {
            text: name.clone(),
            span: self.span(target),
        };
        let existing = self.lookup_local(&name);
        match (existing, ty) {
            (Some(local), None) => {
                if self.in_comptime_block && !self.comptime_locals.contains(&local) {
                    self.error(
                        Diagnostic::error(format!("`{name}` is not known at compile time"))
                            .with_code(codes::CONST_EVAL)
                            .with_primary(ident.span, "a variable of the run-time code")
                            .with_help("comptime blocks change only their own variables"),
                    );
                }
                self.result.res.insert(target, Res::Local(local));
                let local_ty = self.result.locals[local].ty;
                self.result.expr_types.insert(target, local_ty);
                // a variable bound to a reference writes through it, so the value has
                // the referenced type
                let value_ty = self.strip_ref(local_ty);
                let declared = self.result.locals[local].span;
                self.check_with(
                    value,
                    value_ty,
                    Some((declared, format!("`{name}` is declared here"))),
                );
            }
            (Some(local), Some(ty)) => {
                let first = self.result.locals[local].span;
                self.error(
                    Diagnostic::error(format!("`{name}` is already defined"))
                        .with_code(codes::DUPLICATE_DEFINITION)
                        .with_primary(
                            ty.span().unwrap_or(span),
                            "annotations only declare new variables",
                        )
                        .with_secondary(first, "defined here")
                        .with_help(format!(
                            "to change `{name}`, assign it without the annotation; to declare another variable, choose another name"
                        )),
                );
                let local_ty = self.result.locals[local].ty;
                self.check(value, local_ty);
            }
            (None, _) => {
                if let Some(Ok(Path::Const(c))) = self.lookup_path(std::slice::from_ref(&ident)) {
                    let defined = self.db.package.consts[c].name.span;
                    let mut diag = Diagnostic::error(format!("cannot assign to constant `{name}`"))
                        .with_code(codes::BAD_ASSIGNMENT)
                        .with_primary(ident.span, "constants are immutable")
                        .with_secondary(defined, "the constant is defined here");
                    let variable = name.to_lowercase();
                    if variable != name {
                        diag = diag.with_help(format!(
                            "to change the value, copy it into a variable: `{variable} = {name}`"
                        ));
                    }
                    self.error(diag);
                }
                let declared = match ty {
                    Some(ty) => {
                        let declared = self.db.lower_ty(self.module, ty);
                        let declared = self.db.value_type(declared, ty.span());
                        self.check(value, declared);
                        declared
                    }
                    None => {
                        // a reference is bound, not copied: the variable is the slot of the map
                        // or the header in the packet
                        let found = self.infer_raw(value, None);
                        if self.shallow(found) == self.db.types.unit {
                            self.error(
                                Diagnostic::error("this expression has no value")
                                    .with_code(codes::MISMATCHED_TYPES)
                                    .with_primary(self.span(value), "returns nothing"),
                            );
                        }
                        found
                    }
                };
                let local = self.declare(&ident, declared);
                self.result.decl_locals.insert(stmt, local);
                self.result.res.insert(target, Res::Local(local));
                self.result.expr_types.insert(target, declared);
            }
        }
    }

    /// the type of an assignable expression, checking that it is one: a
    /// local, or a field or element of a local or of a reference. The type is
    /// the one of the stored value, without the reference. The value
    /// assigned, if any, shows in the help of an element of a map
    fn place(&mut self, expr: ExprId, value: Option<ExprId>) -> TyId {
        let mut assignable = self.is_place(expr);
        let raw = self.infer_raw(expr, None);
        // `map[i] = v` is an error on its own: the element is an `Option` of a
        // reference, so we report it once here and give no type to check
        // the value against
        if let Expr::Index { base, indices } = &self.body.exprs[expr]
            && let Some(&Res::Map(m)) = self.result.res.get(*base)
            && !self.db.types.is_error(raw)
        {
            let name = self.db.package.maps[m].name.text.clone();
            let index = match indices.as_slice() {
                [index] => self.short_text(*index),
                _ => None,
            };
            let index = index.unwrap_or_else(|| "index".to_string());
            let value = value.and_then(|v| self.short_text(v));
            let value = value.unwrap_or_else(|| "value".to_string());
            return self.error(
                Diagnostic::error(format!("cannot assign to an element of map `{name}`"))
                    .with_code(codes::BAD_ASSIGNMENT)
                    .with_primary(self.span(expr), "maps are written with their methods")
                    .with_help(format!(
                        "write `{name}.insert({index}, {value})`, or change the slot through `{name}.get({index})`"
                    )),
            );
        }
        // a field or element of a reference (`ctx.parse[H](0)?.x`) is always
        // assignable, because the assignment writes through it
        if let Expr::Field { base, .. } | Expr::Index { base, .. } = self.body.exprs[expr]
            && let Some(&base_ty) = self.result.expr_types.get(base)
            && matches!(self.kind(base_ty), TyKind::Ref(_))
        {
            assignable = true;
        }
        let ty = self.strip_ref(raw);
        if let Some(Res::KernelField(_)) = self.result.res.get(expr) {
            return self.error(
                Diagnostic::error("kernel memory is read-only")
                    .with_code(codes::BAD_ASSIGNMENT)
                    .with_primary(self.span(expr), "a field of a kernel struct")
                    .with_note("programs read kernel structs with `bpf_probe_read_kernel`, which cannot write them"),
            );
        }
        if !assignable && !self.db.types.is_error(ty) {
            self.error(
                Diagnostic::error("cannot assign to this expression")
                    .with_code(codes::BAD_ASSIGNMENT)
                    .with_primary(self.span(expr), "not a variable, a field or an element"),
            );
        } else if self.changes_readonly_self(expr) {
            self.readonly_self_error(expr);
        }
        ty
    }

    /// if an expression is `self`, or a part of it, in a method without
    /// `@mut`
    fn changes_readonly_self(&self, expr: ExprId) -> bool {
        let mut root = expr;
        while let Expr::Field { base, .. } | Expr::Index { base, .. } = self.body.exprs[root] {
            root = base;
        }
        match (self.readonly_self, self.result.res.get(root)) {
            (Some(local), Some(Res::Local(l))) => local == *l,
            _ => false,
        }
    }

    /// reporting a change to `self` in a method without `@mut`
    fn readonly_self_error(&mut self, expr: ExprId) {
        self.error(
            Diagnostic::error("`self` cannot be changed in this method")
                .with_code(codes::BAD_ASSIGNMENT)
                .with_primary(self.span(expr), "part of `self`")
                .with_help("decorate the method with `@mut` to change its receiver in place"),
        );
    }

    /// the size of the format string of `print(s)`, as the code generator
    /// builds it: the text with `%` doubled, `%lld` or `%llu` for an integer
    /// or a `bool`, `%s` for bytes, then a newline if missing and a NUL. A
    /// value whose type is still unknown counts as the shortest, so the code
    /// generator reports it if it is too long after all
    fn print_format_size(&self, arg: ExprId) -> usize {
        let text = |t: &str| t.len() + t.matches('%').count();
        let mut size = 0;
        let mut ends_in_newline = false;
        let parts: &[FStrPart] = match &self.body.exprs[arg] {
            Expr::Str(s) => {
                size = text(s);
                ends_in_newline = s.ends_with('\n');
                &[]
            }
            Expr::FStr(parts) => parts,
            _ => return 0,
        };
        for part in parts {
            match part {
                FStrPart::Text(t) => {
                    size += text(t);
                    ends_in_newline = t.ends_with('\n');
                }
                FStrPart::Expr(e) => {
                    let ty = self.result.expr_types.get(*e).map(|&t| self.shallow(t));
                    size += match ty.map(|t| self.kind(t)) {
                        Some(TyKind::Int(_) | TyKind::Bool) => 4,
                        Some(TyKind::Var(v))
                            if matches!(self.vars[v as usize], Var::Unbound { int: true }) =>
                        {
                            4
                        }
                        _ => 2,
                    };
                    ends_in_newline = false;
                }
            }
        }
        size + usize::from(!ends_in_newline) + 1
    }

    /// the text of a short expression, a name or an integer, for a help
    fn short_text(&self, expr: ExprId) -> Option<String> {
        match &self.body.exprs[expr] {
            Expr::Name(name) => Some(name.clone()),
            Expr::Int(n) => Some(n.to_string()),
            _ => None,
        }
    }

    /// if an expression is a local, or a field or element of one. A missing
    /// expression counts, because its error is already reported
    fn is_place(&self, expr: ExprId) -> bool {
        match &self.body.exprs[expr] {
            Expr::Name(name) => self.lookup_local(name).is_some(),
            Expr::Field { base, .. } | Expr::Index { base, .. } => {
                self.expr_segments(expr).is_none() && self.is_place(*base)
            }
            Expr::Missing => true,
            _ => false,
        }
    }

    /// checking the decorators of a loop and recording the bound of
    /// `@bounded(N)` on a `while`. Run-time code needs it because the verifier
    /// rejects a loop it cannot bound
    fn loop_decorators(
        &mut self,
        stmt: StmtId,
        decorators: &[Decorator],
        is_while: bool,
        span: Span,
    ) {
        let mut bounded = false;
        let names: Vec<String> = decorators.iter().map(|d| d.name.to_string()).collect();
        if let Some(both) = decorators
            .iter()
            .find(|d| d.name.to_string() == "no_unroll")
            && names.iter().any(|n| n == "unroll")
        {
            self.error(
                Diagnostic::error("`@unroll` and `@no_unroll` on the same loop")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(both.span, "keep one of them"),
            );
        }
        for decorator in decorators {
            let name = decorator.name.to_string();
            match name.as_str() {
                "unroll" | "no_unroll" if decorator.args.is_empty() => {}
                // the condition is checked with the loop body, where its names are in
                // scope. Here we only check its shape
                "invariant" if decorator.args.len() == 1 && decorator.args[0].name.is_none() => {}
                "invariant" => {
                    self.error(
                        Diagnostic::error("`@invariant` takes one condition")
                            .with_code(codes::BAD_CONTRACT)
                            .with_primary(decorator.span, "expected `@invariant(condition)`"),
                    );
                }
                "unroll" | "no_unroll" => {
                    self.error(
                        Diagnostic::error(format!("`@{name}` takes no arguments"))
                            .with_code(codes::BAD_DECORATOR)
                            .with_primary(decorator.span, "remove the arguments"),
                    );
                }
                "bounded" if !is_while => {
                    self.error(
                        Diagnostic::error("`@bounded` applies to `while` loops")
                            .with_code(codes::BAD_DECORATOR)
                            .with_primary(
                                decorator.span,
                                "`for` loops are bounded by their iterable",
                            ),
                    );
                }
                "bounded" => {
                    bounded = true;
                    let value = match decorator.args.as_slice() {
                        [arg] if arg.name.is_none() => self.db.eval(self.module, &arg.value),
                        _ => None,
                    };
                    if let Some(Value::Int(n)) = value
                        && let Ok(n) = u64::try_from(n)
                        && n > 0
                    {
                        self.result.loop_bounds.insert(stmt, n);
                    } else {
                        self.error(
                            Diagnostic::error("`@bounded` takes the maximum number of iterations")
                                .with_code(codes::BAD_DECORATOR)
                                .with_primary(
                                    decorator.span,
                                    "expected `@bounded(N)` with a positive constant `N`",
                                ),
                        );
                    }
                }
                _ if ProgramKind::DECORATORS.contains(&name.as_str()) => {
                    self.error(
                        Diagnostic::error(format!("`@{name}` applies to functions"))
                            .with_code(codes::BAD_DECORATOR)
                            .with_primary(decorator.span, "not allowed on a loop"),
                    );
                }
                _ => {
                    self.db.diags.push(unknown_decorator(decorator));
                }
            }
        }
        // comptime code needs no bound, the interpreter limits its steps
        if is_while && !bounded && !self.comptime {
            self.error(
                Diagnostic::error("`while` loops need a bound")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(span, "the verifier rejects loops it cannot bound")
                    .with_help("add `@bounded(N)` before the loop, with `N` the maximum number of iterations"),
            );
        }
    }

    /// the element type of what a `for` iterates over: the integers of a
    /// `range` or the elements of an array. `range` is not a value, so we
    /// recognise its call here before inferring anything
    fn iterable(&mut self, iter: ExprId) -> TyId {
        if let Expr::Call { callee, args } = &self.body.exprs[iter]
            && let Some(segments) = self.expr_segments(*callee)
            && let Some(Ok(Path::Builtin(Builtin::Range))) = self.lookup_path(&segments)
        {
            self.result
                .res
                .insert(*callee, Res::Builtin(Builtin::Range));
            let element = self.var(true);
            if args.is_empty() || args.len() > 2 || args.iter().any(|a| a.name.is_some()) {
                self.error(
                    Diagnostic::error("`range` takes one or two integers")
                        .with_code(codes::BAD_ARGUMENTS)
                        .with_primary(self.span(iter), "write `range(n)` or `range(start, end)`"),
                );
            }
            for arg in args {
                self.check(arg.value, element);
            }
            self.result.expr_types.insert(iter, element);
            return element;
        }
        let ty = self.infer(iter, None);
        match self.kind(ty) {
            TyKind::Array(element, _) => element,
            TyKind::Error => ty,
            _ => {
                let found = self.show(ty);
                self.error(
                    Diagnostic::error(format!("cannot iterate over `{found}`"))
                        .with_code(codes::MISMATCHED_TYPES)
                        .with_primary(self.span(iter), "expected `range(...)` or an array"),
                )
            }
        }
    }

    // expressions
    // -------------------------

    /// checking an expression against an expected type, reporting a mismatch.
    /// The type is also passed down as a hint, so a literal gets it with no
    /// inference variable. When a reference is expected the expression is not
    /// read
    fn check(&mut self, expr: ExprId, expected: TyId) {
        self.check_with(expr, expected, None);
    }

    /// [`Self::check`], with a label on what fixed the expected type for a
    /// mismatch
    fn check_with(&mut self, expr: ExprId, expected: TyId, because: Option<(Span, String)>) {
        let found = if matches!(self.kind(expected), TyKind::Ref(_)) {
            self.infer_raw(expr, Some(expected))
        } else {
            self.infer(expr, Some(expected))
        };
        if !self.unify(found, expected) {
            self.mismatch_with(self.span(expr), expected, found, because);
        }
    }

    /// the type of an expression whose value is used. A reference is read
    /// (recorded in `derefs`) and we give the referenced type
    fn infer(&mut self, expr: ExprId, expected: Option<TyId>) -> TyId {
        let ty = self.infer_raw(expr, expected);
        if let TyKind::Ref(inner) = self.kind(ty) {
            self.result.derefs.insert(expr);
            return inner;
        }
        ty
    }

    /// the type of an expression used as a place or bound to a name, keeping
    /// references. The type is recorded in `expr_types`
    fn infer_raw(&mut self, expr: ExprId, expected: Option<TyId>) -> TyId {
        let ty = self.infer_inner(expr, expected);
        self.result.expr_types.insert(expr, ty);
        ty
    }

    /// the referenced type of a reference, or the type itself
    fn strip_ref(&self, ty: TyId) -> TyId {
        match self.kind(ty) {
            TyKind::Ref(inner) => inner,
            _ => ty,
        }
    }

    /// the type of an expression before reading any reference. `expected` is
    /// a hint, not a constraint: callers that need it unify afterwards
    fn infer_inner(&mut self, expr: ExprId, expected: Option<TyId>) -> TyId {
        let span = self.span(expr);
        match &self.body.exprs[expr] {
            Expr::Missing => self.db.types.error,
            Expr::Int(value) => {
                let value = i128::try_from(*value).unwrap_or(i128::MAX);
                self.int_literals.push((expr, value));
                match expected.map(|e| self.kind(e)) {
                    Some(TyKind::Int(_)) => expected.unwrap(),
                    _ => self.var(true),
                }
            }
            Expr::Float(_) if self.comptime => self.db.types.float(),
            Expr::Float(_) => self.float_error(span),
            Expr::Bool(_) => self.db.types.bool,
            Expr::Str(_) | Expr::FStr(_) if !self.in_print => self.error(
                Diagnostic::error("strings can only be passed to `print`")
                    .with_code(codes::STRING_OUTSIDE_PRINT)
                    .with_primary(span, "eBPF programs have no string values")
                    .with_help("write the string directly in `print(...)`"),
            ),
            Expr::Str(_) => self.db.types.str,
            Expr::FStr(parts) => {
                // a value interpolated in the string is not an argument of `print`, so
                // a string there is an error
                self.in_print = false;
                for part in parts {
                    if let FStrPart::Expr(e) = part {
                        self.printable(*e);
                    }
                }
                self.in_print = true;
                self.db.types.str
            }
            Expr::None => {
                let inner = self.var(false);
                self.db.types.option(inner)
            }
            Expr::Ellipsis => self.error(
                Diagnostic::error("`...` is only allowed in protocol methods")
                    .with_code(codes::NOT_SUPPORTED_YET)
                    .with_primary(span, "not a value"),
            ),
            Expr::Name(name) => {
                if let Some(local) = self.lookup_local(name) {
                    self.result.res.insert(expr, Res::Local(local));
                    let ty = self.result.locals[local].ty;
                    if self.in_comptime_block && !self.comptime_locals.contains(&local) {
                        self.error(
                            Diagnostic::error(format!("`{name}` is not known at compile time"))
                                .with_code(codes::CONST_EVAL)
                                .with_primary(span, "a variable of the run-time code")
                                .with_help("comptime blocks read constants and the variables of comptime blocks"),
                        );
                    } else if !self.comptime
                        && self.comptime_locals.contains(&local)
                        && matches!(self.kind(ty), TyKind::Float)
                    {
                        return self.float_error(span);
                    }
                    return ty;
                }
                let ident = Ident {
                    text: name.clone(),
                    span,
                };
                match self.resolve_path(&[ident]) {
                    Some(path) => self.path_value(expr, path, span, expected),
                    None => self.unresolved(name, span),
                }
            }
            Expr::Field { base, name } => {
                if let Some(segments) = self.expr_segments(expr)
                    && let Some(path) = self.resolve_path(&segments)
                {
                    return self.path_value(expr, path, span, expected);
                }
                // a member of an instance: `Maybe[u32].Nothing`
                if let Some(path) = self.instance_path(*base) {
                    let path = match self.member_path(path, &[], std::slice::from_ref(name)) {
                        Some(Ok(path)) => path,
                        Some(Err(diagnostic)) => {
                            self.db.diags.push(diagnostic);
                            Path::Error
                        }
                        None => Path::Error,
                    };
                    return self.path_value(expr, path, span, expected);
                }
                if self
                    .expr_segments(*base)
                    .is_some_and(|s| self.lookup_path(&s).is_none())
                    && let Expr::Name(root) = &self.body.exprs[*base]
                {
                    return self.unresolved(&root.clone(), self.span(*base));
                }
                let name = name.clone();
                self.field(expr, *base, &name)
            }
            Expr::Call { callee, args } => self.call(expr, *callee, args, expected),
            Expr::Index { base, indices } => self.index(*base, indices, span),
            Expr::Unary { op, operand } => {
                let operand = *operand;
                match op {
                    UnaryOp::Not => {
                        self.check(operand, self.db.types.bool);
                        self.db.types.bool
                    }
                    UnaryOp::Neg | UnaryOp::BitNot => {
                        let ty = self.infer(operand, expected);
                        let name = if *op == UnaryOp::Neg {
                            "__neg__"
                        } else {
                            "__invert__"
                        };
                        if let Some(method) = self.operator_method(ty, name) {
                            self.result.res.insert(expr, Res::Operator(method));
                            return self.operator_call(method, None, span);
                        }
                        if *op == UnaryOp::Neg && self.kind(ty) == TyKind::Float {
                            return ty;
                        }
                        if !self.is_int(ty) {
                            return self.bad_operand(op_symbol(*op), ty, self.span(operand));
                        }
                        if *op == UnaryOp::Neg {
                            self.negations.push(expr);
                            // here we range-check `-128` as a whole, not
                            // `128`, because the minimum of a signed type
                            // has no positive counterpart
                            if let Expr::Int(_) = self.body.exprs[operand]
                                && let Some(literal) =
                                    self.int_literals.iter_mut().find(|(e, _)| *e == operand)
                            {
                                *literal = (expr, -literal.1);
                            }
                        }
                        ty
                    }
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let (op, lhs, rhs) = (*op, *lhs, *rhs);
                if op == BinaryOp::Mul
                    && let Expr::Array(elements) = &self.body.exprs[lhs]
                    && elements.len() == 1
                {
                    return self.repeat(expr, elements[0], rhs, expected);
                }
                if matches!(op, BinaryOp::And | BinaryOp::Or) {
                    self.check(lhs, self.db.types.bool);
                    self.check(rhs, self.db.types.bool);
                    return self.db.types.bool;
                }
                // the expected `bool` of a comparison says nothing about its operands
                let hint = if op.is_comparison() { None } else { expected };
                let lhs_ty = self.infer(lhs, hint);
                // an operator of a struct is a method (`__add__`), looked up on the
                // type of the left operand
                if let Some(method) = self.operator_method(lhs_ty, binary_method(op)) {
                    self.result.res.insert(expr, Res::Operator(method));
                    return self.operator_call(method, Some(rhs), span);
                }
                // the amount of a shift is an integer of any type, not necessarily the
                // type of the shifted value
                if let BinaryOp::Shl | BinaryOp::Shr = op {
                    let amount = self.infer(rhs, None);
                    if !self.is_int(amount) {
                        self.bad_operand(op.symbol(), amount, self.span(rhs));
                    }
                    if !self.is_int(lhs_ty) {
                        return self.bad_operand(op.symbol(), lhs_ty, self.span(lhs));
                    }
                    return lhs_ty;
                }
                if self.operator_applies(op, lhs_ty) {
                    self.check(rhs, lhs_ty);
                } else {
                    // the left operand is the error, the right one is not checked
                    // against it, only unified to infer what it can (`o == None`)
                    let rhs_ty = self.infer(rhs, Some(lhs_ty));
                    self.unify(rhs_ty, lhs_ty);
                }
                self.binary_result(op, lhs_ty, span, self.span(lhs))
            }
            Expr::Try(inner) => {
                let inner = *inner;
                let ret_is_option = matches!(self.kind(self.ret), TyKind::Option(_));
                if !ret_is_option && !self.program {
                    let name = self.name;
                    let returns = if self.ret == self.db.types.unit {
                        format!("`{name}` returns nothing")
                    } else {
                        format!(
                            "`{name}` returns `{}`, not an `Option`",
                            self.show(self.ret)
                        )
                    };
                    let mut diag = Diagnostic::error("`?` needs a function that can fail")
                        .with_code(codes::BAD_TRY)
                        .with_primary(span, "`None` would leave the function here");
                    diag = match self.ret_span {
                        Some(ret_span) => diag.with_secondary(ret_span, returns),
                        None => diag.with_note(returns),
                    };
                    self.error(
                        diag.with_note("in eBPF programs, `?` returns the default action of the program type")
                            .with_help("handle the `None` case with `match`, or make the function return an `Option`"),
                    );
                }
                let ty = self.infer(inner, None);
                match self.kind(ty) {
                    TyKind::Option(element) => element,
                    TyKind::Error => ty,
                    // an operand of unknown type becomes an `Option` of unknown type
                    TyKind::Var(_) => {
                        let element = self.var(false);
                        let option = self.db.types.option(element);
                        self.unify(ty, option);
                        element
                    }
                    _ => {
                        let found = self.show(ty);
                        self.error(
                            Diagnostic::error(format!("`?` needs an `Option`, found `{found}`"))
                                .with_code(codes::BAD_TRY)
                                .with_primary(self.span(inner), "not an `Option`"),
                        )
                    }
                }
            }
            Expr::Array(elements) => {
                let elements = elements.clone();
                let (element, len) = match expected.map(|e| self.kind(e)) {
                    Some(TyKind::Array(element, len)) => (element, Some(len)),
                    _ => match elements.first() {
                        Some(&first) => (self.infer(first, None), None),
                        None => {
                            return self.error(
                                Diagnostic::error("cannot infer the type of an empty array")
                                    .with_code(codes::CANNOT_INFER)
                                    .with_primary(span, "arrays have at least one element")
                                    .with_help("to fill an array, repeat a value, as in `[0] * 8`"),
                            );
                        }
                    },
                };
                if let Some(len) = len
                    && len != elements.len() as u64
                {
                    return self.error(
                        Diagnostic::error("wrong number of elements")
                            .with_code(codes::MISMATCHED_TYPES)
                            .with_primary(
                                span,
                                format!("expected {len}, found {}", elements.len()),
                            ),
                    );
                }
                for &e in &elements {
                    self.check(e, element);
                }
                self.db.types.array(element, elements.len() as u64)
            }
        }
    }

    /// the type of `[element] * count`: an array of `count` copies of a
    /// value, `count` being a constant
    fn repeat(
        &mut self,
        expr: ExprId,
        element: ExprId,
        count: ExprId,
        expected: Option<TyId>,
    ) -> TyId {
        let expected_element = match expected.map(|e| self.kind(e)) {
            Some(TyKind::Array(element, _)) => Some(element),
            _ => None,
        };
        let element_ty = match expected_element {
            Some(ty) => {
                self.check(element, ty);
                ty
            }
            None => self.infer(element, None),
        };
        let value = self.const_of(count);
        let count_ty = self.infer(count, None);
        if !self.is_int(count_ty) {
            return self.bad_operand("*", count_ty, self.span(count));
        }
        if value == Err(true) {
            return self.db.types.error;
        }
        let Some(len) = value.ok().and_then(|v| u64::try_from(v).ok()) else {
            return self.error(
                match value {
                    Ok(v) if v < 0 => {
                        Diagnostic::error("the length of `[x] * n` cannot be negative")
                            .with_primary(self.span(count), format!("this is `{v}`"))
                    }
                    Ok(v) => Diagnostic::error("the length of `[x] * n` is too large")
                        .with_primary(self.span(count), format!("this is `{v}`")),
                    Err(_) => Diagnostic::error("the length of `[x] * n` must be a constant")
                        .with_primary(self.span(count), "not known at compile time"),
                }
                .with_code(codes::CONST_EVAL),
            );
        };
        // the element is evaluated once and copied into every slot. The
        // `[element]` on the left is not inferred alone, it gets the array type
        self.result.res.insert(expr, Res::Repeat(len));
        let array = self.db.types.array(element_ty, len);
        self.result.expr_types.insert(self.lhs_of(expr), array);
        array
    }

    /// the left operand of a binary expression
    fn lhs_of(&self, expr: ExprId) -> ExprId {
        match self.body.exprs[expr] {
            Expr::Binary { lhs, .. } => lhs,
            _ => expr,
        }
    }

    /// the enum of variant `i` named by `e`: `e` itself, or when `e` is
    /// generic the instance given by the expected type. `None` when the
    /// expected type does not tell the instance
    fn instance_variant(&mut self, e: EnumId, i: u32, expected: Option<TyId>) -> Option<EnumId> {
        if !self.db.is_template_enum(e) {
            return Some(e);
        }
        match self.kind(expected?) {
            TyKind::Enum(instance)
                if matches!(
                    &self.db.package.enums[instance].instance,
                    Some(Instance { generic: Item::Enum(g), .. }) if *g == e
                ) =>
            {
                let _ = i;
                Some(instance)
            }
            _ => None,
        }
    }

    /// reporting variant `i` of a generic enum whose instance cannot be
    /// found, `label` saying why
    fn uninferred_enum(&mut self, e: EnumId, i: u32, span: Span, label: &str) -> TyId {
        let data = &self.db.package.enums[e];
        let (name, variant) = (&data.name.text, &data.variants[i as usize].name.text);
        self.error(
            Diagnostic::error(format!("cannot infer the generic arguments of `{name}`"))
                .with_code(codes::CANNOT_INFER)
                .with_primary(span, label)
                .with_help(format!("write them, as in `{name}[...].{variant}`")),
        )
    }

    /// the type of a variant of a generic enum built without generic
    /// arguments, which we find from the values (`Maybe.Some(5)`)
    fn inferred_variant_call(
        &mut self,
        e: EnumId,
        i: u32,
        callee: ExprId,
        args: &[Arg],
        span: Span,
    ) -> TyId {
        let package = self.db.package;
        let data = &package.enums[e];
        let names: Vec<String> = data
            .generic_params
            .iter()
            .map(|p| p.name.text.clone())
            .collect();
        let tys: Vec<TyId> = args.iter().map(|a| self.infer(a.value, None)).collect();
        let mut bound: Vec<Option<GenericArgValue>> = vec![None; names.len()];
        if let Some(fields) = &data.variants[i as usize].fields {
            for (field, &ty) in fields.iter().zip(&tys) {
                self.bind_generic(data.module, field, ty, &names, &mut bound);
            }
        }
        let Some(values) = bound.into_iter().collect::<Option<Vec<_>>>() else {
            return self.uninferred_enum(e, i, span, "the values here do not tell them");
        };
        let callee_span = self.span(callee);
        let generic_args = values.into_iter().map(|v| (v, callee_span)).collect();
        let Some(instance_args) =
            self.db
                .check_generic_args(Item::Enum(e), generic_args, callee_span)
        else {
            return self.db.types.error;
        };
        // an instance not in the package yet exists in the next run of the
        // analysis, until then the value has the error type
        let Some(Item::Enum(instance)) = self.db.instantiate(Item::Enum(e), instance_args) else {
            return self.db.types.error;
        };
        self.result.res.insert(callee, Res::Variant(instance, i));
        let fields = self.db.variant_fields[instance][i as usize]
            .clone()
            .unwrap_or_default();
        if fields.len() != args.len() {
            return self.error(
                Diagnostic::error("wrong number of values")
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(
                        span,
                        format!("expected {}, found {}", fields.len(), args.len()),
                    ),
            );
        }
        for ((arg, &ty), &expected) in args.iter().zip(&tys).zip(&fields) {
            if !self.unify(ty, expected) {
                self.mismatch(self.span(arg.value), expected, ty);
            }
        }
        self.db.types.intern(TyKind::Enum(instance))
    }

    /// the type of a path used as a value: a constant, a constant generic
    /// parameter or a variant without fields
    fn path_value(&mut self, expr: ExprId, path: Path, span: Span, expected: Option<TyId>) -> TyId {
        match path {
            Path::Value(v, ty) => {
                self.result.res.insert(expr, Res::ConstValue(v));
                ty
            }
            Path::Const(c) => {
                self.result.res.insert(expr, Res::Const(c));
                self.db.const_info(c).0
            }
            Path::Variant(e, i) => {
                let package = self.db.package;
                let Some(e) = self.instance_variant(e, i, expected) else {
                    return self.uninferred_enum(e, i, span, "nothing here tells them");
                };
                let variant = &package.enums[e].variants[i as usize];
                if variant.fields.is_some() {
                    let n = variant.fields.as_ref().map_or(0, Vec::len);
                    return self.error(
                        Diagnostic::error(format!("variant `{}` has fields", variant.name.text))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(
                                span,
                                format!(
                                    "construct it with {} in parentheses",
                                    vigil_diagnostics::plural(n, "value")
                                ),
                            ),
                    );
                }
                self.result.res.insert(expr, Res::Variant(e, i));
                self.db.types.intern(TyKind::Enum(e))
            }
            Path::Error => self.db.types.error,
            other => self.not_a_value(other, span),
        }
    }

    /// the type of `base.name` when it is not a path: a field of a struct,
    /// or of a kernel struct through a `kptr` or a tracepoint context
    fn field(&mut self, expr: ExprId, base: ExprId, name: &Ident) -> TyId {
        // a field of a referenced struct is read in place, the struct is not
        // copied out, so the base is not marked as read
        let raw = self.infer_raw(base, None);
        let base_ty = self.strip_ref(raw);
        let package = self.db.package;
        match self.kind(base_ty) {
            TyKind::KPtr(pointee) => {
                let TyKind::Kernel(composite) = self.kind(pointee) else {
                    return self.db.types.error;
                };
                self.kernel_field(expr, composite, name, KernelRead::Probe)
            }
            TyKind::TpContext(composite) => {
                self.kernel_field(expr, composite, name, KernelRead::Direct)
            }
            TyKind::Struct(s) => {
                let data = &package.structs[s];
                if let Some(i) = data.fields.iter().position(|f| f.name.text == name.text) {
                    self.result.res.insert(expr, Res::Field(i as u32));
                    return self.db.struct_fields[s][i];
                }
                if data
                    .methods
                    .iter()
                    .any(|&m| package.fns[m].name.text == name.text)
                {
                    return self.error(
                        Diagnostic::error(format!("`{}` is a method", name.text))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(name.span, "call it with `(...)`"),
                    );
                }
                let fields: Vec<_> = data
                    .fields
                    .iter()
                    .map(|f| format!("`{}`", f.name.text))
                    .collect();
                let mut diag =
                    Diagnostic::error(format!("`{}` has no field `{}`", data.name.text, name.text))
                        .with_code(codes::UNKNOWN_FIELD)
                        .with_primary(name.span, "unknown field");
                if !fields.is_empty() {
                    diag = diag.with_note(format!("its fields are {}", fields.join(", ")));
                }
                self.error(diag)
            }
            TyKind::Error => base_ty,
            TyKind::Option(_) => self.error(
                Diagnostic::error(format!("`Option` has no field `{}`", name.text))
                    .with_code(codes::UNKNOWN_FIELD)
                    .with_primary(name.span, "unknown field")
                    .with_help("get the value first, with `?` or `match`"),
            ),
            _ => {
                let found = self.show(base_ty);
                self.error(
                    Diagnostic::error(format!("`{found}` has no field `{}`", name.text))
                        .with_code(codes::UNKNOWN_FIELD)
                        .with_primary(name.span, "unknown field"),
                )
            }
        }
    }

    /// the type of `base[indices]`: an element of an array or of an array
    /// map. Here we also report a generic item with arguments used as a value
    /// and a constant index past the end of an array
    fn index(&mut self, base: ExprId, indices: &[ExprId], span: Span) -> TyId {
        let base_path = self
            .expr_segments(base)
            .and_then(|segments| self.lookup_path(&segments));
        if let Some(Ok(
            path @ (Path::Struct(_)
            | Path::Enum(_)
            | Path::Builtin(_)
            | Path::Fn(_)
            | Path::Method(_)
            | Path::Error),
        )) = base_path
        {
            for &i in indices {
                self.infer(i, None);
            }
            // a name whose import failed, already reported
            if let Path::Error = path {
                return self.db.types.error;
            }
            return self.error(
                Diagnostic::error("an item with generic arguments is not a value")
                    .with_code(codes::WRONG_KIND)
                    .with_primary(span, "not a value")
                    .with_help("call it, as in `Vector[u8, 4](...)` or `argmax[u8, 4](v)`"),
            );
        }
        if let Some(Ok(Path::Map(m))) = base_path {
            return self.map_index(base, m, indices, span);
        }
        let raw = self.infer_raw(base, None);
        let base_ty = self.strip_ref(raw);
        let [index] = indices else {
            for &i in indices {
                self.infer(i, None);
            }
            return self.error(
                Diagnostic::error("arrays take one index")
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(span, format!("found {} indices", indices.len())),
            );
        };
        let index_ty = self.infer(*index, None);
        if !self.is_int(index_ty) {
            let found = self.show(index_ty);
            self.error(
                Diagnostic::error(format!("array indices are integers, found `{found}`"))
                    .with_code(codes::MISMATCHED_TYPES)
                    .with_primary(self.span(*index), "not an integer"),
            );
        }
        match self.kind(base_ty) {
            TyKind::Array(element, len) => {
                // a literal, or a constant expression as `const` computes it
                let constant = match self.body.exprs[*index] {
                    Expr::Int(i) => i128::try_from(i).ok(),
                    _ if self.is_int(index_ty) => self.const_of(*index).ok(),
                    _ => None,
                };
                if let Some(i) = constant
                    && i >= i128::from(len)
                {
                    self.error(
                        Diagnostic::error(format!("index {i} is out of bounds"))
                            .with_code(codes::CONST_EVAL)
                            .with_primary(
                                self.span(*index),
                                format!(
                                    "the array has {}",
                                    vigil_diagnostics::plural(len, "element")
                                ),
                            ),
                    );
                }
                element
            }
            TyKind::Error => base_ty,
            _ => {
                let found = self.show(base_ty);
                self.error(
                    Diagnostic::error(format!("cannot index a value of type `{found}`"))
                        .with_code(codes::MISMATCHED_TYPES)
                        .with_primary(self.span(base), "not an array"),
                )
            }
        }
    }

    /// the type of a call. First we try the forms that are not plain paths,
    /// in order: `kptr[T](address)`, `ctx.parse[H](offset)`, a static method of
    /// an instance, an instance and a method call on a value. Otherwise the
    /// callee is a path: a function, a struct, a variant or a builtin
    fn call(&mut self, expr: ExprId, callee: ExprId, args: &[Arg], expected: Option<TyId>) -> TyId {
        let span = self.span(expr);
        let package = self.db.package;

        // `kptr[T](address)`: an address made a pointer to a kernel struct
        if let Expr::Index { base, indices } = &self.body.exprs[callee]
            && let Some(Ok(Path::Builtin(Builtin::KPtr))) =
                self.expr_segments(*base).and_then(|s| self.lookup_path(&s))
        {
            let pointee = match indices.as_slice() {
                [index] => self
                    .expr_segments(*index)
                    .and_then(|segments| self.resolve_path(&segments)),
                _ => None,
            };
            let Some(Path::Kernel(id)) = pointee else {
                self.skip_args(args);
                if matches!(pointee, Some(Path::Error)) {
                    return self.db.types.error;
                }
                return self.error(
                    Diagnostic::error("`kptr` points to a struct of the kernel")
                        .with_code(codes::WRONG_KIND)
                        .with_primary(span, "write `kptr[task_struct](address)`, importing the struct from `kernel`"),
                );
            };
            self.result.res.insert(callee, Res::KPtrCast);
            let u64_ty = self.db.types.int(IntTy::U64);
            self.args(
                span,
                &[("address".to_string(), u64_ty)],
                args,
                "function",
                None,
            );
            let kernel = self.db.kernel_ty(id);
            let pointer = self.db.types.intern(TyKind::KPtr(kernel));
            self.result.expr_types.insert(callee, pointer);
            return pointer;
        }

        // `ctx.parse[Header](offset)` on a packet context. On the context of
        // another kind of program we report it here
        if let Expr::Index { base, indices } = &self.body.exprs[callee]
            && let Expr::Field {
                base: receiver,
                name,
            } = &self.body.exprs[*base]
            && name.text == "parse"
            && self.expr_segments(*receiver).is_none()
        {
            let receiver_ty = self.infer_raw(*receiver, None);
            let receiver_ty = self.strip_ref(receiver_ty);
            if self.packet_context(receiver_ty) {
                return self.packet_parse(callee, indices, args, span);
            }
            if let TyKind::Struct(s) = self.kind(receiver_ty)
                && self.db.struct_contexts.get(s).is_some()
            {
                self.skip_args(args);
                return self.error(
                    Diagnostic::error("only packet programs can parse packets")
                        .with_code(codes::WRONG_KIND)
                        .with_primary(
                            name.span,
                            "`parse` is a method of `XdpContext` and `TcContext`",
                        ),
                );
            }
        }

        // a static method of an instance: `Vec[u8, 4].zero()`
        let mut instance_target = None;
        if let Expr::Field { base, name } = &self.body.exprs[callee]
            && let Some(path) = self.instance_path(*base)
        {
            let name = name.clone();
            instance_target = Some(
                match self.member_path(path, &[], std::slice::from_ref(&name)) {
                    Some(Ok(path)) => path,
                    Some(Err(diagnostic)) => {
                        self.db.diags.push(diagnostic);
                        Path::Error
                    }
                    None => Path::Error,
                },
            );
        }
        // an instance: `Vec[u8, 4](...)`, `relu[q8_8, 8](v)`
        if instance_target.is_none() {
            instance_target = self.instance_path(callee);
        }

        // a method call on a value: `value.method(args)`
        if instance_target.is_none()
            && let Expr::Field { base, name } = &self.body.exprs[callee]
            && self
                .expr_segments(callee)
                .is_none_or(|s| self.lookup_path(&s).is_none())
        {
            if let Some(Ok(Path::Map(m))) = self
                .expr_segments(*base)
                .and_then(|segments| self.lookup_path(&segments))
            {
                return self.map_method(*base, callee, m, name, args, span);
            }
            // a reference receiver is marked as read, as the builtin methods below
            // need. A method of the package takes it by reference and unmarks it
            let raw = self.infer_raw(*base, None);
            if matches!(self.kind(raw), TyKind::Ref(_)) {
                self.result.derefs.insert(*base);
            }
            let base_ty = self.strip_ref(raw);
            if let Some(ty) = self.builtin_method(callee, base_ty, name, args, span) {
                return ty;
            }
            if name.text == "len" && self.packet_context(base_ty) {
                self.result.res.insert(callee, Res::PacketLen);
                if !args.is_empty() {
                    self.skip_args(args);
                    self.error(
                        Diagnostic::error("`len` takes no arguments")
                            .with_code(codes::BAD_ARGUMENTS)
                            .with_primary(span, "write `ctx.len()`"),
                    );
                }
                let u32_ty = self.db.types.int(IntTy::U32);
                self.result.expr_types.insert(callee, u32_ty);
                return u32_ty;
            }
            let methods = match self.kind(base_ty) {
                TyKind::Struct(s) => &package.structs[s].methods,
                TyKind::Enum(e) => &package.enums[e].methods,
                TyKind::Error => return self.skip_args(args),
                _ => {
                    let found = self.show(base_ty);
                    self.skip_args(args);
                    return self.error(
                        Diagnostic::error(format!("`{found}` has no method `{}`", name.text))
                            .with_code(codes::UNKNOWN_FIELD)
                            .with_primary(name.span, "unknown method"),
                    );
                }
            };
            let Some(&method) = methods
                .iter()
                .find(|&&m| package.fns[m].name.text == name.text)
            else {
                let found = self.show(base_ty);
                self.skip_args(args);
                return self.error(
                    Diagnostic::error(format!("`{found}` has no method `{}`", name.text))
                        .with_code(codes::UNKNOWN_FIELD)
                        .with_primary(name.span, "unknown method"),
                );
            };
            let sig = self.db.fn_sigs[method].clone();
            if !sig.has_self {
                let owner = self.show(base_ty);
                self.skip_args(args);
                return self.error(
                    Diagnostic::error(format!("`{}` does not take `self`", name.text))
                        .with_code(codes::WRONG_KIND)
                        .with_primary(
                            name.span,
                            format!("call it on the type, as `{owner}.{}(...)`", name.text),
                        ),
                );
            }
            // a method takes the receiver by reference. A `@mut` one changes it, so
            // the receiver must be a variable, a part of one or a reference into a map
            self.result.derefs.remove(base);
            if package.fns[method]
                .decorators
                .iter()
                .any(|d| d.name.to_string() == "mut")
            {
                if self.changes_readonly_self(*base) {
                    self.readonly_self_error(*base);
                } else if !matches!(self.kind(raw), TyKind::Ref(_)) && !self.is_place(*base) {
                    self.error(
                        Diagnostic::error(format!("`{}` changes its receiver", name.text))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(self.span(*base), "not a variable")
                            .with_help("store the value in a variable first"),
                    );
                }
            }
            self.result.res.insert(callee, Res::Method(method));
            self.result.expr_types.insert(callee, sig.ret);
            let optional = self.optional_params(method);
            let declared = &package.fns[method].name;
            self.args_with(
                span,
                sig.call_params(),
                &optional,
                args,
                "function",
                Some(declared),
            );
            return sig.ret;
        }

        let target = match (instance_target, self.expr_segments(callee)) {
            (Some(path), _) => path,
            (None, Some(segments)) => match self.resolve_path(&segments) {
                Some(path) => path,
                None if segments.len() == 1 => {
                    self.skip_args(args);
                    return self.unresolved(&segments[0].text, self.span(callee));
                }
                None => {
                    // `CONST.x(...)`: calling a member of a value, which is not a function
                    let ty = self.infer(callee, None);
                    self.skip_args(args);
                    return self.not_callable(ty, callee);
                }
            },
            (None, None) => {
                let ty = self.infer(callee, None);
                self.skip_args(args);
                return self.not_callable(ty, callee);
            }
        };

        match target {
            Path::Fn(f) if self.db.is_template_fn(f) => {
                self.inferred_generic_call(f, callee, args, span)
            }
            Path::Method(f) if self.db.is_template_fn(f) => {
                self.skip_args(args);
                let data = &package.fns[f];
                let name = &data.name.text;
                let list = |params: &[GenericParam]| {
                    let names: Vec<&str> = params.iter().map(|p| p.name.text.as_str()).collect();
                    names.join(", ")
                };
                // here the arguments of a generic struct or enum go on the
                // type, `Ring[T, N].empty(...)`, those of the method after it
                let owner = match data.owner {
                    Some(Owner::Struct(s)) => {
                        let owner = &package.structs[s];
                        Some((&owner.name.text, &owner.generic_params))
                    }
                    Some(Owner::Enum(e)) => {
                        let owner = &package.enums[e];
                        Some((&owner.name.text, &owner.generic_params))
                    }
                    None => None,
                };
                let own = if data.generic_params.is_empty() {
                    String::new()
                } else {
                    format!("[{}]", list(&data.generic_params))
                };
                let (generic, example) = match owner {
                    Some((owner, params)) if !params.is_empty() => {
                        (owner, format!("{owner}[{}].{name}{own}(...)", list(params)))
                    }
                    Some((owner, _)) => (name, format!("{owner}.{name}{own}(...)")),
                    None => (name, format!("{name}{own}(...)")),
                };
                self.error(
                    Diagnostic::error(format!("`{generic}` is generic"))
                        .with_code(codes::BAD_ARGUMENTS)
                        .with_primary(
                            self.span(callee),
                            format!("give its generic arguments, as in `{example}`"),
                        ),
                )
            }
            Path::Struct(s) if self.db.is_template_struct(s) => {
                self.skip_args(args);
                let name = &package.structs[s].name.text;
                self.error(
                    Diagnostic::error(format!("`{name}` is generic"))
                        .with_code(codes::BAD_ARGUMENTS)
                        .with_primary(
                            self.span(callee),
                            format!("give its generic arguments, as in `{name}[...](...)`"),
                        ),
                )
            }
            Path::Fn(f) | Path::Method(f) => {
                let sig: FnSig = self.db.fn_sigs[f].clone();
                if sig.has_self {
                    let name = &package.fns[f].name.text;
                    self.skip_args(args);
                    return self.error(
                        Diagnostic::error(format!("`{name}` is a method that takes `self`"))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(
                                self.span(callee),
                                format!("call it on a value, as `value.{name}(...)`"),
                            ),
                    );
                }
                if package.fns[f].is_comptime && !self.comptime {
                    let name = &package.fns[f].name.text;
                    self.db.diags.push(
                        Diagnostic::error(format!("`{name}` runs at compile time"))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(
                                self.span(callee),
                                "comptime functions cannot be called at run time",
                            )
                            .with_help(format!(
                                "compute the value in a constant: `const TABLE: T = {name}(...)`"
                            )),
                    );
                }
                // a helper has no body to run, it exists only in the kernel
                if self.comptime
                    && package.fns[f]
                        .decorators
                        .iter()
                        .any(|d| d.name.to_string() == "helper")
                {
                    let name = &package.fns[f].name.text;
                    let place = if self.in_comptime_block {
                        format!("called in a comptime block of `{}`", self.name)
                    } else {
                        format!("called in the comptime function `{}`", self.name)
                    };
                    self.db.diags.push(
                        Diagnostic::error(format!("`{name}` cannot be called at compile time"))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(self.span(callee), place)
                            .with_note(
                                "it is a kernel helper, which only exists in the running program",
                            ),
                    );
                }
                self.result.res.insert(callee, Res::Fn(f));
                let optional = self.optional_params(f);
                let declared = &package.fns[f].name;
                self.args_with(
                    span,
                    sig.call_params(),
                    &optional,
                    args,
                    "function",
                    Some(declared),
                );
                sig.ret
            }
            Path::Struct(s) if self.db.struct_contexts.get(s).is_some() => {
                self.skip_args(args);
                let name = &package.structs[s].name.text;
                self.error(
                    Diagnostic::error(format!("`{name}` cannot be constructed"))
                        .with_code(codes::WRONG_KIND)
                        .with_primary(self.span(callee), "contexts are created by the kernel")
                        .with_help("take it as the parameter of a program"),
                )
            }
            Path::Struct(s) => {
                self.result.res.insert(callee, Res::StructCtor(s));
                let data = &package.structs[s];
                let params: Vec<_> = data
                    .fields
                    .iter()
                    .zip(&self.db.struct_fields[s])
                    .map(|(f, &ty)| (f.name.text.clone(), ty))
                    .collect();
                self.args(span, &params, args, "struct", Some(&data.name));
                self.db.types.intern(TyKind::Struct(s))
            }
            Path::Variant(e, i) => {
                let e = match self.instance_variant(e, i, expected) {
                    Some(e) => e,
                    None => return self.inferred_variant_call(e, i, callee, args, span),
                };
                let variant = &package.enums[e].variants[i as usize];
                let Some(fields) = self.db.variant_fields[e][i as usize].clone() else {
                    self.skip_args(args);
                    return self.error(
                        Diagnostic::error(format!("variant `{}` has no fields", variant.name.text))
                            .with_code(codes::BAD_ARGUMENTS)
                            .with_primary(
                                span,
                                format!("write it without parentheses, as `{}`", variant.name.text),
                            ),
                    );
                };
                self.result.res.insert(callee, Res::Variant(e, i));
                let params: Vec<_> = fields
                    .iter()
                    .enumerate()
                    .map(|(n, &ty)| (n.to_string(), ty))
                    .collect();
                self.args(span, &params, args, "variant", Some(&variant.name));
                self.db.types.intern(TyKind::Enum(e))
            }
            Path::Builtin(builtin) => {
                self.result.res.insert(callee, Res::Builtin(builtin));
                // the name as written, for messages
                let name = self
                    .expr_segments(callee)
                    .and_then(|s| s.last().map(|i| i.text.clone()))
                    .unwrap_or_else(|| "this builtin".into());
                self.builtin_call(builtin, &name, args, expected, span)
            }
            Path::Error => self.skip_args(args),
            Path::Const(c) => {
                let ty = self.db.const_info(c).0;
                self.skip_args(args);
                self.not_callable(ty, callee)
            }
            other => {
                self.skip_args(args);
                let (what, help) = match other {
                    Path::Enum(_) => ("an enum", "construct one of its variants"),
                    Path::Namespace => ("a module", "call one of its functions"),
                    Path::Map(_) => ("a map", "use its methods, as in `name.get(key)`"),
                    Path::Protocol => {
                        ("a protocol", "call a function of a type that implements it")
                    }
                    Path::Kernel(_) => (
                        "a kernel type",
                        "reach kernel structs through `kptr[T](address)`",
                    ),
                    _ => ("this", ""),
                };
                let mut diag = Diagnostic::error(format!("cannot call {what}"))
                    .with_code(codes::WRONG_KIND)
                    .with_primary(self.span(callee), "not callable");
                if !help.is_empty() {
                    diag = diag.with_help(help);
                }
                self.error(diag)
            }
        }
    }

    /// the type of a call of a map method, typed by the key and value types
    /// of the map. A lookup gives a reference into the map, not a copy
    fn map_method(
        &mut self,
        base: ExprId,
        callee: ExprId,
        m: MapId,
        name: &Ident,
        args: &[Arg],
        span: Span,
    ) -> TyId {
        let package = self.db.package;
        let map_name = &package.maps[m].name.text;
        self.result.res.insert(base, Res::Map(m));
        let Some(info) = self.db.maps.get(m).copied() else {
            // the map declaration has errors, already reported
            return self.skip_args(args);
        };
        let method = MapMethod::by_name(&name.text).filter(|method| method.applies_to(info.kind));
        let Some(method) = method else {
            self.skip_args(args);
            // a first note lists the methods programs may call, a second one
            // those only tests may call (E0417 is reported elsewhere)
            let quoted = |tests| -> Vec<String> {
                MapMethod::names_for(info.kind, tests)
                    .iter()
                    .map(|n| format!("`{n}`"))
                    .collect()
            };
            let methods = quoted(false);
            let test_only: Vec<String> = quoted(true)
                .into_iter()
                .filter(|n| !methods.contains(n))
                .collect();
            let mut diag = Diagnostic::error(format!(
                "`{}` maps have no method `{}`",
                info.kind.name(),
                name.text
            ))
            .with_code(codes::UNKNOWN_FIELD)
            .with_primary(name.span, format!("not a method of `{map_name}`"))
            .with_note(format!("the methods are {}", methods.join(", ")));
            if !test_only.is_empty() {
                diag = diag.with_note(format!("tests may also call {}", test_only.join(", ")));
            }
            return self.error(diag);
        };
        self.result.res.insert(callee, Res::MapMethod(m, method));
        let key = ("key".to_string(), info.key);
        let value_ref = self.db.types.intern(TyKind::Ref(info.value));
        let (params, ret) = match method {
            MapMethod::Get => (vec![key], self.db.types.option(value_ref)),
            MapMethod::GetOrInsert => (vec![key, ("default".into(), info.value)], value_ref),
            MapMethod::Insert => (vec![key, ("value".into(), info.value)], self.db.types.bool),
            MapMethod::Remove | MapMethod::Contains => (vec![key], self.db.types.bool),
            MapMethod::Emit => (vec![("event".into(), info.value)], self.db.types.bool),
            MapMethod::Count => {
                self.result.test_calls.push((span, "count"));
                (vec![], self.db.types.int(IntTy::U64))
            }
            MapMethod::Last => {
                self.result.test_calls.push((span, "last"));
                (vec![], self.db.types.option(info.value))
            }
            // any context, we check which one after typing the argument below
            MapMethod::TailCall => {
                let context = self.var(false);
                let index = ("index".to_string(), self.db.types.int(IntTy::U32));
                (vec![("ctx".into(), context), index], self.db.types.unit)
            }
        };
        self.args(span, &params, args, "function", None);
        if method == MapMethod::TailCall
            && let Some(ctx) = args.first()
        {
            let ty = self.shallow(self.result.expr_types[ctx.value]);
            let is_context = match self.kind(ty) {
                TyKind::Struct(s) => self.db.struct_contexts.get(s).is_some(),
                TyKind::TpContext(_) | TyKind::Error => true,
                _ => false,
            };
            if !is_context {
                let found = self.show(ty);
                self.error(
                    Diagnostic::error("`tail_call` takes the context of the program")
                        .with_code(codes::MISMATCHED_TYPES)
                        .with_primary(self.span(ctx.value), format!("found `{found}`")),
                );
            }
        }
        self.result.expr_types.insert(callee, ret);
        ret
    }

    /// the type of a method call on a kernel pointer or on a kprobe or LSM
    /// context. `None` when it is not one, so the caller keeps looking
    fn builtin_method(
        &mut self,
        callee: ExprId,
        base_ty: TyId,
        name: &Ident,
        args: &[Arg],
        span: Span,
    ) -> Option<TyId> {
        let kprobe = matches!(
            self.kind(base_ty),
            TyKind::Struct(s) if self.db.struct_contexts.get(s) == Some(&ContextKind::Kprobe)
        );
        let lsm = matches!(
            self.kind(base_ty),
            TyKind::Struct(s) if self.db.struct_contexts.get(s) == Some(&ContextKind::Lsm)
        );
        // a kprobe reads its arguments from the registers of `pt_regs` (how
        // many depends on the architecture), and an LSM hook from 12 words
        let max_args: i128 = if lsm {
            12
        } else {
            self.db.options.arch.arg_offsets().len() as i128
        };
        let (res, ty) = match (self.kind(base_ty), name.text.as_str()) {
            (TyKind::KPtr(_), "is_null") if args.is_empty() => {
                (Res::KPtrIsNull, self.db.types.bool)
            }
            (_, "ret") if kprobe && args.is_empty() => {
                // the return value only exists after the probed function returns. Out
                // of a program (in a function taking the context, which both kinds of
                // program may call) it is allowed
                use crate::program::ProgramKind;
                if let Some(kind) = self.program_kind
                    && !matches!(kind, ProgramKind::Kretprobe | ProgramKind::Uretprobe)
                {
                    return Some(
                        self.error(
                            Diagnostic::error(format!("`ret` is not available in {}", kind.noun()))
                                .with_code(codes::BAD_ARGUMENTS)
                                .with_primary(span, "the probed function has not returned yet")
                                .with_help(
                                    "read the value returned in a `@kretprobe` or `@uretprobe`",
                                ),
                        ),
                    );
                }
                (Res::KprobeRet, self.db.types.int(IntTy::U64))
            }
            (_, "arg") if kprobe || lsm => {
                let index = match args {
                    [arg] if arg.name.is_none() => match self.body.exprs[arg.value] {
                        Expr::Int(n) if (n as i128) < max_args => Some(n as u32),
                        _ => None,
                    },
                    _ => None,
                };
                let Some(index) = index else {
                    self.skip_args(args);
                    return Some(
                        self.error(
                            Diagnostic::error("`arg` takes the position of the argument")
                                .with_code(codes::BAD_ARGUMENTS)
                                .with_primary(
                                    span,
                                    format!(
                                        "expected `ctx.arg(n)` with `n` a literal from 0 to {}",
                                        max_args - 1
                                    ),
                                ),
                        ),
                    );
                };
                (Res::KprobeArg(index), self.db.types.int(IntTy::U64))
            }
            _ => return None,
        };
        self.result.res.insert(callee, res);
        self.result.expr_types.insert(callee, ty);
        Some(ty)
    }

    /// the type of a field of a kernel struct, looking into anonymous
    /// members, recording the access for relocation.
    ///
    /// Integers, enums, pointers and integer arrays of at most 256 bytes are
    /// read. An embedded struct reached through a `kptr` is not: its address is
    /// the value, as a kernel pointer. A pointer to anything but a struct or
    /// union is an address, a `u64`
    fn kernel_field(
        &mut self,
        expr: ExprId,
        composite: u32,
        name: &Ident,
        read: KernelRead,
    ) -> TyId {
        let Some(btf) = self.db.options.btf.clone() else {
            return self.db.types.error;
        };
        let struct_name = btf.name(composite);
        let Some((path, member, offset)) = btf.find_member(composite, &name.text) else {
            return self.error(
                Diagnostic::error(format!("`{struct_name}` has no field `{}`", name.text))
                    .with_code(codes::UNKNOWN_FIELD)
                    .with_primary(name.span, "not in the kernel's BTF"),
            );
        };
        if member.bitfield_size != 0 {
            return self.error(
                Diagnostic::error(format!("`{}` is a bitfield", name.text))
                    .with_code(codes::NOT_SUPPORTED_YET)
                    .with_primary(name.span, "bitfields cannot be read yet"),
            );
        }
        let resolved = btf.resolve(member.ty);
        let (ty, read) = match btf.get(resolved) {
            vigil_btf::Type::Int {
                size,
                signed,
                boolean,
                ..
            } => {
                let int = match (size, signed) {
                    _ if *boolean => None,
                    (1, false) => Some(IntTy::U8),
                    (1, true) => Some(IntTy::I8),
                    (2, false) => Some(IntTy::U16),
                    (2, true) => Some(IntTy::I16),
                    (4, false) => Some(IntTy::U32),
                    (4, true) => Some(IntTy::I32),
                    (8, false) => Some(IntTy::U64),
                    (8, true) => Some(IntTy::I64),
                    _ => return self.unsupported_field(name, &btf.name(member.ty)),
                };
                match int {
                    Some(int) => (self.db.types.int(int), read),
                    None => (self.db.types.bool, read),
                }
            }
            vigil_btf::Type::Enum { size, signed, .. } => {
                let int = match (size, signed) {
                    (1, _) => IntTy::U8,
                    (2, _) => IntTy::U16,
                    (8, true) => IntTy::I64,
                    (8, false) => IntTy::U64,
                    (_, true) => IntTy::I32,
                    _ => IntTy::U32,
                };
                (self.db.types.int(int), read)
            }
            vigil_btf::Type::Ptr(target) => match btf.composite(*target) {
                Some((id, _)) => {
                    let kernel = self.db.kernel_ty(id);
                    (self.db.types.intern(TyKind::KPtr(kernel)), read)
                }
                // other pointers (to strings, to functions) are addresses
                None => (self.db.types.int(IntTy::U64), read),
            },
            vigil_btf::Type::Array { element, len } => {
                let element = btf.resolve(*element);
                let int = match btf.get(element) {
                    vigil_btf::Type::Int {
                        size: 1,
                        signed: false,
                        ..
                    } => IntTy::U8,
                    vigil_btf::Type::Int {
                        size: 1,
                        signed: true,
                        ..
                    } => IntTy::I8,
                    vigil_btf::Type::Int {
                        size: 2, signed, ..
                    } => {
                        if *signed {
                            IntTy::I16
                        } else {
                            IntTy::U16
                        }
                    }
                    vigil_btf::Type::Int {
                        size: 4, signed, ..
                    } => {
                        if *signed {
                            IntTy::I32
                        } else {
                            IntTy::U32
                        }
                    }
                    vigil_btf::Type::Int {
                        size: 8, signed, ..
                    } => {
                        if *signed {
                            IntTy::I64
                        } else {
                            IntTy::U64
                        }
                    }
                    _ => return self.unsupported_field(name, &btf.name(member.ty)),
                };
                if *len == 0 || btf.size(resolved) > 256 {
                    return self.unsupported_field(name, &btf.name(member.ty));
                }
                let int = self.db.types.int(int);
                (self.db.types.array(int, u64::from(*len)), read)
            }
            vigil_btf::Type::Struct(_) | vigil_btf::Type::Union(_) if read == KernelRead::Probe => {
                let kernel = self.db.kernel_ty(resolved);
                (
                    self.db.types.intern(TyKind::KPtr(kernel)),
                    KernelRead::Address,
                )
            }
            _ => return self.unsupported_field(name, &btf.name(member.ty)),
        };
        let index = self.result.kernel_accesses.len() as u32;
        self.result.kernel_accesses.push(KernelAccess {
            composite,
            path,
            offset,
            read,
        });
        self.result.res.insert(expr, Res::KernelField(index));
        ty
    }

    /// reporting a field of a kernel struct whose type cannot be read
    fn unsupported_field(&mut self, name: &Ident, ty: &str) -> TyId {
        self.error(
            Diagnostic::error(format!(
                "field `{}` of type `{ty}` cannot be read",
                name.text
            ))
            .with_code(codes::NOT_SUPPORTED_YET)
            .with_primary(name.span, "unsupported kernel type")
            .with_note(
                "Vigil reads integers, enums, pointers, arrays of integers and embedded structs",
            ),
        )
    }

    /// the method of a struct for an operator, if it has one with a
    /// signature
    fn operator_method(&self, ty: TyId, name: &str) -> Option<FnId> {
        let TyKind::Struct(s) = self.kind(ty) else {
            return None;
        };
        let package = self.db.package;
        package.structs[s]
            .methods
            .iter()
            .copied()
            .find(|&m| package.fns[m].name.text == name && self.db.fn_sigs.get(m).is_some())
    }

    /// the type of a call of an operator method, checking the other operand,
    /// if any, against the parameter after `self`
    fn operator_call(&mut self, method: FnId, operand: Option<ExprId>, span: Span) -> TyId {
        let sig = self.db.fn_sigs[method].clone();
        match (operand, sig.call_params()) {
            (Some(operand), [(_, ty)]) => self.check(operand, *ty),
            (None, []) => {}
            _ => {
                let name = &self.db.package.fns[method].name;
                let expected = if operand.is_some() {
                    "`self` and one value"
                } else {
                    "only `self`"
                };
                return self.error(
                    Diagnostic::error(format!(
                        "`{}` has the wrong number of parameters",
                        name.text
                    ))
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(span, format!("this operator calls `{}`", name.text))
                    .with_secondary(name.span, format!("it should take {expected}")),
                );
            }
        }
        sig.ret
    }

    /// if a type is the context of a packet program (XDP or TC)
    fn packet_context(&self, ty: TyId) -> bool {
        matches!(
            self.kind(ty),
            TyKind::Struct(s) if self.db.struct_contexts.get(s).is_some_and(|c| c.has_packet())
        )
    }

    /// the type of `ctx.parse[Header](offset)`: a reference to the header at
    /// `offset` bytes into the packet, or `None` when the packet is too short
    fn packet_parse(
        &mut self,
        callee: ExprId,
        indices: &[ExprId],
        args: &[Arg],
        span: Span,
    ) -> TyId {
        let header = match indices {
            [index] => self
                .expr_segments(*index)
                .and_then(|segments| self.resolve_path(&segments)),
            _ => None,
        };
        let Some(Path::Struct(header)) = header else {
            self.skip_args(args);
            return self.error(
                Diagnostic::error("`parse` takes the type of a header")
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(span, "write `ctx.parse[Header](offset)`"),
            );
        };
        let header_ty = self.db.types.intern(TyKind::Struct(header));
        if !self.db.storable(header_ty) {
            self.skip_args(args);
            let name = self.show(header_ty);
            return self.error(
                Diagnostic::error(format!("`{name}` cannot be read from a packet"))
                    .with_code(codes::WRONG_KIND)
                    .with_primary(span, "headers are structs of integers and arrays"),
            );
        }
        self.result.res.insert(callee, Res::PacketParse(header));
        let u32_ty = self.db.types.int(IntTy::U32);
        self.args(
            span,
            &[("offset".to_string(), u32_ty)],
            args,
            "function",
            None,
        );
        let reference = self.db.types.intern(TyKind::Ref(header_ty));
        let result = self.db.types.option(reference);
        self.result.expr_types.insert(callee, result);
        result
    }

    /// the type of `map[index]` on an array map: an `Option` of a reference
    /// to the slot
    fn map_index(&mut self, base: ExprId, m: MapId, indices: &[ExprId], span: Span) -> TyId {
        self.result.res.insert(base, Res::Map(m));
        let Some(info) = self.db.maps.get(m).copied() else {
            for &i in indices {
                self.infer(i, None);
            }
            return self.db.types.error;
        };
        if !info.kind.is_array() {
            for &i in indices {
                self.infer(i, None);
            }
            let name = &self.db.package.maps[m].name.text;
            return self.error(
                Diagnostic::error("only array maps can be indexed")
                    .with_code(codes::WRONG_KIND)
                    .with_primary(span, format!("use `{name}.get(key)`")),
            );
        }
        let [index] = indices else {
            for &i in indices {
                self.infer(i, None);
            }
            return self.error(
                Diagnostic::error("maps take one index")
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(span, format!("found {} indices", indices.len())),
            );
        };
        let index_ty = self.infer(*index, None);
        if !self.is_int(index_ty) {
            let found = self.show(index_ty);
            self.error(
                Diagnostic::error(format!("map indices are integers, found `{found}`"))
                    .with_code(codes::MISMATCHED_TYPES)
                    .with_primary(self.span(*index), "not an integer"),
            );
        }
        let value_ref = self.db.types.intern(TyKind::Ref(info.value));
        self.db.types.option(value_ref)
    }

    /// reporting a call of a value that is not a function
    fn not_callable(&mut self, ty: TyId, callee: ExprId) -> TyId {
        if self.db.types.is_error(ty) {
            return ty;
        }
        let found = self.show(ty);
        self.error(
            Diagnostic::error(format!("cannot call a value of type `{found}`"))
                .with_code(codes::WRONG_KIND)
                .with_primary(self.span(callee), "not a function"),
        )
    }

    /// checking the arguments of a call we cannot check as a whole, for the
    /// errors inside them. Returns the error type
    fn skip_args(&mut self, args: &[Arg]) -> TyId {
        for arg in args {
            self.infer(arg.value, None);
        }
        self.db.types.error
    }

    /// the type of a call of a generic function without generic arguments.
    /// We find them by matching the written parameter types with the argument
    /// types (`argmax(v)` with `v: Vector[i32, 3]`) and then check the call
    /// against the instance
    fn inferred_generic_call(&mut self, f: FnId, callee: ExprId, args: &[Arg], span: Span) -> TyId {
        let package = self.db.package;
        let data = &package.fns[f];
        let names: Vec<String> = data
            .generic_params
            .iter()
            .map(|p| p.name.text.clone())
            .collect();
        let tys: Vec<TyId> = args.iter().map(|a| self.infer(a.value, None)).collect();
        let mut bound: Vec<Option<GenericArgValue>> = vec![None; names.len()];
        let mut unknown = false;
        for (i, arg) in args.iter().enumerate() {
            let param = match &arg.name {
                Some(name) => data.params.iter().find(|p| p.name.text == name.text),
                None => data.params.get(i),
            };
            if let (Some(name), None) = (&arg.name, param) {
                unknown = true;
                self.unknown_argument(name);
            }
            // arguments are inferred without a hint, so integer literals keep their
            // variable: in `u32[N]` only `N` is bound, and the literals become `u32`
            // when checked against the instance
            if let Some(ty) = param.and_then(|p| p.ty.as_ref()) {
                self.bind_generic(data.module, ty, tys[i], &names, &mut bound);
            }
        }
        let name = &data.name.text;
        let Some(values) = bound.iter().cloned().collect::<Option<Vec<_>>>() else {
            // an unknown argument is already reported, and may be the one
            // that would tell the parameter
            if unknown {
                return self.db.types.error;
            }
            let missing = names[bound.iter().position(Option::is_none).unwrap_or(0)].clone();
            return self.error(
                Diagnostic::error(format!("cannot infer `{missing}` of `{name}`"))
                    .with_code(codes::CANNOT_INFER)
                    .with_primary(
                        self.span(callee),
                        format!("the arguments of this call do not tell `{missing}`"),
                    )
                    .with_help(format!(
                        "give its generic arguments, as in `{name}[...](...)`"
                    )),
            );
        };
        let callee_span = self.span(callee);
        let generic_args = values.into_iter().map(|v| (v, callee_span)).collect();
        let Some(instance_args) =
            self.db
                .check_generic_args(Item::Fn(f), generic_args, callee_span)
        else {
            return self.db.types.error;
        };
        // an instance not in the package yet exists in the next run of the
        // analysis, until then the call has the error type
        let Some(Item::Fn(instance)) = self.db.instantiate(Item::Fn(f), instance_args) else {
            return self.db.types.error;
        };
        self.result.res.insert(callee, Res::Fn(instance));
        let sig: FnSig = self.db.fn_sigs[instance].clone();
        let params = sig.call_params();
        let mut filled = vec![false; params.len()];
        // where each parameter was given, for the same checks as `args_with`
        let mut given: Vec<Option<Span>> = vec![None; params.len()];
        let mut positional = true;
        for (i, arg) in args.iter().enumerate() {
            let value_span = self.span(arg.value);
            let slot = match &arg.name {
                Some(name) => {
                    positional = false;
                    params.iter().position(|(p, _)| *p == name.text)
                }
                None if !positional => {
                    self.error(
                        Diagnostic::error("positional argument after a keyword argument")
                            .with_code(codes::BAD_ARGUMENTS)
                            .with_primary(value_span, "move it before the keyword arguments"),
                    );
                    continue;
                }
                None => Some(i),
            };
            if let Some(first) = slot.and_then(|s| given.get(s).copied().flatten()) {
                let slot = slot.unwrap_or_default();
                self.error(
                    Diagnostic::error(format!("argument `{}` given twice", params[slot].0))
                        .with_code(codes::BAD_ARGUMENTS)
                        .with_primary(value_span, "given again here")
                        .with_secondary(first, "first given here"),
                );
                continue;
            }
            if let Some(s) = slot.filter(|s| *s < given.len()) {
                given[s] = Some(value_span);
            }
            match slot.and_then(|s| params.get(s).map(|p| (s, p.1))) {
                Some((slot, expected)) => {
                    filled[slot] = true;
                    if !self.unify(tys[i], expected) {
                        self.mismatch(self.span(arg.value), expected, tys[i]);
                    }
                }
                // reported above
                None if arg.name.is_some() => {}
                None => {
                    self.error(
                        Diagnostic::error(format!(
                            "too many arguments: this function takes {}",
                            params.len()
                        ))
                        .with_code(codes::BAD_ARGUMENTS)
                        .with_primary(self.span(arg.value), "unexpected argument")
                        .with_secondary(data.name.span, format!("`{name}` declared here")),
                    );
                }
            }
        }
        // missing parameters are found from what was filled, not from the
        // number of arguments, which counts a repeated one twice
        let optional = self.optional_params(f);
        let unfilled = filled
            .iter()
            .enumerate()
            .any(|(i, f)| !*f && !optional.get(i).copied().unwrap_or(false));
        if unfilled {
            let missing: Vec<&str> = params
                .iter()
                .zip(&filled)
                .enumerate()
                .filter(|(i, (_, f))| !**f && !optional.get(*i).copied().unwrap_or(false))
                .map(|(_, ((name, _), _))| name.as_str())
                .collect();
            self.missing_arguments(span, &missing, Some(&data.name));
        }
        sig.ret
    }

    /// binding the generic parameters `names` that appear in `pattern`, the
    /// written type of a parameter, from the argument type `ty`.
    ///
    /// The first binding of a parameter wins, a later one that disagrees shows
    /// up as a mismatch against the instance. Nothing is bound where the type
    /// does not have the shape of the pattern
    fn bind_generic(
        &mut self,
        module: ModuleId,
        pattern: &TypeRef,
        ty: TyId,
        names: &[String],
        bound: &mut [Option<GenericArgValue>],
    ) {
        let param_index = |path: &PathRef| match path.segments.as_slice() {
            [segment] => names.iter().position(|n| *n == segment.text),
            _ => None,
        };
        // a generic alias is matched as its body, with the written arguments in
        // place of its parameters
        if let TypeRef::Path { path, args, .. } = pattern
            && !args.is_empty()
            && let Some((Def::Item(Item::Alias(a)), used)) =
                self.db
                    .scopes
                    .resolve_prefix(self.db.package, module, &path.segments)
            && used == path.segments.len()
        {
            let alias = &self.db.package.aliases[a];
            if alias.generic_params.len() == args.len() {
                let params: Vec<(String, TypeArgRef)> = alias
                    .generic_params
                    .iter()
                    .map(|p| p.name.text.clone())
                    .zip(args.iter().cloned())
                    .collect();
                let expanded = subst_type(&alias.ty, &params);
                self.bind_generic(alias.module, &expanded, ty, names, bound);
            }
            return;
        }
        match pattern {
            TypeRef::Path { path, args, .. } if args.is_empty() => {
                if let Some(i) = param_index(path)
                    && bound[i].is_none()
                {
                    // a type parameter bound only by literals is `i64`, so here we default
                    // the integer variable now and not at the end
                    let ty = self.zonk(ty);
                    bound[i] = Some(GenericArgValue::Type(ty));
                }
            }
            TypeRef::Path { path, args, span } => {
                let [segment] = path.segments.as_slice() else {
                    return;
                };
                let generic = match self.db.scopes.lookup(module, &segment.text) {
                    Some(Def::Item(item @ (Item::Struct(_) | Item::Enum(_)))) => Some(item),
                    _ => None,
                };
                let Some(generic) = generic else {
                    // `T[N]`, written like a generic application, is an array
                    if let ([length], TyKind::Array(element_ty, n)) =
                        (args.as_slice(), self.kind(ty))
                    {
                        let element = TypeRef::Path {
                            path: path.clone(),
                            args: Vec::new(),
                            span: *span,
                        };
                        self.bind_generic(module, &element, element_ty, names, bound);
                        let length = match length {
                            TypeArgRef::Type(TypeRef::Path { path, args, .. })
                                if args.is_empty() =>
                            {
                                Some(path)
                            }
                            TypeArgRef::Const(ConstExpr::Path(path)) => Some(path),
                            _ => None,
                        };
                        if let Some(i) = length.and_then(param_index)
                            && bound[i].is_none()
                        {
                            bound[i] = Some(GenericArgValue::Const(i128::from(n)));
                        }
                    }
                    return;
                };
                let package = self.db.package;
                let instance = match self.kind(ty) {
                    TyKind::Struct(s) => &package.structs[s].instance,
                    TyKind::Enum(e) => &package.enums[e].instance,
                    _ => return,
                };
                let Some(instance) = instance else {
                    return;
                };
                if instance.generic != generic {
                    return;
                }
                for (arg, value) in args.iter().zip(instance.args.clone()) {
                    match (arg, value) {
                        (TypeArgRef::Type(inner), InstanceArg::Ty(key)) => {
                            let inner_ty = self.db.types.from_key(&key);
                            self.bind_generic(module, inner, inner_ty, names, bound);
                        }
                        // a constant parameter, written like a type
                        (
                            TypeArgRef::Type(TypeRef::Path { path, args, .. }),
                            InstanceArg::Const(v),
                        ) if args.is_empty() => {
                            if let Some(i) = param_index(path)
                                && bound[i].is_none()
                            {
                                bound[i] = Some(GenericArgValue::Const(v));
                            }
                        }
                        (TypeArgRef::Const(ConstExpr::Path(path)), InstanceArg::Const(v)) => {
                            if let Some(i) = param_index(path)
                                && bound[i].is_none()
                            {
                                bound[i] = Some(GenericArgValue::Const(v));
                            }
                        }
                        _ => {}
                    }
                }
            }
            TypeRef::Array(element, len, _) => {
                if let TyKind::Array(element_ty, n) = self.kind(ty) {
                    self.bind_generic(module, element, element_ty, names, bound);
                    if let ConstExpr::Path(path) = len
                        && let Some(i) = param_index(path)
                        && bound[i].is_none()
                    {
                        bound[i] = Some(GenericArgValue::Const(i128::from(n)));
                    }
                }
            }
            TypeRef::Option(inner, _) => {
                if let TyKind::Option(inner_ty) = self.kind(ty) {
                    self.bind_generic(module, inner, inner_ty, names, bound);
                }
            }
            TypeRef::Missing => {}
        }
    }

    /// which parameters, apart from `self`, a call may leave out: the ones
    /// with a default value
    fn optional_params(&self, f: FnId) -> Vec<bool> {
        self.db.package.fns[f]
            .params
            .iter()
            .filter(|p| p.name.text != "self")
            .map(|p| p.default.is_some())
            .collect()
    }

    /// checking the arguments of a call with no default parameters. `what`
    /// names the callee in messages
    fn args(
        &mut self,
        span: Span,
        params: &[(String, TyId)],
        args: &[Arg],
        what: &str,
        declared: Option<&Ident>,
    ) {
        self.args_with(span, params, &[], args, what, declared);
    }

    /// checking the arguments of a call: positional ones by position, then
    /// keyword ones by name. `optional` says which parameters have a default,
    /// and `what` names the callee (the fields of a `"variant"` have no names).
    /// `declared` is the name of the callee where it is declared, if it has
    /// one. An argument matching no parameter is still inferred, for its
    /// errors
    fn args_with(
        &mut self,
        span: Span,
        params: &[(String, TyId)],
        optional: &[bool],
        args: &[Arg],
        what: &str,
        declared: Option<&Ident>,
    ) {
        let declared_here = |diag: Diagnostic| match declared {
            Some(name) => diag.with_secondary(name.span, format!("`{}` declared here", name.text)),
            None => diag,
        };
        let mut filled: Vec<Option<Span>> = vec![None; params.len()];
        let mut positional = true;
        for (i, arg) in args.iter().enumerate() {
            let value_span = self.span(arg.value);
            let slot = match &arg.name {
                None if !positional => {
                    self.infer(arg.value, None);
                    self.error(
                        Diagnostic::error("positional argument after a keyword argument")
                            .with_code(codes::BAD_ARGUMENTS)
                            .with_primary(value_span, "move it before the keyword arguments"),
                    );
                    continue;
                }
                None if i >= params.len() => {
                    self.infer(arg.value, None);
                    self.error(declared_here(
                        Diagnostic::error(format!(
                            "too many arguments: this {what} takes {}",
                            params.len()
                        ))
                        .with_code(codes::BAD_ARGUMENTS)
                        .with_primary(value_span, "unexpected argument"),
                    ));
                    continue;
                }
                None => i,
                Some(name) => {
                    positional = false;
                    if what == "variant" {
                        self.infer(arg.value, None);
                        self.error(
                            Diagnostic::error("variant fields have no names")
                                .with_code(codes::BAD_ARGUMENTS)
                                .with_primary(name.span, "pass the values in order"),
                        );
                        continue;
                    }
                    match params.iter().position(|(p, _)| *p == name.text) {
                        Some(slot) => slot,
                        None => {
                            self.infer(arg.value, None);
                            self.unknown_argument(name);
                            continue;
                        }
                    }
                }
            };
            if let Some(first) = filled[slot] {
                self.infer(arg.value, None);
                self.error(
                    Diagnostic::error(format!("argument `{}` given twice", params[slot].0))
                        .with_code(codes::BAD_ARGUMENTS)
                        .with_primary(value_span, "given again here")
                        .with_secondary(first, "first given here"),
                );
                continue;
            }
            filled[slot] = Some(value_span);
            self.check(arg.value, params[slot].1);
        }
        let missing: Vec<_> = params
            .iter()
            .zip(&filled)
            .enumerate()
            .filter(|(i, (_, f))| f.is_none() && !optional.get(*i).copied().unwrap_or(false))
            .map(|(_, ((name, _), _))| name.as_str())
            .collect();
        if missing.is_empty() {
        } else if what == "variant" {
            self.error(declared_here(
                Diagnostic::error("wrong number of values")
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(
                        span,
                        format!("expected {}, found {}", params.len(), args.len()),
                    ),
            ));
        } else {
            self.missing_arguments(span, &missing, declared);
        }
    }

    /// reporting an argument named after no parameter
    fn unknown_argument(&mut self, name: &Ident) {
        self.error(
            Diagnostic::error(format!("unknown argument `{}`", name.text))
                .with_code(codes::BAD_ARGUMENTS)
                .with_primary(name.span, "no parameter with this name"),
        );
    }

    /// reporting the parameters a call of `declared` leaves without a value
    fn missing_arguments(&mut self, span: Span, missing: &[&str], declared: Option<&Ident>) {
        let names: Vec<_> = missing.iter().map(|n| format!("`{n}`")).collect();
        let message = match names.as_slice() {
            [one] => format!("missing argument {one}"),
            _ => format!("missing arguments {}", names.join(", ")),
        };
        let given: Vec<_> = missing.iter().map(|n| format!("{n}=...")).collect();
        let mut diag = Diagnostic::error(message)
            .with_code(codes::BAD_ARGUMENTS)
            .with_primary(span, format!("add `{}`", given.join(", ")));
        if let Some(name) = declared {
            diag = diag.with_secondary(name.span, format!("`{}` declared here", name.text));
        }
        self.error(diag);
    }

    /// the type of a call of a builtin, written `name`
    fn builtin_call(
        &mut self,
        builtin: Builtin,
        name: &str,
        args: &[Arg],
        expected: Option<TyId>,
        span: Span,
    ) -> TyId {
        let positional = |cx: &mut Self, n: usize, usage: &str| -> bool {
            if args.len() == n && args.iter().all(|a| a.name.is_none()) {
                return true;
            }
            cx.skip_args(args);
            cx.error(
                Diagnostic::error(format!("wrong arguments for `{name}`"))
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(span, format!("expected {usage}")),
            );
            false
        };
        match builtin {
            Builtin::Some => {
                if !positional(self, 1, "one value, as in `Some(x)`") {
                    return self.db.types.error;
                }
                let inner = match expected.map(|e| self.kind(e)) {
                    Some(TyKind::Option(inner)) => {
                        self.check(args[0].value, inner);
                        inner
                    }
                    _ => self.infer(args[0].value, None),
                };
                self.db.types.option(inner)
            }
            Builtin::Int(int) => {
                if !positional(self, 1, &format!("one value, as in `{}(x)`", int.name())) {
                    return self.db.types.int(int);
                }
                let from = self.infer(args[0].value, None);
                let pointer = matches!(self.kind(from), TyKind::KPtr(_));
                let float = self.kind(from) == TyKind::Float;
                if !self.is_int(from) && self.kind(from) != TyKind::Bool && !pointer && !float {
                    let found = self.show(from);
                    self.error(
                        Diagnostic::error(format!("cannot convert `{found}` to `{}`", int.name()))
                            .with_code(codes::MISMATCHED_TYPES)
                            .with_primary(
                                self.span(args[0].value),
                                "expected an integer or `bool`",
                            ),
                    );
                }
                self.db.types.int(int)
            }
            Builtin::Print => {
                if positional(self, 1, "a string, as in `print(f\"x = {x}\")`") {
                    self.in_print = true;
                    self.check(args[0].value, self.db.types.str);
                    self.in_print = false;
                    let size = self.print_format_size(args[0].value);
                    if size > MAX_PRINT_FORMAT {
                        self.error(
                            Diagnostic::error("this message is too long for `print`")
                                .with_code(codes::CODEGEN_LIMIT)
                                .with_primary(
                                    self.span(args[0].value),
                                    format!(
                                        "a format string of {size} bytes, at most {MAX_PRINT_FORMAT}"
                                    ),
                                )
                                .with_note("the format string is built on the 512-byte stack")
                                .with_help("split the message into several `print` calls"),
                        );
                    }
                    if let Expr::FStr(parts) = &self.body.exprs[args[0].value] {
                        let values = parts
                            .iter()
                            .filter(|p| matches!(p, FStrPart::Expr(_)))
                            .count();
                        if values > MAX_PRINT_ARGS {
                            self.error(
                                Diagnostic::error(format!("`print` takes at most {MAX_PRINT_ARGS} values"))
                                    .with_code(codes::BAD_ARGUMENTS)
                                    .with_primary(self.span(args[0].value), format!("{values} values"))
                                    .with_note("`print` uses `bpf_printk`, which formats at most three values; send an event for more"),
                            );
                        }
                    }
                }
                self.db.types.unit
            }
            Builtin::Min | Builtin::Max | Builtin::SaturatingAdd | Builtin::SaturatingSub => {
                if !positional(self, 2, "two integers") {
                    return self.db.types.error;
                }
                let ty = self.infer(args[0].value, expected);
                if !self.int_or_unknown(ty) {
                    self.infer(args[1].value, None);
                    return self.bad_operand(name, ty, self.span(args[0].value));
                }
                self.check(args[1].value, ty);
                if !self.is_int(ty) {
                    return self.bad_operand(name, ty, self.span(args[0].value));
                }
                ty
            }
            Builtin::Clamp => {
                if !positional(self, 3, "three integers, as in `clamp(x, lo, hi)`") {
                    return self.db.types.error;
                }
                let ty = self.infer(args[0].value, expected);
                if !self.int_or_unknown(ty) {
                    self.infer(args[1].value, None);
                    self.infer(args[2].value, None);
                    return self.bad_operand(name, ty, self.span(args[0].value));
                }
                self.check(args[1].value, ty);
                self.check(args[2].value, ty);
                if !self.is_int(ty) {
                    return self.bad_operand(name, ty, self.span(args[0].value));
                }
                ty
            }
            Builtin::Old => {
                if !self.in_contract {
                    self.skip_args(args);
                    return self.error(
                        Diagnostic::error("`old` is only allowed in contracts")
                            .with_code(codes::BAD_CONTRACT)
                            .with_primary(span, "not in a contract"),
                    );
                }
                if !positional(self, 1, "one value, as in `old(self)`") {
                    return self.db.types.error;
                }
                self.infer(args[0].value, expected)
            }
            Builtin::IsSome | Builtin::IsNone | Builtin::Value => {
                if !self.in_contract {
                    self.skip_args(args);
                    self.error(
                        Diagnostic::error(format!("`{name}` is only allowed in contracts"))
                            .with_code(codes::BAD_CONTRACT)
                            .with_primary(span, "not in a contract")
                            .with_help("in code, use `match` or `?`"),
                    );
                    return self.db.types.error;
                }
                if !positional(self, 1, &format!("one `Option`, as in `{name}(x)`")) {
                    return self.db.types.error;
                }
                let ty = self.infer(args[0].value, None);
                let inner = match self.kind(ty) {
                    TyKind::Option(inner) => inner,
                    TyKind::Error => return ty,
                    TyKind::Var(_) => {
                        let inner = self.var(false);
                        let option = self.db.types.option(inner);
                        self.unify(ty, option);
                        inner
                    }
                    _ => {
                        let found = self.show(ty);
                        return self.error(
                            Diagnostic::error("mismatched types")
                                .with_code(codes::MISMATCHED_TYPES)
                                .with_primary(
                                    self.span(args[0].value),
                                    format!("expected an `Option`, found `{found}`"),
                                ),
                        );
                    }
                };
                if builtin == Builtin::Value {
                    inner
                } else {
                    self.db.types.bool
                }
            }
            Builtin::Stage => {
                if !args.is_empty() {
                    self.skip_args(args);
                    self.error(
                        Diagnostic::error("`stage` takes no arguments")
                            .with_code(codes::BAD_ARGUMENTS)
                            .with_primary(span, "write `stage()`"),
                    );
                }
                self.result.stage_calls.push(span);
                self.db.types.unit
            }
            Builtin::Run => {
                self.result.test_calls.push((span, "run"));
                if !positional(
                    self,
                    2,
                    "a program and a packet, as in `run(firewall, pkt)`",
                ) {
                    return self.db.types.error;
                }
                let target = args[0].value;
                // the program is named, not evaluated. A function is not a value, so
                // the name gets the unit type and is not inferred
                self.result.expr_types.insert(target, self.db.types.unit);
                let program = self
                    .expr_segments(target)
                    .and_then(|segments| self.resolve_path(&segments));
                let packet = self.infer(args[1].value, None);
                let damaged = match self.kind(packet) {
                    TyKind::Struct(st) => {
                        let data = &self.db.package.structs[st];
                        let module = &self.db.package.modules[data.module];
                        let testing = module.is_std() && module.path == ["std", "testing"];
                        match data.name.text.as_str() {
                            "Packet" if testing => Some(false),
                            "Damaged" if testing => Some(true),
                            _ => None,
                        }
                    }
                    TyKind::Error => Some(false),
                    _ => None,
                };
                let Some(damaged) = damaged else {
                    let found = self.show(packet);
                    return self.error(
                        Diagnostic::error(format!(
                            "`run` takes a packet of `std.testing`, not `{found}`"
                        ))
                        .with_code(codes::MISMATCHED_TYPES)
                        .with_primary(self.span(args[1].value), "expected `Packet` or `Damaged`"),
                    );
                };
                match program {
                    Some(Path::Fn(f))
                        if matches!(
                            self.db.programs.get(f).map(|p| p.kind),
                            Some(ProgramKind::Xdp | ProgramKind::Tc)
                        ) =>
                    {
                        self.result.runs.insert(target, (f, damaged));
                        self.db.fn_sigs[f].ret
                    }
                    Some(Path::Error) => self.db.types.error,
                    _ => self.error(
                        Diagnostic::error("`run` runs an XDP or TC program of the project")
                            .with_code(codes::WRONG_KIND)
                            .with_primary(self.span(target), "not a packet program")
                            .with_help("name the program, as in `run(firewall, pkt)`"),
                    ),
                }
            }
            Builtin::SetTime => {
                self.result.test_calls.push((span, "set_time"));
                if positional(self, 1, "nanoseconds, as in `set_time(1_000_000)`") {
                    let u64_ty = self.db.types.int(IntTy::U64);
                    self.check(args[0].value, u64_ty);
                }
                self.db.types.unit
            }
            Builtin::Range => {
                self.skip_args(args);
                self.error(
                    Diagnostic::error("`range` can only be iterated by a `for` loop")
                        .with_code(codes::WRONG_KIND)
                        .with_primary(span, "not a value"),
                )
            }
            Builtin::KPtr => {
                self.skip_args(args);
                self.error(
                    Diagnostic::error("`kptr` needs the type it points to")
                        .with_code(codes::WRONG_KIND)
                        .with_primary(span, "write `kptr[task_struct](address)`"),
                )
            }
            Builtin::F64 | Builtin::Math(_) if !self.comptime => {
                self.skip_args(args);
                self.float_error(span)
            }
            Builtin::F64 => {
                if positional(self, 1, "one value, as in `f64(x)`") {
                    let from = self.infer(args[0].value, None);
                    if !self.is_int(from) && self.kind(from) != TyKind::Float {
                        let found = self.show(from);
                        self.error(
                            Diagnostic::error(format!("cannot convert `{found}` to `f64`"))
                                .with_code(codes::MISMATCHED_TYPES)
                                .with_primary(self.span(args[0].value), "expected a number"),
                        );
                    }
                }
                self.db.types.float()
            }
            Builtin::Math(_) => {
                let float = self.db.types.float();
                if positional(self, 1, "one `f64`") {
                    self.check(args[0].value, float);
                }
                float
            }
            Builtin::Bool | Builtin::Str | Builtin::Bytes | Builtin::Option => {
                self.skip_args(args);
                let help = match builtin {
                    Builtin::Bool => Some("compare instead, as in `x != 0`"),
                    Builtin::Option => Some("write `Some(x)` or `None`"),
                    _ => None,
                };
                let mut diag = Diagnostic::error(format!("`{name}` cannot be called"))
                    .with_code(codes::WRONG_KIND)
                    .with_primary(span, "a type without a conversion");
                if let Some(help) = help {
                    diag = diag.with_help(help);
                }
                self.error(diag)
            }
        }
    }

    /// checking a value in a format string: an integer, a `bool` or a byte
    /// string (`bytes[N]`, printed up to its first NUL)
    fn printable(&mut self, expr: ExprId) {
        let ty = self.infer(expr, None);
        let bytes = match self.kind(ty) {
            TyKind::Array(element, _) => {
                matches!(self.kind(element), TyKind::Int(IntTy::U8))
            }
            _ => false,
        };
        if !self.is_int(ty) && self.kind(ty) != TyKind::Bool && !bytes {
            let found = self.show(ty);
            self.error(
                Diagnostic::error(format!("cannot format a value of type `{found}`"))
                    .with_code(codes::MISMATCHED_TYPES)
                    .with_primary(self.span(expr), "expected an integer, `bool` or `bytes[N]`"),
            );
        }
    }

    /// the type of `lhs op rhs` when both sides have type `ty`, checking that
    /// the operator applies. `/` is only for `f64` and `//` only for integers.
    /// `==` and `!=` apply to integers, `bool`, and enums without variant fields
    fn binary_result(&mut self, op: BinaryOp, ty: TyId, span: Span, operand: Span) -> TyId {
        use BinaryOp::*;
        if self.kind(ty) == TyKind::Float {
            return match op {
                Add | Sub | Mul | Div => ty,
                Lt | Le | Gt | Ge | Eq | Ne => self.db.types.bool,
                _ => self.bad_operand(op.symbol(), ty, operand),
            };
        }
        match op {
            Div => {
                let found = self.show(ty);
                self.error(
                    Diagnostic::error("`/` is not defined on integers")
                        .with_code(codes::BAD_OPERAND)
                        .with_primary(span, format!("the operands are `{found}`"))
                        .with_help("use `//` for integer division"),
                )
            }
            Add | Sub | Mul | FloorDiv | Rem | BitAnd | BitOr | BitXor | Shl | Shr => {
                if self.is_int(ty) {
                    ty
                } else {
                    self.bad_operand(op.symbol(), ty, operand)
                }
            }
            Lt | Le | Gt | Ge => {
                if !self.is_int(ty) {
                    self.bad_operand(op.symbol(), ty, operand);
                }
                self.db.types.bool
            }
            Eq | Ne => {
                if !self.comparable(ty) {
                    self.bad_operand(op.symbol(), ty, operand);
                }
                self.db.types.bool
            }
            And | Or => self.db.types.bool,
        }
    }

    /// if `==` and `!=` apply to a type: integers, `bool`, and enums without
    /// variant fields
    fn comparable(&self, ty: TyId) -> bool {
        match self.kind(ty) {
            TyKind::Int(_) | TyKind::Bool | TyKind::Error => true,
            TyKind::Enum(e) => self.db.variant_fields[e].iter().all(Option::is_none),
            _ => self.is_int(ty),
        }
    }

    /// if [`Self::binary_result`] accepts operands of type `ty` for `op`, or
    /// the type is not known yet. When it does not, the other operand is not
    /// checked against `ty`, which would report the same mistake twice
    fn operator_applies(&self, op: BinaryOp, ty: TyId) -> bool {
        use BinaryOp::*;
        match self.kind(ty) {
            TyKind::Var(_) | TyKind::Error => true,
            TyKind::Float => matches!(op, Add | Sub | Mul | Div | Lt | Le | Gt | Ge | Eq | Ne),
            _ if self.is_int(ty) => true,
            _ => matches!(op, Eq | Ne) && self.comparable(ty),
        }
    }

    /// if a type is an integer or not known yet
    fn int_or_unknown(&self, ty: TyId) -> bool {
        matches!(self.kind(ty), TyKind::Var(_) | TyKind::Error) || self.is_int(ty)
    }

    /// reporting an `f64` in run-time code
    fn float_error(&mut self, span: Span) -> TyId {
        self.error(
            Diagnostic::error("floating-point numbers are only allowed in comptime code")
                .with_code(codes::NOT_SUPPORTED_YET)
                .with_primary(span, "the verifier rejects floating point")
                .with_help(
                    "compute the value in a `comptime def`, or use `fixed[I, F]` from `std.fixed`",
                ),
        )
    }

    /// reporting an operator on a type it does not apply to, unless the type
    /// is already an error
    fn bad_operand(&mut self, op: &str, ty: TyId, span: Span) -> TyId {
        if self.db.types.is_error(ty) {
            return ty;
        }
        let found = self.show(ty);
        let label = match op {
            "==" | "!=" => format!("`{op}` applies to integers, `bool` and enums without fields"),
            _ if op.starts_with(char::is_alphabetic) => format!("`{op}` takes integers"),
            _ => format!("`{op}` applies to integers"),
        };
        let mut diag = Diagnostic::error(format!("`{op}` cannot be applied to `{found}`"))
            .with_code(codes::BAD_OPERAND)
            .with_primary(span, label);
        // a struct gets operators from methods
        if let TyKind::Struct(_) = self.kind(ty)
            && let Some(method) = operator_method_name(op)
        {
            diag = diag.with_help(format!(
                "define `{method}` on `{found}` to use `{op}` on its values"
            ));
        } else if let TyKind::Option(_) = self.kind(ty) {
            diag = diag.with_help(match op {
                "==" | "!=" => "test an `Option` with `match`, as in `case None:`",
                _ => "get the value first, with `?` or `match`",
            });
        }
        self.error(diag)
    }

    // patterns
    // -------------------------

    /// checking a pattern against its type, binding its names in the current
    /// scope. `in_or` is set inside an alternative of `|`, where a name cannot
    /// be bound, because it would have no value when another alternative
    /// matches
    fn pat(&mut self, id: PatId, expected: TyId, in_or: bool) {
        let span = self.spans.pats[id];
        match &self.body.pats[id] {
            Pat::Missing | Pat::Wild => {}
            Pat::Bind(name) => {
                if in_or {
                    self.error(
                        Diagnostic::error("alternatives cannot bind names")
                            .with_code(codes::BAD_PATTERN)
                            .with_primary(span, "use a separate `case`"),
                    );
                }
                // a bare name binds, as in Python, even when a constant with that name
                // is in scope. It matches everything, which is rarely wanted, so we report
                // it
                if let Some(Def::Item(Item::Const(_))) =
                    self.db.scopes.lookup(self.module, &name.text)
                {
                    self.error(
                        Diagnostic::error(format!("`{}` binds a new name here", name.text))
                            .with_code(codes::BAD_PATTERN)
                            .with_primary(span, "this pattern matches every value")
                            .with_note(format!(
                                "a name in a pattern captures the value; it does not compare with the constant `{}`",
                                name.text
                            ))
                            .with_help("compare with a dotted name (`module.NAME`) or a guard (`case x if x == NAME:`)"),
                    );
                }
                let local = self.declare(name, expected);
                self.result.pat_locals.insert(id, local);
            }
            Pat::Int { value, negative } => {
                let int = self.var(true);
                if !self.unify(int, expected) {
                    self.pat_mismatch(span, expected, "an integer");
                    return;
                }
                if let TyKind::Int(int) = self.kind(expected) {
                    let value = i128::try_from(*value).unwrap_or(i128::MAX);
                    let value = if *negative { -value } else { value };
                    if !int.contains(value) {
                        self.db.diags.push(out_of_range(value, int, span));
                    }
                }
            }
            Pat::Bool(_) => {
                if !self.unify(self.db.types.bool, expected) {
                    self.pat_mismatch(span, expected, "a `bool`");
                }
            }
            Pat::Str | Pat::Float => {
                self.error(
                    Diagnostic::error("strings and floats cannot be matched")
                        .with_code(codes::BAD_PATTERN)
                        .with_primary(span, "unsupported pattern"),
                );
            }
            Pat::None => {
                let inner = self.var(false);
                let option = self.db.types.option(inner);
                if !self.unify(option, expected) {
                    self.pat_mismatch(span, expected, "an `Option`");
                }
            }
            Pat::Path(path) => match self.resolve_path(&path.segments) {
                Some(Path::Variant(e, i))
                    if self.instance_variant(e, i, Some(expected)).is_none() =>
                {
                    let name = self.db.package.enums[e].name.text.clone();
                    self.pat_mismatch(span, expected, &format!("a `{name}`"));
                }
                Some(Path::Variant(e, i)) => {
                    // a variant of a generic enum is the one of the matched instance
                    let e = self.instance_variant(e, i, Some(expected)).unwrap_or(e);
                    self.result.pat_res.insert(id, Res::Variant(e, i));
                    let enum_ty = self.db.types.intern(TyKind::Enum(e));
                    if !self.unify(enum_ty, expected) {
                        let name = self.show(enum_ty);
                        self.pat_mismatch(span, expected, &format!("`{name}`"));
                    } else if let Some(fields) = &self.db.variant_fields[e][i as usize] {
                        let wildcards = vec!["_"; fields.len()].join(", ");
                        self.error(
                            Diagnostic::error(format!("variant `{path}` has fields"))
                                .with_code(codes::BAD_PATTERN)
                                .with_primary(span, format!("write `{path}({wildcards})`")),
                        );
                    }
                }
                Some(Path::Const(c)) => {
                    self.result.pat_res.insert(id, Res::Const(c));
                    let ty = self.db.const_info(c).0;
                    if !self.unify(ty, expected) {
                        let name = self.show(ty);
                        self.pat_mismatch(span, expected, &format!("`{name}`"));
                    }
                }
                Some(Path::Error) => {}
                Some(_) => {
                    self.error(
                        Diagnostic::error(format!("`{path}` cannot be used in a pattern"))
                            .with_code(codes::BAD_PATTERN)
                            .with_primary(span, "expected a variant or a constant"),
                    );
                }
                None => {
                    self.unresolved(&path.to_string(), span);
                }
            },
            Pat::TupleStruct { path, fields } => {
                let fields = fields.clone();
                let field_tys: Vec<TyId> = match self.resolve_path(&path.segments) {
                    Some(Path::Builtin(Builtin::Some)) => {
                        self.result.pat_res.insert(id, Res::Builtin(Builtin::Some));
                        let inner = self.var(false);
                        let option = self.db.types.option(inner);
                        if !self.unify(option, expected) {
                            self.pat_mismatch(span, expected, "an `Option`");
                        }
                        vec![inner]
                    }
                    Some(Path::Variant(e, i))
                        if self.instance_variant(e, i, Some(expected)).is_none() =>
                    {
                        let name = self.db.package.enums[e].name.text.clone();
                        self.pat_mismatch(span, expected, &format!("a `{name}`"));
                        vec![self.db.types.error; fields.len()]
                    }
                    Some(Path::Variant(e, i)) => {
                        let e = self.instance_variant(e, i, Some(expected)).unwrap_or(e);
                        self.result.pat_res.insert(id, Res::Variant(e, i));
                        let enum_ty = self.db.types.intern(TyKind::Enum(e));
                        if !self.unify(enum_ty, expected) {
                            let name = self.show(enum_ty);
                            self.pat_mismatch(span, expected, &format!("`{name}`"));
                        }
                        match self.db.variant_fields[e][i as usize].clone() {
                            Some(tys) => tys,
                            None => {
                                self.error(
                                    Diagnostic::error(format!("variant `{path}` has no fields"))
                                        .with_code(codes::BAD_PATTERN)
                                        .with_primary(span, "remove the parentheses"),
                                );
                                Vec::new()
                            }
                        }
                    }
                    Some(Path::Error) => Vec::new(),
                    None => {
                        self.unresolved(&path.to_string(), span);
                        Vec::new()
                    }
                    Some(_) => {
                        self.error(
                            Diagnostic::error(format!("`{path}` is not a variant"))
                                .with_code(codes::BAD_PATTERN)
                                .with_primary(span, "expected a variant with fields"),
                        );
                        Vec::new()
                    }
                };
                if !field_tys.is_empty() && field_tys.len() != fields.len() {
                    self.error(
                        Diagnostic::error("wrong number of fields")
                            .with_code(codes::BAD_PATTERN)
                            .with_primary(
                                span,
                                format!("expected {}, found {}", field_tys.len(), fields.len()),
                            ),
                    );
                }
                for (i, &field) in fields.iter().enumerate() {
                    let ty = field_tys.get(i).copied().unwrap_or(self.db.types.error);
                    self.pat(field, ty, in_or);
                }
            }
            Pat::Or(alternatives) => {
                for &alternative in alternatives {
                    self.pat(alternative, expected, true);
                }
            }
        }
    }

    /// reporting a pattern that cannot match the expected type
    fn pat_mismatch(&mut self, span: Span, expected: TyId, found: &str) {
        let expected = self.show(expected);
        self.error(
            Diagnostic::error("mismatched types")
                .with_code(codes::MISMATCHED_TYPES)
                .with_primary(span, format!("expected `{expected}`, found {found}")),
        );
    }

    // exhaustiveness
    // -------------------------

    /// checking if the arms cover every value of `ty`, reporting when they do
    /// not and warning about arms that never match.
    ///
    /// Only the outermost constructor is analysed: a variant is covered when
    /// all its fields are irrefutable, and an arm with a guard covers nothing.
    /// A type without constructors (an integer, a struct) is covered only by a
    /// wildcard or a name
    fn exhaustiveness(&mut self, ty: TyId, arms: &[MatchArm], span: Span) -> bool {
        let package = self.db.package;
        let constructors: Option<Vec<String>> = match self.kind(ty) {
            TyKind::Enum(e) => Some(
                package.enums[e]
                    .variants
                    .iter()
                    .map(|v| {
                        let name = format!("{}.{}", package.enums[e].name.text, v.name.text);
                        match &v.fields {
                            Some(fields) => {
                                format!("{name}({})", vec!["_"; fields.len()].join(", "))
                            }
                            None => name,
                        }
                    })
                    .collect(),
            ),
            TyKind::Option(_) => Some(vec!["None".into(), "Some(_)".into()]),
            TyKind::Bool => Some(vec!["False".into(), "True".into()]),
            TyKind::Error => return true,
            _ => None,
        };
        let total = constructors.as_ref().map_or(0, Vec::len);
        let mut covered = vec![false; total];
        // the pattern of the arm that matches every value, once there is one
        let mut all: Option<Span> = None;
        for arm in arms {
            let coverage = self.coverage(arm.pat);
            if all.is_some()
                || (!coverage.0
                    && !coverage.1.is_empty()
                    && coverage.1.iter().all(|&c| c < total && covered[c]))
            {
                let pat_span = self.spans.pats[arm.pat];
                let mut diag = Diagnostic::warning("unreachable pattern")
                    .with_code(codes::UNREACHABLE_PATTERN);
                diag = match all {
                    Some(earlier) => diag
                        .with_primary(pat_span, "never matched")
                        .with_secondary(earlier, "this arm matches every value"),
                    None => diag.with_primary(pat_span, "earlier arms already match these values"),
                };
                self.db.diags.push(diag.with_help(
                    "remove this arm, or move it before the arms that match its values",
                ));
                continue;
            }
            if arm.guard.is_some() {
                continue;
            }
            if coverage.0 {
                all = Some(self.spans.pats[arm.pat]);
            }
            for c in coverage.1 {
                if c < total {
                    covered[c] = true;
                }
            }
        }
        if all.is_some() || (total > 0 && covered.iter().all(|&c| c)) {
            return true;
        }
        let (label, help) = match constructors {
            Some(names) => {
                let missing: Vec<_> = names.iter().zip(&covered).filter(|(_, c)| !**c).collect();
                let quoted: Vec<_> = missing.iter().map(|(n, _)| format!("`{n}`")).collect();
                let help = match missing.as_slice() {
                    [(one, _)] => format!("add `case {one}:`, or `case _:` for every other value"),
                    _ => "add a `case` for each, or `case _:` for every other value".to_string(),
                };
                (format!("not covered: {}", quoted.join(", ")), help)
            }
            None => (
                format!("not every `{}` is matched", self.show(ty)),
                "add a `case _:` arm for the remaining values".to_string(),
            ),
        };
        self.error(
            Diagnostic::error("non-exhaustive match")
                .with_code(codes::NON_EXHAUSTIVE)
                .with_primary(span, label)
                .with_help(help),
        );
        false
    }

    /// what a pattern covers: everything, or the indices of constructors
    /// (variants, `None`/`Some`, `False`/`True`). A constructor with refutable
    /// fields covers nothing
    fn coverage(&self, pat: PatId) -> (bool, Vec<usize>) {
        match &self.body.pats[pat] {
            Pat::Wild | Pat::Bind(_) | Pat::Missing => (true, Vec::new()),
            Pat::Bool(b) => (false, vec![usize::from(*b)]),
            Pat::None => (false, vec![0]),
            Pat::Path(_) | Pat::TupleStruct { .. } => {
                let (path, fields) = match &self.body.pats[pat] {
                    Pat::Path(p) => (p, &[][..]),
                    Pat::TupleStruct { path, fields } => (path, &fields[..]),
                    _ => unreachable!(),
                };
                let irrefutable = fields.iter().all(|&f| self.coverage(f).0);
                if !irrefutable {
                    return (false, Vec::new());
                }
                let package = self.db.package;
                match self
                    .db
                    .scopes
                    .resolve_prefix(package, self.module, &path.segments)
                {
                    Some((Def::Builtin(Builtin::Some), 1)) => (false, vec![1]),
                    Some((Def::Item(Item::Enum(e)), used)) if used + 1 == path.segments.len() => {
                        let name = &path.segments[used].text;
                        let index = package.enums[e]
                            .variants
                            .iter()
                            .position(|v| v.name.text == *name);
                        (false, index.into_iter().collect())
                    }
                    _ => (false, Vec::new()),
                }
            }
            Pat::Or(alternatives) => {
                let mut all = false;
                let mut constructors = Vec::new();
                for &alternative in alternatives {
                    let (a, c) = self.coverage(alternative);
                    all |= a;
                    constructors.extend(c);
                }
                (all, constructors)
            }
            Pat::Int { .. } | Pat::Str | Pat::Float => (false, Vec::new()),
        }
    }
}

/// the method name of a binary operator on structs, as in Python. `and`
/// and `or` have none, they cannot be overloaded
fn binary_method(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "__add__",
        BinaryOp::Sub => "__sub__",
        BinaryOp::Mul => "__mul__",
        BinaryOp::Div => "__truediv__",
        BinaryOp::FloorDiv => "__floordiv__",
        BinaryOp::Rem => "__mod__",
        BinaryOp::BitAnd => "__and__",
        BinaryOp::BitOr => "__or__",
        BinaryOp::BitXor => "__xor__",
        BinaryOp::Shl => "__lshift__",
        BinaryOp::Shr => "__rshift__",
        BinaryOp::Eq => "__eq__",
        BinaryOp::Ne => "__ne__",
        BinaryOp::Lt => "__lt__",
        BinaryOp::Le => "__le__",
        BinaryOp::Gt => "__gt__",
        BinaryOp::Ge => "__ge__",
        BinaryOp::And | BinaryOp::Or => "",
    }
}

/// the method that gives a struct the operator written `op`, if any. `-`
/// is both `__sub__` and `__neg__`, so it gives none
fn operator_method_name(op: &str) -> Option<&'static str> {
    Some(match op {
        "~" => "__invert__",
        "+" => "__add__",
        "*" => "__mul__",
        "/" => "__truediv__",
        "//" => "__floordiv__",
        "%" => "__mod__",
        "&" => "__and__",
        "|" => "__or__",
        "^" => "__xor__",
        "<<" => "__lshift__",
        ">>" => "__rshift__",
        "==" => "__eq__",
        "!=" => "__ne__",
        "<" => "__lt__",
        "<=" => "__le__",
        ">" => "__gt__",
        ">=" => "__ge__",
        _ => return None,
    })
}

/// how a unary operator is written
fn op_symbol(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Not => "not",
        UnaryOp::Neg => "-",
        UnaryOp::BitNot => "~",
    }
}
