//! the semantic analysis of Vigil: lowering to the HIR, name resolution and
//! type checking.
//!
//! The entry points are [`analyze`] and [`analyze_with`], which give an
//! [`Analysis`] of a package and its diagnostics, and [`evaluate`], which runs
//! a function in the interpreter of comptime code.
//!
//! Each file is lowered to the HIR ([`hir`]), then the signatures are resolved
//! ([`db`]) and each body is checked ([`infer`]). The checks that need every
//! body at once come last.

pub mod comptime;
pub mod db;
pub mod generics;
pub mod hir;
pub mod ide;
pub mod infer;
mod lower;
pub mod maps;
pub mod program;
pub mod resolve;
pub mod stdlib;
pub mod ty;
pub mod walk;

use la_arena::ArenaMap;
use vigil_diagnostics::{Diagnostics, FileId};
use vigil_syntax::ast;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use vigil_btf::Btf;

use vigil_diagnostics::{Diagnostic, Span, codes};

use crate::comptime::{CValue, EvalError, Interp};
use crate::db::{ConstValue, Db, FnSig, Value};
use crate::hir::{ConstId, EnumId, FnId, MapId, ModuleData, Package, StmtId, StructId};
use crate::infer::{InferenceResult, LocalId, Res};
use crate::maps::MapInfo;
use crate::program::{ContextKind, ProgramInfo};
use crate::resolve::Scopes;
use crate::ty::{TyId, TyKind, Types};

/// a source file to analyze, with its module path (`["net", "parse"]` for
/// `src/net/parse.vg`)
pub struct ModuleInput {
    pub file: FileId,
    pub path: Vec<String>,
    pub tree: ast::SourceFile,
    /// whether the module belongs to the standard library (the built-in one
    /// or a `std` dependency), which gives it the privileges of `@helper`
    /// and `@context`. Another module whose path starts with `std` is an
    /// error, it would mix with the standard library
    pub std: bool,
}

/// the result of analysing a package, with everything the later phases need
///
/// The tables are keyed by HIR ids. An entry is missing for an item that was
/// skipped, like a generic template or a function whose body was never needed.
#[derive(Debug)]
pub struct Analysis {
    pub package: Package,
    pub types: Types,
    pub scopes: Scopes,
    pub struct_fields: ArenaMap<StructId, Vec<TyId>>,
    pub variant_fields: ArenaMap<EnumId, Vec<Option<Vec<TyId>>>>,
    pub fn_sigs: ArenaMap<FnId, FnSig>,
    pub struct_contexts: ArenaMap<StructId, ContextKind>,
    pub network_order: ArenaMap<StructId, ()>,
    pub helpers: ArenaMap<FnId, u32>,
    pub programs: ArenaMap<FnId, ProgramInfo>,
    pub maps: ArenaMap<MapId, MapInfo>,
    /// the kernel's types, when the program imports some
    pub btf: Option<Arc<Btf>>,
    pub consts: ArenaMap<ConstId, (TyId, Option<ConstValue>)>,
    /// the values of the constants computed by comptime functions
    pub comptime_consts: ArenaMap<ConstId, CValue>,
    /// what the `comptime:` blocks of each function compute, see [`BlockValues`]
    pub comptime_blocks: ArenaMap<FnId, BlockValues>,
    /// the `stage()` statements that split each program, in order
    pub stages: ArenaMap<FnId, Vec<StmtId>>,
    pub bodies: ArenaMap<FnId, InferenceResult>,
    /// the functions that use `run`, `set_time`, or `count` and `last` of
    /// events, directly or not. Only the simulator of the full Vigil runs them
    pub runs_programs: HashSet<FnId>,
}

/// running a function in the interpreter of comptime code. It returns the
/// result and the final value of the first parameter (the `self` of a `@mut`
/// method).
///
/// The interpreter has the semantics of the compiled code. An index out of
/// bounds is an error, and so is a helper, a map or the packet.
pub fn evaluate(an: &Analysis, f: FnId, args: Vec<CValue>) -> Result<(CValue, CValue), String> {
    let mut interp = Interp::new(
        &an.package,
        &an.bodies,
        &an.types,
        &an.consts,
        &an.comptime_consts,
        &an.fn_sigs,
    );
    interp.call_keeping_self(f, args).map_err(|e| e.message)
}

/// what the target kernels allow. The default suits a lone file: Linux 6.1,
/// no kernel types and the host's architecture
#[derive(Clone, Debug)]
pub struct Options {
    /// the oldest target kernel, which decides the map kinds and helpers we
    /// may use
    pub kernel: vigil_target::KernelVersion,
    /// the kernel's types, for `from kernel import ...`
    pub btf: Option<Arc<Btf>>,
    /// the architecture of the target kernels. It says in which registers a
    /// probed function gets its arguments
    pub arch: vigil_target::Arch,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            kernel: vigil_target::Profile::default_profile().baseline(),
            btf: None,
            arch: vigil_target::Arch::host(),
        }
    }
}

/// a module of the user whose path starts with `std`. Its items would join
/// the namespace of the standard library, or silently replace a module of
/// it, so we reject the name. It gets no privilege of the standard library
/// either way. The error points at the start of the file, because the name
/// comes from the path of the file and not from its text
fn std_name_clash(input: &ModuleInput) -> Diagnostic {
    let name = input.path.join(".");
    let help = if input.path.len() == 1 {
        "rename `std.vg`"
    } else {
        "rename the directory `std/`"
    };
    Diagnostic::error(format!(
        "the module `{name}` has the name of the standard library"
    ))
    .with_code(codes::DUPLICATE_DEFINITION)
    .with_primary(Span::new(input.file, 0..0), "in this file")
    .with_note("`std` and the modules under it are the standard library")
    .with_help(help)
}

/// the analysis of a package, for the default [`Options`]
pub fn analyze(inputs: &[ModuleInput]) -> (Analysis, Diagnostics) {
    analyze_with(inputs, Options::default())
}

/// the analysis of a package for the given target, with the diagnostics
/// sorted by location
pub fn analyze_with(inputs: &[ModuleInput], options: Options) -> (Analysis, Diagnostics) {
    let mut diags = Diagnostics::new();
    let mut package = Package::default();
    for input in inputs {
        if !input.std && input.path.first().is_some_and(|s| s == "std") {
            diags.push(std_name_clash(input));
        }
        let module = package.modules.alloc(ModuleData {
            path: input.path.clone(),
            file: input.file,
            items: Vec::new(),
            imports: Vec::new(),
            unsupported: Vec::new(),
            std: input.std,
        });
        lower::lower_module(&mut package, module, input.file, &input.tree, &mut diags);
    }

    // here we add the instances of generic items in waves. Each check asks
    // for the instances it uses, until none is missing. The diagnostics of
    // these checks are dropped, the final check below repeats them. The
    // limit on the number of items stops an instance that requires another
    // without end (`S[N]` containing `S[N + 1]`).
    let generic = package
        .structs
        .iter()
        .any(|(_, s)| !s.generic_params.is_empty())
        || package
            .enums
            .iter()
            .any(|(_, e)| !e.generic_params.is_empty())
        || package
            .fns
            .iter()
            .any(|(_, f)| !f.generic_params.is_empty());
    if generic {
        for _ in 0..MAX_WAVES {
            let requests = {
                let mut db = Db::new(&package, options.clone(), Diagnostics::new());
                check_package(&mut db, &package);
                db.requests
            };
            let instances = package.structs.len() + package.fns.len();
            if requests.is_empty() || instances > generics::MAX_INSTANCES * 4 {
                break;
            }
            for (item, args) in requests {
                generics::create_instance(&mut package, item, args);
            }
        }
    }

    let mut db = Db::new(&package, options, diags);
    let (bodies, consts) = check_package(&mut db, &package);
    if let Some(&(item, _)) = db.requests.first() {
        // a request left after the last round is one of the items that grow
        let name = resolve::item_name(&package, item);
        db.diags.push(
            Diagnostic::error(format!("too many instances of `{}`", name.text))
                .with_code(codes::CYCLE)
                .with_primary(name.span, "its instances require more instances without end")
                .with_note(format!(
                    "the compiler stops after about {} instances of generic items",
                    generics::MAX_INSTANCES * 4
                ))
                .with_help("make sure an instance does not need another with bigger arguments, as `S[N]` containing `S[N + 1]`"),
        );
    }

    check_recursion(&package, &bodies, &mut db.diags);
    check_contract_purity(&package, &bodies, &db.helpers, &mut db.diags);
    check_helper_kernels(
        &package,
        &bodies,
        &db.helpers,
        db.options.kernel,
        &mut db.diags,
    );
    let runs_programs = check_test_only(&package, &bodies, &db.programs, &mut db.diags);
    let comptime_consts = evaluate_comptime(&mut db, &bodies, &consts);
    let comptime_blocks = evaluate_comptime_blocks(&mut db, &bodies, &consts, &comptime_consts);
    let stages = collect_stages(&mut db, &bodies);
    check_tail_kinds(&mut db, &bodies);
    for diag in db.scopes.take_private_uses() {
        db.diags.push(diag);
    }

    let Db {
        types,
        scopes,
        struct_fields,
        variant_fields,
        fn_sigs,
        struct_contexts,
        network_order,
        helpers,
        programs,
        maps,
        options,
        mut diags,
        ..
    } = db;
    diags.sort_by_location();
    // every instance of a template repeats the template's errors at the
    // same place
    diags.dedup();
    let analysis = Analysis {
        package,
        types,
        scopes,
        struct_fields,
        variant_fields,
        fn_sigs,
        struct_contexts,
        network_order,
        helpers,
        programs,
        maps,
        btf: options.btf,
        consts,
        comptime_consts,
        comptime_blocks,
        stages,
        bodies,
        runs_programs,
    };
    (analysis, diags)
}

/// the maximum number of stages of a program on `kernel`: one more than the
/// tail calls the kernel follows (`MAX_TAIL_CALL_CNT`, 32 before 5.17 and 33
/// after), because each stage after the first is reached by one
fn max_stages(kernel: vigil_target::KernelVersion) -> usize {
    let tail_calls =
        vigil_target::Profile::new(vec![kernel], true).map_or(32, |p| p.limits().max_tail_calls);
    tail_calls as usize + 1
}

/// collecting the `stage()` statements that split each program
///
/// A `stage()` must be at the top of a program's body, not in a loop or a
/// branch, and there are at most [`max_stages`] stages. A variable holding a
/// pointer cannot cross a stage, because the next stage is another program.
/// All of these errors are reported here, as is a `stage()` outside a program.
fn collect_stages(
    db: &mut Db,
    bodies: &ArenaMap<FnId, InferenceResult>,
) -> ArenaMap<FnId, Vec<StmtId>> {
    use crate::hir::{Expr, Stmt};
    let package = db.package;
    let mut stages = ArenaMap::default();
    for (f, res) in bodies.iter() {
        if res.stage_calls.is_empty() {
            continue;
        }
        let data = &package.fns[f];
        let body = &data.body;
        if db.programs.get(f).is_none() {
            for &span in &res.stage_calls {
                db.diags.push(
                    Diagnostic::error("`stage()` is only allowed in a program")
                        .with_code(codes::WRONG_KIND)
                        .with_primary(span, format!("`{}` is not a program", data.name.text))
                        .with_secondary(
                            data.name.span,
                            "declared here, without a program decorator",
                        )
                        .with_note("`stage()` splits a program in two, joined by a tail call"),
                );
            }
            continue;
        }
        let is_stage = |stmt: StmtId| match &body.stmts[stmt] {
            Stmt::Expr(e) => matches!(
                &body.exprs[*e],
                Expr::Call { callee, .. }
                    if res.res.get(*callee) == Some(&Res::Builtin(resolve::Builtin::Stage))
            ),
            _ => false,
        };
        let boundaries: Vec<StmtId> = body.root.iter().copied().filter(|&s| is_stage(s)).collect();
        let top: Vec<Span> = boundaries.iter().map(|&s| data.spans.stmts[s]).collect();
        for &span in &res.stage_calls {
            if !top
                .iter()
                .any(|t| t.start <= span.start && span.end <= t.end)
            {
                db.diags.push(
                    Diagnostic::error("`stage()` is a statement of the body of a program")
                        .with_code(codes::WRONG_KIND)
                        .with_primary(span, "not at the top of the program")
                        .with_note("the program is split there: no loop or branch may contain it"),
                );
            }
        }
        let kernel = db.options.kernel;
        let max = max_stages(kernel);
        if boundaries.len() + 1 > max {
            db.diags.push(
                Diagnostic::error(format!("`{}` has too many stages", data.name.text))
                    .with_code(codes::CODEGEN_LIMIT)
                    .with_primary(
                        data.name.span,
                        format!("{} stages, at most {max}", boundaries.len() + 1),
                    )
                    .with_note(format!(
                        "{kernel} follows {} tail calls at most, one per stage after the first",
                        max - 1
                    )),
            );
        }
        // the statements of each stage. A local used on both sides of a
        // boundary is live across it. Parameters are left out.
        let mut segments: Vec<Vec<StmtId>> = vec![Vec::new()];
        for &stmt in &body.root {
            if is_stage(stmt) {
                segments.push(Vec::new());
            } else {
                segments.last_mut().unwrap().push(stmt);
            }
        }
        let params = data.params.len();
        for (b, &boundary) in boundaries.iter().enumerate() {
            let before: std::collections::HashSet<LocalId> = segments[..=b]
                .iter()
                .flat_map(|s| walk::locals(body, res, s))
                .collect();
            let after: std::collections::HashSet<LocalId> = segments[b + 1..]
                .iter()
                .flat_map(|s| walk::locals(body, res, s))
                .collect();
            let mut live: Vec<LocalId> = before
                .intersection(&after)
                .copied()
                .filter(|l| l.into_raw().into_u32() as usize >= params)
                .collect();
            live.sort_by_key(|l| l.into_raw());
            for local in live {
                let ty = res.locals[local].ty;
                if holds_pointer(db, ty) {
                    let name = &res.locals[local].name;
                    db.diags.push(
                        Diagnostic::error(format!("`{name}` cannot go to the next stage"))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(data.spans.stmts[boundary], "the program is split here")
                            .with_secondary(res.locals[local].span, "a pointer, defined here")
                            .with_help("parse the packet or look up the map again after `stage()`"),
                    );
                }
            }
        }
        stages.insert(f, boundaries);
    }
    stages
}

/// reporting the program arrays whose programs have different types: the
/// kernel refuses a tail call between them. The users of an array are the
/// programs that jump through it, known by the context they pass, then the
/// programs placed in it. The first one decides the type.
fn check_tail_kinds(db: &mut Db, bodies: &ArenaMap<FnId, InferenceResult>) {
    use crate::hir::Expr;
    use crate::program::ContextKind;
    let package = db.package;
    let mut users: HashMap<&str, Vec<(ContextKind, Span, String)>> = HashMap::new();
    for (f, res) in bodies.iter() {
        let data = &package.fns[f];
        for (_, expr) in data.body.exprs.iter() {
            let Expr::Call { callee, args } = expr else {
                continue;
            };
            let Some(Res::MapMethod(m, maps::MapMethod::TailCall)) = res.res.get(*callee) else {
                continue;
            };
            let Some(ctx) = args.first() else {
                continue;
            };
            let context = match db.types.kind(res.expr_types[ctx.value]) {
                TyKind::Struct(s) => db.struct_contexts.get(*s).copied(),
                TyKind::TpContext(_) => Some(ContextKind::Tracepoint),
                _ => None,
            };
            if let Some(context) = context {
                let map = package.maps[*m].name.text.as_str();
                let label = format!("a tail call with context `{}`", context.struct_name());
                let span = data.spans.exprs[ctx.value];
                users.entry(map).or_default().push((context, span, label));
            }
        }
    }
    for (f, info) in db.programs.iter() {
        if let Some((map, _)) = &info.tail {
            let name = &package.fns[f].name;
            let label = format!("`{}` is {}", name.text, info.kind.noun());
            users
                .entry(map.as_str())
                .or_default()
                .push((info.kind.context(), name.span, label));
        }
    }
    let mut maps: Vec<_> = users.into_iter().collect();
    maps.sort_by_key(|(map, _)| *map);
    for (map, users) in maps {
        let (first, at, what) = &users[0];
        for (context, span, label) in &users[1..] {
            if context != first {
                db.diags.push(
                    Diagnostic::error(format!(
                        "the programs of `{map}` must have the same program type"
                    ))
                    .with_code(codes::WRONG_KIND)
                    .with_primary(*span, label.clone())
                    .with_secondary(*at, format!("{what} here"))
                    .with_note("the kernel refuses a tail call to a program of another type"),
                );
            }
        }
    }
}

/// whether a value of the type holds a pointer, directly or in a field
fn holds_pointer(db: &Db, ty: TyId) -> bool {
    match db.types.kind(ty) {
        TyKind::Ref(_) | TyKind::KPtr(_) | TyKind::TpContext(_) => true,
        TyKind::Struct(s) if db.struct_contexts.get(*s).is_some() => true,
        TyKind::Struct(s) => db.struct_fields[*s].iter().any(|&t| holds_pointer(db, t)),
        TyKind::Option(inner) | TyKind::Array(inner, _) => holds_pointer(db, *inner),
        TyKind::Enum(e) => db.variant_fields[*e]
            .iter()
            .flatten()
            .flatten()
            .any(|&t| holds_pointer(db, t)),
        _ => false,
    }
}

/// the values that the `comptime:` blocks of a function give to its
/// variables. For each block, the value after it of every variable of the
/// blocks so far
pub type BlockValues = Vec<(StmtId, Vec<(LocalId, CValue)>)>;

/// running the `comptime:` blocks of the run-time functions. A block the
/// interpreter cannot run is an error
fn evaluate_comptime_blocks(
    db: &mut Db,
    bodies: &ArenaMap<FnId, InferenceResult>,
    consts: &Consts,
    comptime_consts: &ArenaMap<ConstId, CValue>,
) -> ArenaMap<FnId, BlockValues> {
    let package = db.package;
    let mut values = ArenaMap::default();
    // after a type error we run no block, because it could fail again
    if db.diags.has_errors() {
        return values;
    }
    for (f, body) in bodies.iter() {
        if body.comptime_blocks.is_empty() {
            continue;
        }
        let mut interp = Interp::new(
            package,
            bodies,
            &db.types,
            consts,
            comptime_consts,
            &db.fn_sigs,
        );
        match interp.run_blocks(f, &body.comptime_blocks) {
            Ok(result) => {
                values.insert(f, result);
            }
            Err((stmt, error)) => {
                // the error is placed at the expression of the block that
                // fails, or calls what fails
                let span = match error.places.last() {
                    Some(&(g, span)) if g == f => span,
                    _ => package.fns[f].spans.stmts[stmt],
                };
                let diagnostic = Diagnostic::error("cannot run a comptime block")
                    .with_code(codes::CONST_EVAL)
                    .with_primary(span, error.message.clone());
                db.diags
                    .push(eval_details(package, diagnostic, &error, Some(f)));
            }
        }
    }
    values
}

/// computing the constants initialized by calls of comptime functions, plus
/// the constant arrays already computed while checking
///
/// A function may read other such constants, so we retry the calls in rounds
/// until a round computes nothing new. Only the errors of the last round are
/// reported. After a type error anywhere in the package no call is run.
fn evaluate_comptime(
    db: &mut Db,
    bodies: &ArenaMap<FnId, InferenceResult>,
    consts: &Consts,
) -> ArenaMap<ConstId, CValue> {
    let package = db.package;
    let mut values: ArenaMap<ConstId, CValue> = ArenaMap::default();
    for (c, value) in std::mem::take(&mut db.array_consts) {
        values.insert(c, value);
    }
    let mut pending: Vec<(ConstId, FnId, Vec<CValue>)> = Vec::new();
    for (c, f, args) in std::mem::take(&mut db.comptime_consts) {
        let data = &package.consts[c];
        let span = data.value.span().unwrap_or(data.name.span);
        let sig = &db.fn_sigs[f];
        let ty = consts[c].0;
        if db.types.is_error(ty) || db.types.is_error(sig.ret) {
            continue;
        }
        let fn_name = &package.fns[f].name;
        if sig.ret != ty {
            let (ret, ty) = (db.display(sig.ret), db.display(ty));
            let mut diagnostic = Diagnostic::error("mismatched types")
                .with_code(codes::MISMATCHED_TYPES)
                .with_primary(span, format!("returns `{ret}`, but the constant is `{ty}`"));
            if let Some(declared) = data.ty.as_ref().and_then(|t| t.span()) {
                diagnostic = diagnostic.with_secondary(declared, "the type of the constant");
            }
            db.diags.push(diagnostic);
            continue;
        }
        // the arguments as written, for their spans
        let arg_spans: Vec<Option<Span>> = match &data.value {
            hir::ConstExpr::Call(_, written, _) => written.iter().map(|a| a.span()).collect(),
            _ => Vec::new(),
        };
        let arg_span = |i: usize| arg_spans.get(i).copied().flatten().unwrap_or(span);
        // parameters left out take their default values
        let required = sig.defaults.iter().take_while(|d| d.is_none()).count();
        if args.len() > sig.params.len() {
            db.diags.push(
                Diagnostic::error(format!(
                    "too many arguments: this function takes {}",
                    sig.params.len()
                ))
                .with_code(codes::BAD_ARGUMENTS)
                .with_primary(arg_span(sig.params.len()), "unexpected argument")
                .with_secondary(fn_name.span, format!("`{}` declared here", fn_name.text)),
            );
            continue;
        }
        if args.len() < required {
            let names: Vec<String> = sig.params[args.len()..required]
                .iter()
                .map(|(name, _)| format!("`{name}`"))
                .collect();
            let message = match names.as_slice() {
                [one] => format!("missing argument {one}"),
                _ => format!("missing arguments {}", names.join(", ")),
            };
            db.diags.push(
                Diagnostic::error(message)
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(span, "add the missing values, in order")
                    .with_secondary(fn_name.span, format!("`{}` declared here", fn_name.text)),
            );
            continue;
        }
        let mut converted = Vec::new();
        for (i, (arg, &(_, param))) in args.iter().zip(&sig.params).enumerate() {
            let value = match (arg, db.types.kind(param)) {
                (Value::Int(v), TyKind::Int(int)) if int.contains(*v) => Some(CValue::Int(*v)),
                (Value::Int(v), TyKind::Float) => Some(CValue::Float(*v as f64)),
                (Value::Bool(b), TyKind::Bool) => Some(CValue::Bool(*b)),
                _ => None,
            };
            match value {
                Some(value) => converted.push(value),
                None => {
                    let label = match arg {
                        Value::Int(v) if matches!(db.types.kind(param), TyKind::Int(_)) => {
                            format!("{v} does not fit in `{}`", db.display(param))
                        }
                        _ => format!(
                            "expected `{}`, found {}",
                            db.display(param),
                            db::value_kind(arg)
                        ),
                    };
                    db.diags.push(
                        Diagnostic::error("mismatched types")
                            .with_code(codes::MISMATCHED_TYPES)
                            .with_primary(arg_span(i), label),
                    );
                    break;
                }
            }
        }
        if converted.len() == args.len() {
            for default in &sig.defaults[args.len()..] {
                converted.push(match *default {
                    Some(db::ConstValue::Int(v)) => CValue::Int(v),
                    Some(db::ConstValue::Bool(b)) => CValue::Bool(b),
                    // a missing required parameter, already reported above
                    None => break,
                });
            }
            if converted.len() == sig.params.len() {
                pending.push((c, f, converted));
            }
        }
    }
    // after a type error we run no comptime function, as for the blocks,
    // because it would fail again on the same mistake
    if db.diags.has_errors() {
        return values;
    }
    let mut errors: Vec<(ConstId, EvalError)> = Vec::new();
    while !pending.is_empty() {
        errors.clear();
        let before = pending.len();
        let mut rest = Vec::new();
        for (c, f, args) in pending {
            let result = Interp::new(package, bodies, &db.types, consts, &values, &db.fn_sigs)
                .call(f, args.clone());
            match result {
                Ok(value) => {
                    values.insert(c, value);
                }
                Err(error) => {
                    errors.push((c, error));
                    rest.push((c, f, args));
                }
            }
        }
        pending = rest;
        if pending.len() == before {
            break;
        }
    }
    // a constant that needs a constant that failed is not reported, the
    // first error says it all. Constants that need each other are.
    let missing: HashMap<ConstId, Option<ConstId>> =
        errors.iter().map(|(c, e)| (*c, e.missing)).collect();
    let mut reported = Vec::new();
    for (c, error) in &errors {
        let mut chain = vec![*c];
        let mut next = error.missing;
        while let Some(d) = next {
            if chain.contains(&d) {
                break;
            }
            chain.push(d);
            next = missing.get(&d).copied().flatten();
        }
        let cycle = next.is_some();
        // a cycle is reported once, at its first constant
        if error.missing.is_some() && (!cycle || reported.iter().any(|d| chain.contains(d))) {
            continue;
        }
        reported.push(*c);
        let data = &package.consts[*c];
        let span = data.value.span().unwrap_or(data.name.span);
        let mut diagnostic = Diagnostic::error(format!("cannot compute `{}`", data.name.text))
            .with_code(codes::CONST_EVAL)
            .with_primary(span, error.message.clone());
        if cycle {
            let start = chain.iter().position(|&d| Some(d) == next).unwrap_or(0);
            let names: Vec<String> = chain[start..]
                .iter()
                .chain(next.iter())
                .map(|&d| format!("`{}`", package.consts[d].name.text))
                .collect();
            diagnostic = diagnostic.with_note(format!(
                "the values of {} need each other",
                names.join(" → ")
            ));
        }
        db.diags
            .push(eval_details(package, diagnostic, error, None));
    }
    values
}

/// adding to the diagnostic of a failed evaluation where it fails: a label
/// at the failing expression and one at each call that leads to it, with a
/// note naming the functions. `outer` is the function of the primary span,
/// whose place is not labelled again.
fn eval_details(
    package: &Package,
    mut diagnostic: Diagnostic,
    error: &EvalError,
    outer: Option<FnId>,
) -> Diagnostic {
    let mut places: &[(FnId, Span)] = &error.places;
    if let (Some(f), Some(((g, _), rest))) = (outer, places.split_last())
        && f == *g
    {
        places = rest;
    }
    let name = |f: FnId| package.fns[f].name.text.clone();
    for (i, &(f, span)) in places.iter().enumerate() {
        let label = if i == 0 {
            format!("fails here, in `{}`", name(f))
        } else {
            format!("in `{}`, this call fails", name(f))
        };
        diagnostic = diagnostic.with_secondary(span, label);
    }
    if places.len() > 1 {
        let calls: Vec<String> = places
            .iter()
            .rev()
            .map(|&(f, _)| format!("`{}`", name(f)))
            .collect();
        diagnostic = diagnostic.with_note(format!("the failing calls: {}", calls.join(" → ")));
    }
    if let Some(note) = &error.note {
        diagnostic = diagnostic.with_note(note.clone());
    }
    diagnostic
}

/// the types and values of the constants of a package
type Consts = ArenaMap<ConstId, (TyId, Option<ConstValue>)>;

/// the maximum number of rounds of instantiation of generic items
const MAX_WAVES: usize = 16;

/// checking every item of a package that is not a generic template. It
/// returns the checked bodies and the constants
fn check_package(db: &mut Db, package: &Package) -> (ArenaMap<FnId, InferenceResult>, Consts) {
    db.collect_signatures();
    // the methods of an instance are checked only when something calls
    // them, because a method may make no sense for some type arguments. We
    // use a work list, so a method reached from a checked body is checked
    // in turn.
    let lazy = |db: &Db, f: FnId| {
        let data = &package.fns[f];
        data.instance.is_some() && data.generic_params.is_empty() && data.owner.is_some()
            || db.is_template_fn(f)
    };
    let mut bodies: ArenaMap<FnId, InferenceResult> = ArenaMap::default();
    let mut queue: Vec<FnId> = package
        .fns
        .iter()
        .map(|(id, _)| id)
        .filter(|&id| !lazy(db, id))
        .collect();
    queue.reverse();
    while let Some(id) = queue.pop() {
        if bodies.get(id).is_some() || db.is_template_fn(id) {
            continue;
        }
        let result = infer::check_fn(db, id);
        for callee in callees(&result) {
            if bodies.get(callee).is_none() && lazy(db, callee) {
                queue.push(callee);
            }
        }
        bodies.insert(id, result);
    }
    db.enter(None);
    let mut consts = ArenaMap::default();
    for (id, _) in package.consts.iter() {
        consts.insert(id, db.const_info(id));
    }
    (bodies, consts)
}

/// the functions a body calls (by name, as a method or through an operator),
/// sorted and without repetition
pub fn callees(body: &InferenceResult) -> Vec<FnId> {
    let mut callees: Vec<FnId> = body
        .res
        .iter()
        .filter_map(|(_, res)| match res {
            Res::Fn(f) | Res::Method(f) | Res::Operator(f) => Some(*f),
            _ => None,
        })
        .chain(body.aug_operators.iter().map(|(_, &f)| f))
        .collect();
    callees.sort_by_key(|f| f.into_raw());
    callees.dedup();
    callees
}

/// what makes an expression impure, if anything, as a phrase for a message
fn impure_expr(
    body: &hir::Body,
    res: &InferenceResult,
    helpers: &ArenaMap<FnId, u32>,
    package: &Package,
    e: hir::ExprId,
) -> Option<String> {
    use crate::resolve::Builtin;
    if matches!(body.exprs[e], hir::Expr::Try(_)) {
        return Some("`?`, which may leave the function".into());
    }
    Some(match res.res.get(e)? {
        Res::Map(_) | Res::MapMethod(..) => "a map".into(),
        Res::PacketParse(_) | Res::PacketLen => "the packet".into(),
        Res::KernelField(_) => "kernel memory".into(),
        Res::KprobeArg(_) | Res::KprobeRet => "the registers of the probed function".into(),
        Res::Builtin(Builtin::Print) => "`print`".into(),
        Res::Builtin(Builtin::Stage) => "`stage()`".into(),
        Res::Builtin(Builtin::Run) => "`run`, which runs a program".into(),
        Res::Builtin(Builtin::SetTime) => "`set_time`".into(),
        Res::Fn(h) if helpers.get(*h).is_some() => {
            format!("the helper `{}`", package.fns[*h].name.text)
        }
        _ => return None,
    })
}

/// reporting the calls of helpers newer than the oldest target kernel, as
/// the map declarations of `RingBuf` are. The helpers the code generator
/// calls itself are chosen by profile there, so they are not seen here
fn check_helper_kernels(
    package: &Package,
    bodies: &ArenaMap<FnId, InferenceResult>,
    helpers: &ArenaMap<FnId, u32>,
    kernel: vigil_target::KernelVersion,
    diags: &mut vigil_diagnostics::Diagnostics,
) {
    for (f, data) in package.fns.iter() {
        let Some(res) = bodies.get(f) else {
            continue;
        };
        for (e, _) in data.body.exprs.iter() {
            let Some(Res::Fn(h)) = res.res.get(e) else {
                continue;
            };
            let Some(feature) = helpers
                .get(*h)
                .and_then(|&n| vigil_target::Feature::of_helper(n))
            else {
                continue;
            };
            if feature.since() <= kernel {
                continue;
            }
            let name = &package.fns[*h].name.text;
            diags.push(
                Diagnostic::error(format!("the helper `{name}` needs {}", feature.since()))
                    .with_code(codes::UNSUPPORTED_KERNEL)
                    .with_primary(data.spans.exprs[e], "not available in the target kernels")
                    .with_note(format!("the oldest kernel of the project is {kernel}"))
                    .with_help("raise the oldest kernel in `[target] kernels` of vigil.toml"),
            );
        }
    }
}

/// reporting the contracts and loop invariants that are not pure
///
/// A contract only reads its operands and calls pure functions. Recursion is
/// reported elsewhere, so the calls form a DAG and we remember the purity of
/// each function.
fn check_contract_purity(
    package: &Package,
    bodies: &ArenaMap<FnId, InferenceResult>,
    helpers: &ArenaMap<FnId, u32>,
    diags: &mut vigil_diagnostics::Diagnostics,
) {
    /// the first impure operation of a function, or of what it calls
    fn impurity(
        f: FnId,
        package: &Package,
        bodies: &ArenaMap<FnId, InferenceResult>,
        helpers: &ArenaMap<FnId, u32>,
        memo: &mut HashMap<FnId, Option<(Span, String)>>,
    ) -> Option<(Span, String)> {
        if let Some(known) = memo.get(&f) {
            return known.clone();
        }
        // pure until shown otherwise, which ends a cycle. Cycles are
        // reported by `check_recursion`.
        memo.insert(f, None);
        let data = &package.fns[f];
        let found = if helpers.get(f).is_some() {
            Some((data.name.span, format!("the helper `{}`", data.name.text)))
        } else if let Some(res) = bodies.get(f) {
            let direct = data.body.exprs.iter().find_map(|(e, _)| {
                impure_expr(&data.body, res, helpers, package, e)
                    .map(|what| (data.spans.exprs[e], what))
            });
            direct.or_else(|| {
                callees(res)
                    .into_iter()
                    .find_map(|g| impurity(g, package, bodies, helpers, memo))
            })
        } else {
            None
        };
        memo.insert(f, found.clone());
        found
    }

    let mut memo = HashMap::new();
    let mut reported = std::collections::HashSet::new();
    for (f, data) in package.fns.iter() {
        let Some(res) = bodies.get(f) else {
            continue;
        };
        let mut roots: Vec<hir::ExprId> = data.contracts.iter().map(|c| c.expr).collect();
        let mut stmts = Vec::new();
        walk::stmts(&data.body, &data.body.root, &mut stmts);
        for stmt in stmts {
            if let hir::Stmt::For { invariants, .. } | hir::Stmt::While { invariants, .. } =
                &data.body.stmts[stmt]
            {
                roots.extend(invariants);
            }
        }
        for root in roots {
            for e in walk::subexprs(&data.body, root) {
                let span = data.spans.exprs[e];
                let problem = match impure_expr(&data.body, res, helpers, package, e) {
                    Some(what) => Some((format!("uses {what}"), None)),
                    None => match res.res.get(e) {
                        Some(Res::Fn(g) | Res::Method(g) | Res::Operator(g)) => {
                            impurity(*g, package, bodies, helpers, &mut memo).map(|inner| {
                                let name = &package.fns[*g].name.text;
                                (format!("calls `{name}`, which is not pure"), Some(inner))
                            })
                        }
                        _ => None,
                    },
                };
                // the instances of a generic function share its contracts,
                // so we report each place once
                if let Some((label, inner)) = problem
                    && reported.insert(span)
                {
                    let mut diag = Diagnostic::error("a contract must be pure")
                        .with_code(codes::IMPURE_CONTRACT)
                        .with_primary(span, label)
                        .with_note(
                            "contracts and invariants state facts: they cannot read maps, the \
                             packet or the kernel, call helpers, print, or leave with `?`",
                        );
                    if let Some((at, what)) = inner {
                        diag = diag.with_secondary(at, format!("{what} here"));
                    }
                    diags.push(diag);
                    // one error per contract, the outermost cause, because an
                    // expression comes before its subexpressions
                    break;
                }
            }
        }
    }
}

/// the functions that use `run`, `set_time`, or `count` and `last` of events,
/// directly or not. A use reached from a program is an error, because these
/// drive the simulator of the full Vigil and only tests may use them.
fn check_test_only(
    package: &Package,
    bodies: &ArenaMap<FnId, InferenceResult>,
    programs: &ArenaMap<FnId, program::ProgramInfo>,
    diags: &mut vigil_diagnostics::Diagnostics,
) -> HashSet<FnId> {
    fn uses(
        f: FnId,
        bodies: &ArenaMap<FnId, InferenceResult>,
        memo: &mut HashMap<FnId, bool>,
    ) -> bool {
        if let Some(&known) = memo.get(&f) {
            return known;
        }
        // recursion is reported elsewhere, a cycle counts as not using them
        memo.insert(f, false);
        let found = bodies.get(f).is_some_and(|body| {
            !body.test_calls.is_empty() || callees(body).into_iter().any(|g| uses(g, bodies, memo))
        });
        memo.insert(f, found);
        found
    }
    let mut memo = HashMap::new();
    let mut found = HashSet::new();
    for (f, _) in package.fns.iter() {
        if uses(f, bodies, &mut memo) {
            found.insert(f);
        }
    }
    // the functions reached from a program (not a test), each with the first
    // program that reaches it
    let mut reached: HashMap<FnId, FnId> = HashMap::new();
    for (p, info) in programs.iter() {
        if info.kind == program::ProgramKind::Test || !found.contains(&p) {
            continue;
        }
        let mut stack = vec![p];
        while let Some(f) = stack.pop() {
            if !found.contains(&f) || reached.contains_key(&f) {
                continue;
            }
            reached.insert(f, p);
            if let Some(body) = bodies.get(f) {
                stack.extend(callees(body));
            }
        }
    }
    let mut reached: Vec<(FnId, FnId)> = reached.into_iter().collect();
    reached.sort_by_key(|(f, _)| f.into_raw());
    for (f, p) in reached {
        let Some(body) = bodies.get(f) else { continue };
        let program = &package.fns[p].name.text;
        for &(span, name) in &body.test_calls {
            let label = if f == p {
                format!("in the program `{program}`")
            } else {
                format!("reached from the program `{program}`")
            };
            diags.push(
                Diagnostic::error(format!("`{name}` is for tests"))
                    .with_code(codes::TEST_ONLY)
                    .with_primary(span, label)
                    .with_note("`run`, `set_time`, and `count` and `last` of events drive the simulator of `vigil test`: `@test` and `@property` functions, and the functions they call, use them"),
            );
        }
    }
    found
}

/// reporting the recursive functions, with their cycle of calls. The
/// verifier rejects recursion and every call is inlined
fn check_recursion(
    package: &Package,
    bodies: &ArenaMap<FnId, InferenceResult>,
    diags: &mut vigil_diagnostics::Diagnostics,
) {
    #[derive(Clone, Copy, PartialEq)]
    enum State {
        Visiting,
        Done,
    }
    fn visit(
        f: FnId,
        package: &Package,
        bodies: &ArenaMap<FnId, InferenceResult>,
        state: &mut HashMap<FnId, State>,
        stack: &mut Vec<FnId>,
        diags: &mut vigil_diagnostics::Diagnostics,
    ) {
        match state.get(&f) {
            Some(State::Done) => return,
            Some(State::Visiting) => {
                let start = stack.iter().position(|&g| g == f).unwrap_or(0);
                let cycle: Vec<_> = stack[start..]
                    .iter()
                    .chain(std::iter::once(&f))
                    .map(|&g| format!("`{}`", package.fns[g].name.text))
                    .collect();
                let name = &package.fns[f].name;
                // here the note depends on the function, because the
                // verifier never sees a `comptime def`
                let note = if package.fns[f].is_comptime {
                    "recursion is not allowed in Vigil, compile-time functions included; \
                     use a bounded loop instead"
                } else {
                    "the verifier rejects recursion; use a bounded loop instead"
                };
                diags.push(
                    Diagnostic::error(format!("`{}` is recursive", name.text))
                        .with_code(codes::RECURSION)
                        .with_primary(
                            name.span,
                            format!("calls itself through {}", cycle.join(" → ")),
                        )
                        .with_note(note),
                );
                return;
            }
            None => {}
        }
        state.insert(f, State::Visiting);
        stack.push(f);
        if let Some(body) = bodies.get(f) {
            for callee in callees(body) {
                visit(callee, package, bodies, state, stack, diags);
            }
        }
        stack.pop();
        state.insert(f, State::Done);
    }

    let mut state = HashMap::new();
    for (f, _) in package.fns.iter() {
        visit(f, package, bodies, &mut state, &mut Vec::new(), diags);
    }
}
