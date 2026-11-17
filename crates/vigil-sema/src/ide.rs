//! this module implements the editor queries over an [`Analysis`] ([`hover`],
//! [`definition`], [`references`], [`names_at`], [`members_at`], [`symbols`],
//! [`tokens`], ...). They take a file and a byte offset and return plain data.
//! They see only checked bodies, so generic templates and unused instance
//! methods are invisible, and a definition is always the template's.

use vigil_diagnostics::{FileId, Span};

use crate::Analysis;
use crate::hir::{Expr, ExprId, FnId, Item};
use crate::infer::Res;
use crate::ty::TyKind;

/// an expression of a function body, with its span
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct At {
    pub function: FnId,
    pub expr: ExprId,
    pub span: Span,
}

/// does a span of `file` contain `offset`? The end is included, because a
/// cursor right after a name is still on it.
fn contains(span: Span, file: FileId, offset: usize) -> bool {
    span.file == file && span.range().start <= offset && offset <= span.range().end
}

/// the innermost (shortest) expression at `offset` of `file`, in a checked
/// function body
pub fn expr_at(analysis: &Analysis, file: FileId, offset: usize) -> Option<At> {
    let mut best: Option<At> = None;
    for (function, data) in analysis.package.fns.iter() {
        if analysis.bodies.get(function).is_none() {
            continue;
        }
        for (expr, &span) in data.spans.exprs.iter() {
            if contains(span, file, offset)
                && best.is_none_or(|b| span.range().len() < b.span.range().len())
            {
                best = Some(At {
                    function,
                    expr,
                    span,
                });
            }
        }
    }
    best
}

/// a type as shown in an editor
fn type_name(analysis: &Analysis, ty: crate::ty::TyId) -> String {
    analysis.types.display(&analysis.package, ty, &|_| None)
}

/// the signature of a function as written in Vigil, `def f(x: u32) -> bool`,
/// or only the name when there is no signature (a template)
fn signature(analysis: &Analysis, function: FnId) -> String {
    let data = &analysis.package.fns[function];
    match analysis.fn_sigs.get(function) {
        Some(sig) => {
            let params: Vec<String> = sig
                .params
                .iter()
                .map(|(name, ty)| format!("{name}: {}", type_name(analysis, *ty)))
                .collect();
            format!(
                "def {}({}) -> {}",
                data.name.text,
                params.join(", "),
                type_name(analysis, sig.ret)
            )
        }
        None => format!("def {}", data.name.text),
    }
}

/// what to show on hover for the expression at `offset`, with its span: a
/// function signature, the name and type of a variable, constant or field,
/// or else just the type
pub fn hover(analysis: &Analysis, file: FileId, offset: usize) -> Option<(Span, String)> {
    let at = expr_at(analysis, file, offset)?;
    let body = &analysis.bodies[at.function];
    let data = &analysis.package.fns[at.function];
    let ty = body
        .expr_types
        .get(at.expr)
        .map(|&t| type_name(analysis, t));
    let text = match (body.res.get(at.expr), &data.body.exprs[at.expr]) {
        (Some(Res::Fn(f) | Res::Method(f)), _) => signature(analysis, *f),
        (Some(Res::Local(l)), _) => {
            let local = &body.locals[*l];
            format!("{}: {}", local.name, type_name(analysis, local.ty))
        }
        (Some(Res::Const(c)), _) => format!(
            "{}: {}",
            analysis.package.consts[*c].name.text,
            ty.clone().unwrap_or_default()
        ),
        (Some(Res::Map(m) | Res::MapMethod(m, _)), _) => {
            format!("map {}", analysis.package.maps[*m].name.text)
        }
        (_, Expr::Field { name, .. }) => {
            format!("{}: {}", name.text, ty.clone().unwrap_or_default())
        }
        _ => ty?,
    };
    Some((at.span, text))
}

/// go to definition: the span of the name in the definition of what the
/// expression at `offset` names
pub fn definition(analysis: &Analysis, file: FileId, offset: usize) -> Option<Span> {
    let at = expr_at(analysis, file, offset)?;
    definition_of(analysis, at.function, at.expr)
}

/// where the thing named by an expression is defined. For an instance it
/// is the template.
fn definition_of(analysis: &Analysis, function: FnId, expr: ExprId) -> Option<Span> {
    let at = At {
        function,
        expr,
        span: analysis.package.fns[function].spans.exprs[expr],
    };
    let body = &analysis.bodies[at.function];
    let package = &analysis.package;
    let res = body.res.get(at.expr)?;
    Some(match res {
        Res::Local(l) => body.locals[*l].span,
        Res::Const(c) => package.consts[*c].name.span,
        Res::Fn(f) | Res::Method(f) => {
            // an instance has no source of its own, so we point to its
            // generic function
            let f = match package.fns[*f].instance.as_ref().map(|i| i.generic) {
                Some(Item::Fn(g)) => g,
                _ => *f,
            };
            package.fns[f].name.span
        }
        Res::Variant(e, i) => package.enums[*e].variants[*i as usize].name.span,
        Res::StructCtor(s) => package.structs[*s].name.span,
        Res::Map(m) | Res::MapMethod(m, _) => package.maps[*m].name.span,
        Res::Field(i) => {
            let Expr::Field { base, .. } = &package.fns[at.function].body.exprs[at.expr] else {
                return None;
            };
            let mut ty = *body.expr_types.get(*base)?;
            if let TyKind::Ref(inner) = analysis.types.kind(ty) {
                ty = *inner;
            }
            let TyKind::Struct(s) = analysis.types.kind(ty) else {
                return None;
            };
            let s = match package.structs[*s].instance.as_ref().map(|i| i.generic) {
                Some(Item::Struct(g)) => g,
                _ => *s,
            };
            package.structs[s].fields.get(*i as usize)?.name.span
        }
        _ => return None,
    })
}

/// what a completion name stands for
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Local,
    Function,
    Struct,
    Enum,
    Constant,
    Map,
    Module,
    Other,
}

/// the names visible at `offset` of `file`: the locals declared before it,
/// then the items and imports of the module in alphabetical order
pub fn names_at(analysis: &Analysis, file: FileId, offset: usize) -> Vec<(String, Kind)> {
    let mut names = Vec::new();
    // the function around the offset is the last one whose name comes before
    // it. This works because bodies have no recorded end and functions are
    // never nested.
    let around = analysis
        .package
        .fns
        .iter()
        .filter(|(f, d)| {
            d.name.span.file == file
                && d.name.span.range().start <= offset
                && analysis.bodies.get(*f).is_some()
        })
        .max_by_key(|(_, d)| d.name.span.range().start);
    if let Some((f, _)) = around {
        for (_, local) in analysis.bodies[f].locals.iter() {
            if local.span.range().start <= offset && !names.iter().any(|(n, _)| *n == local.name) {
                names.push((local.name.clone(), Kind::Local));
            }
        }
    }
    let module = analysis
        .package
        .modules
        .iter()
        .find(|(_, m)| m.file == file)
        .map(|(id, _)| id);
    if let Some(scope) = module.and_then(|m| analysis.scopes.modules.get(m)) {
        let mut items: Vec<_> = scope
            .iter()
            .map(|(name, (def, _))| {
                let kind = match def {
                    crate::resolve::Def::Item(Item::Fn(_)) => Kind::Function,
                    crate::resolve::Def::Item(Item::Struct(_)) => Kind::Struct,
                    crate::resolve::Def::Item(Item::Enum(_)) => Kind::Enum,
                    crate::resolve::Def::Item(Item::Const(_)) => Kind::Constant,
                    crate::resolve::Def::Item(Item::Map(_)) => Kind::Map,
                    crate::resolve::Def::Namespace(_) => Kind::Module,
                    _ => Kind::Other,
                };
                (name.clone(), kind)
            })
            .collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        names.extend(items);
    }
    names
}

/// what an editor offers after `.`: a field, method, variant, map or context
/// method, or module item
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub name: String,
    pub kind: MemberKind,
    /// a type or signature, shown beside the name
    pub detail: String,
}

/// the kind of a [`Member`] or [`Symbol`]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberKind {
    Field,
    Method,
    Variant,
    Function,
    Struct,
    Enum,
    Constant,
    Map,
    Module,
}

/// the expressions of `file` ending at `end`, longest first. This is what
/// comes right before a `.` or a `(`.
fn exprs_ending_at(analysis: &Analysis, file: FileId, end: usize) -> Vec<At> {
    let mut found = Vec::new();
    for (function, data) in analysis.package.fns.iter() {
        if analysis.bodies.get(function).is_none() || data.name.span.file != file {
            continue;
        }
        for (expr, &span) in data.spans.exprs.iter() {
            if span.file == file && span.range().end == end && !span.range().is_empty() {
                found.push(At {
                    function,
                    expr,
                    span,
                });
            }
        }
    }
    found.sort_by_key(|at| std::cmp::Reverse(at.span.range().len()));
    found
}

/// the signature of a method, with `self` shown without its type
fn method_signature(analysis: &Analysis, f: FnId) -> String {
    signature(analysis, f).replacen("self: ", "", 1)
}

/// the methods of a struct or enum: with `self` (called on values) or
/// without (called on the type)
fn methods(analysis: &Analysis, methods: &[FnId], on_values: bool) -> Vec<Member> {
    methods
        .iter()
        .filter(|&&f| {
            let takes_self = analysis.package.fns[f]
                .params
                .first()
                .is_some_and(|p| p.name.text == "self");
            takes_self == on_values
        })
        .map(|&f| Member {
            name: analysis.package.fns[f].name.text.clone(),
            kind: MemberKind::Method,
            detail: method_signature(analysis, f),
        })
        .collect()
}

/// the members of the expression ending at the `.` at offset `dot`. A map,
/// type or module has its own (map methods, variants, static methods, module
/// items). A value has the fields and methods of its type, and a context
/// has its builtins.
pub fn members_at(analysis: &Analysis, file: FileId, dot: usize) -> Vec<Member> {
    let package = &analysis.package;
    for at in exprs_ending_at(analysis, file, dot) {
        let body = &analysis.bodies[at.function];
        let data = &package.fns[at.function];
        // a name that is not a value (map, type or module), looked up in
        // the module scope
        if let Expr::Name(name) = &data.body.exprs[at.expr]
            && !matches!(body.res.get(at.expr), Some(Res::Local(_) | Res::Const(_)))
            && let Some(def) = analysis.scopes.lookup(data.module, name)
        {
            match def {
                crate::resolve::Def::Item(Item::Map(m)) => {
                    let Some(info) = analysis.maps.get(m) else {
                        return Vec::new();
                    };
                    return crate::maps::MapMethod::names_for(info.kind, true)
                        .into_iter()
                        .map(|name| Member {
                            name: name.to_string(),
                            kind: MemberKind::Method,
                            detail: format!("map method of `{}`", package.maps[m].name.text),
                        })
                        .collect();
                }
                crate::resolve::Def::Item(Item::Enum(e)) => {
                    let mut members: Vec<Member> = package.enums[e]
                        .variants
                        .iter()
                        .map(|v| Member {
                            name: v.name.text.clone(),
                            kind: MemberKind::Variant,
                            detail: format!("{}.{}", package.enums[e].name.text, v.name.text),
                        })
                        .collect();
                    members.extend(methods(analysis, &package.enums[e].methods, false));
                    return members;
                }
                crate::resolve::Def::Item(Item::Struct(s)) => {
                    return methods(analysis, &package.structs[s].methods, false);
                }
                crate::resolve::Def::Namespace(ns) => return module_items(analysis, ns),
                _ => {}
            }
        }
        let Some(&ty) = body.expr_types.get(at.expr) else {
            continue;
        };
        let ty = match analysis.types.kind(ty) {
            TyKind::Ref(inner) => *inner,
            _ => ty,
        };
        return match analysis.types.kind(ty) {
            TyKind::Struct(s) => {
                let s = *s;
                let mut members: Vec<Member> = package.structs[s]
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, field)| Member {
                        name: field.name.text.clone(),
                        kind: MemberKind::Field,
                        detail: analysis
                            .struct_fields
                            .get(s)
                            .and_then(|f| f.get(i))
                            .map_or(String::new(), |&t| type_name(analysis, t)),
                    })
                    .collect();
                members.extend(methods(analysis, &package.structs[s].methods, true));
                // the builtins of program contexts
                let builtins: &[(&str, &str)] = match analysis.struct_contexts.get(s) {
                    Some(c) if c.has_packet() => &[
                        ("parse", "parse[H](offset) -> ref[H]?"),
                        ("len", "len() -> u32"),
                    ],
                    Some(crate::program::ContextKind::Kprobe) => {
                        &[("arg", "arg(n) -> u64"), ("ret", "ret() -> u64")]
                    }
                    _ => &[],
                };
                members.extend(builtins.iter().map(|(name, detail)| Member {
                    name: name.to_string(),
                    kind: MemberKind::Method,
                    detail: detail.to_string(),
                }));
                members
            }
            TyKind::Enum(e) => methods(analysis, &package.enums[*e].methods, true),
            _ => Vec::new(),
        };
    }
    Vec::new()
}

/// the items and submodules of a namespace module (`net.` after
/// `import std.net`), sorted. Names starting with `_` are private and left out.
fn module_items(analysis: &Analysis, ns: crate::resolve::NsId) -> Vec<Member> {
    let namespace = &analysis.scopes.namespaces[ns];
    let mut members: Vec<Member> = namespace
        .children
        .keys()
        .map(|name| Member {
            name: name.clone(),
            kind: MemberKind::Module,
            detail: "module".into(),
        })
        .collect();
    let Some(module) = namespace.module else {
        return members;
    };
    let package = &analysis.package;
    let items = package
        .fns
        .iter()
        .filter(|(_, d)| d.module == module && d.owner.is_none() && d.instance.is_none())
        .map(|(f, d)| Member {
            name: d.name.text.clone(),
            kind: MemberKind::Function,
            detail: signature(analysis, f),
        })
        .chain(
            package
                .structs
                .iter()
                .filter(|(_, d)| d.module == module && d.instance.is_none())
                .map(|(_, d)| Member {
                    name: d.name.text.clone(),
                    kind: MemberKind::Struct,
                    detail: format!("struct {}", d.name.text),
                }),
        )
        .chain(
            package
                .enums
                .iter()
                .filter(|(_, d)| d.module == module && d.instance.is_none())
                .map(|(_, d)| Member {
                    name: d.name.text.clone(),
                    kind: MemberKind::Enum,
                    detail: format!("enum {}", d.name.text),
                }),
        )
        .chain(
            package
                .consts
                .iter()
                .filter(|(_, d)| d.module == module)
                .map(|(_, d)| Member {
                    name: d.name.text.clone(),
                    kind: MemberKind::Constant,
                    detail: "const".into(),
                }),
        );
    members.extend(items.filter(|m| !m.name.starts_with('_')));
    members.sort_by(|a, b| a.name.cmp(&b.name));
    members
}

/// the signature of the call whose `(` is at `paren`, as text, with the
/// text of each parameter. A method's `self` is left out, and a struct
/// constructor takes its fields.
pub fn call_signature(
    analysis: &Analysis,
    file: FileId,
    paren: usize,
) -> Option<(String, Vec<String>)> {
    let package = &analysis.package;
    for at in exprs_ending_at(analysis, file, paren) {
        let body = &analysis.bodies[at.function];
        let function = match body.res.get(at.expr) {
            Some(Res::Fn(f)) => (*f, false),
            Some(Res::Method(f)) => (*f, true),
            Some(Res::StructCtor(s)) => {
                let fields: Vec<String> = package.structs[*s]
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let ty = analysis
                            .struct_fields
                            .get(*s)
                            .and_then(|t| t.get(i))
                            .map_or(String::new(), |&t| type_name(analysis, t));
                        format!("{}: {ty}", f.name.text)
                    })
                    .collect();
                let label = format!("{}({})", package.structs[*s].name.text, fields.join(", "));
                return Some((label, fields));
            }
            _ => continue,
        };
        let (f, method) = function;
        let sig = analysis.fn_sigs.get(f)?;
        let params: Vec<String> = sig
            .params
            .iter()
            .filter(|(name, _)| !(method && name == "self"))
            .map(|(name, ty)| format!("{name}: {}", type_name(analysis, *ty)))
            .collect();
        let label = format!(
            "{}({}) -> {}",
            package.fns[f].name.text,
            params.join(", "),
            type_name(analysis, sig.ret)
        );
        return Some((label, params));
    }
    None
}

/// what a name refers to, for finding all its uses
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    /// a local variable of a function
    Local(FnId, crate::infer::LocalId),
    /// an item, field or variant, by the span of its name. Two uses refer
    /// to the same thing when their definitions have the same span.
    Defined(Span),
}

/// the span of the name in an expression: a whole name, or the field of an
/// access
fn name_span(analysis: &Analysis, function: FnId, expr: ExprId) -> Span {
    let data = &analysis.package.fns[function];
    match &data.body.exprs[expr] {
        Expr::Field { name, .. } => name.span,
        _ => data.spans.exprs[expr],
    }
}

/// what an expression of a checked body refers to, if anything
fn target_of(analysis: &Analysis, function: FnId, expr: ExprId) -> Option<Target> {
    let body = &analysis.bodies[function];
    match body.res.get(expr)? {
        Res::Local(l) => Some(Target::Local(function, *l)),
        _ => definition_of(analysis, function, expr).map(Target::Defined),
    }
}

/// the target of the definition whose name is at `offset`: an item, field
/// or variant of a template, or a local at its declaration
fn defined_at(analysis: &Analysis, file: FileId, offset: usize) -> Option<Target> {
    let package = &analysis.package;
    let mut names: Vec<Span> = Vec::new();
    for (_, d) in package.fns.iter().filter(|(_, d)| d.instance.is_none()) {
        names.push(d.name.span);
    }
    for (_, d) in package.structs.iter().filter(|(_, d)| d.instance.is_none()) {
        names.push(d.name.span);
        names.extend(d.fields.iter().map(|f| f.name.span));
    }
    for (_, d) in package.enums.iter().filter(|(_, d)| d.instance.is_none()) {
        names.push(d.name.span);
        names.extend(d.variants.iter().map(|v| v.name.span));
    }
    names.extend(package.consts.iter().map(|(_, d)| d.name.span));
    names.extend(package.maps.iter().map(|(_, d)| d.name.span));
    if let Some(span) = names.into_iter().find(|&s| contains(s, file, offset)) {
        return Some(Target::Defined(span));
    }
    // a local at its declaration
    for (function, body) in analysis.bodies.iter() {
        for (local, data) in body.locals.iter() {
            if contains(data.span, file, offset) {
                return Some(Target::Local(function, local));
            }
        }
    }
    None
}

/// the spans of the uses of what is named at `offset`, definition first,
/// each once. Only expressions of checked bodies count, not type
/// annotations or imports.
pub fn references(analysis: &Analysis, file: FileId, offset: usize) -> Vec<Span> {
    let target = match expr_at(analysis, file, offset)
        .and_then(|at| target_of(analysis, at.function, at.expr))
    {
        Some(target) => target,
        None => match defined_at(analysis, file, offset) {
            Some(target) => target,
            None => return Vec::new(),
        },
    };
    let mut spans = vec![match target {
        Target::Local(f, l) => analysis.bodies[f].locals[l].span,
        Target::Defined(span) => span,
    }];
    for (function, body) in analysis.bodies.iter() {
        if let Target::Local(f, _) = target
            && f != function
        {
            continue;
        }
        for (expr, _) in analysis.package.fns[function].spans.exprs.iter() {
            if body.res.get(expr).is_some() && target_of(analysis, function, expr) == Some(target) {
                let span = name_span(analysis, function, expr);
                if !spans.contains(&span) {
                    spans.push(span);
                }
            }
        }
    }
    spans
}

/// the spans of the declaration and uses of the local named at `offset`,
/// for renaming. Other names give `None` because their uses may hide in
/// annotations or imports, which [`references`] does not see.
pub fn local_references(analysis: &Analysis, file: FileId, offset: usize) -> Option<Vec<Span>> {
    let target = match expr_at(analysis, file, offset)
        .and_then(|at| target_of(analysis, at.function, at.expr))
    {
        Some(target) => target,
        None => defined_at(analysis, file, offset)?,
    };
    matches!(target, Target::Local(..)).then(|| references(analysis, file, offset))
}

/// a definition of a file, for the editor outline
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub kind: MemberKind,
    /// a type or signature, and the kind for a program
    pub detail: String,
    /// the span of the name
    pub span: Span,
    /// the fields, variants and methods of a struct or enum
    pub children: Vec<Symbol>,
}

/// the definitions of `file` in order (functions, structs, enums, constants
/// and maps, with their members), without instances
pub fn symbols(analysis: &Analysis, file: FileId) -> Vec<Symbol> {
    let package = &analysis.package;
    let function = |f: FnId| {
        let d = &package.fns[f];
        let detail = match analysis.programs.get(f) {
            Some(info) => format!("{} {}", info.kind.noun(), signature(analysis, f)),
            None => signature(analysis, f),
        };
        Symbol {
            name: d.name.text.clone(),
            kind: if d.owner.is_some() {
                MemberKind::Method
            } else {
                MemberKind::Function
            },
            detail,
            span: d.name.span,
            children: Vec::new(),
        }
    };
    let mut symbols: Vec<Symbol> = package
        .fns
        .iter()
        .filter(|(_, d)| d.name.span.file == file && d.owner.is_none() && d.instance.is_none())
        .map(|(f, _)| function(f))
        .collect();
    for (s, d) in package.structs.iter() {
        if d.name.span.file != file || d.instance.is_some() {
            continue;
        }
        let mut children: Vec<Symbol> = d
            .fields
            .iter()
            .enumerate()
            .map(|(i, field)| Symbol {
                name: field.name.text.clone(),
                kind: MemberKind::Field,
                detail: analysis
                    .struct_fields
                    .get(s)
                    .and_then(|f| f.get(i))
                    .map_or(String::new(), |&t| type_name(analysis, t)),
                span: field.name.span,
                children: Vec::new(),
            })
            .collect();
        children.extend(d.methods.iter().map(|&m| function(m)));
        symbols.push(Symbol {
            name: d.name.text.clone(),
            kind: MemberKind::Struct,
            detail: format!("struct {}", d.name.text),
            span: d.name.span,
            children,
        });
    }
    for (_, d) in package.enums.iter() {
        if d.name.span.file != file || d.instance.is_some() {
            continue;
        }
        let mut children: Vec<Symbol> = d
            .variants
            .iter()
            .map(|v| Symbol {
                name: v.name.text.clone(),
                kind: MemberKind::Variant,
                detail: String::new(),
                span: v.name.span,
                children: Vec::new(),
            })
            .collect();
        children.extend(d.methods.iter().map(|&m| function(m)));
        symbols.push(Symbol {
            name: d.name.text.clone(),
            kind: MemberKind::Enum,
            detail: format!("enum {}", d.name.text),
            span: d.name.span,
            children,
        });
    }
    for (_, d) in package
        .consts
        .iter()
        .filter(|(_, d)| d.name.span.file == file)
    {
        symbols.push(Symbol {
            name: d.name.text.clone(),
            kind: MemberKind::Constant,
            detail: "const".into(),
            span: d.name.span,
            children: Vec::new(),
        });
    }
    for (_, d) in package
        .maps
        .iter()
        .filter(|(_, d)| d.name.span.file == file)
    {
        symbols.push(Symbol {
            name: d.name.text.clone(),
            kind: MemberKind::Map,
            detail: if d.is_event { "event" } else { "map" }.into(),
            span: d.name.span,
            children: Vec::new(),
        });
    }
    symbols.sort_by_key(|s| s.span.range().start);
    symbols
}

/// the kinds of names an editor colors. The order breaks ties between
/// tokens with the same span in [`tokens`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenKind {
    Namespace,
    Type,
    Struct,
    Enum,
    Parameter,
    Variable,
    Property,
    EnumMember,
    Function,
    Method,
    Decorator,
    /// a map or event, a variable of the object file
    Map,
    Constant,
}

/// the names of `file` with their kinds, sorted and without overlaps:
/// definitions, written types, decorators and names in checked bodies
pub fn tokens(analysis: &Analysis, file: FileId) -> Vec<(Span, TokenKind)> {
    let package = &analysis.package;
    let mut tokens: Vec<(Span, TokenKind)> = Vec::new();
    let type_ref = |tokens: &mut Vec<(Span, TokenKind)>, ty: &crate::hir::TypeRef| {
        fn walk(ty: &crate::hir::TypeRef, out: &mut Vec<Span>) {
            match ty {
                crate::hir::TypeRef::Path { path, args, .. } => {
                    out.extend(path.segments.last().map(|s| s.span));
                    for arg in args {
                        if let crate::hir::TypeArgRef::Type(t) = arg {
                            walk(t, out);
                        }
                    }
                }
                crate::hir::TypeRef::Option(inner, _) | crate::hir::TypeRef::Array(inner, ..) => {
                    walk(inner, out)
                }
                crate::hir::TypeRef::Missing => {}
            }
        }
        let mut spans = Vec::new();
        walk(ty, &mut spans);
        tokens.extend(spans.into_iter().map(|s| (s, TokenKind::Type)));
    };
    for (function, data) in package.fns.iter() {
        if data.name.span.file != file || data.instance.is_some() {
            continue;
        }
        let kind = if data.owner.is_some() {
            TokenKind::Method
        } else {
            TokenKind::Function
        };
        tokens.push((data.name.span, kind));
        for decorator in &data.decorators {
            let span = Span::new(
                file,
                decorator.span.range().start..decorator.name.span.range().end,
            );
            tokens.push((span, TokenKind::Decorator));
        }
        for param in &data.params {
            tokens.push((param.name.span, TokenKind::Parameter));
            if let Some(ty) = &param.ty {
                type_ref(&mut tokens, ty);
            }
        }
        if let Some(ty) = &data.ret {
            type_ref(&mut tokens, ty);
        }
        for (_, stmt) in data.body.stmts.iter() {
            if let crate::hir::Stmt::Assign { ty: Some(ty), .. } = stmt {
                type_ref(&mut tokens, ty);
            }
        }
        let Some(body) = analysis.bodies.get(function) else {
            continue;
        };
        let params = data.params.len();
        for (expr, _) in data.spans.exprs.iter() {
            let Some(res) = body.res.get(expr) else {
                continue;
            };
            let kind = match res {
                Res::Local(l) if l.into_raw().into_u32() < params as u32 => TokenKind::Parameter,
                Res::Local(_) => TokenKind::Variable,
                Res::Const(_) | Res::ConstValue(_) => TokenKind::Constant,
                Res::Fn(_) => TokenKind::Function,
                Res::Method(_)
                | Res::MapMethod(..)
                | Res::PacketParse(_)
                | Res::PacketLen
                | Res::KprobeArg(_)
                | Res::KprobeRet
                | Res::KPtrIsNull => TokenKind::Method,
                Res::Variant(..) => TokenKind::EnumMember,
                Res::StructCtor(_) => TokenKind::Struct,
                Res::Field(_) | Res::KernelField(_) => TokenKind::Property,
                Res::Map(_) => TokenKind::Map,
                Res::Builtin(_) | Res::KPtrCast => TokenKind::Type,
                Res::Operator(_) | Res::Repeat(_) => continue,
            };
            // we color only names and fields, because the span of a call,
            // operator or index is not a name
            match &data.body.exprs[expr] {
                Expr::Name(_) | Expr::Field { .. } => {
                    tokens.push((name_span(analysis, function, expr), kind))
                }
                _ => {}
            }
        }
    }
    for (_, d) in package.structs.iter() {
        if d.name.span.file != file || d.instance.is_some() {
            continue;
        }
        tokens.push((d.name.span, TokenKind::Struct));
        for decorator in &d.decorators {
            let span = Span::new(
                file,
                decorator.span.range().start..decorator.name.span.range().end,
            );
            tokens.push((span, TokenKind::Decorator));
        }
        for field in &d.fields {
            tokens.push((field.name.span, TokenKind::Property));
            type_ref(&mut tokens, &field.ty);
        }
    }
    for (_, d) in package.enums.iter() {
        if d.name.span.file != file || d.instance.is_some() {
            continue;
        }
        tokens.push((d.name.span, TokenKind::Enum));
        tokens.extend(
            d.variants
                .iter()
                .map(|v| (v.name.span, TokenKind::EnumMember)),
        );
    }
    tokens.extend(
        package
            .consts
            .iter()
            .filter(|(_, d)| d.name.span.file == file)
            .map(|(_, d)| (d.name.span, TokenKind::Constant)),
    );
    tokens.extend(
        package
            .maps
            .iter()
            .filter(|(_, d)| d.name.span.file == file)
            .map(|(_, d)| (d.name.span, TokenKind::Map)),
    );
    tokens.retain(|(span, _)| span.file == file && !span.range().is_empty());
    tokens.sort_by_key(|(span, kind)| (span.range().start, span.range().end, *kind));
    // removing overlaps: a token starting inside the previous one is
    // dropped. At the same start the shortest wins, then the first
    // `TokenKind`.
    let mut out: Vec<(Span, TokenKind)> = Vec::new();
    for (span, kind) in tokens {
        if out
            .last()
            .is_some_and(|(last, _)| span.range().start < last.range().end)
        {
            continue;
        }
        out.push((span, kind));
    }
    out
}
