//! this module implements generic items by monomorphization, and checks the
//! protocols of structs and enums. A generic item is a template, and each use
//! (`Vec[u8, 4]`) is an instance, a copy with its parameters bound.
//!
//! The package cannot grow while a [`Db`] borrows it, so a missing instance
//! is only requested here, and `analyze_with` adds it and checks again.

use std::collections::HashMap;

use vigil_diagnostics::{Diagnostic, Span, codes};

use crate::db::{Db, Value};
use crate::hir::*;
use crate::resolve::Def;
use crate::ty::{TyId, TyKind, display_key};

/// the value of a generic parameter while checking an instance
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvArg {
    /// a type argument
    Type(TyId),
    /// a constant argument, with the type of its parameter (its bound, or
    /// `i64`)
    Const(i128, TyId),
}

/// a bound against runaway instantiation, as `S[N]` containing `S[N + 1]`.
/// `analyze_with` stops when the package has more than four times this many
/// structs and functions.
pub const MAX_INSTANCES: usize = 512;

/// the error for a use of a generic item with the wrong number of generic
/// arguments, pointing also at the item
pub(crate) fn generic_count_error(
    item: &Ident,
    expected: usize,
    found: usize,
    span: Span,
) -> Diagnostic {
    let plural = if expected == 1 { "" } else { "s" };
    Diagnostic::error(format!(
        "`{}` takes {expected} generic argument{plural}, found {found}",
        item.text
    ))
    .with_code(codes::BAD_ARGUMENTS)
    .with_primary(span, format!("expected {expected}"))
    .with_secondary(item.span, format!("`{}` declared here", item.text))
}

/// the instances already in a package, by generic item and arguments
pub(crate) fn instance_table(package: &Package) -> HashMap<(Item, Vec<InstanceArg>), Item> {
    let mut table = HashMap::new();
    for (id, data) in package.structs.iter() {
        if let Some(instance) = &data.instance {
            table.insert((instance.generic, instance.args.clone()), Item::Struct(id));
        }
    }
    for (id, data) in package.enums.iter() {
        if let Some(instance) = &data.instance {
            table.insert((instance.generic, instance.args.clone()), Item::Enum(id));
        }
    }
    for (id, data) in package.fns.iter() {
        // a method is instantiated with its type, so only the instances of
        // generic functions are looked up by arguments
        if let Some(instance) = &data.instance
            && matches!(instance.generic, Item::Fn(g) if !package.fns[g].generic_params.is_empty())
        {
            table.insert((instance.generic, instance.args.clone()), Item::Fn(id));
        }
    }
    table
}

impl Db<'_> {
    /// whether a struct is a generic template, not an instance
    pub fn is_template_struct(&self, s: StructId) -> bool {
        let data = &self.package.structs[s];
        !data.generic_params.is_empty() && data.instance.is_none()
    }

    /// whether an enum is a generic template, not an instance
    pub fn is_template_enum(&self, e: EnumId) -> bool {
        let data = &self.package.enums[e];
        !data.generic_params.is_empty() && data.instance.is_none()
    }

    /// whether a function is a template: generic, or a method of a generic
    /// struct or enum. An instance is never a template.
    pub fn is_template_fn(&self, f: FnId) -> bool {
        let data = &self.package.fns[f];
        if data.instance.is_some() {
            return false;
        }
        !data.generic_params.is_empty()
            || matches!(data.owner, Some(Owner::Struct(s)) if self.is_template_struct(s))
            || matches!(data.owner, Some(Owner::Enum(e)) if self.is_template_enum(e))
    }

    /// the generic parameters of the item of an instance. For a method, its
    /// own and then those of its struct or enum
    fn generic_params_of(&self, item: Item) -> Vec<GenericParam> {
        let package = self.package;
        match item {
            Item::Struct(s) => package.structs[s].generic_params.clone(),
            Item::Enum(e) => package.enums[e].generic_params.clone(),
            Item::Fn(f) => {
                let mut params = package.fns[f].generic_params.clone();
                match package.fns[f].owner {
                    Some(Owner::Struct(s)) => {
                        params.extend(package.structs[s].generic_params.clone())
                    }
                    Some(Owner::Enum(e)) => params.extend(package.enums[e].generic_params.clone()),
                    None => {}
                }
                params
            }
            _ => Vec::new(),
        }
    }

    /// binding the generic parameters of an instance while we check it.
    /// `None` (not an instance) binds nothing
    pub fn enter(&mut self, instance: Option<&Instance>) {
        self.env.clear();
        let Some(instance) = instance else {
            return;
        };
        let package = self.package;
        let params = self.generic_params_of(instance.generic);
        let module = match instance.generic {
            Item::Struct(s) => package.structs[s].module,
            Item::Enum(e) => package.enums[e].module,
            Item::Fn(f) => package.fns[f].module,
            _ => return,
        };
        let mut env = Vec::new();
        for (name, arg) in &instance.env {
            let value = match arg {
                InstanceArg::Ty(key) => EnvArg::Type(self.types.from_key(key)),
                InstanceArg::Const(v) => {
                    let ty = params
                        .iter()
                        .find(|p| p.name.text == *name)
                        .and_then(|p| p.bound.clone())
                        .map(|b| self.lower_ty(module, &b))
                        .unwrap_or_else(|| self.types.int(crate::ty::IntTy::I64));
                    EnvArg::Const(*v, ty)
                }
            };
            env.push((name.clone(), value));
        }
        self.env = env;
    }

    /// the value bound to a generic parameter, if any. The latest binding of a
    /// name wins
    pub fn env_lookup(&self, name: &str) -> Option<EnvArg> {
        self.env
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|&(_, a)| a)
    }

    /// whether a generic parameter is a constant (bounded by an integer type)
    pub(crate) fn is_const_param(&mut self, module: ModuleId, param: &GenericParam) -> bool {
        let Some(bound) = &param.bound else {
            return false;
        };
        if self.protocol_of(module, bound).is_some() {
            return false;
        }
        let ty = self.lower_ty(module, bound);
        matches!(self.types.kind(ty), TyKind::Int(_))
    }

    /// the protocol a type reference names, if any
    pub fn protocol_of(&self, module: ModuleId, ty: &TypeRef) -> Option<ProtocolId> {
        let TypeRef::Path { path, .. } = ty else {
            return None;
        };
        match self
            .scopes
            .resolve_prefix(self.package, module, &path.segments)
        {
            Some((Def::Item(Item::Protocol(p)), used)) if used == path.segments.len() => Some(p),
            _ => None,
        }
    }

    /// the instance of a generic item for some arguments, if it exists.
    /// Otherwise we request it, once, and return `None`.
    pub fn instantiate(&mut self, generic: Item, args: Vec<InstanceArg>) -> Option<Item> {
        if let Some(&item) = self.instances.get(&(generic, args.clone())) {
            return Some(item);
        }
        if !self.requests.contains(&(generic, args.clone())) {
            self.requests.push((generic, args));
        }
        None
    }

    /// checking the lowered arguments of an instance of `generic` against its
    /// parameters: number, kind, range of constants and protocol bounds.
    /// `None` means an error, already reported.
    pub fn check_generic_args(
        &mut self,
        generic: Item,
        args: Vec<(GenericArgValue, Span)>,
        span: Span,
    ) -> Option<Vec<InstanceArg>> {
        let package = self.package;
        let (params, module, ident) = match generic {
            Item::Struct(s) => (
                package.structs[s].generic_params.clone(),
                package.structs[s].module,
                &package.structs[s].name,
            ),
            Item::Enum(e) => (
                package.enums[e].generic_params.clone(),
                package.enums[e].module,
                &package.enums[e].name,
            ),
            Item::Fn(f) => (
                package.fns[f].generic_params.clone(),
                package.fns[f].module,
                &package.fns[f].name,
            ),
            _ => return None,
        };
        if params.len() != args.len() {
            self.diags
                .push(generic_count_error(ident, params.len(), args.len(), span));
            return None;
        }
        let mut instance_args: Vec<InstanceArg> = Vec::new();
        for (index, (param, (arg, arg_span))) in params.iter().zip(args).enumerate() {
            let constant = self.is_const_param(module, param);
            match (constant, arg) {
                (true, GenericArgValue::Const(v)) => {
                    if let Some(bound) = &param.bound {
                        let ty = self.lower_ty(module, bound);
                        if let TyKind::Int(int) = self.types.kind(ty)
                            && !int.contains(v)
                        {
                            self.diags.push(crate::db::out_of_range(v, *int, arg_span));
                            return None;
                        }
                    }
                    instance_args.push(InstanceArg::Const(v));
                }
                (false, GenericArgValue::Type(ty)) => {
                    if let Some(bound) = &param.bound
                        && let Some(protocol) = self.protocol_of(module, bound)
                    {
                        // the arguments of the protocol may name earlier
                        // parameters, as in `M: Model[T]`
                        let bound_args = match bound {
                            TypeRef::Path { args, .. } if !args.is_empty() => {
                                let earlier: Vec<_> = params[..index]
                                    .iter()
                                    .map(|p| p.name.text.clone())
                                    .zip(instance_args.iter().cloned())
                                    .collect();
                                self.lower_with(module, &params, &earlier, args)
                            }
                            _ => Vec::new(),
                        };
                        if !self.implements(ty, protocol, &bound_args) {
                            let shown = self.display(ty);
                            let mut protocol_name = package.protocols[protocol].name.text.clone();
                            if !bound_args.is_empty() {
                                let args: Vec<String> = bound_args
                                    .iter()
                                    .map(|a| match a {
                                        GenericArgValue::Type(t) => self.display(*t),
                                        GenericArgValue::Const(v) => v.to_string(),
                                        GenericArgValue::Error => "?".into(),
                                    })
                                    .collect();
                                protocol_name = format!("{protocol_name}[{}]", args.join(", "));
                            }
                            self.diags.push(
                                Diagnostic::error(format!(
                                    "`{shown}` does not implement `{protocol_name}`"
                                ))
                                .with_code(codes::MISMATCHED_TYPES)
                                .with_primary(
                                    arg_span,
                                    format!("`{}` requires `{protocol_name}`", param.name.text),
                                ),
                            );
                            return None;
                        }
                    }
                    let Some(key) = self.types.key(ty) else {
                        let shown = self.display(ty);
                        self.diags.push(
                            Diagnostic::error(format!("`{shown}` cannot be a generic argument"))
                                .with_code(codes::WRONG_KIND)
                                .with_primary(arg_span, "unsupported generic argument"),
                        );
                        return None;
                    };
                    instance_args.push(InstanceArg::Ty(key));
                }
                (true, GenericArgValue::Type(_)) => {
                    self.diags.push(
                        Diagnostic::error(format!("`{}` is a constant parameter", param.name.text))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(arg_span, "expected a constant, found a type")
                            .with_secondary(param.name.span, "declared here"),
                    );
                    return None;
                }
                (false, GenericArgValue::Const(_)) => {
                    self.diags.push(
                        Diagnostic::error(format!("`{}` is a type parameter", param.name.text))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(arg_span, "expected a type, found a constant")
                            .with_secondary(param.name.span, "declared here"),
                    );
                    return None;
                }
                (_, GenericArgValue::Error) => return None,
            }
        }
        Some(instance_args)
    }

    /// lowering the generic arguments written in a type position, as in
    /// `Vec[u8, N]`, with their spans
    pub fn type_generic_args(
        &mut self,
        module: ModuleId,
        args: &[TypeArgRef],
    ) -> Vec<(GenericArgValue, Span)> {
        args.iter()
            .map(|arg| {
                let span = match arg {
                    TypeArgRef::Type(t) => t.span(),
                    TypeArgRef::Const(c) => c.span(),
                }
                .unwrap_or(Span::new(self.package.modules[module].file, 0..0));
                let value = match arg {
                    TypeArgRef::Const(expr) => match self.eval(module, expr) {
                        Some(Value::Int(v)) => GenericArgValue::Const(v),
                        _ => GenericArgValue::Error,
                    },
                    // a name, which may be a type or a constant: the parser
                    // cannot tell
                    TypeArgRef::Type(ty @ TypeRef::Path { path, args, .. }) => {
                        let constant = args.is_empty() && self.names_constant(module, path);
                        if constant {
                            match self.eval(module, &ConstExpr::Path(path.clone())) {
                                Some(Value::Int(v)) => GenericArgValue::Const(v),
                                _ => GenericArgValue::Error,
                            }
                        } else {
                            let lowered = self.lower_ty(module, ty);
                            if self.types.is_error(lowered) {
                                GenericArgValue::Error
                            } else {
                                GenericArgValue::Type(lowered)
                            }
                        }
                    }
                    TypeArgRef::Type(ty) => {
                        let lowered = self.lower_ty(module, ty);
                        GenericArgValue::Type(lowered)
                    }
                };
                (value, span)
            })
            .collect()
    }

    /// whether a path names a constant item or a constant generic parameter
    pub fn names_constant(&self, module: ModuleId, path: &PathRef) -> bool {
        if let [segment] = path.segments.as_slice()
            && let Some(arg) = self.env_lookup(&segment.text)
        {
            return matches!(arg, EnvArg::Const(..));
        }
        matches!(
            self.scopes.resolve_prefix(self.package, module, &path.segments),
            Some((Def::Item(Item::Const(_)), used)) if used == path.segments.len()
        )
    }

    /// lowering generic arguments written in `module`, with `params` bound to
    /// `bound`. The previous environment is restored afterwards.
    fn lower_with(
        &mut self,
        module: ModuleId,
        params: &[GenericParam],
        bound: &[(String, InstanceArg)],
        args: &[TypeArgRef],
    ) -> Vec<GenericArgValue> {
        let mut env = Vec::new();
        for (name, arg) in bound {
            let value = match arg {
                InstanceArg::Ty(key) => EnvArg::Type(self.types.from_key(key)),
                InstanceArg::Const(v) => {
                    let ty = params
                        .iter()
                        .find(|p| p.name.text == *name)
                        .and_then(|p| p.bound.clone())
                        .map(|b| self.lower_ty(module, &b))
                        .unwrap_or_else(|| self.types.int(crate::ty::IntTy::I64));
                    EnvArg::Const(*v, ty)
                }
            };
            env.push((name.clone(), value));
        }
        let outer = std::mem::replace(&mut self.env, env);
        let lowered = self
            .type_generic_args(module, args)
            .into_iter()
            .map(|(v, _)| v)
            .collect();
        self.env = outer;
        lowered
    }

    /// the declaration of a nominal type (the template's, for an instance)
    fn declaration(&self, ty: TyId) -> Option<Declaration<'_>> {
        let package = self.package;
        Some(match self.types.kind(ty) {
            TyKind::Struct(s) => {
                let data = &package.structs[*s];
                let declaring = match &data.instance {
                    Some(Instance {
                        generic: Item::Struct(g),
                        ..
                    }) => &package.structs[*g],
                    _ => data,
                };
                Declaration {
                    module: declaring.module,
                    generic_params: &declaring.generic_params,
                    bases: &declaring.bases,
                    base_args: &declaring.base_args,
                    instance: data.instance.as_ref(),
                }
            }
            TyKind::Enum(e) => {
                let data = &package.enums[*e];
                let declaring = match &data.instance {
                    Some(Instance {
                        generic: Item::Enum(g),
                        ..
                    }) => &package.enums[*g],
                    _ => data,
                };
                Declaration {
                    module: declaring.module,
                    generic_params: &declaring.generic_params,
                    bases: &declaring.bases,
                    base_args: &declaring.base_args,
                    instance: data.instance.as_ref(),
                }
            }
            _ => return None,
        })
    }

    /// whether a struct or enum lists a protocol with the given arguments.
    /// Empty `args` match any arguments, and the error type implements
    /// everything.
    pub fn implements(&mut self, ty: TyId, protocol: ProtocolId, args: &[GenericArgValue]) -> bool {
        if matches!(self.types.kind(ty), TyKind::Error) {
            return true;
        }
        let Some(declaration) = self.declaration(ty) else {
            return false;
        };
        let package = self.package;
        let module = declaration.module;
        let params = declaration.generic_params.to_vec();
        let env = declaration
            .instance
            .map(|i| i.env.clone())
            .unwrap_or_default();
        let listed: Vec<Vec<TypeArgRef>> = declaration
            .bases
            .iter()
            .zip(declaration.base_args)
            .filter(|(base, _)| {
                matches!(
                    self.scopes.resolve_prefix(package, module, &base.segments),
                    Some((Def::Item(Item::Protocol(p)), _)) if p == protocol
                )
            })
            .map(|(_, a)| a.clone())
            .collect();
        listed.into_iter().any(|base_args| {
            if args.is_empty() {
                return true;
            }
            let lowered = self.lower_with(module, &params, &env, &base_args);
            lowered.len() == args.len() && lowered.iter().zip(args).all(|(a, b)| same_arg(*a, *b))
        })
    }

    /// checking that structs and enums define the methods of their protocols
    /// with the right types. Templates are left out, their instances are
    /// checked.
    pub(crate) fn check_protocols(&mut self) {
        let package = self.package;
        // the method types of a non-generic protocol are lowered once, so an
        // unknown type is reported once. Those of a generic protocol are
        // lowered for each use.
        let mut signatures: HashMap<ProtocolId, Vec<MethodTypes>> = HashMap::new();
        for (protocol, data) in package.protocols.iter() {
            if data.generic_params.is_empty() {
                let signature = self.protocol_signature(protocol, Vec::new());
                signatures.insert(protocol, signature);
            }
        }
        let mut types: Vec<(TyId, String, Vec<FnId>)> = Vec::new();
        for (id, data) in package.structs.iter() {
            if !self.is_template_struct(id) {
                let ty = self.types.intern(TyKind::Struct(id));
                types.push((ty, data.name.text.clone(), data.methods.clone()));
            }
        }
        for (id, data) in package.enums.iter() {
            if !self.is_template_enum(id) {
                let ty = self.types.intern(TyKind::Enum(id));
                types.push((ty, data.name.text.clone(), data.methods.clone()));
            }
        }
        for (ty, name, methods) in types {
            let Some(declaration) = self.declaration(ty) else {
                continue;
            };
            let module = declaration.module;
            let params = declaration.generic_params.to_vec();
            let env = declaration
                .instance
                .map(|i| i.env.clone())
                .unwrap_or_default();
            let bases: Vec<(PathRef, Vec<TypeArgRef>)> = declaration
                .bases
                .iter()
                .cloned()
                .zip(declaration.base_args.iter().cloned())
                .collect();
            for (base, base_args) in bases {
                let protocol = match self.scopes.resolve_prefix(package, module, &base.segments) {
                    Some((Def::Item(Item::Protocol(p)), used)) if used == base.segments.len() => p,
                    Some((Def::Error, _)) => continue,
                    _ => {
                        self.diags.push(
                            Diagnostic::error(format!("`{base}` is not a protocol"))
                                .with_code(codes::WRONG_KIND)
                                .with_primary(base.span, "expected a protocol"),
                        );
                        continue;
                    }
                };
                let protocol_data = &package.protocols[protocol];
                if base_args.len() != protocol_data.generic_params.len() {
                    self.diags.push(generic_count_error(
                        &protocol_data.name,
                        protocol_data.generic_params.len(),
                        base_args.len(),
                        base.span,
                    ));
                    continue;
                }
                let signature = match signatures.get(&protocol) {
                    Some(signature) => signature.clone(),
                    None => {
                        let args = self.lower_with(module, &params, &env, &base_args);
                        if args.iter().any(|a| matches!(a, GenericArgValue::Error)) {
                            continue;
                        }
                        self.protocol_signature(protocol, args)
                    }
                };
                for (required, (param_tys, ret_ty)) in protocol_data.methods.iter().zip(&signature)
                {
                    let found = methods
                        .iter()
                        .copied()
                        .find(|&m| package.fns[m].name.text == required.name.text);
                    let arity =
                        |f: &FnData| f.params.iter().filter(|p| p.name.text != "self").count();
                    let message = match found {
                        None => format!("missing method `{}`", required.name.text),
                        Some(f) if arity(&package.fns[f]) != required.arity => format!(
                            "`{}` takes {} besides `self`, not {}",
                            required.name.text,
                            vigil_diagnostics::plural(required.arity, "parameter"),
                            arity(&package.fns[f])
                        ),
                        Some(f) => match self.signature_mismatch(f, param_tys, *ret_ty) {
                            Some(message) => message,
                            None => continue,
                        },
                    };
                    let protocol_name = &protocol_data.name.text;
                    let mut diagnostic =
                        Diagnostic::error(format!("`{name}` does not implement `{protocol_name}`"))
                            .with_code(codes::MISMATCHED_TYPES)
                            .with_primary(base.span, message)
                            .with_secondary(required.name.span, "required here");
                    if let Some(f) = found {
                        diagnostic =
                            diagnostic.with_secondary(package.fns[f].name.span, "implemented here");
                    }
                    self.diags.push(diagnostic);
                }
            }
        }
    }

    /// the method types of a protocol with its parameters bound to `args`. A
    /// constant argument is an `i64`, an erroneous one the error type.
    fn protocol_signature(
        &mut self,
        protocol: ProtocolId,
        args: Vec<GenericArgValue>,
    ) -> Vec<MethodTypes> {
        let data = &self.package.protocols[protocol];
        let mut env = Vec::new();
        for (param, arg) in data.generic_params.iter().zip(args) {
            let value = match arg {
                GenericArgValue::Type(ty) => EnvArg::Type(ty),
                GenericArgValue::Const(v) => {
                    EnvArg::Const(v, self.types.int(crate::ty::IntTy::I64))
                }
                GenericArgValue::Error => EnvArg::Type(self.types.error),
            };
            env.push((param.name.text.clone(), value));
        }
        let outer = std::mem::replace(&mut self.env, env);
        let signature = data
            .methods
            .iter()
            .map(|m| {
                let params = m
                    .params
                    .iter()
                    .map(|t| t.as_ref().map(|t| self.lower_ty(data.module, t)))
                    .collect();
                let ret = m.ret.as_ref().map(|t| self.lower_ty(data.module, t));
                (params, ret)
            })
            .collect();
        self.env = outer;
        signature
    }
}

/// a struct or enum as declared, for [`Db::implements`] and
/// [`Db::check_protocols`]
struct Declaration<'p> {
    module: ModuleId,
    generic_params: &'p [GenericParam],
    bases: &'p [PathRef],
    base_args: &'p [Vec<TypeArgRef>],
    instance: Option<&'p Instance>,
}

/// whether two lowered generic arguments are the same. An error matches
/// anything, because it is already reported.
fn same_arg(a: GenericArgValue, b: GenericArgValue) -> bool {
    match (a, b) {
        (GenericArgValue::Type(x), GenericArgValue::Type(y)) => x == y,
        (GenericArgValue::Const(x), GenericArgValue::Const(y)) => x == y,
        (GenericArgValue::Error, _) | (_, GenericArgValue::Error) => true,
        _ => false,
    }
}

impl Db<'_> {
    /// how method `f` differs from the types a protocol requires, as a
    /// message. A parameter without a type in the protocol accepts any type.
    fn signature_mismatch(
        &mut self,
        f: FnId,
        params: &[Option<TyId>],
        ret: Option<TyId>,
    ) -> Option<String> {
        let sig = self.fn_sigs.get(f)?.clone();
        let name = &self.package.fns[f].name.text;
        for ((param, found), required) in sig.call_params().iter().zip(params) {
            if let Some(required) = *required
                && !self.types.is_error(required)
                && required != *found
            {
                return Some(format!(
                    "`{name}` takes `{param}: {}`, the protocol requires `{}`",
                    self.display(*found),
                    self.display(required)
                ));
            }
        }
        if let Some(required) = ret
            && !self.types.is_error(required)
            && required != sig.ret
        {
            return Some(format!(
                "`{name}` returns `{}`, the protocol requires `{}`",
                self.display(sig.ret),
                self.display(required)
            ));
        }
        None
    }
}

/// the parameter types and return type of a protocol method (`None` for any
/// type)
type MethodTypes = (Vec<Option<TyId>>, Option<TyId>);

/// a lowered generic argument
#[derive(Clone, Copy, Debug)]
pub enum GenericArgValue {
    /// a type
    Type(TyId),
    /// the value of a constant
    Const(i128),
    /// an argument that could not be lowered, already reported
    Error,
}

/// the arguments of an instance as written in its name: `u8, 4`
fn render_args(package: &Package, args: &[InstanceArg]) -> String {
    let rendered: Vec<_> = args
        .iter()
        .map(|a| match a {
            InstanceArg::Ty(key) => display_key(package, key),
            InstanceArg::Const(v) => v.to_string(),
        })
        .collect();
    rendered.join(", ")
}

/// adding the instance of `generic` for `args` to the package, with its
/// methods. The copy is named after its arguments (`Vec[u8, 4]`) and its
/// [`Instance`] keeps the environment
pub(crate) fn create_instance(package: &mut Package, generic: Item, args: Vec<InstanceArg>) {
    match generic {
        Item::Struct(s) => {
            let mut data = package.structs[s].clone();
            let env: Vec<_> = data
                .generic_params
                .iter()
                .map(|p| p.name.text.clone())
                .zip(args.iter().cloned())
                .collect();
            data.name.text = format!("{}[{}]", data.name.text, render_args(package, &args));
            data.instance = Some(Instance {
                generic,
                args: args.clone(),
                env: env.clone(),
            });
            let methods = std::mem::take(&mut data.methods);
            let id = package.structs.alloc(data);
            for method in methods {
                let mut f = package.fns[method].clone();
                let mut method_env = env.clone();
                // the method's own generic parameters shadow those of the
                // struct and stay unbound: generic methods of generic types
                // are not supported
                method_env
                    .retain(|(name, _)| !f.generic_params.iter().any(|p| p.name.text == *name));
                f.owner = Some(Owner::Struct(id));
                f.instance = Some(Instance {
                    generic: Item::Fn(method),
                    args: args.clone(),
                    env: method_env,
                });
                let fid = package.fns.alloc(f);
                package.structs[id].methods.push(fid);
            }
        }
        Item::Enum(e) => {
            let mut data = package.enums[e].clone();
            let env: Vec<_> = data
                .generic_params
                .iter()
                .map(|p| p.name.text.clone())
                .zip(args.iter().cloned())
                .collect();
            data.name.text = format!("{}[{}]", data.name.text, render_args(package, &args));
            data.instance = Some(Instance {
                generic,
                args: args.clone(),
                env: env.clone(),
            });
            let methods = std::mem::take(&mut data.methods);
            let id = package.enums.alloc(data);
            for method in methods {
                let mut f = package.fns[method].clone();
                let mut method_env = env.clone();
                method_env
                    .retain(|(name, _)| !f.generic_params.iter().any(|p| p.name.text == *name));
                f.owner = Some(Owner::Enum(id));
                f.instance = Some(Instance {
                    generic: Item::Fn(method),
                    args: args.clone(),
                    env: method_env,
                });
                let fid = package.fns.alloc(f);
                package.enums[id].methods.push(fid);
            }
        }
        Item::Fn(f) => {
            let mut data = package.fns[f].clone();
            let env = data
                .generic_params
                .iter()
                .map(|p| p.name.text.clone())
                .zip(args.iter().cloned())
                .collect();
            data.name.text = format!("{}[{}]", data.name.text, render_args(package, &args));
            data.generic_params.clear();
            data.instance = Some(Instance { generic, args, env });
            package.fns.alloc(data);
        }
        _ => {}
    }
}
