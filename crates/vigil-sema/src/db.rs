//! this module collects the signatures of the package (fields, variants,
//! functions, aliases and constants) in a [`Db`]. Bodies are checked only
//! against these signatures, so their order does not matter.
//!
//! aliases and constants depend on each other (array lengths), so we compute
//! them on demand and memoize them. Meeting one in progress is a cycle.

use std::collections::HashSet;

use la_arena::ArenaMap;
use vigil_diagnostics::{Diagnostic, Diagnostics, Span, codes};

use std::collections::HashMap;

use crate::Options;
use crate::comptime::CValue;
use crate::generics::{EnvArg, GenericArgValue};
use crate::hir::*;
use crate::maps::MapInfo;
use crate::program::{ContextKind, ProgramInfo};
use crate::resolve::{Builtin, Def, Scopes};
use crate::ty::{IntTy, TyId, TyKind, Types};

/// the value of a constant known by the type checker: an integer or a bool.
/// Arrays and comptime constants are computed after type checking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstValue {
    Int(i128),
    Bool(bool),
}

/// a value of the constant evaluator, before checking it against a type.
/// Integers are unbounded here (within `i128`). Strings are used as
/// decorator arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Int(i128),
    Bool(bool),
    Str(String),
}

/// the signature of a function
#[derive(Clone, Debug)]
pub struct FnSig {
    /// the parameters by name, including `self`
    pub params: Vec<(String, TyId)>,
    /// is the first parameter `self`?
    pub has_self: bool,
    /// the return type, `()` when none is written
    pub ret: TyId,
    /// the default value of each parameter, if any
    pub defaults: Vec<Option<ConstValue>>,
}

impl FnSig {
    /// the parameters given at a call site (without `self`)
    pub fn call_params(&self) -> &[(String, TyId)] {
        &self.params[usize::from(self.has_self)..]
    }

    /// the defaults of [`FnSig::call_params`]
    pub fn call_defaults(&self) -> &[Option<ConstValue>] {
        &self.defaults[usize::from(self.has_self)..]
    }
}

/// a value computed on demand
#[derive(Clone, Copy)]
enum Memo<T> {
    /// being computed, meeting it again is a cycle
    InProgress,
    /// already computed
    Done(T),
}

/// the signatures of a package and the state of its type checking.
/// One `Db` serves one round of analysis: [`Db::collect_signatures`] fills
/// the tables before any body is checked, and [`crate::Analysis`] takes
/// them at the end.
pub struct Db<'p> {
    /// the package being analysed
    pub package: &'p Package,
    /// the names visible in each module
    pub scopes: Scopes,
    /// the interned types
    pub types: Types,
    /// the field types of each struct, in order
    pub struct_fields: ArenaMap<StructId, Vec<TyId>>,
    /// the payload types of each variant, `None` for a variant without
    /// parentheses
    pub variant_fields: ArenaMap<EnumId, Vec<Option<Vec<TyId>>>>,
    /// the signature of each function that is not a template
    pub fn_sigs: ArenaMap<FnId, FnSig>,
    /// the structs that are the context of a program type
    pub struct_contexts: ArenaMap<StructId, ContextKind>,
    /// the structs whose integer fields are big-endian in memory
    pub network_order: ArenaMap<StructId, ()>,
    /// the kernel helpers, with their numbers
    pub helpers: ArenaMap<FnId, u32>,
    /// the functions that are programs or tests
    pub programs: ArenaMap<FnId, ProgramInfo>,
    /// the kind and sizes of each map
    pub maps: ArenaMap<MapId, MapInfo>,
    /// what the target kernels allow
    pub options: Options,
    aliases: ArenaMap<AliasId, Memo<TyId>>,
    consts: ArenaMap<ConstId, Memo<(TyId, Option<ConstValue>)>>,
    /// the generic parameters bound while checking an instance
    pub(crate) env: Vec<(String, EnvArg)>,
    /// the instances in the package, by generic item and arguments
    pub(crate) instances: HashMap<(Item, Vec<InstanceArg>), Item>,
    /// the instances used but not yet in the package
    pub requests: Vec<(Item, Vec<InstanceArg>)>,
    /// the constants computed by a comptime function after type checking,
    /// with the function and its argument values
    pub comptime_consts: Vec<(ConstId, FnId, Vec<Value>)>,
    /// the array constants written as literals, with their values
    pub array_consts: Vec<(ConstId, CValue)>,
    /// the diagnostics so far
    pub diags: Diagnostics,
}

impl<'p> Db<'p> {
    /// creating an empty `Db` with the scopes of the package built.
    /// Building the scopes may already report errors in `diags`.
    pub fn new(package: &'p Package, options: Options, mut diags: Diagnostics) -> Self {
        let scopes = Scopes::build(package, options.btf.clone(), &mut diags);
        Db {
            package,
            scopes,
            types: Types::default(),
            struct_fields: ArenaMap::default(),
            variant_fields: ArenaMap::default(),
            fn_sigs: ArenaMap::default(),
            struct_contexts: ArenaMap::default(),
            network_order: ArenaMap::default(),
            helpers: ArenaMap::default(),
            programs: ArenaMap::default(),
            maps: ArenaMap::default(),
            options,
            aliases: ArenaMap::default(),
            consts: ArenaMap::default(),
            env: Vec::new(),
            instances: crate::generics::instance_table(package),
            requests: Vec::new(),
            comptime_consts: Vec::new(),
            array_consts: Vec::new(),
            diags,
        }
    }

    /// a type as shown in messages
    pub fn display(&self, ty: TyId) -> String {
        self.types.display(self.package, ty, &|_| None)
    }

    /// reporting a diagnostic and returning the error type
    fn error(&mut self, diagnostic: Diagnostic) -> TyId {
        self.diags.push(diagnostic);
        self.types.error
    }

    /// the main function of the module: it resolves every signature, then
    /// checks decorators, protocols and types that contain themselves.
    /// Function decorators come last because `@tail` names a map and
    /// `@property` reads the signatures of its generators.
    pub fn collect_signatures(&mut self) {
        let package = self.package;
        for (id, data) in package.aliases.iter() {
            // a generic alias is resolved at each use, with its arguments
            if data.generic_params.is_empty() {
                self.alias_ty(id);
            }
            check_reserved(&data.name, &mut self.diags);
        }
        for (id, _) in package.consts.iter() {
            self.const_info(id);
        }
        for (id, data) in package.structs.iter() {
            check_reserved(&data.name, &mut self.diags);
            if self.is_template_struct(id) {
                continue;
            }
            self.check_struct_decorators(id);
            self.enter(data.instance.as_ref());
            let fields = data
                .fields
                .iter()
                .map(|f| {
                    let ty = self.lower_ty(data.module, &f.ty);
                    self.value_type(ty, f.ty.span())
                })
                .collect();
            self.struct_fields.insert(id, fields);
            let names = data.fields.iter().map(|f| &f.name);
            let methods = data.methods.iter().map(|&m| &package.fns[m].name);
            check_unique(names.chain(methods), "member", &mut self.diags);
        }
        self.env.clear();
        for (id, data) in package.enums.iter() {
            check_reserved(&data.name, &mut self.diags);
            if self.is_template_enum(id) {
                continue;
            }
            self.enter(data.instance.as_ref());
            let variants = data
                .variants
                .iter()
                .map(|v| {
                    v.fields.as_ref().map(|fields| {
                        fields
                            .iter()
                            .map(|t| self.lower_ty(data.module, t))
                            .collect()
                    })
                })
                .collect();
            self.variant_fields.insert(id, variants);
            let names = data.variants.iter().map(|v| &v.name);
            let methods = data.methods.iter().map(|&m| &package.fns[m].name);
            check_unique(names.chain(methods), "member", &mut self.diags);
        }
        self.env.clear();
        self.collect_maps();
        for (id, data) in package.fns.iter() {
            if self.is_template_fn(id) {
                continue;
            }
            self.enter(data.instance.as_ref());
            let sig = self.fn_sig(data);
            self.fn_sigs.insert(id, sig);
        }
        self.env.clear();
        for (id, _) in package.fns.iter() {
            if !self.is_template_fn(id) {
                self.check_fn_decorators(id);
            }
        }
        for (_, data) in package.protocols.iter() {
            check_reserved(&data.name, &mut self.diags);
        }
        self.check_protocols();
        self.check_infinite_types();
    }

    /// the signature of a function. A parameter without a type, other than
    /// `self`, is an error.
    fn fn_sig(&mut self, data: &FnData) -> FnSig {
        check_unique(
            data.params.iter().map(|p| &p.name),
            "parameter",
            &mut self.diags,
        );
        let mut has_self = false;
        let mut params = Vec::new();
        for (i, param) in data.params.iter().enumerate() {
            let ty = if param.name.text == "self" {
                match (data.owner, i, &param.ty) {
                    (Some(owner), 0, None) => {
                        has_self = true;
                        let ty = self.types.intern(match owner {
                            Owner::Struct(s) => TyKind::Struct(s),
                            Owner::Enum(e) => TyKind::Enum(e),
                        });
                        // a method takes its receiver by reference, so a
                        // `@mut` method changes it in place
                        self.types.intern(TyKind::Ref(ty))
                    }
                    (Some(_), 0, Some(ty)) => self.error(
                        Diagnostic::error("`self` takes no type annotation")
                            .with_code(codes::WRONG_KIND)
                            .with_primary(ty.span().unwrap_or(param.name.span), "remove this type"),
                    ),
                    _ => self.error(
                        Diagnostic::error("`self` must be the first parameter of a method")
                            .with_code(codes::WRONG_KIND)
                            .with_primary(param.name.span, "unexpected `self`"),
                    ),
                }
            } else {
                match &param.ty {
                    Some(ty) => {
                        let lowered = self.lower_ty(data.module, ty);
                        self.value_type(lowered, ty.span())
                    }
                    None => self.error(
                        Diagnostic::error(format!(
                            "parameter `{}` needs a type annotation",
                            param.name.text
                        ))
                        .with_code(codes::CANNOT_INFER)
                        .with_primary(param.name.span, "missing `: type`"),
                    ),
                }
            };
            params.push((param.name.text.clone(), ty));
        }
        let defaults = self.param_defaults(data, &params);
        let ret = match &data.ret {
            Some(ty) => {
                let lowered = self.lower_ty(data.module, ty);
                self.value_type(lowered, ty.span())
            }
            None => self.types.unit,
        };
        FnSig {
            params,
            defaults,
            has_self,
            ret,
        }
    }

    /// the default value of each parameter, an integer or bool constant.
    /// As in Python, after a parameter with a default every later one
    /// needs one too.
    fn param_defaults(
        &mut self,
        data: &FnData,
        params: &[(String, TyId)],
    ) -> Vec<Option<ConstValue>> {
        let mut defaults = Vec::new();
        let mut first_default: Option<Span> = None;
        for (param, &(_, ty)) in data.params.iter().zip(params) {
            let Some(expr) = &param.default else {
                if let Some(first) = first_default
                    && param.name.text != "self"
                {
                    self.diags.push(
                        Diagnostic::error(format!(
                            "`{}` needs a default value, like the parameters before it",
                            param.name.text
                        ))
                        .with_code(codes::BAD_ARGUMENTS)
                        .with_primary(param.name.span, "no default value")
                        .with_secondary(first, "the first default value"),
                    );
                    first_default = None;
                }
                defaults.push(None);
                continue;
            };
            let span = expr.span().unwrap_or(param.name.span);
            first_default.get_or_insert(span);
            let value = match (self.eval(data.module, expr), self.types.kind(ty).clone()) {
                (Some(Value::Int(v)), TyKind::Int(int)) if int.contains(v) => {
                    Some(ConstValue::Int(v))
                }
                (Some(Value::Int(v)), TyKind::Int(int)) => {
                    self.diags.push(out_of_range(v, int, span));
                    None
                }
                (Some(Value::Bool(b)), TyKind::Bool) => Some(ConstValue::Bool(b)),
                (None, _) | (_, TyKind::Error) => None,
                (Some(found), kind) => {
                    let shown = self.display(ty);
                    let diagnostic = if matches!(kind, TyKind::Int(_) | TyKind::Bool) {
                        Diagnostic::error("mismatched types").with_primary(
                            span,
                            format!("expected `{shown}`, found {}", value_kind(&found)),
                        )
                    } else {
                        Diagnostic::error(format!("a parameter of type `{shown}` has no default"))
                            .with_primary(span, "defaults are integer or `bool` constants")
                    };
                    self.diags
                        .push(diagnostic.with_code(codes::MISMATCHED_TYPES));
                    None
                }
            };
            defaults.push(value);
        }
        defaults
    }

    // lowering types
    // -------------------------

    /// the type a written type stands for in `module`, using the generic
    /// parameters of the environment. An error is reported once and gives
    /// the error type.
    pub fn lower_ty(&mut self, module: ModuleId, ty: &TypeRef) -> TyId {
        match ty {
            TypeRef::Missing => self.types.error,
            TypeRef::Option(inner, _) => {
                let inner = self.lower_ty(module, inner);
                self.types.option(inner)
            }
            TypeRef::Path { path, args, span } => self.lower_path_ty(module, path, args, *span),
            TypeRef::Array(element, len, _) => {
                let element = self.lower_ty(module, element);
                match self.array_len(module, &TypeArgRef::Const(len.clone())) {
                    Some(len) => self.types.array(element, len),
                    None => self.types.error,
                }
            }
        }
    }

    /// the type a path names, with its arguments: a builtin, a kernel
    /// struct, a struct or enum (an instance if generic), an alias or a
    /// generic parameter. One argument after a non-generic type makes an
    /// array, as in `u8[4]`.
    fn lower_path_ty(
        &mut self,
        module: ModuleId,
        path: &PathRef,
        args: &[TypeArgRef],
        span: Span,
    ) -> TyId {
        let package = self.package;
        if let [segment] = path.segments.as_slice()
            && let Some(arg) = self.env_lookup(&segment.text)
        {
            return match arg {
                EnvArg::Type(ty) if args.is_empty() => ty,
                EnvArg::Type(ty) => match args {
                    // `T[N]`, an array of the type parameter
                    [arg] => match self.array_len(module, arg) {
                        Some(len) => self.types.array(ty, len),
                        None => self.types.error,
                    },
                    _ => self.types.error,
                },
                EnvArg::Const(..) => self.error(
                    Diagnostic::error(format!("`{path}` is a constant, not a type"))
                        .with_code(codes::WRONG_KIND)
                        .with_primary(path.span, "expected a type"),
                ),
            };
        }
        let def = match self.scopes.resolve_prefix(package, module, &path.segments) {
            Some((def, used)) if used == path.segments.len() => def,
            Some((Def::Error, _)) => return self.types.error,
            _ => {
                return self.error(
                    Diagnostic::error(format!("cannot find type `{path}`"))
                        .with_code(codes::UNRESOLVED_NAME)
                        .with_primary(path.span, "not found in this scope"),
                );
            }
        };
        let base = match def {
            Def::Builtin(Builtin::Int(int)) => self.types.int(int),
            Def::Builtin(Builtin::Bool) => self.types.bool,
            Def::Builtin(Builtin::F64) => self.types.float(),
            Def::Builtin(Builtin::Str) => self.types.str,
            Def::Builtin(Builtin::Bytes) => {
                let u8 = self.types.int(IntTy::U8);
                return match args {
                    [arg] => match self.array_len(module, arg) {
                        Some(len) => self.types.array(u8, len),
                        None => self.types.error,
                    },
                    _ => self.error(
                        Diagnostic::error("`bytes` needs a length")
                            .with_code(codes::WRONG_KIND)
                            .with_primary(span, "write it as `bytes[N]`"),
                    ),
                };
            }
            Def::Builtin(Builtin::Option) => {
                return match args {
                    [TypeArgRef::Type(inner)] => {
                        let inner = self.lower_ty(module, inner);
                        self.types.option(inner)
                    }
                    _ => self.error(
                        Diagnostic::error("`Option` takes one type argument")
                            .with_code(codes::WRONG_KIND)
                            .with_primary(span, "write it as `Option[T]` or `T?`"),
                    ),
                };
            }
            Def::Builtin(Builtin::KPtr) => {
                let pointee = match args {
                    [TypeArgRef::Type(inner)] => Some(self.lower_ty(module, inner)),
                    _ => None,
                };
                return match pointee.map(|t| (t, self.types.kind(t).clone())) {
                    Some((t, TyKind::Kernel(_))) => self.types.intern(TyKind::KPtr(t)),
                    Some((t, TyKind::Error)) => t,
                    _ => self.error(
                        Diagnostic::error("`kptr` points to a struct of the kernel")
                            .with_code(codes::WRONG_KIND)
                            .with_primary(span, "write it as `kptr[task_struct]`, importing the struct from `kernel`"),
                    ),
                };
            }
            Def::Kernel(id) => {
                if !args.is_empty() {
                    return self.error(
                        Diagnostic::error(format!("`{path}` is not generic"))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(span, "unexpected type arguments"),
                    );
                }
                return self.kernel_ty(id);
            }
            // `TracepointContext[event]`, typed by the struct of its event
            Def::Item(Item::Struct(s))
                if self.struct_contexts.get(s) == Some(&ContextKind::Tracepoint)
                    && !args.is_empty() =>
            {
                let event = match args {
                    [TypeArgRef::Type(inner)] => Some(self.lower_ty(module, inner)),
                    _ => None,
                };
                return match event.map(|t| self.types.kind(t).clone()) {
                    Some(TyKind::Kernel(id)) => self.types.intern(TyKind::TpContext(id)),
                    Some(TyKind::Error) => self.types.error,
                    _ => self.error(
                        Diagnostic::error(
                            "tracepoint contexts are typed by a struct of the kernel",
                        )
                        .with_code(codes::WRONG_KIND)
                        .with_primary(span, "write `TracepointContext[trace_event_raw_<event>]`"),
                    ),
                };
            }
            Def::Item(Item::Enum(e)) if self.is_template_enum(e) => {
                if args.is_empty() {
                    return self.error(
                        Diagnostic::error(format!("`{path}` takes generic arguments"))
                            .with_code(codes::BAD_ARGUMENTS)
                            .with_primary(span, format!("write `{path}[...]`")),
                    );
                }
                let lowered = self.type_generic_args(module, args);
                let Some(instance_args) = self.check_generic_args(Item::Enum(e), lowered, span)
                else {
                    return self.types.error;
                };
                return match self.instantiate(Item::Enum(e), instance_args) {
                    Some(Item::Enum(instance)) => self.types.intern(TyKind::Enum(instance)),
                    _ => self.types.error,
                };
            }
            Def::Item(Item::Struct(s)) if self.is_template_struct(s) => {
                if args.is_empty() {
                    return self.error(
                        Diagnostic::error(format!("`{path}` takes generic arguments"))
                            .with_code(codes::BAD_ARGUMENTS)
                            .with_primary(span, format!("write `{path}[...]`")),
                    );
                }
                let lowered = self.type_generic_args(module, args);
                let Some(instance_args) = self.check_generic_args(Item::Struct(s), lowered, span)
                else {
                    return self.types.error;
                };
                return match self.instantiate(Item::Struct(s), instance_args) {
                    Some(Item::Struct(instance)) => self.types.intern(TyKind::Struct(instance)),
                    _ => self.types.error,
                };
            }
            Def::Item(Item::Protocol(_)) => {
                return self.error(
                    Diagnostic::error(format!("`{path}` is a protocol, not a type"))
                        .with_code(codes::WRONG_KIND)
                        .with_primary(
                            path.span,
                            "protocols bound generic parameters, as in `[M: Model]`",
                        ),
                );
            }
            Def::Item(Item::Struct(s)) => self.types.intern(TyKind::Struct(s)),
            Def::Item(Item::Enum(e)) => self.types.intern(TyKind::Enum(e)),
            Def::Item(Item::Alias(a)) if !package.aliases[a].generic_params.is_empty() => {
                return self.generic_alias_ty(module, a, args, span);
            }
            Def::Item(Item::Alias(a)) => self.alias_ty(a),
            Def::Error => return self.types.error,
            _ => {
                return self.error(
                    Diagnostic::error(format!("`{path}` is not a type"))
                        .with_code(codes::WRONG_KIND)
                        .with_primary(path.span, "expected a type"),
                );
            }
        };
        match args {
            [] => base,
            // `T[N]`, an array
            [arg] => match self.array_len(module, arg) {
                Some(len) => self.types.array(base, len),
                None => self.types.error,
            },
            _ => self.error(
                Diagnostic::error(format!("`{path}` is not generic"))
                    .with_code(codes::WRONG_KIND)
                    .with_primary(span, "unexpected type arguments")
                    .with_note("an array type has one length, as in `u8[4]`"),
            ),
        }
    }

    /// checking that values can have type `ty`. A kernel struct can't,
    /// because it lives in kernel memory and is reached only through `kptr`.
    /// With no span we return the error type without reporting.
    pub fn value_type(&mut self, ty: TyId, span: Option<Span>) -> TyId {
        if let TyKind::Kernel(_) = self.types.kind(ty) {
            let name = self.display(ty);
            let Some(span) = span else {
                return self.types.error;
            };
            return self.error(
                Diagnostic::error(format!("`{name}` lives in kernel memory"))
                    .with_code(codes::WRONG_KIND)
                    .with_primary(span, format!("use `kptr[{name}]`")),
            );
        }
        ty
    }

    /// the type of a kernel struct from its BTF id. We record its name for
    /// messages.
    pub fn kernel_ty(&mut self, id: u32) -> TyId {
        if let Some(btf) = &self.options.btf {
            let name = match btf.get(id) {
                vigil_btf::Type::Struct(c) | vigil_btf::Type::Union(c) => {
                    c.name.clone().unwrap_or_else(|| "<anonymous>".into())
                }
                _ => btf.name(id),
            };
            self.types.name_kernel(id, name);
        }
        self.types.intern(TyKind::Kernel(id))
    }

    /// the length of `T[N]`, a constant from 1 to `u32::MAX`
    fn array_len(&mut self, module: ModuleId, arg: &TypeArgRef) -> Option<u64> {
        let expr = match arg {
            TypeArgRef::Const(expr) => expr.clone(),
            // the parser can't tell `T[N]` with a constant `N` from a type
            // with a type argument
            TypeArgRef::Type(TypeRef::Path { path, args, .. }) if args.is_empty() => {
                ConstExpr::Path(path.clone())
            }
            TypeArgRef::Type(ty) => {
                let span = ty.span()?;
                self.diags.push(
                    Diagnostic::error("expected an array length")
                        .with_code(codes::WRONG_KIND)
                        .with_primary(span, "found a type"),
                );
                return None;
            }
        };
        let span = expr.span()?;
        match self.eval(module, &expr)? {
            Value::Int(n) if (1..=i128::from(u32::MAX)).contains(&n) => Some(n as u64),
            Value::Int(n) => {
                self.diags.push(
                    Diagnostic::error(format!("invalid array length {n}"))
                        .with_code(codes::CONST_EVAL)
                        .with_primary(span, "lengths go from 1 to 4294967295"),
                );
                None
            }
            _ => {
                self.diags.push(
                    Diagnostic::error("expected an array length")
                        .with_code(codes::MISMATCHED_TYPES)
                        .with_primary(span, "not an integer"),
                );
                None
            }
        }
    }

    /// the type of an alias without generic parameters
    pub fn alias_ty(&mut self, id: AliasId) -> TyId {
        let data = &self.package.aliases[id];
        if !data.generic_params.is_empty() {
            return self.error(
                Diagnostic::error(format!("`{}` takes generic arguments", data.name.text))
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(data.name.span, "a generic alias"),
            );
        }
        match self.aliases.get(id) {
            Some(Memo::Done(ty)) => return *ty,
            Some(Memo::InProgress) => {
                let name = &self.package.aliases[id].name;
                return self.error(
                    Diagnostic::error(format!("type alias `{}` refers to itself", name.text))
                        .with_code(codes::CYCLE)
                        .with_primary(name.span, "cycle through this alias"),
                );
            }
            None => {}
        }
        self.aliases.insert(id, Memo::InProgress);
        let data = &self.package.aliases[id];
        let ty = self.lower_ty(data.module, &data.ty);
        self.aliases.insert(id, Memo::Done(ty));
        ty
    }

    /// the type of a use of a generic alias. The arguments are lowered where
    /// the alias is used. This is not memoized because it depends on the
    /// arguments, so the in-progress mark lasts only while we lower it.
    fn generic_alias_ty(
        &mut self,
        module: ModuleId,
        id: AliasId,
        args: &[TypeArgRef],
        span: Span,
    ) -> TyId {
        let package = self.package;
        let data = &package.aliases[id];
        if args.is_empty() {
            return self.error(
                Diagnostic::error(format!("`{}` takes generic arguments", data.name.text))
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(span, format!("write `{}[...]`", data.name.text)),
            );
        }
        if let Some(Memo::InProgress) = self.aliases.get(id) {
            return self.error(
                Diagnostic::error(format!("type alias `{}` refers to itself", data.name.text))
                    .with_code(codes::CYCLE)
                    .with_primary(data.name.span, "cycle through this alias"),
            );
        }
        let lowered = self.type_generic_args(module, args);
        if lowered.len() != data.generic_params.len() {
            return self.error(crate::generics::generic_count_error(
                &data.name,
                data.generic_params.len(),
                lowered.len(),
                span,
            ));
        }
        let mut env = Vec::new();
        for (param, (arg, arg_span)) in data.generic_params.iter().zip(lowered) {
            let constant = self.is_const_param(data.module, param);
            let value = match (constant, arg) {
                (_, GenericArgValue::Error) => return self.types.error,
                (true, GenericArgValue::Const(v)) => {
                    let bound = param.bound.clone().expect("constants have a bound");
                    EnvArg::Const(v, self.lower_ty(data.module, &bound))
                }
                (false, GenericArgValue::Type(ty)) => EnvArg::Type(ty),
                (true, GenericArgValue::Type(_)) => {
                    return self.error(
                        Diagnostic::error(format!("`{}` is a constant parameter", param.name.text))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(arg_span, "expected a constant, found a type")
                            .with_secondary(param.name.span, "declared here"),
                    );
                }
                (false, GenericArgValue::Const(_)) => {
                    return self.error(
                        Diagnostic::error(format!("`{}` is a type parameter", param.name.text))
                            .with_code(codes::WRONG_KIND)
                            .with_primary(arg_span, "expected a type, found a constant")
                            .with_secondary(param.name.span, "declared here"),
                    );
                }
            };
            env.push((param.name.text.clone(), value));
        }
        // the alias sees only its own parameters, not those of the use site
        let outer = std::mem::replace(&mut self.env, env);
        self.aliases.insert(id, Memo::InProgress);
        let ty = self.lower_ty(data.module, &data.ty);
        self.aliases.remove(id);
        self.env = outer;
        ty
    }

    // constants
    // -------------------------

    /// the type and value of a constant. The value is `None` when the
    /// constant has errors or is computed after type checking.
    pub fn const_info(&mut self, id: ConstId) -> (TyId, Option<ConstValue>) {
        match self.consts.get(id) {
            Some(Memo::Done(info)) => return *info,
            Some(Memo::InProgress) => {
                let name = &self.package.consts[id].name;
                self.diags.push(
                    Diagnostic::error(format!("constant `{}` depends on itself", name.text))
                        .with_code(codes::CYCLE)
                        .with_primary(name.span, "cycle through this constant"),
                );
                return (self.types.error, None);
            }
            None => {}
        }
        self.consts.insert(id, Memo::InProgress);
        let info = self.compute_const(id);
        self.consts.insert(id, Memo::Done(info));
        info
    }

    /// computing a constant for [`Db::const_info`]. Without a declared type
    /// it is an `i64`, or a `bool` if its value is one.
    fn compute_const(&mut self, id: ConstId) -> (TyId, Option<ConstValue>) {
        let data = &self.package.consts[id];
        check_reserved(&data.name, &mut self.diags);
        if let ConstExpr::Call(path, args, span) = &data.value {
            return self.comptime_const(id, path, args, *span);
        }
        if is_array_literal(&data.value) {
            return self.array_const(id);
        }
        let declared = data
            .ty
            .as_ref()
            .map(|t| (self.lower_ty(data.module, t), t.span()));
        if let Some((ty, span)) = declared
            && !matches!(
                self.types.kind(ty),
                TyKind::Int(_) | TyKind::Bool | TyKind::Error
            )
        {
            let name = self.display(ty);
            let mut diagnostic = Diagnostic::error(format!("constants cannot have type `{name}`"))
                .with_code(codes::WRONG_KIND)
                .with_primary(
                    span.unwrap_or(data.name.span),
                    "expected an integer or `bool`",
                );
            match self.types.kind(ty) {
                TyKind::Array(..) => {
                    diagnostic = diagnostic
                        .with_help("an array constant is written `[a, b, ...]` or `[x] * N`");
                }
                TyKind::Float => {
                    diagnostic = diagnostic
                        .with_help("compute with `f64` in a `comptime def` and return an integer");
                }
                _ => {}
            }
            self.diags.push(diagnostic);
            return (self.types.error, None);
        }
        let value = self.eval(data.module, &data.value);
        let span = data.value.span().unwrap_or(data.name.span);
        let ty = match declared {
            Some((ty, _)) => ty,
            None => match value {
                Some(Value::Bool(_)) => self.types.bool,
                _ => self.types.int(IntTy::I64),
            },
        };
        let value = match (value, self.types.kind(ty).clone()) {
            (None, _) | (_, TyKind::Error) => None,
            (Some(Value::Int(v)), TyKind::Int(int)) => {
                if int.contains(v) {
                    Some(ConstValue::Int(v))
                } else {
                    self.diags.push(out_of_range(v, int, span));
                    None
                }
            }
            (Some(Value::Bool(b)), TyKind::Bool) => Some(ConstValue::Bool(b)),
            (Some(found), _) => {
                let expected = self.display(ty);
                let mut diagnostic = Diagnostic::error("mismatched types")
                    .with_code(codes::MISMATCHED_TYPES)
                    .with_primary(
                        span,
                        format!("expected `{expected}`, found {}", value_kind(&found)),
                    );
                if let Some((_, Some(declared))) = declared {
                    diagnostic = diagnostic.with_secondary(declared, "the type of the constant");
                }
                self.diags.push(diagnostic);
                None
            }
        };
        (ty, value)
    }

    /// the type of a constant computed by a comptime function. Its type must
    /// be declared. The call is queued in [`Db::comptime_consts`] and run
    /// after type checking.
    fn comptime_const(
        &mut self,
        id: ConstId,
        path: &PathRef,
        args: &[ConstExpr],
        span: Span,
    ) -> (TyId, Option<ConstValue>) {
        let data = &self.package.consts[id];
        let Some(declared) = &data.ty else {
            self.diags.push(
                Diagnostic::error("constants computed by a function need a type")
                    .with_code(codes::CONST_EVAL)
                    .with_primary(
                        data.name.span,
                        format!("write `{}: T = ...`", data.name.text),
                    ),
            );
            return (self.types.error, None);
        };
        let ty = self.lower_ty(data.module, declared);
        if !self.comptime_storable(ty) {
            let name = self.display(ty);
            self.diags.push(
                Diagnostic::error(format!("constants cannot have type `{name}`"))
                    .with_code(codes::WRONG_KIND)
                    .with_primary(
                        declared.span().unwrap_or(data.name.span),
                        "expected integers, `bool`s or arrays of them",
                    ),
            );
            return (self.types.error, None);
        }
        let package = self.package;
        let function = match self
            .scopes
            .resolve_prefix(package, data.module, &path.segments)
        {
            Some((Def::Item(Item::Fn(f)), used))
                if used == path.segments.len() && package.fns[f].is_comptime =>
            {
                f
            }
            Some((Def::Error, _)) => return (ty, None),
            None => {
                self.diags.push(
                    Diagnostic::error(format!("cannot find `{path}` in this scope"))
                        .with_code(codes::UNRESOLVED_NAME)
                        .with_primary(path.span, "not found"),
                );
                return (ty, None);
            }
            Some(_) => {
                self.diags.push(
                    Diagnostic::error(format!("`{path}` is not a comptime function"))
                        .with_code(codes::CONST_EVAL)
                        .with_primary(span, "cannot be evaluated at compile time")
                        .with_help("constants can call functions declared with `comptime def`"),
                );
                return (ty, None);
            }
        };
        let values: Option<Vec<Value>> = args.iter().map(|a| self.eval(data.module, a)).collect();
        if let Some(values) = values {
            self.comptime_consts.push((id, function, values));
        }
        (ty, None)
    }

    /// can comptime code compute values of this type? Only integers, bools
    /// and arrays of them.
    fn comptime_storable(&self, ty: TyId) -> bool {
        match self.types.kind(ty) {
            TyKind::Int(_) | TyKind::Bool => true,
            TyKind::Array(element, _) => self.comptime_storable(*element),
            _ => false,
        }
    }

    /// is the constant a comptime call or an array? These are values of the
    /// program, not of the type checker, so constant expressions can't use
    /// them.
    pub fn is_comptime_const(&self, id: ConstId) -> bool {
        let value = &self.package.consts[id].value;
        matches!(value, ConstExpr::Call(..)) || is_array_literal(value)
    }

    /// the type of an array constant, as `W: i32[3] = [1, 2, 3]` or
    /// `[0] * 3`. Its value is checked and stored in [`Db::array_consts`].
    fn array_const(&mut self, id: ConstId) -> (TyId, Option<ConstValue>) {
        let data = &self.package.consts[id];
        let Some(declared) = &data.ty else {
            self.diags.push(
                Diagnostic::error("array constants need a type")
                    .with_code(codes::CONST_EVAL)
                    .with_primary(
                        data.name.span,
                        format!("write `{}: T[N] = [...]`", data.name.text),
                    ),
            );
            return (self.types.error, None);
        };
        let ty = self.lower_ty(data.module, declared);
        if self.types.is_error(ty) {
            return (ty, None);
        }
        if !matches!(self.types.kind(ty), TyKind::Array(..)) {
            let name = self.display(ty);
            let mut diagnostic = Diagnostic::error("mismatched types")
                .with_code(codes::MISMATCHED_TYPES)
                .with_primary(
                    data.value.span().unwrap_or(data.name.span),
                    format!("expected `{name}`, found an array"),
                );
            if let Some(span) = declared.span() {
                diagnostic = diagnostic.with_secondary(span, "the type of the constant");
            }
            self.diags.push(diagnostic);
            return (self.types.error, None);
        }
        if !self.comptime_storable(ty) {
            let name = self.display(ty);
            self.diags.push(
                Diagnostic::error(format!("`{name}` is not an array of integers or `bool`s"))
                    .with_code(codes::WRONG_KIND)
                    .with_primary(
                        declared.span().unwrap_or(data.name.span),
                        "the type of the constant",
                    )
                    .with_note("array constants hold integers or `bool`s"),
            );
            return (self.types.error, None);
        }
        if let Some(value) = self.array_value(data.module, &data.value, ty) {
            self.array_consts.push((id, value));
        }
        (ty, None)
    }

    /// the value of an array constant, checking the length of every array
    /// and the range of every integer against `ty`
    fn array_value(&mut self, module: ModuleId, expr: &ConstExpr, ty: TyId) -> Option<CValue> {
        let file = self.package.modules[module].file;
        let span = expr.span().unwrap_or(Span::new(file, 0..0));
        match (expr, self.types.kind(ty).clone()) {
            (ConstExpr::Array(items, _), TyKind::Array(element, len)) => {
                if items.len() as u64 != len {
                    self.diags.push(
                        Diagnostic::error(format!(
                            "expected {}, found {}",
                            vigil_diagnostics::plural(len, "element"),
                            items.len()
                        ))
                        .with_code(codes::MISMATCHED_TYPES)
                        .with_primary(span, "wrong length"),
                    );
                    return None;
                }
                let values: Option<Vec<_>> = items
                    .iter()
                    .map(|item| self.array_value(module, item, element))
                    .collect();
                Some(CValue::Array(values?))
            }
            // `[x] * N`
            (ConstExpr::Binary(BinaryOp::Mul, lhs, count, _), TyKind::Array(element, len)) if matches!(**lhs, ConstExpr::Array(ref items, _) if items.len() == 1) =>
            {
                let ConstExpr::Array(items, _) = &**lhs else {
                    return None;
                };
                match self.eval(module, count) {
                    Some(Value::Int(n)) if n == i128::from(len) => {}
                    Some(Value::Int(n)) => {
                        self.diags.push(
                            Diagnostic::error(format!(
                                "expected {}, found {n}",
                                vigil_diagnostics::plural(len, "element")
                            ))
                            .with_code(codes::MISMATCHED_TYPES)
                            .with_primary(span, "wrong length"),
                        );
                        return None;
                    }
                    Some(found) => {
                        self.diags.push(
                            Diagnostic::error("mismatched types")
                                .with_code(codes::MISMATCHED_TYPES)
                                .with_primary(
                                    count.span().unwrap_or(span),
                                    format!("expected an integer, found {}", value_kind(&found)),
                                ),
                        );
                        return None;
                    }
                    None => return None,
                }
                let item = self.array_value(module, &items[0], element)?;
                Some(CValue::Array(vec![item; len as usize]))
            }
            (_, TyKind::Int(int)) => match self.eval(module, expr)? {
                Value::Int(v) if int.contains(v) => Some(CValue::Int(v)),
                Value::Int(v) => {
                    self.diags.push(out_of_range(v, int, span));
                    None
                }
                found => {
                    self.diags.push(
                        Diagnostic::error("mismatched types")
                            .with_code(codes::MISMATCHED_TYPES)
                            .with_primary(
                                span,
                                format!("expected `{}`, found {}", int.name(), value_kind(&found)),
                            ),
                    );
                    None
                }
            },
            (_, TyKind::Bool) => match self.eval(module, expr)? {
                Value::Bool(b) => Some(CValue::Bool(b)),
                found => {
                    self.diags.push(
                        Diagnostic::error("mismatched types")
                            .with_code(codes::MISMATCHED_TYPES)
                            .with_primary(
                                span,
                                format!("expected `bool`, found {}", value_kind(&found)),
                            ),
                    );
                    None
                }
            },
            _ => {
                self.diags.push(
                    Diagnostic::error("expected an array")
                        .with_code(codes::MISMATCHED_TYPES)
                        .with_primary(span, "write `[...]`"),
                );
                None
            }
        }
    }

    /// evaluating a constant expression (literals, constants, constant
    /// generic parameters and operators) with unbounded integers. The caller
    /// checks the range. `None` means an error, already reported.
    pub fn eval(&mut self, module: ModuleId, expr: &ConstExpr) -> Option<Value> {
        match expr {
            ConstExpr::Missing => None,
            ConstExpr::Int(v, span) => match i128::try_from(*v) {
                Ok(v) => Some(Value::Int(v)),
                Err(_) => {
                    self.diags.push(
                        Diagnostic::error("integer is too large")
                            .with_code(codes::CONST_EVAL)
                            .with_primary(*span, "constant expressions compute with 128 bits"),
                    );
                    None
                }
            },
            ConstExpr::Bool(b, _) => Some(Value::Bool(*b)),
            ConstExpr::Str(s, _) => Some(Value::Str(s.clone())),
            ConstExpr::Invalid(span) => {
                self.diags.push(
                    Diagnostic::error("not a constant expression")
                        .with_code(codes::CONST_EVAL)
                        .with_primary(*span, "cannot be evaluated at compile time")
                        .with_note(
                            "constants are made of literals, other constants and operators; \
                             call a `comptime def` to compute others",
                        ),
                );
                None
            }
            ConstExpr::Path(path) => {
                if let [segment] = path.segments.as_slice()
                    && let Some(EnvArg::Const(v, _)) = self.env_lookup(&segment.text)
                {
                    return Some(Value::Int(v));
                }
                let package = self.package;
                match self.scopes.resolve_prefix(package, module, &path.segments) {
                    Some((Def::Item(Item::Const(c)), _)) if self.is_comptime_const(c) => {
                        self.diags.push(
                            Diagnostic::error(format!("`{path}` is computed after type checking"))
                                .with_code(codes::CONST_EVAL)
                                .with_primary(path.span, "cannot be used in constant expressions")
                                .with_note("constants computed by comptime functions can only be used in code"),
                        );
                        None
                    }
                    Some((Def::Item(Item::Const(c)), used)) if used == path.segments.len() => {
                        match self.const_info(c).1? {
                            ConstValue::Int(v) => Some(Value::Int(v)),
                            ConstValue::Bool(b) => Some(Value::Bool(b)),
                        }
                    }
                    Some((Def::Error, _)) => None,
                    None => {
                        self.diags.push(
                            Diagnostic::error(format!("cannot find `{path}` in this scope"))
                                .with_code(codes::UNRESOLVED_NAME)
                                .with_primary(path.span, "not found"),
                        );
                        None
                    }
                    Some(_) => {
                        self.diags.push(
                            Diagnostic::error(format!("`{path}` is not a constant"))
                                .with_code(codes::CONST_EVAL)
                                .with_primary(path.span, "expected a constant"),
                        );
                        None
                    }
                }
            }
            ConstExpr::Unary(op, inner, span) => {
                let value = self.eval(module, inner)?;
                match (op, value) {
                    (UnaryOp::Neg, Value::Int(v)) => Some(Value::Int(-v)),
                    (UnaryOp::BitNot, Value::Int(v)) => Some(Value::Int(!v)),
                    (UnaryOp::Not, Value::Bool(b)) => Some(Value::Bool(!b)),
                    (_, found) => {
                        let (symbol, takes) = match op {
                            UnaryOp::Neg => ("-", "an integer"),
                            UnaryOp::BitNot => ("~", "an integer"),
                            UnaryOp::Not => ("not", "`bool`"),
                        };
                        let found = value_kind(&found);
                        self.diags.push(
                            Diagnostic::error(format!("`{symbol}` cannot be applied to {found}"))
                                .with_code(codes::BAD_OPERAND)
                                .with_primary(
                                    inner.span().unwrap_or(*span),
                                    format!("expected {takes}, found {found}"),
                                ),
                        );
                        None
                    }
                }
            }
            ConstExpr::Array(_, span) => {
                self.diags.push(
                    Diagnostic::error("arrays can only initialize a constant")
                        .with_code(codes::CONST_EVAL)
                        .with_primary(*span, "not allowed inside a constant expression")
                        .with_help("declare `const NAME: T[N] = [...]` and use `NAME`"),
                );
                None
            }
            ConstExpr::Call(_, _, span) => {
                self.diags.push(
                    Diagnostic::error("calls can only initialize a constant")
                        .with_code(codes::CONST_EVAL)
                        .with_primary(*span, "not allowed inside a constant expression")
                        .with_help("declare `const NAME: T = f(...)` and use `NAME`"),
                );
                None
            }
            ConstExpr::Binary(op, lhs, rhs, span) => {
                let lhs = self.eval(module, lhs)?;
                let rhs = self.eval(module, rhs)?;
                let (kinds, amount) = ((value_kind(&lhs), value_kind(&rhs)), rhs.clone());
                let result = eval_binary(*op, lhs, rhs);
                if let Err(message) = &result {
                    self.diags
                        .push(binary_error(*op, kinds, &amount, message, *span));
                }
                result.ok()
            }
        }
    }

    // recursive types
    // -------------------------

    /// reporting structs and enums that contain themselves by value, since
    /// they would have an infinite size
    fn check_infinite_types(&mut self) {
        let package = self.package;
        let mut reported = HashSet::new();
        for (id, data) in package.structs.iter() {
            let start = TyKind::Struct(id);
            if self.contains(&start, &start, &mut HashSet::new()) && reported.insert(start) {
                self.diags.push(infinite(&data.name));
            }
        }
        for (id, data) in package.enums.iter() {
            let start = TyKind::Enum(id);
            if self.contains(&start, &start, &mut HashSet::new()) && reported.insert(start) {
                self.diags.push(infinite(&data.name));
            }
        }
    }

    /// does `outer` contain `target` by value, directly or not? `seen` stops
    /// cycles that don't go through `target`.
    fn contains(&self, outer: &TyKind, target: &TyKind, seen: &mut HashSet<TyKind>) -> bool {
        if !seen.insert(outer.clone()) {
            return false;
        }
        let fields: Vec<TyId> = match outer {
            TyKind::Struct(s) => self.struct_fields.get(*s).cloned().unwrap_or_default(),
            TyKind::Enum(e) => self
                .variant_fields
                .get(*e)
                .into_iter()
                .flatten()
                .flatten()
                .flatten()
                .copied()
                .collect(),
            _ => Vec::new(),
        };
        fields.into_iter().any(|field| {
            let nominal = self.nominal_part(field);
            nominal.as_ref() == Some(target)
                || nominal.is_some_and(|n| self.contains(&n, target, seen))
        })
    }

    /// the struct or enum stored by value in a field of type `ty`, looking
    /// through options and arrays. References and pointers store nothing.
    fn nominal_part(&self, ty: TyId) -> Option<TyKind> {
        match self.types.kind(ty) {
            kind @ (TyKind::Struct(_) | TyKind::Enum(_)) => Some(kind.clone()),
            TyKind::Option(inner) | TyKind::Array(inner, _) => self.nominal_part(*inner),
            _ => None,
        }
    }
}

/// evaluating a binary operation on constants, or the error message
pub(crate) fn eval_binary(op: BinaryOp, lhs: Value, rhs: Value) -> Result<Value, String> {
    use BinaryOp::*;
    let overflow = || "arithmetic overflow".to_string();
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => Ok(match op {
            Add => Value::Int(a.checked_add(b).ok_or_else(overflow)?),
            Sub => Value::Int(a.checked_sub(b).ok_or_else(overflow)?),
            Mul => Value::Int(a.checked_mul(b).ok_or_else(overflow)?),
            FloorDiv | Rem if b == 0 => return Err("division by zero".into()),
            // division truncates toward zero as eBPF does, not toward
            // negative infinity as in Python
            FloorDiv => Value::Int(a.checked_div(b).ok_or_else(overflow)?),
            Rem => Value::Int(a.checked_rem(b).ok_or_else(overflow)?),
            Div => return Err("integer division is written `//`".into()),
            Shl | Shr if !(0..128).contains(&b) => {
                return Err("shift amount out of range".into());
            }
            Shl => {
                let shifted = a << b;
                if shifted >> b != a {
                    return Err(overflow());
                }
                Value::Int(shifted)
            }
            Shr => Value::Int(a >> b),
            BitAnd => Value::Int(a & b),
            BitOr => Value::Int(a | b),
            BitXor => Value::Int(a ^ b),
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            Lt => Value::Bool(a < b),
            Le => Value::Bool(a <= b),
            Gt => Value::Bool(a > b),
            Ge => Value::Bool(a >= b),
            And | Or => return Err("`and` and `or` take `bool` operands".into()),
        }),
        (Value::Bool(a), Value::Bool(b)) => Ok(match op {
            And => Value::Bool(a && b),
            Or => Value::Bool(a || b),
            Eq => Value::Bool(a == b),
            Ne => Value::Bool(a != b),
            _ => return Err(format!("`{}` does not apply to `bool`", op.symbol())),
        }),
        _ => Err("invalid operands for this operator".into()),
    }
}

/// what a constant value is, in the words of the errors
pub(crate) fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Int(_) => "an integer",
        Value::Bool(_) => "`bool`",
        Value::Str(_) => "a string",
    }
}

/// the error of a binary operation on constants, from the message of
/// [`eval_binary`], the kinds of the operands and the right operand. The
/// constants of bodies give the same errors
pub(crate) fn binary_error(
    op: BinaryOp,
    kinds: (&str, &str),
    rhs: &Value,
    message: &str,
    span: Span,
) -> Diagnostic {
    let symbol = op.symbol();
    let error = |label: String| {
        Diagnostic::error(message.to_string())
            .with_code(codes::CONST_EVAL)
            .with_primary(span, label)
    };
    match (kinds, rhs) {
        (("an integer", "an integer"), Value::Int(b)) => match op {
            BinaryOp::FloorDiv | BinaryOp::Rem if *b == 0 => error("the divisor is 0".into()),
            BinaryOp::Shl | BinaryOp::Shr if !(0..128).contains(b) => {
                error(format!("shifts by {b}, the amount goes from 0 to 127"))
            }
            BinaryOp::Div => error("`/` divides `f64` values".into()).with_help("write `//`"),
            BinaryOp::And | BinaryOp::Or => {
                Diagnostic::error(format!("`{symbol}` cannot be applied to integers"))
                    .with_code(codes::BAD_OPERAND)
                    .with_primary(span, "expected `bool` operands, found integers")
                    .with_help("compare the integers first, as in `a != 0`")
            }
            _ => error("the result does not fit in 128 bits".into()),
        },
        (("`bool`", "`bool`"), _) => {
            Diagnostic::error(format!("`{symbol}` cannot be applied to `bool`"))
                .with_code(codes::BAD_OPERAND)
                .with_primary(span, "invalid operands of type `bool`")
        }
        ((a, b), _) => Diagnostic::error("mismatched types")
            .with_code(codes::MISMATCHED_TYPES)
            .with_primary(span, format!("`{symbol}` between {a} and {b}")),
    }
}

/// the error for an integer that does not fit its type
pub(crate) fn out_of_range(value: impl std::fmt::Display, int: IntTy, span: Span) -> Diagnostic {
    Diagnostic::error(format!("`{value}` does not fit in `{}`", int.name()))
        .with_code(codes::LITERAL_OUT_OF_RANGE)
        .with_primary(
            span,
            format!("`{}` goes from {} to {}", int.name(), int.min(), int.max()),
        )
}

/// the error for a type that contains itself
fn infinite(name: &Ident) -> Diagnostic {
    Diagnostic::error(format!("type `{}` contains itself", name.text))
        .with_code(codes::INFINITE_TYPE)
        .with_primary(name.span, "this type would have an infinite size")
        .with_note("eBPF programs have no heap; store an index into a map instead")
}

/// reporting names defined twice. Empty names, left by syntax errors, are
/// skipped.
fn check_unique<'a>(names: impl Iterator<Item = &'a Ident>, what: &str, diags: &mut Diagnostics) {
    let mut seen: Vec<&Ident> = Vec::new();
    for name in names {
        if name.text.is_empty() {
            continue;
        }
        if let Some(first) = seen.iter().find(|n| n.text == name.text) {
            diags.push(
                Diagnostic::error(format!("{what} `{}` is defined twice", name.text))
                    .with_code(codes::DUPLICATE_DEFINITION)
                    .with_primary(name.span, "redefined here")
                    .with_secondary(first.span, "first defined here"),
            );
        } else {
            seen.push(name);
        }
    }
}

/// reporting a type, protocol or constant named as a builtin type, because
/// the name would be ambiguous in signatures
fn check_reserved(name: &Ident, diags: &mut Diagnostics) {
    let reserved = IntTy::ALL.iter().any(|i| i.name() == name.text)
        || matches!(name.text.as_str(), "bool" | "str" | "bytes" | "Option");
    if reserved {
        diags.push(
            Diagnostic::error(format!("`{}` is a builtin type", name.text))
                .with_code(codes::DUPLICATE_DEFINITION)
                .with_primary(name.span, "cannot be redefined"),
        );
    }
}

/// is the expression an array literal, `[...]` or `[x] * N`?
fn is_array_literal(expr: &ConstExpr) -> bool {
    match expr {
        ConstExpr::Array(..) => true,
        ConstExpr::Binary(BinaryOp::Mul, lhs, _, _) => matches!(**lhs, ConstExpr::Array(..)),
        _ => false,
    }
}
