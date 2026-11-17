//! this module checks the decorators that connect functions and structs to
//! the kernel: programs (`@kprobe`, `@xdp`, ...), `@tail`, `@test`,
//! `@property`, contracts, `@context`, `@helper` and `@network_order`.
//! The checks run from [`Db`] while collecting signatures and fill its tables.

use vigil_diagnostics::{Diagnostic, Span, codes};

use crate::db::{Db, Value};
use crate::hir::*;
use crate::maps::MapKind;
use crate::ty::{TyId, TyKind};

/// what the context pointer of a program points to. Each kind is a struct
/// without fields of the standard library, marked with `@context`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextKind {
    /// `struct pt_regs *`, for kprobes, kretprobes, uprobes and uretprobes
    Kprobe,
    /// the record of a tracepoint
    Tracepoint,
    /// `struct xdp_md *`
    Xdp,
    /// `struct __sk_buff *`
    Tc,
    /// the arguments of an LSM hook, as 64-bit words
    Lsm,
}

impl ContextKind {
    fn by_name(name: &str) -> Option<ContextKind> {
        match name {
            "kprobe" => Some(ContextKind::Kprobe),
            "tracepoint" => Some(ContextKind::Tracepoint),
            "xdp" => Some(ContextKind::Xdp),
            "tc" => Some(ContextKind::Tc),
            "lsm" => Some(ContextKind::Lsm),
            _ => None,
        }
    }

    /// the name of the standard library struct for this context
    pub fn struct_name(self) -> &'static str {
        match self {
            ContextKind::Kprobe => "KprobeContext",
            ContextKind::Tracepoint => "TracepointContext",
            ContextKind::Xdp => "XdpContext",
            ContextKind::Tc => "TcContext",
            ContextKind::Lsm => "LsmContext",
        }
    }

    /// the standard library module that defines the struct
    pub fn module(self) -> &'static str {
        match self {
            ContextKind::Kprobe | ContextKind::Tracepoint | ContextKind::Lsm => "std.bpf",
            ContextKind::Xdp | ContextKind::Tc => "std.net",
        }
    }

    /// does the context give access to a packet, with `parse` and `len`?
    pub fn has_packet(self) -> bool {
        matches!(self, ContextKind::Xdp | ContextKind::Tc)
    }
}

/// the type of a program, as named by its decorator
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramKind {
    Kprobe,
    Kretprobe,
    Tracepoint,
    Xdp,
    Tc,
    Uprobe,
    Uretprobe,
    Lsm,
    /// `@test`, loaded as an XDP program and run by the test runner of the
    /// full Vigil
    Test,
}

impl ProgramKind {
    /// the decorators that make a function a program. `@test` is not here
    /// because tests are checked separately.
    pub const DECORATORS: [&str; 8] = [
        "kprobe",
        "kretprobe",
        "tracepoint",
        "xdp",
        "tc",
        "uprobe",
        "uretprobe",
        "lsm",
    ];

    /// the name of the decorator, also the ELF section prefix of a tracing
    /// program
    pub fn name(self) -> &'static str {
        match self {
            ProgramKind::Kprobe => "kprobe",
            ProgramKind::Kretprobe => "kretprobe",
            ProgramKind::Tracepoint => "tracepoint",
            ProgramKind::Xdp => "xdp",
            ProgramKind::Tc => "tc",
            ProgramKind::Uprobe => "uprobe",
            ProgramKind::Uretprobe => "uretprobe",
            ProgramKind::Lsm => "lsm",
            ProgramKind::Test => "test",
        }
    }

    /// the kind of program with its article, for messages ("an XDP program")
    pub fn noun(self) -> &'static str {
        match self {
            ProgramKind::Kprobe => "a kprobe program",
            ProgramKind::Kretprobe => "a kretprobe program",
            ProgramKind::Tracepoint => "a tracepoint program",
            ProgramKind::Xdp => "an XDP program",
            ProgramKind::Tc => "a TC program",
            ProgramKind::Uprobe => "a uprobe program",
            ProgramKind::Uretprobe => "a uretprobe program",
            ProgramKind::Lsm => "an LSM program",
            ProgramKind::Test => "a test",
        }
    }

    /// the context taken by a program of this type
    pub fn context(self) -> ContextKind {
        match self {
            ProgramKind::Kprobe
            | ProgramKind::Kretprobe
            | ProgramKind::Uprobe
            | ProgramKind::Uretprobe => ContextKind::Kprobe,
            ProgramKind::Lsm => ContextKind::Lsm,
            ProgramKind::Tracepoint => ContextKind::Tracepoint,
            // a test takes no context, but it is loaded as an XDP program
            ProgramKind::Xdp | ProgramKind::Test => ContextKind::Xdp,
            ProgramKind::Tc => ContextKind::Tc,
        }
    }

    /// the name of the action enum of a packet program, `None` for others
    pub fn action_enum(self) -> Option<&'static str> {
        match self {
            ProgramKind::Xdp => Some("XdpAction"),
            ProgramKind::Tc => Some("TcAction"),
            _ => None,
        }
    }

    /// what a program returns when `?` meets `None`: letting the packet go
    /// on, with `XDP_PASS` (2) or `TC_ACT_OK` (0). The kernel ignores the
    /// result of a tracing program.
    pub fn default_return(self) -> i64 {
        match self {
            ProgramKind::Xdp => 2,
            _ => 0,
        }
    }

    /// what a program returns when it aborts (index out of bounds, full map).
    /// It is 0 for every kind: `XDP_ABORTED` for XDP, which the
    /// `xdp:xdp_exception` tracepoint reports, and `TC_ACT_OK` for TC, which
    /// has no error action.
    pub fn abort_return(self) -> i64 {
        0
    }
}

/// what makes a function an eBPF program: its type and where it is attached
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramInfo {
    pub kind: ProgramKind,
    /// what the program is attached to: a kernel function, `binary:symbol`,
    /// an LSM hook, `category/name`, or `ingress` / `egress` for TC. Empty
    /// for XDP and tests.
    pub target: String,
    /// the network interface of a packet program, if the decorator names
    /// one. The loader of the full Vigil can override it.
    pub iface: Option<String>,
    /// `@tail(map, index)`: the program goes in a program array for
    /// `tail_call` instead of being attached. The index is within bounds.
    pub tail: Option<(String, u32)>,
}

impl ProgramInfo {
    /// the ELF section of the program. It tells a loader the program type
    /// and, for tracing programs, the target.
    pub fn section(&self) -> String {
        match self.kind {
            ProgramKind::Xdp | ProgramKind::Test => "xdp".into(),
            ProgramKind::Tc => "classifier".into(),
            kind => format!("{}/{}", kind.name(), self.target),
        }
    }
}

/// the error for an unknown decorator, listing the known ones
pub(crate) fn unknown_decorator(decorator: &Decorator) -> Diagnostic {
    Diagnostic::error(format!("unknown decorator `@{}`", decorator.name))
        .with_code(codes::BAD_DECORATOR)
        .with_primary(decorator.span, "not a Vigil decorator")
        .with_note(
            "programs: @kprobe, @kretprobe, @tracepoint, @uprobe, @uretprobe, @lsm, @xdp, @tc, \
             @tail; tests: @test; contracts: @requires, @ensures; methods: @mut; loops: \
             @unroll, @no_unroll, @bounded, @invariant; structs: @network_order",
        )
}

/// the error for a decorator reserved to the standard library
fn std_only(decorator: &Decorator) -> Diagnostic {
    Diagnostic::error(format!(
        "`@{}` is reserved to the standard library",
        decorator.name
    ))
    .with_code(codes::BAD_DECORATOR)
    .with_primary(decorator.span, "not allowed here")
}

impl Db<'_> {
    /// checking the decorators of a struct and recording contexts and
    /// network order structs. Only `@network_order` and, in the standard
    /// library, `@context` are allowed.
    pub(crate) fn check_struct_decorators(&mut self, id: StructId) {
        let package = self.package;
        let data = &package.structs[id];
        let is_std = package.modules[data.module].is_std();
        for decorator in &data.decorators {
            match decorator.name.to_string().as_str() {
                "network_order" if decorator.args.is_empty() => {
                    self.network_order.insert(id, ());
                }
                "network_order" => self.diags.push(
                    Diagnostic::error("`@network_order` takes no arguments")
                        .with_code(codes::BAD_DECORATOR)
                        .with_primary(decorator.span, "write `@network_order`"),
                ),
                "context" if !is_std => self.diags.push(std_only(decorator)),
                "context" => {
                    let kind = match decorator.args.as_slice() {
                        [arg] => match self.eval(data.module, &arg.value) {
                            Some(Value::Str(name)) => ContextKind::by_name(&name),
                            _ => None,
                        },
                        _ => None,
                    };
                    match kind {
                        Some(kind) if data.fields.is_empty() => {
                            self.struct_contexts.insert(id, kind);
                        }
                        Some(_) => self.diags.push(
                            Diagnostic::error("contexts have no fields")
                                .with_code(codes::BAD_DECORATOR)
                                .with_primary(data.name.span, "remove the fields"),
                        ),
                        None => self.diags.push(
                            Diagnostic::error("unknown context kind")
                                .with_code(codes::BAD_DECORATOR)
                                .with_primary(
                                    decorator.span,
                                    "expected a program type, as in `@context(\"xdp\")`",
                                ),
                        ),
                    }
                }
                _ => self.diags.push(
                    Diagnostic::error(format!(
                        "`@{}` is not a decorator of structs",
                        decorator.name
                    ))
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(decorator.span, "not allowed on a struct")
                    .with_help("structs take `@network_order`"),
                ),
            }
        }
    }

    /// checking the decorators of a function and recording helpers and
    /// programs. A function is at most one of program, test, property or
    /// helper. `@tail` comes last because it needs to know if the function
    /// is a program.
    pub(crate) fn check_fn_decorators(&mut self, id: FnId) {
        let package = self.package;
        let data = &package.fns[id];
        let mut first: Option<&Decorator> = None;
        let mut tail = None;
        for decorator in &data.decorators {
            let name = decorator.name.to_string();
            if name == "tail" {
                tail = self
                    .tail(data.module, decorator)
                    .map(|slot| (slot, decorator.span));
                continue;
            }
            let is_program = ProgramKind::DECORATORS.contains(&name.as_str());
            if is_program || name == "helper" || name == "test" || name == "property" {
                if let Some(first) = first {
                    self.diags.push(
                        Diagnostic::error(
                            "a function can be only one program, test, property or helper",
                        )
                        .with_code(codes::BAD_DECORATOR)
                        .with_primary(decorator.span, "second decorator")
                        .with_secondary(first.span, "first one here"),
                    );
                    continue;
                }
                first = Some(decorator);
            }
            if name == "helper" {
                self.helper(id, decorator);
            } else if name == "test" {
                self.test(id, decorator);
            } else if name == "property" {
                self.property(id, decorator);
            } else if is_program {
                self.program(id, decorator, &name);
            } else if name == "mut" {
                let is_method = data.params.first().is_some_and(|p| p.name.text == "self");
                if !is_method {
                    self.diags.push(
                        Diagnostic::error("`@mut` applies to methods that take `self`")
                            .with_code(codes::BAD_DECORATOR)
                            .with_primary(
                                decorator.span,
                                format!("`{}` does not take `self`", data.name.text),
                            ),
                    );
                } else if !decorator.args.is_empty() {
                    self.diags.push(
                        Diagnostic::error("`@mut` takes no arguments")
                            .with_code(codes::BAD_DECORATOR)
                            .with_primary(decorator.span, "write `@mut`"),
                    );
                }
            } else if matches!(name.as_str(), "requires" | "ensures") {
                self.contract_decorator(id, decorator, &name);
            } else if matches!(
                name.as_str(),
                "unroll" | "no_unroll" | "bounded" | "invariant"
            ) {
                self.diags.push(
                    Diagnostic::error(format!("`@{name}` applies to loops"))
                        .with_code(codes::BAD_DECORATOR)
                        .with_primary(decorator.span, "not allowed on a function")
                        .with_help(format!(
                            "write `@{name}` on the line before a `for` or `while` loop"
                        )),
                );
            } else {
                self.diags.push(unknown_decorator(decorator));
            }
        }
        if let Some((slot, span)) = tail {
            match self.programs.get_mut(id) {
                Some(info) => info.tail = Some(slot),
                None => self.diags.push(
                    Diagnostic::error("`@tail` applies to programs")
                        .with_code(codes::BAD_DECORATOR)
                        .with_primary(span, "this function is not a program")
                        .with_help("add the decorator of its program type, as in `@xdp`"),
                ),
            }
        }
    }

    /// checking `@test` (a top-level function without parameters returning
    /// `bool`) and recording the test as a program
    fn test(&mut self, id: FnId, decorator: &Decorator) {
        let data = &self.package.fns[id];
        let sig = &self.fn_sigs[id];
        let returns_bool =
            matches!(self.types.kind(sig.ret), TyKind::Bool) || self.types.is_error(sig.ret);
        if !decorator.args.is_empty()
            || data.owner.is_some()
            || !sig.params.is_empty()
            || !returns_bool
        {
            self.diags.push(
                Diagnostic::error("a test is a function without parameters that returns `bool`")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(data.name.span, "not a test")
                    .with_help("write `@test` before `def name() -> bool:`"),
            );
            return;
        }
        self.programs.insert(
            id,
            ProgramInfo {
                kind: ProgramKind::Test,
                target: String::new(),
                iface: None,
                tail: None,
            },
        );
    }

    /// checking `@property`, `@property(cases=N)` and `@property(x=g)`.
    /// A property is a top-level, non-generic function returning `bool`,
    /// whose parameters are generated by the test runner of the full Vigil,
    /// so they must be data. `cases` goes from 1 to 1 000 000, and `x=g`
    /// names the generator of `x`.
    fn property(&mut self, id: FnId, decorator: &Decorator) {
        let data = &self.package.fns[id];
        let sig = self.fn_sigs[id].clone();
        let returns_bool =
            matches!(self.types.kind(sig.ret), TyKind::Bool) || self.types.is_error(sig.ret);
        if data.owner.is_some() || !data.generic_params.is_empty() || !returns_bool {
            self.diags.push(
                Diagnostic::error("a property is a function that returns `bool`")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(data.name.span, "not a property")
                    .with_help(
                        "write `@property` before `def name(x: u32, ...) -> bool:`, outside any struct and without generic parameters",
                    ),
            );
            return;
        }
        for arg in &decorator.args {
            // `x=g`, the values of `x` are made by `g`
            if let (Some(name), ConstExpr::Path(path)) = (&arg.name, &arg.value)
                && let Some(k) = data.params.iter().position(|p| p.name.text == name.text)
            {
                let ty = sig.params[k].1;
                self.check_generator(data.module, path, ty);
                continue;
            }
            let cases = match (&arg.name, &arg.value) {
                (Some(name), ConstExpr::Int(n, _)) if name.text == "cases" => Some(*n),
                _ => None,
            };
            if !matches!(cases, Some(n) if n > 0 && n <= 1_000_000) {
                self.diags.push(
                    Diagnostic::error("invalid argument for `@property`")
                        .with_code(codes::BAD_DECORATOR)
                        .with_primary(
                            decorator.span,
                            "expected `@property`, `@property(cases=N)` or `@property(x=generator)`",
                        )
                        .with_help("`cases` is the number of values tried, from 1 to 1000000; `x=g` makes the values of the parameter `x` with the function `g`"),
                );
            }
        }
        for (param, (_, ty)) in data.params.iter().zip(&sig.params) {
            if !self.is_data(*ty) {
                self.diags.push(
                    Diagnostic::error("the parameters of a property must be data")
                        .with_code(codes::BAD_DECORATOR)
                        .with_primary(param.name.span, "values of this type cannot be generated")
                        .with_help(
                            "integers, booleans, arrays, structs, enums and options of those; not maps, references, contexts or kernel pointers",
                        ),
                );
            }
        }
    }

    /// checking the generator `path` of a parameter of type `ty`. A
    /// generator is a function (or a method without `self`) with data
    /// parameters, returning `ty` or `ty?`, where `None` discards the case.
    /// It can't be a program, test, property, helper, comptime or generic.
    fn check_generator(&mut self, module: ModuleId, path: &PathRef, ty: TyId) {
        let bad = |message: String, label: &str| {
            Diagnostic::error(message)
                .with_code(codes::BAD_DECORATOR)
                .with_primary(path.span, label.to_string())
        };
        let Some(g) = generator(&self.scopes, self.package, module, path) else {
            self.diags.push(
                bad(format!("`{path}` is not a function"), "not a generator")
                    .with_help("a generator is a function whose parameters are drawn, and which returns the value of the parameter"),
            );
            return;
        };
        let data = &self.package.fns[g];
        let special = data.decorators.iter().any(|d| {
            let n = d.name.to_string();
            ProgramKind::DECORATORS.contains(&n.as_str())
                || ["test", "property", "helper"].contains(&n.as_str())
        });
        // a generic template has no signature of its own
        let sig = self.fn_sigs.get(g).cloned();
        let Some(sig) =
            sig.filter(|_| !special && data.generic_params.is_empty() && !data.is_comptime)
        else {
            self.diags.push(bad(
                format!("`{path}` cannot be a generator"),
                "a program, test, property, helper, comptime or generic function",
            ));
            return;
        };
        let ret = sig.ret;
        let optional = matches!(self.types.kind(ret), TyKind::Option(inner) if *inner == ty);
        if ret != ty && !optional && !self.types.is_error(ret) && !self.types.is_error(ty) {
            let (ret, ty) = (self.display(ret), self.display(ty));
            self.diags.push(bad(
                format!("`{path}` does not make values of the parameter"),
                &format!("returns `{ret}`, the parameter is `{ty}`"),
            )
            .with_help("a generator returns the type of the parameter, or an option of it (`None` discards the case)"));
        }
        for (param, (_, t)) in data.params.iter().zip(&sig.params) {
            if param.name.text == "self" {
                self.diags.push(
                    bad(format!("`{path}` cannot be a generator"), "it takes `self`")
                        .with_secondary(param.name.span, "`self` here")
                        .with_help("a generator is a function, or a method without `self`"),
                );
                return;
            }
            if !self.is_data(*t) {
                self.diags.push(
                    bad(
                        format!("the parameters of `{path}` cannot be generated"),
                        &format!("its parameter `{}` is not data", param.name.text),
                    )
                    .with_secondary(param.name.span, "this parameter")
                    .with_note("generated values are integers, booleans, arrays, structs, enums and options of those"),
                );
                return;
            }
        }
    }

    /// is the type data that a property can take (integers, bools, arrays,
    /// structs, enums and options of those)? The error type counts as data,
    /// to avoid reporting twice.
    fn is_data(&self, ty: TyId) -> bool {
        match self.types.kind(ty) {
            TyKind::Int(_) | TyKind::Bool | TyKind::Error => true,
            TyKind::Array(inner, _) | TyKind::Option(inner) => self.is_data(*inner),
            TyKind::Struct(s) if self.struct_contexts.get(*s).is_some() => false,
            TyKind::Struct(s) => self
                .struct_fields
                .get(*s)
                .is_some_and(|fields| fields.iter().all(|&t| self.is_data(t))),
            TyKind::Enum(e) => self.variant_fields.get(*e).is_some_and(|variants| {
                variants
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|&t| self.is_data(t))
            }),
            _ => false,
        }
    }

    /// checking `@requires(e)` and `@ensures(e)`, with one condition each.
    /// Neither goes on a helper, a test or a `comptime` function. Also no
    /// `@requires` on a program, because the kernel calls it, and no
    /// `@ensures` on a property, since that is what it returns.
    fn contract_decorator(&mut self, id: FnId, decorator: &Decorator, name: &str) {
        let data = &self.package.fns[id];
        let has = |n: &str| data.decorators.iter().any(|d| d.name.to_string() == n);
        let bad = |message: &str, label: &str| {
            Diagnostic::error(message.to_string())
                .with_code(codes::BAD_CONTRACT)
                .with_primary(decorator.span, label.to_string())
        };
        if decorator.args.len() != 1 || decorator.args[0].name.is_some() {
            let message = format!("`@{name}` takes one condition");
            self.diags
                .push(bad(&message, &format!("expected `@{name}(condition)`")));
        } else if has("helper") || has("test") || data.is_comptime {
            let what = if has("helper") {
                "helpers"
            } else if has("test") {
                "tests"
            } else {
                "`comptime` functions"
            };
            let message = format!("{what} have no contracts");
            self.diags.push(bad(&message, "remove this contract"));
        } else if name == "ensures" && has("property") {
            self.diags.push(
                bad(
                    "properties have no `@ensures`",
                    "the property is what it returns",
                )
                .with_help("`@requires` on a property filters the values it is given"),
            );
        } else if name == "requires"
            && data
                .decorators
                .iter()
                .any(|d| ProgramKind::DECORATORS.contains(&d.name.to_string().as_str()))
        {
            self.diags.push(
                bad("programs have no `@requires`", "the kernel calls programs")
                    .with_help("state what a program guarantees with `@ensures`"),
            );
        }
    }

    /// the program array and index of `@tail(stages, 2)`: a `ProgArray` map
    /// in scope and an index below its `max_entries`
    fn tail(&mut self, module: ModuleId, decorator: &Decorator) -> Option<(String, u32)> {
        let usage = |db: &mut Self, label: &str| {
            db.diags.push(
                Diagnostic::error("invalid arguments for `@tail`")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(decorator.span, label.to_string())
                    .with_help(
                        "write `@tail(stages, 0)`, with `map stages: ProgArray(max_entries=N)`",
                    ),
            );
            None
        };
        let [map, index] = decorator.args.as_slice() else {
            return usage(self, "expected a program array and an index");
        };
        let map = match &map.value {
            ConstExpr::Path(path) if path.segments.len() == 1 => {
                match self.scopes.lookup(module, &path.segments[0].text) {
                    Some(crate::resolve::Def::Item(Item::Map(m))) => Some(m),
                    _ => None,
                }
            }
            _ => None,
        };
        let Some(map) = map else {
            return usage(self, "the first argument is not a map");
        };
        let info = self.maps.get(map).copied()?;
        if info.kind != MapKind::ProgArray {
            return usage(self, "the map is not a `ProgArray`");
        }
        let index = match self.eval(module, &index.value) {
            Some(Value::Int(n)) if n >= 0 && n < i128::from(info.max_entries) => n as u32,
            _ => {
                let label = format!("the index is not in 0..{}", info.max_entries);
                return usage(self, &label);
            }
        };
        Some((self.package.maps[map].name.text.clone(), index))
    }

    /// checking `@helper(N)`, used only by the standard library, and
    /// recording the helper number
    fn helper(&mut self, id: FnId, decorator: &Decorator) {
        let package = self.package;
        let data = &package.fns[id];
        if !package.modules[data.module].is_std() {
            self.diags.push(std_only(decorator));
            return;
        }
        let number = match decorator.args.as_slice() {
            [arg] => match self.eval(data.module, &arg.value) {
                Some(Value::Int(n)) => u32::try_from(n).ok(),
                _ => None,
            },
            _ => None,
        };
        let Some(number) = number else {
            self.diags.push(
                Diagnostic::error("`@helper` takes the number of the helper")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(decorator.span, "expected `@helper(N)`"),
            );
            return;
        };
        // helpers take and return scalars in registers. A helper returning
        // an array or struct fills a buffer passed with its size instead, as
        // `bpf_get_current_comm(buf, size)`.
        let sig = &self.fn_sigs[id];
        let scalar = |kind: &TyKind| {
            matches!(
                kind,
                TyKind::Int(_) | TyKind::Bool | TyKind::Unit | TyKind::KPtr(_)
            )
        };
        // the other parameters come after the buffer and its size, as in
        // `bpf_probe_read_user_str(buf, size, address)`, so at most three
        // of them fit in the five registers
        let fills_buffer = matches!(
            self.types.kind(sig.ret),
            TyKind::Array(..) | TyKind::Struct(_)
        ) && self.storable(sig.ret)
            && sig.params.len() <= 3;
        let scalars = sig.params.iter().all(|(_, t)| scalar(self.types.kind(*t)))
            && (scalar(self.types.kind(sig.ret)) || fills_buffer);
        if !scalars || sig.params.len() > 5 {
            self.diags.push(
                Diagnostic::error("helpers take at most five integers and return an integer")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(data.name.span, "unsupported helper signature"),
            );
            return;
        }
        let body_is_ellipsis = match data.body.root.as_slice() {
            [stmt] => matches!(
                &data.body.stmts[*stmt],
                Stmt::Expr(e) if matches!(data.body.exprs[*e], Expr::Ellipsis)
            ),
            _ => false,
        };
        if !body_is_ellipsis {
            self.diags.push(
                Diagnostic::error("the body of a helper is `...`")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(data.name.span, "the kernel implements helpers"),
            );
            return;
        }
        self.helpers.insert(id, number);
    }

    /// the value of a string decorator argument
    fn string_arg(&mut self, module: ModuleId, arg: &DecoratorArg) -> Option<String> {
        match self.eval(module, &arg.value) {
            Some(Value::Str(s)) => Some(s),
            _ => None,
        }
    }

    /// checking the decorator and signature of a program, and recording it.
    /// The target must fit its kind (no whitespace, `category/name`,
    /// `binary:symbol`, an identifier for LSM). The program takes its
    /// context and returns an integer or its action enum.
    fn program(&mut self, id: FnId, decorator: &Decorator, name: &str) {
        let package = self.package;
        let data = &package.fns[id];
        let kind = match name {
            "kprobe" => ProgramKind::Kprobe,
            "kretprobe" => ProgramKind::Kretprobe,
            "tracepoint" => ProgramKind::Tracepoint,
            "xdp" => ProgramKind::Xdp,
            "uprobe" => ProgramKind::Uprobe,
            "uretprobe" => ProgramKind::Uretprobe,
            "lsm" => ProgramKind::Lsm,
            _ => ProgramKind::Tc,
        };
        if data.owner.is_some() {
            self.diags.push(
                Diagnostic::error("methods cannot be eBPF programs")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(decorator.span, "move the program to a top-level function"),
            );
            return;
        }
        let usage = match kind {
            ProgramKind::Tracepoint => "`@tracepoint(\"category/name\")`",
            ProgramKind::Kprobe => "`@kprobe(\"kernel_function\")`",
            ProgramKind::Kretprobe => "`@kretprobe(\"kernel_function\")`",
            ProgramKind::Xdp => "`@xdp` or `@xdp(iface=\"eth0\")`",
            ProgramKind::Tc => "`@tc(direction=\"ingress\", iface=\"eth0\")`",
            ProgramKind::Uprobe => "`@uprobe(\"/path/to/binary:symbol\")`",
            ProgramKind::Uretprobe => "`@uretprobe(\"/path/to/binary:symbol\")`",
            ProgramKind::Lsm => "`@lsm(\"hook\")`, as in `@lsm(\"file_open\")`",
            ProgramKind::Test => "`@test`",
        };
        let bad = |db: &mut Self| {
            db.diags.push(
                Diagnostic::error(format!("invalid arguments for `@{name}`"))
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(decorator.span, format!("expected {usage}")),
            );
        };

        let (target, iface) = match kind {
            ProgramKind::Kprobe
            | ProgramKind::Kretprobe
            | ProgramKind::Tracepoint
            | ProgramKind::Uprobe
            | ProgramKind::Uretprobe
            | ProgramKind::Lsm => {
                let target = match decorator.args.as_slice() {
                    [arg] if arg.name.is_none() => self.string_arg(data.module, arg),
                    _ => None,
                };
                let valid = |t: &String| {
                    !t.is_empty()
                        && !t.contains(char::is_whitespace)
                        && (kind != ProgramKind::Tracepoint
                            || t.split('/').filter(|p| !p.is_empty()).count() == 2)
                        && (!matches!(kind, ProgramKind::Uprobe | ProgramKind::Uretprobe)
                            || t.rsplit_once(':')
                                .is_some_and(|(b, f)| !b.is_empty() && !f.is_empty()))
                        && (kind != ProgramKind::Lsm
                            || t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
                };
                match target.filter(valid) {
                    Some(target) => (target, None),
                    None => return bad(self),
                }
            }
            ProgramKind::Xdp | ProgramKind::Tc => {
                let mut iface = None;
                let mut direction = None;
                for arg in &decorator.args {
                    let value = self.string_arg(data.module, arg);
                    match (arg.name.as_ref().map(|n| n.text.as_str()), value) {
                        (Some("iface"), Some(v)) if !v.is_empty() => iface = Some(v),
                        (Some("direction"), Some(v))
                            if kind == ProgramKind::Tc
                                && matches!(v.as_str(), "ingress" | "egress") =>
                        {
                            direction = Some(v)
                        }
                        _ => return bad(self),
                    }
                }
                let target = match kind {
                    ProgramKind::Tc => direction.unwrap_or_else(|| "ingress".into()),
                    _ => String::new(),
                };
                (target, iface)
            }
            // can't happen, `@test` is checked by `test`
            ProgramKind::Test => (String::new(), None),
        };

        let sig = self.fn_sigs[id].clone();
        let context = kind.context();
        let takes_context = match sig.params.as_slice() {
            [(_, ty)] => {
                matches!(
                    self.types.kind(*ty),
                    TyKind::Struct(s) if self.struct_contexts.get(*s) == Some(&context)
                ) || (context == ContextKind::Tracepoint
                    && matches!(self.types.kind(*ty), TyKind::TpContext(_)))
                    || self.types.is_error(*ty)
            }
            _ => false,
        };
        let span = signature_span(data);
        if !takes_context {
            // the parameter when there is one, the signature otherwise
            let (span, label) = match (sig.params.as_slice(), data.params.first()) {
                ([(_, ty)], Some(param)) => (
                    param
                        .ty
                        .as_ref()
                        .and_then(TypeRef::span)
                        .unwrap_or(param.name.span),
                    format!("found `{}`", self.display(*ty)),
                ),
                ([], _) => (span, "no parameter".to_string()),
                (params, _) => (span, format!("{} parameters", params.len())),
            };
            self.diags.push(
                Diagnostic::error(format!(
                    "{} takes one parameter of type `{}`",
                    kind.noun(),
                    context.struct_name()
                ))
                .with_code(codes::BAD_DECORATOR)
                .with_primary(span, label)
                .with_help(format!(
                    "write `def {}(ctx: {})`, with `from {} import {}`",
                    data.name.text,
                    context.struct_name(),
                    context.module(),
                    context.struct_name()
                )),
            );
            return;
        }
        let returns_action = match (self.types.kind(sig.ret), kind.action_enum()) {
            (TyKind::Enum(e), Some(action)) => package.enums[*e].name.text == action,
            _ => false,
        };
        if !matches!(self.types.kind(sig.ret), TyKind::Int(_) | TyKind::Error) && !returns_action {
            let found = self.display(sig.ret);
            let expected = kind
                .action_enum()
                .map_or("an integer".to_string(), |a| format!("`{a}` or an integer"));
            let example = kind.action_enum().unwrap_or("i32");
            let (span, label) = match data.ret.as_ref().and_then(TypeRef::span) {
                Some(ret) => (ret, format!("found `{found}`")),
                None => (span, "no return type".to_string()),
            };
            self.diags.push(
                Diagnostic::error(format!("{} returns {expected}", kind.noun()))
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(span, label)
                    .with_help(format!("declare the return type, as in `-> {example}`")),
            );
            return;
        }
        self.programs.insert(
            id,
            ProgramInfo {
                kind,
                target,
                iface,
                tail: None,
            },
        );
    }
}

/// the span of `name(params) -> ret`, for errors about a signature
fn signature_span(data: &FnData) -> Span {
    let end = data
        .ret
        .as_ref()
        .and_then(TypeRef::span)
        .or_else(|| data.params.last().map(|p| p.name.span))
        .unwrap_or(data.name.span);
    data.name.span.to(end)
}

/// the function named by a generator argument of `@property` (`x=g` or
/// `x=Type.method`) as seen from `module`. Methods are looked up in the
/// template, not in its instances.
pub fn generator(
    scopes: &crate::resolve::Scopes,
    package: &Package,
    module: ModuleId,
    path: &PathRef,
) -> Option<FnId> {
    use crate::resolve::Def;
    let (def, used) = scopes.resolve_prefix(package, module, &path.segments)?;
    let rest = &path.segments[used..];
    match (def, rest) {
        (Def::Item(Item::Fn(f)), []) => Some(f),
        (Def::Item(Item::Struct(s)), [method]) => package.fns.iter().find_map(|(f, data)| {
            (data.owner == Some(Owner::Struct(s))
                && data.instance.is_none()
                && data.name.text == method.text)
                .then_some(f)
        }),
        (Def::Item(Item::Enum(e)), [method]) => package.fns.iter().find_map(|(f, data)| {
            (data.owner == Some(Owner::Enum(e))
                && data.instance.is_none()
                && data.name.text == method.text)
                .then_some(f)
        }),
        _ => None,
    }
}

/// the generator of each parameter of a property, `None` when the values
/// come from the parameter type
pub fn property_generators(
    scopes: &crate::resolve::Scopes,
    package: &Package,
    f: FnId,
) -> Vec<Option<FnId>> {
    let data = &package.fns[f];
    data.params
        .iter()
        .map(|param| {
            data.decorators
                .iter()
                .filter(|d| d.name.to_string() == "property")
                .flat_map(|d| &d.args)
                .find_map(|arg| match (&arg.name, &arg.value) {
                    (Some(name), ConstExpr::Path(path)) if name.text == param.name.text => {
                        generator(scopes, package, data.module, path)
                    }
                    _ => None,
                })
        })
        .collect()
}
