//! this module builds the module tree and the scope of each module
//! ([`Scopes::build`]), and looks names up in them ([`Scopes::lookup`],
//! [`Scopes::resolve_prefix`]).
//!
//! Every file is a module named by its path under `src/`, and every directory
//! is a namespace. A module may shadow a builtin of the prelude. Imports are
//! not re-exported, and names starting with `_` are private. The namespace
//! `kernel` holds the structs and unions of the kernel's BTF.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use vigil_btf::Btf;

use la_arena::{Arena, ArenaMap, Idx};
use vigil_diagnostics::{Diagnostic, Diagnostics, Span, codes};

use crate::hir::*;
use crate::ty::IntTy;

/// a namespace, in [`Scopes::namespaces`]
pub type NsId = Idx<Namespace>;

/// a node of the module tree: a directory, a file, or both
#[derive(Debug, Default)]
pub struct Namespace {
    /// the namespaces under this one, by name
    pub children: HashMap<String, NsId>,
    /// the module of the file at this path, if there is one
    pub module: Option<ModuleId>,
    /// whether this is the `kernel` module, whose names are the structs and
    /// unions of the kernel's BTF
    pub kernel: bool,
}

/// what a name refers to
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Def {
    /// a module or directory
    Namespace(NsId),
    /// a definition of the package
    Item(Item),
    /// a name of the prelude
    Builtin(Builtin),
    /// a struct or union of the kernel's BTF, by type id
    Kernel(u32),
    /// a name whose definition failed to resolve. It is already reported, so
    /// its uses report nothing more.
    Error,
}

/// a name of the prelude: a builtin type or function
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin {
    Int(IntTy),
    Bool,
    Str,
    /// `bytes[N]`, an alias of `u8[N]`
    Bytes,
    Option,
    Some,
    Print,
    Range,
    Min,
    Max,
    /// `clamp(x, lo, hi)`, without branches. `min` and `max` branch, which
    /// gives the verifier the range of the result, but each branch doubles
    /// the paths it explores and a clamp would need two.
    Clamp,
    /// `saturating_add(a, b)` and `saturating_sub(a, b)`, without branches:
    /// the result, or the bound of the type it would cross
    SaturatingAdd,
    SaturatingSub,
    /// `kptr[T]`: a pointer to a kernel struct
    KPtr,
    /// `f64` and the functions on it: in comptime code only
    F64,
    /// `stage()`, where a program is split into programs joined by tail calls
    Stage,
    Math(Math),
    /// `is_some(e)`, `is_none(e)` and `value(e)` on an `Option`, only in
    /// contracts, which have no `?` or `match`
    IsSome,
    IsNone,
    Value,
    /// `old(e)` in `@ensures`: `e` as it was on entry, for the `self` of a
    /// `@mut` method
    Old,
    /// `run(program, packet)`: the verdict of an XDP or TC program on a
    /// packet of `std.testing`. In tests only, run by the simulator of the
    /// full Vigil.
    Run,
    /// `set_time(ns)`: what `bpf_ktime_get_ns` returns from then on in the
    /// simulator. In tests only.
    SetTime,
}

/// a function on `f64` of the prelude, for comptime code
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Math {
    Exp,
    Log,
    Sqrt,
    Tanh,
    Floor,
    Round,
}

impl Builtin {
    /// the builtin a name of the prelude stands for
    fn by_name(name: &str) -> Option<Builtin> {
        if let Some(int) = IntTy::ALL.into_iter().find(|i| i.name() == name) {
            return Some(Builtin::Int(int));
        }
        Some(match name {
            "bool" => Builtin::Bool,
            "str" => Builtin::Str,
            "bytes" => Builtin::Bytes,
            "Option" => Builtin::Option,
            "Some" => Builtin::Some,
            "print" => Builtin::Print,
            "range" => Builtin::Range,
            "min" => Builtin::Min,
            "max" => Builtin::Max,
            "clamp" => Builtin::Clamp,
            "saturating_add" => Builtin::SaturatingAdd,
            "saturating_sub" => Builtin::SaturatingSub,
            "kptr" => Builtin::KPtr,
            "f64" => Builtin::F64,
            "stage" => Builtin::Stage,
            "is_some" => Builtin::IsSome,
            "is_none" => Builtin::IsNone,
            "value" => Builtin::Value,
            "old" => Builtin::Old,
            "run" => Builtin::Run,
            "set_time" => Builtin::SetTime,
            "exp" => Builtin::Math(Math::Exp),
            "log" => Builtin::Math(Math::Log),
            "sqrt" => Builtin::Math(Math::Sqrt),
            "tanh" => Builtin::Math(Math::Tanh),
            "floor" => Builtin::Math(Math::Floor),
            "round" => Builtin::Math(Math::Round),
            _ => return None,
        })
    }
}

/// the module tree, and what each module can name
#[derive(Debug)]
pub struct Scopes {
    /// every namespace
    pub namespaces: Arena<Namespace>,
    /// the namespace of `src/`
    pub root: NsId,
    /// the kernel's types, for the names of `kernel`
    pub btf: Option<Arc<Btf>>,
    /// the names visible in each module (without the prelude), with the span
    /// that defines them
    pub modules: ArenaMap<ModuleId, HashMap<String, (Def, Span)>>,
    /// the private names reached through a path (`ports._x`), one error per
    /// place. Here we collect them because [`Scopes::resolve_prefix`] has
    /// many callers and no diagnostics, see [`Scopes::take_private_uses`]
    private_uses: Mutex<Vec<Diagnostic>>,
}

impl Scopes {
    /// building the scopes of a package. Duplicate definitions and bad
    /// imports are reported here. The first definition of a name wins, and a
    /// bad import gives [`Def::Error`].
    pub fn build(package: &Package, btf: Option<Arc<Btf>>, diags: &mut Diagnostics) -> Scopes {
        let mut namespaces = Arena::new();
        let root = namespaces.alloc(Namespace::default());
        let mut scopes = Scopes {
            namespaces,
            root,
            btf,
            modules: ArenaMap::default(),
            private_uses: Mutex::default(),
        };
        for (id, module) in package.modules.iter() {
            let ns = scopes.namespace_path(&module.path);
            scopes.namespaces[ns].module = Some(id);
        }
        // `kernel`, unless the project has a module of that name, which hides
        // the kernel's types
        if !scopes.namespaces[root].children.contains_key("kernel") {
            let kernel = scopes.namespaces.alloc(Namespace {
                kernel: true,
                ..Namespace::default()
            });
            scopes.namespaces[root]
                .children
                .insert("kernel".into(), kernel);
        }
        for (id, module) in package.modules.iter() {
            let scope = scopes.module_scope(package, id, module, diags);
            scopes.modules.insert(id, scope);
        }
        scopes
    }

    /// the namespace at a path, made with its parents if missing
    fn namespace_path(&mut self, path: &[String]) -> NsId {
        let mut ns = self.root;
        for segment in path {
            ns = match self.namespaces[ns].children.get(segment) {
                Some(&child) => child,
                None => {
                    let child = self.namespaces.alloc(Namespace::default());
                    self.namespaces[ns].children.insert(segment.clone(), child);
                    child
                }
            };
        }
        ns
    }

    /// the names a module defines and imports. The first definition of a
    /// name wins, any other is reported.
    fn module_scope(
        &self,
        package: &Package,
        id: ModuleId,
        module: &ModuleData,
        diags: &mut Diagnostics,
    ) -> HashMap<String, (Def, Span)> {
        let mut scope: HashMap<String, (Def, Span)> = HashMap::new();
        let mut define = |name: &Ident, def: Def, diags: &mut Diagnostics| {
            if name.text.is_empty() {
                return;
            }
            if let Some(&(_, kept)) = scope.get(&name.text) {
                // items are defined before imports, so here we label the two
                // definitions in the order of the file
                let (first, again) = if name.span.start < kept.start {
                    (name.span, kept)
                } else {
                    (kept, name.span)
                };
                diags.push(
                    Diagnostic::error(format!("`{}` is defined twice", name.text))
                        .with_code(codes::DUPLICATE_DEFINITION)
                        .with_primary(again, "redefined here")
                        .with_secondary(first, "first defined here"),
                );
                return;
            }
            scope.insert(name.text.clone(), (def, name.span));
        };

        for &item in &module.items {
            define(item_name(package, item), Def::Item(item), diags);
        }
        for name in &module.unsupported {
            define(name, Def::Error, diags);
        }
        for import in &module.imports {
            match import {
                Import::From {
                    module: path,
                    names,
                } => {
                    let ns = self.resolve_module(path, diags);
                    for (name, alias) in names {
                        let def = match ns {
                            Some(ns) => self.import_member(package, id, ns, path, name, diags),
                            None => Def::Error,
                        };
                        define(alias.as_ref().unwrap_or(name), def, diags);
                    }
                }
                Import::Module { path, alias } => {
                    let def = self
                        .resolve_module(path, diags)
                        .map_or(Def::Error, Def::Namespace);
                    if let Some(name) = alias.as_ref().or(path.segments.last()) {
                        define(name, def, diags);
                    }
                }
            }
        }
        scope
    }

    /// the namespace at an absolute module path, or `None` (reported) if there
    /// is none
    fn resolve_module(&self, path: &PathRef, diags: &mut Diagnostics) -> Option<NsId> {
        let mut ns = self.root;
        for (i, segment) in path.segments.iter().enumerate() {
            match self.namespaces[ns].children.get(&segment.text) {
                Some(&child) => ns = child,
                None => {
                    let prefix: Vec<_> = path.segments[..=i]
                        .iter()
                        .map(|s| s.text.as_str())
                        .collect();
                    let mut diag =
                        Diagnostic::error(format!("unresolved module `{}`", prefix.join(".")))
                            .with_code(codes::UNRESOLVED_MODULE)
                            .with_primary(segment.span, "no such module");
                    // we list the modules of the analyzed library, which may
                    // not be the built-in one, because the command line can
                    // load another
                    if path.segments[0].text == "std" && i > 0 {
                        let mut names: Vec<_> = self.namespaces[ns]
                            .children
                            .keys()
                            .map(|name| format!("`{}.{name}`", prefix[..i].join(".")))
                            .collect();
                        names.sort();
                        let note = if names.is_empty() {
                            format!("`{}` has no modules", prefix[..i].join("."))
                        } else {
                            format!("`{}` has {}", prefix[..i].join("."), names.join(", "))
                        };
                        diag = diag.with_note(note);
                    }
                    diags.push(diag);
                    return None;
                }
            }
        }
        Some(ns)
    }

    /// what `from path import name` imports. A missing or private name is
    /// reported
    fn import_member(
        &self,
        package: &Package,
        importer: ModuleId,
        ns: NsId,
        path: &PathRef,
        name: &Ident,
        diags: &mut Diagnostics,
    ) -> Def {
        if self.namespaces[ns].kernel {
            return self.kernel_member(name, diags);
        }
        match self.member(package, ns, &name.text) {
            Some(Def::Item(item)) => {
                let module = item_module(package, item);
                if name.text.starts_with('_') && module != importer {
                    diags.push(
                        Diagnostic::error(format!("`{}` is private", name.text))
                            .with_code(codes::PRIVATE_ITEM)
                            .with_primary(name.span, "private to its module")
                            .with_note("names starting with `_` cannot be imported"),
                    );
                    return Def::Error;
                }
                Def::Item(item)
            }
            Some(def) => def,
            None => {
                diags.push(
                    Diagnostic::error(format!("`{}` not found in module `{path}`", name.text))
                        .with_code(codes::UNRESOLVED_NAME)
                        .with_primary(name.span, "not found"),
                );
                Def::Error
            }
        }
    }

    /// the kernel struct or union that `from kernel import name` imports
    fn kernel_member(&self, name: &Ident, diags: &mut Diagnostics) -> Def {
        let Some(btf) = &self.btf else {
            diags.push(
                Diagnostic::error("kernel types need the kernel's BTF")
                    .with_code(codes::UNRESOLVED_MODULE)
                    .with_primary(name.span, "cannot import kernel types")
                    .with_note(format!(
                        "Vigil reads `{}`, or the file named by `btf` in the `[target]` table of vigil.toml",
                        vigil_btf::VMLINUX
                    ))
                    .with_help(
                        "kernels built with CONFIG_DEBUG_INFO_BTF have it; for others, extract it from \
                         their debug info (`pahole -J`) or take it from BTFHub, and name the file with `btf`",
                    ),
            );
            return Def::Error;
        };
        match btf.find_composite(&name.text) {
            Some(id) => Def::Kernel(id),
            None => {
                diags.push(
                    Diagnostic::error(format!(
                        "`{}` is not a struct or union of the kernel",
                        name.text
                    ))
                    .with_code(codes::UNRESOLVED_NAME)
                    .with_primary(name.span, "not found in the kernel's BTF"),
                );
                Def::Error
            }
        }
    }

    /// what a name inside a namespace refers to: an item of its module, or
    /// else a child namespace. Imports are not re-exported.
    pub fn member(&self, package: &Package, ns: NsId, name: &str) -> Option<Def> {
        let namespace = &self.namespaces[ns];
        if namespace.kernel {
            return self
                .btf
                .as_ref()
                .and_then(|btf| btf.find_composite(name))
                .map(Def::Kernel);
        }
        if let Some(module) = namespace.module {
            let item = package.modules[module]
                .items
                .iter()
                .find(|&&item| item_name(package, item).text == name);
            if let Some(&item) = item {
                return Some(Def::Item(item));
            }
        }
        namespace
            .children
            .get(name)
            .map(|&child| Def::Namespace(child))
    }

    /// what a name refers to in a module, first in its scope, then in the
    /// prelude
    pub fn lookup(&self, module: ModuleId, name: &str) -> Option<Def> {
        if let Some((def, _)) = self.modules.get(module).and_then(|s| s.get(name)) {
            return Some(*def);
        }
        Builtin::by_name(name).map(Def::Builtin)
    }

    /// the definition named by the longest prefix of `segments`, with the
    /// length of that prefix. The rest (a variant, a method, a field) is for
    /// the caller.
    pub fn resolve_prefix(
        &self,
        package: &Package,
        module: ModuleId,
        segments: &[Ident],
    ) -> Option<(Def, usize)> {
        let first = segments.first()?;
        let mut def = self.lookup(module, &first.text)?;
        let mut used = 1;
        while let (Def::Namespace(ns), Some(next)) = (def, segments.get(used)) {
            match self.member(package, ns, &next.text) {
                Some(Def::Item(item))
                    if next.text.starts_with('_') && item_module(package, item) != module =>
                {
                    self.private_use(next);
                    def = Def::Error;
                    used += 1;
                }
                Some(member) => {
                    def = member;
                    used += 1;
                }
                None => break,
            }
        }
        Some((def, used))
    }
}

impl Scopes {
    /// recording the error for a private name reached from another module
    fn private_use(&self, name: &Ident) {
        let mut uses = self.private_uses.lock().unwrap();
        if uses.iter().any(|d| d.anchor() == Some(name.span)) {
            return;
        }
        uses.push(
            Diagnostic::error(format!("`{}` is private", name.text))
                .with_code(codes::PRIVATE_ITEM)
                .with_primary(name.span, "private to its module")
                .with_note("names starting with `_` can only be used in their module"),
        );
    }

    /// the errors for the private names reached so far, emptying the list
    pub fn take_private_uses(&self) -> Vec<Diagnostic> {
        std::mem::take(&mut *self.private_uses.lock().unwrap())
    }
}

/// the name of an item, as defined
pub fn item_name(package: &Package, item: Item) -> &Ident {
    match item {
        Item::Fn(id) => &package.fns[id].name,
        Item::Struct(id) => &package.structs[id].name,
        Item::Enum(id) => &package.enums[id].name,
        Item::Const(id) => &package.consts[id].name,
        Item::Alias(id) => &package.aliases[id].name,
        Item::Map(id) => &package.maps[id].name,
        Item::Protocol(id) => &package.protocols[id].name,
    }
}

/// the module that defines an item
fn item_module(package: &Package, item: Item) -> ModuleId {
    match item {
        Item::Fn(id) => package.fns[id].module,
        Item::Struct(id) => package.structs[id].module,
        Item::Enum(id) => package.enums[id].module,
        Item::Const(id) => package.consts[id].module,
        Item::Alias(id) => package.aliases[id].module,
        Item::Map(id) => package.maps[id].module,
        Item::Protocol(id) => package.protocols[id].module,
    }
}
