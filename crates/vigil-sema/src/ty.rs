//! this module defines the interned types. Interning is structural, so
//! once inference variables are resolved two types are equal when their
//! [`TyId`]s are. A `TyId` only means something in the table of one
//! analysis, so generic instances, which outlive it, use a [`TyKey`]

use std::collections::HashMap;
use std::fmt;

use crate::hir::{EnumId, MapId, Package, StructId};

/// an interned type, an index into a [`Types`] table
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TyId(u32);

/// integer types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntTy {
    U8,
    U16,
    U32,
    U64,
    Usize,
    I8,
    I16,
    I32,
    I64,
}

impl IntTy {
    /// all integer types, unsigned first
    pub const ALL: [IntTy; 9] = [
        IntTy::U8,
        IntTy::U16,
        IntTy::U32,
        IntTy::U64,
        IntTy::Usize,
        IntTy::I8,
        IntTy::I16,
        IntTy::I32,
        IntTy::I64,
    ];

    /// the name of the type in source code
    pub fn name(self) -> &'static str {
        match self {
            IntTy::U8 => "u8",
            IntTy::U16 => "u16",
            IntTy::U32 => "u32",
            IntTy::U64 => "u64",
            IntTy::Usize => "usize",
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
        }
    }

    /// the width of the type, in bits
    pub fn bits(self) -> u32 {
        match self {
            IntTy::U8 | IntTy::I8 => 8,
            IntTy::U16 | IntTy::I16 => 16,
            IntTy::U32 | IntTy::I32 => 32,
            // `usize` is as wide as an eBPF register, 64 bits
            IntTy::U64 | IntTy::Usize | IntTy::I64 => 64,
        }
    }

    pub fn signed(self) -> bool {
        matches!(self, IntTy::I8 | IntTy::I16 | IntTy::I32 | IntTy::I64)
    }

    /// the smallest value of the type. We use `i128` because it holds every
    /// bound, `u64::MAX` included
    pub fn min(self) -> i128 {
        if self.signed() {
            -(1i128 << (self.bits() - 1))
        } else {
            0
        }
    }

    /// the largest value of the type
    pub fn max(self) -> i128 {
        if self.signed() {
            (1i128 << (self.bits() - 1)) - 1
        } else {
            (1i128 << self.bits()) - 1
        }
    }

    /// checking if a value fits in the type
    pub fn contains(self, value: i128) -> bool {
        (self.min()..=self.max()).contains(&value)
    }
}

/// a type that does not depend on an interner, used for the type arguments
/// of generic instances, which are requested by one analysis and checked by
/// the next. Only generic arguments have a key (see [`Types::key`]), and a
/// kernel pointer is keyed by the BTF id of its struct
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TyKey {
    Int(IntTy),
    Bool,
    Struct(StructId),
    Enum(EnumId),
    Option(Box<TyKey>),
    Array(Box<TyKey>, u64),
    KPtr(u32),
    Float,
}

/// the structure of a type
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TyKind {
    Int(IntTy),
    Bool,
    /// string literals and format strings, only arguments of `print`
    Str,
    /// the return type of a function without one
    Unit,
    Struct(StructId),
    Enum(EnumId),
    Option(TyId),
    Array(TyId, u64),
    /// a reference to a value stored in a map
    Ref(TyId),
    /// the type of the name of a map, used only through its methods
    Map(MapId),
    /// a kernel struct or union, by BTF type id. Only reached through a
    /// `kptr`
    Kernel(u32),
    /// `kptr[T]`, the address of a kernel struct, read field by field
    KPtr(TyId),
    /// `TracepointContext[T]`, the context of a tracepoint, typed by the
    /// kernel struct of its event (BTF type id)
    TpContext(u32),
    /// `f64`, only in `comptime` code, which runs in the compiler
    Float,
    /// an inference variable, local to the body being checked
    Var(u32),
    /// the type of an erroneous expression. It is compatible with
    /// everything, so we report an error once and not at every use
    Error,
}

/// the interner of an analysis, with the common types at hand
pub struct Types {
    kinds: Vec<TyKind>,
    ids: HashMap<TyKind, TyId>,
    /// names of kernel types, for messages
    kernel_names: HashMap<u32, String>,
    pub unit: TyId,
    pub bool: TyId,
    pub str: TyId,
    pub error: TyId,
}

impl Default for Types {
    fn default() -> Self {
        let mut types = Types {
            kinds: Vec::new(),
            ids: HashMap::new(),
            kernel_names: HashMap::new(),
            unit: TyId(0),
            bool: TyId(0),
            str: TyId(0),
            error: TyId(0),
        };
        types.unit = types.intern(TyKind::Unit);
        types.bool = types.intern(TyKind::Bool);
        types.str = types.intern(TyKind::Str);
        types.error = types.intern(TyKind::Error);
        for int in IntTy::ALL {
            types.intern(TyKind::Int(int));
        }
        types
    }
}

impl fmt::Debug for Types {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Types({} interned)", self.kinds.len())
    }
}

impl Types {
    /// the id of a type, interning it on first use
    pub fn intern(&mut self, kind: TyKind) -> TyId {
        if let Some(&id) = self.ids.get(&kind) {
            return id;
        }
        let id = TyId(u32::try_from(self.kinds.len()).expect("too many types"));
        self.kinds.push(kind.clone());
        self.ids.insert(kind, id);
        id
    }

    /// recording the name of a kernel type for messages, because
    /// [`TyKind::Kernel`] only holds the id
    pub fn name_kernel(&mut self, id: u32, name: String) {
        self.kernel_names.insert(id, name);
    }

    /// the id of a type if it is already interned. Unlike
    /// [`Types::intern`], it needs no `&mut`
    pub fn lookup(&self, kind: &TyKind) -> Option<TyId> {
        self.ids.get(kind).copied()
    }

    /// the kind of an id
    pub fn kind(&self, id: TyId) -> &TyKind {
        &self.kinds[id.0 as usize]
    }

    pub fn int(&mut self, int: IntTy) -> TyId {
        self.intern(TyKind::Int(int))
    }

    pub fn option(&mut self, inner: TyId) -> TyId {
        self.intern(TyKind::Option(inner))
    }

    pub fn array(&mut self, element: TyId, len: u64) -> TyId {
        self.intern(TyKind::Array(element, len))
    }

    pub fn float(&mut self) -> TyId {
        self.intern(TyKind::Float)
    }

    /// checking if the type holds an `f64`, at any depth
    pub fn has_float(&self, id: TyId) -> bool {
        match self.kind(id) {
            TyKind::Float => true,
            TyKind::Option(inner) | TyKind::Array(inner, _) | TyKind::Ref(inner) => {
                self.has_float(*inner)
            }
            _ => false,
        }
    }

    /// checking if the type is [`TyKind::Error`]
    pub fn is_error(&self, id: TyId) -> bool {
        matches!(self.kind(id), TyKind::Error)
    }

    /// showing a type in a message. `var` names the unresolved inference
    /// variables, and the ones it does not name are shown as `_`
    pub fn display(
        &self,
        package: &Package,
        id: TyId,
        var: &dyn Fn(u32) -> Option<String>,
    ) -> String {
        match self.kind(id) {
            TyKind::Int(int) => int.name().to_string(),
            TyKind::Bool => "bool".to_string(),
            TyKind::Float => "f64".to_string(),
            TyKind::Str => "str".to_string(),
            TyKind::Unit => "()".to_string(),
            TyKind::Struct(s) => package.structs[*s].name.text.clone(),
            TyKind::Enum(e) => package.enums[*e].name.text.clone(),
            TyKind::Option(inner) => format!("Option[{}]", self.display(package, *inner, var)),
            TyKind::Array(element, len) => {
                format!("{}[{len}]", self.display(package, *element, var))
            }
            TyKind::Ref(inner) => format!("ref[{}]", self.display(package, *inner, var)),
            TyKind::Kernel(id) => self
                .kernel_names
                .get(id)
                .cloned()
                .unwrap_or_else(|| "<kernel type>".into()),
            TyKind::KPtr(inner) => format!("kptr[{}]", self.display(package, *inner, var)),
            TyKind::TpContext(id) => format!(
                "TracepointContext[{}]",
                self.kernel_names
                    .get(id)
                    .map_or("<kernel type>", String::as_str)
            ),
            TyKind::Map(m) => format!("map `{}`", package.maps[*m].name.text),
            TyKind::Var(v) => var(*v).unwrap_or_else(|| "_".to_string()),
            TyKind::Error => "{unknown}".to_string(),
        }
    }
}

impl Types {
    /// the key of a type, if it can be a generic argument: an integer,
    /// `bool`, `f64`, a struct, an enum, an option or array of these, or a
    /// pointer to a kernel struct
    pub fn key(&self, id: TyId) -> Option<TyKey> {
        Some(match self.kind(id) {
            TyKind::Int(int) => TyKey::Int(*int),
            TyKind::Bool => TyKey::Bool,
            TyKind::Float => TyKey::Float,
            TyKind::Struct(s) => TyKey::Struct(*s),
            TyKind::Enum(e) => TyKey::Enum(*e),
            TyKind::Option(inner) => TyKey::Option(Box::new(self.key(*inner)?)),
            TyKind::Array(inner, len) => TyKey::Array(Box::new(self.key(*inner)?), *len),
            TyKind::KPtr(inner) => match self.kind(*inner) {
                TyKind::Kernel(k) => TyKey::KPtr(*k),
                _ => return None,
            },
            _ => return None,
        })
    }

    /// the type of a key, interned in this table
    pub fn from_key(&mut self, key: &TyKey) -> TyId {
        match key {
            TyKey::Int(int) => self.int(*int),
            TyKey::Bool => self.bool,
            TyKey::Float => self.intern(TyKind::Float),
            TyKey::Struct(s) => self.intern(TyKind::Struct(*s)),
            TyKey::Enum(e) => self.intern(TyKind::Enum(*e)),
            TyKey::Option(inner) => {
                let inner = self.from_key(inner);
                self.option(inner)
            }
            TyKey::Array(inner, len) => {
                let inner = self.from_key(inner);
                self.array(inner, *len)
            }
            TyKey::KPtr(k) => {
                let kernel = self.intern(TyKind::Kernel(*k));
                self.intern(TyKind::KPtr(kernel))
            }
        }
    }
}

/// showing a type key in the name of an instance
pub fn display_key(package: &Package, key: &TyKey) -> String {
    match key {
        TyKey::Int(int) => int.name().to_string(),
        TyKey::Bool => "bool".into(),
        TyKey::Float => "f64".into(),
        TyKey::Struct(s) => package.structs[*s].name.text.clone(),
        TyKey::Enum(e) => package.enums[*e].name.text.clone(),
        TyKey::Option(inner) => format!("Option[{}]", display_key(package, inner)),
        TyKey::Array(inner, len) => format!("{}[{len}]", display_key(package, inner)),
        TyKey::KPtr(_) => "kptr".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_structural() {
        let mut types = Types::default();
        let u8 = types.int(IntTy::U8);
        let a = types.option(u8);
        let b = types.option(u8);
        assert_eq!(a, b);
        assert_ne!(types.array(u8, 4), types.array(u8, 5));
    }

    #[test]
    fn integer_ranges() {
        assert_eq!(IntTy::U8.max(), 255);
        assert_eq!(IntTy::I8.min(), -128);
        assert_eq!(IntTy::U64.max(), u64::MAX as i128);
        assert!(IntTy::I64.contains(-1));
        assert!(!IntTy::U32.contains(-1));
    }
}
