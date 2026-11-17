//! this module checks the declarations of maps and event streams, and
//! defines their methods.
//!
//! ```text
//! map counts: Hash[u32, u64](max_entries=1024)     # also LruHash, PerCpuHash
//! map slots: Array[Counters, 4]                     # also PerCpuArray; keys are u32
//! event alerts: RingBuf[Alert](size=256 * 1024)
//! map stages: ProgArray(max_entries=4)              # programs, for tail calls
//! ```
//!
//! A lookup returns a `ref[V]` into the memory of the map. Keys and values
//! are built on the stack (512 bytes), so they are limited to
//! `MAX_ENTRY_SIZE`, except array values, which are used in place.

use vigil_diagnostics::{Diagnostic, codes};
use vigil_target::Feature;

use crate::db::{Db, Value};
use crate::hir::*;
use crate::ty::{IntTy, TyId, TyKind};

/// the kind of a map, as its declaration names it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapKind {
    Hash,
    /// a hash map that evicts the least recently used entry when full
    LruHash,
    /// a hash map with one value per CPU
    PerCpuHash,
    /// an array, indexed by `u32`, whose entries all exist from the start
    Array,
    /// an array with one value per CPU
    PerCpuArray,
    /// a ring buffer of events, declared with `event`
    RingBuf,
    /// programs to jump to with `tail_call`
    ProgArray,
}

impl MapKind {
    /// every kind, in the order the error for an unknown one lists them
    pub const ALL: [MapKind; 7] = [
        MapKind::Hash,
        MapKind::LruHash,
        MapKind::PerCpuHash,
        MapKind::Array,
        MapKind::PerCpuArray,
        MapKind::RingBuf,
        MapKind::ProgArray,
    ];

    /// the name of the kind, as written in a declaration
    pub fn name(self) -> &'static str {
        match self {
            MapKind::Hash => "Hash",
            MapKind::LruHash => "LruHash",
            MapKind::PerCpuHash => "PerCpuHash",
            MapKind::Array => "Array",
            MapKind::PerCpuArray => "PerCpuArray",
            MapKind::RingBuf => "RingBuf",
            MapKind::ProgArray => "ProgArray",
        }
    }

    /// the kernel's number for the kind: `BPF_MAP_TYPE_*`
    pub fn bpf_type(self) -> u32 {
        match self {
            MapKind::Hash => 1,
            MapKind::Array => 2,
            MapKind::ProgArray => 3,
            MapKind::PerCpuHash => 5,
            MapKind::PerCpuArray => 6,
            MapKind::LruHash => 9,
            MapKind::RingBuf => 27,
        }
    }

    /// whether the map is an array, `Array` or `PerCpuArray`
    pub fn is_array(self) -> bool {
        matches!(self, MapKind::Array | MapKind::PerCpuArray)
    }

    /// whether the map is a hash map, of any of the three kinds
    pub fn is_hash(self) -> bool {
        matches!(self, MapKind::Hash | MapKind::LruHash | MapKind::PerCpuHash)
    }

    /// whether the map holds one value per CPU
    pub fn is_per_cpu(self) -> bool {
        matches!(self, MapKind::PerCpuHash | MapKind::PerCpuArray)
    }
}

/// what a checked map declaration says: the kind, the types, and the size
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapInfo {
    pub kind: MapKind,
    /// the key type: `u32` for arrays and program arrays, `()` for ring
    /// buffers
    pub key: TyId,
    /// the value type: the events for a ring buffer, `u32` for a program
    /// array
    pub value: TyId,
    /// the number of entries, or the size in bytes of a ring buffer
    pub max_entries: u32,
}

/// the methods of maps, resolved by the type checker
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapMethod {
    /// `get(key) -> ref[V]?`, and `map[index]` for an array
    Get,
    /// `get_or_insert(key, default) -> ref[V]`, aborts if the map is full
    GetOrInsert,
    /// `insert(key, value) -> bool`: true on success
    Insert,
    /// `remove(key) -> bool`: true if the key was present
    Remove,
    /// `contains(key) -> bool`
    Contains,
    /// `emit(event) -> bool`, on a ring buffer: false if it is full
    Emit,
    /// `tail_call(ctx, index)` on a program array: jumps to the program at
    /// `index` without return, or goes on if there is none
    TailCall,
    /// `count() -> u64` on a ring buffer, in tests only: the events emitted
    /// so far in the simulator of the full Vigil
    Count,
    /// `last() -> V?` on a ring buffer, in tests only: the last event emitted
    /// in the simulator of the full Vigil
    Last,
}

impl MapMethod {
    /// the method of that name, if maps have one
    pub fn by_name(name: &str) -> Option<MapMethod> {
        Some(match name {
            "get" => MapMethod::Get,
            "get_or_insert" => MapMethod::GetOrInsert,
            "insert" => MapMethod::Insert,
            "remove" => MapMethod::Remove,
            "contains" => MapMethod::Contains,
            "emit" => MapMethod::Emit,
            "tail_call" => MapMethod::TailCall,
            "count" => MapMethod::Count,
            "last" => MapMethod::Last,
            _ => return None,
        })
    }

    /// whether maps of the kind have the method
    pub fn applies_to(self, kind: MapKind) -> bool {
        match self {
            MapMethod::Emit | MapMethod::Count | MapMethod::Last => kind == MapKind::RingBuf,
            MapMethod::TailCall => kind == MapKind::ProgArray,
            MapMethod::GetOrInsert | MapMethod::Remove => kind.is_hash(),
            MapMethod::Get | MapMethod::Insert | MapMethod::Contains => {
                !matches!(kind, MapKind::RingBuf | MapKind::ProgArray)
            }
        }
    }

    /// whether only tests may call the method, because it reads the
    /// simulator of the full Vigil
    pub fn test_only(self) -> bool {
        matches!(self, MapMethod::Count | MapMethod::Last)
    }

    /// the names of the methods maps of the kind have, for messages. Without
    /// `tests`, the methods only tests may call are left out
    pub fn names_for(kind: MapKind, tests: bool) -> Vec<&'static str> {
        [
            "get",
            "get_or_insert",
            "insert",
            "remove",
            "contains",
            "emit",
            "tail_call",
            "count",
            "last",
        ]
        .into_iter()
        .filter(|n| {
            MapMethod::by_name(n).is_some_and(|m| m.applies_to(kind) && (tests || !m.test_only()))
        })
        .collect()
    }
}

/// the largest key or value of a map, well below the stack size because
/// they are built on it
const MAX_ENTRY_SIZE: u64 = 256;
/// the largest value of an array, used in place. The kernel allocates
/// per-CPU values in units of 32 KiB
const MAX_ARRAY_VALUE_SIZE: u64 = 32 * 1024;

impl Db<'_> {
    /// checking the map declarations of the package. Each valid one goes in
    /// [`Db::maps`], an invalid one is reported and left out.
    pub(crate) fn collect_maps(&mut self) {
        let package = self.package;
        for (id, _) in package.maps.iter() {
            if let Some(info) = self.map_info(id) {
                self.maps.insert(id, info);
            }
        }
    }

    /// no map, after reporting why
    fn map_error(&mut self, diagnostic: Diagnostic) -> Option<MapInfo> {
        self.diags.push(diagnostic);
        None
    }

    /// checking a map declaration: the kind (with `event` only for ring
    /// buffers), its arguments and size, and the key and value types. The
    /// first error is reported and gives `None`.
    fn map_info(&mut self, id: MapId) -> Option<MapInfo> {
        let package = self.package;
        let data = &package.maps[id];
        // the declaration of a map of the kind, for the labels
        let usage = |kind: MapKind| match kind {
            MapKind::RingBuf => "`event name: RingBuf[T](size=N)`".to_string(),
            MapKind::ProgArray => "`map name: ProgArray(max_entries=N)`".to_string(),
            k if k.is_array() => format!("`map name: {}[V, N]`", k.name()),
            k => format!("`map name: {}[K, V](max_entries=N)`", k.name()),
        };
        let kind_name = data.kind.to_string();
        let Some(kind) = MapKind::ALL.into_iter().find(|k| k.name() == kind_name) else {
            let kinds: Vec<_> = MapKind::ALL
                .iter()
                .map(|k| format!("`{}`", k.name()))
                .collect();
            return self.map_error(
                Diagnostic::error(format!("unknown map kind `{kind_name}`"))
                    .with_code(codes::WRONG_KIND)
                    .with_primary(data.kind.span, "not a map kind")
                    .with_note(format!("the kinds are {}", kinds.join(", "))),
            );
        };
        if (kind == MapKind::RingBuf) != data.is_event {
            let message = if data.is_event {
                "events are declared with `RingBuf`"
            } else {
                "ring buffers are declared with `event`, not `map`"
            };
            return self.map_error(
                Diagnostic::error(message)
                    .with_code(codes::WRONG_KIND)
                    .with_primary(
                        data.kind.span,
                        format!("expected {}", usage(MapKind::RingBuf)),
                    ),
            );
        }
        if kind == MapKind::RingBuf && self.options.kernel < Feature::RingBuf.since() {
            return self.map_error(
                Diagnostic::error(format!("ring buffers need {}", Feature::RingBuf.since()))
                    .with_code(codes::UNSUPPORTED_KERNEL)
                    .with_primary(data.kind.span, "not available in the target kernels")
                    .with_note(format!(
                        "the oldest kernel of the project is {}",
                        self.options.kernel
                    ))
                    .with_help("raise the oldest kernel in `[target] kernels` of vigil.toml"),
            );
        }

        let bad_args = |db: &mut Self, expected: &str| {
            db.map_error(
                Diagnostic::error(format!("`{}` takes {expected}", kind.name()))
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(data.kind.span, format!("expected {}", usage(kind))),
            )
        };

        let u32_ty = self.types.int(IntTy::U32);
        // where the key and value types are written, for the errors on them
        let (mut key_span, mut value_span) = (None, None);
        let (key, value, entries) = if kind.is_array() {
            let [TypeArgRef::Type(value_ref), length] = data.args.as_slice() else {
                return bad_args(
                    self,
                    &format!("a value type and a length, as in `{}[V, 16]`", kind.name()),
                );
            };
            let value = self.lower_ty(data.module, value_ref);
            value_span = value_ref.span();
            let length = match length {
                TypeArgRef::Const(expr) => expr.clone(),
                TypeArgRef::Type(TypeRef::Path { path, args, .. }) if args.is_empty() => {
                    ConstExpr::Path(path.clone())
                }
                TypeArgRef::Type(_) => {
                    return bad_args(
                        self,
                        &format!("a value type and a length, as in `{}[V, 16]`", kind.name()),
                    );
                }
            };
            (u32_ty, value, Some(length))
        } else if kind == MapKind::ProgArray {
            if !data.args.is_empty() {
                return bad_args(self, "no type arguments, as in `ProgArray(max_entries=4)`");
            }
            (u32_ty, u32_ty, None)
        } else if kind == MapKind::RingBuf {
            let [TypeArgRef::Type(event)] = data.args.as_slice() else {
                return bad_args(self, "the type of the events, as in `RingBuf[Event]`");
            };
            value_span = event.span();
            (self.types.unit, self.lower_ty(data.module, event), None)
        } else {
            let [TypeArgRef::Type(key), TypeArgRef::Type(value)] = data.args.as_slice() else {
                return bad_args(
                    self,
                    &format!("a key type and a value type, as in `{}[K, V]`", kind.name()),
                );
            };
            (key_span, value_span) = (key.span(), value.span());
            let key = self.lower_ty(data.module, key);
            (key, self.lower_ty(data.module, value), None)
        };

        // `max_entries=N` for a hash map or a program array, `size=N` for a
        // ring buffer. An array has its length in its type arguments.
        let param_name = match kind {
            MapKind::RingBuf => Some("size"),
            k if k.is_hash() || k == MapKind::ProgArray => Some("max_entries"),
            _ => None,
        };
        let mut sized = entries;
        for param in &data.params {
            let name = param.name.as_ref().map(|n| n.text.as_str());
            if name.is_some() && name == param_name {
                sized = Some(param.value.clone());
            } else {
                let span = param
                    .name
                    .as_ref()
                    .map(|n| n.span)
                    .or_else(|| param.value.span())
                    .unwrap_or(data.span);
                let expected =
                    param_name.map_or("no parameters".to_string(), |p| format!("`{p}=N`"));
                let mut diagnostic = Diagnostic::error("unexpected map parameter")
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(span, format!("expected {expected}"));
                if kind.is_array() {
                    diagnostic = diagnostic.with_help(format!(
                        "the length of `{}` goes in brackets, as in `{}[V, 16]`",
                        kind.name(),
                        kind.name()
                    ));
                }
                return self.map_error(diagnostic);
            }
        }
        let Some(size_expr) = sized else {
            return self.map_error(
                Diagnostic::error(format!("`{}` needs its size", data.name.text))
                    .with_code(codes::BAD_ARGUMENTS)
                    .with_primary(
                        data.name.span,
                        format!("add `({}=N)`", param_name.unwrap_or("max_entries")),
                    ),
            );
        };
        let size_span = size_expr.span().unwrap_or(data.span);
        let max_entries = match self.eval(data.module, &size_expr) {
            Some(Value::Int(n)) if n > 0 && n <= i128::from(u32::MAX) => n as u32,
            Some(found) => {
                let found = match found {
                    Value::Int(n) => n.to_string(),
                    other => crate::db::value_kind(&other).to_string(),
                };
                return self.map_error(
                    Diagnostic::error("invalid map size")
                        .with_code(codes::CONST_EVAL)
                        .with_primary(
                            size_span,
                            format!(
                                "expected a positive integer that fits in 32 bits, found {found}"
                            ),
                        ),
                );
            }
            None => return None,
        };
        if kind == MapKind::RingBuf && (!max_entries.is_power_of_two() || max_entries % 4096 != 0) {
            return self.map_error(
                Diagnostic::error(
                    "the size of a ring buffer is a power of two and a multiple of 4096",
                )
                .with_code(codes::CONST_EVAL)
                .with_primary(size_span, vigil_diagnostics::plural(max_entries, "byte"))
                .with_help(format!(
                    "use {} bytes",
                    max_entries
                        .max(4096)
                        .checked_next_power_of_two()
                        .unwrap_or(1 << 31)
                )),
            );
        }

        for (ty, what, at) in [(key, "key", key_span), (value, "value", value_span)] {
            let at = at.unwrap_or(data.kind.span);
            if kind == MapKind::RingBuf && what == "key" {
                continue;
            }
            if !self.storable(ty) {
                let shown = self.display(ty);
                return self.map_error(
                    Diagnostic::error(format!("`{shown}` cannot be stored in a map"))
                        .with_code(codes::WRONG_KIND)
                        .with_primary(at, format!("invalid {what} type"))
                        .with_note("maps store integers, booleans, and structs, enums, options and arrays of them"),
                );
            }
            let (size, _) = self.size_of(ty);
            // the values of an array are used in place, through references.
            // Other keys and values are built on the stack.
            let array_value =
                what == "value" && matches!(kind, MapKind::Array | MapKind::PerCpuArray);
            let limit = if array_value {
                MAX_ARRAY_VALUE_SIZE
            } else {
                MAX_ENTRY_SIZE
            };
            if size > limit {
                let note = if array_value {
                    "the kernel allocates per-CPU values in 32 KiB units"
                } else {
                    "keys and values are built on the 512-byte stack; store large values in an `Array` or `PerCpuArray`"
                };
                return self.map_error(
                    Diagnostic::error(format!("the {what} type is too large ({size} bytes)"))
                        .with_code(codes::CODEGEN_LIMIT)
                        .with_primary(at, format!("at most {limit} bytes"))
                        .with_note(note),
                );
            }
        }
        Some(MapInfo {
            kind,
            key,
            value,
            max_entries,
        })
    }

    /// whether values of a type can live in map memory: no references,
    /// contexts, strings or floats
    pub fn storable(&self, ty: TyId) -> bool {
        match self.types.kind(ty) {
            TyKind::Int(_) | TyKind::Bool | TyKind::Error => true,
            TyKind::Struct(s) => {
                self.struct_contexts.get(*s).is_none()
                    && self.struct_fields[*s].iter().all(|&f| self.storable(f))
            }
            TyKind::Enum(e) => self.variant_fields[*e]
                .iter()
                .flatten()
                .flatten()
                .all(|&f| self.storable(f)),
            TyKind::Option(inner) | TyKind::Array(inner, _) => self.storable(*inner),
            // a kernel pointer is stored as its address
            TyKind::KPtr(_) => true,
            TyKind::Str
            | TyKind::Unit
            | TyKind::Ref(_)
            | TyKind::Map(_)
            | TyKind::Var(_)
            | TyKind::Kernel(_)
            | TyKind::TpContext(_)
            | TyKind::Float => false,
        }
    }

    /// the size and alignment of a type in memory, which must agree with the
    /// code generators
    ///
    /// Fields are laid out as in C. A `@network_order` struct (a header) has
    /// no padding, and its alignment is the one of C if its fields fall at
    /// the same offsets as in C, 1 otherwise. An enum is a 4-byte tag when no
    /// variant has values, else an 8-byte tag and the largest payload in
    /// 8-byte words. An option of a reference is a nullable pointer, any
    /// other option a flag byte and the value.
    pub fn size_of(&self, ty: TyId) -> (u64, u64) {
        fn record(fields: impl Iterator<Item = (u64, u64)>) -> (u64, u64) {
            let mut size: u64 = 0;
            let mut align: u64 = 1;
            for (s, a) in fields {
                size = size.next_multiple_of(a) + s;
                align = align.max(a);
            }
            (size.next_multiple_of(align), align)
        }
        fn packed(fields: impl Iterator<Item = (u64, u64)>) -> (u64, u64) {
            let mut size: u64 = 0;
            let mut align: u64 = 1;
            let mut natural = true;
            for (s, a) in fields {
                natural &= size.is_multiple_of(a);
                size += s;
                align = align.max(a);
            }
            if natural && size.is_multiple_of(align) {
                (size, align)
            } else {
                (size, 1)
            }
        }
        match self.types.kind(ty) {
            TyKind::Int(int) => {
                let bytes = u64::from(int.bits() / 8);
                (bytes, bytes)
            }
            TyKind::Bool => (1, 1),
            TyKind::Ref(_) | TyKind::KPtr(_) | TyKind::TpContext(_) => (8, 8),
            TyKind::Struct(s) if self.struct_contexts.get(*s).is_some() => (8, 8),
            TyKind::Struct(s) if self.network_order.get(*s).is_some() => {
                packed(self.struct_fields[*s].iter().map(|&f| self.size_of(f)))
            }
            TyKind::Struct(s) => record(self.struct_fields[*s].iter().map(|&f| self.size_of(f))),
            TyKind::Enum(e) => {
                let largest = self.variant_fields[*e]
                    .iter()
                    .flatten()
                    .map(|fields| record(fields.iter().map(|&f| self.size_of(f))).0)
                    .max()
                    .unwrap_or(0);
                let words = largest.div_ceil(8);
                if words == 0 {
                    (4, 4)
                } else {
                    (8 + 8 * words, 8)
                }
            }
            TyKind::Option(inner) if matches!(self.types.kind(*inner), TyKind::Ref(_)) => (8, 8),
            TyKind::Option(inner) => record([(1, 1), self.size_of(*inner)].into_iter()),
            TyKind::Array(element, len) => {
                let (size, align) = self.size_of(*element);
                (size * len, align)
            }
            TyKind::Str
            | TyKind::Unit
            | TyKind::Map(_)
            | TyKind::Var(_)
            | TyKind::Kernel(_)
            | TyKind::Error => (0, 1),
            TyKind::Float => (8, 8),
        }
    }
}
