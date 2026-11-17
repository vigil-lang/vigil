//! this module defines how Vigil types are represented in LLVM IR and how
//! large they are. [`Layout`] also carries the options of the generator.
//!
//! Integers are `iN`, `bool` is `i1`, contexts are `ptr`, a struct is a
//! named type `%s.N`, `T[N]` is `[N x T]`. A `@network_order` struct has no
//! padding (see [`packed_size_align`]); when that leaves a field where C
//! would not put it, its type is packed, `<{ ... }>`, with alignment 1. An
//! enum is `%e.N = type { i32, [K x i64] }`: the variant index, then room
//! for the largest payload. `Option[T]` is `{ i1, T }`, all zeros for `None`, except
//! `Option[ref[T]]`, a pointer that is null for `None`.
//!
//! We compute sizes here instead of asking LLVM, following [`DATA_LAYOUT`];
//! the room for enum payloads is only right if both agree. Loads and stores
//! also carry an alias tag saying if the memory may be in the packet
//! ([`Layout::packet_types`]), so LLVM can keep a header field in a register
//! across a write to a map value.

use std::collections::HashSet;

use vigil_sema::Analysis;
use vigil_sema::comptime::CValue;
use vigil_sema::hir::{EnumId, StructId};
use vigil_sema::infer::Res;
use vigil_sema::ty::{IntTy, TyId, TyKind};

/// the data layout of little-endian BPF, as clang has it
pub(crate) const DATA_LAYOUT: &str = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
/// the target triple, little-endian BPF
pub(crate) const TRIPLE: &str = "bpfel";

/// the layout of the types of a package, under the options of the generator
pub(crate) struct Layout<'a> {
    pub analysis: &'a Analysis,
    /// packet bounds read as 64-bit words at offsets 0 and 8 of the
    /// context, as userspace interpreters do (for tests)
    pub test_context: bool,
    /// reading kernel memory with `bpf_probe_read_kernel` (Linux 5.5)
    /// instead of `bpf_probe_read`
    pub probe_read_kernel: bool,
    /// the size from which locals and temporaries go in scratch, if they do
    pub scratch: Option<u64>,
    /// running long loops with `bpf_loop` (Linux 5.17)
    pub bpf_loop: bool,
    /// the architecture of the target kernels, which fixes the layout of
    /// `struct pt_regs`
    pub arch: vigil_target::Arch,
    /// whether the generator optimizes (`Options::optimize`)
    pub optimize: bool,
    /// the aggregate types that may be packet memory: the headers parsed
    /// anywhere (`ctx.parse[H]`) and the aggregates inside them
    pub packet_types: HashSet<TyId>,
}

/// the aggregate types that may be packet memory, for
/// [`Layout::packet_types`]: the headers some body parses, closed over the
/// parts of each aggregate
pub(crate) fn packet_types(analysis: &Analysis) -> HashSet<TyId> {
    let mut todo: Vec<TyId> = analysis
        .bodies
        .iter()
        .flat_map(|(_, res)| res.res.values())
        .filter_map(|res| match res {
            Res::PacketParse(s) => analysis.types.lookup(&TyKind::Struct(*s)),
            _ => None,
        })
        .collect();
    let mut found = HashSet::new();
    while let Some(ty) = todo.pop() {
        if !is_aggregate(analysis, ty) || !found.insert(ty) {
            continue;
        }
        todo.extend(components(analysis, ty));
    }
    found
}

/// checking if a type is a struct, array, enum or option
fn is_aggregate(analysis: &Analysis, ty: TyId) -> bool {
    matches!(
        analysis.types.kind(ty),
        TyKind::Struct(_) | TyKind::Array(..) | TyKind::Enum(_) | TyKind::Option(_)
    )
}

/// the types of the parts of an aggregate: fields of a struct or of any
/// variant, element of an array, value of an option. Empty otherwise.
fn components(analysis: &Analysis, ty: TyId) -> Vec<TyId> {
    match analysis.types.kind(ty) {
        TyKind::Struct(s) => analysis.struct_fields.get(*s).cloned().unwrap_or_default(),
        TyKind::Array(element, _) => vec![*element],
        TyKind::Option(inner) => vec![*inner],
        TyKind::Enum(e) => analysis
            .variant_fields
            .get(*e)
            .into_iter()
            .flatten()
            .flatten()
            .flatten()
            .copied()
            .collect(),
        _ => Vec::new(),
    }
}

impl<'a> Layout<'a> {
    /// checking that memory reached from a value of type `ty` is never
    /// packet memory: `ty` is an aggregate that may not be in the packet
    /// and holds none that may. This is enough because Vigil has no casts
    /// and no unions, so two aggregates share memory only when one holds
    /// the other. A non-aggregate gives `false`, meaning we know nothing.
    pub fn packet_free(&self, ty: TyId) -> bool {
        is_aggregate(self.analysis, ty)
            && !self.packet_types.contains(&ty)
            && components(self.analysis, ty)
                .into_iter()
                .all(|c| !is_aggregate(self.analysis, c) || self.packet_free(c))
    }

    pub fn kind(&self, ty: TyId) -> &'a TyKind {
        self.analysis.types.kind(ty)
    }

    pub fn is_unit(&self, ty: TyId) -> bool {
        matches!(self.kind(ty), TyKind::Unit)
    }

    /// the LLVM type of values of `ty`
    pub fn ty(&self, ty: TyId) -> String {
        match self.kind(ty) {
            TyKind::Int(int) => format!("i{}", int.bits()),
            TyKind::Bool => "i1".into(),
            // floats exist only in comptime code, which is never compiled
            TyKind::Float => "double".into(),
            TyKind::Struct(s) if self.analysis.struct_contexts.get(*s).is_some() => "ptr".into(),
            TyKind::Struct(s) => format!("%s.{}", s.into_raw()),
            TyKind::Enum(e) => format!("%e.{}", e.into_raw()),
            TyKind::Option(_) if self.is_niche(ty) => "ptr".into(),
            TyKind::Option(inner) => format!("{{ i1, {} }}", self.ty(*inner)),
            TyKind::Ref(_) | TyKind::KPtr(_) | TyKind::TpContext(_) => "ptr".into(),
            TyKind::Array(element, len) => format!("[{len} x {}]", self.ty(*element)),
            // not representable: the type checker rejects them wherever a
            // value would be needed
            TyKind::Unit
            | TyKind::Str
            | TyKind::Map(_)
            | TyKind::Kernel(_)
            | TyKind::Var(_)
            | TyKind::Error => "void".into(),
        }
    }

    /// checking if values of `ty` hold pointers. Those must stay on the
    /// stack, because the verifier forgets what a pointer stored in a map
    /// points to, so neither scratch nor stage carries can hold one.
    pub fn has_pointer(&self, ty: TyId) -> bool {
        match self.kind(ty) {
            TyKind::Ref(_) | TyKind::KPtr(_) | TyKind::TpContext(_) => true,
            TyKind::Struct(s) if self.analysis.struct_contexts.get(*s).is_some() => true,
            TyKind::Struct(s) => self.analysis.struct_fields[*s]
                .iter()
                .any(|&t| self.has_pointer(t)),
            TyKind::Option(inner) | TyKind::Array(inner, _) => self.has_pointer(*inner),
            TyKind::Enum(e) => self.analysis.variant_fields[*e]
                .iter()
                .flatten()
                .flatten()
                .any(|&t| self.has_pointer(t)),
            _ => false,
        }
    }

    /// checking if values of `ty` go in scratch: scratch is on, they are
    /// large enough, and they hold no pointer
    pub fn in_scratch(&self, ty: TyId) -> bool {
        self.scratch
            .is_some_and(|min| self.size_align(ty).0 >= min && !self.has_pointer(ty))
    }

    /// checking if `ty` is an `Option[ref[T]]`. Those are a pointer that is
    /// null for `None`, which is what map lookups return and the check the
    /// verifier understands.
    pub fn is_niche(&self, ty: TyId) -> bool {
        matches!(self.kind(ty), TyKind::Option(inner) if matches!(self.kind(*inner), TyKind::Ref(_)))
    }

    /// the literal struct holding the payload of a variant, or `None` for a
    /// variant without fields
    pub fn payload_ty(&self, e: EnumId, variant: usize) -> Option<String> {
        let fields = self.analysis.variant_fields[e][variant].as_ref()?;
        let fields: Vec<_> = fields.iter().map(|&t| self.ty(t)).collect();
        Some(format!("{{ {} }}", fields.join(", ")))
    }

    /// the number of 8-byte words reserved for the payloads of an enum,
    /// enough for the largest
    pub fn payload_words(&self, e: EnumId) -> u64 {
        let largest = self.analysis.variant_fields[e]
            .iter()
            .flatten()
            .map(|fields| struct_size_align(fields.iter().map(|&t| self.size_align(t))).0)
            .max()
            .unwrap_or(0);
        largest.div_ceil(8)
    }

    /// size and alignment of values of `ty`, in bytes
    ///
    /// An enum without payloads is just its `i32` index; with payloads, the
    /// index padded to 8 bytes and then the payload words. Types that are
    /// not representable take no room.
    pub fn size_align(&self, ty: TyId) -> (u64, u64) {
        match self.kind(ty) {
            TyKind::Int(int) => {
                let bytes = u64::from(int.bits() / 8);
                (bytes, bytes)
            }
            TyKind::Bool => (1, 1),
            TyKind::Float => (8, 8),
            TyKind::Struct(s) if self.analysis.struct_contexts.get(*s).is_some() => (8, 8),
            TyKind::Struct(s) => {
                let fields = self.analysis.struct_fields[*s]
                    .iter()
                    .map(|&t| self.size_align(t));
                if self.analysis.network_order.get(*s).is_some() {
                    packed_size_align(fields)
                } else {
                    struct_size_align(fields)
                }
            }
            TyKind::Enum(e) => {
                let words = self.payload_words(*e);
                if words == 0 {
                    (4, 4)
                } else {
                    (8 + 8 * words, 8)
                }
            }
            TyKind::Option(_) if self.is_niche(ty) => (8, 8),
            TyKind::Ref(_) | TyKind::KPtr(_) | TyKind::TpContext(_) => (8, 8),
            TyKind::Option(inner) => {
                struct_size_align([(1, 1), self.size_align(*inner)].into_iter())
            }
            TyKind::Array(element, len) => {
                let (size, align) = self.size_align(*element);
                (size * len, align)
            }
            TyKind::Unit
            | TyKind::Str
            | TyKind::Map(_)
            | TyKind::Kernel(_)
            | TyKind::Var(_)
            | TyKind::Error => (0, 1),
        }
    }

    /// checking if the LLVM type of the struct `s` is packed: a
    /// `@network_order` struct with alignment 1. With a larger alignment,
    /// its fields are already where C puts them.
    pub fn is_packed(&self, s: StructId) -> bool {
        self.analysis.network_order.get(s).is_some()
            && self.analysis.struct_fields.get(s).is_some_and(|fields| {
                packed_size_align(fields.iter().map(|&t| self.size_align(t))).1 == 1
            })
    }

    /// the offsets of the fields of the struct `s`, which follow each other
    /// in a `@network_order` struct
    pub fn struct_offsets(&self, s: StructId) -> Vec<u64> {
        let types = &self.analysis.struct_fields[s];
        if self.analysis.network_order.get(s).is_none() {
            return self.field_offsets(types);
        }
        let mut offset = 0;
        types
            .iter()
            .map(|&t| {
                let at = offset;
                offset += self.size_align(t).0;
                at
            })
            .collect()
    }

    /// the offset of part `index` of the aggregate `ty`: a field of a
    /// struct, or the value of an option (index 1). Zero for other parts.
    pub fn part_offset(&self, ty: TyId, index: u32) -> u64 {
        let offsets = match self.kind(ty) {
            TyKind::Struct(s) if self.analysis.struct_contexts.get(*s).is_none() => {
                self.struct_offsets(*s)
            }
            TyKind::Option(inner) if !self.is_niche(ty) => {
                self.field_offsets(&[self.analysis.types.bool, *inner])
            }
            _ => Vec::new(),
        };
        offsets.get(index as usize).copied().unwrap_or(0)
    }

    /// the offsets of the fields of a struct with fields of these types
    pub fn field_offsets(&self, fields: &[TyId]) -> Vec<u64> {
        let mut offset = 0;
        fields
            .iter()
            .map(|&f| {
                let (size, align) = self.size_align(f);
                offset = u64::next_multiple_of(offset, align);
                let at = offset;
                offset += size;
                at
            })
            .collect()
    }

    /// the integer and fractional bits of a `fixed[I, F]` of `std.fixed`,
    /// or `None` for other structs
    fn fixed_point(&self, s: vigil_sema::hir::StructId) -> Option<(u32, u32)> {
        let package = &self.analysis.package;
        let instance = package.structs[s].instance.as_ref()?;
        let vigil_sema::hir::Item::Struct(generic) = instance.generic else {
            return None;
        };
        let data = &package.structs[generic];
        let in_std_fixed = package.modules[data.module].path == ["std", "fixed"];
        match instance.args.as_slice() {
            [
                vigil_sema::hir::InstanceArg::Const(i),
                vigil_sema::hir::InstanceArg::Const(f),
            ] if in_std_fixed && data.name.text == "fixed" => {
                Some((u32::try_from(*i).ok()?, u32::try_from(*f).ok()?))
            }
            _ => None,
        }
    }

    /// the description of the memory layout of a type, to decode values of
    /// maps and ring buffers. A `fixed` is described as a number, and the
    /// integer fields of `@network_order` structs as big-endian.
    pub fn describe(&self, ty: TyId) -> crate::desc::TypeDesc {
        use crate::desc::TypeDesc;
        let package = &self.analysis.package;
        match self.kind(ty) {
            TyKind::Int(int) => TypeDesc::Int {
                bytes: int.bits() / 8,
                signed: int.signed(),
                big_endian: false,
            },
            TyKind::Bool => TypeDesc::Bool,
            // a kernel pointer is stored as an address: userspace can show
            // it but not follow it
            TyKind::KPtr(_) => TypeDesc::Int {
                bytes: 8,
                signed: false,
                big_endian: false,
            },
            TyKind::Struct(s) if let Some((int_bits, frac_bits)) = self.fixed_point(*s) => {
                TypeDesc::Fixed {
                    int_bits,
                    frac_bits,
                }
            }
            TyKind::Struct(s) => {
                let types = &self.analysis.struct_fields[*s];
                let offsets = self.struct_offsets(*s);
                TypeDesc::Struct {
                    name: package.structs[*s].name.text.clone(),
                    fields: package.structs[*s]
                        .fields
                        .iter()
                        .zip(types.iter().zip(offsets))
                        .map(|(f, (&t, offset))| {
                            let mut desc = self.describe(t);
                            if let TypeDesc::Int { big_endian, .. } = &mut desc {
                                *big_endian = self.analysis.network_order.get(*s).is_some();
                            }
                            (f.name.text.clone(), offset, desc)
                        })
                        .collect(),
                }
            }
            TyKind::Enum(e) => TypeDesc::Enum {
                name: package.enums[*e].name.text.clone(),
                variants: package.enums[*e]
                    .variants
                    .iter()
                    .zip(&self.analysis.variant_fields[*e])
                    .map(|(v, fields)| {
                        let fields = fields.clone().unwrap_or_default();
                        let offsets = self.field_offsets(&fields);
                        let described = fields
                            .iter()
                            .zip(offsets)
                            .map(|(&t, offset)| (8 + offset, self.describe(t)))
                            .collect();
                        (v.name.text.clone(), described)
                    })
                    .collect(),
            },
            TyKind::Option(inner) => {
                let offset = self.field_offsets(&[self.analysis.types.bool, *inner])[1];
                TypeDesc::Option {
                    inner: Box::new(self.describe(*inner)),
                    offset,
                }
            }
            TyKind::Array(element, len) => TypeDesc::Array {
                element: Box::new(self.describe(*element)),
                len: *len,
                stride: self.size_align(*element).0,
            },
            _ => TypeDesc::Unit,
        }
    }

    /// `%s.N = type { ... }` and `%e.N = type { ... }` for every struct and
    /// enum of the package
    pub fn type_definitions(&self) -> String {
        let mut out = String::new();
        for (id, _) in self.analysis.package.structs.iter() {
            // contexts are pointers, not structs. Generic templates have no
            // fields here and are never used, only their instances.
            let Some(fields) = self.analysis.struct_fields.get(id) else {
                continue;
            };
            if self.analysis.struct_contexts.get(id).is_some() {
                continue;
            }
            let fields: Vec<_> = fields.iter().map(|&t| self.ty(t)).collect();
            let (open, close) = if self.is_packed(id) {
                ("<{", "}>")
            } else {
                ("{", "}")
            };
            out.push_str(&format!(
                "%s.{} = type {open} {} {close}\n",
                id.into_raw(),
                fields.join(", ")
            ));
        }
        for (id, _) in self.analysis.package.enums.iter() {
            if self.analysis.variant_fields.get(id).is_none() {
                continue;
            }
            out.push_str(&format!(
                "%e.{} = type {{ i32, [{} x i64] }}\n",
                id.into_raw(),
                self.payload_words(id)
            ));
        }
        out
    }
}

/// size and alignment of a struct with fields of these sizes and
/// alignments, in order with padding
fn struct_size_align(fields: impl Iterator<Item = (u64, u64)>) -> (u64, u64) {
    let mut size: u64 = 0;
    let mut align: u64 = 1;
    for (field_size, field_align) in fields {
        size = size.next_multiple_of(field_align);
        size += field_size;
        align = align.max(field_align);
    }
    (size.next_multiple_of(align), align)
}

/// size and alignment of a `@network_order` struct with fields of these
/// sizes and alignments. A header is byte-exact, so the fields follow each
/// other without padding. The alignment is the one of C when that puts the
/// fields at the same offsets (the headers of `std.net`), so their fields
/// are still read with one load each; otherwise it is 1.
pub(crate) fn packed_size_align(fields: impl Iterator<Item = (u64, u64)>) -> (u64, u64) {
    let mut size: u64 = 0;
    let mut align: u64 = 1;
    let mut natural = true;
    for (field_size, field_align) in fields {
        natural &= size.is_multiple_of(field_align);
        size += field_size;
        align = align.max(field_align);
    }
    if natural && size.is_multiple_of(align) {
        (size, align)
    } else {
        (size, 1)
    }
}

impl Layout<'_> {
    /// the LLVM constant of a value computed at compile time, without its
    /// type
    pub fn constant(&self, ty: TyId, value: &CValue) -> String {
        match (self.kind(ty), value) {
            (TyKind::Int(int), CValue::Int(v)) => int_const(*v, *int),
            (TyKind::Bool, CValue::Bool(b)) => b.to_string(),
            (TyKind::Array(element, _), CValue::Array(items)) => {
                let element_ll = self.ty(*element);
                let items: Vec<String> = items
                    .iter()
                    .map(|item| format!("{element_ll} {}", self.constant(*element, item)))
                    .collect();
                format!("[{}]", items.join(", "))
            }
            (TyKind::Struct(s), CValue::Struct(fields)) => {
                let types = &self.analysis.struct_fields[*s];
                let fields: Vec<String> = types
                    .iter()
                    .zip(fields)
                    .map(|(&t, v)| format!("{} {}", self.ty(t), self.constant(t, v)))
                    .collect();
                if self.is_packed(*s) {
                    format!("<{{ {} }}>", fields.join(", "))
                } else {
                    format!("{{ {} }}", fields.join(", "))
                }
            }
            (TyKind::Option(inner), CValue::Option(Some(v))) if !self.is_niche(ty) => format!(
                "{{ i1 true, {} {} }}",
                self.ty(*inner),
                self.constant(*inner, v)
            ),
            // `None`, a niche option, or any other shape: all zeros. Never
            // `poison`, which LLVM may read from any register.
            _ => "zeroinitializer".into(),
        }
    }
}

/// an integer constant of type `int` as LLVM expects it: the value wrapped
/// to the width of the type, in signed form
pub(crate) fn int_const(value: i128, int: IntTy) -> String {
    let bits = int.bits();
    let mask = (1u128 << bits) - 1;
    let wrapped = (value as u128) & mask;
    let signed = if wrapped >> (bits - 1) == 1 {
        wrapped as i128 - (1i128 << bits)
    } else {
        wrapped as i128
    };
    signed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_wrapped_and_signed() {
        assert_eq!(int_const(255, IntTy::U8), "-1");
        assert_eq!(int_const(127, IntTy::U8), "127");
        assert_eq!(int_const(u64::MAX as i128, IntTy::U64), "-1");
        assert_eq!(int_const(-5, IntTy::I32), "-5");
    }

    #[test]
    fn structs_are_padded() {
        assert_eq!(struct_size_align([(1, 1), (4, 4)].into_iter()), (8, 4));
        assert_eq!(struct_size_align([(8, 8), (1, 1)].into_iter()), (16, 8));
        assert_eq!(struct_size_align(std::iter::empty()), (0, 1));
    }

    #[test]
    fn network_structs_are_not_padded() {
        // `a: u8, b: u16, c: u32`: 7 bytes, alignment 1
        assert_eq!(
            packed_size_align([(1, 1), (2, 2), (4, 4)].into_iter()),
            (7, 1)
        );
        // `a: u32, b: u8`: no tail padding either
        assert_eq!(packed_size_align([(4, 4), (1, 1)].into_iter()), (5, 1));
        // `EthHdr` and `Ipv4Hdr` are laid out as in C
        assert_eq!(
            packed_size_align([(6, 1), (6, 1), (2, 2)].into_iter()),
            (14, 2)
        );
        let ipv4 = [
            (1, 1),
            (1, 1),
            (2, 2),
            (2, 2),
            (2, 2),
            (1, 1),
            (1, 1),
            (2, 2),
            (4, 4),
            (4, 4),
        ];
        assert_eq!(packed_size_align(ipv4.into_iter()), (20, 4));
        assert_eq!(packed_size_align(std::iter::empty()), (0, 1));
    }
}
