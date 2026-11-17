//! this crate reads BTF, the BPF Type Format, which is how a Vigil program
//! imports a kernel struct from `/sys/kernel/btf/vmlinux`. The main queries
//! are [`Btf::find_composite`], [`Btf::find_member`] and [`Btf::size`].
//!
//! We decode only what Vigil needs: names, sizes, members and offsets,
//! pointers, arrays, typedefs and qualifiers. Functions, prototypes, data
//! sections and declaration tags become [`Type::Other`], but they still take
//! an id.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// the default location of the kernel BTF
pub const VMLINUX: &str = "/sys/kernel/btf/vmlinux";

/// a BTF type, as far as we decode it. A `u32` in a variant is the id of
/// another type of the same [`Btf`]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Void,
    Int {
        name: String,
        size: u32,
        signed: bool,
        boolean: bool,
    },
    Ptr(u32),
    Array {
        element: u32,
        len: u32,
    },
    Struct(Composite),
    Union(Composite),
    Enum {
        name: Option<String>,
        size: u32,
        signed: bool,
    },
    /// a forward declaration of a struct or union
    Fwd {
        name: String,
        union: bool,
    },
    Typedef {
        name: String,
        ty: u32,
    },
    /// `volatile`, `const`, `restrict` or a type tag. [`Btf::resolve`] looks
    /// through them
    Qualifier(u32),
    Float {
        size: u32,
    },
    /// a global variable, in an object file the maps of `.maps`
    Var {
        name: String,
        ty: u32,
    },
    /// a function, prototype, data section or declaration tag, kept only so
    /// the following ids stay right
    Other,
}

/// the body of a struct or union
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Composite {
    /// `None` when anonymous
    pub name: Option<String>,
    /// size in bytes
    pub size: u32,
    pub members: Vec<Member>,
}

/// a member of a struct or union
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    /// `None` for an anonymous member, whose members are reached directly
    /// (see [`Btf::find_member`])
    pub name: Option<String>,
    pub ty: u32,
    /// offset in bits from the start of the enclosing type
    pub bit_offset: u32,
    /// bitfield width in bits, zero if not a bitfield
    pub bitfield_size: u32,
}

/// why a blob could not be read, as a message for the user
#[derive(Debug, PartialEq, Eq)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// the types of a BTF blob
///
/// a type id indexes `types`. We put `void` at id 0, since BTF does not
/// encode it, so the ids need no adjustment.
#[derive(Debug)]
pub struct Btf {
    types: Vec<Type>,
    /// the type of each name. A complete struct or union wins over an
    /// incomplete one, and a struct or union over a typedef
    names: HashMap<String, u32>,
}

// kinds of type records, as in the kernel `include/uapi/linux/btf.h`
// -------------------------

const KIND_INT: u32 = 1;
const KIND_PTR: u32 = 2;
const KIND_ARRAY: u32 = 3;
const KIND_STRUCT: u32 = 4;
const KIND_UNION: u32 = 5;
const KIND_ENUM: u32 = 6;
const KIND_FWD: u32 = 7;
const KIND_TYPEDEF: u32 = 8;
const KIND_VOLATILE: u32 = 9;
const KIND_CONST: u32 = 10;
const KIND_RESTRICT: u32 = 11;
const KIND_FUNC: u32 = 12;
const KIND_FUNC_PROTO: u32 = 13;
const KIND_VAR: u32 = 14;
const KIND_DATASEC: u32 = 15;
const KIND_FLOAT: u32 = 16;
const KIND_DECL_TAG: u32 = 17;
const KIND_TYPE_TAG: u32 = 18;
const KIND_ENUM64: u32 = 19;

/// a cursor over the type section, in little-endian 32-bit words
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn u32(&mut self) -> Result<u32, Error> {
        let word = self
            .bytes
            .get(self.at..self.at + 4)
            .ok_or_else(|| Error("truncated BTF type section".into()))?;
        self.at += 4;
        Ok(u32::from_le_bytes(word.try_into().unwrap()))
    }

    /// skipping words. A skip past the end shows up in the next read, or
    /// ends the loop of [`Btf::parse`]
    fn skip(&mut self, words: usize) {
        self.at += 4 * words;
    }
}

impl Btf {
    /// reading the BTF in a file, like [`VMLINUX`]
    pub fn from_file(path: impl AsRef<Path>) -> Result<Btf, Error> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| Error(format!("could not read {}: {e}", path.display())))?;
        Btf::parse(&bytes)
    }

    /// parsing a BTF blob
    ///
    /// a 24-byte header, then the type records and the strings, at offsets
    /// from the end of the header. Each record is three words (name, info,
    /// size or type) and data depending on kind and `vlen`. We assume
    /// little-endian, as on both architectures Vigil targets.
    pub fn parse(bytes: &[u8]) -> Result<Btf, Error> {
        let word = |at: usize| -> Result<u32, Error> {
            bytes
                .get(at..at + 4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .ok_or_else(|| Error("truncated BTF header".into()))
        };
        if bytes.len() < 24 || u16::from_le_bytes([bytes[0], bytes[1]]) != 0xeb9f {
            return Err(Error("not BTF (bad magic number)".into()));
        }
        let header_len = word(4)? as usize;
        let (type_off, type_len) = (word(8)? as usize, word(12)? as usize);
        let (str_off, str_len) = (word(16)? as usize, word(20)? as usize);
        let section = |off: usize, len: usize| {
            bytes
                .get(header_len + off..header_len + off + len)
                .ok_or_else(|| Error("BTF section out of bounds".into()))
        };
        let strings = section(str_off, str_len)?;
        // offset 0 is the empty string, an absent name
        let string = |offset: u32| -> Option<String> {
            let start = offset as usize;
            if start == 0 || start >= strings.len() {
                return None;
            }
            let end = strings[start..].iter().position(|&b| b == 0)? + start;
            Some(String::from_utf8_lossy(&strings[start..end]).into_owned())
        };

        let mut reader = Reader {
            bytes: section(type_off, type_len)?,
            at: 0,
        };
        let mut types = vec![Type::Void];
        while reader.at < reader.bytes.len() {
            let name = string(reader.u32()?);
            let info = reader.u32()?;
            let size_or_type = reader.u32()?;
            let vlen = (info & 0xffff) as usize;
            let kind = (info >> 24) & 0x1f;
            let kind_flag = info >> 31 == 1;
            let ty = match kind {
                // flags in the top byte of the encoding: 1 signed, 2 char,
                // 4 bool
                KIND_INT => {
                    let encoding = reader.u32()?;
                    Type::Int {
                        name: name.unwrap_or_default(),
                        size: size_or_type,
                        signed: (encoding >> 24) & 1 == 1,
                        boolean: (encoding >> 24) & 4 == 4,
                    }
                }
                KIND_PTR => Type::Ptr(size_or_type),
                KIND_ARRAY => {
                    let element = reader.u32()?;
                    let _index = reader.u32()?;
                    let len = reader.u32()?;
                    Type::Array { element, len }
                }
                // with the kind flag, the offset word has the bitfield width
                // in its top byte and the bit offset below
                KIND_STRUCT | KIND_UNION => {
                    let mut members = Vec::with_capacity(vlen);
                    for _ in 0..vlen {
                        let member_name = string(reader.u32()?);
                        let member_ty = reader.u32()?;
                        let offset = reader.u32()?;
                        let (bit_offset, bitfield_size) = if kind_flag {
                            (offset & 0x00ff_ffff, offset >> 24)
                        } else {
                            (offset, 0)
                        };
                        members.push(Member {
                            name: member_name,
                            ty: member_ty,
                            bit_offset,
                            bitfield_size,
                        });
                    }
                    let composite = Composite {
                        name,
                        size: size_or_type,
                        members,
                    };
                    if kind == KIND_STRUCT {
                        Type::Struct(composite)
                    } else {
                        Type::Union(composite)
                    }
                }
                KIND_ENUM => {
                    reader.skip(2 * vlen);
                    Type::Enum {
                        name,
                        size: size_or_type,
                        signed: kind_flag,
                    }
                }
                KIND_ENUM64 => {
                    reader.skip(3 * vlen);
                    Type::Enum {
                        name,
                        size: size_or_type,
                        signed: kind_flag,
                    }
                }
                KIND_FWD => Type::Fwd {
                    name: name.unwrap_or_default(),
                    union: kind_flag,
                },
                KIND_TYPEDEF => Type::Typedef {
                    name: name.unwrap_or_default(),
                    ty: size_or_type,
                },
                KIND_VOLATILE | KIND_CONST | KIND_RESTRICT | KIND_TYPE_TAG => {
                    Type::Qualifier(size_or_type)
                }
                KIND_FLOAT => Type::Float { size: size_or_type },
                KIND_FUNC => Type::Other,
                KIND_FUNC_PROTO => {
                    reader.skip(2 * vlen);
                    Type::Other
                }
                KIND_VAR => {
                    reader.skip(1);
                    Type::Var {
                        name: name.unwrap_or_default(),
                        ty: size_or_type,
                    }
                }
                KIND_DECL_TAG => {
                    reader.skip(1);
                    Type::Other
                }
                KIND_DATASEC => {
                    reader.skip(3 * vlen);
                    Type::Other
                }
                other => return Err(Error(format!("unknown BTF kind {other}"))),
            };
            types.push(ty);
        }

        // creating the table of names. Structs and unions go first, a
        // complete one replacing an incomplete one, then typedefs fill only
        // the free names. Forward declarations never enter, `find_composite`
        // reaches them only through a typedef
        let mut names = HashMap::new();
        for (id, ty) in types.iter().enumerate() {
            let name = match ty {
                Type::Struct(c) | Type::Union(c) => c.name.clone(),
                _ => None,
            };
            if let Some(name) = name {
                let complete = matches!(&types[id], Type::Struct(c) | Type::Union(c) if !c.members.is_empty() || c.size > 0);
                if complete || !names.contains_key(&name) {
                    names.insert(name, id as u32);
                }
            }
        }
        for (id, ty) in types.iter().enumerate() {
            if let Type::Typedef { name, .. } = ty {
                names.entry(name.clone()).or_insert(id as u32);
            }
        }
        Ok(Btf { types, names })
    }

    /// the type of an id, `void` past the end
    pub fn get(&self, id: u32) -> &Type {
        self.types.get(id as usize).unwrap_or(&Type::Void)
    }

    /// the number of type ids, `void` included
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// is there nothing but `void`?
    pub fn is_empty(&self) -> bool {
        self.types.len() <= 1
    }

    /// the type behind typedefs and qualifiers. We stop after 64 steps,
    /// because a malformed blob can have a cycle
    pub fn resolve(&self, mut id: u32) -> u32 {
        for _ in 0..64 {
            match self.get(id) {
                Type::Typedef { ty, .. } | Type::Qualifier(ty) => id = *ty,
                _ => break,
            }
        }
        id
    }

    /// the struct or union of a name, through typedefs. A forward declaration
    /// gives the complete type of the same name, if any
    pub fn find_composite(&self, name: &str) -> Option<u32> {
        let id = self.resolve(*self.names.get(name)?);
        match self.get(id) {
            Type::Struct(_) | Type::Union(_) => Some(id),
            Type::Fwd { name, .. } => self.names.get(name).copied().filter(|&i| i != id),
            _ => None,
        }
    }

    /// the struct or union behind a type and its id. A forward declaration is
    /// completed by name
    pub fn composite(&self, id: u32) -> Option<(u32, &Composite)> {
        let id = self.resolve(id);
        match self.get(id) {
            Type::Struct(c) | Type::Union(c) => Some((id, c)),
            Type::Fwd { name, .. } => {
                let complete = self.find_composite(name)?;
                match self.get(complete) {
                    Type::Struct(c) | Type::Union(c) => Some((complete, c)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub fn is_union(&self, id: u32) -> bool {
        matches!(self.get(self.resolve(id)), Type::Union(_))
    }

    /// the size of a type in bytes. Pointers are 8 bytes on every Vigil
    /// target, and types without a size (like `void`) give 0
    pub fn size(&self, id: u32) -> u32 {
        match self.get(self.resolve(id)) {
            Type::Int { size, .. } | Type::Enum { size, .. } | Type::Float { size } => *size,
            Type::Ptr(_) => 8,
            Type::Array { element, len } => self.size(*element) * len,
            Type::Struct(c) | Type::Union(c) => c.size,
            _ => 0,
        }
    }

    /// looking up a member by name, also inside anonymous members as C does.
    /// It returns the path of member indices, the member and its byte offset
    ///
    /// for a bitfield the offset is the byte of its first bit, the bit
    /// offset is in the [`Member`]
    pub fn find_member(&self, composite: u32, name: &str) -> Option<(Vec<u32>, Member, u32)> {
        let (_, c) = self.composite(composite)?;
        for (i, member) in c.members.iter().enumerate() {
            match &member.name {
                Some(n) if n == name => {
                    return Some((vec![i as u32], member.clone(), member.bit_offset / 8));
                }
                None => {
                    if let Some((mut path, found, offset)) = self.find_member(member.ty, name) {
                        path.insert(0, i as u32);
                        return Some((path, found, offset + member.bit_offset / 8));
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// the C name of a type for messages, like `struct s *` or `int[4]`,
    /// without qualifiers
    pub fn name(&self, id: u32) -> String {
        match self.get(id) {
            Type::Void => "void".into(),
            Type::Int { name, .. }
            | Type::Typedef { name, .. }
            | Type::Fwd { name, .. }
            | Type::Var { name, .. } => name.clone(),
            Type::Ptr(t) => format!("{} *", self.name(*t)),
            Type::Array { element, len } => format!("{}[{len}]", self.name(*element)),
            Type::Struct(c) => format!("struct {}", c.name.as_deref().unwrap_or("<anonymous>")),
            Type::Union(c) => format!("union {}", c.name.as_deref().unwrap_or("<anonymous>")),
            Type::Enum { name, .. } => format!("enum {}", name.as_deref().unwrap_or("<anonymous>")),
            Type::Qualifier(t) => self.name(*t),
            Type::Float { .. } => "float".into(),
            Type::Other => "?".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a hand written blob with `int`, `struct s { int a; struct s *next;
    /// union { int b; }; }` and `typedef struct s s_t`
    fn sample() -> Vec<u8> {
        let mut strings = b"\0int\0s\0a\0next\0b\0s_t\0".to_vec();
        strings.push(0);
        // the offset of a name in `strings`
        let name = |s: &str| -> u32 {
            let bytes = s.as_bytes();
            strings
                .windows(bytes.len() + 2)
                .position(|w| w[0] == 0 && &w[1..=bytes.len()] == bytes && w[bytes.len() + 1] == 0)
                .unwrap() as u32
                + 1
        };
        let mut types: Vec<u32> = Vec::new();
        // 1: int
        types.extend([name("int"), KIND_INT << 24, 4, (1 << 24) | 32]);
        // 2: struct s, 3 members, size 24
        types.extend([name("s"), (KIND_STRUCT << 24) | 3, 24]);
        types.extend([name("a"), 1, 0]);
        types.extend([name("next"), 3, 64]);
        types.extend([0, 4, 128]);
        // 3: pointer to struct s
        types.extend([0, KIND_PTR << 24, 2]);
        // 4: anonymous union { int b; }
        types.extend([0, (KIND_UNION << 24) | 1, 4]);
        types.extend([name("b"), 1, 0]);
        // 5: typedef s_t
        types.extend([name("s_t"), KIND_TYPEDEF << 24, 2]);
        let type_bytes: Vec<u8> = types.iter().flat_map(|w| w.to_le_bytes()).collect();
        // the header: magic, version 1, no flags, its length, then offsets
        // and lengths of the two sections
        let mut blob = Vec::new();
        blob.extend(0xeb9fu16.to_le_bytes());
        blob.extend([1, 0]);
        blob.extend(24u32.to_le_bytes());
        blob.extend(0u32.to_le_bytes());
        blob.extend((type_bytes.len() as u32).to_le_bytes());
        blob.extend((type_bytes.len() as u32).to_le_bytes());
        blob.extend((strings.len() as u32).to_le_bytes());
        blob.extend(type_bytes);
        blob.extend(strings);
        blob
    }

    #[test]
    fn parses_structs_and_finds_members() {
        let btf = Btf::parse(&sample()).unwrap();
        let s = btf.find_composite("s").unwrap();
        assert_eq!(btf.find_composite("s_t"), Some(s));
        assert_eq!(btf.size(s), 24);
        let (path, member, offset) = btf.find_member(s, "next").unwrap();
        assert_eq!((path, offset), (vec![1], 8));
        assert_eq!(btf.name(member.ty), "struct s *");
        // `b` is in the anonymous union, the third member
        let (path, _, offset) = btf.find_member(s, "b").unwrap();
        assert_eq!((path, offset), (vec![2, 0], 16));
        assert!(btf.find_member(s, "missing").is_none());
    }

    #[test]
    fn rejects_other_files() {
        assert!(Btf::parse(b"\x7fELF and more bytes here....").is_err());
    }

    /// skipped on a machine without kernel BTF
    #[test]
    fn reads_the_running_kernel() {
        let Ok(btf) = Btf::from_file(VMLINUX) else {
            eprintln!("skipping: no {VMLINUX}");
            return;
        };
        let task = btf.find_composite("task_struct").expect("task_struct");
        for field in ["pid", "tgid", "real_parent", "comm"] {
            assert!(btf.find_member(task, field).is_some(), "{field}");
        }
        let (_, comm, _) = btf.find_member(task, "comm").unwrap();
        assert_eq!(btf.size(comm.ty), 16);
    }
}
