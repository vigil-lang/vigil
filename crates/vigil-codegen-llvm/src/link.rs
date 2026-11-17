//! this module links a program for a formal model of eBPF. The full Vigil
//! runs programs against such a model (not in this repository), and
//! [`model_input`] gives its input as JSON: the linked code ([`link`]), the
//! map sizes ([`maps`]), the context ([`context`]) and the prototypes of the
//! helpers called. [`line_of`] maps a linked instruction back to a source
//! line.
//!
//! The model has no ELF and no relocations, so we give it the program
//! followed by `.text` (the `bpf_loop` callbacks), and rewrite each
//! relocated `lddw`: a map gets source 1 and its index in `.maps`, a
//! callback gets source 4 and the index of its first instruction.

use object::{Object, ObjectSection, ObjectSymbol, RelocationTarget};
use serde_json::{Value, json};
use vigil_sema::program::ProgramKind;

use crate::ProgramDesc;

/// the `lddw` source of a map, for the model
const MAP: u8 = 1;
/// the `lddw` source of a function, for the model
const FUNC: u8 = 4;

/// the size of the `pt_regs` of x86-64, the context of kprobes
const PT_REGS: u64 = 168;
/// a bound on tracepoint records, because the model does not know their
/// formats
const TRACEPOINT_RECORD: u64 = 4096;

/// a map of the object, as the model sees it. Its position in [`maps`] is
/// what a linked `lddw` of the map holds.
pub struct Map {
    /// the name of the map, also its symbol
    pub name: String,
    /// the `BPF_MAP_TYPE_*`
    pub kind: u32,
    pub key_size: u32,
    pub value_size: u32,
    pub max_entries: u32,
}

/// the maps of the object, in the order of their offsets in `.maps`
///
/// We read them back from the BTF, where they are as libbpf declares them:
/// a struct whose members point to arrays, and the array lengths are the
/// values (`int (*max_entries)[1024]`). A map without a type or a
/// `max_entries` is left out, and an unreadable object gives none.
pub fn maps(object: &[u8]) -> Vec<Map> {
    use vigil_btf::Type;
    let Ok(file) = object::File::parse(object) else {
        return Vec::new();
    };
    let Some(section) = file.section_by_name(".maps") else {
        return Vec::new();
    };
    let Some(btf) = file
        .section_by_name(".BTF")
        .and_then(|s| s.data().ok())
        .and_then(|data| vigil_btf::Btf::parse(data).ok())
    else {
        return Vec::new();
    };
    let definitions: std::collections::HashMap<&str, u32> = (0..btf.len() as u32)
        .filter_map(|id| match btf.get(id) {
            Type::Var { name, ty } => Some((name.as_str(), *ty)),
            _ => None,
        })
        .collect();
    let mut symbols: Vec<_> = file
        .symbols()
        .filter(|s| s.section_index() == Some(section.index()))
        .collect();
    symbols.sort_by_key(|s| s.address());
    symbols
        .into_iter()
        .filter_map(|symbol| {
            let name = symbol.name().ok()?;
            let (_, definition) = btf.composite(*definitions.get(name)?)?;
            let field = |field: &str| {
                definition
                    .members
                    .iter()
                    .find(|m| m.name.as_deref() == Some(field))
                    .and_then(|m| match btf.get(btf.resolve(m.ty)) {
                        Type::Ptr(array) => match btf.get(btf.resolve(*array)) {
                            Type::Array { len, .. } => Some(*len),
                            _ => None,
                        },
                        _ => None,
                    })
            };
            Some(Map {
                name: name.to_string(),
                kind: field("type")?,
                key_size: field("key_size").unwrap_or(0),
                value_size: field("value_size").unwrap_or(0),
                max_entries: field("max_entries")?,
            })
        })
        .collect()
}

/// the code of `program` linked for the model (see the module header).
/// A relocation that is not on an `lddw`, or that names data other than a
/// map, is an error, because the model only has maps.
pub fn link(object: &[u8], program: &ProgramDesc) -> Result<Vec<u8>, String> {
    let maps = maps(object);
    let file = object::File::parse(object).map_err(|e| e.to_string())?;
    let section_name = program.info.section();
    let section = file
        .section_by_name(&section_name)
        .ok_or_else(|| format!("no section `{section_name}`"))?;
    let data = section.data().map_err(|e| e.to_string())?;
    let symbol = file
        .symbols()
        .find(|s| s.section_index() == Some(section.index()) && s.name() == Ok(&program.name))
        .ok_or_else(|| format!("no symbol `{}`", program.name))?;
    let start = symbol.address() as usize;
    let end = if symbol.size() > 0 {
        start + symbol.size() as usize
    } else {
        data.len()
    };
    let mut code = data[start..end].to_vec();
    let text = file.section_by_name(".text");
    let text_base = code.len();
    let mut parts = vec![(section, start as u64, end as u64, 0)];
    if let Some(text) = text {
        code.extend_from_slice(text.data().map_err(|e| e.to_string())?);
        let size = text.size();
        parts.push((text, 0, size, text_base));
    }
    let text_index = file.section_by_name(".text").map(|s| s.index());
    for (part, from, to, base) in parts {
        for (offset, relocation) in part.relocations() {
            if offset < from || offset >= to {
                continue;
            }
            let at = base + (offset - from) as usize;
            let RelocationTarget::Symbol(target) = relocation.target() else {
                return Err("a relocation without a symbol".into());
            };
            let target = file.symbol_by_index(target).map_err(|e| e.to_string())?;
            if code.get(at) != Some(&0x18) {
                return Err(format!(
                    "a relocation of instruction {} that is not `lddw`",
                    at / 8
                ));
            }
            let (src, imm) = if text_index.is_some() && target.section_index() == text_index {
                // a callback: a symbol of `.text` plus the addend in the
                // immediate, which is how a section symbol is relocated
                let addend = i32::from_le_bytes(code[at + 4..at + 8].try_into().unwrap());
                let byte = target.address() as i64 + i64::from(addend);
                (FUNC, ((text_base as i64 + byte) / 8) as u32)
            } else {
                let name = target.name().map_err(|e| e.to_string())?;
                let Some(index) = maps.iter().position(|m| m.name == name) else {
                    return Err(format!(
                        "a reference to `{name}`: the model has maps, not other data \
                         (constant arrays live in read-only data)"
                    ));
                };
                (MAP, index as u32)
            };
            code[at + 1] = (code[at + 1] & 0x0f) | (src << 4);
            code[at + 4..at + 8].copy_from_slice(&imm.to_le_bytes());
            code[at + 12..at + 16].copy_from_slice(&0u32.to_le_bytes());
        }
    }
    Ok(code)
}

/// the context of a program type, as the model reads it
///
/// Packet programs must be built with `test_context`, because the model has
/// the 64-bit packet bounds at offsets 0 and 8, as userspace interpreters
/// do.
pub fn context(kind: ProgramKind) -> Value {
    let packet = |ret: Value| {
        json!({
            "size": 16,
            "writable": false,
            "fields": [
                {"offset": 0, "field": "packet"},
                {"offset": 8, "field": "packetEnd"},
            ],
            "ret": ret,
            "packetWritable": true,
        })
    };
    let tracing = |size: u64| json!({"size": size, "writable": false, "fields": [], "ret": null, "packetWritable": false});
    match kind {
        ProgramKind::Xdp | ProgramKind::Test => packet(json!([0, 4])),
        // TC actions start at -1 (`TC_ACT_UNSPEC`), so there is no unsigned
        // range to check the result against
        ProgramKind::Tc => packet(Value::Null),
        ProgramKind::Kprobe
        | ProgramKind::Kretprobe
        | ProgramKind::Uprobe
        | ProgramKind::Uretprobe => tracing(PT_REGS),
        // an LSM hook has at most 12 arguments of 64 bits
        ProgramKind::Lsm => tracing(96),
        ProgramKind::Tracepoint => tracing(TRACEPOINT_RECORD),
    }
}

/// the prototypes of the helpers Vigil calls, as Linux declares them, by
/// helper number. An argument that is not a map, a key or a value is
/// `anything`, because the model does not check that memory.
const HELPERS: &[(u32, &[&str], &str)] = &[
    // `uninitMem:rN`: a stack buffer written by the helper, of the size in
    // rN. Models that do not have it read `anything`.
    (4, &["uninitMem:r2", "anything", "anything"], "scalar"), // probe_read
    (5, &[], "scalar"),                                       // ktime_get_ns
    // trace_printk: Linux checks the format and its size, not the values
    (6, &["anything", "anything"], "scalar"),
    (8, &[], "scalar"),                            // get_smp_processor_id
    (14, &[], "scalar"),                           // get_current_pid_tgid
    (15, &[], "scalar"),                           // get_current_uid_gid
    (16, &["uninitMem:r2", "anything"], "scalar"), // get_current_comm
    (35, &[], "scalar"),                           // get_current_task
    (45, &["uninitMem:r2", "anything", "anything"], "scalar"), // probe_read_str
    (112, &["uninitMem:r2", "anything", "anything"], "scalar"), // probe_read_user
    (113, &["uninitMem:r2", "anything", "anything"], "scalar"), // probe_read_kernel
    (114, &["uninitMem:r2", "anything", "anything"], "scalar"), // probe_read_user_str
    (115, &["uninitMem:r2", "anything", "anything"], "scalar"), // probe_read_kernel_str
    (
        130,
        &["mapPtr", "anything", "anything", "anything"],
        "scalar",
    ), // ringbuf_output
];

/// the prototypes, from [`HELPERS`], of the helpers that `code` calls and
/// the model does not know
fn helpers(code: &[u8]) -> Vec<Value> {
    let mut called: Vec<u32> = code
        .as_chunks::<8>()
        .0
        .iter()
        .filter(|i| i[0] == 0x85 && i[1] >> 4 == 0)
        .map(|i| u32::from_le_bytes(i[4..8].try_into().unwrap()))
        .collect();
    called.sort_unstable();
    called.dedup();
    called
        .into_iter()
        .filter_map(|id| HELPERS.iter().find(|h| h.0 == id))
        .map(|(id, args, ret)| json!({"id": id, "args": args, "ret": ret}))
        .collect()
}

/// what the model reads for `program` (see the module header)
pub fn model_input(object: &[u8], program: &ProgramDesc) -> Result<Value, String> {
    let code = link(object, program)?;
    let hex: String = code.iter().map(|b| format!("{b:02x}")).collect();
    let maps: Vec<Value> = maps(object)
        .iter()
        .map(|m| json!({"keySize": m.key_size, "valueSize": m.value_size}))
        .collect();
    Ok(json!({
        "code": hex,
        "maps": maps,
        "context": context(program.info.kind),
        "helpers": helpers(&code),
    }))
}

/// the source file and line of instruction `pc` of the code [`link`] gives
/// for `program`, from the BTF line info. It is the last record at or
/// before the instruction, in the program section or, past its end, in
/// `.text`.
pub fn line_of(object: &[u8], program: &ProgramDesc, pc: usize) -> Option<(String, u32)> {
    let file = object::File::parse(object).ok()?;
    let section = file.section_by_name(&program.info.section())?;
    let symbol = file
        .symbols()
        .find(|s| s.section_index() == Some(section.index()) && s.name() == Ok(&program.name))?;
    let start = symbol.address() as usize;
    let len = if symbol.size() > 0 {
        symbol.size() as usize
    } else {
        section.size() as usize - start
    };
    let (name, offset) = if pc * 8 < len {
        (program.info.section(), start + pc * 8)
    } else {
        (".text".to_string(), pc * 8 - len)
    };
    let btf = file.section_by_name(".BTF")?.data().ok()?;
    let ext = file.section_by_name(".BTF.ext")?.data().ok()?;
    let u32_at =
        |b: &[u8], at: usize| Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?));
    let strings = u32_at(btf, 4)? as usize + u32_at(btf, 16)? as usize;
    let string = |off: u32| {
        let at = strings + off as usize;
        let bytes = btf.get(at..)?;
        let end = bytes.iter().position(|&b| b == 0)?;
        String::from_utf8(bytes[..end].to_vec()).ok()
    };
    // the line info: a record size, then for each section its name, the
    // number of records and the records (byte offset of the instruction,
    // file, line text, and `line << 10 | column`)
    let header = u32_at(ext, 4)? as usize;
    let mut at = header + u32_at(ext, 16)? as usize;
    let end = at + u32_at(ext, 20)? as usize;
    let record = u32_at(ext, at)? as usize;
    at += 4;
    let mut best: Option<(usize, u32, u32)> = None;
    while at + 8 <= end {
        let section_name = string(u32_at(ext, at)?)?;
        let count = u32_at(ext, at + 4)? as usize;
        at += 8;
        for i in 0..count {
            let r = at + i * record;
            let insn = u32_at(ext, r)? as usize;
            if section_name == name && insn <= offset && best.is_none_or(|(b, _, _)| insn >= b) {
                best = Some((insn, u32_at(ext, r + 4)?, u32_at(ext, r + 12)? >> 10));
            }
        }
        at += count * record;
    }
    let (_, file_name, line) = best?;
    Some((string(file_name)?, line))
}
