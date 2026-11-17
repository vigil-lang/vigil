//! this module writes the programs of the certified backend as a
//! relocatable ELF object: the code sections, `license` and a symbol table,
//! which is all aya or libbpf need. Core has no maps, so there is no BTF and
//! there are no relocations.

const EM_BPF: u16 = 247;
const SHT_PROGBITS: u32 = 1;
const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHF_WRITE: u64 = 1;
const SHF_ALLOC: u64 = 2;
const SHF_EXECINSTR: u64 = 4;
const STB_GLOBAL: u8 = 1;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;

/// a section header, before writing it
struct Section {
    name: u32,
    kind: u32,
    flags: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    align: u64,
    entsize: u64,
}

/// a string table of NUL ended names, starting with the empty name ELF
/// requires
#[derive(Default)]
struct Strings(Vec<u8>);

impl Strings {
    /// adding `s` and returning its offset. There is no sharing, a name added
    /// twice is stored twice
    fn add(&mut self, s: &str) -> u32 {
        if self.0.is_empty() {
            self.0.push(0);
        }
        let at = self.0.len() as u32;
        self.0.extend_from_slice(s.as_bytes());
        self.0.push(0);
        at
    }
}

/// writing a 24-byte ELF64 symbol table entry
fn symbol(out: &mut Vec<u8>, name: u32, info: u8, shndx: u16, value: u64, size: u64) {
    out.extend_from_slice(&name.to_le_bytes());
    out.push(info);
    out.push(0);
    out.extend_from_slice(&shndx.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
}

/// the object of one program `name` in `section` (like `xdp`)
pub fn object(section: &str, name: &str, code: &[u8]) -> Vec<u8> {
    objects(&[(section, name, code)])
}

/// the object of several `(section, name, code)` programs
///
/// programs of the same section go one after the other, as LLVM does, and
/// each symbol gives start and size. Sections follow the order of their
/// first program.
pub fn objects(programs: &[(&str, &str, &[u8])]) -> Vec<u8> {
    let mut shstrtab = Strings::default();
    let mut strtab = Strings::default();
    let license = b"GPL\0";

    // the code of each section, and the name, offset and size of its programs
    type Symbols<'a> = Vec<(&'a str, u64, u64)>;
    let mut code_sections: Vec<(&str, Vec<u8>, Symbols)> = Vec::new();
    for &(section, name, code) in programs {
        let index = match code_sections.iter().position(|(s, ..)| *s == section) {
            Some(i) => i,
            None => {
                code_sections.push((section, Vec::new(), Vec::new()));
                code_sections.len() - 1
            }
        };
        let (_, bytes, symbols) = &mut code_sections[index];
        symbols.push((name, bytes.len() as u64, code.len() as u64));
        bytes.extend_from_slice(code);
    }
    let n = code_sections.len();
    // section indices: 0 is null, code from 1, then `license`, `.symtab`,
    // `.strtab` and `.shstrtab`
    let license_index = n + 1;
    let strtab_index = n + 3;

    // symbol 0 is null, as ELF requires. All the others are global, so the
    // first global is 1 below
    let mut symtab = Vec::new();
    symbol(&mut symtab, 0, 0, 0, 0, 0);
    for (i, (_, _, symbols)) in code_sections.iter().enumerate() {
        for &(name, offset, size) in symbols {
            let program = strtab.add(name);
            symbol(
                &mut symtab,
                program,
                (STB_GLOBAL << 4) | STT_FUNC,
                (i + 1) as u16,
                offset,
                size,
            );
        }
    }
    let license_sym = strtab.add("_license");
    symbol(
        &mut symtab,
        license_sym,
        (STB_GLOBAL << 4) | STT_OBJECT,
        license_index as u16,
        0,
        4,
    );

    let code_names: Vec<u32> = code_sections
        .iter()
        .map(|(section, ..)| shstrtab.add(section))
        .collect();
    let names = [
        shstrtab.add("license"),
        shstrtab.add(".symtab"),
        shstrtab.add(".strtab"),
        shstrtab.add(".shstrtab"),
    ];

    // we write the header last over these 64 bytes, when we know where the
    // section headers are. `place` writes data aligned and gives its offset
    let mut out = vec![0u8; 64];
    let place = |out: &mut Vec<u8>, data: &[u8], align: usize| {
        while !out.len().is_multiple_of(align) {
            out.push(0);
        }
        let at = out.len() as u64;
        out.extend_from_slice(data);
        at
    };
    let code_at: Vec<u64> = code_sections
        .iter()
        .map(|(_, bytes, _)| place(&mut out, bytes, 8))
        .collect();
    let license_at = place(&mut out, license, 1);
    let symtab_at = place(&mut out, &symtab, 8);
    let strtab_at = place(&mut out, &strtab.0, 1);
    let shstrtab_at = place(&mut out, &shstrtab.0, 1);

    let mut sections: Vec<Section> = code_sections
        .iter()
        .zip(code_names.iter().zip(&code_at))
        .map(|((_, bytes, _), (&name, &offset))| Section {
            name,
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            offset,
            size: bytes.len() as u64,
            link: 0,
            info: 0,
            align: 8,
            entsize: 0,
        })
        .collect();
    sections.extend([
        Section {
            name: names[0],
            kind: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_WRITE,
            offset: license_at,
            size: license.len() as u64,
            link: 0,
            info: 0,
            align: 1,
            entsize: 0,
        },
        Section {
            name: names[1],
            kind: SHT_SYMTAB,
            flags: 0,
            offset: symtab_at,
            size: symtab.len() as u64,
            // the string table of the names and the first global symbol
            link: strtab_index as u32,
            info: 1,
            align: 8,
            entsize: 24,
        },
        Section {
            name: names[2],
            kind: SHT_STRTAB,
            flags: 0,
            offset: strtab_at,
            size: strtab.0.len() as u64,
            link: 0,
            info: 0,
            align: 1,
            entsize: 0,
        },
        Section {
            name: names[3],
            kind: SHT_STRTAB,
            flags: 0,
            offset: shstrtab_at,
            size: shstrtab.0.len() as u64,
            link: 0,
            info: 0,
            align: 1,
            entsize: 0,
        },
    ]);
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    // the section headers, after the data
    let headers_at = out.len() as u64;
    out.extend_from_slice(&[0u8; 64]); // the null section
    for s in &sections {
        out.extend_from_slice(&s.name.to_le_bytes());
        out.extend_from_slice(&s.kind.to_le_bytes());
        out.extend_from_slice(&s.flags.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&s.offset.to_le_bytes());
        out.extend_from_slice(&s.size.to_le_bytes());
        out.extend_from_slice(&s.link.to_le_bytes());
        out.extend_from_slice(&s.info.to_le_bytes());
        out.extend_from_slice(&s.align.to_le_bytes());
        out.extend_from_slice(&s.entsize.to_le_bytes());
    }

    // the ELF header: 64-bit, little-endian, relocatable, BPF, no entry point
    // and no program headers
    let mut header = Vec::with_capacity(64);
    header.extend_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0]);
    header.extend_from_slice(&[0; 8]);
    header.extend_from_slice(&1u16.to_le_bytes()); // ET_REL
    header.extend_from_slice(&EM_BPF.to_le_bytes());
    header.extend_from_slice(&1u32.to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes()); // entry
    header.extend_from_slice(&0u64.to_le_bytes()); // program headers
    header.extend_from_slice(&headers_at.to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes()); // flags
    header.extend_from_slice(&64u16.to_le_bytes());
    header.extend_from_slice(&0u16.to_le_bytes());
    header.extend_from_slice(&0u16.to_le_bytes());
    header.extend_from_slice(&64u16.to_le_bytes());
    header.extend_from_slice(&(sections.len() as u16 + 1).to_le_bytes());
    // the index of `.shstrtab`, the last section
    header.extend_from_slice(&(sections.len() as u16).to_le_bytes());
    out[..64].copy_from_slice(&header);
    out
}

#[cfg(test)]
mod tests {
    /// checking the object with `readelf`, if it is installed
    #[test]
    fn readelf_understands_the_object() {
        // `r0 = 2; exit`, passing every packet
        let code = [0xb7, 0, 0, 0, 2, 0, 0, 0, 0x95, 0, 0, 0, 0, 0, 0, 0];
        let object = super::object("xdp", "main", &code);
        let path = std::env::temp_dir().join(format!("vigil-elf-{}.o", std::process::id()));
        std::fs::write(&path, &object).unwrap();
        let Ok(out) = std::process::Command::new("readelf")
            .args(["-S", "-s"])
            .arg(&path)
            .output()
        else {
            return;
        };
        let _ = std::fs::remove_file(&path);
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{text}");
        assert!(text.contains("xdp") && text.contains("license"), "{text}");
        assert!(text.contains("FUNC    GLOBAL DEFAULT    1 main"), "{text}");
    }

    /// two programs of the same section, with another one between them, end
    /// up together in their section
    #[test]
    fn programs_of_one_section_share_it() {
        let code = [0xb7, 0, 0, 0, 2, 0, 0, 0, 0x95, 0, 0, 0, 0, 0, 0, 0];
        let object = super::objects(&[
            ("xdp", "first", &code),
            ("tc", "shaper", &code),
            ("xdp", "second", &code),
        ]);
        use object::{Object, ObjectSection, ObjectSymbol};
        let file = object::File::parse(&*object).unwrap();
        let xdp = file.section_by_name("xdp").unwrap();
        assert_eq!(xdp.size(), 32);
        assert_eq!(file.section_by_name("tc").unwrap().size(), 16);
        let second = file.symbols().find(|s| s.name() == Ok("second")).unwrap();
        assert_eq!(
            (second.section_index(), second.address()),
            (Some(xdp.index()), 16)
        );
        assert!(file.section_by_name("license").is_some());
    }
}
