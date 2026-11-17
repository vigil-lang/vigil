//! this module builds the metadata of the generated module: debug info,
//! CO-RE relocations, alias tags and loop hints. [`Meta`] numbers the nodes
//! while functions are generated and [`Meta::write`] prints them at the
//! end. A node that refers to itself, or to a later node, is reserved first
//! and filled in later.
//!
//! The source text goes in the `DIFile`s because the BPF backend turns the
//! debug info into BTF line info, and the verifier prints those lines in its
//! log. A field of a kernel struct is read at an offset the loader relocates
//! (CO-RE). The relocation global's name depends on every member accessed,
//! so code refers to a placeholder until [`Meta::finish_kernel`].

use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;

use vigil_btf::Btf;
use vigil_diagnostics::{FileId, SourceMap, Span};

/// a field access in a kernel struct (CO-RE), see [`Meta::kernel_access`]
struct KernelAccess {
    /// the BTF id of the struct or union
    composite: u32,
    /// the BTF index of the member at each step
    path: Vec<u32>,
    /// the byte offset of the field in the kernel we build against
    offset: u32,
}

/// the metadata of a module being generated
pub(crate) struct Meta<'a> {
    sources: &'a SourceMap,
    btf: Option<&'a Btf>,
    kernel_accesses: Vec<KernelAccess>,
    /// the members accessed in each kernel struct or union. We describe only
    /// those, which is all the loader needs to relocate.
    kernel_members: HashMap<u32, BTreeSet<u32>>,
    /// the node of each kernel type described, `None` for `void`
    kernel_nodes: HashMap<u32, Option<usize>>,
    /// the last sequence number of `llvm.bpf.passthrough`
    passthroughs: u32,
    /// the last number given by [`Meta::unique`]
    unique: u32,
    /// the nodes, `!0` first
    nodes: Vec<String>,
    /// the `DIFile` of each source file
    files: HashMap<FileId, usize>,
    /// the offsets where the lines of each source file start
    lines: HashMap<FileId, Vec<usize>>,
    /// the shared `DILocation` of each scope, line and column
    locations: HashMap<(usize, u32, u32), usize>,
    /// the compile unit, once there is a function
    unit: Option<usize>,
    /// the node listing the global variables of the unit (the maps), filled
    /// by [`Meta::write`], and those variables
    globals_node: Option<usize>,
    globals: Vec<usize>,
    /// the subroutine types of internal functions, programs and callbacks,
    /// each made once
    internal_type: Option<usize>,
    program_type: Option<usize>,
    callback_type: Option<usize>,
    /// the alias tags: memory that may be in the packet, and memory that
    /// never is (see `Layout::packet_free`)
    tbaa: Option<(usize, usize)>,
}

impl<'a> Meta<'a> {
    pub fn new(sources: &'a SourceMap, btf: Option<&'a Btf>) -> Self {
        Meta {
            sources,
            btf,
            kernel_accesses: Vec::new(),
            kernel_members: HashMap::new(),
            kernel_nodes: HashMap::new(),
            passthroughs: 0,
            unique: 0,
            nodes: Vec::new(),
            files: HashMap::new(),
            lines: HashMap::new(),
            locations: HashMap::new(),
            unit: None,
            globals_node: None,
            globals: Vec::new(),
            internal_type: None,
            program_type: None,
            callback_type: None,
            tbaa: None,
        }
    }

    /// the alias tag of a load or store: with `packet`, for memory that may
    /// be in the packet, otherwise for memory that never is
    ///
    /// The two are siblings under one root, so LLVM takes accesses with
    /// different tags as different memory. An access without a tag may be
    /// to any memory, which is what we leave when we cannot tell.
    pub fn tbaa(&mut self, packet: bool) -> usize {
        let (in_packet, elsewhere) = *self.tbaa.get_or_insert_with(|| {
            let nodes = &mut self.nodes;
            let mut add = |node: String| {
                nodes.push(node);
                nodes.len() - 1
            };
            let root = add("!{!\"vigil memory\"}".into());
            let packet = add(format!("!{{!\"packet\", !{root}, i64 0}}"));
            let other = add(format!("!{{!\"not packet\", !{root}, i64 0}}"));
            (
                add(format!("!{{!{packet}, !{packet}, i64 0}}")),
                add(format!("!{{!{other}, !{other}, i64 0}}")),
            )
        });
        if packet { in_packet } else { elsewhere }
    }

    fn add(&mut self, node: String) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// a node to fill later, for one that refers to itself or to a later
    /// node
    fn reserve(&mut self) -> usize {
        self.add(String::new())
    }

    /// the `DIFile` of a source file, with its text embedded
    fn file(&mut self, file: FileId) -> usize {
        if let Some(&id) = self.files.get(&file) {
            return id;
        }
        let source = self.sources.get(file);
        let node = format!(
            "!DIFile(filename: \"{}\", directory: \"\", source: \"{}\")",
            escape(source.name()),
            escape(source.text())
        );
        let id = self.add(node);
        self.files.insert(file, id);
        id
    }

    /// the compile unit, made with the file of the first function. The list
    /// of its globals is reserved here and filled by [`Meta::write`], when
    /// every map is known.
    fn unit(&mut self, file: FileId) -> usize {
        if let Some(unit) = self.unit {
            return unit;
        }
        let file = self.file(file);
        let globals = self.reserve();
        let unit = self.add(format!(
            "distinct !DICompileUnit(language: DW_LANG_C99, file: !{file}, producer: \"vigil\", \
             isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !{globals})"
        ));
        self.unit = Some(unit);
        self.globals_node = Some(globals);
        unit
    }

    /// the debug info of a map in `.maps`, as libbpf declares them, or
    /// `None` before any function (no compile unit yet)
    ///
    /// It is a struct of pointers to arrays whose lengths are the values of
    /// `fields` (`__uint(type, BPF_MAP_TYPE_HASH)` is `int (*type)[1]`), and
    /// clang turns it into the BTF loaders read. The returned node is the
    /// one to attach to the global.
    pub fn map_definition(&mut self, name: &str, fields: &[(&str, u64)]) -> Option<usize> {
        let unit = self.unit?;
        let int = self.add("!DIBasicType(name: \"int\", size: 32, encoding: DW_ATE_signed)".into());
        let definition = self.reserve();
        let mut members = Vec::new();
        for (i, &(field, value)) in fields.iter().enumerate() {
            let range = self.add(format!("!DISubrange(count: {value})"));
            let ranges = self.add(format!("!{{!{range}}}"));
            let array = self.add(format!(
                "!DICompositeType(tag: DW_TAG_array_type, baseType: !{int}, size: {}, elements: !{ranges})",
                32 * value
            ));
            let pointer = self.add(format!(
                "!DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{array}, size: 64)"
            ));
            members.push(self.add(format!(
                "!DIDerivedType(tag: DW_TAG_member, name: \"{field}\", scope: !{definition}, \
                 baseType: !{pointer}, size: 64, offset: {})",
                64 * i
            )));
        }
        let list: Vec<String> = members.iter().map(|m| format!("!{m}")).collect();
        let elements = self.add(format!("!{{{}}}", list.join(", ")));
        self.nodes[definition] = format!(
            "distinct !DICompositeType(tag: DW_TAG_structure_type, size: {}, elements: !{elements})",
            64 * fields.len()
        );
        let variable = self.add(format!(
            "distinct !DIGlobalVariable(name: \"{}\", scope: !{unit}, type: !{definition}, \
             isLocal: false, isDefinition: true)",
            escape(name)
        ));
        let expression = self.add(format!(
            "!DIGlobalVariableExpression(var: !{variable}, expr: !DIExpression())"
        ));
        self.globals.push(expression);
        Some(expression)
    }

    /// line and column, both from 1, of the start of a span. The column
    /// counts bytes. The line starts of a file are found once, on its first
    /// span.
    pub fn line_col(&mut self, span: Span) -> (u32, u32) {
        let sources = self.sources;
        let starts = self.lines.entry(span.file).or_insert_with(|| {
            let text = sources.get(span.file).text();
            std::iter::once(0)
                .chain(text.match_indices('\n').map(|(i, _)| i + 1))
                .collect()
        });
        let offset = span.start as usize;
        let line = starts.partition_point(|&start| start <= offset);
        let column = offset - starts[line - 1] + 1;
        (line as u32, column as u32)
    }

    /// the subprogram of a function defined at `span`
    ///
    /// A program gets the signature `long (void *ctx)` for its BTF function
    /// info. An internal function is always inlined, so it gets an empty
    /// one: there is no function left in the object to describe.
    pub fn subprogram(&mut self, name: &str, span: Span, program: bool) -> usize {
        let ty = if program {
            match self.program_type {
                Some(ty) => ty,
                None => {
                    let long = self.add(
                        "!DIBasicType(name: \"long\", size: 64, encoding: DW_ATE_signed)".into(),
                    );
                    let pointer = self.add(
                        "!DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)".into(),
                    );
                    let types = self.add(format!("!{{!{long}, !{pointer}}}"));
                    let ty = self.add(format!("!DISubroutineType(types: !{types})"));
                    self.program_type = Some(ty);
                    ty
                }
            }
        } else {
            match self.internal_type {
                Some(ty) => ty,
                None => {
                    let types = self.add("!{null}".into());
                    let ty = self.add(format!("!DISubroutineType(types: !{types})"));
                    self.internal_type = Some(ty);
                    ty
                }
            }
        };
        let params: &[(&str, ParamTy)] = if program {
            &[("ctx", ParamTy::Pointer)]
        } else {
            &[]
        };
        self.subprogram_of_type(name, span, ty, program, params)
    }

    /// a subprogram of a given type, with its parameters named. The kernel
    /// requires the names in the BTF of program functions: aya fills in
    /// missing ones, libbpf does not.
    fn subprogram_of_type(
        &mut self,
        name: &str,
        span: Span,
        ty: usize,
        program: bool,
        params: &[(&str, ParamTy)],
    ) -> usize {
        let unit = self.unit(span.file);
        let file = self.file(span.file);
        let (line, _) = self.line_col(span);
        let flags = if program {
            "DISPFlagDefinition | DISPFlagOptimized"
        } else {
            "DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized"
        };
        let id = self.reserve();
        let mut variables = Vec::new();
        for (i, (param, kind)) in params.iter().enumerate() {
            let param_ty = self.param_type(*kind);
            variables.push(format!(
                "!{}",
                self.add(format!(
                    "!DILocalVariable(name: \"{param}\", arg: {}, scope: !{id}, file: !{file}, \
                     line: {line}, type: !{param_ty})",
                    i + 1
                ))
            ));
        }
        let retained = if variables.is_empty() {
            String::new()
        } else {
            format!(
                ", retainedNodes: !{}",
                self.add(format!("!{{{}}}", variables.join(", ")))
            )
        };
        self.nodes[id] = format!(
            "distinct !DISubprogram(name: \"{}\", scope: !{file}, file: !{file}, line: {line}, \
             type: !{ty}, scopeLine: {line}, spFlags: {flags}, unit: !{unit}{retained})",
            escape(name)
        );
        id
    }

    /// the debug type of a parameter
    fn param_type(&mut self, kind: ParamTy) -> usize {
        match kind {
            ParamTy::Long => {
                self.add("!DIBasicType(name: \"long\", size: 64, encoding: DW_ATE_signed)".into())
            }
            ParamTy::Pointer => self
                .add("!DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)".into()),
        }
    }

    /// the `DILocation` of the start of a span in a scope, one node per
    /// scope, line and column
    pub fn location(&mut self, scope: usize, span: Span) -> usize {
        let (line, column) = self.line_col(span);
        if let Some(&id) = self.locations.get(&(scope, line, column)) {
            return id;
        }
        let id = self.add(format!(
            "!DILocation(line: {line}, column: {column}, scope: !{scope})"
        ));
        self.locations.insert((scope, line, column), id);
        id
    }

    /// loop metadata asking for full unrolling. Distinct and self-referring,
    /// as LLVM requires of loop IDs.
    pub fn unroll(&mut self) -> usize {
        let full = self.add("!{!\"llvm.loop.unroll.full\"}".into());
        let id = self.reserve();
        self.nodes[id] = format!("distinct !{{!{id}, !{full}}}");
        id
    }

    /// loop metadata forbidding unrolling
    pub fn no_unroll(&mut self) -> usize {
        let disable = self.add("!{!\"llvm.loop.unroll.disable\"}".into());
        let id = self.reserve();
        self.nodes[id] = format!("distinct !{{!{id}, !{disable}}}");
        id
    }

    /// a number for names that must be unique in the module
    pub fn unique(&mut self) -> u32 {
        self.unique += 1;
        self.unique
    }

    /// the subprogram of a `bpf_loop` callback, `long (long index, void
    /// *ctx)`. Unlike an internal function, a callback stays a function of
    /// the object, so its BTF describes it.
    pub fn callback_subprogram(&mut self, name: &str, span: Span) -> usize {
        let ty = match self.callback_type {
            Some(ty) => ty,
            None => {
                let long = self
                    .add("!DIBasicType(name: \"long\", size: 64, encoding: DW_ATE_signed)".into());
                let pointer = self.add(
                    "!DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)".into(),
                );
                let types = self.add(format!("!{{!{long}, !{long}, !{pointer}}}"));
                let ty = self.add(format!("!DISubroutineType(types: !{types})"));
                self.callback_type = Some(ty);
                ty
            }
        };
        self.subprogram_of_type(
            name,
            span,
            ty,
            false,
            &[("index", ParamTy::Long), ("ctx", ParamTy::Pointer)],
        )
    }

    /// a new sequence number for `llvm.bpf.passthrough`, which keeps
    /// optimizations from merging or moving what it guards
    pub fn passthrough(&mut self) -> u32 {
        self.passthroughs += 1;
        self.passthroughs
    }

    /// the name of the global holding the offset of a field of a kernel
    /// struct, recording the access. It is a placeholder until
    /// [`Self::finish_kernel`] knows every member accessed.
    pub fn kernel_access(&mut self, composite: u32, path: &[u32], offset: u32) -> String {
        let index = match self
            .kernel_accesses
            .iter()
            .position(|a| a.composite == composite && a.path == path)
        {
            Some(index) => index,
            None => {
                let btf = self
                    .btf
                    .expect("kernel accesses come with the kernel's BTF");
                let mut current = composite;
                for (step, &member) in path.iter().enumerate() {
                    self.kernel_members
                        .entry(current)
                        .or_default()
                        .insert(member);
                    if step + 1 < path.len()
                        && let Some((_, c)) = btf.composite(current)
                        && let Some(m) = c.members.get(member as usize)
                        && let Some((next, _)) = btf.composite(m.ty)
                    {
                        current = next;
                    }
                }
                self.kernel_accesses.push(KernelAccess {
                    composite,
                    path: path.to_vec(),
                    offset,
                });
                self.kernel_accesses.len() - 1
            }
        };
        format!("@\"vigil.core.{index}\"")
    }

    /// the definitions of the relocation globals, and each placeholder with
    /// the name that replaces it. It also describes the kernel types
    /// accessed.
    ///
    /// A global is named like `@"llvm.task_struct:0:2400$0:85"`: the struct,
    /// the relocation kind (0, the byte offset of a field), the offset in
    /// the kernel we build against, and the access path, as positions among
    /// the members described.
    pub fn finish_kernel(&mut self) -> (String, Vec<(String, String)>) {
        let Some(btf) = self.btf else {
            return (String::new(), Vec::new());
        };
        let mut globals = String::new();
        let mut replacements = Vec::new();
        let accesses = std::mem::take(&mut self.kernel_accesses);
        for (index, access) in accesses.iter().enumerate() {
            let node = self.kernel_node(access.composite);
            let mut indices = vec!["0".to_string()];
            let mut current = access.composite;
            for &member in &access.path {
                let position = self.kernel_members[&current]
                    .iter()
                    .position(|&m| m == member)
                    .unwrap_or(0);
                indices.push(position.to_string());
                if let Some((_, c)) = btf.composite(current)
                    && let Some(m) = c.members.get(member as usize)
                    && let Some((next, _)) = btf.composite(m.ty)
                {
                    current = next;
                }
            }
            let struct_name = match btf.composite(access.composite) {
                Some((_, c)) => c.name.clone().unwrap_or_default(),
                None => String::new(),
            };
            let name = format!(
                "llvm.{struct_name}:0:{}${}",
                access.offset,
                indices.join(":")
            );
            if let Some(node) = node {
                let _ = writeln!(
                    globals,
                    "@\"{name}\" = external global i64, !llvm.preserve.access.index !{node} #2"
                );
            }
            replacements.push((format!("@\"vigil.core.{index}\""), format!("@\"{name}\"")));
        }
        (globals, replacements)
    }

    /// the debug info of a kernel type, `None` for `void` and for what is
    /// not described
    ///
    /// Typedefs and qualifiers are resolved away, a forward declaration
    /// stands for its complete type, and an enum is an unsigned integer of
    /// its size. A struct or union lists only the members accessed.
    fn kernel_node(&mut self, id: u32) -> Option<usize> {
        let btf = self.btf?;
        let id = btf.resolve(id);
        if let Some(&node) = self.kernel_nodes.get(&id) {
            return node;
        }
        let node = match btf.get(id).clone() {
            vigil_btf::Type::Int {
                name,
                size,
                signed,
                boolean,
            } => {
                let encoding = match (boolean, signed, size, name.contains("char")) {
                    (true, ..) => "DW_ATE_boolean",
                    (_, true, 1, true) => "DW_ATE_signed_char",
                    (_, false, 1, true) => "DW_ATE_unsigned_char",
                    (_, true, ..) => "DW_ATE_signed",
                    _ => "DW_ATE_unsigned",
                };
                Some(self.add(format!(
                    "!DIBasicType(name: \"{}\", size: {}, encoding: {encoding})",
                    escape(&name),
                    size * 8
                )))
            }
            vigil_btf::Type::Enum { size, .. } => Some(self.add(format!(
                "!DIBasicType(name: \"unsigned int\", size: {}, encoding: DW_ATE_unsigned)",
                size * 8
            ))),
            vigil_btf::Type::Ptr(target) => {
                // reserved and recorded first, because the pointed type may
                // lead back to this pointer
                let node = self.reserve();
                self.kernel_nodes.insert(id, Some(node));
                let base = self
                    .kernel_node(target)
                    .map_or("null".to_string(), |n| format!("!{n}"));
                self.nodes[node] =
                    format!("!DIDerivedType(tag: DW_TAG_pointer_type, baseType: {base}, size: 64)");
                Some(node)
            }
            vigil_btf::Type::Array { element, len } => {
                let base = self
                    .kernel_node(element)
                    .map_or("null".to_string(), |n| format!("!{n}"));
                let range = self.add(format!("!DISubrange(count: {len})"));
                let elements = self.add(format!("!{{!{range}}}"));
                Some(self.add(format!(
                    "!DICompositeType(tag: DW_TAG_array_type, baseType: {base}, size: {}, elements: !{elements})",
                    btf.size(id) * 8
                )))
            }
            vigil_btf::Type::Struct(_)
            | vigil_btf::Type::Union(_)
            | vigil_btf::Type::Fwd { .. } => {
                let (complete, composite) = btf.composite(id)?;
                if complete != id {
                    let node = self.kernel_node(complete);
                    self.kernel_nodes.insert(id, node);
                    return node;
                }
                let composite = composite.clone();
                let node = self.reserve();
                self.kernel_nodes.insert(id, Some(node));
                let members: Vec<u32> = self
                    .kernel_members
                    .get(&id)
                    .map(|m| m.iter().copied().collect())
                    .unwrap_or_default();
                let mut elements = Vec::new();
                for index in members {
                    let member = &composite.members[index as usize];
                    let base = self
                        .kernel_node(member.ty)
                        .map_or("null".to_string(), |n| format!("!{n}"));
                    let name = member
                        .name
                        .as_ref()
                        .map(|n| format!("name: \"{}\", ", escape(n)))
                        .unwrap_or_default();
                    let element = self.add(format!(
                        "!DIDerivedType(tag: DW_TAG_member, {name}scope: !{node}, baseType: {base}, size: {}, offset: {})",
                        btf.size(member.ty) * 8,
                        member.bit_offset
                    ));
                    elements.push(format!("!{element}"));
                }
                let list = self.add(format!("!{{{}}}", elements.join(", ")));
                let tag = if btf.is_union(id) {
                    "DW_TAG_union_type"
                } else {
                    "DW_TAG_structure_type"
                };
                let name = composite
                    .name
                    .as_ref()
                    .map(|n| format!("name: \"{}\", ", escape(n)))
                    .unwrap_or_default();
                self.nodes[node] = format!(
                    "distinct !DICompositeType(tag: {tag}, {name}size: {}, elements: !{list})",
                    composite.size * 8
                );
                Some(node)
            }
            _ => None,
        };
        self.kernel_nodes.insert(id, node);
        node
    }

    /// writing the named metadata and every node, at the end of the module
    pub fn write(&self, out: &mut String) {
        let flags_dwarf = self.nodes.len();
        let flags_debug = flags_dwarf + 1;
        if let Some(unit) = self.unit {
            let _ = writeln!(out, "!llvm.dbg.cu = !{{!{unit}}}");
        }
        let _ = writeln!(
            out,
            "!llvm.module.flags = !{{!{flags_dwarf}, !{flags_debug}}}"
        );
        for (id, node) in self.nodes.iter().enumerate() {
            if Some(id) == self.globals_node {
                let list: Vec<String> = self.globals.iter().map(|g| format!("!{g}")).collect();
                let _ = writeln!(out, "!{id} = !{{{}}}", list.join(", "));
                continue;
            }
            let _ = writeln!(out, "!{id} = {node}");
        }
        // DWARF 5, the version with the sources embedded in the files
        let _ = writeln!(
            out,
            "!{flags_dwarf} = !{{i32 7, !\"Dwarf Version\", i32 5}}"
        );
        let _ = writeln!(
            out,
            "!{flags_debug} = !{{i32 2, !\"Debug Info Version\", i32 3}}"
        );
    }
}

/// escaping a string for LLVM metadata and IR: printable ASCII other than
/// `"` and `\` stays, every other byte becomes `\XX`
pub(crate) fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        if (0x20..0x7f).contains(&byte) && byte != b'"' && byte != b'\\' {
            out.push(byte as char);
        } else {
            let _ = write!(out, "\\{byte:02X}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_quotes_and_control_characters() {
        assert_eq!(escape("a\"b\\c\nd"), "a\\22b\\5Cc\\0Ad");
        assert_eq!(escape("é"), "\\C3\\A9");
    }

    #[test]
    fn computes_lines_and_columns() {
        let mut sources = SourceMap::new();
        let file = sources.add("a.vg", "ab\ncd\n");
        let mut meta = Meta::new(&sources, None);
        assert_eq!(meta.line_col(Span::new(file, 0..1)), (1, 1));
        assert_eq!(meta.line_col(Span::new(file, 4..5)), (2, 2));
    }
}

/// the types of parameters in the debug info
#[derive(Clone, Copy)]
enum ParamTy {
    /// `long`
    Long,
    /// `void *`
    Pointer,
}
