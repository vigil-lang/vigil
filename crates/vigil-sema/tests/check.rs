//! the type-checking cases of `tests/cases/`, and tests that the analysis
//! never panics.
//!
//! A line `# module: a.b` starts a new module in a case, the first one is
//! `main` by default. A case named `ok_*` must give no diagnostics, any other
//! is compared with a snapshot. A case with `kernel` in its name is skipped
//! when there is no BTF. A first line `# target: 5.3` sets the oldest target
//! kernel, Linux 6.1 by default.

use std::path::Path;

use vigil_diagnostics::{Diagnostics, Renderer, SourceMap};
use vigil_sema::ModuleInput;

/// the modules of a case, by name, with their text
fn split_modules(text: &str) -> Vec<(String, String)> {
    let mut modules = vec![("main".to_string(), String::new())];
    for line in text.split_inclusive('\n') {
        if let Some(name) = line.strip_prefix("# module: ") {
            let name = name.trim().to_string();
            if modules.len() == 1 && modules[0].1.trim().is_empty() {
                modules[0].0 = name;
            } else {
                modules.push((name, String::new()));
            }
            continue;
        }
        modules.last_mut().unwrap().1.push_str(line);
    }
    modules
}

/// the kernel's types, read once for all the tests (`None` when the machine
/// has no BTF)
fn btf() -> Option<std::sync::Arc<vigil_btf::Btf>> {
    static BTF: std::sync::OnceLock<Option<std::sync::Arc<vigil_btf::Btf>>> =
        std::sync::OnceLock::new();
    BTF.get_or_init(|| {
        vigil_btf::Btf::from_file(vigil_btf::VMLINUX)
            .ok()
            .map(std::sync::Arc::new)
    })
    .clone()
}

/// the rendered diagnostics of a case, first the parser's, then the
/// analysis'
fn check(text: &str) -> String {
    let mut sources = SourceMap::new();
    let mut diags = Diagnostics::new();
    let mut inputs = Vec::new();
    for (name, code) in split_modules(text) {
        let file = sources.add(format!("{}.vg", name.replace('.', "/")), code.clone());
        let parse = vigil_syntax::parse(&code);
        for error in parse.errors() {
            diags.push(error.to_diagnostic(file));
        }
        inputs.push(ModuleInput {
            file,
            path: name.split('.').map(String::from).collect(),
            tree: parse.tree(),
            std: false,
        });
    }
    for (path, name, code) in vigil_sema::stdlib::MODULES {
        let file = sources.add(*name, *code);
        inputs.push(ModuleInput {
            file,
            path: path.split('.').map(String::from).collect(),
            tree: vigil_syntax::parse(code).tree(),
            std: true,
        });
    }
    let mut options = vigil_sema::Options {
        btf: btf(),
        ..Default::default()
    };
    if let Some(version) = text
        .lines()
        .next()
        .and_then(|l| l.strip_prefix("# target: "))
    {
        options.kernel = version.trim().parse().unwrap();
    }
    let (_, sema_diags) = vigil_sema::analyze_with(&inputs, options);
    diags.extend(sema_diags);
    let mut out = Vec::new();
    Renderer::new(&sources, false)
        .render_all(&diags, &mut out)
        .unwrap();
    String::from_utf8(out).unwrap()
}

/// every case of `tests/cases/`, in the order of their names
#[test]
fn cases() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cases");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "vg"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty());
    for path in paths {
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(&path).unwrap();
        if name.contains("kernel") && btf().is_none() {
            eprintln!("skipping {name}: no kernel BTF");
            continue;
        }
        let output = check(&text);
        if name.starts_with("ok_") {
            assert_eq!(output, "", "{name} should type-check without diagnostics");
        } else {
            assert!(!output.is_empty(), "{name} should produce diagnostics");
            insta::assert_snapshot!(name, output);
        }
    }
}

/// the analysis must not panic on every line prefix of the cases and of
/// `examples/`
#[test]
fn never_panics_on_prefixes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dirs = [root.join("tests/cases"), root.join("../../examples")];
    for dir in dirs {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "vg") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let mut end = 0;
            for line in text.split_inclusive('\n') {
                end += line.len();
                let _ = check(&text[..end]);
            }
        }
    }
}

/// a small xorshift generator, so the mutations are the same for the same
/// seed. The state must not be zero.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// the tokens the mutations insert, the ones most likely to confuse the
/// checker
const PIECES: &[&str] = &[
    "def ",
    "struct ",
    "enum ",
    "protocol ",
    "map ",
    "event ",
    "match ",
    "case ",
    "if ",
    "elif ",
    "else:",
    "for ",
    "while ",
    "in ",
    "return ",
    "comptime ",
    "const ",
    "from ",
    "import ",
    "@xdp",
    "@kprobe(\"x\")",
    "@bounded(4)",
    "self",
    "ctx",
    "?",
    ":",
    "->",
    "=",
    "==",
    "+",
    "*",
    "<<",
    "(",
    ")",
    "[",
    "]",
    ",",
    ".",
    "0",
    "255",
    "-1",
    "u8",
    "u32",
    "i64",
    "bool",
    "q8_8",
    "Option[u32]",
    "T",
    "N",
    "range(",
    "print(f\"{",
    "}\")",
    "\n",
    "\n    ",
    "    ",
    "\t",
    "#",
    "\"",
    "b\"x\"",
    "None",
    "Some(",
    "_",
];

/// making one to four random edits to `text`
fn mutate(rng: &mut Rng, text: &str) -> String {
    let mut lines: Vec<String> = text.split_inclusive('\n').map(String::from).collect();
    for _ in 0..1 + rng.below(4) {
        if lines.is_empty() {
            lines.push(String::new());
        }
        let i = rng.below(lines.len());
        match rng.below(8) {
            // delete or duplicate a range of lines
            0 => {
                let end = (i + 1 + rng.below(6)).min(lines.len());
                lines.drain(i..end);
            }
            1 => {
                let end = (i + 1 + rng.below(6)).min(lines.len());
                let copy: Vec<String> = lines[i..end].to_vec();
                let at = rng.below(lines.len() + 1);
                lines.splice(at..at, copy);
            }
            // swap two lines
            2 => {
                let j = rng.below(lines.len());
                lines.swap(i, j);
            }
            // indent or dedent a line
            3 => lines[i].insert_str(0, "    "),
            4 => {
                let line = &lines[i];
                let trimmed = line.strip_prefix("    ").unwrap_or(line.trim_start());
                lines[i] = trimmed.to_string();
            }
            // insert a piece, or delete some characters, inside a line
            5 | 6 => {
                let line = &mut lines[i];
                let mut at = rng.below(line.len() + 1);
                while !line.is_char_boundary(at) {
                    at -= 1;
                }
                if rng.below(2) == 0 {
                    line.insert_str(at, PIECES[rng.below(PIECES.len())]);
                } else {
                    let mut end = (at + 1 + rng.below(8)).min(line.len());
                    while !line.is_char_boundary(end) {
                        end += 1;
                    }
                    line.replace_range(at..end, "");
                }
            }
            // cut the text short
            _ => lines.truncate(i),
        }
    }
    lines.concat()
}

/// random mutations of the examples, the cases and the standard library
///
/// Nothing may panic, and the formatter must keep the meaning and be
/// idempotent. `VIGIL_MUTATIONS` sets the number of inputs (300 by default)
/// and `VIGIL_SEED` the seed. A failure prints the seed and count to rerun
/// it.
#[test]
fn never_panics_on_mutations() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut corpus = Vec::new();
    for dir in [root.join("tests/cases"), root.join("../../examples")] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "vg") {
                corpus.push(std::fs::read_to_string(&path).unwrap());
            }
        }
    }
    corpus.sort();
    for (_, _, code) in vigil_sema::stdlib::MODULES {
        corpus.push(code.to_string());
    }
    let count: u64 = std::env::var("VIGIL_MUTATIONS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(300);
    let seed: u64 = std::env::var("VIGIL_SEED")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(0x5eed);
    for n in 0..count {
        let input = seed.wrapping_add(n);
        let mut rng = Rng(input.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let original = &corpus[rng.below(corpus.len())];
        let text = mutate(&mut rng, original);
        let result = std::panic::catch_unwind(|| {
            let _ = check(&text);
            match vigil_syntax::fmt::format(&text) {
                Ok(formatted) => {
                    let again = vigil_syntax::fmt::format(&formatted);
                    assert_eq!(
                        again.as_ref(),
                        Ok(&formatted),
                        "formatting is not idempotent"
                    );
                }
                Err(vigil_syntax::fmt::Error::Syntax) => {}
                Err(vigil_syntax::fmt::Error::Changed) => {
                    panic!("the formatter changed the program")
                }
            }
        });
        if result.is_err() {
            panic!("VIGIL_SEED={input} VIGIL_MUTATIONS=1 fails on:\n{text}");
        }
    }
}
