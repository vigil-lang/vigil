//! tests on the programs of `examples/`: they parse without errors, with
//! snapshots of trees and canonical forms, and the parser stays lossless
//! and doesn't panic on every prefix and on many one-character deletions.

use std::path::{Path, PathBuf};

use vigil_syntax::{parse, pretty};

/// name and text of every `.vg` file of `compiler/examples/`, sorted by name
fn examples() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("examples directory")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "vg"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no examples in {}", dir.display());
    paths
        .into_iter()
        .map(|p| {
            let name = p.file_stem().unwrap().to_string_lossy().into_owned();
            (name, std::fs::read_to_string(&p).unwrap())
        })
        .collect()
}

#[test]
fn examples_parse_without_errors() {
    for (name, text) in examples() {
        let parse = parse(&text);
        assert!(parse.errors().is_empty(), "{name}: {:#?}", parse.errors());
        assert_eq!(parse.syntax().to_string(), text, "{name} is not lossless");
        insta::assert_snapshot!(format!("{name}.tree"), parse.debug_tree(false));
    }
}

#[test]
fn pretty_printing_is_canonical() {
    for (name, text) in examples() {
        let printed = pretty(&parse(&text).tree());
        insta::assert_snapshot!(format!("{name}.pretty"), printed);

        let reparsed = parse(&printed);
        assert!(
            reparsed.errors().is_empty(),
            "{name}: {:#?}",
            reparsed.errors()
        );
        assert_eq!(pretty(&reparsed.tree()), printed, "{name}: not a fixpoint");
    }
}

#[test]
fn every_prefix_parses_losslessly() {
    for (name, text) in examples() {
        for (cut, _) in text.char_indices() {
            let prefix = &text[..cut];
            let parse = parse(prefix);
            assert_eq!(parse.syntax().to_string(), prefix, "{name} cut at {cut}");
            // the typed view must not panic on the broken tree either
            let _ = pretty(&parse.tree());
        }
    }
}

#[test]
fn deleting_characters_never_breaks_the_parser() {
    for (name, text) in examples() {
        // one character in three, enough to hit every kind of token and
        // indentation, and much faster than all of them
        for (cut, c) in text.char_indices().step_by(3) {
            let mutated = format!("{}{}", &text[..cut], &text[cut + c.len_utf8()..]);
            let parse = parse(&mutated);
            assert_eq!(
                parse.syntax().to_string(),
                mutated,
                "{name} without byte {cut}"
            );
            let _ = pretty(&parse.tree());
        }
    }
}
