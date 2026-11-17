//! small tests of the grammar and its error recovery. A snapshot has the
//! tree without trivia and then the rendered errors. Every case also checks
//! that the tree is lossless.

use vigil_diagnostics::{Diagnostics, Renderer, SourceMap};
use vigil_syntax::parse;

/// the tree of a text without trivia, and its rendered errors if any
fn check(text: &str) -> String {
    let parse = parse(text);
    assert_eq!(parse.syntax().to_string(), text, "not lossless");
    let mut sources = SourceMap::new();
    let file = sources.add("test.vg", text);
    let mut diags = Diagnostics::new();
    for error in parse.errors() {
        diags.push(error.to_diagnostic(file));
    }
    let mut out = parse.debug_tree(false);
    if !diags.is_empty() {
        out.push_str("\n--- errors ---\n");
        let mut rendered = Vec::new();
        Renderer::new(&sources, false)
            .render_all(&diags, &mut rendered)
            .unwrap();
        out.push_str(&String::from_utf8(rendered).unwrap());
    }
    out
}

#[test]
fn operator_precedence() {
    insta::assert_snapshot!(check(
        "def f():\n    x = not a == b and c or -d.e(1)[2]? * 3 + 4 << 5 & 6 | 7 ^ 8 // 9\n"
    ));
}

#[test]
fn assignments() {
    insta::assert_snapshot!(check(
        "def f():\n    x: u32 = 1\n    y: u8\n    x <<= 2\n    a.b[0] = x\n"
    ));
}

#[test]
fn generic_arguments_are_types_or_constants() {
    insta::assert_snapshot!(check(
        "map m: Array[Mlp[q8_8, N, 2 * N], 1](pinned=True)\ntype T = Option[u32]?\n"
    ));
}

#[test]
fn imports() {
    insta::assert_snapshot!(check(
        "import std.ml as ml\nfrom std.net import (\n    EthHdr,\n    Ipv4Hdr as Ip,\n)\n"
    ));
}

#[test]
fn definitions_with_members() {
    insta::assert_snapshot!(check(
        "protocol Model[T]:\n    def predict(self, x: T) -> u32: ...\n\n\
         enum E:\n    A\n    B(u8, u16)\n\nstruct S(Model): pass\n"
    ));
}

#[test]
fn control_flow() {
    insta::assert_snapshot!(check(
        "def f(n: u32) -> u32:\n    @bounded(16)\n    while n > 0:\n        n -= 1\n        if n == 3: break\n    \
         for i in range(4):\n        continue\n    comptime:\n        pass\n    match n:\n        \
         case Some(x) if x > 1:\n            return x\n        case E.A | -1:\n            pass\n    return 0\n"
    ));
}

#[test]
fn contextual_keywords_are_names_in_statements() {
    insta::assert_snapshot!(check("def f():\n    map = 1\n    type = map\n"));
}

#[test]
fn recovers_from_a_missing_colon() {
    insta::assert_snapshot!(check("def f()\n    return 1\n\ndef g():\n    return 2\n"));
}

#[test]
fn recovers_from_a_bad_statement() {
    insta::assert_snapshot!(check("def f():\n    x = \n    ) y\n    return 1\n"));
}

#[test]
fn recovers_from_garbage_items() {
    insta::assert_snapshot!(check("42 + 1\n    indented\n$\ndef g(): pass\n"));
}

#[test]
fn reports_layout_errors() {
    insta::assert_snapshot!(check("def f():\n    if x:\n        y\n      z\n\tw\n"));
}

#[test]
fn reports_an_unterminated_string() {
    insta::assert_snapshot!(check("def f():\n    print(\"abc)\n"));
}

/// the rendered errors of each text, for the tests of the messages
fn errors(texts: &[&str]) -> String {
    let mut out = String::new();
    for text in texts {
        let parse = parse(text);
        let mut sources = SourceMap::new();
        let file = sources.add("test.vg", *text);
        let mut diags = Diagnostics::new();
        for error in parse.errors() {
            diags.push(error.to_diagnostic(file));
        }
        let mut rendered = Vec::new();
        Renderer::new(&sources, false)
            .render_all(&diags, &mut rendered)
            .unwrap();
        out.push_str(&String::from_utf8(rendered).unwrap());
    }
    out
}

/// one error for a bad list, with the opening bracket when the error is on
/// a later line
#[test]
fn reports_bad_lists_once() {
    insta::assert_snapshot!(errors(&[
        "def f(x u32, y: u32) -> u32:\n    return x\n",
        "def f(x: u32, y: u32 -> u32:\n    return x\n",
        "def f(x: u32) -> u32:\n    return g(x y, 1)\n",
        "def f(x: u32) -> u32:\n    y = [1, 2\n    return y\n",
        "def f(x: u32) -> u32:\n    y = (x + 1\n    return y\n",
        "struct P:\n    x u32\n    y: u32\n",
        "@xdp(\ndef f(x: u32) -> u32:\n    return x\n",
        "def f(x: u32) -> u32:\n    return x)\n",
    ]));
}

/// a header without its `:` gives one error, and its block is still parsed
#[test]
fn recovers_from_a_bad_header() {
    insta::assert_snapshot!(errors(&[
        "def f(x: u32) -> u32:\n    while x < 3 do\n        x += 1\n    return x\n",
        "def f(x: u32) -> u32:\n    if x = 1:\n        return 0\n    return x\n",
    ]));
}

/// helps for what other languages write differently
#[test]
fn explains_syntax_of_other_languages() {
    insta::assert_snapshot!(errors(&[
        "x = 3\n",
        "fn f() -> u32:\n    return 1\n",
        "def f(a: bool, b: bool) -> bool:\n    return a && b\n",
        "def f(x: u32):\n    x++\n",
        "def f(x: bool) -> bool:\n    return !x\n",
        "def f(x: u32):\n    let y = x\n",
        "def f(x: u32):\n    if x:\n        pass\n    else if x > 1:\n        pass\n",
        "def f(x: u32):\n    elif x:\n        pass\n",
        "def f():\n    for (i = 0; i < 3; i += 1):\n        pass\n",
        "def f():\n    while True {\n        pass\n    }\n",
    ]));
}

#[test]
fn parses_lone_expressions() {
    let parse = vigil_syntax::parse_expr("pid >> 32");
    assert!(parse.errors().is_empty());
    assert!(matches!(
        parse.expr(),
        Some(vigil_syntax::ast::Expr::BinExpr(_))
    ));
    let parse = vigil_syntax::parse_expr("a b");
    assert_eq!(parse.errors().len(), 1);
    assert_eq!(parse.syntax().to_string(), "a b");
}

/// deciding if a generic argument is a type looks ahead to its `]`, and a
/// long scan must not look like a stuck parser. `long_brackets.vg` has
/// brackets open over a whole file
#[test]
fn long_generic_arguments_do_not_exhaust_the_parser() {
    let ones = vec!["1"; 300].join(", ");
    let parse = vigil_syntax::parse(&format!("type A = Foo[Bar[{ones}]]\n"));
    assert!(parse.errors().is_empty(), "{:?}", parse.errors());
    let _ = vigil_syntax::parse(include_str!("long_brackets.vg"));
}
