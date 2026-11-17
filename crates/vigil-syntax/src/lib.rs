//! this crate defines the lexer, the parser and the syntax trees of Vigil.
//! The entry points are:
//! - [`parse`]: from source text to a lossless `rowan` tree (its text is
//!   exactly the input, comments included) and the syntax errors
//! - [`ast`]: a typed view of that tree
//! - [`fn@pretty`]: printing a tree in canonical form
//! - [`fmt`]: formatting source text, keeping its comments
//!
//! Parsing never fails. What the parser could not understand goes in an
//! `ERROR` node and what it could not find is missing, so every accessor of
//! [`ast`] returns an `Option`.

pub mod ast;
pub mod fmt;
mod lexer;
mod parser;
mod pretty;
mod syntax_kind;

use std::fmt::Write;

use rowan::{GreenNode, NodeOrToken, TextRange};
use vigil_diagnostics::{Diagnostic, FileId, Span};

pub use ast::AstNode;
pub use pretty::pretty;
pub use syntax_kind::{SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken, VigilLanguage};

/// a syntax error: what is wrong and where
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxError {
    /// one of the syntax codes of `vigil_diagnostics::codes`
    pub code: &'static str,
    /// the headline of the diagnostic
    pub message: String,
    /// a short text shown at the error
    pub label: String,
    /// where the error is. Empty for something missing (see
    /// `Parser::current_range`)
    pub range: TextRange,
    /// another place involved and its label, such as the bracket that is
    /// not closed
    pub secondary: Option<(TextRange, String)>,
    /// a concrete fix, when the mistake is a common one
    pub help: Option<String>,
}

impl SyntaxError {
    pub(crate) fn new(
        code: &'static str,
        message: impl Into<String>,
        label: impl Into<String>,
        range: TextRange,
    ) -> Self {
        SyntaxError {
            code,
            message: message.into(),
            label: label.into(),
            range,
            secondary: None,
            help: None,
        }
    }

    pub(crate) fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    /// the error as a diagnostic of a file, labelled at its range
    pub fn to_diagnostic(&self, file: FileId) -> Diagnostic {
        let span = |range: TextRange| {
            Span::new(file, usize::from(range.start())..usize::from(range.end()))
        };
        let mut diag = Diagnostic::error(&self.message)
            .with_code(self.code)
            .with_primary(span(self.range), &self.label);
        if let Some((range, label)) = &self.secondary {
            diag = diag.with_secondary(span(*range), label);
        }
        if let Some(help) = &self.help {
            diag = diag.with_help(help);
        }
        diag
    }
}

/// the result of parsing: a tree (always, errors or not) and the errors in
/// source order. The tree is kept as a `rowan` green node, cheap to share
#[derive(Clone, Debug)]
pub struct Parse {
    green: GreenNode,
    errors: Vec<SyntaxError>,
}

/// parsing a source file
pub fn parse(text: &str) -> Parse {
    parse_with(text, parser::parse)
}

/// parsing a lone expression, such as an interpolation of a format string.
/// The root is still a `SOURCE_FILE`, with the expression as a child (see
/// [`Parse::expr`])
pub fn parse_expr(text: &str) -> Parse {
    let mut parse = parse_with(text, parser::parse_expression);
    // the caller moves the primary label to its place in the file, a
    // secondary one would stay at an offset of the interpolation text
    for error in &mut parse.errors {
        error.secondary = None;
    }
    parse
}

/// lexing `text` and running a parser entry point over the tokens
fn parse_with(
    text: &str,
    run: fn(&str, &[lexer::Token]) -> (GreenNode, Vec<SyntaxError>),
) -> Parse {
    let (tokens, mut errors) = lexer::tokenize(text);
    let (green, parse_errors) = run(text, &tokens);
    // a bad token gives an error from the lexer and another from the parser
    // within its range (for a bad indentation, at its end). We keep only the
    // lexer's, which says more
    let lexed: Vec<_> = errors.iter().map(|e| e.range).collect();
    errors.extend(parse_errors.into_iter().filter(|e| {
        let at = e.range.start();
        !lexed.iter().any(|l| l.start() <= at && at <= l.end())
    }));
    // stable sort, so the lexer's errors stay first at one position
    errors.sort_by_key(|e| e.range.start());
    Parse { green, errors }
}

impl Parse {
    /// the root of the concrete syntax tree
    pub fn syntax(&self) -> SyntaxNode {
        SyntaxNode::new_root(self.green.clone())
    }

    /// the root as a typed [`ast::SourceFile`]. The cast can't fail, both
    /// entry points make a `SOURCE_FILE` root
    pub fn tree(&self) -> ast::SourceFile {
        ast::SourceFile::cast(self.syntax()).expect("the root is a source file")
    }

    /// the expression of a [`parse_expr`] result, `None` if nothing starts
    /// one
    pub fn expr(&self) -> Option<ast::Expr> {
        self.syntax().children().find_map(ast::Expr::cast)
    }

    pub fn errors(&self) -> &[SyntaxError] {
        &self.errors
    }

    /// an indented dump of the tree, one node or token per line with kind and
    /// range, used by the snapshot tests. Trivia only if `trivia` is set
    pub fn debug_tree(&self, trivia: bool) -> String {
        let mut out = String::new();
        let mut depth = 0;
        for event in self.syntax().preorder_with_tokens() {
            match event {
                rowan::WalkEvent::Enter(element) => {
                    match &element {
                        NodeOrToken::Node(node) => {
                            // writing to a `String` can't fail
                            let _ = writeln!(
                                out,
                                "{:indent$}{:?}@{:?}",
                                "",
                                node.kind(),
                                node.text_range(),
                                indent = depth * 2
                            );
                        }
                        NodeOrToken::Token(token) if trivia || !token.kind().is_trivia() => {
                            let _ = writeln!(
                                out,
                                "{:indent$}{:?}@{:?} {:?}",
                                "",
                                token.kind(),
                                token.text_range(),
                                token.text(),
                                indent = depth * 2
                            );
                        }
                        NodeOrToken::Token(_) => {}
                    }
                    depth += 1;
                }
                rowan::WalkEvent::Leave(_) => depth -= 1,
            }
        }
        out
    }
}
