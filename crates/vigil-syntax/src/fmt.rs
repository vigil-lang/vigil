//! this module implements the formatter (`vigil fmt`). It works on the
//! tokens, not the tree, to keep comments and blank lines, with the spacing
//! of [`fn@crate::pretty`]. A line written over several lines is kept as
//! written, and a one-line line longer than [`WIDTH`] is broken at its
//! widest brackets, one item per line.
//!
//! Working on tokens could change the meaning of a program, so we check the
//! result: it must parse, print the same as the source and keep every
//! comment. If it fails, we try again without breaking lines.
//!
//! The result keeps the line endings of the source: a file whose first line
//! ends in `\r\n` (as on Windows) is written with `\r\n` throughout, any
//! other with `\n`.

use crate::lexer::{Token, tokenize};
use crate::syntax_kind::SyntaxKind::{self, *};

/// why a source was not formatted
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// the source has syntax errors
    Syntax,
    /// the result would mean something else or lose a comment (a formatter
    /// bug)
    Changed,
}

/// the longest line left on one line, in bytes, indentation included
pub const WIDTH: usize = 100;

/// formatting `text`. A source with syntax errors is not formatted, because
/// we couldn't check the result without a tree
pub fn format(text: &str) -> Result<String, Error> {
    let crlf = text
        .find('\n')
        .is_some_and(|end| text[..end].ends_with('\r'));
    if !text.contains('\r') {
        return format_lf(text);
    }
    // here we format with `\n` and put the line endings back, because the
    // formatter and its checks work on lines ending in `\n`
    let formatted = format_lf(&text.replace("\r\n", "\n"))?;
    Ok(if crlf {
        formatted.replace('\n', "\r\n")
    } else {
        formatted
    })
}

/// formatting a text whose lines end in `\n`
fn format_lf(text: &str) -> Result<String, Error> {
    let parse = crate::parse(text);
    if !parse.errors().is_empty() {
        return Err(Error::Syntax);
    }
    let same = |formatted: &str| {
        let again = crate::parse(formatted);
        again.errors().is_empty()
            && crate::pretty(&again.tree()) == crate::pretty(&parse.tree())
            && comments(formatted) == comments(text)
    };
    // breaking a line can change its meaning (a trailing comma after a
    // parenthesised expression, for example), so then we don't break
    for break_lines in [true, false] {
        let formatted = Formatter::new(text, break_lines).run();
        if same(&formatted) {
            return Ok(formatted);
        }
    }
    Err(Error::Changed)
}

/// the comments of a text in order, without trailing whitespace
fn comments(text: &str) -> Vec<String> {
    tokenize(text)
        .0
        .iter()
        .filter(|t| t.kind == COMMENT)
        .map(|t| text[t.range].trim_end().to_string())
        .collect()
}

/// a logical line, with its tokens without trivia
struct Line {
    /// number of open blocks around it
    level: usize,
    tokens: Vec<Token>,
    /// a comment after it on its last physical line
    trailing: Option<String>,
    /// spaces and tabs before the trailing comment
    gap: usize,
    /// for a line written over several lines, the range of text (first to
    /// last token) kept as written
    verbatim: Option<(usize, usize)>,
}

/// a comment on its own line, with its column. A blank line between two of
/// them is stored with no text and column `usize::MAX`
struct CommentLine {
    text: String,
    column: usize,
}

/// the output, before adding the blank lines between top-level definitions
enum Out {
    Line(Line),
    /// a comment line and its indentation level
    Comment(CommentLine, usize),
    Blank,
}

/// a run of the formatter over a text
struct Formatter<'t> {
    text: &'t str,
    /// do we break long lines? (see [`fn@format`])
    break_lines: bool,
}

impl<'t> Formatter<'t> {
    fn new(text: &'t str, break_lines: bool) -> Self {
        Formatter { text, break_lines }
    }

    fn slice(&self, t: &Token) -> &'t str {
        &self.text[t.range]
    }

    /// the column of an offset, from 0, in bytes
    fn column(&self, offset: usize) -> usize {
        offset - self.text[..offset].rfind('\n').map_or(0, |i| i + 1)
    }

    /// the formatted text, unchecked. Comment lines wait in `pending` until
    /// the next logical line, because their indentation depends on it
    fn run(&self) -> String {
        let (tokens, _) = tokenize(self.text);
        let mut outs: Vec<Out> = Vec::new();
        let mut pending: Vec<CommentLine> = Vec::new();
        let mut blank_before_pending = false;
        let mut level = 0usize;
        let mut current: Option<Line> = None;
        // open brackets in the current line
        let mut depth = 0usize;
        // line breaks since the previous line. A `NEWLINE` counts its own
        // break and a comment line doesn't, so there was a blank line if
        // `breaks` is above 0 after a logical line, or above 1 after a
        // comment line (`after_comment`)
        let mut breaks = 0usize;
        let mut after_comment = false;
        let mut prev_line_level = 0usize;
        for token in &tokens {
            match token.kind {
                WHITESPACE => {}
                INDENT => level += 1,
                DEDENT => level = level.saturating_sub(1),
                EOF => {}
                // a line break inside brackets: the line is kept as written
                LINE_BREAK => {
                    breaks += 1;
                    if let Some(line) = &mut current
                        && depth > 0
                        && line.verbatim.is_none()
                    {
                        let start = usize::from(line.tokens[0].range.start());
                        line.verbatim = Some((start, start));
                    }
                }
                // the end of a logical line, here a verbatim line gets its end
                NEWLINE => {
                    if let Some(mut line) = current.take() {
                        if let Some((start, _)) = line.verbatim {
                            let end = line
                                .tokens
                                .last()
                                .map_or(start, |t| usize::from(t.range.end()));
                            line.verbatim = Some((start, end));
                        }
                        prev_line_level = line.level;
                        outs.push(Out::Line(line));
                    }
                    breaks = 0;
                    after_comment = false;
                    depth = 0;
                }
                COMMENT => {
                    let text = self.slice(token).trim_end().to_string();
                    match &mut current {
                        // a trailing comment
                        Some(line) if depth == 0 => {
                            let start = usize::from(token.range.start());
                            let before = self.text[..start].trim_end_matches([' ', '\t']);
                            line.gap = start - before.len();
                            line.trailing = Some(text);
                        }
                        Some(line) => {
                            // inside brackets, the line is kept as written
                            if line.verbatim.is_none() {
                                let start = usize::from(line.tokens[0].range.start());
                                line.verbatim = Some((start, start));
                            }
                        }
                        // a comment on its own line waits for the next logical
                        // line, with the blank line before it if any
                        None => {
                            if breaks > usize::from(after_comment) {
                                if pending.is_empty() {
                                    blank_before_pending = true;
                                } else {
                                    pending.push(CommentLine {
                                        text: String::new(),
                                        column: usize::MAX,
                                    });
                                }
                            }
                            pending.push(CommentLine {
                                text,
                                column: self.column(usize::from(token.range.start())),
                            });
                            breaks = 0;
                            after_comment = true;
                        }
                    }
                }
                _ => {
                    if current.is_none() {
                        // a new logical line. Now we know its indentation, so
                        // we write the blank and comment lines before it
                        let column = self.column(usize::from(token.range.start()));
                        let blank_line = breaks > usize::from(after_comment);
                        if blank_before_pending || (pending.is_empty() && blank_line) {
                            outs.push(Out::Blank);
                        }
                        let blank_after = !pending.is_empty() && blank_line;
                        for comment in pending.drain(..) {
                            if comment.column == usize::MAX {
                                outs.push(Out::Blank);
                                continue;
                            }
                            // a comment deeper than the line ends the block
                            // before it, and keeps that block's level
                            let at = if comment.column > column {
                                prev_line_level.max(level)
                            } else {
                                level
                            };
                            outs.push(Out::Comment(comment, at));
                        }
                        if blank_after {
                            outs.push(Out::Blank);
                        }
                        blank_before_pending = false;
                        current = Some(Line {
                            level,
                            tokens: Vec::new(),
                            trailing: None,
                            gap: 0,
                            verbatim: None,
                        });
                    }
                    match token.kind {
                        L_PAREN | L_BRACK => depth += 1,
                        R_PAREN | R_BRACK => depth = depth.saturating_sub(1),
                        _ => {}
                    }
                    if let Some(line) = &mut current {
                        line.tokens.push(*token);
                    }
                    breaks = 0;
                }
            }
        }
        // closing the last line if there is no final line break. Comments at
        // the end of the file go at the top level
        if let Some(line) = current.take() {
            outs.push(Out::Line(line));
        }
        for comment in pending.drain(..) {
            if comment.column == usize::MAX {
                outs.push(Out::Blank);
            } else {
                outs.push(Out::Comment(comment, 0));
            }
        }
        self.render(outs)
    }

    /// the text of the lines, adding blank lines between top-level
    /// definitions. Never a blank line before the first line
    fn render(&self, outs: Vec<Out>) -> String {
        let mut out = String::new();
        let mut blank = false;
        let mut started = false;
        // was the last line a decorator? Then no blank line before the
        // definition
        let mut after_decorator = false;
        let mut i = 0;
        while i < outs.len() {
            match &outs[i] {
                Out::Blank => blank = started,
                Out::Comment(..) | Out::Line(_) => {
                    // a blank line before each top-level definition, put before
                    // its comments and decorators, not between them
                    let starts_definition = match &outs[i] {
                        Out::Line(line) => line.level == 0 && self.is_definition(line),
                        // the first comment of a block right before a
                        // definition
                        Out::Comment(_, 0)
                            if i == 0 || !matches!(outs[i - 1], Out::Comment(..)) =>
                        {
                            let after = outs[i..].iter().find(|o| !matches!(o, Out::Comment(..)));
                            matches!(after, Some(Out::Line(l)) if l.level == 0 && self.is_definition(l))
                        }
                        _ => false,
                    };
                    let in_comment_block = i > 0 && matches!(outs[i - 1], Out::Comment(..));
                    if started
                        && starts_definition
                        && !after_decorator
                        && !(in_comment_block && matches!(outs[i], Out::Line(_)))
                    {
                        blank = true;
                    }
                    if blank {
                        out.push('\n');
                    }
                    blank = false;
                    started = true;
                    match &outs[i] {
                        Out::Comment(comment, level) => {
                            out.push_str(&"    ".repeat(*level));
                            out.push_str(&comment.text);
                            out.push('\n');
                        }
                        Out::Line(line) => {
                            out.push_str(&"    ".repeat(line.level));
                            let text = self.line(line);
                            if let Some(comment) = &line.trailing {
                                // we keep the gap if the line didn't change,
                                // so aligned comments stay aligned. Otherwise
                                // two spaces
                                let first = usize::from(line.tokens[0].range.start());
                                let last = line
                                    .tokens
                                    .last()
                                    .map_or(first, |t| usize::from(t.range.end()));
                                let gap = if self.text[first..last] == text
                                    && self.column(first) == 4 * line.level
                                {
                                    line.gap.max(2)
                                } else {
                                    2
                                };
                                out.push_str(&text);
                                out.push_str(&" ".repeat(gap));
                                out.push_str(comment);
                            } else {
                                out.push_str(&text);
                            }
                            out.push('\n');
                            after_decorator = line.tokens.first().is_some_and(|t| t.kind == AT);
                        }
                        Out::Blank => {}
                    }
                }
            }
            i += 1;
        }
        out
    }

    /// does a line start a definition (`def`, `struct`, `enum`, `protocol`, or
    /// a decorator)? `protocol` is lexed as a name. `comptime def` doesn't
    /// count
    fn is_definition(&self, line: &Line) -> bool {
        let Some(first) = line.tokens.first() else {
            return false;
        };
        matches!(first.kind, DEF_KW | STRUCT_KW | ENUM_KW | AT)
            || (first.kind == IDENT && matches!(self.slice(first), "protocol"))
    }

    /// the text of a logical line without indentation: as written if
    /// verbatim, broken if too long, else spaced tokens
    fn line(&self, line: &Line) -> String {
        if let Some((start, end)) = line.verbatim {
            return self.text[start..end].to_string();
        }
        let text = self.tokens(&line.tokens, 0);
        if self.break_lines
            && 4 * line.level + text.len() > WIDTH
            && let Some(broken) = self.broken(line)
        {
            return broken;
        }
        text
    }

    /// tokens spaced as the printer does, `depth` brackets in
    fn tokens(&self, tokens: &[Token], mut depth: usize) -> String {
        let mut out = String::new();
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 && self.space(tokens, i, depth) {
                out.push(' ');
            }
            match token.kind {
                L_PAREN | L_BRACK => depth += 1,
                R_PAREN | R_BRACK => depth = depth.saturating_sub(1),
                _ => {}
            }
            out.push_str(self.slice(token));
        }
        out
    }

    /// breaking a long line at its widest outer non-empty group of brackets:
    /// the head, one item per line, the rest. `None` if there is no such
    /// group. A trailing comma in the source is not doubled
    fn broken(&self, line: &Line) -> Option<String> {
        let tokens = &line.tokens;
        let mut groups = Vec::new();
        let (mut depth, mut open) = (0usize, 0);
        for (i, token) in tokens.iter().enumerate() {
            match token.kind {
                L_PAREN | L_BRACK => {
                    if depth == 0 {
                        open = i;
                    }
                    depth += 1;
                }
                R_PAREN | R_BRACK => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 && i > open + 1 {
                        groups.push((open, i));
                    }
                }
                _ => {}
            }
        }
        let width = |&(open, close): &(usize, usize)| {
            usize::from(tokens[close].range.start()) - usize::from(tokens[open].range.end())
        };
        let (open, close) = groups.into_iter().max_by_key(width)?;
        let mut items = Vec::new();
        let (mut depth, mut start) = (0usize, open + 1);
        for i in open + 1..close {
            match tokens[i].kind {
                L_PAREN | L_BRACK => depth += 1,
                R_PAREN | R_BRACK => depth = depth.saturating_sub(1),
                COMMA if depth == 0 => {
                    items.push(&tokens[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        if start < close {
            items.push(&tokens[start..close]);
        }
        let indent = "    ".repeat(line.level);
        let mut out = self.tokens(&tokens[..=open], 0);
        for item in items {
            out.push('\n');
            out.push_str(&indent);
            out.push_str("    ");
            out.push_str(&self.tokens(item, 1));
            out.push(',');
        }
        out.push('\n');
        out.push_str(&indent);
        out.push_str(&self.tokens(&tokens[close..], 0));
        Some(out)
    }

    /// is there a space between tokens `i - 1` and `i`? `depth` is the
    /// bracket depth before token `i`
    fn space(&self, tokens: &[Token], i: usize, depth: usize) -> bool {
        let a = tokens[i - 1].kind;
        let b = tokens[i].kind;
        if matches!(b, R_PAREN | R_BRACK | COMMA | COLON | DOT | QUESTION) {
            return false;
        }
        if matches!(a, L_PAREN | L_BRACK | DOT | AT | TILDE) {
            return false;
        }
        // no spaces in `f(x=1)` and `def f(n=4)`, but spaces in
        // `def f(n: u32 = 4)`. Outside brackets `=` is always spaced
        if depth > 0 && (a == EQ || b == EQ) {
            let at = if b == EQ { i } else { i - 1 };
            return annotated(tokens, at);
        }
        // no space after a unary minus or plus, even before a bracket
        // (`-(1 << n)`): it comes first or after an operator, a bracket, a
        // comma or a keyword
        if matches!(a, MINUS | PLUS) && (i < 2 || is_before_operand(tokens[i - 2].kind)) {
            return false;
        }
        // no space in `f(x)`, `a[i]`, `Map[K, V]`, but a space in `x = (a)`
        // and `return [1]`
        if matches!(b, L_PAREN | L_BRACK) {
            return !matches!(a, IDENT | R_PAREN | R_BRACK | STRING | FSTRING | QUESTION);
        }
        true
    }
}

/// does the `=` at `at` follow a `:` in the same bracketed item? (the
/// default of an annotated parameter)
fn annotated(tokens: &[Token], at: usize) -> bool {
    let mut depth = 0usize;
    for token in tokens[..at].iter().rev() {
        match token.kind {
            R_PAREN | R_BRACK => depth += 1,
            L_PAREN | L_BRACK if depth == 0 => return false,
            L_PAREN | L_BRACK => depth -= 1,
            COMMA if depth == 0 => return false,
            COLON if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

/// does an operand follow a token of this kind? Then a `-` after it is unary
fn is_before_operand(kind: SyntaxKind) -> bool {
    !matches!(
        kind,
        IDENT
            | INT
            | FLOAT
            | STRING
            | FSTRING
            | R_PAREN
            | R_BRACK
            | TRUE_KW
            | FALSE_KW
            | NONE_KW
            | QUESTION
            | ELLIPSIS
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_comments_and_fixes_spacing() {
        let source = "\
# Module comment
from std.net import XdpContext,   XdpAction
# The limit.
const LIMIT: u32 = 10   # trailing



def double(x:u32)->u32:
    # inside
    return x*2
@xdp(iface = \"eth0\")
def main(ctx: XdpContext) -> XdpAction:
    n = double( LIMIT )  # call
    if n > -5:
        return XdpAction.DROP
        # end of the block
    # before pass
    m = [
        1,  # one
        2,
    ]
    return XdpAction.PASS
";
        let expected = "\
# Module comment
from std.net import XdpContext, XdpAction
# The limit.
const LIMIT: u32 = 10   # trailing

def double(x: u32) -> u32:
    # inside
    return x * 2

@xdp(iface=\"eth0\")
def main(ctx: XdpContext) -> XdpAction:
    n = double(LIMIT)  # call
    if n > -5:
        return XdpAction.DROP
        # end of the block
    # before pass
    m = [
        1,  # one
        2,
    ]
    return XdpAction.PASS
";
        assert_eq!(format(source).unwrap(), expected);
        assert_eq!(format(expected).unwrap(), expected);
    }

    #[test]
    fn spaces_defaults_of_annotated_parameters() {
        let source = "def f(x: u32=4, y=2) -> u32:\n    return g(a=1, b=[1, 2])\n";
        let expected = "def f(x: u32 = 4, y=2) -> u32:\n    return g(a=1, b=[1, 2])\n";
        assert_eq!(super::Formatter::new(source, true).run(), expected);
    }

    /// every `.vg` file (examples, standard library, test inputs of the
    /// crates) formats, and formatting is idempotent
    #[test]
    fn formats_every_source() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut files = Vec::new();
        let mut dirs = vec![
            root.join("examples"),
            root.join("../std"),
            root.join("crates"),
        ];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() && !path.ends_with("target") {
                    dirs.push(path);
                } else if path.extension().is_some_and(|e| e == "vg") {
                    files.push(path);
                }
            }
        }
        let mut failures = Vec::new();
        for path in &files {
            let text = std::fs::read_to_string(path).unwrap();
            match format(&text) {
                Ok(once) => {
                    if format(&once).as_ref() != Ok(&once) {
                        failures.push(format!("{}: not idempotent", path.display()));
                    }
                }
                // test inputs with syntax errors on purpose
                Err(Error::Syntax) => {}
                Err(Error::Changed) => failures.push(format!("{}: changed", path.display())),
            }
        }
        assert!(
            failures.is_empty(),
            "{} files:\n{}",
            files.len(),
            failures.join("\n")
        );
    }

    #[test]
    fn breaks_long_lines_at_their_widest_brackets() {
        let long = "def classify(packets: u64, bytes: u64, duration: u64, flags: u32, window: u16, ttl: u8, size: u16) -> u32:\n    return scale(packets, bytes, duration, flags, u32(window), u32(ttl), u32(size), keyword=1, other=2, last=3)\n";
        let formatted = format(long).unwrap();
        assert_eq!(
            formatted,
            "def classify(\n    packets: u64,\n    bytes: u64,\n    duration: u64,\n    flags: u32,\n    window: u16,\n    ttl: u8,\n    size: u16,\n) -> u32:\n    return scale(\n        packets,\n        bytes,\n        duration,\n        flags,\n        u32(window),\n        u32(ttl),\n        u32(size),\n        keyword=1,\n        other=2,\n        last=3,\n    )\n"
        );
        // kept as broken
        assert_eq!(format(&formatted).as_deref(), Ok(formatted.as_str()));
        // lines without brackets stay long
        let import = format!("from std.net import {}\n", ["XdpContext"; 12].join(", "));
        assert_eq!(format(&import).as_deref(), Ok(import.as_str()));
    }

    #[test]
    fn keeps_unary_minus_before_a_bracket() {
        let source = "def f(n: u64, a: i64) -> i64:\n    return -(1 << n) - (a) + -[1][0] * g(-(2), - (3))\n";
        let expected =
            "def f(n: u64, a: i64) -> i64:\n    return -(1 << n) - (a) + -[1][0] * g(-(2), -(3))\n";
        assert_eq!(format(source).as_deref(), Ok(expected));
    }

    /// the standard library and the examples are already formatted
    #[test]
    fn library_and_examples_are_formatted() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut dirs = vec![root.join("examples"), root.join("../std")];
        let mut unformatted = Vec::new();
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() && !path.ends_with(".git") {
                    dirs.push(path);
                } else if path.extension().is_some_and(|e| e == "vg") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    if format(&text).as_deref() != Ok(text.as_str()) {
                        unformatted.push(path.display().to_string());
                    }
                }
            }
        }
        assert!(
            unformatted.is_empty(),
            "not formatted (run `vigil fmt`):\n{}",
            unformatted.join("\n")
        );
    }

    #[test]
    fn keeps_the_line_endings_of_the_source() {
        let lf = "def f() -> u32:\n    # one\n    return 1 + 2\n";
        let crlf = lf.replace('\n', "\r\n");
        assert_eq!(
            format("def f( ) -> u32:\n    # one\n    return 1+2\n").as_deref(),
            Ok(lf)
        );
        let messy = "def f( ) -> u32:\r\n    # one\r\n    return 1+2\r\n";
        assert_eq!(format(messy).as_deref(), Ok(crlf.as_str()));
        assert_eq!(format(&crlf).as_deref(), Ok(crlf.as_str()));
        // mixed endings follow the first line
        let mixed = "def f() -> u32:\r\n    # one\n    return 1 + 2\r\n";
        assert_eq!(format(mixed).as_deref(), Ok(crlf.as_str()));
        let mixed = "def f() -> u32:\n    # one\r\n    return 1 + 2\n";
        assert_eq!(format(mixed).as_deref(), Ok(lf));
    }

    #[test]
    fn refuses_syntax_errors() {
        assert_eq!(format("def f(:\n"), Err(Error::Syntax));
    }
}
