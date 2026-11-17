//! this module implements the lexer, in two passes: `logos` makes the raw
//! tokens, then a layout pass adds `NEWLINE`, `INDENT` and `DEDENT` as in
//! Python. The layout tokens are empty, so the tokens still cover the text
//! exactly. Layout is a separate pass because it depends on open brackets
//! and blank lines, which the `logos` patterns can't see.
//!
//! The layout rules:
//! - a line break is a `NEWLINE`, except inside brackets or at the end of a
//!   blank or comment-only line, where it is trivia
//! - a deeper indentation opens a level (`INDENT`), a shallower one closes
//!   levels (`DEDENT`) and must land on an open level
//! - indentation is spaces only
//! - the end of the file closes the last line and every open level

use logos::Logos;
use rowan::{TextRange, TextSize};

use crate::SyntaxError;
use crate::syntax_kind::SyntaxKind::{self, *};
use vigil_diagnostics::codes;

/// a token: its kind and the range it covers
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: SyntaxKind,
    pub range: TextRange,
}

/// the tokens of `text`, covering it exactly, and the lexical and layout
/// errors
pub fn tokenize(text: &str) -> (Vec<Token>, Vec<SyntaxError>) {
    let mut errors = Vec::new();
    let raw = raw_tokens(text, &mut errors);
    let tokens = Layout::new(text, &mut errors).run(&raw);
    (tokens, errors)
}

/// the raw tokens from `logos`. A character that starts no token, or a
/// string left open at the end of its line, gives an `ERROR_TOKEN` and an
/// error, and lexing goes on
fn raw_tokens(text: &str, errors: &mut Vec<SyntaxError>) -> Vec<Token> {
    let mut lexer = SyntaxKind::lexer(text);
    let mut tokens = Vec::new();
    while let Some(result) = lexer.next() {
        let span = lexer.span();
        let range = TextRange::new(size(span.start), size(span.end));
        let kind = result.unwrap_or(ERROR_TOKEN);
        if kind == ERROR_TOKEN {
            let slice = lexer.slice();
            let error = if slice.starts_with(['"', '\'']) {
                SyntaxError::new(
                    codes::UNEXPECTED_CHARACTER,
                    "unterminated string literal",
                    "missing closing quote",
                    range,
                )
            } else {
                let error = SyntaxError::new(
                    codes::UNEXPECTED_CHARACTER,
                    format!("unexpected character `{}`", slice.escape_debug()),
                    "not part of any token",
                    range,
                );
                match character_help(slice) {
                    Some(help) => error.with_help(help),
                    None => error,
                }
            };
            errors.push(error);
        }
        // the pattern accepts `0x_`, because `_` may separate digits. We
        // report the missing digits here, so that it is not reported later
        // as a literal too large for 128 bits
        if kind == INT
            && let Some((prefix, digits)) = based_digits(lexer.slice())
            && digits.chars().all(|c| c == '_')
        {
            errors.push(SyntaxError::new(
                codes::SYNTAX_ERROR,
                format!("expected digits after `{prefix}`"),
                "integer literal without digits",
                range,
            ));
        }
        tokens.push(Token { kind, range });
    }
    tokens
}

/// a list of widths, as `0, 4 or 8`
fn or_list(widths: &[usize]) -> String {
    match widths {
        [] => String::new(),
        [only] => only.to_string(),
        [init @ .., last] => {
            let init: Vec<_> = init.iter().map(usize::to_string).collect();
            format!("{} or {last}", init.join(", "))
        }
    }
}

/// a help for a character that other languages use
fn character_help(slice: &str) -> Option<&'static str> {
    Some(match slice {
        ";" => "remove the `;`: a statement ends at the end of its line",
        "{" | "}" => "a block is written with `:` and an indented body, without braces",
        "!" => "write `not` for the logical negation",
        _ => return None,
    })
}

/// the prefix and the digits of an integer with a base prefix (`0x`, `0b`,
/// `0o`), `None` for a decimal one
fn based_digits(int: &str) -> Option<(&str, &str)> {
    ["0x", "0b", "0o"]
        .into_iter()
        .find_map(|p| int.strip_prefix(p).map(|digits| (p, digits)))
}

/// an offset as `rowan` stores it
fn size(n: usize) -> TextSize {
    TextSize::try_from(n).expect("source files are limited to 4 GiB")
}

/// the state of the layout pass
struct Layout<'a> {
    text: &'a str,
    errors: &'a mut Vec<SyntaxError>,
    out: Vec<Token>,
    /// widths of the open indentation levels, innermost last. The bottom
    /// one, 0, is never closed
    levels: Vec<usize>,
    /// number of open brackets of both kinds. Inside them, line breaks and
    /// indentation are trivia
    brackets: usize,
    /// true at the start of a logical line (or before any token)
    at_line_start: bool,
}

impl<'a> Layout<'a> {
    fn new(text: &'a str, errors: &'a mut Vec<SyntaxError>) -> Self {
        Layout {
            text,
            errors,
            out: Vec::new(),
            levels: vec![0],
            brackets: 0,
            at_line_start: true,
        }
    }

    /// the raw tokens with the layout tokens added
    fn run(mut self, raw: &[Token]) -> Vec<Token> {
        let mut i = 0;
        while i < raw.len() {
            if self.at_line_start && self.brackets == 0 {
                i = self.line_start(raw, i);
                if i >= raw.len() {
                    break;
                }
            }
            let token = raw[i];
            match token.kind {
                LINE_BREAK if self.brackets == 0 && !self.at_line_start => {
                    self.push(NEWLINE, token.range);
                    self.at_line_start = true;
                }
                LINE_BREAK | WHITESPACE | COMMENT => self.push(token.kind, token.range),
                L_PAREN | L_BRACK => {
                    self.brackets += 1;
                    self.push(token.kind, token.range);
                }
                // a stray closing bracket is reported by the parser, here we
                // only avoid an underflow
                R_PAREN | R_BRACK => {
                    self.brackets = self.brackets.saturating_sub(1);
                    self.push(token.kind, token.range);
                }
                kind => self.push(kind, token.range),
            }
            i += 1;
        }

        // closing the last line and the open levels, because a file may
        // not end with a line break
        let end = TextRange::empty(size(self.text.len()));
        if !self.at_line_start {
            self.push(NEWLINE, end);
        }
        for _ in 1..self.levels.len() {
            self.push(DEDENT, end);
        }
        self.out
    }

    /// the index of the first token of the next logical line, from a line
    /// start at `i`. Blank and comment-only lines become trivia and the
    /// indentation gives its `INDENT` or `DEDENT`s. Returns `raw.len()` if
    /// no logical line is left
    fn line_start(&mut self, raw: &[Token], mut i: usize) -> usize {
        loop {
            let indent = match raw.get(i) {
                Some(t) if t.kind == WHITESPACE => Some(*t),
                _ => None,
            };
            let content = i + usize::from(indent.is_some());
            match raw.get(content).map(|t| t.kind) {
                // a blank or comment-only line, or the end of the file: all
                // trivia, and its indentation doesn't count
                None | Some(LINE_BREAK) | Some(COMMENT) => {
                    let line_end = raw[content..]
                        .iter()
                        .position(|t| t.kind == LINE_BREAK)
                        .map_or(raw.len(), |p| content + p + 1);
                    for token in &raw[i..line_end] {
                        self.push(token.kind, token.range);
                    }
                    if line_end >= raw.len() {
                        return raw.len();
                    }
                    i = line_end;
                }
                Some(_) => {
                    let (width, tabs) = match indent {
                        Some(ws) => {
                            self.push(WHITESPACE, ws.range);
                            self.indent_width(ws.range)
                        }
                        None => (0, false),
                    };
                    let start = raw[content].range.start();
                    let ws = indent.map_or(TextRange::empty(start), |ws| ws.range);
                    self.indent_to(width, tabs, ws);
                    self.at_line_start = false;
                    return content;
                }
            }
        }
    }

    /// the width of an indentation and whether it has tabs. A tab is
    /// reported and counts as one column, any other width would be a guess
    fn indent_width(&mut self, range: TextRange) -> (usize, bool) {
        let ws = &self.text[range];
        let tabs = ws.contains('\t');
        if tabs {
            self.errors.push(SyntaxError::new(
                codes::TAB_INDENTATION,
                "tabs are not allowed in indentation",
                "indent with spaces",
                range,
            ));
        }
        (ws.chars().filter(|&c| c == ' ' || c == '\t').count(), tabs)
    }

    /// emitting the `INDENT` or `DEDENT`s that go from the open levels to an
    /// indentation of `width`, the width of the whitespace `ws`
    fn indent_to(&mut self, width: usize, tabs: bool, ws: TextRange) {
        let here = TextRange::empty(ws.end());
        let open = self.levels.clone();
        let current = *self.levels.last().unwrap();
        if width > current {
            self.levels.push(width);
            self.push(INDENT, here);
            return;
        }
        while width < *self.levels.last().unwrap() {
            self.levels.pop();
            self.push(DEDENT, here);
        }
        // a width between two open levels: the line stays at the level it
        // dedented to. No error when there are tabs, because they were
        // already reported and are likely the cause
        if width != *self.levels.last().unwrap() && !tabs {
            self.errors.push(SyntaxError::new(
                codes::INCONSISTENT_INDENTATION,
                "unindent does not match any outer indentation level",
                format!("expected {} spaces, found {width}", or_list(&open)),
                ws,
            ));
        }
    }

    fn push(&mut self, kind: SyntaxKind, range: TextRange) {
        self.out.push(Token { kind, range });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the significant tokens of an error-free text, by kind, with the text
    /// of names and integers
    fn kinds(text: &str) -> String {
        let (tokens, errors) = tokenize(text);
        assert!(errors.is_empty(), "{errors:?}");
        tokens
            .iter()
            .filter(|t| !t.kind.is_trivia())
            .map(|t| match t.kind {
                IDENT | INT => format!("{:?}({})", t.kind, &text[t.range]),
                kind => format!("{kind:?}"),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn tokens_cover_the_input() {
        let text = "def f(x):\n    # c\n\n    return [1,\n  2]\n";
        let (tokens, _) = tokenize(text);
        let mut end = TextSize::from(0);
        for token in &tokens {
            assert_eq!(token.range.start(), end, "{token:?}");
            end = token.range.end();
        }
        assert_eq!(usize::from(end), text.len());
    }

    #[test]
    fn indentation_produces_indent_and_dedent() {
        assert_eq!(
            kinds("if a:\n    b\n    if c:\n        d\ne\n"),
            "IF_KW IDENT(a) COLON NEWLINE INDENT IDENT(b) NEWLINE IF_KW IDENT(c) COLON \
             NEWLINE INDENT IDENT(d) NEWLINE DEDENT DEDENT IDENT(e) NEWLINE"
        );
    }

    #[test]
    fn blank_lines_and_brackets_are_trivia() {
        assert_eq!(
            kinds("x = f(1,\n      2)\n\n   # comment\ny\n"),
            "IDENT(x) EQ IDENT(f) L_PAREN INT(1) COMMA INT(2) R_PAREN NEWLINE IDENT(y) NEWLINE"
        );
    }

    #[test]
    fn end_of_file_closes_lines_and_levels() {
        assert_eq!(
            kinds("def f():\n    return"),
            "DEF_KW IDENT(f) L_PAREN R_PAREN COLON NEWLINE INDENT RETURN_KW NEWLINE DEDENT"
        );
    }

    #[test]
    fn keywords_and_literals() {
        assert_eq!(
            kinds("define 0x1F 1_000 1.5 f\"{x}\" 'a'"),
            "IDENT(define) INT(0x1F) INT(1_000) FLOAT FSTRING STRING NEWLINE"
        );
    }

    #[test]
    fn reports_bad_indentation() {
        let (_, errors) = tokenize("if a:\n    b\n  c\n");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].code, codes::INCONSISTENT_INDENTATION);
        let (_, errors) = tokenize("if a:\n\tb\n");
        assert_eq!(errors[0].code, codes::TAB_INDENTATION);
    }

    #[test]
    fn reports_integers_without_digits() {
        for (text, prefix) in [("0x_", "0x"), ("0b__", "0b"), ("0o_", "0o")] {
            let (tokens, errors) = tokenize(text);
            assert_eq!(tokens[0].kind, INT);
            assert_eq!(errors.len(), 1, "{text}");
            assert_eq!(errors[0].code, codes::SYNTAX_ERROR);
            assert_eq!(
                errors[0].message,
                format!("expected digits after `{prefix}`")
            );
        }
        assert_eq!(
            kinds("0x_1 0b1_ 0o_7"),
            "INT(0x_1) INT(0b1_) INT(0o_7) NEWLINE"
        );
    }

    #[test]
    fn reports_unexpected_characters() {
        let (_, errors) = tokenize("x = $\ns = \"abc\n");
        let messages: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();
        assert_eq!(
            messages,
            ["unexpected character `$`", "unterminated string literal"]
        );
    }
}
