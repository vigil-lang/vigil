//! this module implements an event-based recursive-descent parser. The
//! [`grammar`] sees only significant tokens and emits events, and
//! [`build_tree`] turns them into a `rowan` tree, putting the trivia back.
//! Entry points: [`parse`] and [`parse_expression`].
//!
//! We use events because a node is often known only after its first child
//! (`a + b` is a `BIN_EXPR` once we see `+`), see [`Parser::open_before`].
//! A grammar loop that consumes nothing runs out of fuel and panics, so a
//! stuck loop is a panic in the tests and not a hang.

mod grammar;

use std::cell::Cell;

use rowan::{GreenNode, GreenNodeBuilder, TextRange, TextSize};
use vigil_diagnostics::codes;

use crate::SyntaxError;
use crate::lexer::Token;
use crate::syntax_kind::SyntaxKind::{self, *};

pub(crate) fn parse(text: &str, tokens: &[Token]) -> (GreenNode, Vec<SyntaxError>) {
    run(text, tokens, grammar::source_file)
}

pub(crate) fn parse_expression(text: &str, tokens: &[Token]) -> (GreenNode, Vec<SyntaxError>) {
    run(text, tokens, grammar::expression_root)
}

/// running a grammar entry point over the tokens
fn run(text: &str, tokens: &[Token], root: fn(&mut Parser)) -> (GreenNode, Vec<SyntaxError>) {
    let mut parser = Parser::new(text, tokens);
    root(&mut parser);
    let green = build_tree(text, tokens, parser.events);
    (green, parser.errors)
}

/// what the grammar did, in order
enum Event {
    /// a node starts
    Open { kind: SyntaxKind },
    /// the innermost open node ends
    Close,
    /// the next significant token is consumed, with another kind if `remap`
    /// says so
    Advance { remap: Option<SyntaxKind> },
}

/// an open node, as the index of its `Open` event. Not `Copy`, so a node is
/// closed once
pub(crate) struct MarkOpened {
    index: usize,
}

/// a closed node, that a new node can still wrap (see
/// [`Parser::open_before`])
#[derive(Clone, Copy)]
pub(crate) struct MarkClosed {
    index: usize,
}

/// the state of the parser: the position in the tokens, events and errors
pub(crate) struct Parser<'t> {
    text: &'t str,
    tokens: &'t [Token],
    /// indices in `tokens` of the significant (non-trivia) tokens
    significant: Vec<usize>,
    /// position in `significant` of the current token
    pos: usize,
    /// lookahead budget, running out means a grammar loop is stuck. A `Cell`
    /// because the lookahead functions take `&self`
    fuel: Cell<u32>,
    events: Vec<Event>,
    errors: Vec<SyntaxError>,
}

/// how many lookaheads are allowed without consuming a token. No rule needs
/// nearly as many
const FUEL: u32 = 256;

impl<'t> Parser<'t> {
    fn new(text: &'t str, tokens: &'t [Token]) -> Self {
        let significant = tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.kind.is_trivia())
            .map(|(i, _)| i)
            .collect();
        Parser {
            text,
            tokens,
            significant,
            pos: 0,
            fuel: Cell::new(FUEL),
            events: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// opening a node at the current token. Its kind is given at close
    pub(crate) fn open(&mut self) -> MarkOpened {
        let mark = MarkOpened {
            index: self.events.len(),
        };
        self.events.push(Event::Open { kind: ERROR });
        mark
    }

    /// opening a node before the closed node `m`, to wrap it (e.g. the left
    /// operand of a binary expression). This shifts the events after it, so
    /// call it right after closing `m`, before opening anything else
    pub(crate) fn open_before(&mut self, m: MarkClosed) -> MarkOpened {
        self.events.insert(m.index, Event::Open { kind: ERROR });
        MarkOpened { index: m.index }
    }

    /// closing the innermost open node, which must be `m`, with `kind`
    pub(crate) fn close(&mut self, m: MarkOpened, kind: SyntaxKind) -> MarkClosed {
        self.events[m.index] = Event::Open { kind };
        self.events.push(Event::Close);
        MarkClosed { index: m.index }
    }

    /// consuming the current token
    pub(crate) fn advance(&mut self) {
        self.advance_as(None);
    }

    /// consuming the current token with another kind, for an `IDENT` that is
    /// a contextual keyword here
    pub(crate) fn advance_remap(&mut self, kind: SyntaxKind) {
        self.advance_as(Some(kind));
    }

    /// consuming the current token. This refills the fuel, because the
    /// parser is not stuck
    fn advance_as(&mut self, remap: Option<SyntaxKind>) {
        assert!(!self.eof(), "advance past the end of the input");
        self.fuel.set(FUEL);
        self.events.push(Event::Advance { remap });
        self.pos += 1;
    }

    pub(crate) fn eof(&self) -> bool {
        self.pos == self.significant.len()
    }

    /// the kind of the significant token `n` places ahead, `EOF` past the
    /// end. Spends one unit of fuel
    pub(crate) fn nth(&self, n: usize) -> SyntaxKind {
        let fuel = self.fuel.get();
        assert!(fuel > 0, "parser is stuck at token {}", self.pos);
        self.fuel.set(fuel - 1);
        self.significant
            .get(self.pos + n)
            .map_or(EOF, |&i| self.tokens[i].kind)
    }

    /// [`Parser::nth`] without fuel, for scans ahead that stop by themselves
    /// at the end of the line. With fuel, a scan longer than [`FUEL`] tokens
    /// would look like a stuck loop
    pub(crate) fn peek(&self, n: usize) -> SyntaxKind {
        self.significant
            .get(self.pos + n)
            .map_or(EOF, |&i| self.tokens[i].kind)
    }

    pub(crate) fn current(&self) -> SyntaxKind {
        self.nth(0)
    }

    pub(crate) fn at(&self, kind: SyntaxKind) -> bool {
        self.nth(0) == kind
    }

    /// is the current token the identifier `word`? Used for contextual
    /// keywords
    pub(crate) fn at_contextual(&self, word: &str) -> bool {
        self.at(IDENT) && self.nth_text(0) == word
    }

    /// the text of the significant token `n` places ahead, empty past the end
    /// and for layout tokens
    pub(crate) fn nth_text(&self, n: usize) -> &'t str {
        self.significant
            .get(self.pos + n)
            .map_or("", |&i| &self.text[self.tokens[i].range])
    }

    /// consuming the current token if it is of `kind`
    pub(crate) fn eat(&mut self, kind: SyntaxKind) -> bool {
        if self.at(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// [`Parser::eat`], with an error if the token is not there. Then nothing
    /// is consumed, because the token may be what the next rule expects
    pub(crate) fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.eat(kind) {
            return true;
        }
        self.error_expected(&kind.describe());
        false
    }

    /// reporting that `what` was expected, and what was found. False if
    /// the error was dropped, because another one is at the same place
    pub(crate) fn error_expected(&mut self, what: &str) -> bool {
        self.error_expected_labeled(what, format!("expected {what}"))
    }

    /// [`Parser::error_expected`] with another label, that says more than
    /// the message
    pub(crate) fn error_expected_labeled(&mut self, what: &str, label: impl Into<String>) -> bool {
        let found = self.describe_current();
        self.error_labeled(format!("expected {what}, found {found}"), label.into())
    }

    /// reporting an error at the current token, the message is also the label
    pub(crate) fn error(&mut self, message: impl Into<String>) -> bool {
        let message = message.into();
        self.error_labeled(message.clone(), message)
    }

    fn error_labeled(&mut self, message: String, label: String) -> bool {
        let range = self.current_range();
        self.error_at(range, message, label)
    }

    /// reporting an error at `range`
    pub(crate) fn error_at(&mut self, range: TextRange, message: String, label: String) -> bool {
        // only the first error at a position is kept, because the next ones
        // are usually other rules failing on the same token
        if self
            .errors
            .last()
            .is_some_and(|e| e.range.start() == range.start())
        {
            return false;
        }
        self.errors
            .push(SyntaxError::new(codes::SYNTAX_ERROR, message, label, range));
        true
    }

    /// adding a help to the error just reported
    pub(crate) fn help(&mut self, help: impl Into<String>) {
        if let Some(error) = self.errors.last_mut() {
            error.help = Some(help.into());
        }
    }

    /// adding a secondary label to the error just reported
    pub(crate) fn secondary(&mut self, range: TextRange, label: impl Into<String>) {
        if let Some(error) = self.errors.last_mut() {
            error.secondary = Some((range, label.into()));
        }
    }

    /// consuming the current token inside an `ERROR` node
    pub(crate) fn advance_with_error(&mut self, message: impl Into<String>) {
        let m = self.open();
        self.error(message);
        self.advance();
        self.close(m, ERROR);
    }

    /// how many errors were reported so far, to know if a rule reported one
    pub(crate) fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// is there a line break in the text between `from` and the current
    /// token?
    pub(crate) fn newline_since(&self, from: TextSize) -> bool {
        let to = self.current_range().start().max(from);
        self.text[TextRange::new(from, to)].contains('\n')
    }

    /// is there a line break in the text between the significant tokens
    /// `n - 1` and `n` places ahead?
    pub(crate) fn line_break_before(&self, n: usize) -> bool {
        let range = |n: usize| {
            self.significant
                .get(self.pos + n)
                .map(|&i| self.tokens[i].range)
        };
        match (n.checked_sub(1).and_then(range), range(n)) {
            (Some(prev), Some(next)) if prev.end() <= next.start() => {
                self.text[TextRange::new(prev.end(), next.start())].contains('\n')
            }
            _ => false,
        }
    }

    /// the indentation of the current line, for an error at an `INDENT`,
    /// whose own range is empty
    pub(crate) fn indentation_range(&self) -> TextRange {
        let at = match self.significant.get(self.pos) {
            Some(&i) => self.tokens[i].range.start(),
            None => TextSize::of(self.text),
        };
        let line_start = self.text[..usize::from(at)]
            .rfind('\n')
            .map_or(0, |n| n + 1);
        TextRange::new(TextSize::try_from(line_start).unwrap_or(at), at)
    }

    /// the range of the significant token before the current one, empty at
    /// the start of the input
    pub(crate) fn prev_range(&self) -> TextRange {
        self.pos
            .checked_sub(1)
            .and_then(|p| self.significant.get(p))
            .map_or(TextRange::empty(TextSize::from(0)), |&i| {
                self.tokens[i].range
            })
    }

    /// the range of an error at the current token. For an empty layout token
    /// or the end of input we use the end of the previous token, where the
    /// reader expects the missing piece
    pub(crate) fn current_range(&self) -> TextRange {
        match self.significant.get(self.pos) {
            Some(&i) if !self.tokens[i].range.is_empty() => self.tokens[i].range,
            _ => {
                let end = self
                    .pos
                    .checked_sub(1)
                    .and_then(|p| self.significant.get(p))
                    .map_or(TextSize::from(0), |&i| self.tokens[i].range.end());
                TextRange::empty(end)
            }
        }
    }

    /// the current token in "found ..." messages. A `NEWLINE` without text is
    /// the one added at the end of a file without a final line break, so it
    /// reads as the end of the file
    fn describe_current(&self) -> String {
        match self.current() {
            kind @ (NEWLINE | INDENT | DEDENT | EOF) => {
                if kind == NEWLINE && self.nth_text(0).is_empty() {
                    EOF.describe()
                } else {
                    kind.describe()
                }
            }
            IDENT => format!("`{}`", self.nth_text(0)),
            kind if kind.is_keyword() || kind.text().is_some() => kind.describe(),
            _ => format!("`{}`", self.nth_text(0).escape_debug()),
        }
    }
}

/// building the tree of the events, with every token (trivia too) once, in
/// order
fn build_tree(text: &str, tokens: &[Token], events: Vec<Event>) -> GreenNode {
    let mut builder = GreenNodeBuilder::new();
    let mut next = 0;
    let mut depth = 0usize;
    let token = |builder: &mut GreenNodeBuilder, kind: SyntaxKind, t: &Token| {
        builder.token(kind.into(), &text[t.range]);
    };
    let flush_trivia = |builder: &mut GreenNodeBuilder, next: &mut usize| {
        while let Some(t) = tokens.get(*next).filter(|t| t.kind.is_trivia()) {
            token(builder, t.kind, t);
            *next += 1;
        }
    };

    for event in events {
        match event {
            Event::Open { kind } => {
                // trivia before a node goes to its parent, so a node starts at
                // its first significant token. The root takes the leading
                // trivia of the file
                if depth > 0 {
                    flush_trivia(&mut builder, &mut next);
                }
                builder.start_node(kind.into());
                depth += 1;
            }
            Event::Close => {
                if depth == 1 {
                    // the trailing trivia goes into the root, which closes last
                    for t in &tokens[next..] {
                        token(&mut builder, t.kind, t);
                    }
                    next = tokens.len();
                }
                builder.finish_node();
                depth -= 1;
            }
            Event::Advance { remap } => {
                flush_trivia(&mut builder, &mut next);
                let t = &tokens[next];
                token(&mut builder, remap.unwrap_or(t.kind), t);
                next += 1;
            }
        }
    }
    assert_eq!(depth, 0, "unbalanced parser events");
    builder.finish()
}
