//! this module defines the kinds of tokens and nodes, and the `rowan`
//! language [`VigilLanguage`] that gives the tree types [`SyntaxNode`],
//! [`SyntaxToken`] and [`SyntaxElement`].

use logos::Logos;

/// the kinds of tokens and nodes of the concrete syntax tree. Token kinds
/// carry their `logos` patterns. Layout tokens come from the layout pass,
/// and contextual keywords (`protocol`, `map`, `event`, `type`) are lexed as
/// `IDENT` and get their kind from the parser only where they start an item.
///
/// The order of the variants matters, because some functions test ranges of
/// discriminants: tokens come before `ERROR_TOKEN` and nodes after it,
/// keywords go from `AND_KW` to `TYPE_KW`, augmented assignments from
/// `PLUS_EQ` to `SHR_EQ`, and `__LAST` is last. A new kind goes in its range
#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u16)]
#[allow(non_camel_case_types)]
pub enum SyntaxKind {
    // trivia
    #[regex(r"[ \t\r]+")]
    WHITESPACE,
    #[regex(r"#[^\n]*", allow_greedy = true)]
    COMMENT,
    /// a raw line break. The layout pass turns it into a `NEWLINE`, or keeps
    /// it as trivia inside brackets and at the end of a blank line
    #[token("\n")]
    LINE_BREAK,

    // layout
    NEWLINE,
    INDENT,
    DEDENT,
    /// the end of the input, for the parser lookahead. Never in a tree
    EOF,

    // literals and names
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*")]
    IDENT,
    #[regex(r"[0-9][0-9_]*|0x[0-9a-fA-F_]+|0b[01_]+|0o[0-7_]+")]
    INT,
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?")]
    FLOAT,
    #[regex(r#""([^"\\\n]|\\.)*""#)]
    #[regex(r#"'([^'\\\n]|\\.)*'"#)]
    STRING,
    #[regex(r#"f"([^"\\\n]|\\.)*""#)]
    #[regex(r#"f'([^'\\\n]|\\.)*'"#)]
    FSTRING,

    // keywords
    #[token("and")]
    AND_KW,
    #[token("as")]
    AS_KW,
    #[token("break")]
    BREAK_KW,
    #[token("case")]
    CASE_KW,
    #[token("comptime")]
    COMPTIME_KW,
    #[token("const")]
    CONST_KW,
    #[token("continue")]
    CONTINUE_KW,
    #[token("def")]
    DEF_KW,
    #[token("elif")]
    ELIF_KW,
    #[token("else")]
    ELSE_KW,
    #[token("enum")]
    ENUM_KW,
    #[token("False")]
    FALSE_KW,
    #[token("for")]
    FOR_KW,
    #[token("from")]
    FROM_KW,
    #[token("if")]
    IF_KW,
    #[token("import")]
    IMPORT_KW,
    #[token("in")]
    IN_KW,
    #[token("match")]
    MATCH_KW,
    #[token("None")]
    NONE_KW,
    #[token("not")]
    NOT_KW,
    #[token("or")]
    OR_KW,
    #[token("pass")]
    PASS_KW,
    #[token("return")]
    RETURN_KW,
    #[token("struct")]
    STRUCT_KW,
    #[token("True")]
    TRUE_KW,
    #[token("while")]
    WHILE_KW,
    // contextual keywords, lexed as `IDENT`
    PROTOCOL_KW,
    MAP_KW,
    EVENT_KW,
    TYPE_KW,

    // punctuation
    #[token("(")]
    L_PAREN,
    #[token(")")]
    R_PAREN,
    #[token("[")]
    L_BRACK,
    #[token("]")]
    R_BRACK,
    #[token(",")]
    COMMA,
    #[token(":")]
    COLON,
    #[token(".")]
    DOT,
    #[token("...")]
    ELLIPSIS,
    #[token("->")]
    ARROW,
    #[token("@")]
    AT,
    #[token("?")]
    QUESTION,
    #[token("=")]
    EQ,
    #[token("+=")]
    PLUS_EQ,
    #[token("-=")]
    MINUS_EQ,
    #[token("*=")]
    STAR_EQ,
    #[token("/=")]
    SLASH_EQ,
    #[token("//=")]
    SLASH2_EQ,
    #[token("%=")]
    PERCENT_EQ,
    #[token("&=")]
    AMP_EQ,
    #[token("|=")]
    PIPE_EQ,
    #[token("^=")]
    CARET_EQ,
    #[token("<<=")]
    SHL_EQ,
    #[token(">>=")]
    SHR_EQ,
    #[token("==")]
    EQ2,
    #[token("!=")]
    NEQ,
    #[token("<")]
    LT,
    #[token("<=")]
    LTEQ,
    #[token(">")]
    GT,
    #[token(">=")]
    GTEQ,
    #[token("+")]
    PLUS,
    #[token("-")]
    MINUS,
    #[token("*")]
    STAR,
    #[token("/")]
    SLASH,
    #[token("//")]
    SLASH2,
    #[token("%")]
    PERCENT,
    #[token("&")]
    AMP,
    #[token("|")]
    PIPE,
    #[token("^")]
    CARET,
    #[token("~")]
    TILDE,
    #[token("<<")]
    SHL,
    #[token(">>")]
    SHR,

    /// a character that starts no token, or a string left open at the end of
    /// its line. The patterns match only the string, the character is a
    /// `logos` error that the lexer turns into this kind
    #[regex(r#""([^"\\\n]|\\.)*"#, priority = 1)]
    #[regex(r#"'([^'\\\n]|\\.)*"#, priority = 1)]
    ERROR_TOKEN,

    // nodes
    SOURCE_FILE,
    ERROR,

    // items
    IMPORT,
    IMPORT_NAME,
    FN_DEF,
    STRUCT_DEF,
    ENUM_DEF,
    PROTOCOL_DEF,
    MAP_DECL,
    EVENT_DECL,
    CONST_DEF,
    TYPE_ALIAS,

    // parts of items
    NAME,
    NAME_REF,
    PATH,
    DECORATOR,
    GENERIC_PARAM_LIST,
    GENERIC_PARAM,
    PARAM_LIST,
    PARAM,
    RET_TYPE,
    BASE_LIST,
    MEMBER_LIST,
    FIELD,
    VARIANT,
    VARIANT_FIELDS,

    // types
    PATH_TYPE,
    OPTION_TYPE,
    ARRAY_TYPE,
    GENERIC_ARG_LIST,
    TYPE_ARG,
    CONST_ARG,

    // statements
    BLOCK,
    EXPR_STMT,
    ASSIGN_STMT,
    AUG_ASSIGN_STMT,
    RETURN_STMT,
    IF_STMT,
    ELIF_CLAUSE,
    ELSE_CLAUSE,
    FOR_STMT,
    WHILE_STMT,
    MATCH_STMT,
    MATCH_ARM,
    BREAK_STMT,
    CONTINUE_STMT,
    PASS_STMT,
    COMPTIME_STMT,

    // expressions
    LITERAL,
    NAME_EXPR,
    PAREN_EXPR,
    ARRAY_EXPR,
    /// `a, b`, only on either side of an assignment
    TUPLE_EXPR,
    PREFIX_EXPR,
    BIN_EXPR,
    CALL_EXPR,
    ARG_LIST,
    ARG,
    INDEX_EXPR,
    INDEX_LIST,
    FIELD_EXPR,
    TRY_EXPR,

    // patterns
    WILDCARD_PAT,
    LITERAL_PAT,
    BIND_PAT,
    PATH_PAT,
    TUPLE_STRUCT_PAT,
    OR_PAT,

    // not a kind: the number of kinds, the bound checked by
    // `VigilLanguage::kind_from_raw`
    #[doc(hidden)]
    __LAST,
}

use SyntaxKind::*;

impl SyntaxKind {
    /// whitespace, comments and raw line breaks: skipped by the parser, kept
    /// in the tree
    pub fn is_trivia(self) -> bool {
        matches!(self, WHITESPACE | COMMENT | LINE_BREAK)
    }

    /// is it a keyword, contextual or not?
    pub fn is_keyword(self) -> bool {
        (AND_KW as u16..=TYPE_KW as u16).contains(&(self as u16))
    }

    /// is it a token kind, not a node kind?
    pub fn is_token(self) -> bool {
        (self as u16) <= ERROR_TOKEN as u16
    }

    /// the fixed text of a keyword or punctuation, `None` for other tokens
    /// and for nodes
    pub fn text(self) -> Option<&'static str> {
        Some(match self {
            AND_KW => "and",
            AS_KW => "as",
            BREAK_KW => "break",
            CASE_KW => "case",
            COMPTIME_KW => "comptime",
            CONST_KW => "const",
            CONTINUE_KW => "continue",
            DEF_KW => "def",
            ELIF_KW => "elif",
            ELSE_KW => "else",
            ENUM_KW => "enum",
            FALSE_KW => "False",
            FOR_KW => "for",
            FROM_KW => "from",
            IF_KW => "if",
            IMPORT_KW => "import",
            IN_KW => "in",
            MATCH_KW => "match",
            NONE_KW => "None",
            NOT_KW => "not",
            OR_KW => "or",
            PASS_KW => "pass",
            PROTOCOL_KW => "protocol",
            RETURN_KW => "return",
            STRUCT_KW => "struct",
            TRUE_KW => "True",
            WHILE_KW => "while",
            MAP_KW => "map",
            EVENT_KW => "event",
            TYPE_KW => "type",
            L_PAREN => "(",
            R_PAREN => ")",
            L_BRACK => "[",
            R_BRACK => "]",
            COMMA => ",",
            COLON => ":",
            DOT => ".",
            ELLIPSIS => "...",
            ARROW => "->",
            AT => "@",
            QUESTION => "?",
            EQ => "=",
            PLUS_EQ => "+=",
            MINUS_EQ => "-=",
            STAR_EQ => "*=",
            SLASH_EQ => "/=",
            SLASH2_EQ => "//=",
            PERCENT_EQ => "%=",
            AMP_EQ => "&=",
            PIPE_EQ => "|=",
            CARET_EQ => "^=",
            SHL_EQ => "<<=",
            SHR_EQ => ">>=",
            EQ2 => "==",
            NEQ => "!=",
            LT => "<",
            LTEQ => "<=",
            GT => ">",
            GTEQ => ">=",
            PLUS => "+",
            MINUS => "-",
            STAR => "*",
            SLASH => "/",
            SLASH2 => "//",
            PERCENT => "%",
            AMP => "&",
            PIPE => "|",
            CARET => "^",
            TILDE => "~",
            SHL => "<<",
            SHR => ">>",
            _ => return None,
        })
    }

    /// the kind in "expected ..." messages: the text in backticks, or a
    /// description such as "a name"
    pub fn describe(self) -> String {
        if let Some(text) = self.text() {
            return format!("`{text}`");
        }
        match self {
            IDENT => "a name",
            INT => "an integer",
            FLOAT => "a float",
            STRING => "a string",
            FSTRING => "a format string",
            NEWLINE => "a line break",
            INDENT => "an indented block",
            DEDENT => "the end of the block",
            EOF => "the end of the file",
            _ => "a token",
        }
        .to_string()
    }

    /// is it an augmented assignment operator (`+=`, `<<=`, ...)?
    pub fn is_aug_assign(self) -> bool {
        (PLUS_EQ as u16..=SHR_EQ as u16).contains(&(self as u16))
    }
}

impl From<SyntaxKind> for rowan::SyntaxKind {
    fn from(kind: SyntaxKind) -> Self {
        rowan::SyntaxKind(kind as u16)
    }
}

/// the `rowan` language of Vigil trees. It has no values, it only gives the
/// kind type to `rowan`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VigilLanguage {}

impl rowan::Language for VigilLanguage {
    type Kind = SyntaxKind;

    fn kind_from_raw(raw: rowan::SyntaxKind) -> SyntaxKind {
        assert!(raw.0 < __LAST as u16, "invalid syntax kind {}", raw.0);
        // SAFETY: `SyntaxKind` is `repr(u16)` with discriminants from 0
        // without gaps, and `raw.0` is below `__LAST`
        unsafe { std::mem::transmute::<u16, SyntaxKind>(raw.0) }
    }

    fn kind_to_raw(kind: SyntaxKind) -> rowan::SyntaxKind {
        kind.into()
    }
}

pub type SyntaxNode = rowan::SyntaxNode<VigilLanguage>;
pub type SyntaxToken = rowan::SyntaxToken<VigilLanguage>;
pub type SyntaxElement = rowan::SyntaxElement<VigilLanguage>;
