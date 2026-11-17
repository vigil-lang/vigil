//! this module defines the grammar of Vigil, as recursive-descent functions
//! over a [`Parser`]:
//!
//! ```text
//! file         = item*
//! item         = decorator* ( import | fn_def | struct_def | enum_def
//!                | protocol_def | map_decl | event_decl | const_def
//!                | type_alias )
//! decorator    = "@" path arg_list? NEWLINE
//! import       = "from" path "import" ( "(" names ")" | names ) NEWLINE
//!              | "import" path ( "as" NAME )? NEWLINE
//! names        = ( NAME ( "as" NAME )? ),*
//! fn_def       = "comptime"? "def" NAME generic_params? params
//!                ( "->" type )? ":" block
//! struct_def   = "struct" NAME generic_params? bases? ":" members
//! enum_def     = "enum" NAME generic_params? bases? ":" members
//! protocol_def = "protocol" NAME generic_params? bases? ":" members
//! bases        = "(" type,* ")"
//! members      = NEWLINE INDENT member+ DEDENT | "pass" NEWLINE
//! member       = NAME ":" type NEWLINE                (struct field)
//!              | NAME ( "(" type,* ")" )? NEWLINE     (enum variant)
//!              | decorator* fn_def | "pass" NEWLINE
//! map_decl     = "map" NAME ":" type arg_list? NEWLINE  ("event" alike)
//! const_def    = "const" NAME ( ":" type )? "=" expr NEWLINE
//! type_alias   = "type" NAME generic_params? "=" type NEWLINE
//! generic_params = "[" ( NAME ( ":" type )? ),* "]"
//! params       = "(" ( NAME ( ":" type )? ( "=" expr )? ),* ")"
//! type         = path ( "[" ( type | expr ),* "]" )? ( "?" | "[" expr "]" )*
//!
//! block        = NEWLINE INDENT stmt+ DEDENT | simple_stmt
//! stmt         = if_stmt | decorator* for_stmt | decorator* while_stmt
//!              | match_stmt | "comptime" ":" block | simple_stmt
//! simple_stmt  = ( "return" expr? | "break" | "continue" | "pass"
//!                | exprs "=" exprs
//!                | expr ( ":" type ( "=" expr )? | AUG_OP expr )? ) NEWLINE
//! exprs        = expr ( "," expr )*
//! if_stmt      = "if" expr ":" block ( "elif" expr ":" block )*
//!                ( "else" ":" block )?
//! for_stmt     = "for" NAME "in" expr ":" block
//! while_stmt   = "while" expr ":" block
//! match_stmt   = "match" expr ":" NEWLINE INDENT
//!                ( "case" pattern ( "if" expr )? ":" block )+ DEDENT
//!
//! expr         = precedence climbing over, from loosest to tightest:
//!                or, and, not, comparisons, |, ^, &, << >>, + -, * / // %,
//!                unary - ~, postfix call/index/field/?
//! arg_list     = "(" ( ( NAME "=" )? expr ),* ")"
//! pattern      = "_" | "-"? literal | NAME | path | path "(" pattern,* ")"
//!              | pattern ( "|" pattern )+
//! ```
//!
//! `x,*` is a comma-separated list with an optional trailing comma.
//! `protocol`, `map`, `event` and `type` are keywords only at the start of
//! an item, followed by a name.
//!
//! Error recovery: a rule that doesn't find what it expects reports it and
//! returns, and the caller goes on as if the part were there. Input that
//! makes no sense is skipped to the end of the line (and its indented
//! block) by [`skip_line`]. Every loop goes through [`progress`] or
//! [`delimited`], so no loop can spin in place.

use rowan::TextRange;

use crate::parser::{MarkClosed, Parser};
use crate::syntax_kind::SyntaxKind::{self, *};

/// a whole file: items until the end of the input
pub(crate) fn source_file(p: &mut Parser) {
    let m = p.open();
    while !p.eof() {
        progress(p, item);
    }
    p.close(m, SOURCE_FILE);
}

/// a lone expression, for interpolations of format strings. Anything after
/// it is an error, kept in an `ERROR` node
pub(crate) fn expression_root(p: &mut Parser) {
    let m = p.open();
    expr(p);
    // the `NEWLINE` the lexer adds at the end of the text
    p.eat(NEWLINE);
    if !p.eof() {
        let rest = p.open();
        p.error_expected("the end of the expression");
        while !p.eof() {
            p.advance();
        }
        p.close(rest, ERROR);
    }
    p.close(m, SOURCE_FILE);
}

// items
// -------------------------

/// an item with its decorators, which go inside the item's node. The kind
/// is known only after the decorators
fn item(p: &mut Parser) {
    let m = p.open();
    let errors = p.error_count();
    while p.at(AT) {
        decorator(p);
    }
    let kind = match p.current() {
        FROM_KW | IMPORT_KW => import(p),
        DEF_KW | COMPTIME_KW => fn_def(p),
        STRUCT_KW => owner_def(p, STRUCT_KW, STRUCT_DEF),
        ENUM_KW => owner_def(p, ENUM_KW, ENUM_DEF),
        // a contextual keyword must be followed by a name, otherwise it is
        // an ordinary name and can't start an item
        IDENT if p.at_contextual("protocol") && p.nth(1) == IDENT => {
            owner_def(p, PROTOCOL_KW, PROTOCOL_DEF)
        }
        CONST_KW => const_def(p),
        IDENT if p.at_contextual("map") && p.nth(1) == IDENT => map_decl(p, MAP_KW, MAP_DECL),
        IDENT if p.at_contextual("event") && p.nth(1) == IDENT => map_decl(p, EVENT_KW, EVENT_DECL),
        IDENT if p.at_contextual("type") && p.nth(1) == IDENT => type_alias(p),
        INDENT => {
            unexpected_indentation(p);
            skip_block(p);
            ERROR
        }
        // a bad decorator that took the rest of the file, already reported
        EOF if p.error_count() > errors => ERROR,
        _ => {
            if p.error_expected("an item")
                && let Some(help) = item_help(p)
            {
                p.help(help);
            }
            skip_line(p);
            ERROR
        }
    };
    p.close(m, kind);
}

/// a help for a line that starts no item, when it looks like a function of
/// another language or a statement
fn item_help(p: &Parser) -> Option<String> {
    let word = p.nth_text(0);
    if p.at(IDENT) && matches!(word, "fn" | "func" | "function" | "fun") {
        return Some("functions are defined with `def`".to_string());
    }
    let name = match (p.nth(0), p.nth(1)) {
        (IDENT, IDENT) if matches!(word, "let" | "var") => p.nth_text(1),
        (IDENT, EQ | COLON) => word,
        _ => return None,
    };
    Some(format!(
        "statements go in the body of a function; for a constant, write `const {name} = ...`"
    ))
}

/// the error of a line indented more than the line before it
fn unexpected_indentation(p: &mut Parser) {
    let range = p.indentation_range();
    p.error_at(
        range,
        "unexpected indentation".to_string(),
        "indented more than the line before".to_string(),
    );
}

/// `@path(args)`, on its own line
fn decorator(p: &mut Parser) {
    let m = p.open();
    p.advance(); // `@`
    path(p);
    if p.at(L_PAREN) {
        arg_list(p);
    }
    expect_newline(p);
    p.close(m, DECORATOR);
}

/// `import path as alias` or `from path import names`. The names may be in
/// parentheses, to span several lines
fn import(p: &mut Parser) -> SyntaxKind {
    if p.eat(IMPORT_KW) {
        path(p);
        if p.eat(AS_KW) {
            name(p);
        }
    } else {
        p.advance(); // `from`
        path(p);
        p.expect(IMPORT_KW);
        if p.eat(L_PAREN) {
            delimited(p, R_PAREN, import_name);
        } else {
            import_name(p);
            while p.eat(COMMA) {
                import_name(p);
            }
        }
    }
    expect_newline(p);
    IMPORT
}

fn import_name(p: &mut Parser) {
    let m = p.open();
    name_ref(p);
    if p.eat(AS_KW) {
        name(p);
    }
    p.close(m, IMPORT_NAME);
}

/// a function definition, from `comptime` or `def`. The caller opens and
/// closes its node, so the decorators are inside
fn fn_def(p: &mut Parser) -> SyntaxKind {
    p.eat(COMPTIME_KW);
    p.expect(DEF_KW);
    name(p);
    if p.at(L_BRACK) {
        generic_params(p);
    }
    if p.at(L_PAREN) {
        params(p);
    } else {
        p.error_expected("a parameter list");
    }
    if p.at(ARROW) {
        let m = p.open();
        p.advance();
        type_(p);
        p.close(m, RET_TYPE);
    }
    header_colon(p);
    block(p);
    FN_DEF
}

/// `struct`, `enum` and `protocol`: a header and members. We remap the
/// keyword because `protocol` is lexed as a name
fn owner_def(p: &mut Parser, keyword: SyntaxKind, kind: SyntaxKind) -> SyntaxKind {
    p.advance_remap(keyword);
    name(p);
    if p.at(L_BRACK) {
        generic_params(p);
    }
    if p.at(L_PAREN) {
        let m = p.open();
        p.advance();
        delimited(p, R_PAREN, |p| {
            type_(p);
        });
        p.close(m, BASE_LIST);
    }
    header_colon(p);
    members(p, kind);
    kind
}

/// the members of an owner: an indented block, or `pass` on the header line
/// (`struct S(Model): pass`)
fn members(p: &mut Parser, owner: SyntaxKind) {
    let m = p.open();
    if p.at(PASS_KW) {
        pass_stmt(p);
    } else if p.expect(NEWLINE) {
        if p.at(INDENT) {
            p.advance();
            while !p.at(DEDENT) && !p.eof() {
                progress(p, |p| member(p, owner));
            }
            p.expect(DEDENT);
        } else {
            p.error_expected(&INDENT.describe());
        }
    }
    p.close(m, MEMBER_LIST);
}

/// one member: fields in a struct, variants in an enum, methods in all
/// three. Only a method may be decorated
fn member(p: &mut Parser, owner: SyntaxKind) {
    let m = p.open();
    let decorated = p.at(AT);
    while p.at(AT) {
        decorator(p);
    }
    let kind = match p.current() {
        DEF_KW | COMPTIME_KW => fn_def(p),
        PASS_KW if !decorated => {
            p.advance();
            expect_newline(p);
            PASS_STMT
        }
        IDENT if !decorated && owner == STRUCT_DEF && p.nth(1) == COLON => {
            name(p);
            p.advance(); // `:`
            type_(p);
            expect_newline(p);
            FIELD
        }
        IDENT if !decorated && owner == ENUM_DEF => {
            name(p);
            if p.at(L_PAREN) {
                let fields = p.open();
                p.advance();
                delimited(p, R_PAREN, |p| {
                    type_(p);
                });
                p.close(fields, VARIANT_FIELDS);
            }
            expect_newline(p);
            VARIANT
        }
        // `x u32`, a field without its `:`
        IDENT if !decorated && owner == STRUCT_DEF && p.nth(1) == IDENT => {
            p.advance();
            p.error_expected_labeled("`:`", "expected `:` between the field name and its type");
            skip_line(p);
            ERROR
        }
        _ => {
            p.error_expected(match owner {
                STRUCT_DEF => "a field or a method",
                ENUM_DEF => "a variant or a method",
                _ => "a method",
            });
            skip_line(p);
            ERROR
        }
    };
    p.close(m, kind);
}

/// a `map` or an `event` declaration, they have the same shape
fn map_decl(p: &mut Parser, keyword: SyntaxKind, kind: SyntaxKind) -> SyntaxKind {
    p.advance_remap(keyword);
    name(p);
    p.expect(COLON);
    type_(p);
    if p.at(L_PAREN) {
        arg_list(p);
    }
    expect_newline(p);
    kind
}

fn const_def(p: &mut Parser) -> SyntaxKind {
    p.advance();
    name(p);
    if p.eat(COLON) {
        type_(p);
    }
    p.expect(EQ);
    expr(p);
    expect_newline(p);
    CONST_DEF
}

fn type_alias(p: &mut Parser) -> SyntaxKind {
    p.advance_remap(TYPE_KW);
    name(p);
    if p.at(L_BRACK) {
        generic_params(p);
    }
    p.expect(EQ);
    type_(p);
    expect_newline(p);
    TYPE_ALIAS
}

/// `[T, N: u32]`: type parameters with an optional bound and constant
/// parameters with their type. Here they look the same
fn generic_params(p: &mut Parser) {
    let m = p.open();
    p.advance(); // `[`
    delimited(p, R_BRACK, |p| {
        let m = p.open();
        name(p);
        if p.eat(COLON) {
            type_(p);
        }
        p.close(m, GENERIC_PARAM);
    });
    p.close(m, GENERIC_PARAM_LIST);
}

fn params(p: &mut Parser) {
    let m = p.open();
    p.advance(); // `(`
    delimited(p, R_PAREN, |p| {
        let m = p.open();
        name(p);
        if p.eat(COLON) {
            type_(p);
        } else if p.at(IDENT) {
            // `x u32`, the list then skips to its `)`
            p.error_expected_labeled("`:`", "expected `:` between the parameter and its type");
        }
        if p.eat(EQ) {
            expr(p);
        }
        p.close(m, PARAM);
    });
    p.close(m, PARAM_LIST);
}

// names and paths
// -------------------------

/// a name being defined
fn name(p: &mut Parser) {
    if p.at(IDENT) {
        let m = p.open();
        p.advance();
        p.close(m, NAME);
    } else {
        p.error_expected(&IDENT.describe());
    }
}

/// a name being used
fn name_ref(p: &mut Parser) {
    if p.at(IDENT) {
        let m = p.open();
        p.advance();
        p.close(m, NAME_REF);
    } else {
        p.error_expected(&IDENT.describe());
    }
}

/// a dotted path, `std.net` or `XdpAction.DROP`. A dot without a name after
/// it is left to the caller
fn path(p: &mut Parser) {
    let m = p.open();
    name_ref(p);
    while p.at(DOT) && p.nth(1) == IDENT {
        p.advance();
        name_ref(p);
    }
    p.close(m, PATH);
}

// types
// -------------------------

/// a type. Nothing is consumed if the current token can't start one
fn type_(p: &mut Parser) {
    if !p.at(IDENT) {
        p.error_expected("a type");
        return;
    }
    let m = p.open();
    path(p);
    if p.at(L_BRACK) {
        generic_args(p);
    }
    let mut ty = p.close(m, PATH_TYPE);
    // `T?` and `T[N]` wrap the type so far. The first bracket was already
    // taken as generic arguments, so `u8[4]` is a `PATH_TYPE` with argument
    // `4`, and `u8[4][2]` an array of two `u8[4]`
    loop {
        if p.at(QUESTION) {
            let m = p.open_before(ty);
            p.advance();
            ty = p.close(m, OPTION_TYPE);
        } else if p.at(L_BRACK) {
            let m = p.open_before(ty);
            p.advance();
            let open = p.prev_range();
            let len = p.open();
            expr(p);
            p.close(len, CONST_ARG);
            if !p.at(R_BRACK) {
                expect_closing(p, R_BRACK, open);
                skip_to_close(p);
            }
            p.eat(R_BRACK);
            ty = p.close(m, ARRAY_TYPE);
        } else {
            break;
        }
    }
}

/// `[...]` after a path, each argument a type or a constant expression
fn generic_args(p: &mut Parser) {
    let m = p.open();
    p.advance(); // `[`
    delimited(p, R_BRACK, |p| {
        let m = p.open();
        if type_arg_ahead(p) {
            type_(p);
            p.close(m, TYPE_ARG);
        } else {
            expr(p);
            p.close(m, CONST_ARG);
        }
    });
    p.close(m, GENERIC_ARG_LIST);
}

/// do the tokens ahead form a type (a path, generic arguments, `?`s) up to
/// the end of the argument? Otherwise (`4`, `N * 2`) it is a constant
/// expression. A name alone is taken as a type, later phases fix it if it
/// is a constant. We use [`Parser::peek`] here, which spends no fuel
fn type_arg_ahead(p: &Parser) -> bool {
    if p.peek(0) != IDENT {
        return false;
    }
    let mut n = 1;
    loop {
        match p.peek(n) {
            DOT if p.peek(n + 1) == IDENT => n += 2,
            QUESTION => n += 1,
            L_BRACK => {
                let mut depth = 0;
                loop {
                    match p.peek(n) {
                        L_BRACK => depth += 1,
                        R_BRACK => depth -= 1,
                        EOF | NEWLINE => return false,
                        _ => {}
                    }
                    n += 1;
                    if depth == 0 {
                        break;
                    }
                }
            }
            COMMA | R_BRACK => return true,
            _ => return false,
        }
    }
}

// statements
// -------------------------

/// the body of a compound statement or a function: an indented block, or a
/// simple statement on the same line (`if n == 3: break`)
fn block(p: &mut Parser) {
    let m = p.open();
    if p.at(NEWLINE) {
        p.advance();
        if p.at(INDENT) {
            p.advance();
            while !p.at(DEDENT) && !p.eof() {
                progress(p, stmt);
            }
            p.expect(DEDENT);
        } else {
            p.error_expected(&INDENT.describe());
        }
    } else {
        simple_stmt(p);
    }
    p.close(m, BLOCK);
}

/// a statement. Only loops may be decorated
fn stmt(p: &mut Parser) {
    match p.current() {
        AT => {
            let m = p.open();
            while p.at(AT) {
                decorator(p);
            }
            let kind = match p.current() {
                FOR_KW => for_stmt(p),
                WHILE_KW => while_stmt(p),
                _ => {
                    p.error_expected("a `for` or `while` loop after the decorators");
                    skip_line(p);
                    ERROR
                }
            };
            p.close(m, kind);
        }
        IF_KW => if_stmt(p),
        FOR_KW => {
            let m = p.open();
            let kind = for_stmt(p);
            p.close(m, kind);
        }
        WHILE_KW => {
            let m = p.open();
            let kind = while_stmt(p);
            p.close(m, kind);
        }
        MATCH_KW => match_stmt(p),
        COMPTIME_KW if p.nth(1) == COLON => {
            let m = p.open();
            p.advance();
            p.advance();
            block(p);
            p.close(m, COMPTIME_STMT);
        }
        INDENT => {
            let m = p.open();
            unexpected_indentation(p);
            skip_block(p);
            p.close(m, ERROR);
        }
        _ => simple_stmt(p),
    }
}

/// a statement on one logical line. We parse an expression first, then `=`,
/// `:` or an augmented operator after it makes an assignment
fn simple_stmt(p: &mut Parser) {
    let m = p.open();
    // `let x = ...`, from another language
    let declared = (p.at_contextual("let") || p.at_contextual("var")) && p.nth(1) == IDENT;
    let declared = declared.then(|| p.nth_text(1));
    let kind = match p.current() {
        RETURN_KW => {
            p.advance();
            if !p.at(NEWLINE) {
                expr(p);
            }
            RETURN_STMT
        }
        BREAK_KW => {
            p.advance();
            BREAK_STMT
        }
        CONTINUE_KW => {
            p.advance();
            CONTINUE_STMT
        }
        PASS_KW => {
            p.advance();
            PASS_STMT
        }
        kind if starts_expr(kind) => {
            let tuple = expr_list(p);
            match p.current() {
                EQ => {
                    p.advance();
                    expr_list(p);
                    ASSIGN_STMT
                }
                // a tuple can only be the target of a plain `=`
                _ if tuple => {
                    p.error_expected("`=` after the targets of an assignment");
                    EXPR_STMT
                }
                COLON => {
                    p.advance();
                    type_(p);
                    if p.eat(EQ) {
                        expr(p);
                    }
                    ASSIGN_STMT
                }
                kind if kind.is_aug_assign() => {
                    p.advance();
                    expr(p);
                    AUG_ASSIGN_STMT
                }
                _ => EXPR_STMT,
            }
        }
        kind => {
            if p.error_expected("a statement") && matches!(kind, ELIF_KW | ELSE_KW) {
                p.help(format!(
                    "`{}` continues an `if`, and goes at the same indentation as the `if`",
                    p.nth_text(0)
                ));
            }
            skip_line(p);
            p.close(m, ERROR);
            return;
        }
    };
    let before = p.error_count();
    expect_newline(p);
    if let Some(name) = declared
        && p.error_count() > before
    {
        p.help(format!(
            "a variable is declared by its first assignment: write `{name} = ...`"
        ));
    }
    p.close(m, kind);
}

/// `pass`, as the members of an owner
fn pass_stmt(p: &mut Parser) {
    let m = p.open();
    p.advance();
    expect_newline(p);
    p.close(m, PASS_STMT);
}

fn if_stmt(p: &mut Parser) {
    let m = p.open();
    p.advance(); // `if`
    expr(p);
    header_colon(p);
    block(p);
    while p.at(ELIF_KW) {
        let m = p.open();
        p.advance();
        expr(p);
        header_colon(p);
        block(p);
        p.close(m, ELIF_CLAUSE);
    }
    if p.at(ELSE_KW) {
        let m = p.open();
        p.advance();
        if !p.at(IF_KW) {
            header_colon(p);
        } else {
            if p.error_expected("`:`") {
                p.help("write `elif` instead of `else if`");
            }
            skip_to_newline(p);
        }
        block(p);
        p.close(m, ELSE_CLAUSE);
    }
    p.close(m, IF_STMT);
}

fn for_stmt(p: &mut Parser) -> SyntaxKind {
    p.advance(); // `for`
    if !p.at(IDENT) {
        // `for (i = 0; ...)`, from another language. The rest of the
        // header would only give more errors
        if p.error_expected(&IDENT.describe()) {
            p.help("a loop over a range is written `for i in range(n):`");
        }
        skip_line(p);
        return FOR_STMT;
    }
    name(p);
    p.expect(IN_KW);
    expr(p);
    header_colon(p);
    block(p);
    FOR_STMT
}

fn while_stmt(p: &mut Parser) -> SyntaxKind {
    p.advance(); // `while`
    expr(p);
    header_colon(p);
    block(p);
    WHILE_STMT
}

/// `match`, whose block has only `case` arms. Anything else is skipped one
/// line at a time
fn match_stmt(p: &mut Parser) {
    let m = p.open();
    p.advance(); // `match`
    expr(p);
    header_colon(p);
    if p.expect(NEWLINE) && p.expect(INDENT) {
        while !p.at(DEDENT) && !p.eof() {
            if p.at(CASE_KW) {
                match_arm(p);
            } else {
                let m = p.open();
                p.error_expected("`case`");
                skip_line(p);
                p.close(m, ERROR);
            }
        }
        p.expect(DEDENT);
    }
    p.close(m, MATCH_STMT);
}

fn match_arm(p: &mut Parser) {
    let m = p.open();
    p.advance(); // `case`
    pattern(p);
    if p.eat(IF_KW) {
        expr(p);
    }
    header_colon(p);
    block(p);
    p.close(m, MATCH_ARM);
}

// expressions
// -------------------------

/// can a token of this kind start an expression?
fn starts_expr(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        INT | FLOAT
            | STRING
            | FSTRING
            | TRUE_KW
            | FALSE_KW
            | NONE_KW
            | ELLIPSIS
            | IDENT
            | L_PAREN
            | L_BRACK
            | NOT_KW
            | MINUS
            | TILDE
    )
}

/// the binding power of the operand of `not`, between `and` and comparisons:
/// `not a == b` is `not (a == b)`, `not a and b` is `(not a) and b`, as in
/// Python
const NOT_BP: u8 = 5;
/// the binding power of the operand of unary `-` and `~`, above every binary
/// operator: `-a * b` is `(-a) * b`. Postfix binds tighter, `-a.b` is
/// `-(a.b)`
const UNARY_BP: u8 = 21;

/// the left and right binding powers of a binary operator. The right one is
/// greater, so every operator is left associative
fn infix_bp(kind: SyntaxKind) -> Option<(u8, u8)> {
    Some(match kind {
        OR_KW => (1, 2),
        AND_KW => (3, 4),
        EQ2 | NEQ | LT | LTEQ | GT | GTEQ => (7, 8),
        PIPE => (9, 10),
        CARET => (11, 12),
        AMP => (13, 14),
        SHL | SHR => (15, 16),
        PLUS | MINUS => (17, 18),
        STAR | SLASH | SLASH2 | PERCENT => (19, 20),
        _ => return None,
    })
}

/// an expression
pub(crate) fn expr(p: &mut Parser) {
    expr_bp(p, 0);
}

/// an expression or a `TUPLE_EXPR` `a, b, ...`, for both sides of an
/// assignment. Returns true for a tuple
fn expr_list(p: &mut Parser) -> bool {
    let Some(first) = expr_bp(p, 0) else {
        return false;
    };
    if !p.at(COMMA) {
        return false;
    }
    let m = p.open_before(first);
    while p.eat(COMMA) {
        expr(p);
    }
    p.close(m, TUPLE_EXPR);
    true
}

/// precedence climbing: an expression whose binary operators bind at least
/// as tightly as `min_bp`. Postfix operators are always applied. `None` if
/// there is no operand
fn expr_bp(p: &mut Parser, min_bp: u8) -> Option<MarkClosed> {
    let mut lhs = match p.current() {
        NOT_KW => prefix(p, NOT_BP),
        MINUS | TILDE => prefix(p, UNARY_BP),
        _ => atom(p)?,
    };
    loop {
        match p.current() {
            L_PAREN => {
                let m = p.open_before(lhs);
                arg_list(p);
                lhs = p.close(m, CALL_EXPR);
            }
            L_BRACK => {
                let m = p.open_before(lhs);
                let list = p.open();
                p.advance();
                delimited(p, R_BRACK, expr);
                p.close(list, INDEX_LIST);
                lhs = p.close(m, INDEX_EXPR);
            }
            DOT => {
                let m = p.open_before(lhs);
                p.advance();
                name_ref(p);
                lhs = p.close(m, FIELD_EXPR);
            }
            QUESTION => {
                let m = p.open_before(lhs);
                p.advance();
                lhs = p.close(m, TRY_EXPR);
            }
            op => {
                let Some((left, right)) = infix_bp(op) else {
                    break;
                };
                if left < min_bp {
                    break;
                }
                let m = p.open_before(lhs);
                let op_end = p.current_range().end();
                p.advance();
                let before = p.error_count();
                let missing = expr_bp(p, right).is_none();
                if missing {
                    p.error_expected("an expression");
                    // `&&`, `++`: an operator of another language
                    if p.error_count() > before
                        && p.current_range().start() == op_end
                        && let Some(help) = doubled_operator_help(op, p.current())
                    {
                        p.help(help);
                    }
                }
                lhs = p.close(m, BIN_EXPR);
                // the rest of the line is reported once, by the statement
                if missing {
                    break;
                }
            }
        }
    }
    Some(lhs)
}

/// a help for two operators written together, that are one operator in
/// other languages
fn doubled_operator_help(first: SyntaxKind, second: SyntaxKind) -> Option<&'static str> {
    Some(match (first, second) {
        (AMP, AMP) => "write `and` instead of `&&`",
        (PIPE, PIPE) => "write `or` instead of `||`",
        (PLUS, PLUS) => "Vigil has no `++`: write `+= 1`",
        (MINUS, MINUS) => "Vigil has no `--`: write `-= 1`",
        _ => return None,
    })
}

/// a prefix operator and its operand, which binds at least as tightly as
/// `bp`
fn prefix(p: &mut Parser, bp: u8) -> MarkClosed {
    let m = p.open();
    p.advance();
    if expr_bp(p, bp).is_none() {
        p.error_expected("an expression");
    }
    p.close(m, PREFIX_EXPR)
}

/// a literal, a name or a bracketed expression. `None` (consuming nothing)
/// if the current token starts none of them
fn atom(p: &mut Parser) -> Option<MarkClosed> {
    let m = p.open();
    let kind = match p.current() {
        INT | FLOAT | STRING | FSTRING | TRUE_KW | FALSE_KW | NONE_KW | ELLIPSIS => {
            p.advance();
            LITERAL
        }
        IDENT => {
            name_ref(p);
            NAME_EXPR
        }
        L_PAREN => {
            p.advance();
            let open = p.prev_range();
            expr(p);
            if !p.at(R_PAREN) {
                expect_closing(p, R_PAREN, open);
                // `(a, b)`: the rest up to the `)` gives no more errors
                skip_to_close(p);
            }
            p.eat(R_PAREN);
            PAREN_EXPR
        }
        L_BRACK => {
            p.advance();
            delimited(p, R_BRACK, expr);
            ARRAY_EXPR
        }
        _ => {
            p.error_expected("an expression");
            // we close the node as an empty `ERROR`, so the events balance
            p.close(m, ERROR);
            return None;
        }
    };
    Some(p.close(m, kind))
}

/// `(a, name=b)`. A name followed by `=` is a keyword argument, otherwise it
/// starts the value
fn arg_list(p: &mut Parser) {
    let m = p.open();
    p.advance(); // `(`
    delimited(p, R_PAREN, |p| {
        let m = p.open();
        if p.at(IDENT) && p.nth(1) == EQ {
            name(p);
            p.advance();
        }
        expr(p);
        p.close(m, ARG);
    });
    p.close(m, ARG_LIST);
}

// patterns
// -------------------------

/// a pattern, or alternatives separated by `|`
fn pattern(p: &mut Parser) {
    let Some(first) = single_pattern(p) else {
        return;
    };
    if p.at(PIPE) {
        let m = p.open_before(first);
        while p.eat(PIPE) {
            single_pattern(p);
        }
        p.close(m, OR_PAT);
    }
}

/// a pattern without `|`. `_` is lexed as a name, and is the wildcard
/// unless a path goes on from it. A name followed by `.` or `(` is a path,
/// otherwise a binding
fn single_pattern(p: &mut Parser) -> Option<MarkClosed> {
    let m = p.open();
    let kind = match p.current() {
        IDENT if p.nth_text(0) == "_" && !matches!(p.nth(1), DOT | L_PAREN) => {
            p.advance();
            WILDCARD_PAT
        }
        INT | FLOAT | STRING | TRUE_KW | FALSE_KW | NONE_KW => {
            p.advance();
            LITERAL_PAT
        }
        MINUS if matches!(p.nth(1), INT | FLOAT) => {
            p.advance();
            p.advance();
            LITERAL_PAT
        }
        IDENT if matches!(p.nth(1), DOT | L_PAREN) => {
            path(p);
            if p.at(L_PAREN) {
                p.advance();
                delimited(p, R_PAREN, pattern);
                TUPLE_STRUCT_PAT
            } else {
                PATH_PAT
            }
        }
        IDENT => {
            name(p);
            BIND_PAT
        }
        _ => {
            p.error_expected("a pattern");
            p.close(m, ERROR);
            return None;
        }
    };
    Some(p.close(m, kind))
}

// helper functions
// -------------------------

/// running `f`, consuming one token as an error if `f` consumed nothing, so
/// the calling loop terminates
fn progress(p: &mut Parser, f: impl FnOnce(&mut Parser)) {
    let before = p.pos;
    f(p);
    if p.pos == before {
        p.advance_with_error(format!("unexpected {}", p.current().describe()));
    }
}

/// parsing `item,*` up to and including `close`, after the opening bracket.
/// A trailing comma is allowed. A `NEWLINE` also ends the list, because
/// there is none inside brackets, so it means the brackets don't match
fn delimited(p: &mut Parser, close: SyntaxKind, mut item: impl FnMut(&mut Parser)) {
    let open = p.prev_range();
    while !p.at(close) && !p.at(NEWLINE) && !p.eof() {
        let before = p.pos;
        item(p);
        if p.pos == before {
            // a line that starts with no element (`def` after `@xdp(`): the
            // bracket was likely never closed, and the list ends here
            if p.newline_since(p.prev_range().end()) {
                unclosed_label(p, close, open);
                break;
            }
            p.advance_with_error(format!("unexpected {}", p.current().describe()));
            continue;
        }
        if p.at(close) || p.eat(COMMA) {
            continue;
        }
        // something else after an element: we report it once and skip to
        // the closing bracket, so the rest of the line gives no more errors
        if p.error_expected(&format!("`,` or {}", close.describe())) {
            unclosed_label(p, close, open);
        }
        skip_to_close(p);
        p.eat(close);
        return;
    }
    expect_closing(p, close, open);
}

/// expecting the bracket that closes the one at `open`
fn expect_closing(p: &mut Parser, close: SyntaxKind, open: TextRange) {
    if !p.eat(close) && p.error_expected(&close.describe()) {
        unclosed_label(p, close, open);
    }
}

/// pointing at the opening bracket of an error, when the error is on
/// another line. Inside brackets the line breaks are ignored, so a missing
/// `)` is often found lines later
fn unclosed_label(p: &mut Parser, close: SyntaxKind, open: TextRange) {
    if p.newline_since(open.end()) {
        let bracket = if close == R_BRACK { "[" } else { "(" };
        p.secondary(open, format!("this `{bracket}` is not closed"));
    }
}

/// skipping up to the closing bracket of a list (not included), past nested
/// brackets, inside an `ERROR` node. Only if it is on the current line: an
/// unclosed list would take the lines after it, and the rule after the list
/// recovers better
fn skip_to_close(p: &mut Parser) {
    let mut depth = 0usize;
    let mut n = 0;
    loop {
        if n > 0 && p.line_break_before(n) {
            return;
        }
        match p.peek(n) {
            NEWLINE | INDENT | DEDENT | EOF => return,
            R_PAREN | R_BRACK if depth == 0 => break,
            L_PAREN | L_BRACK => depth += 1,
            R_PAREN | R_BRACK => depth -= 1,
            _ => {}
        }
        n += 1;
    }
    let m = p.open();
    for _ in 0..n {
        p.advance();
    }
    p.close(m, ERROR);
}

/// the end of a logical line. Anything else is reported and the rest of the
/// line goes in an `ERROR` node
fn expect_newline(p: &mut Parser) {
    if p.eat(NEWLINE) {
        return;
    }
    match p.current() {
        // a closing bracket here closes nothing of this line
        kind @ (R_PAREN | R_BRACK) => {
            let range = p.current_range();
            p.error_at(
                range,
                format!("unmatched {}", kind.describe()),
                "no opening bracket before it".to_string(),
            );
        }
        _ => {
            p.error_expected(&NEWLINE.describe());
        }
    }
    let m = p.open();
    skip_line(p);
    p.close(m, ERROR);
}

/// the `:` at the end of the header of a compound statement or a
/// definition. Without it we skip the rest of the line, so the indented
/// block after it is still parsed as the body
fn header_colon(p: &mut Parser) {
    if p.eat(COLON) {
        return;
    }
    // `if x = 1:`
    if p.error_expected("`:`") && p.at(EQ) {
        p.help("write `==` to compare");
    }
    skip_to_newline(p);
}

/// skipping the rest of the logical line, without its `NEWLINE`, inside an
/// `ERROR` node
fn skip_to_newline(p: &mut Parser) {
    if matches!(p.current(), NEWLINE | INDENT | DEDENT | EOF) {
        return;
    }
    let m = p.open();
    while !matches!(p.current(), NEWLINE | INDENT | DEDENT | EOF) {
        p.advance();
    }
    p.close(m, ERROR);
}

/// skipping to the end of the logical line and the indented block after it,
/// if any. It stops before a `DEDENT`, which closes an enclosing block
fn skip_line(p: &mut Parser) {
    while !p.eof() && !p.at(DEDENT) {
        if p.eat(NEWLINE) {
            if p.at(INDENT) {
                skip_block(p);
            }
            return;
        }
        if p.at(INDENT) {
            skip_block(p);
            return;
        }
        p.advance();
    }
}

/// skipping an indented block, from its `INDENT` to the matching `DEDENT`
fn skip_block(p: &mut Parser) {
    let mut depth = 0usize;
    while !p.eof() {
        match p.current() {
            INDENT => depth += 1,
            DEDENT => depth -= 1,
            _ => {}
        }
        p.advance();
        if depth == 0 {
            return;
        }
    }
}
