//! this module implements the canonical printer of the AST (four-space
//! indentation, spaces around binary operators). The formatter uses it to
//! check its output (see [`crate::fmt`]). Comments and blank lines are not
//! in the AST, so they are not printed. Parentheses are printed only where
//! the source has them, and a missing part is printed as `<?>`.

use crate::ast::*;

/// what a missing part prints as. It doesn't parse, so a printed broken tree
/// is never taken for a correct program
const MISSING: &str = "<?>";

/// the canonical text of a file
pub fn pretty(file: &SourceFile) -> String {
    let mut p = Printer::default();
    let mut previous: Option<Group> = None;
    for item in file.items() {
        let group = Group::of(&item);
        if let Some(previous) = previous
            && (previous != group || group == Group::Definition)
        {
            p.blank_line();
        }
        p.item(&item);
        previous = Some(group);
    }
    p.out
}

/// groups of items, for blank lines. A blank line goes between items of
/// different groups and between two definitions
#[derive(Clone, Copy, PartialEq, Eq)]
enum Group {
    Import,
    Declaration,
    Definition,
}

impl Group {
    fn of(item: &Item) -> Group {
        match item {
            Item::Import(_) => Group::Import,
            Item::MapDecl(_) | Item::EventDecl(_) | Item::ConstDef(_) | Item::TypeAlias(_) => {
                Group::Declaration
            }
            Item::FnDef(_) | Item::StructDef(_) | Item::EnumDef(_) | Item::ProtocolDef(_) => {
                Group::Definition
            }
        }
    }
}

/// the text printed so far and the indentation depth
#[derive(Default)]
struct Printer {
    out: String,
    /// levels of indentation, four spaces each
    indent: usize,
}

impl Printer {
    /// writing a line at the current indentation
    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn blank_line(&mut self) {
        self.out.push('\n');
    }

    /// running `f` one level deeper
    fn indented(&mut self, f: impl FnOnce(&mut Self)) {
        self.indent += 1;
        f(self);
        self.indent -= 1;
    }

    // items

    fn item(&mut self, item: &Item) {
        match item {
            Item::Import(i) => self.line(&import(i)),
            Item::FnDef(f) => self.fn_def(f),
            Item::StructDef(s) => self.owner_def("struct", s),
            Item::EnumDef(e) => self.owner_def("enum", e),
            Item::ProtocolDef(d) => self.owner_def("protocol", d),
            Item::MapDecl(m) => self.map_like("map", m),
            Item::EventDecl(e) => self.map_like("event", e),
            Item::ConstDef(c) => {
                let ty = c.ty().map(|t| format!(": {}", ty(&t))).unwrap_or_default();
                self.line(&format!(
                    "const {}{ty} = {}",
                    name(c.name()),
                    opt_expr(c.value())
                ));
            }
            Item::TypeAlias(a) => self.line(&format!(
                "type {}{} = {}",
                name(a.name()),
                generic_params(a.generic_params()),
                opt_type(a.ty())
            )),
        }
    }

    /// writing each decorator on its own line
    fn decorators(&mut self, decorators: impl Iterator<Item = Decorator>) {
        for d in decorators {
            let path = d.path().map(|p| p.to_string());
            let args = d.arg_list().map(|a| args(&a)).unwrap_or_default();
            self.line(&format!("@{}{args}", path.as_deref().unwrap_or(MISSING)));
        }
    }

    fn fn_def(&mut self, f: &FnDef) {
        self.decorators(f.decorators());
        let comptime = if f.is_comptime() { "comptime " } else { "" };
        let params = f
            .param_list()
            .map(|list| {
                let params: Vec<_> = list.params().map(|p| param(&p)).collect();
                params.join(", ")
            })
            .unwrap_or_default();
        let ret = f
            .ret_type()
            .map(|t| format!(" -> {}", ty(&t)))
            .unwrap_or_default();
        self.line(&format!(
            "{comptime}def {}{}({params}){ret}:",
            name(f.name()),
            generic_params(f.generic_params())
        ));
        self.body(f.body());
    }

    fn owner_def(&mut self, keyword: &str, def: &impl OwnerDef) {
        self.decorators(def.decorators());
        let bases = def
            .bases()
            .map(|b| {
                let types: Vec<_> = b.types().map(|t| ty(&t)).collect();
                format!("({})", types.join(", "))
            })
            .unwrap_or_default();
        self.line(&format!(
            "{keyword} {}{}{bases}:",
            name(def.name()),
            generic_params(def.generic_params())
        ));
        self.indented(|p| {
            // blank lines around methods, not between fields or variants
            let mut previous_was_method = false;
            for (i, member) in def.members().enumerate() {
                let is_method = matches!(member, Member::FnDef(_));
                if i > 0 && (is_method || previous_was_method) {
                    p.blank_line();
                }
                previous_was_method = is_method;
                match member {
                    Member::Field(f) => {
                        p.line(&format!("{}: {}", name(f.name()), opt_type(f.ty())))
                    }
                    Member::Variant(v) => {
                        let fields = v
                            .fields()
                            .map(|f| {
                                let types: Vec<_> = f.types().map(|t| ty(&t)).collect();
                                format!("({})", types.join(", "))
                            })
                            .unwrap_or_default();
                        p.line(&format!("{}{fields}", name(v.name())));
                    }
                    Member::FnDef(f) => p.fn_def(&f),
                    Member::PassStmt(_) => p.line("pass"),
                }
            }
        });
    }

    fn map_like(&mut self, keyword: &str, decl: &impl MapLike) {
        self.decorators(decl.decorators());
        let args = decl.arg_list().map(|a| args(&a)).unwrap_or_default();
        self.line(&format!(
            "{keyword} {}: {}{args}",
            name(decl.name()),
            opt_type(decl.ty())
        ));
    }

    // statements

    /// writing an indented block, always on its own lines (even for
    /// `if n == 3: break`). An empty or missing body is `<?>`
    fn body(&mut self, block: Option<Block>) {
        self.indented(|p| match block {
            Some(block) => {
                let mut empty = true;
                for stmt in block.stmts() {
                    p.stmt(&stmt);
                    empty = false;
                }
                if empty {
                    p.line(MISSING);
                }
            }
            None => p.line(MISSING),
        });
    }

    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::ExprStmt(s) => self.line(&opt_expr(s.expr())),
            Stmt::AssignStmt(s) => {
                let ty = s.ty().map(|t| format!(": {}", ty(&t))).unwrap_or_default();
                // `x: u32` alone has no value, but without an annotation a
                // missing value is an error and we show it
                let value = match (s.value(), s.ty().is_some()) {
                    (Some(v), _) => format!(" = {}", expr(&v)),
                    (None, true) => String::new(),
                    (None, false) => format!(" = {MISSING}"),
                };
                self.line(&format!("{}{ty}{value}", opt_expr(s.target())));
            }
            Stmt::AugAssignStmt(s) => {
                let op = s.op().map(|t| t.text().to_string());
                self.line(&format!(
                    "{} {} {}",
                    opt_expr(s.target()),
                    op.as_deref().unwrap_or(MISSING),
                    opt_expr(s.value())
                ));
            }
            Stmt::ReturnStmt(s) => match s.expr() {
                Some(e) => self.line(&format!("return {}", expr(&e))),
                None => self.line("return"),
            },
            Stmt::IfStmt(s) => {
                self.line(&format!("if {}:", opt_expr(s.condition())));
                self.body(s.then_block());
                for elif in s.elifs() {
                    self.line(&format!("elif {}:", opt_expr(elif.condition())));
                    self.body(elif.block());
                }
                if let Some(else_clause) = s.else_clause() {
                    self.line("else:");
                    self.body(else_clause.block());
                }
            }
            Stmt::ForStmt(s) => {
                self.decorators(s.decorators());
                self.line(&format!(
                    "for {} in {}:",
                    name(s.binding()),
                    opt_expr(s.iterable())
                ));
                self.body(s.body());
            }
            Stmt::WhileStmt(s) => {
                self.decorators(s.decorators());
                self.line(&format!("while {}:", opt_expr(s.condition())));
                self.body(s.body());
            }
            Stmt::MatchStmt(s) => {
                self.line(&format!("match {}:", opt_expr(s.scrutinee())));
                self.indented(|p| {
                    for arm in s.arms() {
                        let guard = arm
                            .guard()
                            .map(|g| format!(" if {}", expr(&g)))
                            .unwrap_or_default();
                        p.line(&format!("case {}{guard}:", opt_pattern(arm.pattern())));
                        p.body(arm.body());
                    }
                });
            }
            Stmt::BreakStmt(_) => self.line("break"),
            Stmt::ContinueStmt(_) => self.line("continue"),
            Stmt::PassStmt(_) => self.line("pass"),
            Stmt::ComptimeStmt(s) => {
                self.line("comptime:");
                self.body(s.body());
            }
        }
    }
}

/// an import on one line, without the parentheses of a `from` import
fn import(i: &Import) -> String {
    let path = i.path().map(|p| p.to_string());
    let path = path.as_deref().unwrap_or(MISSING);
    if !i.is_from() {
        return match i.alias() {
            Some(alias) => format!("import {path} as {}", alias.text()),
            None => format!("import {path}"),
        };
    }
    let names: Vec<_> = i
        .names()
        .map(|n| {
            let name = n.name_ref().map(|r| r.text());
            let name = name.as_deref().unwrap_or(MISSING).to_string();
            match n.alias() {
                Some(alias) => format!("{name} as {}", alias.text()),
                None => name,
            }
        })
        .collect();
    format!("from {path} import {}", names.join(", "))
}

/// a name, or `<?>`
fn name(name: Option<Name>) -> String {
    name.map(|n| n.text()).unwrap_or_else(|| MISSING.into())
}

/// `[T, N: u32]`, or nothing if there is no list
fn generic_params(list: Option<GenericParamList>) -> String {
    let Some(list) = list else {
        return String::new();
    };
    let params: Vec<_> = list
        .params()
        .map(|p| match p.bound() {
            Some(bound) => format!("{}: {}", name(p.name()), ty(&bound)),
            None => name(p.name()),
        })
        .collect();
    format!("[{}]", params.join(", "))
}

fn param(p: &Param) -> String {
    let mut out = name(p.name());
    if let Some(t) = p.ty() {
        out.push_str(&format!(": {}", ty(&t)));
    }
    if let Some(default) = p.default() {
        out.push_str(&format!(" = {}", expr(&default)));
    }
    out
}

/// `(a, name=b)`, no spaces around the `=` of a keyword argument
fn args(list: &ArgList) -> String {
    let args: Vec<_> = list
        .args()
        .map(|a| {
            let value = opt_expr(a.value());
            match a.name() {
                Some(n) => format!("{}={value}", n.text()),
                None => value,
            }
        })
        .collect();
    format!("({})", args.join(", "))
}

/// a type
fn ty(t: &Type) -> String {
    match t {
        Type::PathType(t) => {
            let path = t.path().map(|p| p.to_string());
            let path = path.as_deref().unwrap_or(MISSING);
            match t.generic_args() {
                Some(list) => {
                    let args: Vec<_> = list
                        .args()
                        .map(|a| match a {
                            GenericArg::TypeArg(a) => opt_type(a.ty()),
                            GenericArg::ConstArg(a) => opt_expr(a.expr()),
                        })
                        .collect();
                    format!("{path}[{}]", args.join(", "))
                }
                None => path.to_string(),
            }
        }
        Type::OptionType(t) => format!("{}?", opt_type(t.inner())),
        Type::ArrayType(t) => format!(
            "{}[{}]",
            opt_type(t.element()),
            opt_expr(t.len().and_then(|l| l.expr()))
        ),
    }
}

fn opt_type(t: Option<Type>) -> String {
    t.map(|t| ty(&t)).unwrap_or_else(|| MISSING.into())
}

/// an expression, on one line
fn expr(e: &Expr) -> String {
    match e {
        Expr::Literal(l) => l
            .token()
            .map(|t| t.text().to_string())
            .unwrap_or_else(|| MISSING.into()),
        Expr::NameExpr(n) => n
            .name_ref()
            .map(|r| r.text())
            .unwrap_or_else(|| MISSING.into()),
        Expr::ParenExpr(p) => format!("({})", opt_expr(p.expr())),
        Expr::ArrayExpr(a) => {
            let elements: Vec<_> = a.elements().map(|e| expr(&e)).collect();
            format!("[{}]", elements.join(", "))
        }
        Expr::TupleExpr(t) => {
            let elements: Vec<_> = t.elements().map(|e| expr(&e)).collect();
            elements.join(", ")
        }
        Expr::PrefixExpr(p) => {
            let op = p.op().map(|t| t.text().to_string());
            let op = op.as_deref().unwrap_or(MISSING);
            // `not` is a word and needs a space, `-` and `~` don't
            let space = if op == "not" { " " } else { "" };
            format!("{op}{space}{}", opt_expr(p.operand()))
        }
        Expr::BinExpr(b) => {
            let op = b.op().map(|t| t.text().to_string());
            format!(
                "{} {} {}",
                opt_expr(b.lhs()),
                op.as_deref().unwrap_or(MISSING),
                opt_expr(b.rhs())
            )
        }
        // the parser always gives a call an argument list, `()` is a fallback
        Expr::CallExpr(c) => format!(
            "{}{}",
            opt_expr(c.callee()),
            c.arg_list()
                .map(|a| args(&a))
                .unwrap_or_else(|| "()".into())
        ),
        Expr::IndexExpr(i) => {
            let indices: Vec<_> = i.indices().map(|e| expr(&e)).collect();
            format!("{}[{}]", opt_expr(i.base()), indices.join(", "))
        }
        Expr::FieldExpr(f) => format!(
            "{}.{}",
            opt_expr(f.base()),
            f.field()
                .map(|r| r.text())
                .unwrap_or_else(|| MISSING.into())
        ),
        Expr::TryExpr(t) => format!("{}?", opt_expr(t.expr())),
    }
}

fn opt_expr(e: Option<Expr>) -> String {
    e.map(|e| expr(&e)).unwrap_or_else(|| MISSING.into())
}

/// a pattern
fn pattern(p: &Pattern) -> String {
    match p {
        Pattern::WildcardPat(_) => "_".into(),
        Pattern::LiteralPat(l) => l.text(),
        Pattern::BindPat(b) => name(b.name()),
        Pattern::PathPat(p) => p
            .path()
            .map(|p| p.to_string())
            .unwrap_or_else(|| MISSING.into()),
        Pattern::TupleStructPat(t) => {
            let path = t.path().map(|p| p.to_string());
            let fields: Vec<_> = t.fields().map(|f| pattern(&f)).collect();
            format!(
                "{}({})",
                path.as_deref().unwrap_or(MISSING),
                fields.join(", ")
            )
        }
        Pattern::OrPat(o) => {
            let alternatives: Vec<_> = o.alternatives().map(|a| pattern(&a)).collect();
            alternatives.join(" | ")
        }
    }
}

fn opt_pattern(p: Option<Pattern>) -> String {
    p.map(|p| pattern(&p)).unwrap_or_else(|| MISSING.into())
}
