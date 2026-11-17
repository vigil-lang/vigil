//! this module lowers the typed AST of one file to the HIR, without
//! resolving names. It also desugars `elif` chains, tuple assignments and
//! format strings, and reports the errors that need only the syntax.

use rowan::TextRange;
use vigil_diagnostics::{Diagnostic, Diagnostics, FileId, Span, codes};
use vigil_syntax::ast::{self, AstNode, MapLike, OwnerDef};
use vigil_syntax::{SyntaxKind, SyntaxToken};

use crate::hir::*;

/// the main function of the module: lowering the items of one file into
/// `package`, as the module `module`
pub(crate) fn lower_module(
    package: &mut Package,
    module: ModuleId,
    file: FileId,
    tree: &ast::SourceFile,
    diags: &mut Diagnostics,
) {
    let mut cx = Ctx {
        package,
        module,
        file,
        offset: 0,
        diags,
    };
    for item in tree.items() {
        cx.item(&item);
    }
}

/// the state of lowering one file
struct Ctx<'a> {
    package: &'a mut Package,
    module: ModuleId,
    file: FileId,
    /// added to every text offset. It is non-zero inside a format string,
    /// because each interpolation is parsed alone, from offset 0.
    offset: u32,
    diags: &'a mut Diagnostics,
}

impl Ctx<'_> {
    /// the span in the file of a range of the text being lowered
    fn span(&self, range: TextRange) -> Span {
        let start = u32::from(range.start()) + self.offset;
        let end = u32::from(range.end()) + self.offset;
        Span::new(self.file, start as usize..end as usize)
    }

    fn node_span(&self, node: &impl AstNode) -> Span {
        self.span(node.syntax().text_range())
    }

    /// reporting `a < b < c`, a comparison whose left operand is a comparison
    /// without parentheses. Python would read it as `a < b and b < c`, and
    /// the type checker as `(a < b) < c`, so we reject it here, where the
    /// parentheses are still visible, with one error instead of two type
    /// errors. It returns true when reported
    fn chained_comparison(&mut self, b: &ast::BinExpr) -> bool {
        let op = |b: &ast::BinExpr| b.op().and_then(|t| binary_op(t.kind()));
        let Some(outer) = op(b).filter(|op| op.is_comparison()) else {
            return false;
        };
        let Some(ast::Expr::BinExpr(lhs)) = b.lhs() else {
            return false;
        };
        let Some(inner) = op(&lhs).filter(|op| op.is_comparison()) else {
            return false;
        };
        let text =
            |e: Option<ast::Expr>| e.map(|e| e.syntax().text().to_string().trim().to_string());
        // a longer chain `a < b < c < d` gets the general help, the precise
        // one would be long
        let longer = matches!(lhs.lhs(), Some(ast::Expr::BinExpr(e)) if op(&e).is_some_and(|op| op.is_comparison()));
        let help = match (text(lhs.lhs()), text(lhs.rhs()), text(b.rhs())) {
            (Some(first), Some(middle), Some(last)) if !longer => format!(
                "compare twice: `{first} {} {middle} and {middle} {} {last}`",
                inner.symbol(),
                outer.symbol()
            ),
            _ => "compare twice, joined by `and`: `a < b and b < c`".to_string(),
        };
        let diag = Diagnostic::error("comparisons do not chain in Vigil")
            .with_code(codes::BAD_OPERAND)
            .with_primary(self.node_span(b), "two comparisons in a row")
            .with_help(help);
        self.diags.push(diag);
        true
    }

    fn token_span(&self, token: &SyntaxToken) -> Span {
        self.span(token.text_range())
    }

    /// the identifier of a name. A missing name is an empty identifier at
    /// `fallback`, so later errors have somewhere to point.
    fn ident(&self, name: Option<ast::Name>, fallback: Span) -> Ident {
        match name.and_then(|n| n.ident()) {
            Some(token) => Ident {
                text: token.text().to_string(),
                span: self.token_span(&token),
            },
            None => Ident {
                text: String::new(),
                span: fallback,
            },
        }
    }

    fn name_ref(&self, name: &ast::NameRef) -> Ident {
        Ident {
            text: name.text(),
            span: self.node_span(name),
        }
    }

    fn path(&self, path: &ast::Path) -> PathRef {
        PathRef {
            segments: path.segments().map(|s| self.name_ref(&s)).collect(),
            span: self.node_span(path),
        }
    }

    // items
    // -------------------------

    /// lowering a top-level item and recording it in the module
    fn item(&mut self, item: &ast::Item) {
        let span = self.node_span(item);
        match item {
            ast::Item::Import(i) => self.import(i, span),
            ast::Item::FnDef(f) => {
                let id = self.fn_def(f, None);
                self.push(Item::Fn(id));
            }
            ast::Item::StructDef(s) => self.struct_def(s, span),
            ast::Item::EnumDef(e) => self.enum_def(e, span),
            ast::Item::ProtocolDef(p) => self.protocol(p, span),
            ast::Item::MapDecl(m) => self.map_decl(m, false, span),
            ast::Item::EventDecl(e) => self.map_decl(e, true, span),
            ast::Item::ConstDef(c) => {
                let data = ConstData {
                    name: self.ident(c.name(), span),
                    module: self.module,
                    ty: c.ty().map(|t| self.type_ref(&t)),
                    value: self.const_expr(c.value()),
                };
                let id = self.package.consts.alloc(data);
                self.push(Item::Const(id));
            }
            ast::Item::TypeAlias(a) => {
                let generic_params = self.generic_params(a.generic_params());
                let data = AliasData {
                    name: self.ident(a.name(), span),
                    module: self.module,
                    generic_params,
                    ty: a.ty().map_or(TypeRef::Missing, |t| self.type_ref(&t)),
                };
                let id = self.package.aliases.alloc(data);
                self.push(Item::Alias(id));
            }
        }
    }

    fn generic_params(&mut self, list: Option<ast::GenericParamList>) -> Vec<GenericParam> {
        list.into_iter()
            .flat_map(|l| l.params())
            .map(|p| {
                let span = self.node_span(&p);
                GenericParam {
                    name: self.ident(p.name(), span),
                    bound: p.bound().map(|t| self.type_ref(&t)),
                }
            })
            .collect()
    }

    /// lowering a protocol. We keep only the method signatures, without
    /// bodies and without counting `self` as a parameter.
    fn protocol(&mut self, def: &ast::ProtocolDef, span: Span) {
        let generic_params = self.generic_params(def.generic_params());
        let methods = def
            .members()
            .filter_map(|m| match m {
                ast::Member::FnDef(f) => {
                    let name = self.ident(f.name(), self.node_span(&f));
                    let params: Vec<Option<TypeRef>> = f
                        .param_list()
                        .into_iter()
                        .flat_map(|l| l.params())
                        .filter(|p| p.name().is_none_or(|n| n.text() != "self"))
                        .map(|p| p.ty().map(|t| self.type_ref(&t)))
                        .collect();
                    let ret = f.ret_type().map(|t| self.type_ref(&t));
                    Some(ProtocolMethod {
                        name,
                        arity: params.len(),
                        params,
                        ret,
                    })
                }
                _ => None,
            })
            .collect();
        let id = self.package.protocols.alloc(ProtocolData {
            name: self.ident(def.name(), span),
            module: self.module,
            generic_params,
            methods,
        });
        self.push(Item::Protocol(id));
    }

    /// lowering a `map` or `event` declaration, which takes no decorator.
    /// A kind that is not a plain path becomes an empty path.
    fn map_decl(&mut self, decl: &impl MapLike, is_event: bool, span: Span) {
        for d in decl.decorators() {
            self.diags.push(
                Diagnostic::error("decorators are not allowed here")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(
                        self.node_span(&d),
                        if is_event {
                            "events take no decorator"
                        } else {
                            "maps take no decorator"
                        },
                    ),
            );
        }
        let name = self.ident(decl.name(), span);
        let (kind, args) = match decl.ty().map(|t| self.type_ref(&t)) {
            Some(TypeRef::Path { path, args, .. }) => (path, args),
            _ => (
                PathRef {
                    segments: Vec::new(),
                    span,
                },
                Vec::new(),
            ),
        };
        let params = decl
            .arg_list()
            .into_iter()
            .flat_map(|list| list.args())
            .map(|a| DecoratorArg {
                name: a.name().map(|n| self.ident(Some(n), span)),
                value: self.const_expr(a.value()),
            })
            .collect();
        let id = self.package.maps.alloc(MapData {
            name,
            module: self.module,
            is_event,
            kind,
            args,
            params,
            span,
        });
        self.push(Item::Map(id));
    }

    /// recording an item in the module being lowered
    fn push(&mut self, item: Item) {
        self.package.modules[self.module].items.push(item);
    }

    fn import(&mut self, import: &ast::Import, span: Span) {
        let Some(path) = import.path() else {
            return;
        };
        let path = self.path(&path);
        let hir = if import.is_from() {
            let names = import
                .names()
                .filter_map(|n| {
                    let name = self.name_ref(&n.name_ref()?);
                    let alias = n.alias().map(|a| self.ident(Some(a), span));
                    Some((name, alias))
                })
                .collect();
            Import::From {
                module: path,
                names,
            }
        } else {
            Import::Module {
                path,
                alias: import.alias().map(|a| self.ident(Some(a), span)),
            }
        };
        self.package.modules[self.module].imports.push(hir);
    }

    /// reporting decorators on an enum. A struct keeps its decorators
    /// (`@context`, `@network_order`), checked with the signatures.
    fn owner_checks(&mut self, def: &impl OwnerDef, keeps_decorators: bool) {
        if keeps_decorators {
            return;
        }
        for d in def.decorators() {
            self.diags.push(
                Diagnostic::error("decorators are not allowed here")
                    .with_code(codes::BAD_DECORATOR)
                    .with_primary(self.node_span(&d), "enums take no decorator"),
            );
        }
    }

    /// lowering a struct with its fields and methods. We allocate the struct
    /// first, so its methods can name it as their owner.
    fn struct_def(&mut self, def: &ast::StructDef, span: Span) {
        self.owner_checks(def, true);
        let decorators = def.decorators().map(|d| self.decorator(&d)).collect();
        let generic_params = self.generic_params(def.generic_params());
        let (bases, base_args): (Vec<_>, Vec<_>) = def
            .bases()
            .into_iter()
            .flat_map(|b| b.types())
            .filter_map(|t| match self.type_ref(&t) {
                TypeRef::Path { path, args, .. } => Some((path, args)),
                _ => None,
            })
            .unzip();
        let id = self.package.structs.alloc(StructData {
            name: self.ident(def.name(), span),
            module: self.module,
            decorators,
            generic_params,
            instance: None,
            bases,
            base_args,
            fields: Vec::new(),
            methods: Vec::new(),
        });
        for member in def.members() {
            match member {
                ast::Member::Field(f) => {
                    let field = FieldData {
                        name: self.ident(f.name(), self.node_span(&f)),
                        ty: f.ty().map_or(TypeRef::Missing, |t| self.type_ref(&t)),
                    };
                    self.package.structs[id].fields.push(field);
                }
                ast::Member::FnDef(f) => {
                    let method = self.fn_def(&f, Some(Owner::Struct(id)));
                    self.package.structs[id].methods.push(method);
                }
                ast::Member::Variant(_) | ast::Member::PassStmt(_) => {}
            }
        }
        self.push(Item::Struct(id));
    }

    /// lowering an enum with its variants and methods
    fn enum_def(&mut self, def: &ast::EnumDef, span: Span) {
        self.owner_checks(def, false);
        let generic_params = self.generic_params(def.generic_params());
        let (bases, base_args): (Vec<_>, Vec<_>) = def
            .bases()
            .into_iter()
            .flat_map(|b| b.types())
            .filter_map(|t| match self.type_ref(&t) {
                TypeRef::Path { path, args, .. } => Some((path, args)),
                _ => None,
            })
            .unzip();
        let id = self.package.enums.alloc(EnumData {
            name: self.ident(def.name(), span),
            module: self.module,
            generic_params,
            instance: None,
            bases,
            base_args,
            variants: Vec::new(),
            methods: Vec::new(),
        });
        for member in def.members() {
            match member {
                ast::Member::Variant(v) => {
                    let fields = v
                        .fields()
                        .map(|f| f.types().map(|t| self.type_ref(&t)).collect());
                    let variant = VariantData {
                        name: self.ident(v.name(), self.node_span(&v)),
                        fields,
                    };
                    self.package.enums[id].variants.push(variant);
                }
                ast::Member::FnDef(f) => {
                    let method = self.fn_def(&f, Some(Owner::Enum(id)));
                    self.package.enums[id].methods.push(method);
                }
                ast::Member::Field(_) | ast::Member::PassStmt(_) => {}
            }
        }
        self.push(Item::Enum(id));
    }

    /// lowering a function or method, with its body and contracts
    fn fn_def(&mut self, f: &ast::FnDef, owner: Option<Owner>) -> FnId {
        let span = self.node_span(f);
        let name = self.ident(f.name(), span);
        let generic_params = self.generic_params(f.generic_params());
        let decorators = f.decorators().map(|d| self.decorator(&d)).collect();
        let params = f
            .param_list()
            .into_iter()
            .flat_map(|list| list.params())
            .map(|p| Param {
                name: self.ident(p.name(), self.node_span(&p)),
                ty: p.ty().map(|t| self.type_ref(&t)),
                default: p.default().map(|d| self.const_expr(Some(d))),
            })
            .collect();
        let ret = f.ret_type().map(|t| self.type_ref(&t));

        let mut body = Body::default();
        let mut spans = BodySpans::default();
        let mut contracts = Vec::new();
        {
            let mut lowerer = BodyLowerer {
                cx: self,
                body: &mut body,
                spans: &mut spans,
            };
            if let Some(block) = f.body() {
                body_root(&mut lowerer, &block);
            }
            // contract expressions go in the body, to be typed with it. A
            // contract without exactly one positional argument is skipped
            // here and reported when checking decorators.
            for d in f.decorators() {
                let kind = match d.path().map(|p| p.syntax().text().to_string()).as_deref() {
                    Some("requires") => ContractKind::Requires,
                    Some("ensures") => ContractKind::Ensures,
                    _ => continue,
                };
                let span = lowerer.cx.node_span(&d);
                let mut args = d.arg_list().into_iter().flat_map(|list| list.args());
                if let Some(arg) = args.next().filter(|a| a.name().is_none()) {
                    let expr = lowerer.expr(arg.value(), span);
                    contracts.push(Contract { kind, expr, span });
                }
            }
        }

        self.package.fns.alloc(FnData {
            name,
            module: self.module,
            owner,
            decorators,
            is_comptime: f.is_comptime(),
            generic_params,
            instance: None,
            params,
            ret,
            body,
            spans,
            contracts,
        })
    }

    /// lowering a decorator, with its arguments as constant expressions
    fn decorator(&mut self, d: &ast::Decorator) -> Decorator {
        let span = self.node_span(d);
        let name = d.path().map_or(
            PathRef {
                segments: Vec::new(),
                span,
            },
            |p| self.path(&p),
        );
        let args = d
            .arg_list()
            .into_iter()
            .flat_map(|list| list.args())
            .map(|a| DecoratorArg {
                name: a.name().map(|n| self.ident(Some(n), span)),
                value: self.const_expr(a.value()),
            })
            .collect();
        Decorator { name, args, span }
    }

    // types and constant expressions
    // -------------------------

    /// a type as written
    fn type_ref(&mut self, ty: &ast::Type) -> TypeRef {
        let span = self.node_span(ty);
        match ty {
            ast::Type::PathType(t) => {
                let Some(path) = t.path() else {
                    return TypeRef::Missing;
                };
                let args = t
                    .generic_args()
                    .into_iter()
                    .flat_map(|list| list.args())
                    .map(|a| match a {
                        ast::GenericArg::TypeArg(a) => {
                            TypeArgRef::Type(a.ty().map_or(TypeRef::Missing, |t| self.type_ref(&t)))
                        }
                        ast::GenericArg::ConstArg(a) => {
                            TypeArgRef::Const(self.const_expr(a.expr()))
                        }
                    })
                    .collect();
                TypeRef::Path {
                    path: self.path(&path),
                    args,
                    span,
                }
            }
            ast::Type::OptionType(t) => {
                let inner = t.inner().map_or(TypeRef::Missing, |t| self.type_ref(&t));
                TypeRef::Option(Box::new(inner), span)
            }
            ast::Type::ArrayType(t) => {
                let element = t.element().map_or(TypeRef::Missing, |t| self.type_ref(&t));
                let len = self.const_expr(t.len().and_then(|l| l.expr()));
                TypeRef::Array(Box::new(element), len, span)
            }
        }
    }

    /// an expression as a constant expression. Anything else becomes
    /// [`ConstExpr::Invalid`], reported only if something evaluates it.
    fn const_expr(&mut self, expr: Option<ast::Expr>) -> ConstExpr {
        let Some(expr) = expr else {
            return ConstExpr::Missing;
        };
        let span = self.node_span(&expr);
        match &expr {
            ast::Expr::Literal(l) => match l.token() {
                Some(t) if t.kind() == SyntaxKind::INT => match self.int(&t) {
                    Some(v) => ConstExpr::Int(v, span),
                    None => ConstExpr::Missing,
                },
                Some(t) if t.kind() == SyntaxKind::TRUE_KW => ConstExpr::Bool(true, span),
                Some(t) if t.kind() == SyntaxKind::FALSE_KW => ConstExpr::Bool(false, span),
                Some(t) if t.kind() == SyntaxKind::STRING => {
                    ConstExpr::Str(unescape(t.text()), span)
                }
                _ => ConstExpr::Invalid(span),
            },
            ast::Expr::NameExpr(_) | ast::Expr::FieldExpr(_) => match self.expr_path(&expr) {
                Some(path) => ConstExpr::Path(path),
                None => ConstExpr::Invalid(span),
            },
            ast::Expr::ParenExpr(p) => self.const_expr(p.expr()),
            ast::Expr::ArrayExpr(a) => ConstExpr::Array(
                a.elements().map(|e| self.const_expr(Some(e))).collect(),
                span,
            ),
            ast::Expr::PrefixExpr(p) => {
                let op = p.op().and_then(|t| unary_op(t.kind()));
                match op {
                    Some(op) => ConstExpr::Unary(op, Box::new(self.const_expr(p.operand())), span),
                    None => ConstExpr::Invalid(span),
                }
            }
            ast::Expr::BinExpr(b) if self.chained_comparison(b) => ConstExpr::Missing,
            ast::Expr::BinExpr(b) => {
                let op = b.op().and_then(|t| binary_op(t.kind()));
                match op {
                    Some(op) => ConstExpr::Binary(
                        op,
                        Box::new(self.const_expr(b.lhs())),
                        Box::new(self.const_expr(b.rhs())),
                        span,
                    ),
                    None => ConstExpr::Invalid(span),
                }
            }
            ast::Expr::CallExpr(c) => {
                let path = c.callee().and_then(|callee| self.expr_path(&callee));
                let args: Option<Vec<_>> = c
                    .arg_list()
                    .into_iter()
                    .flat_map(|l| l.args())
                    .map(|a| a.name().is_none().then(|| self.const_expr(a.value())))
                    .collect();
                match (path, args) {
                    (Some(path), Some(args)) => ConstExpr::Call(path, args, span),
                    _ => ConstExpr::Invalid(span),
                }
            }
            _ => ConstExpr::Invalid(span),
        }
    }

    /// the path of `a.b.c`, if the expression has only names and fields
    fn expr_path(&self, expr: &ast::Expr) -> Option<PathRef> {
        match expr {
            ast::Expr::NameExpr(n) => {
                let name = self.name_ref(&n.name_ref()?);
                Some(PathRef {
                    span: name.span,
                    segments: vec![name],
                })
            }
            ast::Expr::FieldExpr(f) => {
                let mut path = self.expr_path(&f.base()?)?;
                path.segments.push(self.name_ref(&f.field()?));
                path.span = self.node_span(f);
                Some(path)
            }
            _ => None,
        }
    }

    /// the value of an integer literal, reporting one too large for 128 bits
    fn int(&mut self, token: &SyntaxToken) -> Option<u128> {
        let value = parse_int(token.text());
        if value.is_none() {
            self.diags.push(
                Diagnostic::error("integer literal is too large")
                    .with_code(codes::LITERAL_OUT_OF_RANGE)
                    .with_primary(self.token_span(token), "does not fit in 128 bits"),
            );
        }
        value
    }
}

/// the prefix operator of a token
fn unary_op(kind: SyntaxKind) -> Option<UnaryOp> {
    Some(match kind {
        SyntaxKind::NOT_KW => UnaryOp::Not,
        SyntaxKind::MINUS => UnaryOp::Neg,
        SyntaxKind::TILDE => UnaryOp::BitNot,
        _ => return None,
    })
}

/// the infix operator of a token
fn binary_op(kind: SyntaxKind) -> Option<BinaryOp> {
    use SyntaxKind::*;
    Some(match kind {
        OR_KW => BinaryOp::Or,
        AND_KW => BinaryOp::And,
        EQ2 => BinaryOp::Eq,
        NEQ => BinaryOp::Ne,
        LT => BinaryOp::Lt,
        LTEQ => BinaryOp::Le,
        GT => BinaryOp::Gt,
        GTEQ => BinaryOp::Ge,
        PIPE => BinaryOp::BitOr,
        CARET => BinaryOp::BitXor,
        AMP => BinaryOp::BitAnd,
        SHL => BinaryOp::Shl,
        SHR => BinaryOp::Shr,
        PLUS => BinaryOp::Add,
        MINUS => BinaryOp::Sub,
        STAR => BinaryOp::Mul,
        SLASH => BinaryOp::Div,
        SLASH2 => BinaryOp::FloorDiv,
        PERCENT => BinaryOp::Rem,
        _ => return None,
    })
}

/// the binary operator of an augmented assignment, `+` for `+=`
fn aug_op(kind: SyntaxKind) -> Option<BinaryOp> {
    use SyntaxKind::*;
    Some(match kind {
        PLUS_EQ => BinaryOp::Add,
        MINUS_EQ => BinaryOp::Sub,
        STAR_EQ => BinaryOp::Mul,
        SLASH_EQ => BinaryOp::Div,
        SLASH2_EQ => BinaryOp::FloorDiv,
        PERCENT_EQ => BinaryOp::Rem,
        AMP_EQ => BinaryOp::BitAnd,
        PIPE_EQ => BinaryOp::BitOr,
        CARET_EQ => BinaryOp::BitXor,
        SHL_EQ => BinaryOp::Shl,
        SHR_EQ => BinaryOp::Shr,
        _ => return None,
    })
}

/// the value of an integer literal (`42`, `0xFF`, `0b1010`, `0o17`, with `_`
/// anywhere), or `None` if it has no digits or does not fit in 128 bits
pub(crate) fn parse_int(text: &str) -> Option<u128> {
    let text = text.replace('_', "");
    let (digits, radix) = match text.get(..2) {
        Some("0x") => (&text[2..], 16),
        Some("0b") => (&text[2..], 2),
        Some("0o") => (&text[2..], 8),
        _ => (&text[..], 10),
    };
    if digits.is_empty() {
        return None;
    }
    u128::from_str_radix(digits, radix).ok()
}

/// the value of a string literal, without quotes and with escapes applied.
/// An unknown escape gives the character after the backslash.
fn unescape(token: &str) -> String {
    let inner = &token[1..token.len().saturating_sub(1).max(1)];
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

// bodies
// -------------------------

/// the state of lowering one body: the file context and the node arenas
struct BodyLowerer<'a, 'b> {
    cx: &'a mut Ctx<'b>,
    body: &'a mut Body,
    spans: &'a mut BodySpans,
}

/// lowering the block of a function as the root of its body
fn body_root(l: &mut BodyLowerer, block: &ast::Block) {
    l.body.root = l.block(Some(block.clone()));
}

impl BodyLowerer<'_, '_> {
    /// adding an expression to the body, with its span
    fn alloc_expr(&mut self, expr: Expr, span: Span) -> ExprId {
        let id = self.body.exprs.alloc(expr);
        self.spans.exprs.insert(id, span);
        id
    }

    fn alloc_stmt(&mut self, stmt: Stmt, span: Span) -> StmtId {
        let id = self.body.stmts.alloc(stmt);
        self.spans.stmts.insert(id, span);
        id
    }

    fn alloc_pat(&mut self, pat: Pat, span: Span) -> PatId {
        let id = self.body.pats.alloc(pat);
        self.spans.pats.insert(id, span);
        id
    }

    /// lowering the statements of a block. It is not a plain map because a
    /// tuple assignment becomes several statements.
    fn block(&mut self, block: Option<ast::Block>) -> Block {
        let mut out = Vec::new();
        for stmt in block.into_iter().flat_map(|b| b.stmts()) {
            match &stmt {
                ast::Stmt::AssignStmt(s)
                    if matches!(s.target(), Some(ast::Expr::TupleExpr(_)))
                        || matches!(s.value(), Some(ast::Expr::TupleExpr(_))) =>
                {
                    self.tuple_assign(s, &mut out)
                }
                _ => out.push(self.stmt(&stmt)),
            }
        }
        out
    }

    /// lowering `a, b = b, a`. We compute every value into a hidden variable
    /// before assigning any target, and declare targets not defined yet.
    /// The hidden names `tuple.N.i` have dots, so they never clash with
    /// source names. `N` is the number of statements so far.
    fn tuple_assign(&mut self, s: &ast::AssignStmt, out: &mut Vec<StmtId>) {
        let span = self.cx.node_span(s);
        let elements = |e: Option<ast::Expr>| -> Vec<ast::Expr> {
            match e {
                Some(ast::Expr::TupleExpr(t)) => t.elements().collect(),
                Some(e) => vec![e],
                None => Vec::new(),
            }
        };
        let (targets, values) = (elements(s.target()), elements(s.value()));
        if let Some(ty) = s.ty() {
            self.cx.diags.push(
                Diagnostic::error("a tuple of variables has no type annotation")
                    .with_code(codes::BAD_ASSIGNMENT)
                    .with_primary(self.cx.node_span(&ty), "remove this annotation")
                    .with_help("to give the variables types, declare them first, one per line"),
            );
            return;
        }
        if targets.len() != values.len() {
            self.cx.diags.push(
                Diagnostic::error(format!(
                    "{} for {}",
                    vigil_diagnostics::plural(targets.len(), "target"),
                    vigil_diagnostics::plural(values.len(), "value")
                ))
                .with_code(codes::BAD_ASSIGNMENT)
                .with_primary(span, "each target takes one value, as in `a, b = b, a`"),
            );
            return;
        }
        let hidden: Vec<String> = (0..values.len())
            .map(|i| format!("tuple.{}.{i}", out.len() + self.body.stmts.len()))
            .collect();
        for (name, value) in hidden.iter().zip(values) {
            let value_span = self.cx.node_span(&value);
            let target = self.alloc_expr(Expr::Name(name.clone()), value_span);
            let value = self.expr(Some(value), value_span);
            out.push(self.alloc_stmt(
                Stmt::Assign {
                    target,
                    ty: None,
                    value: Some(value),
                },
                span,
            ));
        }
        for (name, target) in hidden.iter().zip(targets) {
            let target_span = self.cx.node_span(&target);
            let target = self.expr(Some(target), target_span);
            let value = self.alloc_expr(Expr::Name(name.clone()), target_span);
            out.push(self.alloc_stmt(
                Stmt::Assign {
                    target,
                    ty: None,
                    value: Some(value),
                },
                span,
            ));
        }
    }

    /// lowering one statement
    fn stmt(&mut self, stmt: &ast::Stmt) -> StmtId {
        let span = self.cx.node_span(stmt);
        let hir = match stmt {
            ast::Stmt::ExprStmt(s) => Stmt::Expr(self.expr(s.expr(), span)),
            ast::Stmt::AssignStmt(s) => Stmt::Assign {
                target: self.expr(s.target(), span),
                ty: s.ty().map(|t| self.cx.type_ref(&t)),
                value: s.value().map(|v| self.expr(Some(v), span)),
            },
            ast::Stmt::AugAssignStmt(s) => {
                let op = s.op().and_then(|t| aug_op(t.kind()));
                match op {
                    Some(op) => Stmt::AugAssign {
                        target: self.expr(s.target(), span),
                        op,
                        value: self.expr(s.value(), span),
                    },
                    None => Stmt::Missing,
                }
            }
            ast::Stmt::ReturnStmt(s) => Stmt::Return(s.expr().map(|e| self.expr(Some(e), span))),
            ast::Stmt::IfStmt(s) => {
                let cond = self.expr(s.condition(), span);
                let then = self.block(s.then_block());
                // `elif`s become nested `if`s, built from the last one up,
                // so the `else` of each one is the next `if`
                let mut otherwise = s.else_clause().map(|e| self.block(e.block()));
                let elifs: Vec<_> = s.elifs().collect();
                for elif in elifs.iter().rev() {
                    let elif_span = self.cx.node_span(elif);
                    let cond = self.expr(elif.condition(), elif_span);
                    let then = self.block(elif.block());
                    let nested = self.alloc_stmt(
                        Stmt::If {
                            cond,
                            then,
                            otherwise,
                        },
                        elif_span,
                    );
                    otherwise = Some(vec![nested]);
                }
                Stmt::If {
                    cond,
                    then,
                    otherwise,
                }
            }
            ast::Stmt::ForStmt(s) => Stmt::For {
                decorators: s.decorators().map(|d| self.cx.decorator(&d)).collect(),
                invariants: self.invariants(s.decorators(), span),
                binding: self.cx.ident(s.binding(), span),
                iter: self.expr(s.iterable(), span),
                body: self.block(s.body()),
            },
            ast::Stmt::WhileStmt(s) => Stmt::While {
                decorators: s.decorators().map(|d| self.cx.decorator(&d)).collect(),
                invariants: self.invariants(s.decorators(), span),
                cond: self.expr(s.condition(), span),
                body: self.block(s.body()),
            },
            ast::Stmt::MatchStmt(s) => Stmt::Match {
                scrutinee: self.expr(s.scrutinee(), span),
                arms: s
                    .arms()
                    .map(|arm| {
                        let arm_span = self.cx.node_span(&arm);
                        MatchArm {
                            pat: self.pat(arm.pattern(), arm_span),
                            guard: arm.guard().map(|g| self.expr(Some(g), arm_span)),
                            body: self.block(arm.body()),
                        }
                    })
                    .collect(),
            },
            ast::Stmt::BreakStmt(_) => Stmt::Break,
            ast::Stmt::ContinueStmt(_) => Stmt::Continue,
            ast::Stmt::PassStmt(_) => Stmt::Pass,
            ast::Stmt::ComptimeStmt(s) => Stmt::Comptime(self.block(s.body())),
        };
        self.alloc_stmt(hir, span)
    }

    /// the expressions of the `@invariant(e)` decorators of a loop.
    /// `fallback` is the span of a missing one.
    fn invariants(
        &mut self,
        decorators: impl Iterator<Item = ast::Decorator>,
        fallback: Span,
    ) -> Vec<ExprId> {
        let mut out = Vec::new();
        for d in decorators {
            if d.path().map(|p| p.syntax().text().to_string()).as_deref() != Some("invariant") {
                continue;
            }
            let mut args = d.arg_list().into_iter().flat_map(|list| list.args());
            if let Some(arg) = args.next().filter(|a| a.name().is_none()) {
                out.push(self.expr(arg.value(), fallback));
            }
        }
        out
    }

    /// lowering an expression. `fallback` is the span of a missing one.
    fn expr(&mut self, expr: Option<ast::Expr>, fallback: Span) -> ExprId {
        let Some(expr) = expr else {
            return self.alloc_expr(Expr::Missing, fallback);
        };
        let span = self.cx.node_span(&expr);
        let hir = match &expr {
            ast::Expr::Literal(l) => match l.token() {
                Some(t) => self.literal(&t),
                None => Expr::Missing,
            },
            ast::Expr::NameExpr(n) => match n.name_ref() {
                Some(r) => Expr::Name(r.text()),
                None => Expr::Missing,
            },
            ast::Expr::ParenExpr(p) => return self.expr(p.expr(), span),
            ast::Expr::ArrayExpr(a) => {
                Expr::Array(a.elements().map(|e| self.expr(Some(e), span)).collect())
            }
            // the parser builds a tuple only on the sides of a plain `=`,
            // which [`Self::block`] lowers apart. Here we get one only after a
            // syntax error (`a, b: T = ...`), already reported
            ast::Expr::TupleExpr(_) => Expr::Missing,
            ast::Expr::PrefixExpr(p) => match p.op().and_then(|t| unary_op(t.kind())) {
                Some(op) => Expr::Unary {
                    op,
                    operand: self.expr(p.operand(), span),
                },
                None => Expr::Missing,
            },
            ast::Expr::BinExpr(b) if self.cx.chained_comparison(b) => Expr::Missing,
            ast::Expr::BinExpr(b) => match b.op().and_then(|t| binary_op(t.kind())) {
                Some(op) => Expr::Binary {
                    op,
                    lhs: self.expr(b.lhs(), span),
                    rhs: self.expr(b.rhs(), span),
                },
                None => Expr::Missing,
            },
            ast::Expr::CallExpr(c) => Expr::Call {
                callee: self.expr(c.callee(), span),
                args: c
                    .arg_list()
                    .into_iter()
                    .flat_map(|list| list.args())
                    .map(|a| Arg {
                        name: a.name().map(|n| self.cx.ident(Some(n), span)),
                        value: self.expr(a.value(), span),
                    })
                    .collect(),
            },
            ast::Expr::IndexExpr(i) => Expr::Index {
                base: self.expr(i.base(), span),
                indices: i.indices().map(|e| self.expr(Some(e), span)).collect(),
            },
            ast::Expr::FieldExpr(f) => match f.field() {
                Some(field) => Expr::Field {
                    base: self.expr(f.base(), span),
                    name: self.cx.name_ref(&field),
                },
                None => Expr::Missing,
            },
            ast::Expr::TryExpr(t) => Expr::Try(self.expr(t.expr(), span)),
        };
        self.alloc_expr(hir, span)
    }

    /// the expression of a literal token
    fn literal(&mut self, token: &SyntaxToken) -> Expr {
        match token.kind() {
            SyntaxKind::INT => self.cx.int(token).map_or(Expr::Missing, Expr::Int),
            SyntaxKind::FLOAT => Expr::Float(token.text().replace('_', "").parse().unwrap_or(0.0)),
            SyntaxKind::STRING => Expr::Str(unescape(token.text())),
            SyntaxKind::FSTRING => self.fstring(token),
            SyntaxKind::TRUE_KW => Expr::Bool(true),
            SyntaxKind::FALSE_KW => Expr::Bool(false),
            SyntaxKind::NONE_KW => Expr::None,
            SyntaxKind::ELLIPSIS => Expr::Ellipsis,
            _ => Expr::Missing,
        }
    }

    /// the parts of `f"a {x} b"`: text and interpolated expressions. `{{`
    /// and `}}` are literal braces, and an unmatched brace is reported.
    fn fstring(&mut self, token: &SyntaxToken) -> Expr {
        let text = token.text();
        let start = u32::from(token.text_range().start()) + self.cx.offset;
        // skipping `f` and the quotes
        let content = &text[2..text.len() - 1];
        let content_start = start + 2;
        let at = |cx: &Ctx, from: usize, to: usize| {
            Span::new(
                cx.file,
                (content_start as usize + from)..(content_start as usize + to),
            )
        };

        let mut parts = Vec::new();
        let mut literal = String::new();
        let bytes = content.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'{' if bytes.get(i + 1) == Some(&b'{') => {
                    literal.push('{');
                    i += 2;
                }
                b'}' if bytes.get(i + 1) == Some(&b'}') => {
                    literal.push('}');
                    i += 2;
                }
                b'{' => {
                    let Some(len) = content[i + 1..].find('}') else {
                        self.cx.diags.push(
                            Diagnostic::error("unclosed `{` in format string")
                                .with_code(codes::SYNTAX_ERROR)
                                .with_primary(at(self.cx, i, i + 1), "this `{` is never closed")
                                .with_help("write `{{` for a literal brace"),
                        );
                        break;
                    };
                    if !literal.is_empty() {
                        parts.push(FStrPart::Text(std::mem::take(&mut literal)));
                    }
                    let expr_start = i + 1;
                    let source = &content[expr_start..expr_start + len];
                    parts.push(FStrPart::Expr(self.interpolation(
                        source,
                        content_start + expr_start as u32,
                        at(self.cx, i, expr_start + len + 1),
                    )));
                    i = expr_start + len + 1;
                }
                b'}' => {
                    self.cx.diags.push(
                        Diagnostic::error("unmatched `}` in format string")
                            .with_code(codes::SYNTAX_ERROR)
                            .with_primary(at(self.cx, i, i + 1), "no `{` before it")
                            .with_help("write `}}` for a literal brace"),
                    );
                    i += 1;
                }
                // escapes, as in a plain string
                b'\\' if i + 1 < bytes.len() => {
                    let c = content[i + 1..].chars().next().unwrap();
                    literal.push(match c {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '0' => '\0',
                        other => other,
                    });
                    i += 1 + c.len_utf8();
                }
                _ => {
                    let c = content[i..].chars().next().unwrap();
                    literal.push(c);
                    i += c.len_utf8();
                }
            }
        }
        if !literal.is_empty() {
            parts.push(FStrPart::Text(literal));
        }
        Expr::FStr(parts)
    }

    /// the expression of an interpolation, parsed from its own text, which
    /// starts at `offset` in the file. `braces` is the span of the whole
    /// `{...}`, used for an empty one.
    fn interpolation(&mut self, source: &str, offset: u32, braces: Span) -> ExprId {
        let parse = vigil_syntax::parse_expr(source);
        let saved = std::mem::replace(&mut self.cx.offset, offset);
        for error in parse.errors() {
            let mut error = error.clone();
            error.message = error
                .message
                .replace("the end of the file", "the end of the interpolation");
            let mut diag = error.to_diagnostic(self.cx.file);
            // the parser located the error in the interpolation text, so we
            // move it to its place in the file
            diag.labels[0].span = self.cx.span(error.range);
            self.cx.diags.push(diag);
        }
        let id = self.expr(parse.expr(), braces);
        self.cx.offset = saved;
        id
    }

    /// lowering a pattern. `fallback` is the span of a missing one, and a
    /// literal pattern is recognized by its text.
    fn pat(&mut self, pat: Option<ast::Pattern>, fallback: Span) -> PatId {
        let Some(pat) = pat else {
            return self.alloc_pat(Pat::Missing, fallback);
        };
        let span = self.cx.node_span(&pat);
        let hir = match &pat {
            ast::Pattern::WildcardPat(_) => Pat::Wild,
            ast::Pattern::BindPat(b) => Pat::Bind(self.cx.ident(b.name(), span)),
            ast::Pattern::LiteralPat(l) => {
                let text = l.text();
                let (negative, digits) = match text.strip_prefix('-') {
                    Some(rest) => (true, rest.trim_start()),
                    None => (false, text.as_str()),
                };
                match digits {
                    "True" => Pat::Bool(true),
                    "False" => Pat::Bool(false),
                    "None" => Pat::None,
                    d if d.starts_with(['"', '\'']) => Pat::Str,
                    d if d.contains('.') && !d.starts_with("0x") => Pat::Float,
                    d => match parse_int(d) {
                        Some(value) => Pat::Int { value, negative },
                        None => Pat::Missing,
                    },
                }
            }
            ast::Pattern::PathPat(p) => match p.path() {
                Some(path) => Pat::Path(self.cx.path(&path)),
                None => Pat::Missing,
            },
            ast::Pattern::TupleStructPat(t) => match t.path() {
                Some(path) => Pat::TupleStruct {
                    path: self.cx.path(&path),
                    fields: t.fields().map(|f| self.pat(Some(f), span)).collect(),
                },
                None => Pat::Missing,
            },
            ast::Pattern::OrPat(o) => {
                Pat::Or(o.alternatives().map(|a| self.pat(Some(a), span)).collect())
            }
        };
        self.alloc_pat(hir, span)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_integer_literals() {
        assert_eq!(parse_int("1_000"), Some(1000));
        assert_eq!(parse_int("0xFF_FF"), Some(0xFFFF));
        assert_eq!(parse_int("0b101"), Some(5));
        assert_eq!(parse_int("0o17"), Some(15));
        assert_eq!(parse_int("0x"), None);
        assert_eq!(parse_int(&"9".repeat(40)), None);
    }

    #[test]
    fn unescapes_strings() {
        assert_eq!(unescape(r#""a\nb\"c""#), "a\nb\"c");
        assert_eq!(unescape("'x'"), "x");
    }
}
