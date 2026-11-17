//! this module defines a typed view of the syntax tree. Each type wraps a
//! [`SyntaxNode`] (enums like [`Expr`] wrap one of several kinds), and
//! accessors walk the children each time. A part is found by kind, not by
//! position, and is an `Option` because a tree with errors may miss it.
//! When two children have the same kind, the second is found as the second
//! child of that kind (see [`AssignStmt::value`]).

use crate::syntax_kind::SyntaxKind::{self, *};
use crate::syntax_kind::{SyntaxNode, SyntaxToken};

/// a typed node of the tree
pub trait AstNode: Sized {
    /// can a node of this kind be viewed as `Self`?
    fn can_cast(kind: SyntaxKind) -> bool;
    /// the node as `Self`, if its kind fits
    fn cast(node: SyntaxNode) -> Option<Self>;
    /// the untyped node, for its range, text and tokens
    fn syntax(&self) -> &SyntaxNode;
}

// a struct wrapping a node of one kind
macro_rules! node {
    ($name:ident, $kind:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(SyntaxNode);

        impl AstNode for $name {
            fn can_cast(kind: SyntaxKind) -> bool {
                kind == $kind
            }
            fn cast(node: SyntaxNode) -> Option<Self> {
                Self::can_cast(node.kind()).then(|| $name(node))
            }
            fn syntax(&self) -> &SyntaxNode {
                &self.0
            }
        }
    };
}

// an enum over node types, one variant per type with the same name. The
// kinds are distinct, so the order of the variants doesn't matter
macro_rules! enum_node {
    ($name:ident { $($variant:ident),* $(,)? }) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant($variant),)*
        }

        impl AstNode for $name {
            fn can_cast(kind: SyntaxKind) -> bool {
                $($variant::can_cast(kind))||*
            }
            fn cast(node: SyntaxNode) -> Option<Self> {
                $(if $variant::can_cast(node.kind()) {
                    return Some($name::$variant($variant(node)));
                })*
                None
            }
            fn syntax(&self) -> &SyntaxNode {
                match self {
                    $($name::$variant(n) => n.syntax(),)*
                }
            }
        }
    };
}

/// the first child of `parent` that is an `N`
fn child<N: AstNode>(parent: &SyntaxNode) -> Option<N> {
    parent.children().find_map(N::cast)
}

/// the children of `parent` that are `N`s, in order
fn children<N: AstNode>(parent: &SyntaxNode) -> impl Iterator<Item = N> + use<N> {
    parent.children().filter_map(N::cast)
}

/// the first token of `kind` directly under `parent`
fn token(parent: &SyntaxNode, kind: SyntaxKind) -> Option<SyntaxToken> {
    parent
        .children_with_tokens()
        .filter_map(|e| e.into_token())
        .find(|t| t.kind() == kind)
}

/// the first significant token directly under `parent`, used for operators
/// (operands are nodes) and literals
fn first_token(parent: &SyntaxNode) -> Option<SyntaxToken> {
    parent
        .children_with_tokens()
        .filter_map(|e| e.into_token())
        .find(|t| !t.kind().is_trivia())
}

// source file and items
// -------------------------

node!(SourceFile, SOURCE_FILE);
impl SourceFile {
    pub fn items(&self) -> impl Iterator<Item = Item> + use<> {
        children(&self.0)
    }
}

enum_node!(Item {
    Import,
    FnDef,
    StructDef,
    EnumDef,
    ProtocolDef,
    MapDecl,
    EventDecl,
    ConstDef,
    TypeAlias,
});

node!(Import, IMPORT);
impl Import {
    /// is it `from path import ...` and not `import path`?
    pub fn is_from(&self) -> bool {
        token(&self.0, FROM_KW).is_some()
    }
    pub fn path(&self) -> Option<Path> {
        child(&self.0)
    }
    pub fn names(&self) -> impl Iterator<Item = ImportName> + use<> {
        children(&self.0)
    }
    /// the alias of `import path as alias`. `None` for a `from` import, whose
    /// aliases are in the [`ImportName`] nodes
    pub fn alias(&self) -> Option<Name> {
        child(&self.0)
    }
}

node!(ImportName, IMPORT_NAME);
impl ImportName {
    pub fn name_ref(&self) -> Option<NameRef> {
        child(&self.0)
    }
    pub fn alias(&self) -> Option<Name> {
        child(&self.0)
    }
}

node!(Decorator, DECORATOR);
impl Decorator {
    pub fn path(&self) -> Option<Path> {
        child(&self.0)
    }
    pub fn arg_list(&self) -> Option<ArgList> {
        child(&self.0)
    }
}

node!(FnDef, FN_DEF);
impl FnDef {
    pub fn decorators(&self) -> impl Iterator<Item = Decorator> + use<> {
        children(&self.0)
    }
    pub fn is_comptime(&self) -> bool {
        token(&self.0, COMPTIME_KW).is_some()
    }
    pub fn name(&self) -> Option<Name> {
        child(&self.0)
    }
    pub fn generic_params(&self) -> Option<GenericParamList> {
        child(&self.0)
    }
    pub fn param_list(&self) -> Option<ParamList> {
        child(&self.0)
    }
    pub fn ret_type(&self) -> Option<Type> {
        child::<RetType>(&self.0).and_then(|r| child(&r.0))
    }
    pub fn body(&self) -> Option<Block> {
        child(&self.0)
    }
}

// the `-> type` of a function, `FnDef::ret_type` looks inside it
node!(RetType, RET_TYPE);

node!(GenericParamList, GENERIC_PARAM_LIST);
impl GenericParamList {
    pub fn params(&self) -> impl Iterator<Item = GenericParam> + use<> {
        children(&self.0)
    }
}

node!(GenericParam, GENERIC_PARAM);
impl GenericParam {
    pub fn name(&self) -> Option<Name> {
        child(&self.0)
    }
    /// the protocol bound of a type parameter or the type of a constant
    /// parameter. Same syntax, the type checker tells them apart
    pub fn bound(&self) -> Option<Type> {
        child(&self.0)
    }
}

node!(ParamList, PARAM_LIST);
impl ParamList {
    pub fn params(&self) -> impl Iterator<Item = Param> + use<> {
        children(&self.0)
    }
}

node!(Param, PARAM);
impl Param {
    pub fn name(&self) -> Option<Name> {
        child(&self.0)
    }
    pub fn ty(&self) -> Option<Type> {
        child(&self.0)
    }
    /// the default value, after `=`
    pub fn default(&self) -> Option<Expr> {
        child(&self.0)
    }
}

/// the common shape of `struct`, `enum` and `protocol` definitions. The
/// allowed members differ, but here they are all [`Member`]s
pub trait OwnerDef: AstNode {
    fn decorators(&self) -> impl Iterator<Item = Decorator> {
        children(self.syntax())
    }
    fn name(&self) -> Option<Name> {
        child(self.syntax())
    }
    fn generic_params(&self) -> Option<GenericParamList> {
        child(self.syntax())
    }
    /// the protocols in parentheses after the name
    fn bases(&self) -> Option<BaseList> {
        child(self.syntax())
    }
    /// the members in order, none if the member list is missing
    fn members(&self) -> impl Iterator<Item = Member> {
        child::<MemberList>(self.syntax())
            .into_iter()
            .flat_map(|list| children(&list.0))
    }
}

node!(StructDef, STRUCT_DEF);
impl OwnerDef for StructDef {}
node!(EnumDef, ENUM_DEF);
impl OwnerDef for EnumDef {}
node!(ProtocolDef, PROTOCOL_DEF);
impl OwnerDef for ProtocolDef {}

node!(BaseList, BASE_LIST);
impl BaseList {
    pub fn types(&self) -> impl Iterator<Item = Type> + use<> {
        children(&self.0)
    }
}

node!(MemberList, MEMBER_LIST);

enum_node!(Member {
    Field,
    Variant,
    FnDef,
    PassStmt
});

node!(Field, FIELD);
impl Field {
    pub fn name(&self) -> Option<Name> {
        child(&self.0)
    }
    pub fn ty(&self) -> Option<Type> {
        child(&self.0)
    }
}

node!(Variant, VARIANT);
impl Variant {
    pub fn name(&self) -> Option<Name> {
        child(&self.0)
    }
    /// the payload types, if the variant has a parenthesised list
    pub fn fields(&self) -> Option<VariantFields> {
        child(&self.0)
    }
}

node!(VariantFields, VARIANT_FIELDS);
impl VariantFields {
    pub fn types(&self) -> impl Iterator<Item = Type> + use<> {
        children(&self.0)
    }
}

/// the common shape of `map` and `event` declarations
pub trait MapLike: AstNode {
    fn decorators(&self) -> impl Iterator<Item = Decorator> {
        children(self.syntax())
    }
    fn name(&self) -> Option<Name> {
        child(self.syntax())
    }
    fn ty(&self) -> Option<Type> {
        child(self.syntax())
    }
    /// the arguments after the type, `(pinned=True)`
    fn arg_list(&self) -> Option<ArgList> {
        child(self.syntax())
    }
}

node!(MapDecl, MAP_DECL);
impl MapLike for MapDecl {}
node!(EventDecl, EVENT_DECL);
impl MapLike for EventDecl {}

node!(ConstDef, CONST_DEF);
impl ConstDef {
    pub fn name(&self) -> Option<Name> {
        child(&self.0)
    }
    pub fn ty(&self) -> Option<Type> {
        child(&self.0)
    }
    pub fn value(&self) -> Option<Expr> {
        child(&self.0)
    }
}

node!(TypeAlias, TYPE_ALIAS);
impl TypeAlias {
    pub fn name(&self) -> Option<Name> {
        child(&self.0)
    }
    pub fn generic_params(&self) -> Option<GenericParamList> {
        child(&self.0)
    }
    pub fn ty(&self) -> Option<Type> {
        child(&self.0)
    }
}

// names and paths
// -------------------------

node!(Name, NAME);
impl Name {
    pub fn ident(&self) -> Option<SyntaxToken> {
        token(&self.0, IDENT)
    }
    /// the name as a string, empty if the identifier is missing
    pub fn text(&self) -> String {
        self.ident()
            .map(|t| t.text().to_string())
            .unwrap_or_default()
    }
}

node!(NameRef, NAME_REF);
impl NameRef {
    pub fn ident(&self) -> Option<SyntaxToken> {
        token(&self.0, IDENT)
    }
    /// the name as a string, empty if the identifier is missing
    pub fn text(&self) -> String {
        self.ident()
            .map(|t| t.text().to_string())
            .unwrap_or_default()
    }
}

node!(Path, PATH);
impl Path {
    pub fn segments(&self) -> impl Iterator<Item = NameRef> + use<> {
        children(&self.0)
    }
}

/// the segments joined by dots, without trivia
impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let segments: Vec<_> = self.segments().map(|s| s.text()).collect();
        f.write_str(&segments.join("."))
    }
}

// types
// -------------------------

enum_node!(Type {
    PathType,
    OptionType,
    ArrayType
});

node!(PathType, PATH_TYPE);
impl PathType {
    pub fn path(&self) -> Option<Path> {
        child(&self.0)
    }
    pub fn generic_args(&self) -> Option<GenericArgList> {
        child(&self.0)
    }
}

node!(OptionType, OPTION_TYPE);
impl OptionType {
    pub fn inner(&self) -> Option<Type> {
        child(&self.0)
    }
}

node!(ArrayType, ARRAY_TYPE);
impl ArrayType {
    /// the type of the elements, `u8[4]` in `u8[4][2]`. A single bracket
    /// after a path is a [`PathType`] with a generic argument, not an array
    /// (see the parser's `type_`)
    pub fn element(&self) -> Option<Type> {
        child(&self.0)
    }
    pub fn len(&self) -> Option<ConstArg> {
        child(&self.0)
    }
}

node!(GenericArgList, GENERIC_ARG_LIST);
impl GenericArgList {
    pub fn args(&self) -> impl Iterator<Item = GenericArg> + use<> {
        children(&self.0)
    }
}

enum_node!(GenericArg { TypeArg, ConstArg });

node!(TypeArg, TYPE_ARG);
impl TypeArg {
    pub fn ty(&self) -> Option<Type> {
        child(&self.0)
    }
}

node!(ConstArg, CONST_ARG);
impl ConstArg {
    pub fn expr(&self) -> Option<Expr> {
        child(&self.0)
    }
}

// statements
// -------------------------

node!(Block, BLOCK);
impl Block {
    pub fn stmts(&self) -> impl Iterator<Item = Stmt> + use<> {
        children(&self.0)
    }
}

enum_node!(Stmt {
    ExprStmt,
    AssignStmt,
    AugAssignStmt,
    ReturnStmt,
    IfStmt,
    ForStmt,
    WhileStmt,
    MatchStmt,
    BreakStmt,
    ContinueStmt,
    PassStmt,
    ComptimeStmt,
});

node!(ExprStmt, EXPR_STMT);
impl ExprStmt {
    pub fn expr(&self) -> Option<Expr> {
        child(&self.0)
    }
}

node!(AssignStmt, ASSIGN_STMT);
/// `target = value`, `target: type = value`, or `target: type`. A tuple
/// target `a, b = ...` is a [`TupleExpr`]
impl AssignStmt {
    pub fn target(&self) -> Option<Expr> {
        child(&self.0)
    }
    /// the annotation of `target: type = value`
    pub fn ty(&self) -> Option<Type> {
        child(&self.0)
    }
    /// the second expression child, the first is the target. An annotation
    /// is a type, not an expression, so it doesn't count
    pub fn value(&self) -> Option<Expr> {
        children(&self.0).nth(1)
    }
}

node!(AugAssignStmt, AUG_ASSIGN_STMT);
impl AugAssignStmt {
    pub fn target(&self) -> Option<Expr> {
        child(&self.0)
    }
    /// the operator token, `+=` or another
    pub fn op(&self) -> Option<SyntaxToken> {
        self.0
            .children_with_tokens()
            .filter_map(|e| e.into_token())
            .find(|t| t.kind().is_aug_assign())
    }
    pub fn value(&self) -> Option<Expr> {
        children(&self.0).nth(1)
    }
}

node!(ReturnStmt, RETURN_STMT);
impl ReturnStmt {
    pub fn expr(&self) -> Option<Expr> {
        child(&self.0)
    }
}

node!(IfStmt, IF_STMT);
impl IfStmt {
    pub fn condition(&self) -> Option<Expr> {
        child(&self.0)
    }
    /// the block after the condition. The `elif` and `else` blocks are in
    /// their own nodes
    pub fn then_block(&self) -> Option<Block> {
        child(&self.0)
    }
    pub fn elifs(&self) -> impl Iterator<Item = ElifClause> + use<> {
        children(&self.0)
    }
    pub fn else_clause(&self) -> Option<ElseClause> {
        child(&self.0)
    }
}

node!(ElifClause, ELIF_CLAUSE);
impl ElifClause {
    pub fn condition(&self) -> Option<Expr> {
        child(&self.0)
    }
    pub fn block(&self) -> Option<Block> {
        child(&self.0)
    }
}

node!(ElseClause, ELSE_CLAUSE);
impl ElseClause {
    pub fn block(&self) -> Option<Block> {
        child(&self.0)
    }
}

node!(ForStmt, FOR_STMT);
impl ForStmt {
    pub fn decorators(&self) -> impl Iterator<Item = Decorator> + use<> {
        children(&self.0)
    }
    /// the name bound by the loop, `i` in `for i in range(4)`
    pub fn binding(&self) -> Option<Name> {
        child(&self.0)
    }
    pub fn iterable(&self) -> Option<Expr> {
        child(&self.0)
    }
    pub fn body(&self) -> Option<Block> {
        child(&self.0)
    }
}

node!(WhileStmt, WHILE_STMT);
impl WhileStmt {
    pub fn decorators(&self) -> impl Iterator<Item = Decorator> + use<> {
        children(&self.0)
    }
    pub fn condition(&self) -> Option<Expr> {
        child(&self.0)
    }
    pub fn body(&self) -> Option<Block> {
        child(&self.0)
    }
}

node!(MatchStmt, MATCH_STMT);
impl MatchStmt {
    /// the expression matched on
    pub fn scrutinee(&self) -> Option<Expr> {
        child(&self.0)
    }
    pub fn arms(&self) -> impl Iterator<Item = MatchArm> + use<> {
        children(&self.0)
    }
}

node!(MatchArm, MATCH_ARM);
impl MatchArm {
    pub fn pattern(&self) -> Option<Pattern> {
        child(&self.0)
    }
    /// the condition after `if`, the only expression child (a pattern is not
    /// an expression)
    pub fn guard(&self) -> Option<Expr> {
        child(&self.0)
    }
    pub fn body(&self) -> Option<Block> {
        child(&self.0)
    }
}

node!(BreakStmt, BREAK_STMT);
node!(ContinueStmt, CONTINUE_STMT);
node!(PassStmt, PASS_STMT);

node!(ComptimeStmt, COMPTIME_STMT);
impl ComptimeStmt {
    pub fn body(&self) -> Option<Block> {
        child(&self.0)
    }
}

// expressions
// -------------------------

enum_node!(Expr {
    Literal,
    NameExpr,
    ParenExpr,
    ArrayExpr,
    TupleExpr,
    PrefixExpr,
    BinExpr,
    CallExpr,
    IndexExpr,
    FieldExpr,
    TryExpr,
});

node!(Literal, LITERAL);
impl Literal {
    /// the token of the literal, its kind says which sort it is
    pub fn token(&self) -> Option<SyntaxToken> {
        first_token(&self.0)
    }
}

node!(NameExpr, NAME_EXPR);
impl NameExpr {
    pub fn name_ref(&self) -> Option<NameRef> {
        child(&self.0)
    }
}

node!(ParenExpr, PAREN_EXPR);
impl ParenExpr {
    pub fn expr(&self) -> Option<Expr> {
        child(&self.0)
    }
}

node!(ArrayExpr, ARRAY_EXPR);
impl ArrayExpr {
    pub fn elements(&self) -> impl Iterator<Item = Expr> + use<> {
        children(&self.0)
    }
}

node!(TupleExpr, TUPLE_EXPR);
impl TupleExpr {
    pub fn elements(&self) -> impl Iterator<Item = Expr> + use<> {
        children(&self.0)
    }
}

node!(PrefixExpr, PREFIX_EXPR);
impl PrefixExpr {
    /// the operator: `not`, `-` or `~`
    pub fn op(&self) -> Option<SyntaxToken> {
        first_token(&self.0)
    }
    pub fn operand(&self) -> Option<Expr> {
        child(&self.0)
    }
}

node!(BinExpr, BIN_EXPR);
impl BinExpr {
    pub fn lhs(&self) -> Option<Expr> {
        child(&self.0)
    }
    /// the operator, the only token directly under the node
    pub fn op(&self) -> Option<SyntaxToken> {
        first_token(&self.0)
    }
    /// the second expression child, the first is the left operand
    pub fn rhs(&self) -> Option<Expr> {
        children(&self.0).nth(1)
    }
}

node!(CallExpr, CALL_EXPR);
impl CallExpr {
    pub fn callee(&self) -> Option<Expr> {
        child(&self.0)
    }
    pub fn arg_list(&self) -> Option<ArgList> {
        child(&self.0)
    }
}

node!(ArgList, ARG_LIST);
impl ArgList {
    pub fn args(&self) -> impl Iterator<Item = Arg> + use<> {
        children(&self.0)
    }
}

node!(Arg, ARG);
impl Arg {
    /// the name of a keyword argument `name=value`, `None` for a positional
    /// one
    pub fn name(&self) -> Option<Name> {
        child(&self.0)
    }
    pub fn value(&self) -> Option<Expr> {
        child(&self.0)
    }
}

node!(IndexExpr, INDEX_EXPR);
impl IndexExpr {
    pub fn base(&self) -> Option<Expr> {
        child(&self.0)
    }
    /// the indices. They are in their own `INDEX_LIST`, so they are not
    /// confused with the base
    pub fn indices(&self) -> impl Iterator<Item = Expr> + use<> {
        child::<IndexList>(&self.0)
            .into_iter()
            .flat_map(|list| children(&list.0))
    }
}

node!(IndexList, INDEX_LIST);

node!(FieldExpr, FIELD_EXPR);
impl FieldExpr {
    pub fn base(&self) -> Option<Expr> {
        child(&self.0)
    }
    pub fn field(&self) -> Option<NameRef> {
        child(&self.0)
    }
}

node!(TryExpr, TRY_EXPR);
/// `expr?`
impl TryExpr {
    pub fn expr(&self) -> Option<Expr> {
        child(&self.0)
    }
}

// patterns
// -------------------------

enum_node!(Pattern {
    WildcardPat,
    LiteralPat,
    BindPat,
    PathPat,
    TupleStructPat,
    OrPat,
});

node!(WildcardPat, WILDCARD_PAT);

node!(LiteralPat, LITERAL_PAT);
impl LiteralPat {
    /// the literal as written, with a leading `-` (a separate token)
    pub fn text(&self) -> String {
        self.0
            .children_with_tokens()
            .filter_map(|e| e.into_token())
            .filter(|t| !t.kind().is_trivia())
            .map(|t| t.text().to_string())
            .collect()
    }
}

node!(BindPat, BIND_PAT);
impl BindPat {
    pub fn name(&self) -> Option<Name> {
        child(&self.0)
    }
}

node!(PathPat, PATH_PAT);
impl PathPat {
    pub fn path(&self) -> Option<Path> {
        child(&self.0)
    }
}

node!(TupleStructPat, TUPLE_STRUCT_PAT);
/// `Some(x)`: a path and the patterns of its fields
impl TupleStructPat {
    pub fn path(&self) -> Option<Path> {
        child(&self.0)
    }
    pub fn fields(&self) -> impl Iterator<Item = Pattern> + use<> {
        children(&self.0)
    }
}

node!(OrPat, OR_PAT);
impl OrPat {
    pub fn alternatives(&self) -> impl Iterator<Item = Pattern> + use<> {
        children(&self.0)
    }
}
