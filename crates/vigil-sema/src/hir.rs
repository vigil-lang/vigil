//! this module defines the HIR, a package after lowering, with names still
//! unresolved. Items live in the arenas of the [`Package`], and the nodes of a
//! function in its [`Body`]. Types stay as written ([`TypeRef`]) and spans
//! are kept apart, in [`BodySpans`].

use la_arena::{Arena, ArenaMap, Idx};
use vigil_diagnostics::{FileId, Span};

/// a module of the package
pub type ModuleId = Idx<ModuleData>;
pub type FnId = Idx<FnData>;
pub type StructId = Idx<StructData>;
pub type EnumId = Idx<EnumData>;
pub type ConstId = Idx<ConstData>;
pub type AliasId = Idx<AliasData>;
pub type MapId = Idx<MapData>;
pub type ProtocolId = Idx<ProtocolData>;

/// an expression of a [`Body`]
pub type ExprId = Idx<Expr>;
pub type StmtId = Idx<Stmt>;
pub type PatId = Idx<Pat>;

/// a name with the span where it is written
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ident {
    /// the name, empty where the parser recovered from an error
    pub text: String,
    /// where it is written
    pub span: Span,
}

/// a dotted path: `std.net`, `XdpAction.DROP`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathRef {
    pub segments: Vec<Ident>,
    pub span: Span,
}

impl ModuleData {
    /// whether the module is in the standard library, which may use
    /// `@helper` and `@context`. Here we look at where the module was loaded
    /// from, not at its path, because a user module could be named `std`
    pub fn is_std(&self) -> bool {
        self.std
    }
}

impl std::fmt::Display for PathRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let segments: Vec<_> = self.segments.iter().map(|s| s.text.as_str()).collect();
        f.write_str(&segments.join("."))
    }
}

/// every item of a package, in arenas shared by all its modules. Instances
/// of generic items are added to the same arenas when needed.
#[derive(Clone, Debug, Default)]
pub struct Package {
    pub modules: Arena<ModuleData>,
    pub fns: Arena<FnData>,
    pub structs: Arena<StructData>,
    pub enums: Arena<EnumData>,
    pub consts: Arena<ConstData>,
    pub aliases: Arena<AliasData>,
    pub maps: Arena<MapData>,
    pub protocols: Arena<ProtocolData>,
}

/// a module: one source file
#[derive(Clone, Debug)]
pub struct ModuleData {
    /// the dotted path of the module, relative to the project's `src/`
    pub path: Vec<String>,
    /// the source file
    pub file: FileId,
    /// the items it defines, in order
    pub items: Vec<Item>,
    /// its imports, in order
    pub imports: Vec<Import>,
    /// the names of declarations not supported yet, already reported, so
    /// they resolve silently. Lowering puts nothing here at present.
    pub unsupported: Vec<Ident>,
    /// whether it was loaded as part of the standard library, see
    /// [`crate::ModuleInput::std`]
    pub std: bool,
}

/// a top-level definition of a module
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Item {
    Fn(FnId),
    Struct(StructId),
    Enum(EnumId),
    Const(ConstId),
    Alias(AliasId),
    Map(MapId),
    Protocol(ProtocolId),
}

/// an import statement
#[derive(Clone, Debug)]
pub enum Import {
    /// `from module import name as alias, ...`
    From {
        module: PathRef,
        names: Vec<(Ident, Option<Ident>)>,
    },
    /// `import module as alias`, without an alias the last segment is bound
    Module { path: PathRef, alias: Option<Ident> },
}

/// a decorator, `@name(args)`, with its arguments as constant expressions
#[derive(Clone, Debug)]
pub struct Decorator {
    pub name: PathRef,
    pub args: Vec<DecoratorArg>,
    pub span: Span,
}

/// an argument of a decorator or of a map declaration: `value` or
/// `name=value`
#[derive(Clone, Debug)]
pub struct DecoratorArg {
    pub name: Option<Ident>,
    pub value: ConstExpr,
}

/// whose method a function is
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Owner {
    Struct(StructId),
    Enum(EnumId),
}

/// a function or a method
#[derive(Clone, Debug)]
pub struct FnData {
    pub name: Ident,
    pub module: ModuleId,
    /// the struct or enum of a method, `None` for a function
    pub owner: Option<Owner>,
    pub decorators: Vec<Decorator>,
    /// whether it is a `comptime def`, run only at compile time
    pub is_comptime: bool,
    pub generic_params: Vec<GenericParam>,
    /// what it instantiates, for an instance or a method of an instance
    pub instance: Option<Instance>,
    pub params: Vec<Param>,
    pub ret: Option<TypeRef>,
    pub body: Body,
    pub spans: BodySpans,
    /// `@requires(e)` and `@ensures(e)`. The expressions are in `body`, so
    /// they are typed with it
    pub contracts: Vec<Contract>,
}

/// what a contract of a function states
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContractKind {
    /// `@requires(e)`: what callers guarantee
    Requires,
    /// `@ensures(e)`: what the function guarantees when it returns, with
    /// `result` as the returned value
    Ensures,
}

/// a contract of a function: an expression of its body, typed with its
/// parameters (and `result` for `@ensures`), never run
#[derive(Clone, Debug)]
pub struct Contract {
    pub kind: ContractKind,
    pub expr: ExprId,
    pub span: Span,
}

/// a generic parameter: a type (`T`, `M: Model`) or a constant (`N: usize`)
#[derive(Clone, Debug)]
pub struct GenericParam {
    pub name: Ident,
    /// a protocol for a type parameter, an integer type for a constant one
    pub bound: Option<TypeRef>,
}

/// the value of a generic parameter in an instance
///
/// A type is kept as a [`crate::ty::TyKey`], not a `TyId`, because the
/// package outlives the type tables of each round of analysis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InstanceArg {
    /// a type argument
    Ty(crate::ty::TyKey),
    /// a constant argument
    Const(i128),
}

/// what an instance instantiates: the generic item and the values of its
/// parameters
#[derive(Clone, Debug)]
pub struct Instance {
    /// the template: for a method of an instance, the template's method
    pub generic: Item,
    /// the arguments, in the order of the parameters
    pub args: Vec<InstanceArg>,
    /// the arguments by parameter name (for a method, those of its struct or
    /// enum)
    pub env: Vec<(String, InstanceArg)>,
}

/// a protocol: the methods its implementations must define
#[derive(Clone, Debug)]
pub struct ProtocolData {
    pub name: Ident,
    pub module: ModuleId,
    /// `protocol Classifier[In]`, bound to the arguments of each type that
    /// lists it, as in `Classifier[Vector[T, 4]]`
    pub generic_params: Vec<GenericParam>,
    pub methods: Vec<ProtocolMethod>,
}

/// a method a protocol requires
#[derive(Clone, Debug)]
pub struct ProtocolMethod {
    pub name: Ident,
    /// the number of parameters besides `self`
    pub arity: usize,
    /// their types, a parameter without one accepts any type
    pub params: Vec<Option<TypeRef>>,
    /// the return type, any if missing
    pub ret: Option<TypeRef>,
}

/// a parameter of a function
#[derive(Clone, Debug)]
pub struct Param {
    pub name: Ident,
    /// the written type, `self` has none
    pub ty: Option<TypeRef>,
    /// `n: u32 = 4`: a constant, given when a call leaves the parameter out
    pub default: Option<ConstExpr>,
}

/// a struct
#[derive(Clone, Debug)]
pub struct StructData {
    pub name: Ident,
    pub module: ModuleId,
    pub decorators: Vec<Decorator>,
    pub generic_params: Vec<GenericParam>,
    /// for an instance of a generic struct: what it instantiates
    pub instance: Option<Instance>,
    /// the protocols listed after the name: `struct S(Model):`
    pub bases: Vec<PathRef>,
    /// the generic arguments written after each protocol
    pub base_args: Vec<Vec<TypeArgRef>>,
    pub fields: Vec<FieldData>,
    pub methods: Vec<FnId>,
}

/// a field of a struct
#[derive(Clone, Debug)]
pub struct FieldData {
    pub name: Ident,
    pub ty: TypeRef,
}

/// an enum, whose variants may carry values
#[derive(Clone, Debug)]
pub struct EnumData {
    pub name: Ident,
    pub module: ModuleId,
    pub generic_params: Vec<GenericParam>,
    /// for an instance of a generic enum: what it instantiates
    pub instance: Option<Instance>,
    /// the protocols the enum implements
    pub bases: Vec<PathRef>,
    /// the generic arguments written after each protocol
    pub base_args: Vec<Vec<TypeArgRef>>,
    pub variants: Vec<VariantData>,
    pub methods: Vec<FnId>,
}

/// a variant of an enum
#[derive(Clone, Debug)]
pub struct VariantData {
    pub name: Ident,
    /// the types of its values: `None` without parentheses, `Some` (maybe
    /// empty) otherwise
    pub fields: Option<Vec<TypeRef>>,
}

/// a constant, `const NAME: T = value`
#[derive(Clone, Debug)]
pub struct ConstData {
    pub name: Ident,
    pub module: ModuleId,
    pub ty: Option<TypeRef>,
    pub value: ConstExpr,
}

/// a `map` or `event` declaration: `map name: Kind[args](params)`
#[derive(Clone, Debug)]
pub struct MapData {
    pub name: Ident,
    pub module: ModuleId,
    pub is_event: bool,
    /// the kind as written (`Hash`, `Array`, `RingBuf`, ...)
    pub kind: PathRef,
    /// the arguments in brackets, as `Hash[K, V]`, `Array[V, 16]` or
    /// `RingBuf[Event]`
    pub args: Vec<TypeArgRef>,
    /// the arguments in parentheses, as `max_entries=N`
    pub params: Vec<DecoratorArg>,
    pub span: Span,
}

/// a type alias, `type Name = T`
#[derive(Clone, Debug)]
pub struct AliasData {
    pub name: Ident,
    pub module: ModuleId,
    /// `type Vec4[T] = Vector[T, 4]`: expanded at each use
    pub generic_params: Vec<GenericParam>,
    pub ty: TypeRef,
}

/// a type as written
#[derive(Clone, Debug)]
pub enum TypeRef {
    /// a named type with its bracketed arguments: generic arguments, or an
    /// array length after a non-generic type
    Path {
        path: PathRef,
        args: Vec<TypeArgRef>,
        span: Span,
    },
    /// `T?`
    Option(Box<TypeRef>, Span),
    /// `T[N]` after generic arguments: `Vec[u8, 4][2]`
    Array(Box<TypeRef>, ConstExpr, Span),
    /// the parser recovered from an error here
    Missing,
}

impl TypeRef {
    /// where the type is written, `None` for a missing one
    pub fn span(&self) -> Option<Span> {
        match self {
            TypeRef::Path { span, .. } | TypeRef::Option(_, span) | TypeRef::Array(_, _, span) => {
                Some(*span)
            }
            TypeRef::Missing => None,
        }
    }
}

/// a bracketed argument of a type, as written. The parser cannot tell a
/// constant name from a type, so `Vec[u8, N]` has two `Type` arguments until
/// names are resolved.
#[derive(Clone, Debug)]
pub enum TypeArgRef {
    Type(TypeRef),
    Const(ConstExpr),
}

/// replacing the names of `params` in `ty` by their arguments, to expand a
/// generic alias
pub fn subst_type(ty: &TypeRef, params: &[(String, TypeArgRef)]) -> TypeRef {
    let lookup = |path: &PathRef| match path.segments.as_slice() {
        [segment] => params
            .iter()
            .find(|(n, _)| *n == segment.text)
            .map(|(_, a)| a),
        _ => None,
    };
    match ty {
        TypeRef::Path { path, args, span } => {
            if args.is_empty()
                && let Some(TypeArgRef::Type(t)) = lookup(path)
            {
                return t.clone();
            }
            TypeRef::Path {
                path: path.clone(),
                args: args.iter().map(|a| subst_arg(a, params)).collect(),
                span: *span,
            }
        }
        TypeRef::Option(inner, span) => TypeRef::Option(Box::new(subst_type(inner, params)), *span),
        TypeRef::Array(inner, len, span) => TypeRef::Array(
            Box::new(subst_type(inner, params)),
            subst_const(len, params),
            *span,
        ),
        TypeRef::Missing => TypeRef::Missing,
    }
}

/// a type argument with the names of `params` replaced by their arguments
fn subst_arg(arg: &TypeArgRef, params: &[(String, TypeArgRef)]) -> TypeArgRef {
    match arg {
        // a bare name is replaced by its argument, a type or a constant
        TypeArgRef::Type(TypeRef::Path { path, args, .. }) if args.is_empty() => {
            match path.segments.as_slice() {
                [segment] => match params.iter().find(|(n, _)| *n == segment.text) {
                    Some((_, a)) => a.clone(),
                    None => arg.clone(),
                },
                _ => arg.clone(),
            }
        }
        TypeArgRef::Type(t) => TypeArgRef::Type(subst_type(t, params)),
        TypeArgRef::Const(c) => TypeArgRef::Const(subst_const(c, params)),
    }
}

/// replacing the names of `params` in a constant expression by their
/// arguments. A name bound to a bare named type stays a path, because it may
/// be a constant the parser did not recognize.
fn subst_const(c: &ConstExpr, params: &[(String, TypeArgRef)]) -> ConstExpr {
    match c {
        ConstExpr::Path(path) => match path.segments.as_slice() {
            [segment] => match params.iter().find(|(n, _)| *n == segment.text) {
                Some((_, TypeArgRef::Const(v))) => v.clone(),
                Some((_, TypeArgRef::Type(TypeRef::Path { path, args, .. })))
                    if args.is_empty() =>
                {
                    ConstExpr::Path(path.clone())
                }
                _ => c.clone(),
            },
            _ => c.clone(),
        },
        ConstExpr::Unary(op, e, span) => {
            ConstExpr::Unary(*op, Box::new(subst_const(e, params)), *span)
        }
        ConstExpr::Binary(op, a, b, span) => ConstExpr::Binary(
            *op,
            Box::new(subst_const(a, params)),
            Box::new(subst_const(b, params)),
            *span,
        ),
        ConstExpr::Call(f, args, span) => ConstExpr::Call(
            f.clone(),
            args.iter().map(|a| subst_const(a, params)).collect(),
            *span,
        ),
        ConstExpr::Array(items, span) => ConstExpr::Array(
            items.iter().map(|a| subst_const(a, params)).collect(),
            *span,
        ),
        _ => c.clone(),
    }
}

/// the expressions allowed where a value must be known at compile time
/// (constant initializers, array lengths, decorator arguments)
#[derive(Clone, Debug)]
pub enum ConstExpr {
    Int(u128, Span),
    Bool(bool, Span),
    /// a string, with its escapes applied
    Str(String, Span),
    /// a constant, or a constant generic parameter
    Path(PathRef),
    Unary(UnaryOp, Box<ConstExpr>, Span),
    Binary(BinaryOp, Box<ConstExpr>, Box<ConstExpr>, Span),
    /// a call of a `comptime` function, with positional arguments only
    Call(PathRef, Vec<ConstExpr>, Span),
    /// `[1, 2, 3]`, the value of an array constant
    Array(Vec<ConstExpr>, Span),
    /// anything else (a float, a call with named arguments, ...), not a
    /// constant expression
    Invalid(Span),
    /// nothing written, or an integer literal too large for 128 bits, already
    /// reported
    Missing,
}

impl ConstExpr {
    /// where the expression is written, `None` for a missing one
    pub fn span(&self) -> Option<Span> {
        match self {
            ConstExpr::Int(_, s)
            | ConstExpr::Bool(_, s)
            | ConstExpr::Str(_, s)
            | ConstExpr::Unary(_, _, s)
            | ConstExpr::Binary(_, _, _, s)
            | ConstExpr::Call(_, _, s)
            | ConstExpr::Array(_, s)
            | ConstExpr::Invalid(s) => Some(*s),
            ConstExpr::Path(p) => Some(p.span),
            ConstExpr::Missing => None,
        }
    }
}

// function bodies
// -------------------------

/// the body of a function: arenas of its nodes, and its top-level block
#[derive(Clone, Debug, Default)]
pub struct Body {
    pub exprs: Arena<Expr>,
    pub stmts: Arena<Stmt>,
    pub pats: Arena<Pat>,
    /// the statements of the body, in order
    pub root: Block,
}

/// the source spans of the nodes of a [`Body`], by id
#[derive(Clone, Debug, Default)]
pub struct BodySpans {
    pub exprs: ArenaMap<ExprId, Span>,
    pub stmts: ArenaMap<StmtId, Span>,
    pub pats: ArenaMap<PatId, Span>,
}

/// a sequence of statements
pub type Block = Vec<StmtId>;

/// a prefix operator
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// `not`
    Not,
    /// `-`
    Neg,
    /// `~`
    BitNot,
}

/// an infix operator
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    BitOr,
    BitXor,
    BitAnd,
    Shl,
    Shr,
    Add,
    Sub,
    Mul,
    /// `/`, reserved for fixed-point division
    Div,
    /// `//`, integer division
    FloorDiv,
    Rem,
}

impl BinaryOp {
    /// if the operator is a comparison, giving a `bool`. `and` and `or` do
    /// not count, their operands are `bool` already
    pub fn is_comparison(self) -> bool {
        use BinaryOp::*;
        matches!(self, Eq | Ne | Lt | Le | Gt | Ge)
    }

    /// how the operator is written
    pub fn symbol(self) -> &'static str {
        match self {
            BinaryOp::Or => "or",
            BinaryOp::And => "and",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::BitAnd => "&",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::FloorDiv => "//",
            BinaryOp::Rem => "%",
        }
    }
}

/// an expression of a body. Parentheses leave no trace
#[derive(Clone, Debug)]
pub enum Expr {
    /// the parser recovered from an error here, or lowering reported one
    Missing,
    Int(u128),
    Float(f64),
    Bool(bool),
    /// a string, with its escapes applied
    Str(String),
    /// a format string, `f"a {x} b"`
    FStr(Vec<FStrPart>),
    None,
    /// `...`: the body of a helper, and an error anywhere else
    Ellipsis,
    /// a name, unresolved
    Name(String),
    Field {
        base: ExprId,
        name: Ident,
    },
    Call {
        callee: ExprId,
        args: Vec<Arg>,
    },
    Index {
        base: ExprId,
        indices: Vec<ExprId>,
    },
    Unary {
        op: UnaryOp,
        operand: ExprId,
    },
    Binary {
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    /// `e?`: the value of an option, or leave the function
    Try(ExprId),
    /// `[a, b, c]`
    Array(Vec<ExprId>),
}

/// a piece of a format string
#[derive(Clone, Debug)]
pub enum FStrPart {
    /// literal text, with `{{`, `}}` and escapes already applied
    Text(String),
    /// an interpolated expression
    Expr(ExprId),
}

/// an argument of a call: positional, or `name=value`
#[derive(Clone, Debug)]
pub struct Arg {
    pub name: Option<Ident>,
    pub value: ExprId,
}

/// a statement of a body
#[derive(Clone, Debug)]
pub enum Stmt {
    Expr(ExprId),
    /// `target = value` or `target: ty = value`, a declaration or an
    /// assignment (the type checker decides)
    Assign {
        target: ExprId,
        ty: Option<TypeRef>,
        value: Option<ExprId>,
    },
    /// `target op= value`
    AugAssign {
        target: ExprId,
        op: BinaryOp,
        value: ExprId,
    },
    Return(Option<ExprId>),
    If {
        cond: ExprId,
        then: Block,
        /// the `else` block. An `elif` chain becomes nested `If`s, each alone
        /// in the `else` block of the one before.
        otherwise: Option<Block>,
    },
    For {
        decorators: Vec<Decorator>,
        /// `@invariant(e)`, what holds at the start of every iteration. The
        /// decorator is also in `decorators`.
        invariants: Vec<ExprId>,
        binding: Ident,
        iter: ExprId,
        body: Block,
    },
    While {
        decorators: Vec<Decorator>,
        /// as for `For`
        invariants: Vec<ExprId>,
        cond: ExprId,
        body: Block,
    },
    Match {
        scrutinee: ExprId,
        arms: Vec<MatchArm>,
    },
    Break,
    Continue,
    Pass,
    /// `comptime:`, a block run at compile time
    Comptime(Block),
    /// the parser recovered from an error here
    Missing,
}

/// an arm of a `match`: `case pat if guard:` and its block
#[derive(Clone, Debug)]
pub struct MatchArm {
    pub pat: PatId,
    pub guard: Option<ExprId>,
    pub body: Block,
}

/// a pattern of a `match` arm
#[derive(Clone, Debug)]
pub enum Pat {
    /// the parser recovered from an error here
    Missing,
    /// `_`
    Wild,
    /// a name, bound to the value
    Bind(Ident),
    /// an integer literal, possibly negated
    Int {
        value: u128,
        negative: bool,
    },
    Bool(bool),
    /// a string literal. Strings cannot be matched, so we keep it only to
    /// report it.
    Str,
    /// a float literal, kept only to be reported, as a string is
    Float,
    None,
    /// a variant without values
    Path(PathRef),
    /// a variant with its values: `Some(x)`, `Transport.Udp(u)`
    TupleStruct {
        path: PathRef,
        fields: Vec<PatId>,
    },
    /// `a | b`
    Or(Vec<PatId>),
}
