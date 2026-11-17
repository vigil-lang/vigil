//! this crate defines Vigil Core, the subset of Vigil that the Lean
//! formalization of the full Vigil works on: XDP programs over scalars,
//! local arrays, bounded loops and the packet bytes. The types serialize to
//! the same JSON that Lean derives (the *Vigil Core JSON*).
//!
//! - [`export`]: a type-checked Vigil program in the subset, as Core
//! - [`print`](mod@print): Core as Vigil source
//! - [`generate`]: random well-typed Core programs, for differential tests
//! - [`types`]: the type checker, the `check` of the formalization
//!
//! As in the formalization, a local is an index into [`Program::locals`],
//! not a name. [`print`](mod@print) makes up the names `v0`, `v1`, ...

pub mod export;
pub mod generate;
pub mod print;
pub mod types;

use serde::{Deserialize, Serialize};

/// the width of an integer type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Bits {
    B8,
    B16,
    B32,
    B64,
}

impl Bits {
    /// the width in bits
    pub fn count(self) -> u32 {
        match self {
            Bits::B8 => 8,
            Bits::B16 => 16,
            Bits::B32 => 32,
            Bits::B64 => 64,
        }
    }
}

/// an integer type: signedness and width
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntTy {
    pub signed: bool,
    pub bits: Bits,
}

impl IntTy {
    pub const U8: IntTy = IntTy {
        signed: false,
        bits: Bits::B8,
    };
    pub const U32: IntTy = IntTy {
        signed: false,
        bits: Bits::B32,
    };

    /// the name of the type in Vigil: `u8`, `i64`, ...
    pub fn name(self) -> String {
        format!(
            "{}{}",
            if self.signed { "i" } else { "u" },
            self.bits.count()
        )
    }

    /// the least value of the type. An `i128` holds the range of every
    /// integer type
    pub fn min(self) -> i128 {
        if self.signed {
            -(1i128 << (self.bits.count() - 1))
        } else {
            0
        }
    }

    /// the greatest value of the type
    pub fn max(self) -> i128 {
        if self.signed {
            (1i128 << (self.bits.count() - 1)) - 1
        } else {
            (1i128 << self.bits.count()) - 1
        }
    }
}

/// the type of a local: an integer, a boolean or a fixed length array of
/// integers. Only a local can be an array, never an expression.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Ty {
    Int { t: IntTy },
    Bool,
    Array { elem: IntTy, len: u64 },
}

/// unary operators: `-`, `~` and `not`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnOp {
    Neg,
    Bitnot,
    Not,
}

/// binary operators: arithmetic, bitwise, comparisons and logical ones.
/// [`BinOp::Div`] is Vigil's `//`. Vigil's `/` is not in Core.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Band,
    Bor,
    Bxor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl BinOp {
    /// is this a comparison, an operator on integers giving a boolean?
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }

    /// is this `and` or `or`?
    pub fn is_logical(self) -> bool {
        matches!(self, BinOp::And | BinOp::Or)
    }
}

/// expressions of Core
///
/// an operator carries the integer type it works on, so we can read the
/// type of an expression without an environment. For `not`, `and` and `or`
/// that type means nothing, and we write `u8` there.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Expr {
    Lit {
        t: IntTy,
        value: i128,
    },
    Bool {
        value: bool,
    },
    Var {
        x: usize,
    },
    Index {
        array: usize,
        i: Box<Expr>,
    },
    /// the packet byte at an offset, a `u32`
    Packet {
        offset: Box<Expr>,
    },
    /// the packet length, a `u32`
    Len,
    Un {
        op: UnOp,
        t: IntTy,
        e: Box<Expr>,
    },
    Bin {
        op: BinOp,
        t: IntTy,
        a: Box<Expr>,
        b: Box<Expr>,
    },
    Cast {
        t: IntTy,
        e: Box<Expr>,
    },
}

/// the verdicts of an XDP program, in the order of the `XDP_*` values
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Aborted,
    Drop,
    Pass,
    Tx,
    Redirect,
}

impl Action {
    /// every verdict, in the order of its XDP value
    pub const ALL: [Action; 5] = [
        Action::Aborted,
        Action::Drop,
        Action::Pass,
        Action::Tx,
        Action::Redirect,
    ];

    /// the XDP value, that is its position in [`Action::ALL`]
    pub fn value(self) -> u32 {
        Action::ALL.iter().position(|&a| a == self).unwrap() as u32
    }

    /// the name of the `XdpAction` variant with this value
    pub fn variant(self) -> &'static str {
        ["ABORTED", "DROP", "PASS", "TX", "REDIRECT"][self.value() as usize]
    }
}

/// statements of Core
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Stmt {
    Skip,
    Assign {
        x: usize,
        e: Expr,
    },
    Store {
        array: usize,
        i: Expr,
        e: Expr,
    },
    PacketStore {
        offset: Expr,
        e: Expr,
    },
    Seq {
        a: Box<Stmt>,
        b: Box<Stmt>,
    },
    Ite {
        cond: Expr,
        thn: Box<Stmt>,
        els: Box<Stmt>,
    },
    /// `for x in range(count)`, with `x` of type `t`. The count is a
    /// constant, so every loop is bounded
    Loop {
        x: usize,
        t: IntTy,
        count: u64,
        body: Box<Stmt>,
    },
    Ret {
        action: Action,
    },
}

impl Stmt {
    /// building a right nested sequence, [`Stmt::Skip`] when empty
    pub fn seq(stmts: Vec<Stmt>) -> Stmt {
        stmts
            .into_iter()
            .rev()
            .reduce(|b, a| Stmt::Seq {
                a: Box::new(a),
                b: Box::new(b),
            })
            .unwrap_or(Stmt::Skip)
    }
}

/// a Core program: the types of its locals and its body
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Program {
    pub locals: Vec<Ty>,
    pub body: Stmt,
}

/// flattening sequences and dropping `skip`s
///
/// two programs that differ only in the grouping of statements get the same
/// normal form, so we compare this after a round trip (to source and back)
/// that does not keep the grouping.
pub fn normalize(p: &Program) -> Program {
    fn stmt(s: &Stmt) -> Stmt {
        let mut out = Vec::new();
        collect(s, &mut out);
        Stmt::seq(out)
    }
    fn collect(s: &Stmt, out: &mut Vec<Stmt>) {
        match s {
            Stmt::Seq { a, b } => {
                collect(a, out);
                collect(b, out);
            }
            Stmt::Skip => {}
            Stmt::Ite { cond, thn, els } => out.push(Stmt::Ite {
                cond: cond.clone(),
                thn: Box::new(stmt(thn)),
                els: Box::new(stmt(els)),
            }),
            Stmt::Loop { x, t, count, body } => out.push(Stmt::Loop {
                x: *x,
                t: *t,
                count: *count,
                body: Box::new(stmt(body)),
            }),
            other => out.push(other.clone()),
        }
    }
    Program {
        locals: p.locals.clone(),
        body: stmt(&p.body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // this is the JSON that Lean derives for the same program, so a `serde`
    // change that breaks the format fails here
    #[test]
    fn serializes_like_lean() {
        let u8 = IntTy::U8;
        let p = Program {
            locals: vec![Ty::Int { t: u8 }, Ty::Bool],
            body: Stmt::seq(vec![
                Stmt::Assign {
                    x: 0,
                    e: Expr::Lit { t: u8, value: 3 },
                },
                Stmt::Ret {
                    action: Action::Drop,
                },
            ]),
        };
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(
            json,
            r#"{"locals":[{"int":{"t":{"signed":false,"bits":"b8"}}},"bool"],"body":{"seq":{"a":{"assign":{"x":0,"e":{"lit":{"t":{"signed":false,"bits":"b8"},"value":3}}}},"b":{"ret":{"action":"drop"}}}}}"#
        );
        assert_eq!(serde_json::from_str::<Program>(&json).unwrap(), p);
    }
}
