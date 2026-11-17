//! this module implements the type checker of Core, case for case the one
//! of the Lean formalization (`typeOf`, `checkStmt`, `check`). Only programs
//! that pass [`check`] are covered by the type safety theorem `run_safe`.

use crate::{Expr, IntTy, Program, Stmt, Ty, UnOp};

/// the type of `e` given the types of the locals, `None` if ill typed
///
/// an array is not a value: a [`Expr::Var`] naming one is ill typed, only
/// [`Expr::Index`] reads it. An array index can have any integer type, but
/// a packet offset must be a `u32`.
pub fn type_of(locals: &[Ty], e: &Expr) -> Option<Ty> {
    let int = |t: IntTy| Some(Ty::Int { t });
    match e {
        Expr::Lit { t, .. } => int(*t),
        Expr::Bool { .. } => Some(Ty::Bool),
        Expr::Var { x } => match locals.get(*x)? {
            Ty::Array { .. } => None,
            t => Some(t.clone()),
        },
        Expr::Index { array, i } => match locals.get(*array)? {
            Ty::Array { elem, .. } if is_int(&type_of(locals, i)) => int(*elem),
            _ => None,
        },
        Expr::Packet { offset } => (type_of(locals, offset)? == Ty::Int { t: IntTy::U32 })
            .then_some(Ty::Int { t: IntTy::U8 }),
        Expr::Len => int(IntTy::U32),
        Expr::Un {
            op: UnOp::Not, e, ..
        } => (type_of(locals, e)? == Ty::Bool).then_some(Ty::Bool),
        Expr::Un { t, e, .. } => {
            (type_of(locals, e)? == Ty::Int { t: *t }).then_some(Ty::Int { t: *t })
        }
        Expr::Bin { op, t, a, b } => {
            let (a, b) = (type_of(locals, a)?, type_of(locals, b)?);
            if op.is_logical() {
                (a == Ty::Bool && b == Ty::Bool).then_some(Ty::Bool)
            } else if a == (Ty::Int { t: *t }) && b == a {
                Some(if op.is_comparison() { Ty::Bool } else { a })
            } else {
                None
            }
        }
        Expr::Cast { t, e } => match type_of(locals, e)? {
            Ty::Int { .. } | Ty::Bool => int(*t),
            Ty::Array { .. } => None,
        },
    }
}

/// is this an integer type?
fn is_int(t: &Option<Ty>) -> bool {
    matches!(t, Some(Ty::Int { .. }))
}

/// checking a statement, given the types of the locals
///
/// only a scalar can be assigned, and only a value of its own type. There
/// are no implicit conversions, a [`Expr::Cast`] is always explicit.
pub fn check_stmt(locals: &[Ty], s: &Stmt) -> bool {
    match s {
        Stmt::Skip | Stmt::Ret { .. } => true,
        Stmt::Assign { x, e } => match locals.get(*x) {
            Some(t @ (Ty::Int { .. } | Ty::Bool)) => type_of(locals, e).as_ref() == Some(t),
            _ => false,
        },
        Stmt::Store { array, i, e } => match locals.get(*array) {
            Some(Ty::Array { elem, .. }) => {
                is_int(&type_of(locals, i)) && type_of(locals, e) == Some(Ty::Int { t: *elem })
            }
            _ => false,
        },
        Stmt::PacketStore { offset, e } => {
            type_of(locals, offset) == Some(Ty::Int { t: IntTy::U32 })
                && type_of(locals, e) == Some(Ty::Int { t: IntTy::U8 })
        }
        Stmt::Seq { a, b } => check_stmt(locals, a) && check_stmt(locals, b),
        Stmt::Ite { cond, thn, els } => {
            type_of(locals, cond) == Some(Ty::Bool)
                && check_stmt(locals, thn)
                && check_stmt(locals, els)
        }
        Stmt::Loop { x, t, body, .. } => {
            locals.get(*x) == Some(&Ty::Int { t: *t }) && check_stmt(locals, body)
        }
    }
}

/// checking a whole program
pub fn check(p: &Program) -> bool {
    check_stmt(&p.locals, &p.body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinOp, Bits};

    const U16: IntTy = IntTy {
        signed: false,
        bits: Bits::B16,
    };
    const I64: IntTy = IntTy {
        signed: true,
        bits: Bits::B64,
    };

    fn shl(t: IntTy, amount: IntTy) -> Program {
        Program {
            locals: vec![Ty::Int { t: U16 }],
            body: Stmt::Assign {
                x: 0,
                e: Expr::Bin {
                    op: BinOp::Shl,
                    t,
                    a: Box::new(Expr::Var { x: 0 }),
                    b: Box::new(Expr::Lit {
                        t: amount,
                        value: 8,
                    }),
                },
            },
        }
    }

    // in Vigil the shift amount can have another type than the value, in
    // Core it can't, so the exporter converts it
    #[test]
    fn shift_operands_have_the_type_of_the_shift() {
        assert!(check(&shl(U16, U16)));
        assert!(!check(&shl(U16, I64)));
    }
}
