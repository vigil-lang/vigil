//! this module computes ranges of the integer values of a function, so the
//! code generator can leave out bounds checks: an index whose range is
//! inside the array needs no check (`FnGen::element_ptr`).
//!
//! It is an abstract interpretation over intervals, sound and simple rather
//! than precise. It is flow-insensitive: the range of a local is the union
//! of everything assigned to it. Only locals declared by an assignment or a
//! `for` are followed; parameters, pattern bindings, comptime variables,
//! calls, fields and map values get the range of their type. Operations
//! wrap as the instructions do (`arith` in `function.rs`), and a result
//! that wraps gets the whole type.
//!
//! We iterate down from the range of the type, not up from empty. Every
//! step is then sound, so we can stop after [`ROUNDS`] rounds without a
//! fixed point. Going up would need a widening, and stopping early there
//! would be unsound.

use std::collections::{HashMap, HashSet};

use vigil_sema::db::ConstValue;
use vigil_sema::hir::*;
use vigil_sema::infer::{InferenceResult, LocalId, Res};
use vigil_sema::resolve::Builtin;
use vigil_sema::ty::{IntTy, TyKind};
use vigil_sema::{Analysis, walk};

/// the values `lo..=hi`, never empty. The bounds are `i128` so that every
/// value of every integer type fits, and also sums and differences of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Range {
    pub lo: i128,
    pub hi: i128,
}

impl Range {
    /// the range of `v` alone
    pub fn exact(v: i128) -> Range {
        Range { lo: v, hi: v }
    }

    /// every value of `int`
    pub fn of(int: IntTy) -> Range {
        Range {
            lo: int.min(),
            hi: int.max(),
        }
    }

    /// the smallest range holding both
    fn join(self, other: Range) -> Range {
        Range {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// the range itself if it is inside `int`, otherwise all of `int`,
    /// because a value outside the type wraps
    fn within(self, int: IntTy) -> Range {
        if self.lo >= int.min() && self.hi <= int.max() {
            self
        } else {
            Range::of(int)
        }
    }

    /// checking if every value is at least 0
    fn nonnegative(self) -> bool {
        self.lo >= 0
    }
}

/// `v` rounded up to a power of two minus one, the largest value with no
/// bit above the highest bit of `v`. For non-negative `v`; 0 otherwise.
fn mask_of(v: i128) -> i128 {
    if v <= 0 {
        0
    } else {
        (1i128 << (128 - v.leading_zeros())) - 1
    }
}

/// the most rounds of the descending iteration
const ROUNDS: usize = 16;

/// the ranges of the locals of a function. A local that is not followed has
/// none, and reads as the range of its type.
#[derive(Clone, Debug, Default)]
pub(crate) struct Ranges {
    locals: HashMap<LocalId, Range>,
}

/// the ranges of expressions of a function, given the ranges of its locals
pub(crate) struct Eval<'a> {
    pub an: &'a Analysis,
    pub body: &'a Body,
    pub res: &'a InferenceResult,
    pub ranges: &'a Ranges,
}

impl Ranges {
    /// the ranges of the locals of a function body (see the module header)
    pub fn of_function(an: &Analysis, f: FnId, body: &Body, res: &InferenceResult) -> Ranges {
        let mut stmts = Vec::new();
        walk::stmts(body, &body.root, &mut stmts);
        // the locals followed: integers declared by an assignment or a
        // `for`, and not assigned in a comptime block, which we do not see
        let mut followed: HashSet<LocalId> = HashSet::new();
        for &stmt in &stmts {
            for local in [res.decl_locals.get(stmt), res.for_locals.get(stmt)]
                .into_iter()
                .flatten()
            {
                if let TyKind::Int(_) = an.types.kind(res.locals[*local].ty) {
                    followed.insert(*local);
                }
            }
        }
        if let Some(blocks) = an.comptime_blocks.get(f) {
            for (_, values) in blocks {
                for (local, _) in values {
                    followed.remove(local);
                }
            }
        }
        // a local without a range yet reads as its type, the top where we
        // start. Each round recomputes every range from the previous round,
        // and we stop early when a round changes nothing.
        let mut ranges = Ranges::default();
        for _ in 0..ROUNDS {
            let mut next: HashMap<LocalId, Range> = HashMap::new();
            {
                let eval = Eval {
                    an,
                    body,
                    res,
                    ranges: &ranges,
                };
                let mut add = |local: LocalId, range: Range| {
                    if !followed.contains(&local) {
                        return;
                    }
                    next.entry(local)
                        .and_modify(|r| *r = r.join(range))
                        .or_insert(range);
                };
                for &stmt in &stmts {
                    match &body.stmts[stmt] {
                        Stmt::Assign { target, value, .. } => {
                            let local = match res.decl_locals.get(stmt) {
                                Some(&local) => Some(local),
                                None => match res.res.get(*target) {
                                    Some(Res::Local(local)) => Some(*local),
                                    _ => None,
                                },
                            };
                            let Some(local) = local else { continue };
                            if !followed.contains(&local) {
                                continue;
                            }
                            let range = match value {
                                Some(value) => eval.expr(*value),
                                // declared without a value: zero
                                None => Some(Range::exact(0)),
                            };
                            add(local, range.unwrap_or(eval.local_type(local)));
                        }
                        Stmt::AugAssign { target, op, value } => {
                            let Some(Res::Local(local)) = res.res.get(*target).copied() else {
                                continue;
                            };
                            if !followed.contains(&local) {
                                continue;
                            }
                            let int = eval.int_of_local(local);
                            let range = int.and_then(|int| {
                                let a = eval.local(local);
                                let b = eval.expr(*value)?;
                                eval.binary(*op, a, b, int)
                            });
                            add(local, range.unwrap_or(eval.local_type(local)));
                        }
                        Stmt::For { iter, .. } => {
                            let Some(&local) = res.for_locals.get(stmt) else {
                                continue;
                            };
                            add(local, eval.loop_variable(*iter, local));
                        }
                        _ => {}
                    }
                }
            }
            if next == ranges.locals {
                break;
            }
            ranges.locals = next;
        }
        ranges
    }
}

impl Eval<'_> {
    /// the integer type of an expression, `None` for other types
    fn int_of(&self, e: ExprId) -> Option<IntTy> {
        let ty = *self.res.expr_types.get(e)?;
        match self.an.types.kind(ty) {
            TyKind::Int(int) => Some(*int),
            _ => None,
        }
    }

    /// the integer type of a local, `None` for other types
    fn int_of_local(&self, local: LocalId) -> Option<IntTy> {
        match self.an.types.kind(self.res.locals[local].ty) {
            TyKind::Int(int) => Some(*int),
            _ => None,
        }
    }

    /// the range of the type of a local; just 0 if it is not an integer
    fn local_type(&self, local: LocalId) -> Range {
        self.int_of_local(local).map_or(Range::exact(0), Range::of)
    }

    /// the range of a local: what the iteration found, or its type
    fn local(&self, local: LocalId) -> Range {
        self.ranges
            .locals
            .get(&local)
            .copied()
            .unwrap_or_else(|| self.local_type(local))
    }

    /// the range of the variable of a `for`. Over `range(a, b)` it is at
    /// least `a` and less than `b`; over `range(b)`, at least 0 and less
    /// than `b`. Over an array, or a range that may be empty, it is the
    /// range of its type (the element type for an array).
    fn loop_variable(&self, iter: ExprId, local: LocalId) -> Range {
        let whole = self.local_type(local);
        if let Expr::Call { callee, args } = &self.body.exprs[iter]
            && self.res.res.get(*callee) == Some(&Res::Builtin(Builtin::Range))
        {
            let start = match args.as_slice() {
                [start, _] => self.expr(start.value),
                _ => Some(Range::exact(0)),
            };
            let end = args.last().and_then(|a| self.expr(a.value));
            if let (Some(start), Some(end)) = (start, end)
                && end.hi > start.lo
            {
                return Range {
                    lo: start.lo,
                    hi: end.hi - 1,
                };
            }
        }
        whole
    }

    /// the range of an integer expression, or `None` if it is not one
    pub fn expr(&self, e: ExprId) -> Option<Range> {
        let int = self.int_of(e)?;
        let whole = Range::of(int);
        let range = match &self.body.exprs[e] {
            Expr::Int(v) => Range::exact(i128::try_from(*v).ok()?).within(int),
            Expr::Name(_) => match self.res.res.get(e).copied() {
                Some(Res::Local(local)) if self.int_of_local(local).is_some() => self.local(local),
                Some(Res::ConstValue(v)) => Range::exact(v).within(int),
                Some(Res::Const(c)) => match self.an.consts[c].1 {
                    Some(ConstValue::Int(v)) => Range::exact(v).within(int),
                    _ => whole,
                },
                _ => whole,
            },
            Expr::Binary { op, lhs, rhs }
                if !matches!(self.res.res.get(e), Some(Res::Operator(_))) =>
            {
                match (self.expr(*lhs), self.expr(*rhs)) {
                    (Some(a), Some(b)) => self.binary(*op, a, b, int).unwrap_or(whole),
                    _ => whole,
                }
            }
            Expr::Call { callee, args } => match self.res.res.get(*callee).copied() {
                // a conversion keeps the value when it fits, and wraps
                // otherwise
                Some(Res::Builtin(Builtin::Int(_))) => args
                    .first()
                    .and_then(|a| self.expr(a.value))
                    .map_or(whole, |r| r.within(int)),
                Some(Res::Builtin(Builtin::Min | Builtin::Max)) => {
                    let min = self.res.res.get(*callee) == Some(&Res::Builtin(Builtin::Min));
                    match (
                        args.first().and_then(|a| self.expr(a.value)),
                        args.get(1).and_then(|a| self.expr(a.value)),
                    ) {
                        (Some(a), Some(b)) if min => Range {
                            lo: a.lo.min(b.lo),
                            hi: a.hi.min(b.hi),
                        },
                        (Some(a), Some(b)) => Range {
                            lo: a.lo.max(b.lo),
                            hi: a.hi.max(b.hi),
                        },
                        _ => whole,
                    }
                }
                Some(Res::Builtin(b @ (Builtin::SaturatingAdd | Builtin::SaturatingSub))) => {
                    match (
                        args.first().and_then(|a| self.expr(a.value)),
                        args.get(1).and_then(|a| self.expr(a.value)),
                    ) {
                        (Some(a), Some(c)) => {
                            let (lo, hi) = if b == Builtin::SaturatingAdd {
                                (a.lo + c.lo, a.hi + c.hi)
                            } else {
                                (a.lo - c.hi, a.hi - c.lo)
                            };
                            Range {
                                lo: lo.clamp(whole.lo, whole.hi),
                                hi: hi.clamp(whole.lo, whole.hi),
                            }
                        }
                        _ => whole,
                    }
                }
                // `clamp(x, lo, hi)` is `min(max(x, lo), hi)`
                Some(Res::Builtin(Builtin::Clamp)) => {
                    let get = |i: usize| args.get(i).and_then(|a| self.expr(a.value));
                    match (get(0), get(1), get(2)) {
                        (Some(x), Some(lo), Some(hi)) => {
                            let at_least = Range {
                                lo: x.lo.max(lo.lo),
                                hi: x.hi.max(lo.hi),
                            };
                            Range {
                                lo: at_least.lo.min(hi.lo),
                                hi: at_least.hi.min(hi.hi),
                            }
                        }
                        _ => whole,
                    }
                }
                _ => whole,
            },
            _ => whole,
        };
        Some(range)
    }

    /// the range of `a op b` in `int`, as the instructions compute it;
    /// `None` for an operation we do not follow
    pub fn binary(&self, op: BinaryOp, a: Range, b: Range, int: IntTy) -> Option<Range> {
        let whole = Range::of(int);
        let range = match op {
            BinaryOp::Add => Range {
                lo: a.lo + b.lo,
                hi: a.hi + b.hi,
            }
            .within(int),
            BinaryOp::Sub => Range {
                lo: a.lo - b.hi,
                hi: a.hi - b.lo,
            }
            .within(int),
            BinaryOp::Mul => {
                let products = [
                    a.lo.checked_mul(b.lo),
                    a.lo.checked_mul(b.hi),
                    a.hi.checked_mul(b.lo),
                    a.hi.checked_mul(b.hi),
                ];
                match products {
                    [Some(p), Some(q), Some(r), Some(t)] => Range {
                        lo: p.min(q).min(r).min(t),
                        hi: p.max(q).max(r).max(t),
                    }
                    .within(int),
                    _ => whole,
                }
            }
            // with a non-negative operand, the result is between 0 and that
            // operand
            BinaryOp::BitAnd => match (a.nonnegative(), b.nonnegative()) {
                (true, true) => Range {
                    lo: 0,
                    hi: a.hi.min(b.hi),
                },
                (true, false) => Range { lo: 0, hi: a.hi },
                (false, true) => Range { lo: 0, hi: b.hi },
                (false, false) => whole,
            },
            BinaryOp::BitOr | BinaryOp::BitXor if a.nonnegative() && b.nonnegative() => {
                let lo = if op == BinaryOp::BitOr {
                    a.lo.max(b.lo)
                } else {
                    0
                };
                Range {
                    lo,
                    hi: mask_of(a.hi.max(b.hi)),
                }
                .within(int)
            }
            // the shift amount is taken modulo the width. With an unknown
            // amount, the result is between 0 and `a`.
            BinaryOp::Shr if a.nonnegative() => {
                let width = i128::from(int.bits());
                if b.lo == b.hi {
                    let k = (b.lo & (width - 1)) as u32;
                    Range {
                        lo: a.lo >> k,
                        hi: a.hi >> k,
                    }
                } else {
                    Range { lo: 0, hi: a.hi }
                }
            }
            BinaryOp::Shl if a.nonnegative() && b.lo == b.hi => {
                let k = (b.lo & i128::from(int.bits() - 1)) as u32;
                Range {
                    lo: a.lo << k,
                    hi: a.hi << k,
                }
                .within(int)
            }
            // only unsigned division and remainder. eBPF gives `x // 0 == 0`
            // and `x % 0 == x`, so a divisor that may be 0 bounds the result
            // by `x`.
            BinaryOp::FloorDiv if !int.signed() => {
                if b.lo >= 1 {
                    Range {
                        lo: a.lo / b.hi,
                        hi: a.hi / b.lo,
                    }
                } else {
                    Range { lo: 0, hi: a.hi }
                }
            }
            BinaryOp::Rem if !int.signed() => {
                if b.lo >= 1 {
                    Range {
                        lo: 0,
                        hi: a.hi.min(b.hi - 1),
                    }
                } else {
                    Range { lo: 0, hi: a.hi }
                }
            }
            _ => return None,
        };
        Some(range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_round_up_to_powers_of_two() {
        assert_eq!(mask_of(0), 0);
        assert_eq!(mask_of(1), 1);
        assert_eq!(mask_of(5), 7);
        assert_eq!(mask_of(2047), 2047);
        assert_eq!(mask_of(2048), 4095);
    }
}
