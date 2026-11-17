//! this module implements flat walks over the HIR of a function body, for
//! the passes that need every statement, local or subexpression of a block
//! and not its tree

use std::collections::HashSet;

use crate::hir::*;
use crate::infer::{InferenceResult, LocalId, Res};

/// the statements of a block and its nested blocks, pushed on `out` in
/// source order, each statement before the ones nested in it
pub fn stmts(body: &Body, block: &Block, out: &mut Vec<StmtId>) {
    for &stmt in block {
        out.push(stmt);
        match &body.stmts[stmt] {
            Stmt::If {
                then, otherwise, ..
            } => {
                stmts(body, then, out);
                if let Some(otherwise) = otherwise {
                    stmts(body, otherwise, out);
                }
            }
            Stmt::For { body: inner, .. } | Stmt::While { body: inner, .. } => {
                stmts(body, inner, out)
            }
            Stmt::Match { arms, .. } => {
                for arm in arms {
                    stmts(body, &arm.body, out);
                }
            }
            _ => {}
        }
    }
}

/// the locals a block reads, writes or declares, nested blocks included.
/// The split into tail-called segments and the frame of a `bpf_loop` body
/// are computed from this
pub fn locals(body: &Body, res: &InferenceResult, block: &Block) -> HashSet<LocalId> {
    let mut walker = Locals {
        body,
        res,
        found: HashSet::new(),
    };
    let mut all = Vec::new();
    stmts(body, block, &mut all);
    for stmt in all {
        walker.stmt(stmt);
    }
    walker.found
}

/// the state of [`locals`]: the body, the inference result (which local a
/// name refers to) and the locals found so far
struct Locals<'a> {
    body: &'a Body,
    res: &'a InferenceResult,
    found: HashSet<LocalId>,
}

impl Locals<'_> {
    /// collecting the locals of a statement. Nested blocks are skipped
    /// because [`stmts`] already lists their statements
    fn stmt(&mut self, stmt: StmtId) {
        if let Some(&local) = self.res.decl_locals.get(stmt) {
            self.found.insert(local);
        }
        if let Some(&local) = self.res.for_locals.get(stmt) {
            self.found.insert(local);
        }
        match &self.body.stmts[stmt] {
            Stmt::Expr(e) | Stmt::Return(Some(e)) => self.expr(*e),
            Stmt::Assign { target, value, .. } => {
                self.expr(*target);
                if let Some(value) = value {
                    self.expr(*value);
                }
            }
            Stmt::AugAssign { target, value, .. } => {
                self.expr(*target);
                self.expr(*value);
            }
            Stmt::If { cond, .. } | Stmt::While { cond, .. } => self.expr(*cond),
            Stmt::For { iter, .. } => self.expr(*iter),
            Stmt::Match { scrutinee, arms } => {
                self.expr(*scrutinee);
                for arm in arms {
                    self.pat(arm.pat);
                    if let Some(guard) = arm.guard {
                        self.expr(guard);
                    }
                }
            }
            _ => {}
        }
    }

    /// collecting the locals bound by a pattern and its sub-patterns
    fn pat(&mut self, pat: PatId) {
        if let Some(&local) = self.res.pat_locals.get(pat) {
            self.found.insert(local);
        }
        match &self.body.pats[pat] {
            Pat::TupleStruct { fields, .. } => {
                for &field in fields {
                    self.pat(field);
                }
            }
            Pat::Or(pats) => {
                for &p in pats {
                    self.pat(p);
                }
            }
            _ => {}
        }
    }

    /// collecting the locals an expression and its subexpressions refer to
    fn expr(&mut self, e: ExprId) {
        if let Some(Res::Local(local)) = self.res.res.get(e) {
            self.found.insert(*local);
        }
        match &self.body.exprs[e] {
            Expr::FStr(parts) => {
                for part in parts {
                    if let FStrPart::Expr(e) = part {
                        self.expr(*e);
                    }
                }
            }
            Expr::Field { base, .. } | Expr::Unary { operand: base, .. } | Expr::Try(base) => {
                self.expr(*base)
            }
            Expr::Call { callee, args } => {
                self.expr(*callee);
                for arg in args {
                    self.expr(arg.value);
                }
            }
            Expr::Index { base, indices } => {
                self.expr(*base);
                for &index in indices {
                    self.expr(index);
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.expr(*lhs);
                self.expr(*rhs);
            }
            Expr::Array(items) => {
                for &item in items {
                    self.expr(item);
                }
            }
            _ => {}
        }
    }
}

/// an expression and the expressions inside it, each before its children.
/// The code generator counts these to estimate the size of a block
pub fn subexprs(body: &Body, root: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    let mut todo = vec![root];
    while let Some(e) = todo.pop() {
        out.push(e);
        match &body.exprs[e] {
            Expr::FStr(parts) => todo.extend(parts.iter().filter_map(|p| match p {
                FStrPart::Expr(e) => Some(*e),
                FStrPart::Text(_) => None,
            })),
            Expr::Field { base, .. } | Expr::Unary { operand: base, .. } | Expr::Try(base) => {
                todo.push(*base)
            }
            Expr::Call { callee, args } => {
                todo.push(*callee);
                todo.extend(args.iter().map(|a| a.value));
            }
            Expr::Index { base, indices } => {
                todo.push(*base);
                todo.extend(indices);
            }
            Expr::Binary { lhs, rhs, .. } => todo.extend([*lhs, *rhs]),
            Expr::Array(items) => todo.extend(items),
            _ => {}
        }
    }
    out
}
