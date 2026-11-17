//! this module prints Core as Vigil source, an XDP program named `core`
//! where local `x` is `vx`. Literals carry their type (`u8(250)`) and every
//! operation is parenthesized, so we don't rely on inference or precedence.
//!
//! Printing is not the inverse of exporting: locals are declared at the top
//! and a `for` declares its own variable. [`as_printed`] gives the program
//! that the printed source means, to compare with its export.

use std::fmt::Write;

use crate::{BinOp, Expr, Program, Stmt, Ty, UnOp};

/// collecting the loop variables, which are not declared at the top
fn loop_variables(s: &Stmt, out: &mut Vec<usize>) {
    match s {
        Stmt::Loop { x, body, .. } => {
            out.push(*x);
            loop_variables(body, out);
        }
        Stmt::Seq { a, b } => {
            loop_variables(a, out);
            loop_variables(b, out);
        }
        Stmt::Ite { thn, els, .. } => {
            loop_variables(thn, out);
            loop_variables(els, out);
        }
        _ => {}
    }
}

/// the program that the source of [`print`](print()) means
///
/// the locals set to zero, then the body of `p`, then
/// `return XdpAction.PASS`. Locals are numbered as Vigil does it: the ones
/// at the top first, then one for each `for`, in source order.
pub fn as_printed(p: &Program) -> Program {
    let mut loops = Vec::new();
    loop_variables(&p.body, &mut loops);
    let mut locals = Vec::new();
    let mut renamed = std::collections::HashMap::new();
    let mut stmts = Vec::new();
    for (x, ty) in p.locals.iter().enumerate() {
        if loops.contains(&x) {
            continue;
        }
        let n = locals.len();
        renamed.insert(x, n);
        locals.push(ty.clone());
        match ty {
            Ty::Int { t } => stmts.push(Stmt::Assign {
                x: n,
                e: Expr::Lit { t: *t, value: 0 },
            }),
            Ty::Bool => stmts.push(Stmt::Assign {
                x: n,
                e: Expr::Bool { value: false },
            }),
            Ty::Array { elem, len } => {
                for i in 0..*len {
                    stmts.push(Stmt::Store {
                        array: n,
                        i: Expr::Lit {
                            t: crate::IntTy::U32,
                            value: i as i128,
                        },
                        e: Expr::Lit { t: *elem, value: 0 },
                    });
                }
            }
        }
    }
    let mut r = Renumber {
        old: &p.locals,
        locals: &mut locals,
        names: renamed,
    };
    let body = r.stmt(&p.body);
    let mut flat = Vec::new();
    flatten(&body, &mut flat);
    stmts.extend(flat);
    stmts.push(Stmt::Ret {
        action: crate::Action::Pass,
    });
    Program {
        locals,
        body: Stmt::seq(stmts),
    }
}

/// the renumbering of locals done by [`as_printed`]
struct Renumber<'a> {
    /// the types of the locals of the original program
    old: &'a [Ty],
    /// the types of the locals of the new program, one more for each loop
    locals: &'a mut Vec<Ty>,
    /// the new number of each old local, in the current scope
    names: std::collections::HashMap<usize, usize>,
}

impl Renumber<'_> {
    fn x(&self, x: usize) -> usize {
        self.names[&x]
    }

    fn stmt(&mut self, s: &Stmt) -> Stmt {
        match s {
            Stmt::Skip => Stmt::Skip,
            Stmt::Assign { x, e } => Stmt::Assign {
                x: self.x(*x),
                e: self.expr(e),
            },
            Stmt::Store { array, i, e } => Stmt::Store {
                array: self.x(*array),
                i: self.expr(i),
                e: self.expr(e),
            },
            Stmt::PacketStore { offset, e } => Stmt::PacketStore {
                offset: self.expr(offset),
                e: self.expr(e),
            },
            Stmt::Seq { a, b } => Stmt::Seq {
                a: Box::new(self.stmt(a)),
                b: Box::new(self.stmt(b)),
            },
            Stmt::Ite { cond, thn, els } => Stmt::Ite {
                cond: self.expr(cond),
                thn: Box::new(self.stmt(thn)),
                els: Box::new(self.stmt(els)),
            },
            Stmt::Loop { x, t, count, body } => {
                // a new local for the variable of this `for`. We restore the
                // old number after the body, for a later loop over the same
                // Core local
                let n = self.locals.len();
                self.locals.push(self.old[*x].clone());
                let saved = self.names.insert(*x, n);
                let body = self.stmt(body);
                match saved {
                    Some(previous) => self.names.insert(*x, previous),
                    None => self.names.remove(x),
                };
                Stmt::Loop {
                    x: n,
                    t: *t,
                    count: *count,
                    body: Box::new(body),
                }
            }
            Stmt::Ret { action } => Stmt::Ret { action: *action },
        }
    }

    fn expr(&self, e: &Expr) -> Expr {
        let b = |e: &Expr| Box::new(self.expr(e));
        match e {
            Expr::Var { x } => Expr::Var { x: self.x(*x) },
            Expr::Index { array, i } => Expr::Index {
                array: self.x(*array),
                i: b(i),
            },
            Expr::Packet { offset } => Expr::Packet { offset: b(offset) },
            Expr::Un { op, t, e } => Expr::Un {
                op: *op,
                t: *t,
                e: b(e),
            },
            Expr::Bin { op, t, a, b: rhs } => Expr::Bin {
                op: *op,
                t: *t,
                a: b(a),
                b: b(rhs),
            },
            Expr::Cast { t, e } => Expr::Cast { t: *t, e: b(e) },
            other => other.clone(),
        }
    }
}

/// the statements of nested sequences, in order
fn flatten(s: &Stmt, out: &mut Vec<Stmt>) {
    match s {
        Stmt::Seq { a, b } => {
            flatten(a, out);
            flatten(b, out);
        }
        Stmt::Skip => {}
        other => out.push(other.clone()),
    }
}

/// the Vigil source of a Core program
pub fn print(p: &Program) -> String {
    let mut out = String::from(
        "from std.net import XdpContext, XdpAction, Byte\n\n@xdp\ndef core(ctx: XdpContext) -> XdpAction:\n",
    );
    let mut loops = Vec::new();
    loop_variables(&p.body, &mut loops);
    for (x, ty) in p.locals.iter().enumerate() {
        if loops.contains(&x) {
            continue;
        }
        let _ = match ty {
            Ty::Int { t } => writeln!(out, "    v{x}: {0} = {0}(0)", t.name()),
            Ty::Bool => writeln!(out, "    v{x}: bool = False"),
            Ty::Array { elem, len } => {
                writeln!(out, "    v{x}: {0}[{len}] = [{0}(0)] * {len}", elem.name())
            }
        };
    }
    stmt(&p.body, 1, &mut out);
    out.push_str("    return XdpAction.PASS\n");
    out
}

/// indentation for nesting `depth`
fn indent(depth: usize) -> String {
    "    ".repeat(depth)
}

/// printing a statement at a depth
fn stmt(s: &Stmt, depth: usize, out: &mut String) {
    let pad = indent(depth);
    let _ = match s {
        Stmt::Skip => Ok(()),
        Stmt::Assign { x, e } => writeln!(out, "{pad}v{x} = {}", expr(e)),
        Stmt::Store { array, i, e } => writeln!(out, "{pad}v{array}[{}] = {}", expr(i), expr(e)),
        Stmt::PacketStore { offset, e } => writeln!(
            out,
            "{pad}ctx.parse[Byte]({})?.value = {}",
            expr(offset),
            expr(e)
        ),
        Stmt::Seq { a, b } => {
            stmt(a, depth, out);
            stmt(b, depth, out);
            Ok(())
        }
        Stmt::Ite { cond, thn, els } => {
            let _ = writeln!(out, "{pad}if {}:", expr(cond));
            block(thn, depth + 1, out);
            if **els != Stmt::Skip {
                let _ = writeln!(out, "{pad}else:");
                block(els, depth + 1, out);
            }
            Ok(())
        }
        Stmt::Loop { x, t, count, body } => {
            let _ = writeln!(out, "{pad}for v{x} in range({}({count})):", t.name());
            block(body, depth + 1, out);
            Ok(())
        }
        Stmt::Ret { action } => writeln!(out, "{pad}return XdpAction.{}", action.variant()),
    };
}

/// printing the body of an `if`, `else` or `for`. We write `pass` when it
/// is empty, because a block can't be empty
fn block(s: &Stmt, depth: usize, out: &mut String) {
    let before = out.len();
    stmt(s, depth, out);
    if out.len() == before {
        let _ = writeln!(out, "{}pass", indent(depth));
    }
}

/// the source of an expression, with every operation parenthesized
pub fn expr(e: &Expr) -> String {
    match e {
        Expr::Lit { t, value } => format!("{}({value})", t.name()),
        Expr::Bool { value } => if *value { "True" } else { "False" }.into(),
        Expr::Var { x } => format!("v{x}"),
        Expr::Index { array, i } => format!("v{array}[{}]", expr(i)),
        Expr::Packet { offset } => format!("ctx.parse[Byte]({})?.value", expr(offset)),
        Expr::Len => "ctx.len()".into(),
        Expr::Un { op, e, .. } => match op {
            UnOp::Neg => format!("(-{})", expr(e)),
            UnOp::Bitnot => format!("(~{})", expr(e)),
            UnOp::Not => format!("(not {})", expr(e)),
        },
        Expr::Bin { op, a, b, .. } => {
            let symbol = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "//",
                BinOp::Rem => "%",
                BinOp::Band => "&",
                BinOp::Bor => "|",
                BinOp::Bxor => "^",
                BinOp::Shl => "<<",
                BinOp::Shr => ">>",
                BinOp::Eq => "==",
                BinOp::Ne => "!=",
                BinOp::Lt => "<",
                BinOp::Le => "<=",
                BinOp::Gt => ">",
                BinOp::Ge => ">=",
                BinOp::And => "and",
                BinOp::Or => "or",
            };
            format!("({} {symbol} {})", expr(a), expr(b))
        }
        Expr::Cast { t, e } => format!("{}({})", t.name(), expr(e)),
    }
}
