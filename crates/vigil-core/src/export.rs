//! this module exports a type-checked Vigil program to Core, or reports the
//! first construct outside the subset. The result always passes
//! [`crate::types::check`], otherwise it is a bug of the exporter.
//!
//! The subset: XDP and TC programs without tail calls, over scalars and
//! local integer arrays, with assignments (an array literal or an array
//! copy becomes one store per element), `if`/`else`, `for i in range(N)`
//! with a constant `N`, packet bytes `ctx.parse[Byte](offset)?.value`,
//! `ctx.len()` and returns.
//! A TC program is exported as XDP, with `OK`, `SHOT` and `REDIRECT` as
//! `PASS`, `DROP` and `REDIRECT`. The backend maps them back.

use std::collections::HashMap;

use vigil_sema::Analysis;
use vigil_sema::hir::{self, BinaryOp, Expr as HExpr, FnId, Stmt as HStmt, StmtId, UnaryOp};
use vigil_sema::infer::{InferenceResult, LocalId, Res};
use vigil_sema::program::ProgramKind;
use vigil_sema::resolve::Builtin;
use vigil_sema::ty::{IntTy as SIntTy, TyId, TyKind};

use crate::{Action, BinOp, Bits, Expr, IntTy, Program, Stmt, Ty, UnOp};

/// why a program is not in Vigil Core
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported {
    /// what is outside the subset, as a message for the user
    pub message: String,
    /// where it is, byte offsets in the file
    pub span: vigil_diagnostics::Span,
}

type Result<T> = std::result::Result<T, Unsupported>;

/// the Core of the program `f`, or the first reason it has none
///
/// the locals are the ones of `f` in inference order, without the context
/// parameter and `result`
pub fn export(analysis: &Analysis, f: FnId) -> Result<Program> {
    let data = &analysis.package.fns[f];
    let fail = |message: &str| Unsupported {
        message: message.to_string(),
        span: data.name.span,
    };
    let info = analysis
        .programs
        .get(f)
        .ok_or_else(|| fail("not a program"))?;
    if !matches!(info.kind, ProgramKind::Xdp | ProgramKind::Tc) {
        return Err(fail("Vigil Core has only XDP and TC programs"));
    }
    if info.tail.is_some() {
        return Err(fail("Vigil Core has no tail calls"));
    }
    let res = &analysis.bodies[f];
    let mut e = Exporter {
        an: analysis,
        body: &data.body,
        spans: &data.spans,
        res,
        slots: HashMap::new(),
        locals: Vec::new(),
    };
    // the context is not a Core local, the packet is read with
    // `Expr::Packet`. Neither is `result`, only the contracts see it
    for (id, local) in res.locals.iter().skip(data.params.len()) {
        if res.result_local == Some(id) {
            continue;
        }
        let ty = e.ty(local.ty).ok_or_else(|| Unsupported {
            message: format!("`{}` has a type Vigil Core does not have", local.name),
            span: local.span,
        })?;
        e.slots.insert(id, e.locals.len());
        e.locals.push(ty);
    }
    let body = e.block(&data.body.root)?;
    let program = Program {
        locals: e.locals,
        body,
    };
    // `run_safe` and the certified backend only cover well-typed programs,
    // so an ill-typed export is an error, not a warning
    if !crate::types::check(&program) {
        return Err(fail(&format!(
            "its export to Vigil Core does not type-check, a bug in Vigil: please report it at {} \
             with the program",
            vigil_diagnostics::ISSUES
        )));
    }
    Ok(program)
}

/// the state of an export: the function and the Core local of each of its
/// locals
struct Exporter<'a> {
    an: &'a Analysis,
    body: &'a hir::Body,
    spans: &'a hir::BodySpans,
    res: &'a InferenceResult,
    slots: HashMap<LocalId, usize>,
    locals: Vec<Ty>,
}

/// the Core version of a Vigil integer type
fn int_ty(int: SIntTy) -> IntTy {
    let bits = match int.bits() {
        8 => Bits::B8,
        16 => Bits::B16,
        32 => Bits::B32,
        _ => Bits::B64,
    };
    IntTy {
        signed: int.signed(),
        bits,
    }
}

impl Exporter<'_> {
    fn unsupported_expr<T>(&self, e: hir::ExprId, what: &str) -> Result<T> {
        Err(Unsupported {
            message: format!("{what} is not in Vigil Core"),
            span: self.spans.exprs[e],
        })
    }

    fn unsupported_stmt<T>(&self, s: StmtId, what: &str) -> Result<T> {
        Err(Unsupported {
            message: format!("{what} is not in Vigil Core"),
            span: self.spans.stmts[s],
        })
    }

    /// the Core type of a Vigil type, if any: integer, `bool` or integer
    /// array
    fn ty(&self, ty: TyId) -> Option<Ty> {
        match self.an.types.kind(ty) {
            TyKind::Int(int) => Some(Ty::Int { t: int_ty(*int) }),
            TyKind::Bool => Some(Ty::Bool),
            TyKind::Array(element, len) => match self.an.types.kind(*element) {
                TyKind::Int(int) => Some(Ty::Array {
                    elem: int_ty(*int),
                    len: *len,
                }),
                _ => None,
            },
            _ => None,
        }
    }

    /// the integer type inferred for an expression
    fn int_of(&self, e: hir::ExprId) -> Result<IntTy> {
        match self.res.expr_types.get(e).map(|&t| self.an.types.kind(t)) {
            Some(TyKind::Int(int)) => Ok(int_ty(*int)),
            _ => self.unsupported_expr(e, "a value that is not an integer"),
        }
    }

    /// the Core local an expression names, if any
    fn local(&self, e: hir::ExprId) -> Option<usize> {
        match self.res.res.get(e) {
            Some(Res::Local(local)) => self.slots.get(local).copied(),
            _ => None,
        }
    }

    fn block(&self, block: &hir::Block) -> Result<Stmt> {
        let stmts: Result<Vec<_>> = block.iter().map(|&s| self.stmt(s)).collect();
        Ok(Stmt::seq(stmts?))
    }

    /// the offset, if `e` is `ctx.parse[Byte](offset)?.value`. `Byte` must
    /// be the one of `std.net`, not a header of the same name elsewhere
    fn packet_byte(&self, e: hir::ExprId) -> Option<hir::ExprId> {
        let HExpr::Field { base, .. } = &self.body.exprs[e] else {
            return None;
        };
        let HExpr::Try(call) = &self.body.exprs[*base] else {
            return None;
        };
        let HExpr::Call { callee, args } = &self.body.exprs[*call] else {
            return None;
        };
        let Some(Res::PacketParse(header)) = self.res.res.get(*callee) else {
            return None;
        };
        let data = &self.an.package.structs[*header];
        let std_net = self.an.package.modules[data.module].path == ["std", "net"];
        (std_net && data.name.text == "Byte")
            .then(|| args.first().map(|a| a.value))
            .flatten()
    }

    fn stmt(&self, s: StmtId) -> Result<Stmt> {
        match &self.body.stmts[s] {
            HStmt::Pass | HStmt::Missing => Ok(Stmt::Skip),
            HStmt::Assign {
                target,
                value: Some(value),
                ..
            } => {
                // `a = [x, y]` or `a = [x] * N`: Core has no array values, so
                // we store the elements one by one
                if let Some(elements) = self.array_literal(*value) {
                    let array = match self.res.decl_locals.get(s) {
                        Some(local) => Some(self.slots[local]),
                        None => self.local(*target),
                    };
                    let Some(array) = array else {
                        return self.unsupported_stmt(s, "this assignment");
                    };
                    let stores: Result<Vec<_>> = elements
                        .into_iter()
                        .enumerate()
                        .map(|(i, element)| {
                            Ok(Stmt::Store {
                                array,
                                i: Expr::Lit {
                                    t: IntTy::U32,
                                    value: i as i128,
                                },
                                e: self.expr(element)?,
                            })
                        })
                        .collect();
                    return Ok(Stmt::seq(stores?));
                }
                // `b = a` with `a` an array: Core has no array values either,
                // so we copy the elements one by one
                if let Some(source) = self.local(*value)
                    && let Ty::Array { len, .. } = self.locals[source]
                {
                    let array = match self.res.decl_locals.get(s) {
                        Some(local) => Some(self.slots[local]),
                        None => self.local(*target),
                    };
                    let Some(array) = array else {
                        return self.unsupported_stmt(s, "this assignment");
                    };
                    let index = |i: u64| Expr::Lit {
                        t: IntTy::U32,
                        value: i128::from(i),
                    };
                    let copies = (0..len).map(|i| Stmt::Store {
                        array,
                        i: index(i),
                        e: Expr::Index {
                            array: source,
                            i: Box::new(index(i)),
                        },
                    });
                    return Ok(Stmt::seq(copies.collect()));
                }
                let e = self.expr(*value)?;
                if let Some(&local) = self.res.decl_locals.get(s) {
                    return Ok(Stmt::Assign {
                        x: self.slots[&local],
                        e,
                    });
                }
                self.assign(s, *target, e)
            }
            HStmt::AugAssign { target, op, value } => {
                let t = self.int_of(*target)?;
                let current = self.expr(*target)?;
                let op = self.binop(*value, *op)?;
                let e = Expr::Bin {
                    op,
                    t,
                    a: Box::new(current),
                    b: Box::new(self.second_operand(op, t, *value)?),
                };
                self.assign(s, *target, e)
            }
            HStmt::If {
                cond,
                then,
                otherwise,
            } => Ok(Stmt::Ite {
                cond: self.expr(*cond)?,
                thn: Box::new(self.block(then)?),
                els: Box::new(match otherwise {
                    Some(b) => self.block(b)?,
                    None => Stmt::Skip,
                }),
            }),
            HStmt::For { iter, body, .. } => {
                // the errors point at the iterated expression, because the
                // span of the whole loop runs on past its body
                let HExpr::Call { callee, args } = &self.body.exprs[*iter] else {
                    return self.unsupported_expr(*iter, "a loop over an array");
                };
                if self.res.res.get(*callee) != Some(&Res::Builtin(Builtin::Range)) {
                    return self.unsupported_expr(*iter, "this loop");
                }
                let [count] = args.as_slice() else {
                    return self.unsupported_expr(*iter, "`range(start, end)`");
                };
                let Some(count) = self.constant(count.value) else {
                    return self.unsupported_expr(*iter, "a loop of a length known at run time");
                };
                let binding = self.res.for_locals[s];
                let Ty::Int { t } = self.locals[self.slots[&binding]].clone() else {
                    return self.unsupported_stmt(s, "this loop");
                };
                Ok(Stmt::Loop {
                    x: self.slots[&binding],
                    t,
                    count: count.max(0) as u64,
                    body: Box::new(self.block(body)?),
                })
            }
            HStmt::Return(Some(value)) => match self.res.res.get(*value) {
                Some(Res::Variant(e, i)) if self.an.package.enums[*e].name.text == "XdpAction" => {
                    Ok(Stmt::Ret {
                        action: Action::ALL[*i as usize],
                    })
                }
                // the `TcAction` variants follow the `TC_ACT_*` values: `OK` is
                // 0, `SHOT` 2 and `REDIRECT` 7
                Some(Res::Variant(e, i)) if self.an.package.enums[*e].name.text == "TcAction" => {
                    let action = match i {
                        0 => Action::Pass,
                        2 => Action::Drop,
                        7 => Action::Redirect,
                        _ => {
                            let name = &self.an.package.enums[*e].variants[*i as usize].name.text;
                            return self.unsupported_stmt(
                                s,
                                &format!(
                                    "the TC verdict `{name}` (only `OK`, `SHOT` and `REDIRECT`)"
                                ),
                            );
                        }
                    };
                    Ok(Stmt::Ret { action })
                }
                _ => self.unsupported_stmt(
                    s,
                    "returning something other than `XdpAction.X` or `TcAction.X`",
                ),
            },
            _ => self.unsupported_stmt(s, "this statement"),
        }
    }

    /// `target = e`, where the target is a local, an array element or a
    /// packet byte
    fn assign(&self, s: StmtId, target: hir::ExprId, e: Expr) -> Result<Stmt> {
        if let Some(x) = self.local(target) {
            return Ok(Stmt::Assign { x, e });
        }
        if let Some(offset) = self.packet_byte(target) {
            return Ok(Stmt::PacketStore {
                offset: self.expr(offset)?,
                e,
            });
        }
        if let HExpr::Index { base, indices } = &self.body.exprs[target]
            && let Some(array) = self.local(*base)
        {
            return Ok(Stmt::Store {
                array,
                i: self.expr(indices[0])?,
                e,
            });
        }
        self.unsupported_stmt(s, "this assignment")
    }

    /// the value of a literal, `5` or `-5`
    fn literal(&self, e: hir::ExprId) -> Option<i128> {
        match &self.body.exprs[e] {
            HExpr::Int(v) => i128::try_from(*v).ok(),
            HExpr::Unary {
                op: UnaryOp::Neg,
                operand,
            } => match &self.body.exprs[*operand] {
                HExpr::Int(v) => i128::try_from(*v).ok().map(|v| -v),
                _ => None,
            },
            _ => None,
        }
    }

    /// the elements of `[x, y, ...]`, or `N` copies of `x` for `[x] * N`
    fn array_literal(&self, e: hir::ExprId) -> Option<Vec<hir::ExprId>> {
        match &self.body.exprs[e] {
            HExpr::Array(items) => Some(items.clone()),
            HExpr::Binary { lhs, .. } => match (self.res.res.get(e), &self.body.exprs[*lhs]) {
                (Some(Res::Repeat(n)), HExpr::Array(items)) if items.len() == 1 => {
                    Some(vec![items[0]; *n as usize])
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// the value of a compile time expression: a constant, a literal or a
    /// conversion of one like `u32(8)`
    fn constant(&self, e: hir::ExprId) -> Option<i128> {
        match self.res.res.get(e) {
            Some(Res::ConstValue(v)) => Some(*v),
            Some(Res::Const(c)) => match self.an.consts[*c].1 {
                Some(vigil_sema::db::ConstValue::Int(v)) => Some(v),
                _ => None,
            },
            _ => match &self.body.exprs[e] {
                HExpr::Int(v) => i128::try_from(*v).ok(),
                HExpr::Call { callee, args }
                    if matches!(
                        self.res.res.get(*callee),
                        Some(Res::Builtin(Builtin::Int(_)))
                    ) =>
                {
                    self.constant(args.first()?.value)
                }
                _ => None,
            },
        }
    }

    /// the Core operator of a Vigil one. Vigil's `/` has none
    fn binop(&self, e: hir::ExprId, op: BinaryOp) -> Result<BinOp> {
        Ok(match op {
            BinaryOp::Add => BinOp::Add,
            BinaryOp::Sub => BinOp::Sub,
            BinaryOp::Mul => BinOp::Mul,
            BinaryOp::FloorDiv => BinOp::Div,
            BinaryOp::Rem => BinOp::Rem,
            BinaryOp::BitAnd => BinOp::Band,
            BinaryOp::BitOr => BinOp::Bor,
            BinaryOp::BitXor => BinOp::Bxor,
            BinaryOp::Shl => BinOp::Shl,
            BinaryOp::Shr => BinOp::Shr,
            BinaryOp::Eq => BinOp::Eq,
            BinaryOp::Ne => BinOp::Ne,
            BinaryOp::Lt => BinOp::Lt,
            BinaryOp::Le => BinOp::Le,
            BinaryOp::Gt => BinOp::Gt,
            BinaryOp::Ge => BinOp::Ge,
            BinaryOp::And => BinOp::And,
            BinaryOp::Or => BinOp::Or,
            BinaryOp::Div => return self.unsupported_expr(e, "`/`"),
        })
    }

    /// the second operand of `op` on `t`
    ///
    /// in Vigil a shift amount can have another integer type, in Core it
    /// can't, so we convert it. The result is the same, because the amount
    /// is taken modulo the width and a conversion keeps the low bits.
    fn second_operand(&self, op: BinOp, t: IntTy, e: hir::ExprId) -> Result<Expr> {
        let b = self.expr(e)?;
        if !matches!(op, BinOp::Shl | BinOp::Shr) || self.int_of(e)? == t {
            return Ok(b);
        }
        Ok(match b {
            Expr::Lit { value, .. } => Expr::Lit { t, value },
            b => Expr::Cast { t, e: Box::new(b) },
        })
    }

    /// the Core of an expression
    fn expr(&self, e: hir::ExprId) -> Result<Expr> {
        if let Some(offset) = self.packet_byte(e) {
            return Ok(Expr::Packet {
                offset: Box::new(self.expr(offset)?),
            });
        }
        match &self.body.exprs[e] {
            HExpr::Int(v) => Ok(Expr::Lit {
                t: self.int_of(e)?,
                value: i128::try_from(*v).unwrap_or(i128::MAX),
            }),
            HExpr::Bool(b) => Ok(Expr::Bool { value: *b }),
            HExpr::Name(_) => {
                if let Some(x) = self.local(e) {
                    return Ok(Expr::Var { x });
                }
                match self.constant(e) {
                    Some(value) => Ok(Expr::Lit {
                        t: self.int_of(e)?,
                        value,
                    }),
                    None => self.unsupported_expr(e, "a name other than a local or a constant"),
                }
            }
            HExpr::Index { base, indices } => match self.local(*base) {
                Some(array) => Ok(Expr::Index {
                    array,
                    i: Box::new(self.expr(indices[0])?),
                }),
                None => self.unsupported_expr(e, "indexing something other than a local array"),
            },
            HExpr::Unary { op, operand } => {
                let core = match op {
                    UnaryOp::Neg => UnOp::Neg,
                    UnaryOp::BitNot => UnOp::Bitnot,
                    UnaryOp::Not => UnOp::Not,
                };
                // `-5` is a Core literal, not a negation
                if *op == UnaryOp::Neg
                    && let HExpr::Int(v) = &self.body.exprs[*operand]
                {
                    return Ok(Expr::Lit {
                        t: self.int_of(e)?,
                        value: -(i128::try_from(*v).unwrap_or(i128::MAX)),
                    });
                }
                let t = match core {
                    UnOp::Not => IntTy::U8,
                    _ => self.int_of(*operand)?,
                };
                Ok(Expr::Un {
                    op: core,
                    t,
                    e: Box::new(self.expr(*operand)?),
                })
            }
            HExpr::Binary { op, lhs, rhs } => {
                match self.res.res.get(e) {
                    Some(Res::Operator(_)) => {
                        return self.unsupported_expr(e, "an operator defined by a method");
                    }
                    Some(Res::Repeat(_)) => {
                        return self.unsupported_expr(e, "an array built by `[x] * N`");
                    }
                    _ => {}
                }
                let op = self.binop(e, *op)?;
                let t = if op.is_logical() {
                    IntTy::U8
                } else {
                    self.int_of(*lhs)?
                };
                Ok(Expr::Bin {
                    op,
                    t,
                    a: Box::new(self.expr(*lhs)?),
                    b: Box::new(self.second_operand(op, t, *rhs)?),
                })
            }
            HExpr::Call { callee, args } => match self.res.res.get(*callee) {
                // `u8(250)` is a literal, not a conversion
                Some(Res::Builtin(Builtin::Int(int))) if self.literal(args[0].value).is_some() => {
                    Ok(Expr::Lit {
                        t: int_ty(*int),
                        value: self.literal(args[0].value).unwrap(),
                    })
                }
                Some(Res::Builtin(Builtin::Int(int))) => Ok(Expr::Cast {
                    t: int_ty(*int),
                    e: Box::new(self.expr(args[0].value)?),
                }),
                Some(Res::PacketLen) => Ok(Expr::Len),
                _ => self.unsupported_expr(e, "this call"),
            },
            _ => self.unsupported_expr(e, "this expression"),
        }
    }
}
