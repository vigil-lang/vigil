//! this module generates random well-typed Core programs for differential
//! tests (Lean interpreter, generated code and kernel must agree). It is
//! deterministic in the seed, so a failure is reproduced from the seed.
//!
//! The programs are small but biased towards the tricky cases: literals at
//! the bounds, indices just past the end, offsets in or out of the packet,
//! division and shifts by any value, and casts between all integer types.

use crate::{Action, BinOp, Bits, Expr, IntTy, Program, Stmt, Ty, UnOp};

/// a small deterministic random generator, 64-bit xorshift. The state must
/// not be zero, otherwise it gives only zeros
pub struct Rng(pub u64);

impl Rng {
    /// the next value of the sequence
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// a value in `0..n`, or 0 when `n` is 0. The modulo gives a small bias,
    /// fine for tests
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }

    /// true with probability `percent` in a hundred
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }

    /// picking an element of a non-empty slice
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

/// every integer type
const INTS: [IntTy; 8] = [
    IntTy {
        signed: false,
        bits: Bits::B8,
    },
    IntTy {
        signed: false,
        bits: Bits::B16,
    },
    IntTy {
        signed: false,
        bits: Bits::B32,
    },
    IntTy {
        signed: false,
        bits: Bits::B64,
    },
    IntTy {
        signed: true,
        bits: Bits::B8,
    },
    IntTy {
        signed: true,
        bits: Bits::B16,
    },
    IntTy {
        signed: true,
        bits: Bits::B32,
    },
    IntTy {
        signed: true,
        bits: Bits::B64,
    },
];

/// how large programs get
#[derive(Clone, Copy)]
pub struct Limits {
    /// max statements in the whole program
    pub statements: usize,
    /// max nesting of `if` and `for`
    pub depth: usize,
    /// max count of a loop
    pub loop_count: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            statements: 12,
            depth: 3,
            loop_count: 12,
        }
    }
}

/// the state while generating one program
struct Gen<'a> {
    rng: &'a mut Rng,
    locals: Vec<Ty>,
    /// the locals for loop variables, one per nesting level, and whether an
    /// enclosing loop is using it
    loop_vars: Vec<(usize, bool)>,
    limits: Limits,
    /// statements still to generate
    budget: usize,
}

/// a random well-typed program
///
/// the locals are two to five integers, one boolean, one or two arrays and
/// a `u32` loop variable for each nesting level. Only its loop writes a loop
/// variable.
pub fn program(rng: &mut Rng, limits: Limits) -> Program {
    let mut locals = Vec::new();
    for _ in 0..rng.below(4) + 2 {
        locals.push(Ty::Int { t: rng.pick(&INTS) });
    }
    locals.push(Ty::Bool);
    for _ in 0..rng.below(2) + 1 {
        locals.push(Ty::Array {
            elem: rng.pick(&INTS),
            len: rng.below(6) + 1,
        });
    }
    let mut loop_vars = Vec::new();
    for _ in 0..limits.depth {
        loop_vars.push((locals.len(), false));
        locals.push(Ty::Int { t: IntTy::U32 });
    }
    let mut g = Gen {
        rng,
        locals,
        loop_vars,
        limits,
        budget: limits.statements,
    };
    let body = g.block(0);
    Program {
        locals: g.locals,
        body,
    }
}

impl Gen<'_> {
    /// the locals of a scalar type, without the loop variables
    fn scalars(&self, ty: &Ty) -> Vec<usize> {
        (0..self.locals.len())
            .filter(|&x| self.locals[x] == *ty && !self.loop_vars.iter().any(|&(l, _)| l == x))
            .collect()
    }

    /// the local arrays
    fn arrays(&self) -> Vec<usize> {
        (0..self.locals.len())
            .filter(|&x| matches!(self.locals[x], Ty::Array { .. }))
            .collect()
    }

    /// a literal: a small value, a bound or anything else
    fn literal(&mut self, t: IntTy) -> Expr {
        // a negative literal is a negation in Vigil, so the least signed
        // value has no literal. A 64-bit unsigned literal stops at `i64::MAX`
        let max = t.max().min(i64::MAX as i128);
        let min = if t.signed { t.min() + 1 } else { 0 };
        let value = match self.rng.below(4) {
            0 => self.rng.below(8) as i128,
            1 => min,
            2 => max,
            _ => {
                let span = (max - min) as u128 + 1;
                min + (self.rng.next_u64() as u128 % span) as i128
            }
        };
        Expr::Lit { t, value }
    }

    /// an integer expression, at most `depth` operators deep
    fn int(&mut self, t: IntTy, depth: usize) -> Expr {
        if depth == 0 || self.rng.chance(25) {
            let vars = self.scalars(&Ty::Int { t });
            // the variables of enclosing loops, if of this type. A loop
            // variable is read only inside its loop
            let loops: Vec<usize> = self
                .loop_vars
                .iter()
                .filter(|&&(_, used)| used && t == IntTy::U32)
                .map(|&(x, _)| x)
                .collect();
            return match self.rng.below(4) {
                0 if !vars.is_empty() => Expr::Var {
                    x: self.rng.pick(&vars),
                },
                1 if !loops.is_empty() => Expr::Var {
                    x: self.rng.pick(&loops),
                },
                _ => self.literal(t),
            };
        }
        let d = depth - 1;
        match self.rng.below(9) {
            0 | 1 => {
                let op = self.rng.pick(&[
                    BinOp::Add,
                    BinOp::Sub,
                    BinOp::Mul,
                    BinOp::Div,
                    BinOp::Rem,
                    BinOp::Band,
                    BinOp::Bor,
                    BinOp::Bxor,
                    BinOp::Shl,
                    BinOp::Shr,
                ]);
                Expr::Bin {
                    op,
                    t,
                    a: Box::new(self.int(t, d)),
                    b: Box::new(self.int(t, d)),
                }
            }
            2 => Expr::Un {
                // Vigil negates only signed integers, an unsigned `-` would
                // have no source
                op: if t.signed {
                    self.rng.pick(&[UnOp::Neg, UnOp::Bitnot])
                } else {
                    UnOp::Bitnot
                },
                t,
                e: Box::new(self.int(t, d)),
            },
            3 => {
                let from = self.rng.pick(&INTS);
                Expr::Cast {
                    t,
                    e: Box::new(self.int(from, d)),
                }
            }
            4 => Expr::Cast {
                t,
                e: Box::new(Expr::Packet {
                    offset: Box::new(self.offset(d)),
                }),
            },
            5 => Expr::Cast {
                t,
                e: Box::new(Expr::Len),
            },
            6 => {
                let arrays: Vec<usize> = self
                    .arrays()
                    .into_iter()
                    .filter(|&a| matches!(self.locals[a], Ty::Array { elem, .. } if elem == t))
                    .collect();
                match arrays.first() {
                    Some(_) => {
                        let array = self.rng.pick(&arrays);
                        let index_ty = self.rng.pick(&INTS);
                        Expr::Index {
                            array,
                            i: Box::new(self.index(index_ty, array, d)),
                        }
                    }
                    None => self.literal(t),
                }
            }
            _ => {
                let vars = self.scalars(&Ty::Int { t });
                match vars.first() {
                    Some(_) => Expr::Var {
                        x: self.rng.pick(&vars),
                    },
                    None => self.literal(t),
                }
            }
        }
    }

    /// an array index: mostly in bounds or one past the end, sometimes
    /// anything
    fn index(&mut self, t: IntTy, array: usize, depth: usize) -> Expr {
        let Ty::Array { len, .. } = self.locals[array] else {
            unreachable!()
        };
        if self.rng.chance(70) {
            let bound = (len as i128 + 1).min(t.max());
            Expr::Lit {
                t,
                value: self.rng.below(bound as u64) as i128,
            }
        } else {
            self.int(t, depth)
        }
    }

    /// a packet offset, a `u32`, mostly small so it falls both inside and
    /// outside short packets
    fn offset(&mut self, depth: usize) -> Expr {
        if self.rng.chance(70) {
            Expr::Lit {
                t: IntTy::U32,
                value: self.rng.below(24) as i128,
            }
        } else {
            self.int(IntTy::U32, depth)
        }
    }

    /// a `bool` expression, at most `depth` operators deep
    fn boolean(&mut self, depth: usize) -> Expr {
        if depth == 0 || self.rng.chance(20) {
            let vars = self.scalars(&Ty::Bool);
            return match vars.first() {
                Some(_) if self.rng.chance(50) => Expr::Var {
                    x: self.rng.pick(&vars),
                },
                _ => Expr::Bool {
                    value: self.rng.chance(50),
                },
            };
        }
        let d = depth - 1;
        match self.rng.below(4) {
            0 => Expr::Un {
                op: UnOp::Not,
                t: IntTy::U8,
                e: Box::new(self.boolean(d)),
            },
            1 => Expr::Bin {
                op: self.rng.pick(&[BinOp::And, BinOp::Or]),
                t: IntTy::U8,
                a: Box::new(self.boolean(d)),
                b: Box::new(self.boolean(d)),
            },
            _ => {
                let t = self.rng.pick(&INTS);
                Expr::Bin {
                    op: self.rng.pick(&[
                        BinOp::Eq,
                        BinOp::Ne,
                        BinOp::Lt,
                        BinOp::Le,
                        BinOp::Gt,
                        BinOp::Ge,
                    ]),
                    t,
                    a: Box::new(self.int(t, d)),
                    b: Box::new(self.int(t, d)),
                }
            }
        }
    }

    /// one to four statements, within the budget
    fn block(&mut self, nesting: usize) -> Stmt {
        let mut stmts = Vec::new();
        let count = self.rng.below(4) + 1;
        for _ in 0..count {
            if self.budget == 0 {
                break;
            }
            self.budget -= 1;
            stmts.push(self.stmt(nesting));
        }
        Stmt::seq(stmts)
    }

    /// a statement at a nesting level. At the deepest level, `if` and `for`
    /// become a boolean assignment
    fn stmt(&mut self, nesting: usize) -> Stmt {
        let depth = 3;
        match self.rng.below(10) {
            0..=2 => {
                let scalars: Vec<usize> = (0..self.locals.len())
                    .filter(|&x| {
                        !matches!(self.locals[x], Ty::Array { .. })
                            && !self.loop_vars.iter().any(|&(l, _)| l == x)
                    })
                    .collect();
                let x = self.rng.pick(&scalars);
                let e = match self.locals[x].clone() {
                    Ty::Int { t } => self.int(t, depth),
                    _ => self.boolean(depth),
                };
                Stmt::Assign { x, e }
            }
            3 => {
                let arrays = self.arrays();
                let array = self.rng.pick(&arrays);
                let Ty::Array { elem, .. } = self.locals[array] else {
                    unreachable!()
                };
                let index_ty = self.rng.pick(&INTS);
                let i = self.index(index_ty, array, 1);
                Stmt::Store {
                    array,
                    i,
                    e: self.int(elem, depth),
                }
            }
            4 => Stmt::PacketStore {
                offset: self.offset(1),
                e: self.int(IntTy::U8, depth),
            },
            5 | 6 if nesting < self.limits.depth => Stmt::Ite {
                cond: self.boolean(depth),
                thn: Box::new(self.block(nesting + 1)),
                els: Box::new(if self.rng.chance(50) {
                    self.block(nesting + 1)
                } else {
                    Stmt::Skip
                }),
            },
            7 if nesting < self.limits.depth => {
                let Some(slot) = self.loop_vars.iter().position(|&(_, used)| !used) else {
                    return Stmt::Skip;
                };
                self.loop_vars[slot].1 = true;
                let x = self.loop_vars[slot].0;
                let count = self.rng.below(self.limits.loop_count) + 1;
                let body = self.block(nesting + 1);
                self.loop_vars[slot].1 = false;
                Stmt::Loop {
                    x,
                    t: IntTy::U32,
                    count,
                    body: Box::new(body),
                }
            }
            8 if self.rng.chance(30) => Stmt::Ret {
                action: self.rng.pick(&Action::ALL),
            },
            _ => {
                let x = self.scalars(&Ty::Bool)[0];
                Stmt::Assign {
                    x,
                    e: self.boolean(depth),
                }
            }
        }
    }
}
