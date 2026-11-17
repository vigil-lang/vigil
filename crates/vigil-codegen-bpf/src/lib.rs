//! this crate compiles Vigil Core to eBPF instructions, without LLVM. The
//! entry point is [`compile_for`], and [`elf`] wraps the result in an object.
//!
//! It is the compiler of the Lean formalization of the full Vigil,
//! instruction for instruction: both give the same bytes, which the model
//! verifier checks. A change here and not in Lean breaks that, so the
//! comments give the Lean names.
//!
//! The shape of the code:
//!
//! - `data` and `data_end` are loaded once in `r7` and `r8`. For TC the
//!   context is also kept in `r6`, for `ctx.len()`
//! - expressions use `r0`-`r5` and `r9`, one register per nesting level, so
//!   seven levels at most
//! - every local, array element and loop counter has its own zeroed 8-byte
//!   stack slot
//! - results are sign or zero extended to 64 bits after any operation that
//!   can leave other bits, as in the Lean semantics
//! - packet bytes only at constant offsets, each with a bounds check, and an
//!   access past the end passes the packet
//! - arrays are indexed by a chain of comparisons, so every stack access is
//!   at a constant offset. Out of bounds aborts
//! - a loop keeps its counter in a stack slot and jumps back to its head

pub mod elf;

use vigil_core::{Action, BinOp, Bits, Expr, IntTy, Program, Stmt, Ty, UnOp};

/// the first offset a program can't reach, `maxPacketOffset` in Lean. An
/// access here or beyond passes the packet
const MAX_PACKET_OFFSET: u64 = 0xFFFF;

/// the registers we use. `r6` only keeps the context, for TC
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reg {
    R0 = 0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    R10,
}

/// the registers for expressions, by depth. These are the ones left after
/// the packet bounds (`r7`, `r8`) and the frame pointer (`r10`)
const TEMPS: [Reg; 7] = [
    Reg::R0,
    Reg::R1,
    Reg::R2,
    Reg::R3,
    Reg::R4,
    Reg::R5,
    Reg::R9,
];

/// an instruction slot, the fields of its 8-byte encoding. An `lddw` takes
/// two slots, the second with the upper half of the constant
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Insn {
    opcode: u8,
    dst: u8,
    src: u8,
    off: i16,
    imm: i32,
}

/// the opcode parts we use: class (`ALU64`, `JMP`), source (`K` immediate,
/// `X` register), operation, and the full opcodes of loads and stores
mod op {
    pub const ALU64: u8 = 0x07;
    pub const JMP: u8 = 0x05;
    pub const K: u8 = 0x00;
    pub const X: u8 = 0x08;
    pub const ADD: u8 = 0x00;
    pub const SUB: u8 = 0x10;
    pub const MUL: u8 = 0x20;
    pub const DIV: u8 = 0x30;
    pub const OR: u8 = 0x40;
    pub const AND: u8 = 0x50;
    pub const LSH: u8 = 0x60;
    pub const RSH: u8 = 0x70;
    pub const NEG: u8 = 0x80;
    pub const MOD: u8 = 0x90;
    pub const XOR: u8 = 0xa0;
    pub const MOV: u8 = 0xb0;
    pub const ARSH: u8 = 0xc0;
    pub const JA: u8 = 0x00;
    pub const JEQ: u8 = 0x10;
    pub const JGT: u8 = 0x20;
    pub const JGE: u8 = 0x30;
    pub const JNE: u8 = 0x50;
    pub const JSGT: u8 = 0x60;
    pub const JSGE: u8 = 0x70;
    pub const EXIT: u8 = 0x90;
    pub const JLT: u8 = 0xa0;
    pub const JLE: u8 = 0xb0;
    pub const JSLT: u8 = 0xc0;
    pub const JSLE: u8 = 0xd0;
    pub const LDXDW: u8 = 0x79;
    pub const LDXW: u8 = 0x61;
    pub const LDXB: u8 = 0x71;
    pub const STXDW: u8 = 0x7b;
    pub const STXB: u8 = 0x73;
    pub const LDDW: u8 = 0x18;
}

fn alu_k(code: u8, dst: Reg, imm: i64) -> Insn {
    Insn {
        opcode: op::ALU64 | op::K | code,
        dst: dst as u8,
        src: 0,
        off: 0,
        imm: imm as i32,
    }
}

fn alu_x(code: u8, dst: Reg, src: Reg) -> Insn {
    Insn {
        opcode: op::ALU64 | op::X | code,
        dst: dst as u8,
        src: src as u8,
        off: 0,
        imm: 0,
    }
}

fn neg(dst: Reg) -> Insn {
    Insn {
        opcode: op::ALU64 | op::NEG,
        dst: dst as u8,
        src: 0,
        off: 0,
        imm: 0,
    }
}

fn jmp_k(code: u8, dst: Reg, imm: i64, off: i64) -> Insn {
    Insn {
        opcode: op::JMP | op::K | code,
        dst: dst as u8,
        src: 0,
        off: off as i16,
        imm: imm as i32,
    }
}

fn jmp_x(code: u8, dst: Reg, src: Reg, off: i64) -> Insn {
    Insn {
        opcode: op::JMP | op::X | code,
        dst: dst as u8,
        src: src as u8,
        off: off as i16,
        imm: 0,
    }
}

fn ja(off: i64) -> Insn {
    Insn {
        opcode: op::JMP | op::JA,
        dst: 0,
        src: 0,
        off: off as i16,
        imm: 0,
    }
}

fn exit() -> Insn {
    Insn {
        opcode: op::JMP | op::EXIT,
        dst: 0,
        src: 0,
        off: 0,
        imm: 0,
    }
}

fn load(code: u8, dst: Reg, src: Reg, off: i64) -> Insn {
    Insn {
        opcode: code,
        dst: dst as u8,
        src: src as u8,
        off: off as i16,
        imm: 0,
    }
}

fn store(code: u8, dst: Reg, off: i64, src: Reg) -> Insn {
    Insn {
        opcode: code,
        dst: dst as u8,
        src: src as u8,
        off: off as i16,
        imm: 0,
    }
}

/// the offset of stack slot `k` from `r10`. Slot 0 is just below the frame
/// pointer
fn slot_off(k: usize) -> i64 {
    -8 * (k as i64 + 1)
}

fn load_slot(r: Reg, k: usize) -> Insn {
    load(op::LDXDW, r, Reg::R10, slot_off(k))
}

fn store_slot(k: usize, r: Reg) -> Insn {
    store(op::STXDW, Reg::R10, slot_off(k), r)
}

fn bits(b: Bits) -> u32 {
    match b {
        Bits::B8 => 8,
        Bits::B16 => 16,
        Bits::B32 => 32,
        Bits::B64 => 64,
    }
}

/// `v` as an integer of type `t`, extended to 64 bits (`IntTy.norm` in Lean)
fn norm_value(t: IntTy, v: u64) -> u64 {
    let b = bits(t.bits);
    if b == 64 {
        return v;
    }
    let low = v & ((1u64 << b) - 1);
    if t.signed && low >> (b - 1) == 1 {
        low | !((1u64 << b) - 1)
    } else {
        low
    }
}

/// normalizing register `r` for type `t`, as [`norm_value`]: nothing for 64
/// bits, a mask for narrow unsigned types, two shifts otherwise
///
/// `u32` uses the shifts too, because immediates are sign extended, so the
/// mask `0xFFFF_FFFF` would be all ones
fn norm(t: IntTy, r: Reg) -> Vec<Insn> {
    let b = bits(t.bits);
    match (t.signed, b) {
        (_, 64) => vec![],
        (true, b) => {
            let s = 64 - i64::from(b);
            vec![alu_k(op::LSH, r, s), alu_k(op::ARSH, r, s)]
        }
        (false, 32) => vec![alu_k(op::LSH, r, 32), alu_k(op::RSH, r, 32)],
        (false, b) => vec![alu_k(op::AND, r, (1i64 << b) - 1)],
    }
}

/// loading `c` in `r`: a `mov` if it fits a sign-extended 32-bit immediate,
/// an `lddw` otherwise
fn load_const(r: Reg, c: u64) -> Vec<Insn> {
    let signed = c as i64;
    if (-(1i64 << 31)..(1i64 << 31)).contains(&signed) {
        vec![alu_k(op::MOV, r, signed)]
    } else {
        vec![
            Insn {
                opcode: op::LDDW,
                dst: r as u8,
                src: 0,
                off: 0,
                imm: c as u32 as i32,
            },
            Insn {
                opcode: 0,
                dst: 0,
                src: 0,
                off: 0,
                imm: (c >> 32) as u32 as i32,
            },
        ]
    }
}

/// `r` as a boolean: 0 stays 0, anything else becomes 1
fn to_bool(r: Reg) -> Vec<Insn> {
    vec![jmp_k(op::JEQ, r, 0, 1), alu_k(op::MOV, r, 1)]
}

/// setting `r` to 1 if the jump `code` on `r` and `rb` is taken, 0 otherwise
fn compare(code: u8, r: Reg, rb: Reg) -> Vec<Insn> {
    vec![
        jmp_x(code, r, rb, 2),
        alu_k(op::MOV, r, 0),
        ja(1),
        alu_k(op::MOV, r, 1),
    ]
}

/// returning the XDP verdict `a`. This is the only place that emits an
/// `exit`, and [`compile_for`] relies on it to rewrite verdicts per target
fn exit_with(a: Action) -> Vec<Insn> {
    vec![alu_k(op::MOV, Reg::R0, i64::from(a.value())), exit()]
}

/// does `e` read nothing (no local, array, packet byte or length)?
fn closed(e: &Expr) -> bool {
    match e {
        Expr::Lit { .. } | Expr::Bool { .. } => true,
        Expr::Var { .. } | Expr::Index { .. } | Expr::Packet { .. } | Expr::Len => false,
        Expr::Un { e, .. } | Expr::Cast { e, .. } => closed(e),
        Expr::Bin { a, b, .. } => closed(a) && closed(b),
    }
}

/// the value of a closed expression, `eval [] []` in Lean
///
/// [`expr`] folds every closed expression, and this also gives the constant
/// packet offsets that the bounds checks need
fn constant(e: &Expr) -> Option<u64> {
    if !closed(e) {
        return None;
    }
    eval(e)
}

/// the value of an expression, `None` if it reads a local, an array or the
/// packet. `and` and `or` short-circuit, like their compiled code
fn eval(e: &Expr) -> Option<u64> {
    Some(match e {
        Expr::Lit { t, value } => norm_value(*t, *value as i64 as u64),
        Expr::Bool { value } => u64::from(*value),
        Expr::Un { op, t, e } => {
            let v = eval(e)?;
            match op {
                UnOp::Neg => norm_value(*t, 0u64.wrapping_sub(v)),
                UnOp::Bitnot => norm_value(*t, !v),
                UnOp::Not => u64::from(v == 0),
            }
        }
        Expr::Bin {
            op: BinOp::And,
            a,
            b,
            ..
        } => {
            if eval(a)? == 0 {
                0
            } else {
                u64::from(eval(b)? != 0)
            }
        }
        Expr::Bin {
            op: BinOp::Or,
            a,
            b,
            ..
        } => {
            if eval(a)? != 0 {
                1
            } else {
                u64::from(eval(b)? != 0)
            }
        }
        Expr::Bin { op, t, a, b } => binop(*op, *t, eval(a)?, eval(b)?),
        Expr::Cast { t, e } => norm_value(*t, eval(e)?),
        _ => return None,
    })
}

/// a binary operation on two values of type `t`, `binop` in Lean
///
/// corner cases as in eBPF: division by zero gives 0, remainder by zero
/// gives the dividend, and shift counts are modulo the width. [`eval`]
/// handles `and` and `or` itself, to short-circuit.
fn binop(op: BinOp, t: IntTy, a: u64, b: u64) -> u64 {
    let w = bits(t.bits);
    let shift = (b % u64::from(w)) as u32;
    let (sa, sb) = (a as i64, b as i64);
    let lt = if t.signed { sa < sb } else { a < b };
    let le = if t.signed { sa <= sb } else { a <= b };
    let n = |v| norm_value(t, v);
    match op {
        BinOp::Add => n(a.wrapping_add(b)),
        BinOp::Sub => n(a.wrapping_sub(b)),
        BinOp::Mul => n(a.wrapping_mul(b)),
        BinOp::Div => n(if b == 0 {
            0
        } else if t.signed {
            sa.wrapping_div(sb) as u64
        } else {
            a / b
        }),
        BinOp::Rem => n(if b == 0 {
            a
        } else if t.signed {
            sa.wrapping_rem(sb) as u64
        } else {
            a % b
        }),
        BinOp::Band => n(a & b),
        BinOp::Bor => n(a | b),
        BinOp::Bxor => n(a ^ b),
        BinOp::Shl => n(a << shift),
        BinOp::Shr => n(if t.signed {
            (sa >> shift) as u64
        } else {
            a >> shift
        }),
        BinOp::Eq => u64::from(a == b),
        BinOp::Ne => u64::from(a != b),
        BinOp::Lt => u64::from(lt),
        BinOp::Le => u64::from(le),
        BinOp::Gt => u64::from(!le),
        BinOp::Ge => u64::from(!lt),
        BinOp::And | BinOp::Or => u64::from(a != 0 && b != 0),
    }
}

fn len(code: &[Insn]) -> i64 {
    code.len() as i64
}

/// reading the packet byte at offset `n` into `r`
///
/// first the bounds check the verifiers (Linux and model) need: if
/// `data + n + 1` is past `data_end`, we pass the packet. At
/// [`MAX_PACKET_OFFSET`] or beyond we read nothing. Load offsets are signed
/// 16 bits, so from `0x7fff` on we compute the address in `r`.
fn packet_byte(r: Reg, n: u64) -> Vec<Insn> {
    if n >= MAX_PACKET_OFFSET {
        return exit_with(Action::Pass);
    }
    let mut code = vec![
        alu_x(op::MOV, r, Reg::R7),
        alu_k(op::ADD, r, n as i64 + 1),
        jmp_x(op::JLE, r, Reg::R8, 2),
    ];
    code.extend(exit_with(Action::Pass));
    if n < 0x7fff {
        code.push(load(op::LDXB, r, Reg::R7, n as i64));
    } else {
        code.extend([
            alu_x(op::MOV, r, Reg::R7),
            alu_k(op::ADD, r, n as i64),
            load(op::LDXB, r, r, 0),
        ]);
    }
    code
}

/// signed division (`DIV`) or remainder (`MOD`) of `r` by `rb`, from the
/// unsigned operations, which are all eBPF has
///
/// we take absolute values, divide, and negate the result when needed. `rs`
/// keeps the sign: the xor of the operands for a division, the dividend
/// for a remainder. `rb` is overwritten.
fn signed_div_mod(code: u8, t: IntTy, r: Reg, rb: Reg, rs: Reg) -> Vec<Insn> {
    let mut out = if code == op::DIV {
        vec![alu_x(op::MOV, rs, r), alu_x(op::XOR, rs, rb)]
    } else {
        vec![alu_x(op::MOV, rs, r)]
    };
    out.extend([
        jmp_k(op::JSGE, r, 0, 1),
        neg(r),
        jmp_k(op::JSGE, rb, 0, 1),
        neg(rb),
        alu_x(code, r, rb),
        jmp_k(op::JSGE, rs, 0, 1),
        neg(r),
    ]);
    out.extend(norm(t, r));
    out
}

/// a binary operation of type `t` on `r` and `rb`, result in `r`. `rs` is
/// scratch for signed division
///
/// we normalize only what can leave bits outside the type. Bitwise ops,
/// unsigned division and remainder and right shifts keep values normalized.
/// Shift counts are masked first, as in [`binop`]. `and` and `or` give
/// nothing here, [`expr`] compiles them with jumps.
fn bin_code(op_: BinOp, t: IntTy, r: Reg, rb: Reg, rs: Reg) -> Vec<Insn> {
    let shift_mask = alu_k(op::AND, rb, i64::from(bits(t.bits)) - 1);
    let with_norm = |first: Insn| {
        let mut v = vec![first];
        v.extend(norm(t, r));
        v
    };
    let pick = |s, u| if t.signed { s } else { u };
    match op_ {
        BinOp::Add => with_norm(alu_x(op::ADD, r, rb)),
        BinOp::Sub => with_norm(alu_x(op::SUB, r, rb)),
        BinOp::Mul => with_norm(alu_x(op::MUL, r, rb)),
        BinOp::Band => vec![alu_x(op::AND, r, rb)],
        BinOp::Bor => vec![alu_x(op::OR, r, rb)],
        BinOp::Bxor => vec![alu_x(op::XOR, r, rb)],
        BinOp::Div if t.signed => signed_div_mod(op::DIV, t, r, rb, rs),
        BinOp::Div => vec![alu_x(op::DIV, r, rb)],
        BinOp::Rem if t.signed => signed_div_mod(op::MOD, t, r, rb, rs),
        BinOp::Rem => vec![alu_x(op::MOD, r, rb)],
        BinOp::Shl => {
            let mut v = vec![shift_mask, alu_x(op::LSH, r, rb)];
            v.extend(norm(t, r));
            v
        }
        BinOp::Shr => vec![shift_mask, alu_x(pick(op::ARSH, op::RSH), r, rb)],
        BinOp::Eq => compare(op::JEQ, r, rb),
        BinOp::Ne => compare(op::JNE, r, rb),
        BinOp::Lt => compare(pick(op::JSLT, op::JLT), r, rb),
        BinOp::Le => compare(pick(op::JSLE, op::JLE), r, rb),
        BinOp::Gt => compare(pick(op::JSGT, op::JGT), r, rb),
        BinOp::Ge => compare(pick(op::JSGE, op::JGE), r, rb),
        BinOp::And | BinOp::Or => vec![],
    }
}

/// where the locals live on the stack
struct Frame {
    /// the slot of each local, for an array the slot of element 0
    slots: Vec<usize>,
    /// the length of each array local, 0 for the others
    lens: Vec<usize>,
    /// the first slot of the loop counters, one per nesting level
    counters: usize,
    /// where the code runs, for the code of `ctx.len()`
    target: Target,
}

/// the register of depth `d`, or an error past the seven we have
fn tmp(d: usize) -> Result<Reg, String> {
    TEMPS
        .get(d)
        .copied()
        .ok_or_else(|| "an expression too deep (more than 7 registers)".to_string())
}

/// error for a packet byte at a variable offset, which neither this backend
/// nor the model reads, while the LLVM backend does
const VARIABLE_OFFSET: &str =
    "a packet byte at a variable offset (the certified backend reads only constant offsets)";

/// comparing `idx` with every index of array `a`, each followed by `access`
/// to that slot. Out of bounds aborts
///
/// each link is three instructions, so the jump ending a link skips the
/// following links and the abort
fn element_chain(f: &Frame, a: usize, idx: Reg, access: impl Fn(usize) -> Insn) -> Vec<Insn> {
    let n = f.lens[a];
    let mut code = Vec::new();
    for k in 0..n {
        code.extend([
            jmp_k(op::JNE, idx, k as i64, 2),
            access(f.slots[a] + k),
            ja(3 * (n as i64 - k as i64) - 1),
        ]);
    }
    code.extend(exit_with(Action::Aborted));
    code
}

/// compiling `e` into the register of depth `d`, without touching the
/// registers below
fn expr(f: &Frame, d: usize, e: &Expr) -> Result<Vec<Insn>, String> {
    let r = tmp(d)?;
    if let Some(c) = constant(e) {
        return Ok(load_const(r, c));
    }
    Ok(match e {
        Expr::Lit { .. } | Expr::Bool { .. } => unreachable!("constants are folded"),
        Expr::Var { x } => vec![load_slot(r, f.slots[*x])],
        Expr::Index { array, i } => {
            let mut code = expr(f, d, i)?;
            code.extend(element_chain(f, *array, r, |k| load_slot(r, k)));
            code
        }
        Expr::Packet { offset } => match constant(offset) {
            Some(n) => packet_byte(r, n),
            None => return Err(VARIABLE_OFFSET.into()),
        },
        // for TC, `__sk_buff.len` at offset 0 of the context, as the LLVM
        // backend reads it, because it counts the non linear data too.
        // Otherwise `data_end - data`, as a `u32`
        Expr::Len if f.target == Target::Tc => vec![load(op::LDXW, r, Reg::R6, 0)],
        Expr::Len => {
            let mut code = vec![alu_x(op::MOV, r, Reg::R8), alu_x(op::SUB, r, Reg::R7)];
            code.extend(norm(IntTy::U32, r));
            code
        }
        Expr::Un { op: o, t, e } => {
            let mut code = expr(f, d, e)?;
            match o {
                UnOp::Neg => {
                    code.push(neg(r));
                    code.extend(norm(*t, r));
                }
                UnOp::Bitnot => {
                    code.push(alu_k(op::XOR, r, -1));
                    code.extend(norm(*t, r));
                }
                UnOp::Not => code.extend([
                    jmp_k(op::JEQ, r, 0, 2),
                    alu_k(op::MOV, r, 0),
                    ja(1),
                    alu_k(op::MOV, r, 1),
                ]),
            }
            code
        }
        // `b` is skipped when `a` is 0. Both use the same register, because
        // `a` is not needed after the test
        Expr::Bin {
            op: BinOp::And,
            a,
            b,
            ..
        } => {
            let mut code = expr(f, d, a)?;
            let cb = expr(f, d, b)?;
            code.push(jmp_k(op::JEQ, r, 0, len(&cb) + 2));
            code.extend(cb);
            code.extend(to_bool(r));
            code
        }
        // `b` is skipped and the result is 1 when `a` is not 0
        Expr::Bin {
            op: BinOp::Or,
            a,
            b,
            ..
        } => {
            let mut code = expr(f, d, a)?;
            let cb = expr(f, d, b)?;
            code.extend([
                jmp_k(op::JEQ, r, 0, 2),
                alu_k(op::MOV, r, 1),
                ja(len(&cb) + 2),
            ]);
            code.extend(cb);
            code.extend(to_bool(r));
            code
        }
        Expr::Bin { op: o, t, a, b } => {
            let mut code = expr(f, d, a)?;
            code.extend(expr(f, d + 1, b)?);
            let rb = tmp(d + 1)?;
            let rs = if matches!(o, BinOp::Div | BinOp::Rem) && t.signed {
                tmp(d + 2)?
            } else {
                rb
            };
            code.extend(bin_code(*o, *t, r, rb, rs));
            code
        }
        Expr::Cast { t, e } => {
            let mut code = expr(f, d, e)?;
            code.extend(norm(*t, r));
            code
        }
    })
}

/// compiling `s` inside `depth` loops. A loop at that depth keeps its
/// counter in slot `counters + depth` of the [`Frame`]
fn stmt(f: &Frame, depth: usize, s: &Stmt) -> Result<Vec<Insn>, String> {
    Ok(match s {
        Stmt::Skip => vec![],
        Stmt::Assign { x, e } => {
            let mut code = expr(f, 0, e)?;
            code.push(store_slot(f.slots[*x], Reg::R0));
            code
        }
        // the value first, then the place, as Python evaluates them
        Stmt::Store { array, i, e } => {
            let mut code = expr(f, 0, e)?;
            code.extend(expr(f, 1, i)?);
            code.extend(element_chain(f, *array, Reg::R1, |k| {
                store_slot(k, Reg::R0)
            }));
            code
        }
        // same as `packet_byte`, with a store instead of a load
        Stmt::PacketStore { offset, e } => {
            let mut code = expr(f, 0, e)?;
            let Some(n) = constant(offset) else {
                return Err(VARIABLE_OFFSET.into());
            };
            if n < MAX_PACKET_OFFSET {
                code.extend([
                    alu_x(op::MOV, Reg::R1, Reg::R7),
                    alu_k(op::ADD, Reg::R1, n as i64 + 1),
                    jmp_x(op::JLE, Reg::R1, Reg::R8, 2),
                ]);
                code.extend(exit_with(Action::Pass));
                if n < 0x7fff {
                    code.push(store(op::STXB, Reg::R7, n as i64, Reg::R0));
                } else {
                    code.extend([
                        alu_x(op::MOV, Reg::R1, Reg::R7),
                        alu_k(op::ADD, Reg::R1, n as i64),
                        store(op::STXB, Reg::R1, 0, Reg::R0),
                    ]);
                }
            } else {
                code.extend(exit_with(Action::Pass));
            }
            code
        }
        Stmt::Seq { a, b } => {
            let mut code = stmt(f, depth, a)?;
            code.extend(stmt(f, depth, b)?);
            code
        }
        Stmt::Ite { cond, thn, els } => {
            let mut code = expr(f, 0, cond)?;
            let ca = stmt(f, depth, thn)?;
            let cb = stmt(f, depth, els)?;
            code.push(jmp_k(op::JEQ, Reg::R0, 0, len(&ca) + 1));
            code.extend(ca);
            code.push(ja(len(&cb)));
            code.extend(cb);
            code
        }
        // the counter starts at 0 and the head exits once it reaches
        // `count`. Otherwise `x` gets the counter as a `t`, the body runs
        // and the counter is incremented before jumping back. It lives on
        // the stack because the body can use every register
        Stmt::Loop { x, t, count, body } => {
            if *count >= 1 << 31 {
                return Err("a loop of 2^31 iterations or more".into());
            }
            let k = f.counters + depth;
            let cbody = stmt(f, depth + 1, body)?;
            let mut rest = vec![alu_x(op::MOV, Reg::R1, Reg::R0)];
            rest.extend(norm(*t, Reg::R1));
            rest.push(store_slot(f.slots[*x], Reg::R1));
            rest.extend(cbody);
            rest.extend([
                load_slot(Reg::R0, k),
                alu_k(op::ADD, Reg::R0, 1),
                store_slot(k, Reg::R0),
            ]);
            let head = [
                load_slot(Reg::R0, k),
                jmp_k(op::JGE, Reg::R0, *count as i64, len(&rest) + 1),
            ];
            let mut code = vec![alu_k(op::MOV, Reg::R0, 0), store_slot(k, Reg::R0)];
            code.extend(head);
            let back = -(head.len() as i64 + len(&rest) + 1);
            code.extend(rest);
            code.push(ja(back));
            code
        }
        Stmt::Ret { action } => exit_with(*action),
    })
}

/// the successors of instruction `pc`: none for `exit`, the target of `ja`,
/// next and target for a conditional jump, and past both slots of `lddw`
fn successors(code: &[Insn], pc: usize) -> Vec<usize> {
    let Some(i) = code.get(pc) else {
        return vec![];
    };
    let target = |off: i16| {
        let t = pc as i64 + 1 + i64::from(off);
        (t >= 0).then_some(t as usize)
    };
    match i.opcode {
        0x95 => vec![],
        0x05 => target(i.off).into_iter().collect(),
        op::LDDW => vec![pc + 2],
        o if o & 0x07 == op::JMP => std::iter::once(pc + 1).chain(target(i.off)).collect(),
        _ => vec![pc + 1],
    }
}

/// removing unreachable instructions and retargeting jumps, `removeDead` in
/// Lean
///
/// Linux rejects unreachable instructions, and we leave some: code after a
/// `return`, the final pass of a body that always returns, and the rest of
/// a statement after an access past [`MAX_PACKET_OFFSET`].
fn remove_dead(code: &[Insn]) -> Vec<Insn> {
    let mut seen = vec![false; code.len()];
    let mut todo = vec![0];
    while let Some(pc) = todo.pop() {
        if pc < code.len() && !seen[pc] {
            seen[pc] = true;
            // the second slot of an `lddw` is kept with the first
            if code[pc].opcode == op::LDDW && pc + 1 < code.len() {
                seen[pc + 1] = true;
            }
            todo.extend(successors(code, pc).into_iter().rev());
        }
    }
    // the new position of each instruction, and of the end
    let mut index = vec![0i64; code.len() + 1];
    let mut n = 0;
    for pc in 0..code.len() {
        index[pc] = n;
        if seen[pc] {
            n += 1;
        }
    }
    index[code.len()] = n;
    // the second slot of an `lddw` is data, never a jump, whatever its bits
    let after_lddw = |pc: usize| pc > 0 && code[pc - 1].opcode == op::LDDW && seen[pc - 1];
    code.iter()
        .enumerate()
        .filter(|(pc, _)| seen[*pc])
        .map(|(pc, i)| {
            let mut i = *i;
            let is_jump = i.opcode & 0x07 == op::JMP && i.opcode != 0x95;
            if is_jump && !after_lddw(pc) {
                let t = (pc as i64 + 1 + i64::from(i.off)) as usize;
                i.off = (index[t] - index[pc] - 1) as i16;
            }
            i
        })
        .collect()
}

/// the deepest loop nesting in `s`, that is the counter slots it needs
fn loop_depth(s: &Stmt) -> usize {
    match s {
        Stmt::Seq { a, b } | Stmt::Ite { thn: a, els: b, .. } => loop_depth(a).max(loop_depth(b)),
        Stmt::Loop { body, .. } => loop_depth(body) + 1,
        _ => 0,
    }
}

/// [`compile_for`] with [`Target::Xdp`] when `kernel`, [`Target::Model`]
/// otherwise
pub fn compile(p: &Program, kernel: bool) -> Result<Vec<u8>, String> {
    compile_for(p, if kernel { Target::Xdp } else { Target::Model })
}

/// where the code runs. Targets differ only in the loads of the packet
/// bounds, in the verdicts and, for TC, in the code of `ctx.len()`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// the eBPF model of the Lean formalization: 64-bit bounds at offsets 0
    /// and 8, XDP verdicts. The interpreter of `vigil-driver` runs it
    Model,
    /// XDP in Linux: `struct xdp_md`, 32-bit bounds at offsets 0 and 4
    Xdp,
    /// TC in Linux: `struct __sk_buff`, 32-bit bounds at offsets 76 and 80,
    /// `TC_ACT_*` verdicts, and the length is `__sk_buff.len` at offset 0
    Tc,
}

impl Target {
    /// the load opcode for the packet bounds and the offsets of `data` and
    /// `data_end`
    fn bounds(self) -> (u8, i64, i64) {
        match self {
            Target::Model => (op::LDXDW, 0, 8),
            Target::Xdp => (op::LDXW, 0, 4),
            Target::Tc => (op::LDXW, 76, 80),
        }
    }

    /// the value returned for `action`
    ///
    /// Core has the XDP verdicts. For TC, an abort becomes `TC_ACT_OK` like
    /// a pass, because the language says an aborting program returns 0 for
    /// every kind (`ProgramKind::abort_return`), as the LLVM backend does.
    /// `TX` is an error because TC has no such verdict.
    pub fn verdict(self, action: Action) -> Result<i64, String> {
        match (self, action) {
            (Target::Model | Target::Xdp, a) => Ok(i64::from(a.value())),
            (Target::Tc, Action::Pass | Action::Aborted) => Ok(0),
            (Target::Tc, Action::Drop) => Ok(2),
            (Target::Tc, Action::Redirect) => Ok(7),
            (Target::Tc, Action::Tx) => Err("TC programs have no `TX` verdict".into()),
        }
    }
}

/// the encoded instructions of `p` for `target`
///
/// we generate the code once with XDP verdicts, then change the verdicts
/// for the target. So [`Target::Model`] gets the code of the Lean compiler,
/// and other targets get the same shape with a few loads and immediates
/// changed (TC also keeps the context in `r6` and reads its length there).
///
/// errors say what the backend can't compile: a too deep expression, a
/// variable packet offset, a loop of 2^31 iterations or more, too many
/// locals for the stack, or a program too long for 16-bit jumps.
pub fn compile_for(p: &Program, target: Target) -> Result<Vec<u8>, String> {
    let mut slots = Vec::new();
    let mut lens = Vec::new();
    let mut next = 0;
    for ty in &p.locals {
        slots.push(next);
        let n = match ty {
            Ty::Array { len, .. } => *len as usize,
            _ => 1,
        };
        lens.push(if matches!(ty, Ty::Array { .. }) { n } else { 0 });
        next += n;
    }
    let total = next + loop_depth(&p.body);
    // 64 slots of 8 bytes are the 512 bytes of the eBPF stack
    if total > 64 {
        return Err(format!(
            "{total} slots of 8 bytes: more than the 512 bytes of the stack"
        ));
    }
    let f = Frame {
        slots,
        lens,
        counters: next,
        target,
    };
    let (width, data, data_end) = target.bounds();
    let mut code = vec![
        load(width, Reg::R7, Reg::R1, data),
        load(width, Reg::R8, Reg::R1, data_end),
    ];
    // here we keep the context for `ctx.len()`, because `r1` is a temporary.
    // Only TC needs it, so the model gets the code of Lean
    if target == Target::Tc {
        code.push(alu_x(op::MOV, Reg::R6, Reg::R1));
    }
    // zeroing every slot, so none is read before written
    code.push(alu_k(op::MOV, Reg::R0, 0));
    code.extend((0..total).map(|k| store_slot(k, Reg::R0)));
    code.extend(stmt(&f, 0, &p.body)?);
    // a body that ends without `return` passes the packet
    code.extend(exit_with(Action::Pass));
    // jump offsets are signed 16 bits. We check before removing dead code,
    // which can only make it shorter
    if code.len() >= 0x7fff {
        return Err("a program too long for 16-bit jumps".into());
    }
    let mut code = remove_dead(&code);
    // rewriting the verdicts, the pairs `r0 = action; exit` emitted by
    // `exit_with`, the only place that emits `exit`
    for i in 0..code.len().saturating_sub(1) {
        if code[i + 1] != exit() {
            continue;
        }
        if let Some(&action) = Action::ALL
            .iter()
            .find(|a| code[i] == alu_k(op::MOV, Reg::R0, i64::from(a.value())))
        {
            code[i] = alu_k(op::MOV, Reg::R0, target.verdict(action)?);
        }
    }
    // encoding: opcode, src in the high and dst in the low half of the next
    // byte, then offset and immediate, little-endian
    Ok(code
        .iter()
        .flat_map(|i| {
            let mut b = [0u8; 8];
            b[0] = i.opcode;
            b[1] = (i.src << 4) | i.dst;
            b[2..4].copy_from_slice(&i.off.to_le_bytes());
            b[4..8].copy_from_slice(&i.imm.to_le_bytes());
            b
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the instructions of encoded code, as opcode and register byte
    fn insns(code: &[u8]) -> Vec<(u8, u8)> {
        code.chunks(8).map(|i| (i[0], i[1])).collect()
    }

    /// `mov r0, imm`, encoded
    fn mov_r0(imm: i32) -> Vec<u8> {
        let mut b = vec![0xb7, 0, 0, 0];
        b.extend(imm.to_le_bytes());
        b
    }

    // an abort returns 0 for every kind, so `TC_ACT_OK` for TC, as in the
    // LLVM backend
    #[test]
    fn a_tc_abort_returns_ok() {
        let p = Program {
            locals: vec![],
            body: Stmt::Ret {
                action: Action::Aborted,
            },
        };
        let code = compile_for(&p, Target::Tc).unwrap();
        assert!(code.chunks(8).any(|i| i == mov_r0(0)));
        assert!(!code.chunks(8).any(|i| i == mov_r0(2)));
        assert_eq!(Target::Tc.verdict(Action::Drop), Ok(2));
    }

    // for TC `ctx.len()` is `__sk_buff.len`, read through the context kept
    // in `r6`, and the other targets compute `data_end - data`
    #[test]
    fn tc_reads_the_length_in_the_context() {
        let p = Program {
            locals: vec![Ty::Int { t: IntTy::U32 }],
            body: Stmt::Assign { x: 0, e: Expr::Len },
        };
        let tc = insns(&compile_for(&p, Target::Tc).unwrap());
        // `mov r6, r1` and `r0 = *(u32 *)(r6 + 0)`
        assert!(tc.contains(&(0xbf, 0x16)));
        assert!(tc.contains(&(op::LDXW, 0x60)));
        for target in [Target::Model, Target::Xdp] {
            let code = insns(&compile_for(&p, target).unwrap());
            assert!(
                code.iter()
                    .all(|&(_, regs)| regs & 0x0f != 6 && regs >> 4 != 6)
            );
            // `r0 -= r7`
            assert!(code.contains(&(0x1f, 0x70)));
        }
    }
}
