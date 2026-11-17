//! this module implements a small eBPF disassembler, used by
//! `--emit asm`, the statistics of programs, and tests that check
//! instruction patterns. The listing is in the style of the verifier log.

use std::fmt::Write;

/// a decoded instruction. An `lddw` takes two slots and is one instruction
/// with the 64-bit immediate of both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Insn {
    pub opcode: u8,
    /// destination register
    pub dst: u8,
    /// source register
    pub src: u8,
    pub offset: i16,
    /// the immediate, sign-extended; all 64 bits for an `lddw`
    pub imm: i64,
    /// the slot of the instruction, in 8-byte units
    pub slot: usize,
}

impl Insn {
    /// the class of the instruction, its three low bits
    pub fn class(&self) -> u8 {
        self.opcode & 0x07
    }

    /// checking if this is an `lddw`, the load of a 64-bit immediate
    pub fn is_lddw(&self) -> bool {
        self.opcode == 0x18
    }

    /// the (negative) offset at which the instruction accesses the stack
    /// through `r10`, if it does
    pub fn stack_offset(&self) -> Option<i16> {
        let memory = matches!(self.class(), 0x01..=0x03);
        let base = if self.class() == 0x01 {
            self.src
        } else {
            self.dst
        };
        (memory && base == 10).then_some(self.offset)
    }
}

/// decoding the instructions of a program section. Trailing bytes short of
/// a slot are ignored.
pub fn decode(code: &[u8]) -> Vec<Insn> {
    let mut insns = Vec::new();
    let mut slot = 0;
    while slot * 8 + 8 <= code.len() {
        let bytes = &code[slot * 8..slot * 8 + 8];
        let mut insn = Insn {
            opcode: bytes[0],
            dst: bytes[1] & 0x0f,
            src: bytes[1] >> 4,
            offset: i16::from_le_bytes([bytes[2], bytes[3]]),
            imm: i64::from(i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])),
            slot,
        };
        if insn.is_lddw() && slot * 8 + 16 <= code.len() {
            let high = i32::from_le_bytes(code[slot * 8 + 12..slot * 8 + 16].try_into().unwrap());
            insn.imm = (insn.imm & 0xffff_ffff) | (i64::from(high) << 32);
            slot += 1;
        }
        insns.push(insn);
        slot += 1;
    }
    insns
}

/// the text of the operation of an arithmetic instruction
fn alu_op(op: u8) -> &'static str {
    match op >> 4 {
        0x0 => "+=",
        0x1 => "-=",
        0x2 => "*=",
        0x3 => "/=",
        0x4 => "|=",
        0x5 => "&=",
        0x6 => "<<=",
        0x7 => ">>=",
        0x8 => "neg",
        0x9 => "%=",
        0xa => "^=",
        0xb => "=",
        0xc => "s>>=",
        0xd => "bswap",
        _ => "?",
    }
}

/// the text of the comparison of a conditional jump
fn jmp_op(op: u8) -> &'static str {
    match op >> 4 {
        0x1 => "==",
        0x2 => ">",
        0x3 => ">=",
        0x4 => "&",
        0x5 => "!=",
        0x6 => "s>",
        0x7 => "s>=",
        0xa => "<",
        0xb => "<=",
        0xc => "s<",
        0xd => "s<=",
        _ => "?",
    }
}

/// the type of the memory a load or store accesses
fn size(opcode: u8) -> &'static str {
    match opcode & 0x18 {
        0x00 => "u32",
        0x08 => "u16",
        0x10 => "u8",
        _ => "u64",
    }
}

/// an instruction in the verifier style: `r1 = *(u32 *)(r1 +4)`
pub fn render(insn: &Insn) -> String {
    let (d, s, off, imm) = (insn.dst, insn.src, insn.offset, insn.imm);
    let source = |register: bool| {
        if register {
            format!("r{s}")
        } else {
            imm.to_string()
        }
    };
    match insn.class() {
        0x04 | 0x07 => {
            let wide = insn.class() == 0x07;
            let r = if wide { "r" } else { "w" };
            let op = alu_op(insn.opcode);
            let register = insn.opcode & 0x08 != 0;
            match op {
                "neg" => format!("{r}{d} = -{r}{d}"),
                "bswap" => format!("{r}{d} = bswap{imm}({r}{d})"),
                _ => {
                    let src = if register {
                        format!("{r}{s}")
                    } else {
                        imm.to_string()
                    };
                    format!("{r}{d} {op} {src}")
                }
            }
        }
        0x05 | 0x06 => {
            let r = if insn.class() == 0x06 { "w" } else { "r" };
            match insn.opcode >> 4 {
                0x0 => format!("goto {:+}", off),
                0x8 => format!("call {imm}"),
                0x9 => "exit".into(),
                _ => {
                    let register = insn.opcode & 0x08 != 0;
                    let src = if register {
                        format!("{r}{s}")
                    } else {
                        source(false)
                    };
                    format!("if {r}{d} {} {src} goto {:+}", jmp_op(insn.opcode), off)
                }
            }
        }
        0x00 if insn.is_lddw() => {
            if s == 1 {
                format!("r{d} = map[{imm}]")
            } else {
                format!("r{d} = {imm} ll")
            }
        }
        0x01 => format!("r{d} = *({} *)(r{s} {off:+})", size(insn.opcode)),
        0x02 => format!("*({} *)(r{d} {off:+}) = {imm}", size(insn.opcode)),
        0x03 if insn.opcode & 0xe0 == 0xc0 => {
            format!("lock *({} *)(r{d} {off:+}) += r{s}", size(insn.opcode))
        }
        0x03 => format!("*({} *)(r{d} {off:+}) = r{s}", size(insn.opcode)),
        _ => format!("unknown opcode {:#04x}", insn.opcode),
    }
}

/// the listing of a program: one instruction per line, after its slot
pub fn listing(code: &[u8]) -> String {
    let mut out = String::new();
    for insn in decode(code) {
        let _ = writeln!(out, "{:4}: {}", insn.slot, render(&insn));
    }
    out
}

/// size and stack use of a program
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stats {
    /// instructions, an `lddw` counting as two, as for the verifier limit
    pub insns: usize,
    /// bytes of stack below `r10` used by the program
    pub stack: usize,
}

/// size and stack use of the code of a program. The stack is the deepest
/// access through `r10` itself, so an access through a copy of `r10` is not
/// seen.
pub fn stats(code: &[u8]) -> Stats {
    let insns = decode(code);
    let stack = insns
        .iter()
        .filter_map(Insn::stack_offset)
        .map(|off| usize::from(off.unsigned_abs()))
        .max()
        .unwrap_or(0);
    Stats {
        insns: code.len() / 8,
        stack,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_common_instructions() {
        // r1 = *(u32 *)(r1 + 4); if r2 > r1 goto +3; call 1; exit
        let code = [
            0x61, 0x11, 0x04, 0x00, 0, 0, 0, 0, //
            0x2d, 0x12, 0x03, 0x00, 0, 0, 0, 0, //
            0x85, 0x00, 0x00, 0x00, 1, 0, 0, 0, //
            0x95, 0x00, 0x00, 0x00, 0, 0, 0, 0,
        ];
        assert_eq!(
            listing(&code),
            "   0: r1 = *(u32 *)(r1 +4)\n   1: if r2 > r1 goto +3\n   2: call 1\n   3: exit\n"
        );
    }

    #[test]
    fn measures_stack_use() {
        // *(u64 *)(r10 - 16) = r1; exit
        let code = [
            0x7b, 0x1a, 0xf0, 0xff, 0, 0, 0, 0, 0x95, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(
            stats(&code),
            Stats {
                insns: 2,
                stack: 16
            }
        );
    }
}
