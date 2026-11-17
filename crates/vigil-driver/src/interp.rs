//! this module implements a small eBPF interpreter, enough to run the code
//! of the certified backend on a packet, also in the browser. [`run`] uses
//! the context of [`vigil_codegen_bpf::Target::Model`]: `r1` points to the
//! packet start and end, as 64-bit words at offsets 0 and 8. So the code
//! returns XDP verdicts, also for TC ([`verdict`] names them for the kind),
//! and `ctx.len()` is `data_end - data`, the packet length here because
//! our packets are linear.
//!
//! A fault is what the verifier would reject: a bad access, a store into
//! the context, a jump out of the program, falling off the end, an ALU
//! write to `r10` (by an ALU instruction, a load or `lddw`), an unknown instruction or a helper call. We also stop
//! after [`MAX_STEPS`]. Division by zero is not a fault, as in eBPF. A
//! fault message starts with the instruction index, from 0 as in
//! `--emit asm`.

use vigil_codegen_bpf::Target;
use vigil_core::Action;
use vigil_sema::program::ProgramKind;

// addresses of the memory regions. They are far apart, so an access that
// runs off one region cannot land in another

/// address of the context
const CTX: u64 = 0x1000_0000;
/// address of the first byte of the packet
const PACKET: u64 = 0x2000_0000;
/// address of the lowest byte of the stack
const STACK: u64 = 0x3000_0000;
/// stack size, the same as the kernel
const STACK_SIZE: u64 = 512;
/// the maximum number of instructions of a run, because code given to
/// [`run`] may not come from the certified backend and may not end
pub const MAX_STEPS: u64 = 1_000_000;

/// the result of a run that reached `exit`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// `r0` at `exit`
    pub value: u64,
    /// number of executed instructions
    pub steps: u64,
}

/// the memory of a run: the three regions a program can reach
struct Memory<'a> {
    ctx: [u8; 16],
    packet: &'a mut [u8],
    stack: [u8; STACK_SIZE as usize],
}

impl Memory<'_> {
    /// the `size` bytes at `addr`, or a fault if they are not inside one
    /// region. A store into the context is a fault too, because the
    /// verifier rejects it for XDP and for the `__sk_buff` fields the
    /// certified backend reads

    fn region(&mut self, addr: u64, size: u64, store: bool) -> Result<&mut [u8], String> {
        let within = |base: u64, len: u64| addr >= base && addr.wrapping_add(size) <= base + len;
        let access = if store { "store" } else { "load" };
        let a = if size == 8 { "an" } else { "a" };
        if within(CTX, 16) {
            let o = (addr - CTX) as usize;
            if store {
                return Err(format!(
                    "{a} {size}-byte store into the context at offset {o}, \
                     but the context is read-only (the verifier rejects the store)"
                ));
            }
            return Ok(&mut self.ctx[o..o + size as usize]);
        }
        if within(PACKET, self.packet.len() as u64) {
            let o = (addr - PACKET) as usize;
            return Ok(&mut self.packet[o..o + size as usize]);
        }
        if within(STACK, STACK_SIZE) {
            let o = (addr - STACK) as usize;
            return Ok(&mut self.stack[o..o + size as usize]);
        }
        Err(format!(
            "{a} {size}-byte {access} at {}",
            place(addr, self.packet.len() as u64)
        ))
    }

    /// loading `size` bytes at `addr`, little-endian, zero-extended
    fn load(&mut self, addr: u64, size: u64) -> Result<u64, String> {
        let bytes = self.region(addr, size, false)?;
        let mut word = [0u8; 8];
        word[..bytes.len()].copy_from_slice(bytes);
        Ok(u64::from_le_bytes(word))
    }

    /// storing the low `size` bytes of `value` at `addr`, little-endian
    fn store(&mut self, addr: u64, size: u64, value: u64) -> Result<(), String> {
        let bytes = self.region(addr, size, true)?;
        let n = bytes.len();
        bytes.copy_from_slice(&value.to_le_bytes()[..n]);
        Ok(())
    }
}

/// where `addr` is, for a fault message: an offset in the region it is
/// closest to, and why it is out of bounds. The stack offset is from `r10`,
/// as the code writes it
fn place(addr: u64, packet_len: u64) -> String {
    // half the distance between two regions
    let near = |base: u64| addr.wrapping_sub(base).wrapping_add(0x0800_0000) < 0x1000_0000;
    let offset = |base: u64| addr.wrapping_sub(base) as i64;
    if near(CTX) {
        format!(
            "context offset {}, outside the 16 bytes of the context",
            offset(CTX)
        )
    } else if near(PACKET) {
        format!(
            "packet offset {}, outside the {packet_len}-byte packet \
             (an access must be checked against `data_end` first)",
            offset(PACKET)
        )
    } else if near(STACK) {
        format!(
            "stack offset {} from r10, outside the {STACK_SIZE}-byte stack",
            offset(STACK + STACK_SIZE)
        )
    } else {
        format!("address {addr:#x}, which is not in the context, the packet or the stack")
    }
}

/// the access size in bytes of a load or store
fn size_of(opcode: u8) -> u64 {
    match opcode & 0x18 {
        0x00 => 4,
        0x08 => 2,
        0x10 => 1,
        _ => 8,
    }
}

/// an ALU operation on 64 bits, or 32 when `wide` is false. On 32 bits we
/// truncate the operands and zero-extend the result, as eBPF does. Shifts
/// are modulo the width, division by zero gives 0 and remainder by zero
/// gives the dividend.
fn alu(op: u8, dst: u64, src: u64, wide: bool) -> Result<u64, String> {
    let (bits, mask) = if wide {
        (64, u64::MAX)
    } else {
        (32, 0xFFFF_FFFF)
    };
    let (d, s) = (dst & mask, src & mask);
    let shift = (s & (bits - 1)) as u32;
    let r = match op {
        0x00 => d.wrapping_add(s),
        0x10 => d.wrapping_sub(s),
        0x20 => d.wrapping_mul(s),
        0x30 => d.checked_div(s).unwrap_or(0),
        0x40 => d | s,
        0x50 => d & s,
        0x60 => d << shift,
        0x70 => d >> shift,
        0x80 => d.wrapping_neg(),
        0x90 => d.checked_rem(s).unwrap_or(d),
        0xa0 => d ^ s,
        0xb0 => s,
        0xc0 => {
            if wide {
                ((d as i64) >> shift) as u64
            } else {
                ((d as u32 as i32) >> shift) as u32 as u64
            }
        }
        _ => return Err(format!("unknown ALU operation {op:#04x}")),
    };
    Ok(r & mask)
}

/// checking if a conditional jump is taken, on 64 bits, or 32 when `wide`
/// is false
fn condition(op: u8, d: u64, s: u64, wide: bool) -> Result<bool, String> {
    let (d, s) = if wide {
        (d, s)
    } else {
        (d & 0xFFFF_FFFF, s & 0xFFFF_FFFF)
    };
    let (sd, ss) = if wide {
        (d as i64, s as i64)
    } else {
        (d as u32 as i32 as i64, s as u32 as i32 as i64)
    };
    Ok(match op {
        0x10 => d == s,
        0x20 => d > s,
        0x30 => d >= s,
        0x40 => d & s != 0,
        0x50 => d != s,
        0x60 => sd > ss,
        0x70 => sd >= ss,
        0xa0 => d < s,
        0xb0 => d <= s,
        0xc0 => sd < ss,
        0xd0 => sd <= ss,
        _ => return Err(format!("unknown jump operation {op:#04x}")),
    })
}

/// running the encoded instructions of `code` on `packet`, which may be
/// modified. Registers start at zero, except `r1` (the context) and `r10`
/// (the top of the stack). The error describes the first fault.
pub fn run(code: &[u8], packet: &mut [u8]) -> Result<Outcome, String> {
    if code.len() % 8 != 0 {
        return Err(format!(
            "the code is {} bytes, not a whole number of 8-byte instructions",
            code.len()
        ));
    }
    let n = code.len() / 8;
    let insn = |pc: usize| {
        let b = &code[pc * 8..pc * 8 + 8];
        (
            b[0],
            b[1] & 0x0f,
            b[1] >> 4,
            i16::from_le_bytes([b[2], b[3]]),
            i32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        )
    };
    let len = packet.len() as u64;
    let mut mem = Memory {
        ctx: [0; 16],
        packet,
        stack: [0; STACK_SIZE as usize],
    };
    mem.ctx[..8].copy_from_slice(&PACKET.to_le_bytes());
    mem.ctx[8..].copy_from_slice(&(PACKET + len).to_le_bytes());
    let mut r = [0u64; 11];
    r[1] = CTX;
    r[10] = STACK + STACK_SIZE;
    let mut pc = 0usize;
    let mut steps = 0u64;
    loop {
        if pc >= n {
            return Err(format!(
                "ran past the last instruction ({}) without reaching `exit`",
                n.saturating_sub(1)
            ));
        }
        steps += 1;
        if steps > MAX_STEPS {
            return Err(format!(
                "stopped after {MAX_STEPS} instructions without reaching `exit`, \
                 at instruction {pc}; the program may loop forever"
            ));
        }
        // the faults below are of the instruction at `pc`
        let here = pc;
        let at = |e: String| format!("instruction {here}: {e}");
        let (opcode, dst, src, off, imm) = insn(pc);
        let (dst, src) = (dst as usize, src as usize);
        if dst > 10 || src > 10 {
            return Err(at(format!(
                "register r{} does not exist (eBPF has r0 to r10)",
                dst.max(src)
            )));
        }
        let class = opcode & 0x07;
        // every instruction that writes its destination: ALU, LDX and `lddw`.
        // r10 is read-only, so writing it is a fault whatever the source
        let writes_dst = matches!(class, 0x04 | 0x07 | 0x01) || opcode == 0x18;
        if writes_dst && dst == 10 {
            return Err(at(
                "writes r10, the frame pointer, which is read-only".into()
            ));
        }
        let operand = |r: &[u64; 11]| {
            if opcode & 0x08 != 0 {
                r[src]
            } else {
                imm as i64 as u64
            }
        };
        match class {
            // ALU (32 bits) and ALU64
            0x04 | 0x07 => {
                let wide = class == 0x07;
                let s = operand(&r);
                r[dst] = alu(opcode & 0xf0, r[dst], s, wide).map_err(at)?;
                pc += 1;
            }
            // JMP and JMP32. `ja` and `exit` exist only in JMP
            0x05 | 0x06 => {
                let op = opcode & 0xf0;
                match op {
                    0x00 if class == 0x05 => pc = jump(pc, off, n).map_err(at)?,
                    0x90 if class == 0x05 => {
                        return Ok(Outcome { value: r[0], steps });
                    }
                    0x80 => {
                        return Err(at(format!(
                            "calls helper {imm}, but the interpreter has no helpers \
                             (the certified backend never calls one)"
                        )));
                    }
                    _ => {
                        let s = operand(&r);
                        if condition(op, r[dst], s, class == 0x05).map_err(at)? {
                            pc = jump(pc, off, n).map_err(at)?;
                        } else {
                            pc += 1;
                        }
                    }
                }
            }
            // LDX
            0x01 => {
                let addr = r[src].wrapping_add(off as i64 as u64);
                r[dst] = mem.load(addr, size_of(opcode)).map_err(at)?;
                pc += 1;
            }
            // STX and ST, storing a register or an immediate
            0x03 | 0x02 => {
                let addr = r[dst].wrapping_add(off as i64 as u64);
                let value = if class == 0x03 {
                    r[src]
                } else {
                    imm as i64 as u64
                };
                mem.store(addr, size_of(opcode), value).map_err(at)?;
                pc += 1;
            }
            // LD: only `lddw`, which takes two slots, the second with the
            // high half of the constant
            0x00 if opcode == 0x18 => {
                if pc + 1 >= n {
                    return Err(at("`lddw` takes two slots, but it is the last one".into()));
                }
                let (_, _, _, _, high) = insn(pc + 1);
                r[dst] = (imm as u32 as u64) | ((high as u32 as u64) << 32);
                pc += 2;
            }
            _ => {
                return Err(at(format!("unknown or unsupported opcode {opcode:#04x}")));
            }
        }
    }
}

/// the target of a jump by `off` at `pc`, if it is inside the program of
/// `n` instructions. Offsets count from the next instruction.
fn jump(pc: usize, off: i16, n: usize) -> Result<usize, String> {
    let target = pc as i64 + 1 + off as i64;
    if target < 0 || target as usize >= n {
        return Err(format!(
            "jumps to {target}, outside the program of {}",
            vigil_diagnostics::plural(n, "instruction")
        ));
    }
    Ok(target as usize)
}

/// the name of an XDP verdict, or `unknown`
pub fn xdp_verdict(value: u64) -> &'static str {
    match value {
        0 => "XDP_ABORTED",
        1 => "XDP_DROP",
        2 => "XDP_PASS",
        3 => "XDP_TX",
        4 => "XDP_REDIRECT",
        _ => "unknown",
    }
}

/// the value and name of the verdict a program of `kind` returns, given the
/// XDP verdict `value` of the model code. For TC we map it as the certified
/// backend does, so an abort is `TC_ACT_OK`, as in the kernel
pub fn verdict(kind: ProgramKind, value: u64) -> (u64, &'static str) {
    if kind != ProgramKind::Tc {
        return (value, xdp_verdict(value));
    }
    let tc = usize::try_from(value)
        .ok()
        .and_then(|v| Action::ALL.get(v))
        .and_then(|&a| Target::Tc.verdict(a).ok());
    match tc {
        Some(0) => (0, "TC_ACT_OK"),
        Some(2) => (2, "TC_ACT_SHOT"),
        Some(7) => (7, "TC_ACT_REDIRECT"),
        _ => (value, "unknown"),
    }
}

/// the accepted packet format, for error messages
pub const HEX_FORMAT: &str = "the packet is bytes in hexadecimal, two digits each, as `aabbcc` \
     or `aa bb cc`; spaces, newlines, `:` and `,` may separate the digits, and a word may \
     start with `0x`";

/// parsing a packet in hex: `aa bb cc`, `0xaabb`, with spaces, newlines,
/// `:` or `,` between bytes. We join the words into one run of digits, so
/// `a abb` is the two bytes `aa bb`. An error names the first bad
/// character and its position, counting characters from 1.
pub fn parse_hex(text: &str) -> Result<Vec<u8>, String> {
    let mut digits = Vec::new();
    let mut chars = text.chars().enumerate().peekable();
    let mut word_start = true;
    while let Some((i, c)) = chars.next() {
        if c.is_whitespace() || c == ':' || c == ',' {
            word_start = true;
            continue;
        }
        // here we skip the `0x` at the start of a word
        if word_start && c == '0' && chars.peek().is_some_and(|&(_, x)| x == 'x') {
            chars.next();
            word_start = false;
            continue;
        }
        word_start = false;
        match c.to_digit(16) {
            Some(d) => digits.push(d as u8),
            None => {
                return Err(format!(
                    "`{}` at position {} is not a hexadecimal digit",
                    c.escape_default(),
                    i + 1
                ));
            }
        }
    }
    if digits.len() % 2 != 0 {
        return Err(format!(
            "{} hexadecimal digits, an odd number: a byte is two digits",
            digits.len()
        ));
    }
    Ok(digits.chunks(2).map(|d| d[0] << 4 | d[1]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// encoding one instruction
    fn enc(opcode: u8, dst: u8, src: u8, off: i16, imm: i32) -> Vec<u8> {
        let mut v = vec![opcode, dst | (src << 4)];
        v.extend(off.to_le_bytes());
        v.extend(imm.to_le_bytes());
        v
    }

    #[test]
    fn reads_the_packet_through_the_context() {
        // r2 = *(u64 *)(r1 + 0); r0 = *(u8 *)(r2 + 1); exit
        let code = [
            enc(0x79, 2, 1, 0, 0),
            enc(0x71, 0, 2, 1, 0),
            enc(0x95, 0, 0, 0, 0),
        ]
        .concat();
        let mut packet = vec![7, 42];
        assert_eq!(run(&code, &mut packet).unwrap().value, 42);
    }

    // a read at offset 5 of a packet of two bytes
    #[test]
    fn rejects_reads_past_the_packet() {
        let code = [
            enc(0x79, 2, 1, 0, 0),
            enc(0x71, 0, 2, 5, 0),
            enc(0x95, 0, 0, 0, 0),
        ]
        .concat();
        assert!(run(&code, &mut [1, 2]).is_err());
    }

    // the model code returns XDP verdicts: an abort (0) is `TC_ACT_OK` and a
    // drop (1) is `TC_ACT_SHOT` for TC
    #[test]
    fn names_verdicts_by_kind() {
        assert_eq!(verdict(ProgramKind::Xdp, 1), (1, "XDP_DROP"));
        assert_eq!(verdict(ProgramKind::Tc, 0), (0, "TC_ACT_OK"));
        assert_eq!(verdict(ProgramKind::Tc, 1), (2, "TC_ACT_SHOT"));
        assert_eq!(verdict(ProgramKind::Tc, 2), (0, "TC_ACT_OK"));
        assert_eq!(verdict(ProgramKind::Tc, 4), (7, "TC_ACT_REDIRECT"));
    }

    #[test]
    fn parses_hex() {
        assert_eq!(
            parse_hex("0xaa bb:cc,dd").unwrap(),
            vec![0xaa, 0xbb, 0xcc, 0xdd]
        );
        assert_eq!(parse_hex("a abb").unwrap(), vec![0xaa, 0xbb]);
        assert_eq!(parse_hex("").unwrap(), Vec::<u8>::new());
    }

    // the error names the bad character and its position, from 1
    #[test]
    fn locates_bad_hex() {
        assert_eq!(
            parse_hex("aa bg").unwrap_err(),
            "`g` at position 5 is not a hexadecimal digit"
        );
        assert_eq!(
            parse_hex("abc").unwrap_err(),
            "3 hexadecimal digits, an odd number: a byte is two digits"
        );
        // `x` only starts a `0x` prefix at the start of a word
        assert!(parse_hex("a0xb").is_err());
    }

    // the verifier rejects stores into the XDP and TC context, so here
    // both `stx` and `st` into it fault, at any offset
    #[test]
    fn rejects_stores_into_the_context() {
        // *(u64 *)(r1 + 0) = r1; exit
        let stx = [enc(0x7b, 1, 1, 0, 0), enc(0x95, 0, 0, 0, 0)].concat();
        let err = run(&stx, &mut [1, 2]).unwrap_err();
        assert!(
            err.starts_with("instruction 0: an 8-byte store into the context at offset 0"),
            "{err}"
        );
        // r0 = 0; *(u32 *)(r1 + 8) = 7; exit
        let st = [
            enc(0xb7, 0, 0, 0, 0),
            enc(0x62, 1, 0, 8, 7),
            enc(0x95, 0, 0, 0, 0),
        ]
        .concat();
        let err = run(&st, &mut [1, 2]).unwrap_err();
        assert!(
            err.starts_with("instruction 1: a 4-byte store into the context at offset 8"),
            "{err}"
        );
        // a store into the stack is fine: *(u64 *)(r10 - 8) = r1; exit
        let stack = [enc(0x7b, 10, 1, -8, 0), enc(0x95, 0, 0, 0, 0)].concat();
        assert!(run(&stack, &mut [1, 2]).is_ok());
    }

    // r10 is read-only whatever writes it: an ALU instruction, a load or
    // `lddw`; reading it (as the base of a stack access) is fine
    #[test]
    fn rejects_writes_to_the_frame_pointer() {
        let fault = "writes r10, the frame pointer, which is read-only";
        // r10 = 0
        let alu = [enc(0xb7, 10, 0, 0, 0), enc(0x95, 0, 0, 0, 0)].concat();
        // r10 = *(u64 *)(r1 + 0)
        let ldx = [enc(0x79, 10, 1, 0, 0), enc(0x95, 0, 0, 0, 0)].concat();
        // r10 = 0x1 ll
        let lddw = [
            enc(0x18, 10, 0, 0, 1),
            enc(0x00, 0, 0, 0, 0),
            enc(0x95, 0, 0, 0, 0),
        ]
        .concat();
        for code in [alu, ldx, lddw] {
            let err = run(&code, &mut [1, 2]).unwrap_err();
            assert_eq!(err, format!("instruction 0: {fault}"));
        }
        // r0 = *(u64 *)(r10 - 8) after storing there: reads r10, no fault
        let read = [
            enc(0x7b, 10, 1, -8, 0),
            enc(0x79, 0, 10, -8, 0),
            enc(0x95, 0, 0, 0, 0),
        ]
        .concat();
        assert!(run(&read, &mut [1, 2]).is_ok());
    }

    // faults name the instruction and where the access went
    #[test]
    fn describes_faults() {
        let past = [
            enc(0x79, 2, 1, 0, 0),
            enc(0x71, 0, 2, 5, 0),
            enc(0x95, 0, 0, 0, 0),
        ]
        .concat();
        assert_eq!(
            run(&past, &mut [1, 2]).unwrap_err(),
            "instruction 1: a 1-byte load at packet offset 5, outside the 2-byte packet \
             (an access must be checked against `data_end` first)"
        );
        let below = [enc(0x79, 0, 10, -520, 0), enc(0x95, 0, 0, 0, 0)].concat();
        assert_eq!(
            run(&below, &mut []).unwrap_err(),
            "instruction 0: an 8-byte load at stack offset -520 from r10, outside the 512-byte stack"
        );
        let jump = [enc(0x05, 0, 0, 3, 0), enc(0x95, 0, 0, 0, 0)].concat();
        assert_eq!(
            run(&jump, &mut []).unwrap_err(),
            "instruction 0: jumps to 4, outside the program of 2 instructions"
        );
        let end = enc(0xb7, 0, 0, 0, 0);
        assert_eq!(
            run(&end, &mut []).unwrap_err(),
            "ran past the last instruction (0) without reaching `exit`"
        );
    }
}
