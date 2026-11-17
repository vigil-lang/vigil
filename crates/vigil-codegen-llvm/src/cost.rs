//! this module bounds the work of the verifier: how many instructions it
//! processes for a program (`processed N insns` in its log), computed from
//! the instructions alone by [`estimate`].
//!
//! The verifier walks at least one path from the entry to an `exit`, so the
//! shortest path is a lower bound. With no cycles and no calls into other
//! code, walking every path without pruning gives an upper bound. Loops,
//! BPF-to-BPF calls and `bpf_loop` callbacks leave the upper bound unknown.

use std::collections::{HashMap, VecDeque};

use crate::disasm::{Insn, decode};

/// the number of the helper `bpf_loop`
const BPF_LOOP: i64 = 181;

/// how many instructions the verifier processes for a program
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    /// at least
    pub lower: u64,
    /// at most, when known
    pub upper: Option<u64>,
}

/// the bounds on the work of the verifier for the code of a program (see
/// the module header). An `lddw` is one node of the graph, as for the
/// verifier, and jumps go from slots to instructions.
pub fn estimate(code: &[u8]) -> Cost {
    let insns = decode(code);
    if insns.is_empty() {
        return Cost {
            lower: 0,
            upper: Some(0),
        };
    }
    let index: HashMap<usize, usize> = insns.iter().enumerate().map(|(i, n)| (n.slot, i)).collect();
    let mut calls_code = false;
    let successors: Vec<Vec<usize>> = insns
        .iter()
        .enumerate()
        .map(|(i, insn)| {
            calls_code |= calls_other_code(insn);
            let next = (i + 1 < insns.len()).then_some(i + 1);
            let target = || {
                let slot = insn.slot as i64 + 1 + i64::from(insn.offset);
                usize::try_from(slot)
                    .ok()
                    .and_then(|s| index.get(&s).copied())
            };
            let jump = matches!(insn.class(), 0x05 | 0x06);
            match (jump, insn.opcode & 0xf0) {
                (true, 0x90) => Vec::new(),
                (true, 0x80) => next.into_iter().collect(),
                (true, 0x00) if insn.class() == 0x05 => target().into_iter().collect(),
                (true, _) => next.into_iter().chain(target()).collect(),
                (false, _) => next.into_iter().collect(),
            }
        })
        .collect();
    Cost {
        lower: shortest(&successors),
        upper: if calls_code { None } else { paths(&successors) },
    }
}

/// checking if the verifier walks code outside the program at `insn`: a
/// BPF-to-BPF call, the address of a function (a callback), or a call of
/// `bpf_loop`
fn calls_other_code(insn: &Insn) -> bool {
    let call = insn.opcode == 0x85;
    (call && insn.src == 1)
        || (call && insn.src == 0 && insn.imm == BPF_LOOP)
        || (insn.is_lddw() && insn.src == 4)
}

/// the number of instructions on the shortest path from the first one to
/// an `exit`, or 0 if there is none. Breadth-first, because every
/// instruction weighs one.
fn shortest(successors: &[Vec<usize>]) -> u64 {
    let mut depth = vec![u64::MAX; successors.len()];
    let mut queue = VecDeque::from([0]);
    depth[0] = 1;
    while let Some(i) = queue.pop_front() {
        if successors[i].is_empty() {
            return depth[i];
        }
        for &s in &successors[i] {
            if depth[s] == u64::MAX {
                depth[s] = depth[i] + 1;
                queue.push_back(s);
            }
        }
    }
    0
}

/// the sum of the lengths of the paths from the first instruction, or
/// `None` when there is a cycle. It saturates at `u64::MAX`, because the
/// number of paths can be exponential.
fn paths(successors: &[Vec<usize>]) -> Option<u64> {
    // depth-first, with an explicit stack of instructions and the next
    // successor to visit. A successor still on the stack is a back edge, so
    // a cycle.
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        New,
        Open,
        Done,
    }
    let n = successors.len();
    let mut mark = vec![Mark::New; n];
    // for each instruction, the number of paths from it and their total
    // length, known once it is done
    let mut count = vec![0u64; n];
    let mut length = vec![0u64; n];
    let mut stack = vec![(0, 0)];
    mark[0] = Mark::Open;
    while let Some(&mut (i, ref mut next)) = stack.last_mut() {
        if let Some(&s) = successors[i].get(*next) {
            *next += 1;
            match mark[s] {
                Mark::Open => return None,
                Mark::New => {
                    mark[s] = Mark::Open;
                    stack.push((s, 0));
                }
                Mark::Done => {}
            }
            continue;
        }
        stack.pop();
        mark[i] = Mark::Done;
        if successors[i].is_empty() {
            count[i] = 1;
            length[i] = 1;
        } else {
            for &s in &successors[i] {
                count[i] = count[i].saturating_add(count[s]);
                length[i] = length[i].saturating_add(length[s]);
            }
            // this instruction is on every path from it, so it adds one to
            // the length of each
            length[i] = length[i].saturating_add(count[i]);
        }
    }
    Some(length[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code(insns: &[[u8; 8]]) -> Vec<u8> {
        insns.concat()
    }

    const EXIT: [u8; 8] = [0x95, 0, 0, 0, 0, 0, 0, 0];
    const MOV0: [u8; 8] = [0xb7, 0, 0, 0, 0, 0, 0, 0];

    #[test]
    fn counts_a_straight_line() {
        let cost = estimate(&code(&[MOV0, MOV0, EXIT]));
        assert_eq!(
            cost,
            Cost {
                lower: 3,
                upper: Some(3)
            }
        );
    }

    #[test]
    fn bounds_both_sides_of_a_branch() {
        // 0: if r1 == 0 goto +2; 1: r0 = 0; 2: r0 = 0; 3: exit
        let branch = [0x15, 0x01, 2, 0, 0, 0, 0, 0];
        let cost = estimate(&code(&[branch, MOV0, MOV0, EXIT]));
        // the short path is 0, 3 and the long one 0, 1, 2, 3
        assert_eq!(
            cost,
            Cost {
                lower: 2,
                upper: Some(6)
            }
        );
    }

    #[test]
    fn loops_have_no_upper_bound() {
        // 0: r0 = 0; 1: if r1 != 0 goto -2; 2: exit
        let back = [0x55, 0x01, 0xfe, 0xff, 0, 0, 0, 0];
        let cost = estimate(&code(&[MOV0, back, EXIT]));
        assert_eq!(
            cost,
            Cost {
                lower: 3,
                upper: None
            }
        );
    }

    #[test]
    fn callbacks_have_no_upper_bound() {
        let bpf_loop = [0x85, 0, 0, 0, 181, 0, 0, 0];
        let cost = estimate(&code(&[bpf_loop, EXIT]));
        assert_eq!(
            cost,
            Cost {
                lower: 2,
                upper: None
            }
        );
    }

    #[test]
    fn a_wide_load_is_one_instruction() {
        let lddw = [0x18, 0, 0, 0, 1, 0, 0, 0];
        let high = [0, 0, 0, 0, 0, 0, 0, 0];
        let cost = estimate(&code(&[lddw, high, EXIT]));
        assert_eq!(
            cost,
            Cost {
                lower: 2,
                upper: Some(2)
            }
        );
    }
}
