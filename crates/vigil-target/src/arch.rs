//! this module defines the target architectures. The only part of the
//! kernel ABI that changes between them is the layout of `struct pt_regs`,
//! where kprobes and uprobes find the arguments and the return value.

use std::fmt;

/// an architecture we generate code for
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {
    X86_64,
    Aarch64,
}

impl Arch {
    pub const ALL: [Arch; 2] = [Arch::X86_64, Arch::Aarch64];

    /// the host architecture, the default target. Any host that is not
    /// aarch64 counts as x86-64.
    pub const fn host() -> Arch {
        if cfg!(target_arch = "aarch64") {
            Arch::Aarch64
        } else {
            Arch::X86_64
        }
    }

    /// the canonical name, used by [`Arch::parse`] and diagnostics
    pub const fn name(self) -> &'static str {
        match self {
            Arch::X86_64 => "x86_64",
            Arch::Aarch64 => "aarch64",
        }
    }

    /// parsing the canonical name or a common alias (`x86-64`, `amd64`,
    /// `arm64`)
    pub fn parse(name: &str) -> Option<Arch> {
        match name {
            "x86_64" | "x86-64" | "amd64" => Some(Arch::X86_64),
            "aarch64" | "arm64" => Some(Arch::Aarch64),
            _ => None,
        }
    }

    /// offsets in `struct pt_regs` of the argument registers, in order:
    /// `rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9` on x86-64, `x0` to `x7` on
    /// arm64. The length is the number of arguments a probe can read.
    pub const fn arg_offsets(self) -> &'static [u32] {
        match self {
            Arch::X86_64 => &[112, 104, 96, 88, 72, 64],
            Arch::Aarch64 => &[0, 8, 16, 24, 32, 40, 48, 56],
        }
    }

    /// offset in `struct pt_regs` of the return value: `rax` on x86-64,
    /// `x0` on arm64
    pub const fn ret_offset(self) -> u32 {
        match self {
            Arch::X86_64 => 80,
            Arch::Aarch64 => 0,
        }
    }
}

impl Default for Arch {
    fn default() -> Self {
        Arch::host()
    }
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
