//! this module defines every diagnostic code, so that a number is never
//! used twice. `E` is an error, `W` a warning, and the hundreds give the
//! phase. Codes of the full Vigil stay here even when unused, to keep their
//! numbers taken.

// project and manifest (`vigil.toml`)
pub const INVALID_MANIFEST: &str = "E0001";
pub const INVALID_KERNEL_VERSION: &str = "E0002";
pub const UNSUPPORTED_KERNEL: &str = "E0003";
pub const EMPTY_KERNEL_LIST: &str = "E0004";
pub const INVALID_PACKAGE_NAME: &str = "E0005";
pub const PROJECT_NOT_FOUND: &str = "E0006";
pub const IO_ERROR: &str = "E0007";
/// a malformed `[dependencies]` entry, or one naming a library that can't
/// be used (a missing directory, two libraries with one name)
pub const INVALID_DEPENDENCY: &str = "E0008";
/// a git dependency that could not be fetched or resolved to a commit
pub const DEPENDENCY_UNAVAILABLE: &str = "E0009";
/// options of a command that do not go together (`--certified` with
/// `--emit ir`), or a `--program` that is missing or names no program, in
/// `vigil core` and `vigil simulate`
pub const INVALID_OPTIONS: &str = "E0010";
/// a `--packet` that is not bytes in hexadecimal
pub const INVALID_PACKET: &str = "E0011";

// syntax
pub const SYNTAX_ERROR: &str = "E0200";
pub const UNEXPECTED_CHARACTER: &str = "E0201";
pub const TAB_INDENTATION: &str = "E0202";
pub const INCONSISTENT_INDENTATION: &str = "E0203";

// names and modules
pub const UNRESOLVED_NAME: &str = "E0300";
pub const UNRESOLVED_MODULE: &str = "E0301";
pub const DUPLICATE_DEFINITION: &str = "E0302";
pub const PRIVATE_ITEM: &str = "E0303";
pub const CYCLE: &str = "E0304";
pub const INFINITE_TYPE: &str = "E0305";
pub const NOT_SUPPORTED_YET: &str = "E0390";

// types
pub const MISMATCHED_TYPES: &str = "E0400";
pub const UNKNOWN_FIELD: &str = "E0401";
pub const WRONG_KIND: &str = "E0402";
pub const BAD_ARGUMENTS: &str = "E0403";
pub const BAD_OPERAND: &str = "E0404";
pub const CANNOT_INFER: &str = "E0405";
pub const LITERAL_OUT_OF_RANGE: &str = "E0406";
pub const BAD_ASSIGNMENT: &str = "E0407";
pub const BAD_TRY: &str = "E0408";
pub const MISSING_RETURN: &str = "E0409";
pub const NON_EXHAUSTIVE: &str = "E0410";
pub const OUTSIDE_LOOP: &str = "E0411";
pub const BAD_DECORATOR: &str = "E0412";
pub const BAD_PATTERN: &str = "E0413";
pub const CONST_EVAL: &str = "E0414";
pub const RECURSION: &str = "E0415";
pub const STRING_OUTSIDE_PRINT: &str = "E0416";
pub const TEST_ONLY: &str = "E0417";

// code generation and the kernel. `VERIFIER_REJECTED`, `KERNEL` and
// `TEST_FAILED` are unused here (no kernel loading, no tests)
pub const TOOLCHAIN: &str = "E0500";
pub const CODEGEN_LIMIT: &str = "E0501";
pub const VERIFIER_REJECTED: &str = "E0510";
pub const KERNEL: &str = "E0511";
/// a fault of a program run on a packet by the eBPF interpreter of
/// `vigil simulate`: what the verifier would reject, or too many steps
pub const SIMULATION_FAULT: &str = "E0512";
pub const TEST_FAILED: &str = "E0520";

// contracts. Only the first two are used here, the rest belong to
// the verifier
pub const BAD_CONTRACT: &str = "E0600";
pub const IMPURE_CONTRACT: &str = "E0601";
pub const POSTCONDITION_FAILS: &str = "E0610";
pub const PRECONDITION_FAILS: &str = "E0611";
pub const MAY_ABORT: &str = "E0612";
pub const INVARIANT_ENTRY: &str = "E0613";
pub const INVARIANT_PRESERVED: &str = "E0614";
pub const ILL_DEFINED_CONTRACT: &str = "E0615";
pub const VERIFIER_INTERNAL: &str = "E0620";
pub const LEAN_CHECK_FAILED: &str = "E0621";

// driver, unused here
pub const NOT_IMPLEMENTED: &str = "E0100";

// warnings. `COST_ESTIMATE` and the `W06xx` ones (verification) are
// unused here
pub const NO_SOURCES: &str = "W0001";
pub const DUPLICATE_KERNEL: &str = "W0002";
/// something about dependencies that is ignored: a `std` dependency of a
/// library (only the project being built chooses `std`), or an invalid
/// `vigil.lock`
pub const IGNORED_DEPENDENCY: &str = "W0003";
pub const UNREACHABLE_PATTERN: &str = "W0100";
pub const COST_ESTIMATE: &str = "W0500";
pub const NOT_VERIFIED: &str = "W0600";
pub const VERIFY_UNKNOWN: &str = "W0601";
pub const VACUOUS_REQUIRES: &str = "W0602";
pub const LEAN_TIMEOUT: &str = "W0603";
