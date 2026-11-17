//! this crate defines source files, spans and diagnostics, shared by
//! every phase of the compiler. A phase collects [`Diagnostic`]s in
//! [`Diagnostics`], the text lives in a [`SourceMap`] and a [`Renderer`]
//! prints the reports.

pub mod codes;
mod diagnostic;
mod render;
mod source;

pub use diagnostic::{Diagnostic, Diagnostics, Label, Severity};
pub use render::Renderer;
/// where to report a bug of the compiler, named by the messages of internal
/// errors
pub const ISSUES: &str = "https://github.com/vigil-lang/vigil/issues";

pub use source::{FileId, SourceFile, SourceMap, Span};

/// a number and a noun, in the plural unless the number is 1, for messages:
/// `plural(1, "value")` is `1 value`, `plural(3, "value")` is `3 values`
pub fn plural<N: std::fmt::Display + PartialEq + From<u8>>(n: N, noun: &str) -> String {
    if n == N::from(1) {
        format!("{n} {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

#[cfg(test)]
mod tests {
    use super::plural;

    #[test]
    fn plural_agrees_with_the_number() {
        assert_eq!(plural(1usize, "value"), "1 value");
        assert_eq!(plural(0u32, "value"), "0 values");
        assert_eq!(plural(2u64, "target"), "2 targets");
    }
}
