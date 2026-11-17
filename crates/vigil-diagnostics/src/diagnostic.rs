//! this module defines diagnostics and the collection a phase reports
//! them in. A [`Diagnostic`] is built in one expression, with
//! [`Diagnostic::error`] (or a sibling) followed by the `with_*` methods.

use crate::source::Span;

/// how bad a diagnostic is. Only [`Severity::Error`] counts for
/// [`Diagnostics::has_errors`]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

/// a message attached to a span of the source
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    pub span: Span,
    /// what to say at the span. If empty, the span is just underlined
    pub message: String,
    /// true when the label marks where the problem is, false when it only
    /// gives context. See [`Diagnostic::anchor`]
    pub primary: bool,
}

/// a problem found in a program or a project: what is wrong, where, and
/// how to fix it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    /// one of the codes of [`crate::codes`]
    pub code: Option<&'static str>,
    /// the headline of the report
    pub message: String,
    pub labels: Vec<Label>,
    /// facts explaining the problem, each shown as a `Note:` line
    pub notes: Vec<String>,
    /// ways to fix it, each shown as a `Help:` line
    pub help: Vec<String>,
}

impl Diagnostic {
    /// a diagnostic with a message and nothing else
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Diagnostic {
            severity,
            code: None,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
            help: Vec::new(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }

    pub fn note(message: impl Into<String>) -> Self {
        Self::new(Severity::Note, message)
    }

    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    /// adding a label on where the problem is
    pub fn with_primary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            message: message.into(),
            primary: true,
        });
        self
    }

    /// adding a label that gives context
    pub fn with_secondary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            message: message.into(),
            primary: false,
        });
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help.push(help.into());
        self
    }

    /// the span the report is anchored at: the first primary label, or else
    /// the first label. `None` when there are no labels, and then no source
    /// excerpt is rendered
    pub fn anchor(&self) -> Option<Span> {
        self.labels
            .iter()
            .find(|l| l.primary)
            .or_else(|| self.labels.first())
            .map(|l| l.span)
    }

    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

/// the diagnostics of a phase, in the order they were reported
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Diagnostics {
    items: Vec<Diagnostic>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.items.push(diagnostic);
    }

    /// appending the diagnostics of another phase after these
    pub fn extend(&mut self, other: Diagnostics) {
        self.items.extend(other.items);
    }

    pub fn has_errors(&self) -> bool {
        self.items.iter().any(Diagnostic::is_error)
    }

    pub fn error_count(&self) -> usize {
        self.count(Severity::Error)
    }

    pub fn warning_count(&self) -> usize {
        self.count(Severity::Warning)
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Diagnostic> {
        self.items.iter()
    }

    /// sorting by file and position. The sort is stable, so diagnostics at
    /// the same place keep their order. Those without a location come first
    pub fn sort_by_location(&mut self) {
        self.items
            .sort_by_key(|d| d.anchor().map(|s| (s.file, s.start)));
    }

    /// removing repeated diagnostics. Only adjacent ones are removed, so
    /// call [`Self::sort_by_location`] first
    pub fn dedup(&mut self) {
        self.items.dedup();
    }

    fn count(&self, severity: Severity) -> usize {
        self.items.iter().filter(|d| d.severity == severity).count()
    }
}

impl From<Diagnostic> for Diagnostics {
    fn from(diagnostic: Diagnostic) -> Self {
        Diagnostics {
            items: vec![diagnostic],
        }
    }
}

impl IntoIterator for Diagnostics {
    type Item = Diagnostic;
    type IntoIter = std::vec::IntoIter<Diagnostic>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}
