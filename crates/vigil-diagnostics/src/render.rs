//! this module prints diagnostics for the user. A diagnostic with a location
//! is drawn by `ariadne` with source excerpts, one without (an I/O error, a
//! missing file) gets only the header line, the help and the notes.

use std::io::{self, Write};

use ariadne::{Color, Config, IndexType, Report, ReportKind};

use crate::diagnostic::{Diagnostic, Diagnostics, Severity};
use crate::source::SourceMap;

/// printing diagnostics against the files of a [`SourceMap`], with colour
/// or not
pub struct Renderer<'a> {
    sources: &'a SourceMap,
    color: bool,
}

impl<'a> Renderer<'a> {
    pub fn new(sources: &'a SourceMap, color: bool) -> Self {
        Renderer { sources, color }
    }

    /// writing one diagnostic
    pub fn render(&self, diagnostic: &Diagnostic, out: &mut dyn Write) -> io::Result<()> {
        match diagnostic.anchor() {
            Some(anchor) => self.render_with_source(diagnostic, anchor, out),
            None => self.render_plain(diagnostic, out),
        }
    }

    /// writing every diagnostic, in the order of the collection
    pub fn render_all(&self, diagnostics: &Diagnostics, out: &mut dyn Write) -> io::Result<()> {
        for diagnostic in diagnostics.iter() {
            self.render(diagnostic, out)?;
        }
        Ok(())
    }

    /// a report with source excerpts. We give ariadne byte offsets, as in
    /// [`crate::Span`], not characters (its default)
    fn render_with_source(
        &self,
        diagnostic: &Diagnostic,
        anchor: crate::Span,
        out: &mut dyn Write,
    ) -> io::Result<()> {
        let name = |file| self.sources.get(file).name().to_string();
        let config = Config::default()
            .with_color(self.color)
            .with_index_type(IndexType::Byte);

        let mut report = Report::build(
            kind(diagnostic.severity),
            (name(anchor.file), anchor.range()),
        )
        .with_config(config)
        .with_message(&diagnostic.message);
        if let Some(code) = diagnostic.code {
            report = report.with_code(code);
        }
        // here we sort the labels by position, because ariadne starts a new
        // excerpt when a label is on an earlier line than the previous one
        let mut labels: Vec<_> = diagnostic.labels.iter().collect();
        labels.sort_by_key(|l| (l.span.file, l.span.start));
        for label in labels {
            let color = if label.primary {
                severity_color(diagnostic.severity)
            } else {
                Color::Blue
            };
            let mut ariadne_label =
                ariadne::Label::new((name(label.span.file), label.span.range())).with_color(color);
            if !label.message.is_empty() {
                ariadne_label = ariadne_label.with_message(&label.message);
            }
            report = report.with_label(ariadne_label);
        }
        for note in &diagnostic.notes {
            report = report.with_note(note);
        }
        for help in &diagnostic.help {
            report = report.with_help(help);
        }

        // ariadne only gets the files the labels point into, each once
        let mut files: Vec<_> = diagnostic.labels.iter().map(|l| l.span.file).collect();
        files.sort();
        files.dedup();
        let cache = ariadne::sources(files.into_iter().map(|file| {
            let source = self.sources.get(file);
            (source.name().to_string(), source.text().to_string())
        }));
        report.finish().write(cache, out)
    }

    /// a report without a location: a header line like ariadne's, written
    /// by hand, then the help and the notes, in ariadne's order
    fn render_plain(&self, diagnostic: &Diagnostic, out: &mut dyn Write) -> io::Result<()> {
        let code = diagnostic
            .code
            .map(|c| format!("[{c}] "))
            .unwrap_or_default();
        let header = match diagnostic.severity {
            Severity::Error => "Error",
            Severity::Warning => "Warning",
            Severity::Note => "Advice",
        };
        let header = self.paint(
            &format!("{code}{header}:"),
            severity_color(diagnostic.severity),
        );
        writeln!(out, "{header} {}", diagnostic.message)?;
        for help in &diagnostic.help {
            writeln!(out, "    {} {help}", self.paint("Help:", Color::Fixed(115)))?;
        }
        for note in &diagnostic.notes {
            writeln!(out, "    {} {note}", self.paint("Note:", Color::Fixed(115)))?;
        }
        Ok(())
    }

    /// the text in a colour, when the renderer uses colour
    fn paint(&self, text: &str, color: Color) -> String {
        if self.color {
            use ariadne::Fmt;
            text.fg(color).to_string()
        } else {
            text.to_string()
        }
    }
}

/// the ariadne report kind of a severity. A note is ariadne's "Advice"
fn kind(severity: Severity) -> ReportKind<'static> {
    match severity {
        Severity::Error => ReportKind::Error,
        Severity::Warning => ReportKind::Warning,
        Severity::Note => ReportKind::Advice,
    }
}

/// the colour of the header and primary labels of a severity
fn severity_color(severity: Severity) -> Color {
    match severity {
        Severity::Error => Color::Red,
        Severity::Warning => Color::Yellow,
        Severity::Note => Color::Fixed(147),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Span;

    /// rendering without colour, so snapshots are plain text
    fn render(sources: &SourceMap, diagnostic: &Diagnostic) -> String {
        let mut out = Vec::new();
        Renderer::new(sources, false)
            .render(diagnostic, &mut out)
            .unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn renders_labels_notes_and_help() {
        let mut sources = SourceMap::new();
        let text = "def f(x: u32) -> u64:\n    return x\n";
        let file = sources.add("main.vg", text);
        let ret = text.find("u64").unwrap();
        let x = text.rfind('x').unwrap();
        let diagnostic = Diagnostic::error("mismatched types")
            .with_code("E9999")
            .with_primary(Span::new(file, x..x + 1), "expected `u64`, found `u32`")
            .with_secondary(Span::new(file, ret..ret + 3), "return type declared here")
            .with_note("integer types are never converted implicitly")
            .with_help("use `u64(x)`");
        insta::assert_snapshot!(render(&sources, &diagnostic));
    }

    #[test]
    fn renders_diagnostics_without_location() {
        let sources = SourceMap::new();
        let diagnostic = Diagnostic::error("could not read `vigil.toml`")
            .with_code("E9998")
            .with_note("permission denied");
        insta::assert_snapshot!(render(&sources, &diagnostic));
    }
}
