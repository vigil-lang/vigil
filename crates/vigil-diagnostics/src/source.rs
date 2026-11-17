//! this module defines source files and spans into them. The [`SourceMap`]
//! owns the text of every file and gives a [`FileId`] for each, and a
//! [`Span`] is a pair of byte offsets into one file.

use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// a file of a [`SourceMap`], as its index there
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId(u32);

/// a byte range inside one source file. The offsets are `u32` to keep spans
/// small, so a file is limited to 4 GiB
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub file: FileId,
    /// offset of the first byte
    pub start: u32,
    /// offset just past the last byte
    pub end: u32,
}

impl Span {
    /// the span of a byte range of a file. Panics if the range is inverted
    /// or does not fit in `u32`
    pub fn new(file: FileId, range: Range<usize>) -> Self {
        assert!(range.start <= range.end, "inverted span {range:?}");
        Span {
            file,
            start: offset(range.start),
            end: offset(range.end),
        }
    }

    /// the byte range, for slicing the text of the file
    pub fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }

    /// the smallest span covering `self` and `other`, which must be in the
    /// same file
    pub fn to(self, other: Span) -> Span {
        assert_eq!(self.file, other.file, "joining spans of different files");
        Span {
            file: self.file,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

fn offset(n: usize) -> u32 {
    u32::try_from(n).expect("source files are limited to 4 GiB")
}

/// the text of a file, with the name shown in diagnostics
#[derive(Debug)]
pub struct SourceFile {
    name: String,
    path: Option<PathBuf>,
    text: String,
}

impl SourceFile {
    /// the name shown in diagnostics
    pub fn name(&self) -> &str {
        &self.name
    }

    /// where the file was read from, `None` for an in-memory file
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

/// the text of every file seen by the compiler
#[derive(Debug, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
    /// texts used in place of the files at these paths, like unsaved editor
    /// buffers. Nothing in this repository sets them
    overlays: std::collections::HashMap<PathBuf, String>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// registering an in-memory file, with a name but no path
    pub fn add(&mut self, name: impl Into<String>, text: impl Into<String>) -> FileId {
        self.push(SourceFile {
            name: name.into(),
            path: None,
            text: text.into(),
        })
    }

    /// an empty map that reads the files at these paths from `overlays`
    /// instead of the disk
    pub fn with_overlays(overlays: std::collections::HashMap<PathBuf, String>) -> Self {
        SourceMap {
            files: Vec::new(),
            overlays,
        }
    }

    /// reading a file (from its overlay if it has one, else from disk) and
    /// registering it. Loading a path twice registers it twice
    pub fn load(&mut self, path: &Path) -> io::Result<FileId> {
        let text = match self.overlays.get(path) {
            Some(text) => text.clone(),
            None => std::fs::read_to_string(path)?,
        };
        let name = display_name(path);
        Ok(self.push(SourceFile {
            name,
            path: Some(path.to_path_buf()),
            text,
        }))
    }

    /// the file of an id. Panics on an id from another map
    pub fn get(&self, id: FileId) -> &SourceFile {
        &self.files[id.0 as usize]
    }

    /// every file, in the order they were registered
    pub fn iter(&self) -> impl Iterator<Item = (FileId, &SourceFile)> {
        self.files
            .iter()
            .enumerate()
            .map(|(i, f)| (FileId(i as u32), f))
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    fn push(&mut self, file: SourceFile) -> FileId {
        let id = FileId(u32::try_from(self.files.len()).expect("too many source files"));
        self.files.push(file);
        id
    }
}

/// the name of a path in diagnostics: the path without a leading `./`
fn display_name(path: &Path) -> String {
    let path = path.strip_prefix(".").unwrap_or(path);
    path.display().to_string()
}
