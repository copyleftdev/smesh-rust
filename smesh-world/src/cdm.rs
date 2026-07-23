//! Canonical Document Model: Tier 0 output. Extractors never see raw formats.
//!
//! Every CDM span is dual-anchored — addressable in canonical text for
//! machine citation, and mapped back to native coordinates so the human
//! dashboard highlights the original artifact.

use crate::WorldError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// Content-addressed document identity: `BLAKE3(original bytes)`.
/// Re-ingesting identical bytes is idempotent by construction.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DocId([u8; 32]);

impl DocId {
    pub fn from_bytes(original: &[u8]) -> Self {
        Self(*blake3::hash(original).as_bytes())
    }

    pub fn as_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Debug for DocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DocId({}…)", &self.as_hex()[..12])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceFormat {
    Markdown,
    Pdf,
    Eml,
    Yaml,
    Json,
    Csv,
    Html,
    Unknown,
}

/// Native coordinates in the original artifact, per format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeAnchor {
    PdfPage {
        page: u32,
    },
    EmailLine {
        message_id: String,
        line: u32,
    },
    MarkdownHeading {
        heading_path: Vec<String>,
        line: u32,
    },
    Line {
        line: u32,
    },
    ByteOffset {
        offset: usize,
    },
}

/// A half-open byte range `start..end` into a document's canonical text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CdmSpan {
    pub start: usize,
    pub end: usize,
}

impl CdmSpan {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// Structured metadata harvested at intake. Email headers are native
/// provenance: `From:` + `Date:` yield attribution and temporal edges free.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub date: Option<String>,
    pub department: Option<String>,
    pub extra: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CdmDocument {
    pub id: DocId,
    pub format: SourceFormat,
    pub canonical_text: String,
    pub anchors: Vec<(CdmSpan, NativeAnchor)>,
    pub metadata: DocMetadata,
}

impl CdmDocument {
    pub fn ingest(
        original_bytes: &[u8],
        format: SourceFormat,
        canonical_text: String,
        anchors: Vec<(CdmSpan, NativeAnchor)>,
        metadata: DocMetadata,
    ) -> Self {
        Self {
            id: DocId::from_bytes(original_bytes),
            format,
            canonical_text,
            anchors,
            metadata,
        }
    }

    pub fn span_text(&self, span: CdmSpan) -> Result<&str, WorldError> {
        let len = self.canonical_text.len();
        if span.start >= span.end
            || span.end > len
            || !self.canonical_text.is_char_boundary(span.start)
            || !self.canonical_text.is_char_boundary(span.end)
        {
            return Err(WorldError::InvalidSpan {
                start: span.start,
                end: span.end,
                len,
            });
        }
        Ok(&self.canonical_text[span.start..span.end])
    }

    /// Resolve a canonical span back to its native coordinates: the innermost
    /// anchor whose span contains it.
    pub fn native_anchor(&self, span: CdmSpan) -> Option<&NativeAnchor> {
        self.anchors
            .iter()
            .filter(|(s, _)| s.start <= span.start && span.end <= s.end)
            .min_by_key(|(s, _)| s.end - s.start)
            .map(|(_, a)| a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> CdmDocument {
        CdmDocument::ingest(
            b"raw-bytes",
            SourceFormat::Markdown,
            "Claims must be filed within 30 days.".to_owned(),
            vec![
                (
                    CdmSpan::new(0, 36),
                    NativeAnchor::MarkdownHeading {
                        heading_path: vec!["Claims".into(), "Filing".into()],
                        line: 12,
                    },
                ),
                (CdmSpan::new(0, 6), NativeAnchor::Line { line: 12 }),
            ],
            DocMetadata::default(),
        )
    }

    #[test]
    fn doc_id_is_content_addressed_and_idempotent() {
        assert_eq!(DocId::from_bytes(b"same"), DocId::from_bytes(b"same"));
        assert_ne!(DocId::from_bytes(b"same"), DocId::from_bytes(b"other"));
    }

    #[test]
    fn span_text_extracts_and_rejects_out_of_bounds() {
        let d = doc();
        assert_eq!(d.span_text(CdmSpan::new(0, 6)).unwrap(), "Claims");
        assert!(matches!(
            d.span_text(CdmSpan::new(10, 200)),
            Err(WorldError::InvalidSpan { .. })
        ));
        assert!(d.span_text(CdmSpan::new(5, 5)).is_err());
    }

    #[test]
    fn span_text_rejects_non_char_boundaries() {
        let d = CdmDocument::ingest(
            b"x",
            SourceFormat::Markdown,
            "Prévue".to_owned(),
            vec![],
            DocMetadata::default(),
        );
        assert!(d.span_text(CdmSpan::new(0, 3)).is_err());
        assert_eq!(d.span_text(CdmSpan::new(0, 4)).unwrap(), "Pré");
    }

    #[test]
    fn native_anchor_prefers_innermost_containing_span() {
        let d = doc();
        assert!(matches!(
            d.native_anchor(CdmSpan::new(0, 6)),
            Some(NativeAnchor::Line { line: 12 })
        ));
        assert!(matches!(
            d.native_anchor(CdmSpan::new(7, 11)),
            Some(NativeAnchor::MarkdownHeading { .. })
        ));
        assert!(d.native_anchor(CdmSpan::new(0, 999)).is_none());
    }
}
