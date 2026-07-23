//! Tier 0 — the Registrar of Record.
//!
//! Any artifact in, CDM out. Detection and extraction are deterministic
//! code; per the model policy, an LLM is reserved for triaging genuinely
//! ambiguous blobs and never touches fidelity. Tier 1 experts only ever see
//! the CDM this module produces.

mod eml;
mod markdown;
mod pdf;
pub(crate) mod pdf_render;

use crate::cdm::{CdmDocument, SourceFormat};
use crate::WorldError;

pub struct Registrar;

impl Registrar {
    /// Deterministic format detection. Order matters: magic bytes first,
    /// then structural sniffs on the text, markdown as the text fallback.
    pub fn sniff(bytes: &[u8]) -> SourceFormat {
        if bytes.starts_with(b"%PDF-") {
            return SourceFormat::Pdf;
        }
        let Ok(text) = std::str::from_utf8(bytes) else {
            return SourceFormat::Unknown;
        };
        if looks_like_email(text) {
            return SourceFormat::Eml;
        }
        let trimmed = text.trim_start();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return SourceFormat::Json;
        }
        let lower = trimmed.get(..14).unwrap_or(trimmed).to_ascii_lowercase();
        if lower.starts_with("<!doctype html") || lower.starts_with("<html") {
            return SourceFormat::Html;
        }
        SourceFormat::Markdown
    }

    pub fn ingest(bytes: &[u8]) -> Result<CdmDocument, WorldError> {
        match Self::sniff(bytes) {
            SourceFormat::Markdown => markdown::ingest(bytes),
            SourceFormat::Eml => eml::ingest(bytes),
            SourceFormat::Pdf => pdf::ingest(bytes),
            other => Err(WorldError::UnsupportedFormat(other)),
        }
    }
}

/// An email is a header block (`Key: value` lines, folding allowed) that
/// includes `From:` plus `Subject:` or `Message-ID:` before the first blank
/// line.
fn looks_like_email(text: &str) -> bool {
    let mut saw_from = false;
    let mut saw_subject_or_id = false;
    for line in text.lines() {
        if line.trim().is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("from:") {
            saw_from = true;
        } else if lower.starts_with("subject:") || lower.starts_with("message-id:") {
            saw_subject_or_id = true;
        } else if !line.starts_with([' ', '\t']) && !line.contains(':') {
            return false;
        }
    }
    saw_from && saw_subject_or_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniff_detects_by_magic_and_structure() {
        assert_eq!(Registrar::sniff(b"%PDF-1.7 ..."), SourceFormat::Pdf);
        assert_eq!(Registrar::sniff(b"{\"a\": 1}"), SourceFormat::Json);
        assert_eq!(
            Registrar::sniff(b"<!DOCTYPE html><html></html>"),
            SourceFormat::Html
        );
        assert_eq!(
            Registrar::sniff(b"From: a@b.c\nSubject: hi\n\nbody"),
            SourceFormat::Eml
        );
        assert_eq!(
            Registrar::sniff(b"# Handbook\n\ntext"),
            SourceFormat::Markdown
        );
        assert_eq!(Registrar::sniff(&[0xff, 0xfe, 0x00]), SourceFormat::Unknown);
    }

    #[test]
    fn prose_with_colons_is_not_an_email() {
        assert_eq!(
            Registrar::sniff(b"From: the desk of the CEO\nnote that follows\n"),
            SourceFormat::Markdown
        );
    }

    #[test]
    fn unsupported_formats_are_an_explicit_error() {
        assert!(matches!(
            Registrar::ingest(b"{\"not\": \"yet\"}"),
            Err(WorldError::UnsupportedFormat(SourceFormat::Json))
        ));
    }
}
