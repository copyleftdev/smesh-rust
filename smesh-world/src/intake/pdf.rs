//! PDF → CDM via `pdf-extract` per-page text (the Prévue/AkamaiForms
//! lineage). Each page's text becomes a page-spanning anchor, so a citation
//! resolves to "policy-manual.pdf, page 3" on the dashboard.
//!
//! A PDF with no extractable text (pure scan) is `Malformed` rather than a
//! silently empty document — the OCR path is future Tier 0 work and its
//! absence must be loud.

use crate::cdm::{CdmDocument, CdmSpan, DocMetadata, NativeAnchor, SourceFormat};
use crate::WorldError;

pub fn ingest(bytes: &[u8]) -> Result<CdmDocument, WorldError> {
    let pages = pdf_extract::extract_text_from_mem_by_pages(bytes)
        .map_err(|e| WorldError::Malformed(format!("pdf extraction failed: {e}")))?;

    let mut canonical = String::new();
    let mut anchors = Vec::new();
    for (idx, page) in pages.iter().enumerate() {
        let text = page.replace("\r\n", "\n");
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !canonical.is_empty() {
            canonical.push_str("\n\n");
        }
        let start = canonical.len();
        canonical.push_str(trimmed);
        anchors.push((
            CdmSpan::new(start, canonical.len()),
            NativeAnchor::PdfPage {
                page: u32::try_from(idx + 1).unwrap_or(u32::MAX),
            },
        ));
    }

    if canonical.is_empty() {
        return Err(WorldError::Malformed(
            "pdf has no extractable text (scanned image? OCR not yet supported)".into(),
        ));
    }

    Ok(CdmDocument::ingest(
        bytes,
        SourceFormat::Pdf,
        canonical,
        anchors,
        DocMetadata::default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::Citation;
    use crate::intake::pdf_render::render_pdf;

    fn fixture() -> Vec<u8> {
        render_pdf(&[
            vec![
                "Meridian Clinical Policy Manual".to_owned(),
                "Prior authorization is required for specialty drugs.".to_owned(),
            ],
            vec!["Claims must be filed within 30 days.".to_owned()],
        ])
    }

    #[test]
    fn pages_become_page_anchored_spans() {
        let doc = ingest(&fixture()).unwrap();
        assert_eq!(doc.format, SourceFormat::Pdf);

        let p1 = doc.canonical_text.find("Prior authorization").unwrap();
        match doc.native_anchor(CdmSpan::new(p1, p1 + 19)).unwrap() {
            NativeAnchor::PdfPage { page } => assert_eq!(*page, 1),
            other => panic!("unexpected anchor {other:?}"),
        }

        let p2 = doc.canonical_text.find("30 days").unwrap();
        match doc.native_anchor(CdmSpan::new(p2, p2 + 7)).unwrap() {
            NativeAnchor::PdfPage { page } => assert_eq!(*page, 2),
            other => panic!("unexpected anchor {other:?}"),
        }
    }

    #[test]
    fn grounded_citation_round_trips_through_pdf() {
        let doc = ingest(&fixture()).unwrap();
        let start = doc.canonical_text.find("specialty drugs").unwrap();
        let citation = Citation::grounded(&doc, CdmSpan::new(start, start + 15)).unwrap();
        assert_eq!(citation.quote, "specialty drugs");
        assert!(citation.verify_against(&doc).is_ok());
    }

    #[test]
    fn rendering_is_deterministic_so_doc_ids_are_stable() {
        assert_eq!(fixture(), fixture());
        assert_eq!(
            ingest(&fixture()).unwrap().id,
            ingest(&fixture()).unwrap().id
        );
    }

    #[test]
    fn textless_pdf_is_loudly_malformed() {
        let empty = render_pdf(&[vec![]]);
        assert!(matches!(ingest(&empty), Err(WorldError::Malformed(_))));
    }

    #[test]
    fn garbage_bytes_are_malformed_not_a_panic() {
        assert!(matches!(
            ingest(b"%PDF-1.4 garbage"),
            Err(WorldError::Malformed(_))
        ));
    }
}
