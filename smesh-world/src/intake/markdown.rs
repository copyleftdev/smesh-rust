//! Markdown → CDM. Canonical text is the LF-normalized source; every
//! non-empty line is anchored with its heading path so a citation renders as
//! "Handbook › Claims › Filing, line 12" on the dashboard.

use crate::cdm::{CdmDocument, CdmSpan, DocMetadata, NativeAnchor, SourceFormat};
use crate::WorldError;

pub fn ingest(bytes: &[u8]) -> Result<CdmDocument, WorldError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| WorldError::Malformed(format!("markdown is not UTF-8: {e}")))?;
    let canonical = text.replace("\r\n", "\n");

    let mut anchors = Vec::new();
    let mut heading_stack: Vec<(u8, String)> = Vec::new();
    let mut title = None;
    let mut offset = 0usize;

    for (idx, line) in canonical.split('\n').enumerate() {
        let line_no = u32::try_from(idx + 1).unwrap_or(u32::MAX);
        if let Some((level, heading)) = parse_heading(line) {
            while heading_stack.last().is_some_and(|(l, _)| *l >= level) {
                heading_stack.pop();
            }
            heading_stack.push((level, heading.clone()));
            if level == 1 && title.is_none() {
                title = Some(heading);
            }
        }
        if !line.trim().is_empty() {
            anchors.push((
                CdmSpan::new(offset, offset + line.len()),
                NativeAnchor::MarkdownHeading {
                    heading_path: heading_stack.iter().map(|(_, h)| h.clone()).collect(),
                    line: line_no,
                },
            ));
        }
        offset += line.len() + 1;
    }

    Ok(CdmDocument::ingest(
        bytes,
        SourceFormat::Markdown,
        canonical,
        anchors,
        DocMetadata {
            title,
            ..DocMetadata::default()
        },
    ))
}

fn parse_heading(line: &str) -> Option<(u8, String)> {
    let hashes = line.bytes().take_while(|b| *b == b'#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = line[hashes..].strip_prefix(' ')?;

    // Only a *closing sequence* is decoration, and CommonMark requires it to be
    // preceded by a space. `trim_end_matches` removed hashes unconditionally,
    // so "## Sprint #" became "Sprint" and "### C#" became "C" — the heading
    // silently lost the character that identified it.
    let trimmed = rest.trim();
    let heading = match trimmed.rsplit_once(' ') {
        Some((before, tail)) if !tail.is_empty() && tail.bytes().all(|b| b == b'#') => {
            before.trim()
        }
        _ => trimmed,
    };
    if heading.is_empty() {
        return None;
    }
    Some((
        u8::try_from(hashes).expect("hashes <= 6"),
        heading.to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::Citation;

    const HANDBOOK: &str = "# Meridian Handbook\n\n## Claims\n\n### Filing\n\nClaims must be filed within 30 days.\n\n## Finance\n\nFinance owns the general ledger.\n";

    #[test]
    fn title_comes_from_first_h1() {
        let doc = ingest(HANDBOOK.as_bytes()).unwrap();
        assert_eq!(doc.metadata.title.as_deref(), Some("Meridian Handbook"));
    }

    #[test]
    fn heading_paths_track_nesting_and_sibling_resets() {
        let doc = ingest(HANDBOOK.as_bytes()).unwrap();
        let filing = doc.canonical_text.find("Claims must").unwrap();
        let span = CdmSpan::new(
            filing,
            filing + "Claims must be filed within 30 days.".len(),
        );
        match doc.native_anchor(span).unwrap() {
            NativeAnchor::MarkdownHeading { heading_path, line } => {
                assert_eq!(heading_path, &["Meridian Handbook", "Claims", "Filing"]);
                assert_eq!(*line, 7);
            }
            other => panic!("unexpected anchor {other:?}"),
        }

        let ledger = doc.canonical_text.find("Finance owns").unwrap();
        match doc.native_anchor(CdmSpan::new(ledger, ledger + 7)).unwrap() {
            NativeAnchor::MarkdownHeading { heading_path, .. } => {
                assert_eq!(heading_path, &["Meridian Handbook", "Finance"]);
            }
            other => panic!("unexpected anchor {other:?}"),
        }
    }

    #[test]
    fn crlf_input_is_normalized_but_identity_is_original_bytes() {
        let crlf = HANDBOOK.replace('\n', "\r\n");
        let doc = ingest(crlf.as_bytes()).unwrap();
        assert_eq!(doc.canonical_text, HANDBOOK);
        assert_ne!(doc.id, ingest(HANDBOOK.as_bytes()).unwrap().id);
    }

    #[test]
    fn grounded_citation_round_trips_through_the_adapter() {
        let doc = ingest(HANDBOOK.as_bytes()).unwrap();
        let start = doc.canonical_text.find("30 days").unwrap();
        let citation = Citation::grounded(&doc, CdmSpan::new(start, start + 7)).unwrap();
        assert_eq!(citation.quote, "30 days");
        assert!(citation.verify_against(&doc).is_ok());
    }

    #[test]
    fn hash_runs_and_missing_space_are_not_headings() {
        assert!(parse_heading("#######").is_none());
        assert!(parse_heading("#NoSpace").is_none());
        assert_eq!(
            parse_heading("## Trailing ##"),
            Some((2, "Trailing".into()))
        );
    }
}
