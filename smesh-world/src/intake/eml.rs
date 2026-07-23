//! RFC 5322 `.eml` → CDM. Single-part text bodies only for now — the
//! Meridian corpus controls its own email shape; MIME multipart lands with
//! the attachment-scoping work (vendor-attachment defect).
//!
//! Headers are native provenance: `From:` + `Date:` become attribution and
//! temporal metadata without any model in the loop. `X-Department` is
//! honored because Meridian's renderer stamps it.

use crate::cdm::{CdmDocument, CdmSpan, DocMetadata, NativeAnchor, SourceFormat};
use crate::WorldError;
use std::collections::BTreeMap;

pub fn ingest(bytes: &[u8]) -> Result<CdmDocument, WorldError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| WorldError::Malformed(format!("eml is not UTF-8: {e}")))?;
    let normalized = text.replace("\r\n", "\n");
    let (head, body) = normalized
        .split_once("\n\n")
        .ok_or_else(|| WorldError::Malformed("missing header/body separator".into()))?;

    let headers = parse_headers(head);
    let message_id = headers
        .get("message-id")
        .map(|v| v.trim_matches(['<', '>']).to_owned())
        .ok_or_else(|| WorldError::Malformed("missing Message-ID header".into()))?;

    let canonical = body.to_owned();
    let mut anchors = Vec::new();
    let mut offset = 0usize;
    for (idx, line) in canonical.split('\n').enumerate() {
        if !line.trim().is_empty() {
            anchors.push((
                CdmSpan::new(offset, offset + line.len()),
                NativeAnchor::EmailLine {
                    message_id: message_id.clone(),
                    line: u32::try_from(idx + 1).unwrap_or(u32::MAX),
                },
            ));
        }
        offset += line.len() + 1;
    }

    let mut extra = BTreeMap::new();
    extra.insert("message-id".to_owned(), message_id);
    if let Some(to) = headers.get("to") {
        extra.insert("to".to_owned(), to.clone());
    }

    Ok(CdmDocument::ingest(
        bytes,
        SourceFormat::Eml,
        canonical,
        anchors,
        DocMetadata {
            title: headers.get("subject").cloned(),
            author: headers.get("from").cloned(),
            date: headers.get("date").cloned(),
            department: headers.get("x-department").cloned(),
            extra,
        },
    ))
}

/// Lowercased header map with RFC 5322 folding (continuation lines start
/// with whitespace) unfolded.
fn parse_headers(head: &str) -> BTreeMap<String, String> {
    let mut headers: BTreeMap<String, String> = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in head.lines() {
        if line.starts_with([' ', '\t']) {
            if let Some(key) = &current {
                if let Some(value) = headers.get_mut(key) {
                    value.push(' ');
                    value.push_str(line.trim());
                }
            }
        } else if let Some((key, value)) = line.split_once(':') {
            let key = key.trim().to_ascii_lowercase();
            headers.insert(key.clone(), value.trim().to_owned());
            current = Some(key);
        }
    }
    headers
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEMO: &str = "From: Dana Reyes <dana.reyes@meridianmutual.example>\nTo: claims-all@meridianmutual.example\nSubject: Filing window change,\n effective immediately\nDate: Tue, 21 Jul 2026 09:14:00 -0700\nMessage-ID: <memo-4417@meridianmutual.example>\nX-Department: Claims\n\nTeam,\n\nThe filing window is now 45 days, superseding the handbook's 30.\n\nDana\n";

    #[test]
    fn headers_become_metadata_including_folded_subject() {
        let doc = ingest(MEMO.as_bytes()).unwrap();
        assert_eq!(
            doc.metadata.title.as_deref(),
            Some("Filing window change, effective immediately")
        );
        assert_eq!(
            doc.metadata.author.as_deref(),
            Some("Dana Reyes <dana.reyes@meridianmutual.example>")
        );
        assert_eq!(doc.metadata.department.as_deref(), Some("Claims"));
        assert_eq!(
            doc.metadata.extra.get("message-id").map(String::as_str),
            Some("memo-4417@meridianmutual.example")
        );
    }

    #[test]
    fn body_lines_anchor_to_message_id_coordinates() {
        let doc = ingest(MEMO.as_bytes()).unwrap();
        let start = doc.canonical_text.find("45 days").unwrap();
        match doc.native_anchor(CdmSpan::new(start, start + 7)).unwrap() {
            NativeAnchor::EmailLine { message_id, line } => {
                assert_eq!(message_id, "memo-4417@meridianmutual.example");
                assert_eq!(*line, 3);
            }
            other => panic!("unexpected anchor {other:?}"),
        }
    }

    #[test]
    fn headers_are_not_part_of_canonical_text() {
        let doc = ingest(MEMO.as_bytes()).unwrap();
        assert!(doc.canonical_text.starts_with("Team,"));
        assert!(!doc.canonical_text.contains("Message-ID"));
    }

    #[test]
    fn missing_separator_and_missing_message_id_are_malformed() {
        assert!(matches!(
            ingest(b"From: a@b.c\nSubject: x"),
            Err(WorldError::Malformed(_))
        ));
        assert!(matches!(
            ingest(b"From: a@b.c\nSubject: x\n\nbody\n"),
            Err(WorldError::Malformed(_))
        ));
    }

    #[test]
    fn crlf_emails_normalize_identically() {
        let crlf = MEMO.replace('\n', "\r\n");
        let doc = ingest(crlf.as_bytes()).unwrap();
        assert_eq!(
            doc.canonical_text,
            ingest(MEMO.as_bytes()).unwrap().canonical_text
        );
    }
}
