//! Corpus packet: named CDM documents plus the prompt block extractors read.
//! Tier 1 sees canonical text and intake metadata only — never raw formats.

use smesh_world::{CdmDocument, Registrar, WorldError};

pub struct NamedDoc {
    pub name: String,
    pub doc: CdmDocument,
}

/// Ingest raw artifacts through the Registrar into a packet, failing loudly
/// on any artifact Tier 0 cannot handle.
pub fn ingest_all(artifacts: &[(String, Vec<u8>)]) -> Result<Vec<NamedDoc>, WorldError> {
    artifacts
        .iter()
        .map(|(name, bytes)| {
            Registrar::ingest(bytes).map(|doc| NamedDoc {
                name: name.clone(),
                doc,
            })
        })
        .collect()
}

/// Render the packet as the document block shared by every extractor prompt.
pub fn prompt_block(docs: &[NamedDoc]) -> String {
    let mut out = String::new();
    for named in docs {
        out.push_str(&format!("=== document: {} ===\n", named.name));
        let m = &named.doc.metadata;
        if let Some(title) = &m.title {
            out.push_str(&format!("title: {title}\n"));
        }
        if let Some(author) = &m.author {
            out.push_str(&format!("author: {author}\n"));
        }
        if let Some(date) = &m.date {
            out.push_str(&format!("date: {date}\n"));
        }
        if let Some(department) = &m.department {
            out.push_str(&format!("department: {department}\n"));
        }
        out.push_str("---\n");
        out.push_str(&named.doc.canonical_text);
        out.push_str("\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use smesh_world::meridian;

    fn meridian_packet() -> Vec<NamedDoc> {
        let artifacts: Vec<(String, Vec<u8>)> = meridian::corpus()
            .artifacts
            .into_iter()
            .map(|a| (a.name.to_owned(), a.bytes))
            .collect();
        ingest_all(&artifacts).unwrap()
    }

    #[test]
    fn meridian_ingests_fully_into_a_packet() {
        assert_eq!(meridian_packet().len(), 9);
    }

    #[test]
    fn prompt_block_carries_names_metadata_and_text() {
        let block = prompt_block(&meridian_packet());
        assert!(block.contains("=== document: memo-4417.eml ==="));
        assert!(block.contains("department: Claims"));
        assert!(block.contains("the claims filing window is 45 days"));
        assert!(block.contains("=== document: clinical-policy-manual.pdf ==="));
    }
}
