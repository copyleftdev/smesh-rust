//! Candidate emissions and verdicts.
//!
//! Provenance is enforced at the type layer, not the prompt layer: a
//! `Citation` can only be constructed against a real document with a valid
//! span whose text matches the quote, and `CandidateEdge` construction checks
//! role capabilities. You cannot prompt your way past this module.

use crate::cdm::{CdmDocument, CdmSpan, DocId};
use crate::ontology::EdgeKind;
use crate::role::{Capability, WorldRole};
use crate::WorldError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    pub doc: DocId,
    pub span: CdmSpan,
    pub quote: String,
}

impl Citation {
    /// The only constructor: the quote is extracted from the document, so a
    /// citation whose quote diverges from its span cannot exist.
    pub fn grounded(doc: &CdmDocument, span: CdmSpan) -> Result<Self, WorldError> {
        let quote = doc.span_text(span)?.to_owned();
        Ok(Self {
            doc: doc.id,
            span,
            quote,
        })
    }

    /// Re-verify against the (possibly re-fetched) document — the Grounding
    /// Auditor's first check before any semantic entailment work.
    pub fn verify_against(&self, doc: &CdmDocument) -> Result<(), WorldError> {
        if doc.span_text(self.span)? == self.quote {
            Ok(())
        } else {
            Err(WorldError::QuoteMismatch)
        }
    }
}

/// How an edge earns its place in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceClass {
    /// Extracted, span-cited, mesh-corroborated.
    CorpusDerived { citations: Vec<Citation> },
    /// No document exists; the provenance IS the human's signature.
    /// Highest-trust class — captured tribal knowledge.
    HumanAttested { reviewer: crate::delta::ReviewerId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Judgment {
    Corroborate,
    Refute,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    pub by: WorldRole,
    pub judgment: Judgment,
    pub rationale: String,
}

impl Verdict {
    pub fn new(by: WorldRole, judgment: Judgment, rationale: String) -> Result<Self, WorldError> {
        if !by.has_capability(Capability::Verdict) {
            return Err(WorldError::CapabilityDenied(by, Capability::Verdict));
        }
        Ok(Self {
            by,
            judgment,
            rationale,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateEdge {
    pub subject: String,
    pub kind: EdgeKind,
    pub object: String,
    pub provenance: ProvenanceClass,
    pub emitted_by: WorldRole,
    pub verdicts: Vec<Verdict>,
}

impl CandidateEdge {
    pub fn emit(
        emitted_by: WorldRole,
        subject: String,
        kind: EdgeKind,
        object: String,
        citations: Vec<Citation>,
    ) -> Result<Self, WorldError> {
        if !emitted_by.has_capability(Capability::EmitCandidate) {
            return Err(WorldError::CapabilityDenied(
                emitted_by,
                Capability::EmitCandidate,
            ));
        }
        if citations.is_empty() {
            return Err(WorldError::MissingCitation);
        }
        Ok(Self {
            subject,
            kind,
            object,
            provenance: ProvenanceClass::CorpusDerived { citations },
            emitted_by,
            verdicts: Vec::new(),
        })
    }

    /// Stable identity used by ratification records and dedup.
    ///
    /// Length-prefixed rather than delimited. A bare separator means a subject
    /// containing it can impersonate another candidate — `("a|b", X, "c")` and
    /// `("a", X, "b|c")` produced the same key. That key decides which
    /// ratification decision applies to which edge, so a collision is two
    /// different claims sharing one human approval.
    pub fn key(&self) -> String {
        Self::key_parts(&self.subject, self.kind, &self.object)
    }

    /// The key computation, over its parts.
    pub fn key_parts(subject: &str, kind: EdgeKind, object: &str) -> String {
        format!(
            "{}:{}|{:?}|{}:{}",
            subject.len(),
            subject,
            kind,
            object.len(),
            object
        )
    }

    pub fn record(&mut self, verdict: Verdict) {
        self.verdicts.push(verdict);
    }

    pub fn corroborations(&self) -> u32 {
        self.count(Judgment::Corroborate)
    }

    pub fn refutations(&self) -> u32 {
        self.count(Judgment::Refute)
    }

    /// A contested edge has live evidence on both sides — it must reach a
    /// human, never be resolved by the mesh.
    pub fn is_contested(&self) -> bool {
        self.corroborations() > 0 && self.refutations() > 0
    }

    pub fn meets_consensus(&self) -> bool {
        let policy = self.kind.consensus_policy();
        self.corroborations() >= policy.min_corroborations
            && self.refutations() <= policy.max_refutations
    }

    fn count(&self, judgment: Judgment) -> u32 {
        u32::try_from(
            self.verdicts
                .iter()
                .filter(|v| v.judgment == judgment)
                .count(),
        )
        .unwrap_or(u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn keys_cannot_collide_across_a_separator() {
        // The key decides which ratification decision applies to which edge, so
        // two different claims sharing one is two claims sharing one approval.
        let a = CandidateEdge::key_parts("a|b", EdgeKind::ReportsTo, "c");
        let b = CandidateEdge::key_parts("a", EdgeKind::ReportsTo, "b|c");
        assert_ne!(a, b);
    }

    use super::*;
    use crate::cdm::{DocMetadata, SourceFormat};

    fn doc() -> CdmDocument {
        CdmDocument::ingest(
            b"handbook-v1",
            SourceFormat::Markdown,
            "The Claims department must file within 30 days.".to_owned(),
            vec![],
            DocMetadata::default(),
        )
    }

    fn citation(d: &CdmDocument) -> Citation {
        Citation::grounded(d, CdmSpan::new(4, 21)).unwrap()
    }

    #[test]
    fn citation_quote_is_extracted_not_asserted() {
        let d = doc();
        let c = citation(&d);
        assert_eq!(c.quote, "Claims department");
        assert!(c.verify_against(&d).is_ok());
    }

    #[test]
    fn citation_verification_catches_document_drift() {
        let d = doc();
        let c = citation(&d);
        let mutated = CdmDocument::ingest(
            b"handbook-v2",
            SourceFormat::Markdown,
            "The Claims division must file within 30 days.".to_owned(),
            vec![],
            DocMetadata::default(),
        );
        assert_eq!(c.verify_against(&mutated), Err(WorldError::QuoteMismatch));
    }

    #[test]
    fn only_extractors_emit_and_citations_are_mandatory() {
        let d = doc();
        let err = CandidateEdge::emit(
            WorldRole::GroundingAuditor,
            "Claims".into(),
            EdgeKind::GovernedBy,
            "30-day filing rule".into(),
            vec![citation(&d)],
        )
        .unwrap_err();
        assert!(matches!(err, WorldError::CapabilityDenied(..)));

        let err = CandidateEdge::emit(
            WorldRole::Policy,
            "Claims".into(),
            EdgeKind::GovernedBy,
            "30-day filing rule".into(),
            vec![],
        )
        .unwrap_err();
        assert_eq!(err, WorldError::MissingCitation);
    }

    #[test]
    fn only_verifiers_issue_verdicts() {
        let err = Verdict::new(
            WorldRole::Policy,
            Judgment::Corroborate,
            "looks right".into(),
        )
        .unwrap_err();
        assert!(matches!(err, WorldError::CapabilityDenied(..)));
        assert!(Verdict::new(
            WorldRole::GroundingAuditor,
            Judgment::Corroborate,
            "span entails claim".into()
        )
        .is_ok());
    }

    #[test]
    fn consensus_and_contested_semantics() {
        let d = doc();
        let mut edge = CandidateEdge::emit(
            WorldRole::Policy,
            "Claims".into(),
            EdgeKind::GovernedBy,
            "30-day filing rule".into(),
            vec![citation(&d)],
        )
        .unwrap();
        assert!(!edge.meets_consensus());

        let corroborate = |r| Verdict::new(r, Judgment::Corroborate, "entailed".into()).unwrap();
        edge.record(corroborate(WorldRole::GroundingAuditor));
        edge.record(corroborate(WorldRole::ContradictionSentinel));
        assert!(edge.meets_consensus());
        assert!(!edge.is_contested());

        edge.record(
            Verdict::new(
                WorldRole::ContradictionSentinel,
                Judgment::Refute,
                "conflicts with Finance rule".into(),
            )
            .unwrap(),
        );
        assert!(edge.is_contested());
        assert!(!edge.meets_consensus());
    }
}
