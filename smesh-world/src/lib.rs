//! # SMESH World
//!
//! Signed organizational world models built by a mesh of experts and ratified
//! by humans. See `WORLD-MODEL.md` at the workspace root for the full design.
//!
//! - **AION is the truth plane**: signed, append-only, content-addressed.
//! - **SMESH is the refinery**: experts propose, the field filters.
//! - **The human is the merge authority**: nothing is signed unratified.
//!
//! Governing principle — separation of powers: extractors propose, verifiers
//! judge, one non-LLM curator signs. No role holds two of those powers, and
//! the type system enforces it.

pub mod candidate;
pub mod cdm;
pub mod corpus;
pub mod delta;
pub mod intake;
pub mod meridian;
pub mod ontology;
pub mod role;

pub use candidate::{CandidateEdge, Citation, Judgment, ProvenanceClass, Verdict};
pub use cdm::{CdmDocument, CdmSpan, DocId, NativeAnchor, SourceFormat};
pub use corpus::{DefectManifest, GoldEdge, GoldGraph, PlantedDefect, Scorecard};
pub use delta::{
    Changeset, EvidenceView, Lane, RatificationRecord, Ratified, RejectedView, ReviewDecision,
    ReviewerId, Signature, Signed, SignedChangeset, Staged, StagedRun,
};
pub use intake::Registrar;
pub use ontology::{ConsensusPolicy, EdgeKind, StructuralConstraint, TransportKind};
pub use role::{Capability, ModelPolicy, Tier, WorldRole};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum WorldError {
    #[error("span {start}..{end} is invalid for document of length {len}")]
    InvalidSpan {
        start: usize,
        end: usize,
        len: usize,
    },
    #[error("quote does not match the cited span text")]
    QuoteMismatch,
    #[error("role {0:?} lacks capability {1:?}")]
    CapabilityDenied(role::WorldRole, role::Capability),
    #[error("corpus-derived candidates require at least one citation")]
    MissingCitation,
    #[error("human-attested candidates require a reviewer signature")]
    MissingAttestation,
    #[error("changeset is empty; nothing to stage")]
    EmptyChangeset,
    #[error("no adapter for format {0:?}")]
    UnsupportedFormat(cdm::SourceFormat),
    #[error("malformed artifact: {0}")]
    Malformed(String),
    #[error("ratification record does not cover candidate {0}")]
    UnreviewedCandidate(String),
    #[error("ratification rules on {0}, which is not in this changeset")]
    ForeignRatification(String),
}

/// A window of surrounding text, clipped to character boundaries.
///
/// Slicing a `str` by byte offset panics when the cut lands inside a multi-byte
/// character, and the text here comes from arbitrary documents. Lived in two
/// places, correct in one of them; keeping a single copy is the point.
pub fn context_window(text: &str, start: usize, end: usize, radius: usize) -> &str {
    let mut lo = start.saturating_sub(radius);
    while lo > 0 && !text.is_char_boundary(lo) {
        lo -= 1;
    }
    let mut hi = (end + radius).min(text.len());
    while hi < text.len() && !text.is_char_boundary(hi) {
        hi += 1;
    }
    &text[lo..hi]
}

#[cfg(test)]
mod context_window_tests {
    #[test]
    fn a_window_never_splits_a_character() {
        let text = "policy: it’s ninety days — confirmed";
        for start in 0..text.len() {
            for radius in [0usize, 1, 3, 200] {
                // Must not panic for any offset or radius.
                let _ = super::context_window(text, start, start, radius);
            }
        }
    }
}
