//! Changeset lifecycle: the mesh never publishes — it stages.
//!
//! A staged changeset is a PR against the world. Type-state makes the
//! ratification protocol unskippable: `sign` exists only on
//! `Changeset<Ratified>`, and a `Ratified` value can only be produced by
//! `ratify`, which demands a reviewer decision for every candidate.

use crate::candidate::CandidateEdge;
use crate::ontology::EdgeKind;
use crate::WorldError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ReviewerId(pub String);

/// Ed25519 signature bytes. Key handling lives with the dashboard identity
/// layer; the kernel only requires that a signature is present and bound.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature(pub Vec<u8>);

/// Ratification is always total; attention is tiered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lane {
    /// Summarized batch, spot-checkable.
    Green,
    /// Individually surfaced with evidence.
    Amber,
    /// Mandatory individual decision — why the human is in the loop.
    Red,
}

impl Lane {
    pub fn assign(edge: &CandidateEdge) -> Lane {
        if edge.is_contested() {
            return Lane::Red;
        }
        let normative = matches!(
            edge.kind,
            EdgeKind::GovernedBy | EdgeKind::Requires | EdgeKind::Triggers | EdgeKind::Supersedes
        );
        if normative || edge.corroborations() < 2 {
            Lane::Amber
        } else {
            Lane::Green
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReviewDecision {
    Approve,
    /// The human version wins; the mesh-vs-human delta is a labeled error
    /// signal fed back into expert reputation.
    Edit {
        amended: CandidateEdge,
    },
    Reject {
        reason: String,
    },
    /// Decays back into the field for more evidence.
    Defer,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatificationRecord {
    pub reviewer: ReviewerId,
    pub decisions: BTreeMap<String, ReviewDecision>,
    pub signature: Signature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Staged;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ratified;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signed;

#[derive(Debug, Clone, PartialEq)]
pub struct Changeset<State> {
    pub base_rev: String,
    pub edges: Vec<CandidateEdge>,
    state: std::marker::PhantomData<State>,
}

/// What ratification produced besides the ratified changeset itself.
#[derive(Debug, Clone, PartialEq)]
pub struct RatificationOutcome {
    pub changeset: Changeset<Ratified>,
    pub ratification: RatificationRecord,
    pub killed: Vec<(CandidateEdge, String)>,
    pub parked: Vec<CandidateEdge>,
}

impl Changeset<Staged> {
    pub fn stage(base_rev: String, edges: Vec<CandidateEdge>) -> Result<Self, WorldError> {
        if edges.is_empty() {
            return Err(WorldError::EmptyChangeset);
        }
        Ok(Self {
            base_rev,
            edges,
            state: std::marker::PhantomData,
        })
    }

    pub fn lanes(&self) -> Vec<(Lane, &CandidateEdge)> {
        self.edges.iter().map(|e| (Lane::assign(e), e)).collect()
    }

    /// Every candidate must carry a decision — ultimate sign-off is total.
    pub fn ratify(self, record: RatificationRecord) -> Result<RatificationOutcome, WorldError> {
        for edge in &self.edges {
            if !record.decisions.contains_key(&edge.key()) {
                return Err(WorldError::UnreviewedCandidate(edge.key()));
            }
        }
        let mut retained = Vec::new();
        let mut killed = Vec::new();
        let mut parked = Vec::new();
        for edge in self.edges {
            match record.decisions.get(&edge.key()).cloned() {
                Some(ReviewDecision::Approve) => retained.push(edge),
                Some(ReviewDecision::Edit { amended }) => retained.push(amended),
                Some(ReviewDecision::Reject { reason }) => killed.push((edge, reason)),
                Some(ReviewDecision::Defer) => parked.push(edge),
                None => unreachable!("coverage checked above"),
            }
        }
        if retained.is_empty() {
            return Err(WorldError::EmptyChangeset);
        }
        Ok(RatificationOutcome {
            changeset: Changeset {
                base_rev: self.base_rev,
                edges: retained,
                state: std::marker::PhantomData,
            },
            ratification: record,
            killed,
            parked,
        })
    }
}

/// One citation prepared for human eyes: the quote, where it lives in the
/// original artifact, and enough surrounding context to judge it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceView {
    pub doc_name: String,
    pub quote: String,
    pub anchor: String,
    pub context: String,
}

/// A firewalled emission, summarized for the transparency panel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedView {
    pub role: String,
    pub summary: String,
    pub reason: String,
}

/// The refinery → dashboard handoff: everything a reviewer needs to ratify,
/// serialized to disk between the mesh run and the human session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StagedRun {
    pub base_rev: String,
    pub candidates: Vec<CandidateEdge>,
    /// Evidence per candidate `key()`, in citation order.
    pub evidence: std::collections::BTreeMap<String, Vec<EvidenceView>>,
    pub rejected: Vec<RejectedView>,
    pub contradictions_caught: usize,
    pub scorecard: Option<crate::corpus::Scorecard>,
}

/// The Curator's output: a new revision whose identity commits to the base
/// revision, the surviving edges, and the human ratification that authorized
/// them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignedChangeset {
    pub base_rev: String,
    pub new_rev: String,
    pub edges: Vec<CandidateEdge>,
    pub ratification: RatificationRecord,
}

impl Changeset<Ratified> {
    /// Only reachable through `ratify` — the type system is the protocol.
    /// Attach a ratification, refusing one that does not cover these edges.
    ///
    /// The reviewer's signature covers the base revision, their identity and
    /// the decision map — it does not cover the edges. So a record signed for
    /// one changeset could previously be stapled onto a different one with the
    /// same base revision, and the result verified, having been approved by
    /// nobody. The premise of this whole design is that nothing reaches the
    /// signed graph without a human ratification record; that only holds if the
    /// record demonstrably concerns *these* edges.
    ///
    /// The decision map is signed, so requiring an exact correspondence between
    /// it and the edges binds the signature to the edge set without changing
    /// what is signed.
    pub fn sign(self, ratification: RatificationRecord) -> Result<SignedChangeset, WorldError> {
        let edge_keys: BTreeSet<String> = self.edges.iter().map(|e| e.key()).collect();
        let decided: BTreeSet<String> = ratification.decisions.keys().cloned().collect();

        if let Some(missing) = edge_keys.difference(&decided).next() {
            return Err(WorldError::UnreviewedCandidate(missing.clone()));
        }
        if let Some(extra) = decided.difference(&edge_keys).next() {
            return Err(WorldError::ForeignRatification(extra.clone()));
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(self.base_rev.as_bytes());
        hasher.update(&ratification.signature.0);
        for edge in &self.edges {
            hasher.update(edge.key().as_bytes());
        }
        Ok(SignedChangeset {
            base_rev: self.base_rev,
            new_rev: hasher.finalize().to_hex().to_string(),
            edges: self.edges,
            ratification,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{Citation, Judgment, Verdict};
    use crate::cdm::{CdmDocument, CdmSpan, DocMetadata, SourceFormat};
    use crate::role::WorldRole;

    fn doc() -> CdmDocument {
        CdmDocument::ingest(
            b"handbook",
            SourceFormat::Markdown,
            "Claims reports to Operations. Finance owns the ledger.".to_owned(),
            vec![],
            DocMetadata::default(),
        )
    }

    fn edge(d: &CdmDocument, subject: &str, kind: EdgeKind, object: &str) -> CandidateEdge {
        CandidateEdge::emit(
            WorldRole::Structure,
            subject.into(),
            kind,
            object.into(),
            vec![Citation::grounded(d, CdmSpan::new(0, 6)).unwrap()],
        )
        .unwrap()
    }

    fn corroborated(mut e: CandidateEdge, times: u32) -> CandidateEdge {
        for _ in 0..times {
            e.record(
                Verdict::new(
                    WorldRole::GroundingAuditor,
                    Judgment::Corroborate,
                    "ok".into(),
                )
                .unwrap(),
            );
        }
        e
    }

    fn record(reviewer: &str, decisions: Vec<(String, ReviewDecision)>) -> RatificationRecord {
        RatificationRecord {
            reviewer: ReviewerId(reviewer.into()),
            decisions: decisions.into_iter().collect(),
            signature: Signature(vec![7; 64]),
        }
    }

    #[test]
    fn staging_rejects_empty_changesets() {
        assert_eq!(
            Changeset::stage("rev0".into(), vec![]).unwrap_err(),
            WorldError::EmptyChangeset
        );
    }

    #[test]
    fn lane_assignment_tiers_attention() {
        let d = doc();
        let green = corroborated(edge(&d, "Claims", EdgeKind::MemberOf, "Meridian"), 2);
        let amber = corroborated(edge(&d, "Claims", EdgeKind::GovernedBy, "30-day rule"), 2);
        let mut red = corroborated(edge(&d, "Claims", EdgeKind::ReportsTo, "Ops"), 1);
        red.record(
            Verdict::new(
                WorldRole::ContradictionSentinel,
                Judgment::Refute,
                "conflicting org chart".into(),
            )
            .unwrap(),
        );
        assert_eq!(Lane::assign(&green), Lane::Green);
        assert_eq!(Lane::assign(&amber), Lane::Amber);
        assert_eq!(Lane::assign(&red), Lane::Red);
    }

    #[test]
    fn ratification_must_cover_every_candidate() {
        let d = doc();
        let a = edge(&d, "Claims", EdgeKind::ReportsTo, "Ops");
        let b = edge(&d, "Finance", EdgeKind::Owns, "Ledger");
        let staged = Changeset::stage("rev0".into(), vec![a.clone(), b]).unwrap();
        let partial = record("dj", vec![(a.key(), ReviewDecision::Approve)]);
        assert!(matches!(
            staged.ratify(partial),
            Err(WorldError::UnreviewedCandidate(_))
        ));
    }

    #[test]
    fn decisions_route_to_retained_killed_parked() {
        let d = doc();
        let a = edge(&d, "Claims", EdgeKind::ReportsTo, "Ops");
        let b = edge(&d, "Finance", EdgeKind::Owns, "Ledger");
        let c = edge(&d, "HR", EdgeKind::MemberOf, "Meridian");
        let amended = edge(&d, "Finance", EdgeKind::Owns, "General Ledger");
        let staged =
            Changeset::stage("rev0".into(), vec![a.clone(), b.clone(), c.clone()]).unwrap();
        let outcome = staged
            .ratify(record(
                "dj",
                vec![
                    (a.key(), ReviewDecision::Approve),
                    (
                        b.key(),
                        ReviewDecision::Edit {
                            amended: amended.clone(),
                        },
                    ),
                    (c.key(), ReviewDecision::Defer),
                ],
            ))
            .unwrap();
        assert_eq!(outcome.changeset.edges, vec![a, amended]);
        assert!(outcome.killed.is_empty());
        assert_eq!(outcome.parked, vec![c]);
    }

    #[test]
    fn fully_rejected_changesets_cannot_be_signed() {
        let d = doc();
        let a = edge(&d, "Claims", EdgeKind::ReportsTo, "Ops");
        let staged = Changeset::stage("rev0".into(), vec![a.clone()]).unwrap();
        let outcome = staged.ratify(record(
            "dj",
            vec![(
                a.key(),
                ReviewDecision::Reject {
                    reason: "stale org chart".into(),
                },
            )],
        ));
        assert_eq!(outcome.unwrap_err(), WorldError::EmptyChangeset);
    }

    #[test]
    fn signing_commits_to_base_edges_and_ratification() {
        let d = doc();
        let a = edge(&d, "Claims", EdgeKind::ReportsTo, "Ops");
        let staged = Changeset::stage("rev0".into(), vec![a.clone()]).unwrap();
        let outcome = staged
            .ratify(record("dj", vec![(a.key(), ReviewDecision::Approve)]))
            .unwrap();
        let signed = outcome
            .changeset
            .clone()
            .sign(outcome.ratification.clone())
            .unwrap();
        assert_eq!(signed.base_rev, "rev0");
        assert_eq!(signed.new_rev.len(), 64);

        let mut other_sig = outcome.ratification.clone();
        other_sig.signature = Signature(vec![9; 64]);
        let resigned = outcome.changeset.sign(other_sig).unwrap();
        assert_ne!(signed.new_rev, resigned.new_rev);
    }

    #[test]
    fn a_ratification_cannot_be_moved_to_a_changeset_it_did_not_approve() {
        // The type-state already stops an unratified changeset being signed.
        // What it does not stop is swapping records between two ratified ones:
        // the reviewer's signature covers the base revision, their name and the
        // decisions, not the edges. So a record approving one changeset could
        // be stapled onto another with the same base, and the result would
        // verify while having been approved by nobody. That is the single thing
        // this design promises cannot happen.
        let d = doc();
        let approved = edge(&d, "Claims", EdgeKind::ReportsTo, "Ops");
        let smuggled = edge(&d, "Claims", EdgeKind::ReportsTo, "Finance");

        let honest = Changeset::stage("rev0".into(), vec![approved.clone()])
            .unwrap()
            .ratify(record(
                "dj",
                vec![(approved.key(), ReviewDecision::Approve)],
            ))
            .unwrap();

        // A separate changeset, properly ratified on its own terms.
        let other = Changeset::stage("rev0".into(), vec![smuggled.clone()])
            .unwrap()
            .ratify(record(
                "dj",
                vec![(smuggled.key(), ReviewDecision::Approve)],
            ))
            .unwrap();

        // Now swap: sign the second changeset with the first one's record.
        assert_eq!(
            other
                .changeset
                .sign(honest.ratification.clone())
                .unwrap_err(),
            WorldError::UnreviewedCandidate(smuggled.key()),
            "an edge nobody approved must not ride in on someone else's signature"
        );

        // Its own record still works, so the check rejects the swap and not
        // every signature.
        assert!(honest.changeset.sign(honest.ratification).is_ok());
    }
}
