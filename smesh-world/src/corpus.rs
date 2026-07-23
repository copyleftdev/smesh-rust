//! Meridian Mutual: the instrumented showcase corpus.
//!
//! The gold graph is authored first; documents are rendered from it with a
//! planted-defect manifest. The answer key is exact by construction, so every
//! mesh run is scored — and the scorecard is the permanent regression gate.

use crate::candidate::{CandidateEdge, ProvenanceClass};
use crate::ontology::EdgeKind;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GoldEdge {
    pub subject: String,
    pub kind: EdgeKind,
    pub object: String,
}

impl GoldEdge {
    pub fn key(&self) -> String {
        format!("{}|{:?}|{}", self.subject, self.kind, self.object)
    }
}

/// Each planted defect targets a specific subsystem of the mesh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlantedDefect {
    /// Same term, different meaning per department — Lexicon scoping.
    TermCollision {
        term: String,
        departments: Vec<String>,
    },
    /// Two departments with directly contradictory rules — Sentinel → red lane.
    CrossDepartmentContradiction { edge_a: GoldEdge, edge_b: GoldEdge },
    /// Policy PDF superseded by a later email memo — cross-format supersession.
    CrossFormatSupersession {
        superseded: GoldEdge,
        superseding: GoldEdge,
    },
    /// Same policy in two formats, slightly divergent — consensus mechanics.
    DivergentDuplicate { canonical: GoldEdge },
    /// Rule existing only in an email thread — tribal-knowledge analog.
    EmailOnlyRule { edge: GoldEdge },
    /// A question the corpus deliberately never answers — confabulation trap.
    NegativeSpace { question: String },
    /// Attached vendor policy that must not be ingested as Meridian's.
    VendorAttachment { vendor: String },
    /// Generic-corp boilerplate contradicting Meridian's actual quirky rule.
    BoilerplateTrap { meridian_rule: GoldEdge },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DefectManifest {
    pub defects: Vec<PlantedDefect>,
}

impl DefectManifest {
    pub fn planted_contradictions(&self) -> usize {
        self.defects
            .iter()
            .filter(|d| {
                matches!(
                    d,
                    PlantedDefect::CrossDepartmentContradiction { .. }
                        | PlantedDefect::BoilerplateTrap { .. }
                )
            })
            .count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldGraph {
    pub edges: Vec<GoldEdge>,
}

/// The headline numbers. Confabulation rate targets zero — an edge asserted
/// in no document and carrying no human signature is the founding failure
/// mode this whole system exists to eliminate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scorecard {
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub confabulated: usize,
    pub contradictions_planted: usize,
    pub contradictions_caught: usize,
}

impl Scorecard {
    pub fn evaluate(
        gold: &GoldGraph,
        observed: &[CandidateEdge],
        manifest: &DefectManifest,
        contradictions_caught: usize,
    ) -> Self {
        let gold_keys: BTreeSet<String> = gold.edges.iter().map(GoldEdge::key).collect();
        let observed_keys: BTreeSet<String> = observed.iter().map(CandidateEdge::key).collect();

        let true_positives = observed_keys.intersection(&gold_keys).count();
        let false_positives = observed_keys.difference(&gold_keys).count();
        let false_negatives = gold_keys.difference(&observed_keys).count();
        let confabulated = observed
            .iter()
            .filter(|e| {
                !gold_keys.contains(&e.key())
                    && match &e.provenance {
                        ProvenanceClass::CorpusDerived { citations } => citations.is_empty(),
                        ProvenanceClass::HumanAttested { .. } => false,
                    }
            })
            .count();

        Self {
            true_positives,
            false_positives,
            false_negatives,
            confabulated,
            contradictions_planted: manifest.planted_contradictions(),
            contradictions_caught,
        }
    }

    pub fn precision(&self) -> f64 {
        ratio(
            self.true_positives,
            self.true_positives + self.false_positives,
        )
    }

    pub fn recall(&self) -> f64 {
        ratio(
            self.true_positives,
            self.true_positives + self.false_negatives,
        )
    }

    pub fn contradiction_detection_rate(&self) -> f64 {
        ratio(self.contradictions_caught, self.contradictions_planted)
    }

    pub fn passes_gate(&self) -> bool {
        self.confabulated == 0 && self.contradiction_detection_rate() >= 1.0
    }
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        1.0
    } else {
        num as f64 / den as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::Citation;
    use crate::cdm::{CdmDocument, CdmSpan, DocMetadata, SourceFormat};
    use crate::role::WorldRole;

    fn gold() -> GoldGraph {
        GoldGraph {
            edges: vec![
                GoldEdge {
                    subject: "Claims".into(),
                    kind: EdgeKind::ReportsTo,
                    object: "Ops".into(),
                },
                GoldEdge {
                    subject: "Finance".into(),
                    kind: EdgeKind::Owns,
                    object: "Ledger".into(),
                },
            ],
        }
    }

    fn manifest() -> DefectManifest {
        DefectManifest {
            defects: vec![
                PlantedDefect::CrossDepartmentContradiction {
                    edge_a: gold().edges[0].clone(),
                    edge_b: gold().edges[1].clone(),
                },
                PlantedDefect::NegativeSpace {
                    question: "What is the remote-work policy?".into(),
                },
            ],
        }
    }

    fn observed(subject: &str, kind: EdgeKind, object: &str) -> CandidateEdge {
        let doc = CdmDocument::ingest(
            b"gold-rendered",
            SourceFormat::Markdown,
            "Claims reports to Ops.".to_owned(),
            vec![],
            DocMetadata::default(),
        );
        CandidateEdge::emit(
            WorldRole::Structure,
            subject.into(),
            kind,
            object.into(),
            vec![Citation::grounded(&doc, CdmSpan::new(0, 6)).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn scorecard_computes_precision_and_recall() {
        let hits = vec![
            observed("Claims", EdgeKind::ReportsTo, "Ops"),
            observed("HR", EdgeKind::Owns, "Handbook"),
        ];
        let card = Scorecard::evaluate(&gold(), &hits, &manifest(), 1);
        assert_eq!(card.true_positives, 1);
        assert_eq!(card.false_positives, 1);
        assert_eq!(card.false_negatives, 1);
        assert_eq!(card.precision(), 0.5);
        assert_eq!(card.recall(), 0.5);
        assert_eq!(card.contradiction_detection_rate(), 1.0);
    }

    #[test]
    fn cited_false_positives_are_wrong_but_not_confabulated() {
        let hits = vec![observed("HR", EdgeKind::Owns, "Handbook")];
        let card = Scorecard::evaluate(&gold(), &hits, &manifest(), 1);
        assert_eq!(card.false_positives, 1);
        assert_eq!(card.confabulated, 0);
        assert!(card.passes_gate());
    }

    #[test]
    fn gate_fails_when_planted_contradictions_slip_through() {
        let hits = vec![observed("Claims", EdgeKind::ReportsTo, "Ops")];
        let card = Scorecard::evaluate(&gold(), &hits, &manifest(), 0);
        assert!(!card.passes_gate());
    }

    #[test]
    fn manifest_counts_planted_contradictions() {
        assert_eq!(manifest().planted_contradictions(), 1);
    }
}
