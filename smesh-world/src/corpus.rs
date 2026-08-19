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

    /// Semantic match against an observed candidate: same kind class,
    /// alias-aware subject, canonicalized object with containment tolerance.
    /// Exact-key equality measured naming agreement, not truth — the first
    /// live run scored `Claims Requires 45-day filing window` as both a
    /// false positive and a false negative against `Claims GovernedBy
    /// 45-day filing window`.
    pub fn matches(&self, observed: &CandidateEdge, aliases: &AliasTable) -> bool {
        kind_class(self.kind) == kind_class(observed.kind)
            && aliases.same_subject(&self.subject, &observed.subject)
            && objects_match(&self.object, &observed.object)
    }
}

/// Groups of edge kinds that assert the same class of fact. Extractors
/// legitimately disagree about `GovernedBy` vs `Requires`; the scorecard
/// should not.
fn kind_class(kind: EdgeKind) -> u8 {
    match kind {
        EdgeKind::GovernedBy | EdgeKind::Requires | EdgeKind::Triggers => 0,
        EdgeKind::DefinesTerm => 1,
        EdgeKind::ScopedTo | EdgeKind::MemberOf => 2,
        EdgeKind::Owns | EdgeKind::Operates => 3,
        EdgeKind::ReportsTo => 4,
        EdgeKind::Precedes => 5,
        EdgeKind::Supersedes => 6,
    }
}

/// Lowercased, punctuation-free, whitespace-collapsed comparison form.
pub fn canon(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    for c in s.chars() {
        if c.is_alphanumeric() {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.extend(c.to_lowercase());
        } else {
            pending_space = true;
        }
    }
    out
}

fn objects_match(gold: &str, observed: &str) -> bool {
    let (g, o) = (canon(gold), canon(observed));
    if g == o {
        return true;
    }
    let shorter = g.len().min(o.len());
    shorter >= 8 && (g.contains(&o) || o.contains(&g))
}

/// Declared name equivalences (e.g. `HR` ↔ `Human Resources`) so the gold
/// graph is not hostage to one arbitrary spelling.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasTable {
    pub groups: Vec<Vec<String>>,
}

impl AliasTable {
    pub fn same_subject(&self, a: &str, b: &str) -> bool {
        let (ca, cb) = (canon(a), canon(b));
        if ca == cb {
            return true;
        }
        self.groups.iter().any(|group| {
            let canon_group: Vec<String> = group.iter().map(|g| canon(g)).collect();
            canon_group.contains(&ca) && canon_group.contains(&cb)
        })
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
    #[serde(default)]
    pub aliases: AliasTable,
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
        let mut deduped: Vec<&CandidateEdge> = Vec::new();
        let mut seen: BTreeSet<(u8, String, String)> = BTreeSet::new();
        for e in observed {
            if seen.insert((kind_class(e.kind), canon(&e.subject), canon(&e.object))) {
                deduped.push(e);
            }
        }

        let true_positives = gold
            .edges
            .iter()
            .filter(|g| deduped.iter().any(|o| g.matches(o, &gold.aliases)))
            .count();
        let false_negatives = gold.edges.len() - true_positives;
        let false_positives = deduped
            .iter()
            .filter(|o| !gold.edges.iter().any(|g| g.matches(o, &gold.aliases)))
            .count();
        let confabulated = deduped
            .iter()
            .filter(|e| {
                !gold.edges.iter().any(|g| g.matches(e, &gold.aliases))
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
                    object: "General Ledger".into(),
                },
            ],
            aliases: AliasTable {
                groups: vec![vec!["Claims".into(), "Claims department".into()]],
            },
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

    #[test]
    fn canonical_matching_forgives_naming_noise_not_substance() {
        let hits = vec![
            observed("Claims department", EdgeKind::ReportsTo, "Ops"),
            observed("finance", EdgeKind::Operates, "the General Ledger"),
        ];
        let card = Scorecard::evaluate(&gold(), &hits, &manifest(), 1);
        assert_eq!(
            card.true_positives, 2,
            "alias subject, kind-class sibling, and object containment all match"
        );
        assert_eq!(card.false_positives, 0);

        let miss = vec![observed("Claims", EdgeKind::ReportsTo, "Legal")];
        let card = Scorecard::evaluate(&gold(), &miss, &manifest(), 1);
        assert_eq!(card.true_positives, 0, "different substance stays a miss");
    }

    #[test]
    fn case_variant_duplicates_collapse_before_scoring() {
        let hits = vec![
            observed("Claims", EdgeKind::ReportsTo, "Ops"),
            observed("claims", EdgeKind::ReportsTo, "OPS"),
        ];
        let card = Scorecard::evaluate(&gold(), &hits, &manifest(), 1);
        assert_eq!(card.true_positives, 1);
        assert_eq!(card.false_positives, 0);
    }

    #[test]
    fn canon_normalizes_case_punctuation_and_whitespace() {
        assert_eq!(canon("IT-Security"), "it security");
        assert_eq!(canon("  Prior   Authorization. "), "prior authorization");
        assert_eq!(canon("45-day filing window"), "45 day filing window");
    }
}
