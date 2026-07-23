//! The expert roster: Tiers 0–3, capability sets, model policy.
//!
//! Separation of powers is encoded here and asserted by tests: no role may
//! both emit candidates and issue verdicts; only the Curator writes AION;
//! the Curator is not an LLM.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    Intake,
    Extraction,
    Verification,
    Stewardship,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    CorpusRead,
    GraphRead,
    EmitCandidate,
    Verdict,
    Escalate,
    OntologyMap,
    ProposeEdgeKind,
    AionWrite,
}

/// Which model a role runs, and the diversity constraint that keeps
/// correlated hallucinations from surviving verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelPolicy {
    /// Deterministic code path; no model involved.
    None,
    /// LLM used only to triage ambiguous inputs; fidelity work is code.
    TriageOnly(&'static str),
    Fixed(&'static str),
    /// Must run a model family distinct from every Extraction-tier model.
    DistinctFamilyFromExtraction(&'static str),
}

pub const EXTRACTION_MODEL: &str = "moonshotai/kimi-k3";
pub const VERIFICATION_MODEL: &str = "anthropic/claude-sonnet-4.5";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorldRole {
    /// Tier 0 — any artifact in, CDM out.
    Registrar,
    /// Tier 1 — org verbiage: defined terms, acronyms, canonical names.
    Lexicon,
    /// Tier 1 — normative statements, effective dates, supersession.
    Policy,
    /// Tier 1 — org chart, roles, systems, ownership.
    Structure,
    /// Tier 1 — workflows, lifecycles, temporal ordering.
    Process,
    /// Tier 2 — does the cited span entail the claim?
    GroundingAuditor,
    /// Tier 2 — candidate deltas vs the current world revision.
    ContradictionSentinel,
    /// Tier 3 — owns the edge vocabulary; new kinds only via escalation.
    Ontologist,
    /// Tier 3 — sole AION writer; deliberately not an LLM.
    Curator,
}

impl WorldRole {
    pub const ALL: [WorldRole; 9] = [
        WorldRole::Registrar,
        WorldRole::Lexicon,
        WorldRole::Policy,
        WorldRole::Structure,
        WorldRole::Process,
        WorldRole::GroundingAuditor,
        WorldRole::ContradictionSentinel,
        WorldRole::Ontologist,
        WorldRole::Curator,
    ];

    pub fn tier(&self) -> Tier {
        match self {
            WorldRole::Registrar => Tier::Intake,
            WorldRole::Lexicon | WorldRole::Policy | WorldRole::Structure | WorldRole::Process => {
                Tier::Extraction
            }
            WorldRole::GroundingAuditor | WorldRole::ContradictionSentinel => Tier::Verification,
            WorldRole::Ontologist | WorldRole::Curator => Tier::Stewardship,
        }
    }

    pub fn capabilities(&self) -> &'static [Capability] {
        use Capability::*;
        match self {
            WorldRole::Registrar => &[CorpusRead],
            WorldRole::Lexicon | WorldRole::Policy | WorldRole::Structure | WorldRole::Process => {
                &[CorpusRead, EmitCandidate]
            }
            WorldRole::GroundingAuditor => &[CorpusRead, Verdict],
            WorldRole::ContradictionSentinel => &[GraphRead, Verdict, Escalate],
            WorldRole::Ontologist => &[GraphRead, OntologyMap, ProposeEdgeKind, Escalate],
            WorldRole::Curator => &[GraphRead, AionWrite],
        }
    }

    pub fn has_capability(&self, cap: Capability) -> bool {
        self.capabilities().contains(&cap)
    }

    pub fn model_policy(&self) -> ModelPolicy {
        match self {
            WorldRole::Registrar => ModelPolicy::TriageOnly(EXTRACTION_MODEL),
            WorldRole::Lexicon | WorldRole::Policy | WorldRole::Structure | WorldRole::Process => {
                ModelPolicy::Fixed(EXTRACTION_MODEL)
            }
            WorldRole::GroundingAuditor | WorldRole::ContradictionSentinel => {
                ModelPolicy::DistinctFamilyFromExtraction(VERIFICATION_MODEL)
            }
            WorldRole::Ontologist => ModelPolicy::Fixed(VERIFICATION_MODEL),
            WorldRole::Curator => ModelPolicy::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_role_both_proposes_and_judges() {
        for role in WorldRole::ALL {
            assert!(
                !(role.has_capability(Capability::EmitCandidate)
                    && role.has_capability(Capability::Verdict)),
                "{role:?} violates separation of powers"
            );
        }
    }

    #[test]
    fn only_the_curator_writes_aion() {
        for role in WorldRole::ALL {
            assert_eq!(
                role.has_capability(Capability::AionWrite),
                role == WorldRole::Curator
            );
        }
    }

    #[test]
    fn signing_authority_is_code_not_a_model() {
        assert_eq!(WorldRole::Curator.model_policy(), ModelPolicy::None);
    }

    #[test]
    fn extractors_cannot_touch_the_graph() {
        for role in WorldRole::ALL
            .iter()
            .filter(|r| r.tier() == Tier::Extraction)
        {
            assert!(!role.has_capability(Capability::GraphRead));
            assert!(!role.has_capability(Capability::AionWrite));
        }
    }

    #[test]
    fn verifiers_run_a_distinct_model_family() {
        for role in WorldRole::ALL
            .iter()
            .filter(|r| r.tier() == Tier::Verification)
        {
            match role.model_policy() {
                ModelPolicy::DistinctFamilyFromExtraction(m) => {
                    let extraction_family = EXTRACTION_MODEL.split('/').next().unwrap();
                    assert_ne!(m.split('/').next().unwrap(), extraction_family);
                }
                other => panic!("{role:?} must verify cross-family, got {other:?}"),
            }
        }
    }
}
