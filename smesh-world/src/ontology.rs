//! The typed edge vocabulary — rev 0 of the world model.
//!
//! AION's four edge kinds (semantic/causal/temporal/provenance) are the
//! transport layer; the org world model speaks a typed vocabulary on top.
//! Consensus thresholds are per-edge-kind, not global.

use serde::{Deserialize, Serialize};

/// AION transport-layer edge kind a typed edge lowers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportKind {
    Semantic,
    Causal,
    Temporal,
    Provenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    DefinesTerm,
    GovernedBy,
    ReportsTo,
    Owns,
    Operates,
    MemberOf,
    Precedes,
    Requires,
    Triggers,
    Supersedes,
    ScopedTo,
}

/// Structural invariants the Contradiction Sentinel enforces mechanically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralConstraint {
    /// At most one such edge per subject per timeslice (e.g. `reports_to`).
    UniquePerSubjectPerTimeslice,
    /// The relation must form no cycles (e.g. `supersedes`).
    Acyclic,
}

/// How much independent agreement an edge kind must earn before the Curator
/// will include it in a changeset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsensusPolicy {
    pub min_corroborations: u32,
    pub max_refutations: u32,
}

impl EdgeKind {
    pub const ALL: [EdgeKind; 11] = [
        EdgeKind::DefinesTerm,
        EdgeKind::GovernedBy,
        EdgeKind::ReportsTo,
        EdgeKind::Owns,
        EdgeKind::Operates,
        EdgeKind::MemberOf,
        EdgeKind::Precedes,
        EdgeKind::Requires,
        EdgeKind::Triggers,
        EdgeKind::Supersedes,
        EdgeKind::ScopedTo,
    ];

    pub fn transport(&self) -> TransportKind {
        match self {
            EdgeKind::DefinesTerm | EdgeKind::MemberOf | EdgeKind::ScopedTo => {
                TransportKind::Semantic
            }
            EdgeKind::GovernedBy
            | EdgeKind::ReportsTo
            | EdgeKind::Owns
            | EdgeKind::Operates
            | EdgeKind::Requires
            | EdgeKind::Triggers => TransportKind::Causal,
            EdgeKind::Precedes => TransportKind::Temporal,
            EdgeKind::Supersedes => TransportKind::Provenance,
        }
    }

    /// Normative and identity-bearing edges demand more agreement than
    /// vocabulary edges.
    pub fn consensus_policy(&self) -> ConsensusPolicy {
        let min_corroborations = match self {
            EdgeKind::DefinesTerm | EdgeKind::MemberOf | EdgeKind::ScopedTo => 1,
            EdgeKind::ReportsTo | EdgeKind::Owns | EdgeKind::Operates | EdgeKind::Precedes => 2,
            EdgeKind::GovernedBy
            | EdgeKind::Requires
            | EdgeKind::Triggers
            | EdgeKind::Supersedes => 2,
        };
        ConsensusPolicy {
            min_corroborations,
            max_refutations: 0,
        }
    }

    pub fn structural_constraint(&self) -> Option<StructuralConstraint> {
        match self {
            EdgeKind::ReportsTo => Some(StructuralConstraint::UniquePerSubjectPerTimeslice),
            EdgeKind::Supersedes | EdgeKind::Precedes => Some(StructuralConstraint::Acyclic),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_edge_kind_lowers_to_a_transport() {
        for kind in EdgeKind::ALL {
            let _ = kind.transport();
        }
    }

    #[test]
    fn normative_edges_demand_more_agreement_than_vocabulary() {
        assert!(
            EdgeKind::GovernedBy.consensus_policy().min_corroborations
                > EdgeKind::DefinesTerm.consensus_policy().min_corroborations
        );
    }

    #[test]
    fn no_edge_kind_tolerates_unresolved_refutations() {
        for kind in EdgeKind::ALL {
            assert_eq!(kind.consensus_policy().max_refutations, 0);
        }
    }

    #[test]
    fn supersession_is_acyclic_and_reporting_is_unique() {
        assert_eq!(
            EdgeKind::Supersedes.structural_constraint(),
            Some(StructuralConstraint::Acyclic)
        );
        assert_eq!(
            EdgeKind::ReportsTo.structural_constraint(),
            Some(StructuralConstraint::UniquePerSubjectPerTimeslice)
        );
    }
}
