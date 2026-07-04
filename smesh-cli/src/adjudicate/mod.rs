//! Pharmaceutical claim adjudication driven by signed AION formulary policies.
//!
//! Incoming claims are decided by a mesh of deterministic policy agents that
//! cite the exact signed rules they apply; the AION `ai_constraints` govern what
//! the mesh may decide autonomously. Every decision carries a full audit trail
//! and the cryptographic provenance of the policy that drove it.

pub mod claim;
pub mod engine;
pub mod policy;
pub mod report;

/// Adjudicate the built-in sample batch against the signed policies.
pub fn adjudicate_samples() -> Vec<engine::Adjudication> {
    let policies = policy::load_all();
    claim::sample_claims()
        .iter()
        .map(|c| engine::adjudicate(c, &policies))
        .collect()
}
