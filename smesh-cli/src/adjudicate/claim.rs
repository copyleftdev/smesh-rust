//! Incoming pharmaceutical claim model + a sample batch that exercises each
//! decision path.

use serde::{Deserialize, Serialize};

/// Which plan (and therefore which signed policy) adjudicates the claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Plan {
    PartD,
    Commercial,
    Medicaid,
}

impl Plan {
    pub fn policy_key(&self) -> &'static str {
        match self {
            Plan::PartD => "cms_part_d",
            Plan::Commercial => "commercial_pbm",
            Plan::Medicaid => "medicaid_pdl",
        }
    }
}

/// An incoming pharmacy claim to adjudicate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub id: String,
    pub drug: String,
    /// ATC classification code (e.g. "N06AB" → antidepressant).
    pub atc: String,
    pub plan: Plan,
    /// Formulary tier the drug sits on (1..=6).
    pub tier: u32,
    pub quantity: u32,
    pub days_supply: u32,
    pub cost_usd: f64,
    pub diagnosis: String,
    pub prior_auth_obtained: bool,
    pub step_therapy_completed: bool,
    /// Patient already stable on this drug (grandfathering exemption).
    pub patient_stable_on_drug: bool,
}

impl Claim {
    pub fn is_specialty(&self) -> bool {
        self.tier >= 5
    }
}

/// A curated batch of claims, each hitting a different rule + governance path.
pub fn sample_claims() -> Vec<Claim> {
    vec![
        // Clean generic — straight approve.
        Claim {
            id: "RX-1001".into(),
            drug: "Metformin 500mg".into(),
            atc: "A10BA02".into(),
            plan: Plan::PartD,
            tier: 1,
            quantity: 60,
            days_supply: 30,
            cost_usd: 4.0,
            diagnosis: "Type 2 diabetes".into(),
            prior_auth_obtained: false,
            step_therapy_completed: false,
            patient_stable_on_drug: true,
        },
        // Protected class (antidepressant) on a step-therapy tier, step NOT done.
        // Step therapy is PROHIBITED for protected classes → must not block.
        Claim {
            id: "RX-1002".into(),
            drug: "Sertraline 100mg".into(),
            atc: "N06AB06".into(),
            plan: Plan::PartD,
            tier: 3,
            quantity: 30,
            days_supply: 30,
            cost_usd: 12.0,
            diagnosis: "Major depressive disorder".into(),
            prior_auth_obtained: false,
            step_therapy_completed: false,
            patient_stable_on_drug: false,
        },
        // Non-preferred brand needing prior auth (tier 4 default), not obtained.
        Claim {
            id: "RX-1003".into(),
            drug: "Jardiance 25mg".into(),
            atc: "A10BK03".into(),
            plan: Plan::PartD,
            tier: 4,
            quantity: 30,
            days_supply: 30,
            cost_usd: 620.0,
            diagnosis: "Type 2 diabetes".into(),
            prior_auth_obtained: false,
            step_therapy_completed: true,
            patient_stable_on_drug: false,
        },
        // Specialty tier-5 drug, high cost, no prior auth → PA required.
        Claim {
            id: "RX-1004".into(),
            drug: "Humira 40mg/0.4mL".into(),
            atc: "L04AB04".into(),
            plan: Plan::Commercial,
            tier: 5,
            quantity: 2,
            days_supply: 28,
            cost_usd: 6900.0,
            diagnosis: "Rheumatoid arthritis".into(),
            prior_auth_obtained: false,
            step_therapy_completed: true,
            patient_stable_on_drug: false,
        },
        // Non-formulary / high-cost specialty the mesh would deny — but AION
        // prohibits autonomous denial → must ESCALATE to a human.
        Claim {
            id: "RX-1005".into(),
            drug: "Zolgensma (off-label)".into(),
            atc: "M09AX09".into(),
            plan: Plan::PartD,
            tier: 5,
            quantity: 1,
            days_supply: 1,
            cost_usd: 2100000.0,
            diagnosis: "Off-label indication".into(),
            prior_auth_obtained: false,
            step_therapy_completed: false,
            patient_stable_on_drug: false,
        },
        // Stable patient on a non-preferred step-therapy drug → grandfathered.
        Claim {
            id: "RX-1006".into(),
            drug: "Atorvastatin 40mg".into(),
            atc: "C10AA05".into(),
            plan: Plan::Medicaid,
            tier: 3,
            quantity: 30,
            days_supply: 30,
            cost_usd: 9.0,
            diagnosis: "Hyperlipidemia".into(),
            prior_auth_obtained: false,
            step_therapy_completed: false,
            patient_stable_on_drug: true,
        },
    ]
}
