//! Load the signed AION formulary policies and their verified provenance.
//!
//! The rule payloads are the exact YAML sealed inside the corresponding
//! `.aion` files in the `formulary-aion` project; `provenance.json` records the
//! real signature-verification result (author id, verified, content hash) so
//! every decision can cite which signed policy — and which approved rule —
//! drove it.

use serde::Deserialize;
use serde_yaml::Value;

// The verified policy payloads, bundled at compile time.
const CMS_PART_D: &str = include_str!("policies/cms_part_d.yaml");
const COMMERCIAL_PBM: &str = include_str!("policies/commercial_pbm.yaml");
const MEDICAID_PDL: &str = include_str!("policies/medicaid_pdl.yaml");
const FDA_21CFR: &str = include_str!("policies/fda_21cfr.yaml");
const PROVENANCE: &str = include_str!("policies/provenance.json");

/// Cryptographic provenance for one signed `.aion` policy.
#[derive(Debug, Clone, Deserialize)]
pub struct Provenance {
    pub policy: String,
    pub name: String,
    pub aion_file: String,
    pub author_id: u64,
    pub verified: bool,
    pub sha256: String,
}

/// A formulary tier and its default utilization-management flags.
#[derive(Debug, Clone)]
pub struct Tier {
    pub tier: u32,
    pub label: String,
    pub prior_auth_default: bool,
    pub step_therapy_default: bool,
    #[allow(dead_code)] // parsed policy metadata, retained for completeness
    pub specialty_threshold_usd: Option<f64>,
}

/// A CMS protected drug class (all-or-substantially-all coverage; step therapy
/// prohibited).
#[derive(Debug, Clone)]
pub struct ProtectedClass {
    pub id: String,
    pub name: String,
    pub atc_codes: Vec<String>,
}

/// The AION `ai_constraints` block — what the mesh is permitted to do.
#[derive(Debug, Clone, Default)]
pub struct AiConstraints {
    #[allow(dead_code)] // retained for audit display of what the mesh *may* do
    pub allowed: Vec<String>,
    pub prohibited: Vec<String>,
    pub human_oversight_required_for: Vec<String>,
}

impl AiConstraints {
    pub fn prohibits(&self, op: &str) -> bool {
        self.prohibited.iter().any(|o| o == op)
    }
    pub fn requires_human(&self, op: &str) -> bool {
        self.human_oversight_required_for.iter().any(|o| o == op)
    }
}

/// A parsed, signed formulary policy.
#[derive(Debug, Clone)]
pub struct Policy {
    pub key: String,
    pub name: String,
    pub regulation: String,
    pub tiers: Vec<Tier>,
    pub protected_classes: Vec<ProtectedClass>,
    #[allow(dead_code)] // parsed from the policy; protection is enforced via protected_classes
    pub step_therapy_prohibited_classes: Vec<String>,
    pub ai: AiConstraints,
    pub provenance: Provenance,
}

impl Policy {
    /// Find the protected class (if any) matching an ATC code prefix.
    pub fn protected_class_for_atc(&self, atc: &str) -> Option<&ProtectedClass> {
        self.protected_classes
            .iter()
            .find(|c| c.atc_codes.iter().any(|code| atc.starts_with(code)))
    }

    pub fn tier(&self, n: u32) -> Option<&Tier> {
        self.tiers.iter().find(|t| t.tier == n)
    }
}

/// Load all bundled policies with their provenance.
pub fn load_all() -> Vec<Policy> {
    let provs: Vec<Provenance> = serde_json::from_str(PROVENANCE).unwrap_or_default();
    let sources = [
        ("cms_part_d", CMS_PART_D),
        ("commercial_pbm", COMMERCIAL_PBM),
        ("medicaid_pdl", MEDICAID_PDL),
        ("fda_21cfr", FDA_21CFR),
    ];
    sources
        .iter()
        .filter_map(|(key, yaml)| {
            let prov = provs.iter().find(|p| p.policy == *key)?.clone();
            parse_policy(key, yaml, prov)
        })
        .collect()
}

fn parse_policy(key: &str, yaml: &str, provenance: Provenance) -> Option<Policy> {
    let v: Value = serde_yaml::from_str(yaml).ok()?;

    let regulation = v
        .get("metadata")
        .and_then(|m| m.get("regulation").or_else(|| m.get("organization")))
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();

    let tiers = v
        .get("formulary_tiers")
        .and_then(|t| t.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|t| {
                    Some(Tier {
                        tier: t.get("tier")?.as_u64()? as u32,
                        label: t
                            .get("label")
                            .and_then(|l| l.as_str())
                            .unwrap_or("")
                            .to_string(),
                        prior_auth_default: t
                            .get("prior_auth_default")
                            .and_then(|b| b.as_bool())
                            .unwrap_or(false),
                        step_therapy_default: t
                            .get("step_therapy_default")
                            .and_then(|b| b.as_bool())
                            .unwrap_or(false),
                        specialty_threshold_usd: t
                            .get("specialty_threshold_usd")
                            .and_then(|x| x.as_f64()),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let protected_classes = v
        .get("protected_classes")
        .and_then(|c| c.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|c| {
                    Some(ProtectedClass {
                        id: c.get("id")?.as_str()?.to_string(),
                        name: c.get("name")?.as_str()?.to_string(),
                        atc_codes: c
                            .get("atc_codes")
                            .and_then(|a| a.as_sequence())
                            .map(|s| {
                                s.iter()
                                    .filter_map(|x| x.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let step_therapy_prohibited_classes = v
        .get("utilization_management")
        .and_then(|u| u.get("step_therapy"))
        .and_then(|s| s.get("prohibited_classes"))
        .and_then(|p| p.as_sequence())
        .map(|s| {
            s.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let ai = v
        .get("ai_constraints")
        .map(|a| AiConstraints {
            allowed: str_list(a.get("allowed_operations")),
            prohibited: str_list(a.get("prohibited_operations")),
            human_oversight_required_for: str_list(a.get("human_oversight_required_for")),
        })
        .unwrap_or_default();

    Some(Policy {
        key: key.to_string(),
        name: provenance.name.clone(),
        regulation,
        tiers,
        protected_classes,
        step_therapy_prohibited_classes,
        ai,
        provenance,
    })
}

fn str_list(v: Option<&Value>) -> Vec<String> {
    v.and_then(|x| x.as_sequence())
        .map(|s| {
            s.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policies_load_with_provenance() {
        let policies = load_all();
        assert_eq!(policies.len(), 4);
        for p in &policies {
            assert!(p.provenance.verified, "{} should be verified", p.key);
            assert!(!p.provenance.sha256.is_empty());
        }
    }

    #[test]
    fn test_cms_part_d_rules_parsed() {
        let policies = load_all();
        let cms = policies.iter().find(|p| p.key == "cms_part_d").unwrap();
        // Six protected classes, six tiers, step therapy prohibited for them.
        assert_eq!(cms.protected_classes.len(), 6);
        assert!(cms.tiers.len() >= 5);
        assert!(cms
            .step_therapy_prohibited_classes
            .iter()
            .any(|c| c == "antidepressants"));
        // Tier 5 (specialty) defaults to prior auth.
        assert!(cms.tier(5).unwrap().prior_auth_default);
        // The AION governance layer forbids autonomous denial.
        assert!(cms.ai.prohibits("autonomous_coverage_denial"));
    }

    #[test]
    fn test_protected_class_lookup() {
        let policies = load_all();
        let cms = policies.iter().find(|p| p.key == "cms_part_d").unwrap();
        // N06A → antidepressants (protected).
        assert_eq!(
            cms.protected_class_for_atc("N06AB").map(|c| c.name.as_str()),
            Some("antidepressants")
        );
        // A random ATC is not protected.
        assert!(cms.protected_class_for_atc("A10BA").is_none());
    }
}
