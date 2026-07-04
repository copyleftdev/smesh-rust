//! The adjudication mesh: policy agents evaluate a claim against the signed
//! AION rules, their findings reach consensus in a shared signal field, and the
//! AION `ai_constraints` act as a governance gate on the final decision.

use smesh_core::{Field, Signal, SignalType};

use super::claim::Claim;
use super::policy::Policy;

/// One agent's disposition on a claim, in ascending severity.
#[derive(Debug, Clone, PartialEq)]
pub enum Disposition {
    Approve,
    Flag(String),
    Pend(String),
    Deny(String),
}

impl Disposition {
    fn severity(&self) -> u8 {
        match self {
            Disposition::Approve => 0,
            Disposition::Flag(_) => 1,
            Disposition::Pend(_) => 2,
            Disposition::Deny(_) => 3,
        }
    }
    fn tag(&self) -> &'static str {
        match self {
            Disposition::Approve => "approve",
            Disposition::Flag(_) => "flag",
            Disposition::Pend(_) => "pend",
            Disposition::Deny(_) => "deny",
        }
    }
    pub fn reason(&self) -> &str {
        match self {
            Disposition::Approve => "approved",
            Disposition::Flag(r) | Disposition::Pend(r) | Disposition::Deny(r) => r,
        }
    }
}

/// A single agent finding, with the signed rule it cites.
#[derive(Debug, Clone)]
pub struct Finding {
    pub agent: String,
    pub disposition: Disposition,
    pub citation: String,
    pub detail: String,
}

/// The final decision after the governance gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Approve,
    PendPriorAuth,
    PendStepTherapy,
    EscalateHuman,
    DenyDraft,
}

impl Decision {
    pub fn label(&self) -> &'static str {
        match self {
            Decision::Approve => "APPROVE",
            Decision::PendPriorAuth => "PEND · PRIOR AUTH",
            Decision::PendStepTherapy => "PEND · STEP THERAPY",
            Decision::EscalateHuman => "ESCALATE · HUMAN REVIEW",
            Decision::DenyDraft => "DENY (DRAFT)",
        }
    }
    pub fn code(&self) -> &'static str {
        match self {
            Decision::Approve => "approve",
            Decision::PendPriorAuth => "pend_pa",
            Decision::PendStepTherapy => "pend_step",
            Decision::EscalateHuman => "escalate",
            Decision::DenyDraft => "deny_draft",
        }
    }
}

/// A full adjudication result with audit trail and provenance.
#[derive(Debug, Clone)]
pub struct Adjudication {
    pub claim: Claim,
    pub policy_name: String,
    pub regulation: String,
    pub aion_file: String,
    pub author_id: u64,
    pub verified: bool,
    pub sha256: String,
    pub findings: Vec<Finding>,
    pub decision: Decision,
    pub rationale: String,
    /// The AION `ai_constraints` rule that governed the final decision, if any.
    pub gating_constraint: Option<String>,
    /// Per-disposition agent counts, from the shared signal field.
    pub consensus: Vec<(String, u32)>,
}

/// Adjudicate one claim against the matching signed policy.
pub fn adjudicate(claim: &Claim, policies: &[Policy]) -> Adjudication {
    let policy = policies
        .iter()
        .find(|p| p.key == claim.plan.policy_key())
        .or_else(|| policies.first())
        .expect("at least one policy loaded");

    let findings = run_agents(claim, policy);

    // Fuse the findings through a shared signal field: same-disposition findings
    // reinforce, so consensus per disposition is the field's reinforcement count.
    let mut field = Field::new();
    for f in &findings {
        let sig = Signal::builder(SignalType::Coordination)
            .payload(f.disposition.tag().as_bytes().to_vec())
            .confidence(0.9)
            .origin("adjudication-mesh")
            .build();
        field.emit_anonymous(sig);
    }
    let mut consensus: Vec<(String, u32)> = field
        .active_signals()
        .filter_map(|s| {
            let tag = std::str::from_utf8(&s.payload).ok()?.to_string();
            Some((tag, s.reinforcement_count + 1))
        })
        .collect();
    consensus.sort_by_key(|(_, n)| std::cmp::Reverse(*n));

    // The most severe finding is the mesh's tentative decision.
    let tentative = findings
        .iter()
        .max_by_key(|f| f.disposition.severity())
        .map(|f| f.disposition.clone())
        .unwrap_or(Disposition::Approve);

    // Governance gate: apply the AION ai_constraints to the tentative decision.
    let (decision, rationale, gating_constraint) = govern(&tentative, claim, policy);

    Adjudication {
        claim: claim.clone(),
        policy_name: policy.name.clone(),
        regulation: policy.regulation.clone(),
        aion_file: policy.provenance.aion_file.clone(),
        author_id: policy.provenance.author_id,
        verified: policy.provenance.verified,
        sha256: policy.provenance.sha256.clone(),
        findings,
        decision,
        rationale,
        gating_constraint,
        consensus,
    }
}

fn run_agents(claim: &Claim, policy: &Policy) -> Vec<Finding> {
    let mut out = Vec::new();
    let tier = policy.tier(claim.tier);

    // 1. Formulary agent — placement.
    let tier_label = tier.map(|t| t.label.clone()).unwrap_or_else(|| "unlisted".into());
    out.push(Finding {
        agent: "Formulary".into(),
        disposition: Disposition::Approve,
        citation: format!("{} · tier {} ({})", policy.name, claim.tier, tier_label),
        detail: format!("{} listed on tier {}", claim.drug, claim.tier),
    });

    // 2. Prior-Authorization agent.
    let pa_default = tier.map(|t| t.prior_auth_default).unwrap_or(false);
    if pa_default && !claim.prior_auth_obtained {
        out.push(Finding {
            agent: "Prior-Auth".into(),
            disposition: Disposition::Pend("Prior authorization required before dispensing".into()),
            citation: format!("{} · UM.prior_authorization (tier {} default)", policy.name, claim.tier),
            detail: "Tier requires PA; none on file".into(),
        });
    } else {
        out.push(Finding {
            agent: "Prior-Auth".into(),
            disposition: Disposition::Approve,
            citation: format!("{} · UM.prior_authorization", policy.name),
            detail: if claim.prior_auth_obtained { "PA on file".into() } else { "No PA required".into() },
        });
    }

    // 3. Step-Therapy agent — protected classes are exempt (prohibited).
    let step_default = tier.map(|t| t.step_therapy_default).unwrap_or(false);
    let protected = policy.protected_class_for_atc(&claim.atc);
    if step_default && !claim.step_therapy_completed {
        if let Some(pc) = protected {
            out.push(Finding {
                agent: "Step-Therapy".into(),
                disposition: Disposition::Flag(format!(
                    "Step therapy prohibited for protected class '{}' — access protected",
                    pc.name
                )),
                citation: format!("{} · step_therapy.prohibited_classes ({})", policy.name, pc.id),
                detail: "42 CFR §423.120(b)(2)(v): may not apply step therapy to protected classes".into(),
            });
        } else if claim.patient_stable_on_drug {
            out.push(Finding {
                agent: "Step-Therapy".into(),
                disposition: Disposition::Flag("Grandfathered — stable patient exempt".into()),
                citation: format!("{} · step_therapy.grandfathering", policy.name),
                detail: "Stable patient on non-preferred drug; step therapy waived".into(),
            });
        } else {
            out.push(Finding {
                agent: "Step-Therapy".into(),
                disposition: Disposition::Pend("Step therapy must be completed first".into()),
                citation: format!("{} · UM.step_therapy (tier {} default)", policy.name, claim.tier),
                detail: "Preferred alternative not yet tried".into(),
            });
        }
    }

    // 4. Coverage / medical-necessity agent.
    let off_label = claim.diagnosis.to_ascii_lowercase().contains("off-label");
    if off_label || claim.cost_usd > 1_000_000.0 {
        out.push(Finding {
            agent: "Coverage".into(),
            disposition: Disposition::Deny(
                "Non-covered indication / exceeds medical-necessity criteria".into(),
            ),
            citation: format!("{} · coverage_determination", policy.name),
            detail: "Clinical determination required for this indication".into(),
        });
    } else if claim.is_specialty() {
        out.push(Finding {
            agent: "Coverage".into(),
            disposition: Disposition::Flag("Specialty drug — verify medical necessity".into()),
            citation: format!("{} · specialty ({:.0} USD)", policy.name, claim.cost_usd),
            detail: "Specialty-tier utilization review".into(),
        });
    }

    out
}

/// Apply the AION governance layer. A tentative DENY can never be autonomous.
fn govern(
    tentative: &Disposition,
    _claim: &Claim,
    policy: &Policy,
) -> (Decision, String, Option<String>) {
    match tentative {
        Disposition::Deny(reason) => {
            let prohibits_auto = policy.ai.prohibits("autonomous_coverage_denial");
            let needs_human = policy.ai.requires_human("coverage_determination_final_decision");
            if prohibits_auto && needs_human {
                (
                    Decision::EscalateHuman,
                    format!(
                        "Mesh consensus is DENY ({reason}); AION policy forbids autonomous denial → escalated to human pharmacist for the final coverage determination."
                    ),
                    Some(
                        "prohibited: autonomous_coverage_denial · human_oversight_required: coverage_determination_final_decision"
                            .into(),
                    ),
                )
            } else {
                (
                    Decision::DenyDraft,
                    format!("Denial drafted ({reason}) — requires human sign-off."),
                    Some("denial requires human confirmation".into()),
                )
            }
        }
        Disposition::Pend(reason) => {
            let kind = if reason.to_ascii_lowercase().contains("step") {
                Decision::PendStepTherapy
            } else {
                Decision::PendPriorAuth
            };
            (kind, reason.clone(), None)
        }
        Disposition::Flag(note) => (
            Decision::Approve,
            format!("Approved — all applicable rules satisfied. Note: {note}"),
            None,
        ),
        Disposition::Approve => (
            Decision::Approve,
            "Approved — all applicable rules satisfied.".into(),
            None,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::super::claim::sample_claims;
    use super::super::policy::load_all;
    use super::*;

    fn adj_for(id: &str) -> Adjudication {
        let policies = load_all();
        let claim = sample_claims().into_iter().find(|c| c.id == id).unwrap();
        adjudicate(&claim, &policies)
    }

    #[test]
    fn test_clean_generic_approves() {
        assert_eq!(adj_for("RX-1001").decision, Decision::Approve);
    }

    #[test]
    fn test_protected_class_step_therapy_is_not_blocked() {
        // Antidepressant on a step-therapy tier with step NOT done must still
        // approve — step therapy is prohibited for protected classes.
        let a = adj_for("RX-1002");
        assert_eq!(a.decision, Decision::Approve);
        assert!(a
            .findings
            .iter()
            .any(|f| f.detail.contains("protected classes")));
    }

    #[test]
    fn test_prior_auth_pends() {
        assert_eq!(adj_for("RX-1003").decision, Decision::PendPriorAuth);
    }

    #[test]
    fn test_autonomous_denial_is_escalated() {
        // The off-label $2.1M claim: mesh says deny, AION forbids autonomous
        // denial → must escalate to a human, never auto-deny.
        let a = adj_for("RX-1005");
        assert_eq!(a.decision, Decision::EscalateHuman);
        assert!(a.gating_constraint.is_some());
        assert!(a
            .gating_constraint
            .as_ref()
            .unwrap()
            .contains("autonomous_coverage_denial"));
    }
}
