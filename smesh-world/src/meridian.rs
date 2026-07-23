//! Meridian Mutual — the instrumented showcase corpus.
//!
//! The gold graph is authored first; every document below is rendered *from*
//! it, so the answer key is exact by construction. Each planted defect
//! targets a specific mesh subsystem (see `WORLD-MODEL.md` §7). Rendering is
//! fully deterministic: same call, same bytes, same `DocId`s.

use crate::corpus::{DefectManifest, GoldEdge, GoldGraph, PlantedDefect};
use crate::intake::pdf_render::render_pdf;
use crate::ontology::EdgeKind;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Artifact {
    pub name: &'static str,
    pub bytes: Vec<u8>,
}

/// Ties a gold edge to the artifact and quote that prove it — the ground
/// truth for the Phase 5 provenance-integrity score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Evidence {
    pub edge_key: String,
    pub artifact: &'static str,
    pub quote: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MeridianCorpus {
    pub gold: GoldGraph,
    pub manifest: DefectManifest,
    pub artifacts: Vec<Artifact>,
    pub evidence: Vec<Evidence>,
}

fn edge(subject: &str, kind: EdgeKind, object: &str) -> GoldEdge {
    GoldEdge {
        subject: subject.to_owned(),
        kind,
        object: object.to_owned(),
    }
}

pub fn corpus() -> MeridianCorpus {
    let claim_legal = edge(
        "Claim (Legal)",
        EdgeKind::DefinesTerm,
        "a formal demand for coverage under a policy",
    );
    let claim_finance = edge(
        "Claim (Finance)",
        EdgeKind::DefinesTerm,
        "an expense reimbursement entry in the general ledger",
    );
    let filing_45 = edge("Claims", EdgeKind::GovernedBy, "45-day filing window");
    let filing_30 = edge("Claims", EdgeKind::GovernedBy, "30-day filing window");
    let supersession = edge(
        "45-day filing window",
        EdgeKind::Supersedes,
        "30-day filing window",
    );
    let portal_only = edge("Claims", EdgeKind::GovernedBy, "portal-only intake");
    let pto_rule = edge(
        "HR",
        EdgeKind::GovernedBy,
        "PTO accrues 1 day per fortnight worked, capped at 26 days",
    );
    let ledger = edge("Finance", EdgeKind::Owns, "general ledger");
    let prior_auth = edge(
        "Clinical Policy",
        EdgeKind::GovernedBy,
        "prior authorization required for specialty drugs",
    );
    let token_rotation = edge("IT-Security", EdgeKind::GovernedBy, "90-day token rotation");
    let hr_equipment = edge(
        "HR",
        EdgeKind::GovernedBy,
        "equipment returned within 14 days of departure",
    );
    let it_equipment = edge(
        "IT-Security",
        EdgeKind::GovernedBy,
        "equipment returned within 7 days of departure",
    );

    let gold = GoldGraph {
        edges: vec![
            claim_legal.clone(),
            edge("Claim (Legal)", EdgeKind::ScopedTo, "Legal"),
            claim_finance.clone(),
            edge("Claim (Finance)", EdgeKind::ScopedTo, "Finance"),
            filing_45.clone(),
            supersession,
            portal_only.clone(),
            pto_rule.clone(),
            ledger.clone(),
            prior_auth.clone(),
            token_rotation.clone(),
        ],
    };

    let manifest = DefectManifest {
        defects: vec![
            PlantedDefect::TermCollision {
                term: "Claim".to_owned(),
                departments: vec!["Legal".to_owned(), "Finance".to_owned()],
            },
            PlantedDefect::CrossDepartmentContradiction {
                edge_a: hr_equipment,
                edge_b: it_equipment,
            },
            PlantedDefect::CrossFormatSupersession {
                superseded: filing_30,
                superseding: filing_45.clone(),
            },
            PlantedDefect::DivergentDuplicate {
                canonical: pto_rule.clone(),
            },
            PlantedDefect::EmailOnlyRule {
                edge: portal_only.clone(),
            },
            PlantedDefect::NegativeSpace {
                question: "What is the travel reimbursement mileage rate?".to_owned(),
            },
            PlantedDefect::VendorAttachment {
                vendor: "Acme Retention Systems".to_owned(),
            },
            PlantedDefect::BoilerplateTrap {
                meridian_rule: pto_rule.clone(),
            },
        ],
    };

    let artifacts = vec![
        Artifact {
            name: "employee-handbook.md",
            bytes: EMPLOYEE_HANDBOOK.as_bytes().to_vec(),
        },
        Artifact {
            name: "it-security-policy.md",
            bytes: IT_SECURITY_POLICY.as_bytes().to_vec(),
        },
        Artifact {
            name: "legal-definitions.md",
            bytes: LEGAL_DEFINITIONS.as_bytes().to_vec(),
        },
        Artifact {
            name: "finance-glossary.md",
            bytes: FINANCE_GLOSSARY.as_bytes().to_vec(),
        },
        Artifact {
            name: "wiki-export.md",
            bytes: WIKI_EXPORT.as_bytes().to_vec(),
        },
        Artifact {
            name: "memo-4417.eml",
            bytes: MEMO_4417.as_bytes().to_vec(),
        },
        Artifact {
            name: "ops-bulletin.eml",
            bytes: OPS_BULLETIN.as_bytes().to_vec(),
        },
        Artifact {
            name: "vendor-notice.eml",
            bytes: VENDOR_NOTICE.as_bytes().to_vec(),
        },
        Artifact {
            name: "clinical-policy-manual.pdf",
            bytes: clinical_policy_manual_pdf(),
        },
    ];

    let evidence = vec![
        Evidence {
            edge_key: claim_legal.key(),
            artifact: "legal-definitions.md",
            quote: "a Claim means a formal demand for coverage under a policy",
        },
        Evidence {
            edge_key: claim_finance.key(),
            artifact: "finance-glossary.md",
            quote: "a Claim means an expense reimbursement entry in the general ledger",
        },
        Evidence {
            edge_key: filing_45.key(),
            artifact: "memo-4417.eml",
            quote: "the claims filing window is 45 days",
        },
        Evidence {
            edge_key: portal_only.key(),
            artifact: "ops-bulletin.eml",
            quote: "All claims arrive via the provider portal only",
        },
        Evidence {
            edge_key: pto_rule.key(),
            artifact: "employee-handbook.md",
            quote: "PTO accrues at 1 day per fortnight worked, capped at 26 days",
        },
        Evidence {
            edge_key: ledger.key(),
            artifact: "finance-glossary.md",
            quote: "Finance owns the general ledger",
        },
        Evidence {
            edge_key: prior_auth.key(),
            artifact: "clinical-policy-manual.pdf",
            quote: "Prior authorization is required for specialty drugs.",
        },
        Evidence {
            edge_key: token_rotation.key(),
            artifact: "it-security-policy.md",
            quote: "All access tokens rotate every 90 days",
        },
    ];

    MeridianCorpus {
        gold,
        manifest,
        artifacts,
        evidence,
    }
}

const EMPLOYEE_HANDBOOK: &str = "\
# Meridian Mutual Employee Handbook

## Claims

Claims must be filed within 30 days of the date of service.

## Human Resources

PTO accrues at 1 day per fortnight worked, capped at 26 days.

Departing employees must return company equipment within 14 days.

## Finance

Expense reports are reviewed by the Finance department monthly.
";

const IT_SECURITY_POLICY: &str = "\
# Meridian IT-Security Policy

All access tokens rotate every 90 days.

Departing employees must return company equipment within 7 days.
";

const LEGAL_DEFINITIONS: &str = "\
# Meridian Legal and Compliance Definitions

In Legal usage, a Claim means a formal demand for coverage under a policy.
";

const FINANCE_GLOSSARY: &str = "\
# Meridian Finance Glossary

In Finance usage, a Claim means an expense reimbursement entry in the general ledger.

Finance owns the general ledger.
";

const WIKI_EXPORT: &str = "\
# Meridian Wiki - Benefits FAQ

Employees accrue PTO at the industry standard rate of 10 days per year.
";

const MEMO_4417: &str = "\
From: Dana Reyes <dana.reyes@meridianmutual.example>
To: claims-all@meridianmutual.example
Subject: Filing window change, effective immediately
Date: Tue, 21 Jul 2026 09:14:00 -0700
Message-ID: <memo-4417@meridianmutual.example>
X-Department: Claims

Team,

Effective immediately the claims filing window is 45 days, superseding
the 30-day window stated in the handbook and the clinical policy manual.

Dana
";

const OPS_BULLETIN: &str = "\
From: Ops Desk <ops@meridianmutual.example>
To: claims-all@meridianmutual.example
Subject: Fax intake decommissioned
Date: Wed, 08 Jul 2026 15:02:00 -0700
Message-ID: <ops-bulletin-812@meridianmutual.example>
X-Department: Claims

As of this week, fax intake is decommissioned.
All claims arrive via the provider portal only.

Ops Desk
";

const VENDOR_NOTICE: &str = "\
From: Acme Retention <notice@acmeretention.example>
To: procurement@meridianmutual.example
Subject: Updated Acme retention terms
Date: Mon, 06 Jul 2026 11:30:00 -0700
Message-ID: <acme-terms-2026@acmeretention.example>

Per our updated terms, Acme Retention Systems retains customer data for
10 years from the date of collection.

Acme Retention Systems
";

fn clinical_policy_manual_pdf() -> Vec<u8> {
    render_pdf(&[
        vec![
            "Meridian Clinical Policy Manual".to_owned(),
            "Prior authorization is required for specialty drugs.".to_owned(),
            "Claims must be filed within 30 days of the date of service.".to_owned(),
        ],
        vec![
            "Appendix A - Benefits summary".to_owned(),
            "PTO accrues at 1 day per fortnight worked, capped at 25 days.".to_owned(),
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::Citation;
    use crate::cdm::{CdmSpan, SourceFormat};
    use crate::intake::Registrar;

    #[test]
    fn every_artifact_ingests_through_the_registrar() {
        for artifact in corpus().artifacts {
            let doc = Registrar::ingest(&artifact.bytes)
                .unwrap_or_else(|e| panic!("{} failed: {e}", artifact.name));
            let expected = match artifact.name.rsplit('.').next().unwrap() {
                "md" => SourceFormat::Markdown,
                "eml" => SourceFormat::Eml,
                "pdf" => SourceFormat::Pdf,
                other => panic!("unexpected extension {other}"),
            };
            assert_eq!(doc.format, expected, "{}", artifact.name);
        }
    }

    #[test]
    fn every_evidence_quote_grounds_a_citation() {
        let corpus = corpus();
        for ev in &corpus.evidence {
            let artifact = corpus
                .artifacts
                .iter()
                .find(|a| a.name == ev.artifact)
                .unwrap_or_else(|| panic!("evidence names unknown artifact {}", ev.artifact));
            let doc = Registrar::ingest(&artifact.bytes).unwrap();
            let start = doc
                .canonical_text
                .find(ev.quote)
                .unwrap_or_else(|| panic!("{:?} not found in {}", ev.quote, ev.artifact));
            let citation =
                Citation::grounded(&doc, CdmSpan::new(start, start + ev.quote.len())).unwrap();
            assert_eq!(citation.quote, ev.quote);
            assert!(
                doc.native_anchor(citation.span).is_some(),
                "{}",
                ev.artifact
            );
        }
    }

    #[test]
    fn every_gold_edge_with_corpus_provenance_has_evidence() {
        let corpus = corpus();
        let evidenced: Vec<&str> = corpus
            .evidence
            .iter()
            .map(|e| e.edge_key.as_str())
            .collect();
        for gold_edge in &corpus.gold.edges {
            let structural = matches!(gold_edge.kind, EdgeKind::ScopedTo | EdgeKind::Supersedes);
            if !structural {
                assert!(
                    evidenced.contains(&gold_edge.key().as_str()),
                    "gold edge {} has no supporting evidence",
                    gold_edge.key()
                );
            }
        }
    }

    #[test]
    fn negative_space_is_truly_absent() {
        for artifact in corpus().artifacts {
            if let Ok(doc) = Registrar::ingest(&artifact.bytes) {
                assert!(
                    !doc.canonical_text.to_ascii_lowercase().contains("mileage"),
                    "{} answers the negative-space question",
                    artifact.name
                );
            }
        }
    }

    #[test]
    fn divergent_duplicate_actually_diverges() {
        let corpus = corpus();
        let handbook = corpus
            .artifacts
            .iter()
            .find(|a| a.name == "employee-handbook.md")
            .unwrap();
        let manual = corpus
            .artifacts
            .iter()
            .find(|a| a.name == "clinical-policy-manual.pdf")
            .unwrap();
        let handbook_doc = Registrar::ingest(&handbook.bytes).unwrap();
        let manual_doc = Registrar::ingest(&manual.bytes).unwrap();
        assert!(handbook_doc.canonical_text.contains("capped at 26 days"));
        assert!(manual_doc.canonical_text.contains("capped at 25 days"));
    }

    #[test]
    fn rendering_is_deterministic_across_calls() {
        let (a, b) = (corpus(), corpus());
        for (x, y) in a.artifacts.iter().zip(b.artifacts.iter()) {
            assert_eq!(x.bytes, y.bytes, "{} is nondeterministic", x.name);
        }
        assert_eq!(a.gold, b.gold);
        assert_eq!(a.manifest, b.manifest);
    }

    #[test]
    fn manifest_plants_two_scored_contradictions() {
        assert_eq!(corpus().manifest.planted_contradictions(), 2);
    }
}
