//! Offline demo session: a realistic staged run assembled from the Meridian
//! corpus with no network and no refinery — grounded citations, mixed
//! verdicts, all three lanes populated. Drives UI work and tests.

use smesh_world::meridian;
use smesh_world::{
    CandidateEdge, CdmDocument, CdmSpan, Citation, EdgeKind, EvidenceView, Judgment, Registrar,
    RejectedView, StagedRun, Verdict, WorldRole,
};
use std::collections::BTreeMap;

struct Bench {
    docs: Vec<(String, CdmDocument)>,
}

impl Bench {
    fn new() -> Self {
        let docs = meridian::corpus()
            .artifacts
            .into_iter()
            .map(|a| {
                let doc = Registrar::ingest(&a.bytes).expect("meridian ingests");
                (a.name.to_owned(), doc)
            })
            .collect();
        Self { docs }
    }

    fn doc(&self, name: &str) -> &CdmDocument {
        &self
            .docs
            .iter()
            .find(|(n, _)| n == name)
            .expect("known doc")
            .1
    }

    fn grounded(
        &self,
        role: WorldRole,
        subject: &str,
        kind: EdgeKind,
        object: &str,
        doc_name: &str,
        quote: &str,
    ) -> (CandidateEdge, EvidenceView) {
        let doc = self.doc(doc_name);
        let start = doc
            .canonical_text
            .find(quote)
            .unwrap_or_else(|| panic!("{quote:?} not in {doc_name}"));
        let span = CdmSpan::new(start, start + quote.len());
        let citation = Citation::grounded(doc, span).expect("demo quotes ground");
        let view = EvidenceView {
            doc_name: doc_name.to_owned(),
            quote: quote.to_owned(),
            anchor: doc
                .native_anchor(span)
                .map(ToString::to_string)
                .unwrap_or_default(),
            context: smesh_world::context_window(&doc.canonical_text, span.start, span.end, 120)
                .to_owned(),
        };
        let edge = CandidateEdge::emit(role, subject.into(), kind, object.into(), vec![citation])
            .expect("demo emissions are valid");
        (edge, view)
    }
}

fn corroborate(by: WorldRole, why: &str) -> Verdict {
    Verdict::new(by, Judgment::Corroborate, why.into()).expect("verifier role")
}

fn refute(by: WorldRole, why: &str) -> Verdict {
    Verdict::new(by, Judgment::Refute, why.into()).expect("verifier role")
}

/// A staged run with 2 green, 2 amber, and 4 red (two contested pairs).
pub fn demo_staged_run() -> StagedRun {
    let bench = Bench::new();
    let mut candidates = Vec::new();
    let mut evidence: BTreeMap<String, Vec<EvidenceView>> = BTreeMap::new();

    let mut push = |edge: CandidateEdge, view: EvidenceView| {
        evidence.insert(edge.key(), vec![view]);
        candidates.push(edge);
    };

    let (mut legal_term, v) = bench.grounded(
        WorldRole::Lexicon,
        "Claim (Legal)",
        EdgeKind::DefinesTerm,
        "a formal demand for coverage under a policy",
        "legal-definitions.md",
        "a Claim means a formal demand for coverage under a policy",
    );
    legal_term.record(corroborate(
        WorldRole::GroundingAuditor,
        "quote entails the definition",
    ));
    legal_term.record(corroborate(
        WorldRole::ContradictionSentinel,
        "no conflicts",
    ));
    push(legal_term, v);

    let (mut ledger, v) = bench.grounded(
        WorldRole::Structure,
        "Finance",
        EdgeKind::Owns,
        "general ledger",
        "finance-glossary.md",
        "Finance owns the general ledger",
    );
    ledger.record(corroborate(
        WorldRole::GroundingAuditor,
        "direct statement of ownership",
    ));
    ledger.record(corroborate(
        WorldRole::ContradictionSentinel,
        "no conflicts",
    ));
    push(ledger, v);

    let (mut tokens, v) = bench.grounded(
        WorldRole::Policy,
        "IT-Security",
        EdgeKind::GovernedBy,
        "90-day token rotation",
        "it-security-policy.md",
        "All access tokens rotate every 90 days",
    );
    tokens.record(corroborate(
        WorldRole::GroundingAuditor,
        "quote states the rotation rule",
    ));
    push(tokens, v);

    let (mut pto, v) = bench.grounded(
        WorldRole::Policy,
        "HR",
        EdgeKind::GovernedBy,
        "PTO fortnight accrual capped at 26 days",
        "employee-handbook.md",
        "PTO accrues at 1 day per fortnight worked, capped at 26 days",
    );
    pto.record(corroborate(
        WorldRole::GroundingAuditor,
        "quote states the accrual rule",
    ));
    push(pto, v);

    let (mut filing_45, v) = bench.grounded(
        WorldRole::Policy,
        "Claims",
        EdgeKind::GovernedBy,
        "45-day filing window",
        "memo-4417.eml",
        "the claims filing window is 45 days",
    );
    filing_45.record(corroborate(
        WorldRole::GroundingAuditor,
        "memo states the new window",
    ));
    filing_45.record(refute(
        WorldRole::ContradictionSentinel,
        "conflicts with \"30-day filing window\" for the same subject",
    ));
    push(filing_45, v);

    let (mut filing_30, v) = bench.grounded(
        WorldRole::Policy,
        "Claims",
        EdgeKind::GovernedBy,
        "30-day filing window",
        "employee-handbook.md",
        "Claims must be filed within 30 days of the date of service",
    );
    filing_30.record(corroborate(
        WorldRole::GroundingAuditor,
        "handbook states the old window",
    ));
    filing_30.record(refute(
        WorldRole::ContradictionSentinel,
        "conflicts with \"45-day filing window\" for the same subject",
    ));
    push(filing_30, v);

    let (mut hr_equipment, v) = bench.grounded(
        WorldRole::Policy,
        "HR",
        EdgeKind::GovernedBy,
        "equipment returned within 14 days of departure",
        "employee-handbook.md",
        "Departing employees must return company equipment within 14 days",
    );
    hr_equipment.record(corroborate(
        WorldRole::GroundingAuditor,
        "quote states the deadline",
    ));
    hr_equipment.record(refute(
        WorldRole::ContradictionSentinel,
        "contradicts candidate asserting \"equipment returned within 7 days of departure\"",
    ));
    push(hr_equipment, v);

    let (mut it_equipment, v) = bench.grounded(
        WorldRole::Policy,
        "IT-Security",
        EdgeKind::GovernedBy,
        "equipment returned within 7 days of departure",
        "it-security-policy.md",
        "Departing employees must return company equipment within 7 days",
    );
    it_equipment.record(corroborate(
        WorldRole::GroundingAuditor,
        "quote states the deadline",
    ));
    it_equipment.record(refute(
        WorldRole::ContradictionSentinel,
        "contradicts candidate asserting \"equipment returned within 14 days of departure\"",
    ));
    push(it_equipment, v);

    StagedRun {
        base_rev: "meridian-rev0".into(),
        candidates,
        evidence,
        rejected: vec![RejectedView {
            role: "Policy".into(),
            summary: "Claims --[GovernedBy]--> wormhole intake".into(),
            reason: "quote not found verbatim in document".into(),
        }],
        contradictions_caught: 2,
        scorecard: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smesh_world::Lane;

    #[test]
    fn demo_populates_all_three_lanes_with_grounded_evidence() {
        let run = demo_staged_run();
        let lanes: Vec<Lane> = run.candidates.iter().map(Lane::assign).collect();
        assert_eq!(lanes.iter().filter(|l| **l == Lane::Green).count(), 2);
        assert_eq!(lanes.iter().filter(|l| **l == Lane::Amber).count(), 2);
        assert_eq!(lanes.iter().filter(|l| **l == Lane::Red).count(), 4);
        for c in &run.candidates {
            let views = &run.evidence[&c.key()];
            assert!(!views.is_empty());
            assert!(!views[0].anchor.is_empty());
            assert!(views[0].context.contains(&views[0].quote));
        }
    }
}
