//! Tier 2 verification.
//!
//! The Grounding Auditor re-verifies every citation mechanically, then asks
//! a cross-family model whether the quote entails the claim. The
//! Contradiction Sentinel is deterministic code where it can be (same
//! subject + kind, supersession resolution) and a cross-family model where
//! it must be (contradictions across different subjects). Verifiers cannot
//! emit facts — only verdicts.

use crate::packet::NamedDoc;
use crate::roster;
use crate::{extract_json, Oracle, RefineryError};
use serde::Deserialize;
use smesh_world::role::ModelPolicy;
use smesh_world::{
    CandidateEdge, EdgeKind, Judgment, ProvenanceClass, StructuralConstraint, Verdict, WorldRole,
};
use std::collections::BTreeMap;

fn verifier_model(role: WorldRole) -> &'static str {
    match role.model_policy() {
        ModelPolicy::DistinctFamilyFromExtraction(m) => m,
        other => unreachable!("verifier {role:?} has policy {other:?}"),
    }
}

#[derive(Debug, Deserialize)]
struct AuditorResponse {
    verdict: String,
    #[serde(default)]
    rationale: String,
}

fn context_window(text: &str, start: usize, end: usize) -> &str {
    smesh_world::context_window(text, start, end, 200)
}

/// Run the Grounding Auditor over every corpus-derived candidate.
pub async fn audit_grounding(
    oracle: &dyn Oracle,
    docs: &[NamedDoc],
    candidates: &mut [CandidateEdge],
) -> Result<(), RefineryError> {
    let model = verifier_model(WorldRole::GroundingAuditor);
    let system = roster::auditor_system();
    for candidate in candidates.iter_mut() {
        let ProvenanceClass::CorpusDerived { citations } = &candidate.provenance else {
            continue;
        };
        // Indexing panicked on a candidate that reached here with no citations.
        // The extractor is a language model, so "cannot happen" is a claim about
        // a model's output rather than about this code.
        let Some(citation) = citations.first().cloned() else {
            return Err(RefineryError::Parse(
                "candidate reached verification with no citation to audit".into(),
            ));
        };
        let named = docs
            .iter()
            .find(|d| d.doc.id == citation.doc)
            .ok_or_else(|| RefineryError::Parse("citation references unknown doc".into()))?;
        citation.verify_against(&named.doc)?;

        let context = context_window(
            &named.doc.canonical_text,
            citation.span.start,
            citation.span.end,
        );
        let prompt = format!(
            "Claim: \"{}\" --[{:?}]--> \"{}\"\nCited quote: \"{}\"\nContext:\n...{}...",
            candidate.subject, candidate.kind, candidate.object, citation.quote, context
        );
        let response = oracle.complete(model, &system, &prompt).await?;
        let parsed: AuditorResponse = serde_json::from_value(extract_json(&response)?)
            .map_err(|e| RefineryError::Parse(format!("auditor verdict: {e}")))?;
        let judgment = if parsed.verdict.eq_ignore_ascii_case("corroborate") {
            Judgment::Corroborate
        } else {
            Judgment::Refute
        };
        candidate.record(Verdict::new(
            WorldRole::GroundingAuditor,
            judgment,
            parsed.rationale,
        )?);
    }
    Ok(())
}

/// Run the Contradiction Sentinel. Returns the number of conflict pairs
/// flagged — the `contradictions_caught` input to the scorecard.
pub async fn sentinel_pass(
    oracle: &dyn Oracle,
    candidates: &mut [CandidateEdge],
) -> Result<usize, RefineryError> {
    let supersedes: Vec<(String, String)> = candidates
        .iter()
        .filter(|c| c.kind == EdgeKind::Supersedes)
        .map(|c| (c.subject.clone(), c.object.clone()))
        .collect();
    let superseded_by = |loser: &str| -> Option<&str> {
        supersedes
            .iter()
            .find(|(_, l)| l == loser)
            .map(|(w, _)| w.as_str())
    };
    let related = |a: &str, b: &str| {
        supersedes
            .iter()
            .any(|(w, l)| (w == a && l == b) || (w == b && l == a))
    };

    let mut refutations: BTreeMap<usize, String> = BTreeMap::new();
    let mut conflicts = 0usize;
    // Contradictions are counted per pair. A candidate that conflicts with
    // several others added one per conflict, so a single disputed claim
    // inflated the total and the detection rate computed from it.
    let mut counted_pairs: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    let mut count_pair = move |i: usize, j: usize| counted_pairs.insert((i.min(j), i.max(j)));

    let mut groups: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (i, c) in candidates.iter().enumerate() {
        let conflict_prone = c.kind == EdgeKind::GovernedBy
            || c.kind.structural_constraint()
                == Some(StructuralConstraint::UniquePerSubjectPerTimeslice);
        if conflict_prone {
            groups
                .entry((c.subject.clone(), format!("{:?}", c.kind)))
                .or_default()
                .push(i);
        }
    }
    for indices in groups.values() {
        for (a_pos, &i) in indices.iter().enumerate() {
            for &j in &indices[a_pos + 1..] {
                let (oi, oj) = (&candidates[i].object, &candidates[j].object);
                if oi == oj {
                    continue;
                }
                if let Some(winner) = superseded_by(oi) {
                    refutations
                        .entry(i)
                        .or_insert_with(|| format!("superseded by {winner}"));
                } else if let Some(winner) = superseded_by(oj) {
                    refutations
                        .entry(j)
                        .or_insert_with(|| format!("superseded by {winner}"));
                } else {
                    if count_pair(i, j) {
                        conflicts += 1;
                    }
                    refutations
                        .entry(i)
                        .or_insert_with(|| format!("conflicts with {oj:?} for the same subject"));
                    refutations
                        .entry(j)
                        .or_insert_with(|| format!("conflicts with {oi:?} for the same subject"));
                }
            }
        }
    }

    if candidates.len() > 1 {
        // Evidence quotes ride along: two edges can share a bland name
        // ("PTO accrual") while their quoted substance conflicts — the
        // Meridian boilerplate trap taught us names alone are not enough.
        let listing: String = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let quote = match &c.provenance {
                    ProvenanceClass::CorpusDerived { citations } => citations
                        .first()
                        .map(|ci| ci.quote.chars().take(160).collect::<String>())
                        .unwrap_or_default(),
                    ProvenanceClass::HumanAttested { .. } => String::new(),
                };
                format!(
                    "{i}. \"{}\" --[{:?}]--> \"{}\"  evidence: \"{quote}\"\n",
                    c.subject, c.kind, c.object
                )
            })
            .collect();
        let response = oracle
            .complete(
                verifier_model(WorldRole::ContradictionSentinel),
                &roster::sentinel_system(),
                &listing,
            )
            .await?;
        let pairs: Vec<Vec<usize>> = serde_json::from_value(extract_json(&response)?)
            .map_err(|e| RefineryError::Parse(format!("sentinel pairs: {e}")))?;
        for pair in pairs {
            let [i, j] = pair.as_slice() else { continue };
            let (i, j) = (*i, *j);
            if i >= candidates.len() || j >= candidates.len() || i == j {
                continue;
            }
            if related(&candidates[i].object, &candidates[j].object) {
                continue;
            }
            if count_pair(i, j) {
                conflicts += 1;
            }
            let (oi, oj) = (candidates[i].object.clone(), candidates[j].object.clone());
            refutations
                .entry(i)
                .or_insert_with(|| format!("contradicts candidate asserting {oj:?}"));
            refutations
                .entry(j)
                .or_insert_with(|| format!("contradicts candidate asserting {oi:?}"));
        }
    }

    for (i, candidate) in candidates.iter_mut().enumerate() {
        let verdict = match refutations.get(&i) {
            Some(rationale) => Verdict::new(
                WorldRole::ContradictionSentinel,
                Judgment::Refute,
                rationale.clone(),
            )?,
            None => Verdict::new(
                WorldRole::ContradictionSentinel,
                Judgment::Corroborate,
                "no conflicts within the candidate set".into(),
            )?,
        };
        candidate.record(verdict);
    }
    Ok(conflicts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::ingest_all;
    use async_trait::async_trait;
    use smesh_world::{CdmSpan, Citation};

    struct Scripted(&'static str);

    #[async_trait]
    impl Oracle for Scripted {
        async fn complete(&self, _: &str, _: &str, _: &str) -> Result<String, RefineryError> {
            Ok(self.0.to_owned())
        }
    }

    fn candidate(subject: &str, kind: EdgeKind, object: &str) -> CandidateEdge {
        let docs = ingest_all(&[(
            "doc.md".to_owned(),
            b"# Doc\n\nClaims must be filed within 30 days. The window is 45 days now.\n".to_vec(),
        )])
        .unwrap();
        let citation = Citation::grounded(&docs[0].doc, CdmSpan::new(8, 14)).unwrap();
        CandidateEdge::emit(
            WorldRole::Policy,
            subject.into(),
            kind,
            object.into(),
            vec![citation],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn unresolved_same_subject_conflict_contests_both() {
        let mut candidates = vec![
            candidate("Claims", EdgeKind::GovernedBy, "30-day filing window"),
            candidate("Claims", EdgeKind::GovernedBy, "45-day filing window"),
        ];
        let conflicts = sentinel_pass(&Scripted("[]"), &mut candidates)
            .await
            .unwrap();
        assert_eq!(conflicts, 1);
        assert_eq!(candidates[0].refutations(), 1);
        assert_eq!(candidates[1].refutations(), 1);
    }

    #[tokio::test]
    async fn supersession_resolves_the_conflict_deterministically() {
        let mut candidates = vec![
            candidate("Claims", EdgeKind::GovernedBy, "30-day filing window"),
            candidate("Claims", EdgeKind::GovernedBy, "45-day filing window"),
            candidate(
                "45-day filing window",
                EdgeKind::Supersedes,
                "30-day filing window",
            ),
        ];
        let conflicts = sentinel_pass(&Scripted("[]"), &mut candidates)
            .await
            .unwrap();
        assert_eq!(conflicts, 0);
        assert_eq!(candidates[0].refutations(), 1, "superseded rule is refuted");
        assert_eq!(candidates[1].refutations(), 0, "superseding rule survives");
    }

    #[tokio::test]
    async fn semantic_pairs_from_the_model_contest_both_sides() {
        let mut candidates = vec![
            candidate("HR", EdgeKind::GovernedBy, "equipment back in 14 days"),
            candidate(
                "IT-Security",
                EdgeKind::GovernedBy,
                "equipment back in 7 days",
            ),
        ];
        let conflicts = sentinel_pass(&Scripted("[[0,1]]"), &mut candidates)
            .await
            .unwrap();
        assert_eq!(conflicts, 1);
        assert!(candidates[0].refutations() == 1 && candidates[1].refutations() == 1);
    }

    #[tokio::test]
    async fn auditor_records_cross_model_verdicts() {
        let docs = ingest_all(&[(
            "doc.md".to_owned(),
            b"# Doc\n\nClaims must be filed within 30 days. The window is 45 days now.\n".to_vec(),
        )])
        .unwrap();
        let mut candidates = vec![candidate("Claims", EdgeKind::GovernedBy, "30-day window")];
        audit_grounding(
            &Scripted(r#"{"verdict": "corroborate", "rationale": "entailed"}"#),
            &docs,
            &mut candidates,
        )
        .await
        .unwrap();
        assert_eq!(candidates[0].corroborations(), 1);
    }
}
