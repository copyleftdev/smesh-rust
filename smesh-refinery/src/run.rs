//! Orchestration: corpus in, staged changeset + report out.
//!
//! The refinery's output stops exactly where the human's authority begins —
//! a `Changeset<Staged>` with lanes. Ratification and signing are Phase 4.

use crate::extract::{run_extractors, RejectedEmission};
use crate::packet::{ingest_all, prompt_block, NamedDoc};
use crate::verify::{audit_grounding, sentinel_pass};
use crate::{Oracle, RefineryError};
use smesh_world::corpus::Scorecard;
use smesh_world::meridian;
use smesh_world::{CandidateEdge, Changeset, Lane, Staged};

pub struct RunReport {
    pub docs: Vec<NamedDoc>,
    pub staged: Option<Changeset<Staged>>,
    pub rejected: Vec<RejectedEmission>,
    pub contradictions_caught: usize,
    pub scorecard: Option<Scorecard>,
}

impl RunReport {
    pub fn lane_counts(&self) -> (usize, usize, usize) {
        let mut counts = (0, 0, 0);
        if let Some(staged) = &self.staged {
            for (lane, _) in staged.lanes() {
                match lane {
                    Lane::Green => counts.0 += 1,
                    Lane::Amber => counts.1 += 1,
                    Lane::Red => counts.2 += 1,
                }
            }
        }
        counts
    }

    /// Human-readable run summary for the CLI.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let (green, amber, red) = self.lane_counts();
        out.push_str(&format!(
            "documents ingested: {}\nrejected emissions (confabulation firewall): {}\ncontradiction pairs flagged: {}\n",
            self.docs.len(),
            self.rejected.len(),
            self.contradictions_caught
        ));
        for r in &self.rejected {
            out.push_str(&format!(
                "  rejected [{:?}] {} --[{}]--> {} : {}\n",
                r.role, r.emission.subject, r.emission.kind, r.emission.object, r.reason
            ));
        }
        match &self.staged {
            None => out.push_str("staged changeset: EMPTY — nothing survived\n"),
            Some(staged) => {
                out.push_str(&format!(
                    "staged changeset vs {}: {} candidates (green {green} / amber {amber} / red {red})\n",
                    staged.base_rev,
                    staged.edges.len()
                ));
                for (lane, edge) in staged.lanes() {
                    out.push_str(&format!(
                        "  [{lane:?}] \"{}\" --[{:?}]--> \"{}\"  (+{}/-{})\n",
                        edge.subject,
                        edge.kind,
                        edge.object,
                        edge.corroborations(),
                        edge.refutations()
                    ));
                }
            }
        }
        if let Some(card) = &self.scorecard {
            out.push_str(&format!(
                "scorecard: precision {:.2} recall {:.2} confabulated {} contradictions {}/{} gate {}\n",
                card.precision(),
                card.recall(),
                card.confabulated,
                card.contradictions_caught,
                card.contradictions_planted,
                if card.passes_gate() { "PASS" } else { "FAIL" }
            ));
        }
        out
    }
}

/// Refine an arbitrary corpus: ingest, extract, audit, sentinel, stage.
pub async fn refine(
    oracle: &dyn Oracle,
    artifacts: &[(String, Vec<u8>)],
    base_rev: &str,
) -> Result<RunReport, RefineryError> {
    let docs = ingest_all(artifacts)?;
    let block = prompt_block(&docs);

    let mut outcome = run_extractors(oracle, &docs, &block).await?;
    audit_grounding(oracle, &docs, &mut outcome.candidates).await?;
    let contradictions_caught = sentinel_pass(oracle, &mut outcome.candidates).await?;

    let staged = if outcome.candidates.is_empty() {
        None
    } else {
        Some(Changeset::stage(
            base_rev.to_owned(),
            outcome.candidates.clone(),
        )?)
    };

    Ok(RunReport {
        docs,
        staged,
        rejected: outcome.rejected,
        contradictions_caught,
        scorecard: None,
    })
}

/// Refine the Meridian benchmark and score the run against its gold graph.
pub async fn refine_meridian(oracle: &dyn Oracle) -> Result<RunReport, RefineryError> {
    let corpus = meridian::corpus();
    let artifacts: Vec<(String, Vec<u8>)> = corpus
        .artifacts
        .iter()
        .map(|a| (a.name.to_owned(), a.bytes.clone()))
        .collect();
    let mut report = refine(oracle, &artifacts, "meridian-rev0").await?;

    let observed: &[CandidateEdge] = report
        .staged
        .as_ref()
        .map(|s| s.edges.as_slice())
        .unwrap_or(&[]);
    report.scorecard = Some(Scorecard::evaluate(
        &corpus.gold,
        observed,
        &corpus.manifest,
        report.contradictions_caught,
    ));
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Routes by system-prompt content: extractor roles get scripted
    /// emissions, the auditor corroborates, the sentinel finds nothing new.
    struct MeridianScript;

    #[async_trait]
    impl Oracle for MeridianScript {
        async fn complete(
            &self,
            _model: &str,
            system: &str,
            _prompt: &str,
        ) -> Result<String, RefineryError> {
            if system.contains("Policy expert") {
                Ok(r#"[
                    {"subject": "Claims", "kind": "GovernedBy", "object": "45-day filing window",
                     "doc": "memo-4417.eml", "quote": "the claims filing window is 45 days"},
                    {"subject": "Claims", "kind": "GovernedBy", "object": "30-day filing window",
                     "doc": "employee-handbook.md", "quote": "Claims must be filed within 30 days of the date of service."},
                    {"subject": "Claims", "kind": "GovernedBy", "object": "wormhole intake",
                     "doc": "employee-handbook.md", "quote": "claims teleport through the wormhole"}
                ]"#
                .to_owned())
            } else if system.contains("Lexicon expert") {
                Ok(r#"[
                    {"subject": "Claim (Legal)", "kind": "DefinesTerm",
                     "object": "a formal demand for coverage under a policy",
                     "doc": "legal-definitions.md",
                     "quote": "a Claim means a formal demand for coverage under a policy"}
                ]"#
                .to_owned())
            } else if system.contains("Structure expert") || system.contains("Process expert") {
                Ok("[]".to_owned())
            } else if system.contains("grounding auditor") {
                Ok(r#"{"verdict": "corroborate", "rationale": "entailed"}"#.to_owned())
            } else if system.contains("contradiction sentinel") {
                Ok("[]".to_owned())
            } else {
                Err(RefineryError::Parse(format!(
                    "unscripted system: {system:.60}"
                )))
            }
        }
    }

    #[tokio::test]
    async fn scripted_meridian_run_stages_scores_and_firewalls() {
        let report = refine_meridian(&MeridianScript).await.unwrap();

        assert_eq!(
            report.rejected.len(),
            1,
            "wormhole quote must be firewalled"
        );
        assert!(report.rejected[0].reason.contains("not found verbatim"));

        let staged = report.staged.as_ref().unwrap();
        assert_eq!(staged.edges.len(), 3);

        assert_eq!(
            report.contradictions_caught, 1,
            "30 vs 45 with no supersedes edge is an unresolved conflict"
        );
        let contested = staged.edges.iter().filter(|e| e.is_contested()).count();
        assert_eq!(contested, 2, "both filing rules reach the human contested");
        let (_, _, red) = report.lane_counts();
        assert_eq!(red, 2);

        let card = report.scorecard.as_ref().unwrap();
        assert_eq!(card.confabulated, 0, "firewalled emissions never score");
        assert!(card.precision() > 0.6, "2 of 3 staged edges are gold");
        assert!(
            card.recall() < 0.5,
            "most of the gold graph was not extracted"
        );
    }
}
