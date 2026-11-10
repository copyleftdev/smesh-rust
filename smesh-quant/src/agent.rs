//! A finished research run as a mesh agent, reachable over A2A.
//!
//! [`FindingsBackend`] answers research questions from a run's recorded
//! evidence and nothing else. It never generates text: Jev picks which of the
//! run's findings bear on the question (one yes/no per finding, all in one
//! request), and the answer is those findings exactly as recorded, with the
//! metrics behind them and the run's caveats. An answer can therefore never
//! claim more than the walk-forward showed.
//!
//! As an agent its capability describes what the evidence covers, so in a
//! mesh with an affinity source (Jev) quant questions reach it and other
//! questions do not.

use std::path::{Path, PathBuf};
use std::time::Instant;

use async_trait::async_trait;
use serde_json::{json, Map, Value};
use smesh_agent::{
    AgentConfig, AgentRole, GenerateRequest, GenerateResponse, JevClient, LlmAgent, LlmBackend,
    LlmError, LlmProvider,
};

/// What a research agent says it does.
pub const CAPABILITY: &str = "a quant researcher holding read-only walk-forward evidence on \
    Kalshi prediction markets: whether hourly Bitcoin and Ether above/below odds are well \
    priced against a realized-volatility model and fee-aware paper trading, and whether CPI, \
    payrolls and Fed ladders forecast the releases and the SPY/TLT reaction, cross-checked \
    against Massive and Kraken market data";

/// A recorded finding with its track.
#[derive(Debug, Clone)]
pub struct RecordedFinding {
    /// `crypto` or `macro`.
    pub track: String,
    /// Hypothesis id.
    pub id: String,
    /// What it claims.
    pub label: String,
    /// Final strength.
    pub strength: f64,
    /// Whether it reached consensus.
    pub consensus: bool,
    /// Confirming windows.
    pub confirmations: u64,
    /// Contradicting windows.
    pub contradictions: u64,
}

/// Read the findings from a run's `metrics.json`.
pub fn load_findings(metrics: &Value) -> Vec<RecordedFinding> {
    metrics["findings"]
        .as_array()
        .map(|fs| {
            fs.iter()
                .filter_map(|pair| {
                    let track = pair.get(0)?.as_str()?.to_string();
                    let f = pair.get(1)?;
                    Some(RecordedFinding {
                        track,
                        id: f["id"].as_str()?.to_string(),
                        label: f["label"].as_str()?.to_string(),
                        strength: f["strength"].as_f64().unwrap_or(0.0),
                        consensus: f["consensus"].as_bool().unwrap_or(false),
                        confirmations: f["confirmations"].as_u64().unwrap_or(0),
                        contradictions: f["contradictions"].as_u64().unwrap_or(0),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The Jev request that asks which findings bear on `question`.
pub fn relevance_question(
    question: &str,
    findings: &[RecordedFinding],
) -> (Value, Map<String, Value>) {
    let catalog: Map<String, Value> = findings
        .iter()
        .map(|f| (f.id.clone(), json!(f.label)))
        .collect();
    let state = json!({"question": question, "findings": catalog});
    let questions = findings
        .iter()
        .map(|f| {
            (
                format!("about_{}", f.id.replace('.', "_")),
                json!({
                    "type": "noul",
                    "instructions": format!(
                        "Is the research claim `findings.{}` about what `question` asks?",
                        f.id
                    ),
                }),
            )
        })
        .collect();
    (state, questions)
}

fn track_evidence(metrics: &Value, track: &str) -> String {
    let m = &metrics[track];
    let n = |v: &Value| {
        v.as_f64()
            .map(|x| format!("{x:.4}"))
            .unwrap_or_else(|| "n/a".into())
    };
    match track {
        "crypto" => format!(
            "Crypto evidence: {} samples, {} walk-forward windows, {} out-of-sample predictions. \
             Log loss: Kalshi {}, model {}, blend {}. Paper strategy: {} trades, {} per trade after fees.",
            m["samples"], m["windows"], m["test_predictions"],
            n(&m["logloss"]["market"]), n(&m["logloss"]["model"]), n(&m["logloss"]["blend"]),
            m["paper"]["trades"], n(&m["paper"]["pnl_per_trade"]),
        ),
        "macro" => format!(
            "Macro evidence: {} releases ({}), {} walk-forward windows. Threshold Brier: Kalshi {} vs climatology {}.",
            m["releases"], m["by_series"], m["windows"],
            n(&m["brier"]["kalshi"]), n(&m["brier"]["climatology"]),
        ),
        _ => String::new(),
    }
}

/// Compose the answer from the chosen findings, verbatim from the record.
pub fn compose(run: &str, metrics: &Value, chosen: &[&RecordedFinding]) -> String {
    let mut out =
        format!("From recorded research run `{run}` (read-only data, walk-forward validated):\n");
    let mut tracks: Vec<&str> = Vec::new();
    for f in chosen {
        out.push_str(&format!(
            "\n- {} — {} (strength {:.2}; {} windows confirmed, {} contradicted).",
            f.label,
            if f.consensus {
                "SUPPORTED by consensus"
            } else {
                "not supported"
            },
            f.strength,
            f.confirmations,
            f.contradictions
        ));
        if !tracks.contains(&f.track.as_str()) {
            tracks.push(&f.track);
        }
    }
    for t in tracks {
        out.push_str(&format!("\n\n{}", track_evidence(metrics, t)));
    }
    out.push_str("\n\nCaveats: several claims were tested on the same windows; macro windows hold one release each; this is evidence from one recorded period, not a forecast.");
    out
}

/// Choose findings from Jev's per-finding relevance scores (aligned with
/// `findings`; `None` where Jev gave no answer).
///
/// Findings scored 0.5 or more, most relevant first. If none clears 0.5,
/// the two closest, so a question is never answered with nothing. If Jev
/// scored none at all, every finding, as when Jev is unreachable.
pub fn pick_relevant<'a>(
    findings: &'a [RecordedFinding],
    scores: &[Option<f64>],
) -> Vec<&'a RecordedFinding> {
    let mut scored: Vec<(f64, &RecordedFinding)> = findings
        .iter()
        .zip(scores)
        .filter_map(|(f, s)| s.map(|p| (p, f)))
        .collect();
    if scored.is_empty() {
        tracing::warn!("Jev scored no findings; answering with every finding");
        return findings.iter().collect();
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let relevant: Vec<&RecordedFinding> = scored
        .iter()
        .filter(|(p, _)| *p >= 0.5)
        .map(|(_, f)| *f)
        .collect();
    if relevant.is_empty() {
        scored.into_iter().take(2).map(|(_, f)| f).collect()
    } else {
        relevant
    }
}

/// Answers research questions from one run's recorded findings.
pub struct FindingsBackend {
    run: String,
    metrics: Value,
    findings: Vec<RecordedFinding>,
    jev: Option<JevClient>,
}

impl FindingsBackend {
    /// Load the run at `run_dir`. With `jev`, answers include only the
    /// findings Jev judges relevant; without it, all findings.
    pub fn load(run_dir: &Path, jev: Option<JevClient>) -> anyhow::Result<Self> {
        let metrics: Value = serde_json::from_slice(&std::fs::read(run_dir.join("metrics.json"))?)?;
        let findings = load_findings(&metrics);
        anyhow::ensure!(
            !findings.is_empty(),
            "{} has no findings",
            run_dir.display()
        );
        Ok(Self {
            run: run_dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            metrics,
            findings,
            jev,
        })
    }

    /// The findings this backend can answer from.
    pub fn findings(&self) -> &[RecordedFinding] {
        &self.findings
    }

    async fn choose(&self, question: &str) -> Vec<&RecordedFinding> {
        let Some(jev) = &self.jev else {
            return self.findings.iter().collect();
        };
        let (state, questions) = relevance_question(question, &self.findings);
        match jev.system_one(state, questions).await {
            Ok(r) => {
                let scores: Vec<Option<f64>> = self
                    .findings
                    .iter()
                    .map(|f| r.noul(&format!("about_{}", f.id.replace('.', "_"))))
                    .collect();
                pick_relevant(&self.findings, &scores)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Jev unavailable; answering with every finding");
                self.findings.iter().collect()
            }
        }
    }
}

#[async_trait]
impl LlmBackend for FindingsBackend {
    fn provider(&self) -> LlmProvider {
        LlmProvider::A2a
    }

    fn model(&self) -> &str {
        &self.run
    }

    async fn is_available(&self) -> bool {
        true
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec![self.run.clone()])
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, LlmError> {
        let started = Instant::now();
        // The agent wraps the task in a prompt; the question is its description.
        let question = request
            .prompt
            .lines()
            .find_map(|l| l.strip_prefix("Description: "))
            .unwrap_or(&request.prompt);
        let chosen = self.choose(question).await;
        Ok(GenerateResponse {
            content: compose(&self.run, &self.metrics, &chosen),
            provider: LlmProvider::A2a,
            model: self.run.clone(),
            latency: started.elapsed(),
            input_tokens: None,
            output_tokens: None,
            stop_reason: None,
        })
    }
}

/// A mesh agent named `name` answering from the run at `run_dir`.
pub fn research_agent(
    name: &str,
    run_dir: PathBuf,
    jev: Option<JevClient>,
) -> anyhow::Result<LlmAgent> {
    let backend = FindingsBackend::load(&run_dir, jev)?;
    Ok(LlmAgent::new(
        AgentConfig::new(name, AgentRole::Analyst).with_capability(CAPABILITY),
        std::sync::Arc::new(backend),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics() -> Value {
        json!({
            "crypto": {"samples": 10, "windows": 2, "test_predictions": 8,
                       "logloss": {"market": 0.5, "model": 0.6, "blend": 0.49},
                       "paper": {"trades": 3, "pnl_per_trade": -0.01}},
            "macro": {"releases": 5, "by_series": {"CPI": 5}, "windows": 2,
                      "brier": {"kalshi": 0.1, "climatology": 0.2}},
            "findings": [
                ["crypto", {"id": "crypto.odds_calibrated", "label": "Odds are calibrated", "strength": 0.4,
                            "consensus": true, "confirmations": 3, "contradictions": 1}],
                ["macro", {"id": "macro.kalshi_beats_climatology", "label": "Kalshi beats climatology", "strength": 0.1,
                           "consensus": false, "confirmations": 1, "contradictions": 2}]
            ]
        })
    }

    #[test]
    fn test_loads_findings_from_metrics() {
        let fs = load_findings(&metrics());
        assert_eq!(fs.len(), 2);
        assert_eq!(fs[0].track, "crypto");
        assert!(fs[0].consensus);
        assert_eq!(fs[1].contradictions, 2);
    }

    #[test]
    fn test_answer_is_verbatim_from_the_record() {
        let m = metrics();
        let fs = load_findings(&m);
        let answer = compose("quant-full", &m, &[&fs[0]]);
        assert!(answer.contains("Odds are calibrated — SUPPORTED by consensus (strength 0.40; 3 windows confirmed, 1 contradicted)"));
        assert!(answer.contains("Crypto evidence: 10 samples"));
        // Only the chosen finding's track is described.
        assert!(!answer.contains("Macro evidence"));
        assert!(answer.contains("Caveats"));
    }

    #[test]
    fn test_pick_relevant_never_answers_with_nothing() {
        let fs = load_findings(&metrics());
        let ids = |v: Vec<&RecordedFinding>| v.iter().map(|f| f.id.clone()).collect::<Vec<_>>();
        // Jev answered nothing: every finding, not an empty answer.
        assert_eq!(pick_relevant(&fs, &[None, None]).len(), 2);
        // One clearly relevant.
        assert_eq!(
            ids(pick_relevant(&fs, &[Some(0.9), Some(0.1)])),
            vec!["crypto.odds_calibrated"]
        );
        // None clears 0.5: the closest, most relevant first.
        assert_eq!(
            ids(pick_relevant(&fs, &[Some(0.2), Some(0.4)])),
            vec!["macro.kalshi_beats_climatology", "crypto.odds_calibrated"]
        );
        // Partially scored.
        assert_eq!(
            ids(pick_relevant(&fs, &[None, Some(0.8)])),
            vec!["macro.kalshi_beats_climatology"]
        );
    }

    #[test]
    fn test_relevance_question_asks_one_noul_per_finding() {
        let fs = load_findings(&metrics());
        let (state, qs) = relevance_question("are the odds calibrated?", &fs);
        assert_eq!(state["question"], "are the odds calibrated?");
        assert_eq!(qs.len(), 2);
        assert_eq!(qs["about_crypto_odds_calibrated"]["type"], "noul");
    }
}
