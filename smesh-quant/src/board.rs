//! Hypotheses as SMESH signals: proposed, reinforced, contradicted, doubted
//! and left to decay, so that what survives is the mesh's consensus.
//!
//! Each hypothesis is one signal on a [`smesh_core::Field`]. The field's clock
//! advances one unit per walk-forward window. A hypothesis's strength is the
//! signal's effective intensity, the product of three things:
//!
//! - **Recency (intensity).** It decays exponentially on the field's clock and
//!   is restored whenever a window confirms the claim, so a claim no recent
//!   window supports fades even if it once did.
//! - **Evidence (confidence).** The margin of confirming over contradicting
//!   windows, shrunk towards zero when there are few of them:
//!   `max(0, (for - against) / (for + against + SHRINK))`. A coin-flip record
//!   earns nothing; ten windows for and none against earn 0.63.
//! - **The skeptic's doubt.** A factor on the window it is raised in (small
//!   samples, too few trades). It does not compound across windows.
//!
//! smesh-core's reinforcement boost (up to +50% for many distinct
//! reinforcing windows) multiplies the result, capped at 1. Every change is
//! journaled for the visualization.
//!
//! An earlier version cut confidence by a constant factor on every
//! contradiction while confirmations added a shrinking boost. Contradictions
//! then compounded without limit: a claim confirmed in 25 windows and
//! contradicted in 8 ended at 0.14. Evidence now counts symmetrically.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Value};
use smesh_core::{DecayFunction, Field, Signal, SignalType};
use smesh_runtime::Journal;

/// Decay per window without a confirmation: e^-0.35 = 0.70 a window.
pub const DECAY_PER_WINDOW: f64 = 0.35;
/// Pseudo-windows that shrink confidence when there is little evidence.
pub const SHRINK: f64 = 6.0;
/// Strength at or above which a hypothesis is a finding.
pub const CONSENSUS_THRESHOLD: f64 = 0.30;

#[derive(Debug, Clone)]
struct Hypothesis {
    hash: String,
    label: String,
    confirmations: u32,
    contradictions: u32,
    /// The skeptic's doubt this window (1 = none).
    doubt: f64,
}

impl Hypothesis {
    fn confidence(&self) -> f64 {
        let (c, k) = (self.confirmations as f64, self.contradictions as f64);
        ((c - k) / (c + k + SHRINK)).max(0.0) * self.doubt
    }
}

/// A hypothesis at the end of the run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Finding {
    /// Hypothesis id.
    pub id: String,
    /// What it claims.
    pub label: String,
    /// Final strength.
    pub strength: f64,
    /// Final confidence.
    pub confidence: f64,
    /// Windows that confirmed it.
    pub confirmations: u32,
    /// Windows that contradicted it.
    pub contradictions: u32,
    /// Whether it reached the consensus threshold.
    pub consensus: bool,
}

/// The hypothesis field for one research track.
pub struct Board {
    name: String,
    field: Field,
    journal: Arc<Journal>,
    hypotheses: BTreeMap<String, Hypothesis>,
    window: u64,
}

impl Board {
    /// A board whose own events go to `journal` (the field's node).
    pub fn new(name: &str, journal: Arc<Journal>) -> Self {
        Self {
            name: name.to_string(),
            field: Field::new(),
            journal,
            hypotheses: BTreeMap::new(),
            window: 0,
        }
    }

    fn signal(&self, id: &str) -> Option<&Signal> {
        self.field.get_signal(&self.hypotheses.get(id)?.hash)
    }

    fn strength(&self, id: &str) -> (f64, f64, u32) {
        self.signal(id)
            .map(|s| {
                (
                    s.effective_intensity(self.field.current_time),
                    s.confidence,
                    s.reinforcement_count,
                )
            })
            .unwrap_or((0.0, 0.0, 0))
    }

    fn sync_confidence(&mut self, id: &str) {
        let Some(h) = self.hypotheses.get(id) else {
            return;
        };
        let (hash, confidence) = (h.hash.clone(), h.confidence());
        if let Some(s) = self.field.get_signal_mut(&hash) {
            s.confidence = confidence;
        }
    }

    fn record(&self, by: &Journal, kind: &str, id: &str, extra: Value) {
        let (intensity, confidence, reinforcements) = self.strength(id);
        let (c, k) = self
            .hypotheses
            .get(id)
            .map(|h| (h.confirmations, h.contradictions))
            .unwrap_or((0, 0));
        let mut data = json!({
            "board": self.name, "id": id, "window": self.window,
            "intensity": intensity, "confidence": confidence,
            "reinforcements": reinforcements, "for": c, "against": k, "by": by.node(),
        });
        if let (Value::Object(d), Value::Object(e)) = (&mut data, extra) {
            d.extend(e);
        }
        by.record(kind, data);
    }

    /// Propose a hypothesis. Every hypothesis starts equal: full intensity
    /// and no confidence until a window has tested it. Proposing one that
    /// exists does nothing.
    pub fn propose(&mut self, by: &Journal, id: &str, label: &str, why: Value) {
        if self.hypotheses.contains_key(id) {
            return;
        }
        let signal = Signal::builder(SignalType::Custom)
            .payload(format!("{}:{id}", self.name).into_bytes())
            .intensity(1.0)
            .confidence(0.0)
            .decay_function(DecayFunction::Exponential)
            .decay_rate(DECAY_PER_WINDOW)
            .ttl(1e6)
            .origin(by.node())
            .build();
        let hash = self.field.emit_anonymous(signal);
        self.hypotheses.insert(
            id.to_string(),
            Hypothesis {
                hash,
                label: label.to_string(),
                confirmations: 0,
                contradictions: 0,
                doubt: 1.0,
            },
        );
        by.record(
            "entity",
            json!({"id": id, "type": "hypothesis", "label": label, "group": self.name}),
        );
        self.record(by, "signal.emit", id, json!({"label": label, "why": why}));
    }

    /// A window agrees: reinforce, restore intensity, recount confidence.
    pub fn confirm(&mut self, by: &Journal, id: &str, evidence: Value) {
        let Some(h) = self.hypotheses.get_mut(id) else {
            return;
        };
        h.confirmations += 1;
        let hash = h.hash.clone();
        let now = self.field.current_time;
        let reinforcer = format!("{}#w{}", by.node(), self.window);
        if let Some(s) = self.field.get_signal_mut(&hash) {
            s.reinforce(&reinforcer);
            s.created_at = now;
            s.current_intensity = s.intensity;
        }
        self.sync_confidence(id);
        self.record(by, "signal.reinforce", id, json!({"evidence": evidence}));
    }

    /// A window disagrees: recount confidence. Intensity is left to decay.
    pub fn contradict(&mut self, by: &Journal, id: &str, evidence: Value) {
        let Some(h) = self.hypotheses.get_mut(id) else {
            return;
        };
        h.contradictions += 1;
        self.sync_confidence(id);
        self.record(by, "signal.contradict", id, json!({"evidence": evidence}));
    }

    /// The skeptic's doubt for this window: confidence is multiplied by
    /// `factor` until the next tick. Several doubts in one window keep the
    /// strongest, not their product.
    pub fn dampen(&mut self, by: &Journal, id: &str, factor: f64, reason: &str) {
        let Some(h) = self.hypotheses.get_mut(id) else {
            return;
        };
        h.doubt = h.doubt.min(factor.clamp(0.0, 1.0));
        self.sync_confidence(id);
        self.record(
            by,
            "signal.dampen",
            id,
            json!({"factor": factor, "reason": reason}),
        );
    }

    /// Advance one window: doubts lapse, every hypothesis decays, and each
    /// one's new strength is journaled (one `signal.tick` per hypothesis).
    pub fn tick(&mut self) {
        self.window += 1;
        self.field.tick(1.0);
        let ids: Vec<String> = self.hypotheses.keys().cloned().collect();
        for id in &ids {
            if let Some(h) = self.hypotheses.get_mut(id) {
                h.doubt = 1.0;
            }
            self.sync_confidence(id);
        }
        let journal = Arc::clone(&self.journal);
        for id in ids {
            if self.signal(&id).is_some() {
                self.record(&journal, "signal.tick", &id, json!({}));
            } else {
                self.record(&journal, "signal.expire", &id, json!({}));
            }
        }
    }

    /// Current strength of every hypothesis, strongest first.
    pub fn findings(&self) -> Vec<Finding> {
        let mut out: Vec<Finding> = self
            .hypotheses
            .iter()
            .map(|(id, h)| {
                let (strength, confidence, _) = self.strength(id);
                Finding {
                    id: id.clone(),
                    label: h.label.clone(),
                    strength,
                    confidence,
                    confirmations: h.confirmations,
                    contradictions: h.contradictions,
                    consensus: strength >= CONSENSUS_THRESHOLD,
                }
            })
            .collect();
        out.sort_by(|a, b| b.strength.total_cmp(&a.strength));
        out
    }

    /// Journal the final state: one `finding` per hypothesis.
    pub fn conclude(&self) -> Vec<Finding> {
        let findings = self.findings();
        for f in &findings {
            self.journal.record(
                "finding",
                json!({"board": self.name, "id": f.id, "label": f.label,
                       "strength": f.strength, "confidence": f.confidence,
                       "confirmations": f.confirmations, "contradictions": f.contradictions,
                       "consensus": f.consensus}),
            );
        }
        findings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board() -> (Board, Arc<Journal>) {
        let j = Journal::disabled();
        (Board::new("test", Arc::clone(&j)), j)
    }

    /// Run `pattern` (true = confirm) one window at a time.
    fn run(b: &mut Board, j: &Journal, id: &str, pattern: &[bool]) {
        for &ok in pattern {
            b.tick();
            if ok {
                b.confirm(j, id, json!({}));
            } else {
                b.contradict(j, id, json!({}));
            }
        }
    }

    fn get(b: &Board, id: &str) -> Finding {
        b.findings().into_iter().find(|f| f.id == id).unwrap()
    }

    #[test]
    fn test_an_untested_claim_has_no_strength() {
        let (mut b, j) = board();
        b.propose(&j, "h", "claim", json!({}));
        assert_eq!(get(&b, "h").strength, 0.0);
    }

    #[test]
    fn test_a_mostly_right_claim_reaches_consensus_despite_contradictions() {
        // The case the old multiplicative rule got wrong: 25 for, 8 against.
        let (mut b, j) = board();
        b.propose(&j, "h", "claim", json!({}));
        let mut pattern = vec![true; 25];
        for i in 0..8 {
            pattern.insert(i * 4, false);
        }
        pattern.push(true);
        run(&mut b, &j, "h", &pattern);
        let f = get(&b, "h");
        assert!(f.consensus, "{f:?}");
    }

    #[test]
    fn test_a_coin_flip_record_never_reaches_consensus() {
        let (mut b, j) = board();
        b.propose(&j, "h", "claim", json!({}));
        let pattern: Vec<bool> = (0..32).map(|i| i % 2 == 0).collect();
        run(&mut b, &j, "h", &pattern);
        assert!(!get(&b, "h").consensus);
    }

    #[test]
    fn test_one_lucky_window_is_not_consensus() {
        let (mut b, j) = board();
        b.propose(&j, "h", "claim", json!({}));
        run(&mut b, &j, "h", &[true]);
        assert!(!get(&b, "h").consensus, "{:?}", get(&b, "h"));
    }

    #[test]
    fn test_a_claim_no_recent_window_supports_fades() {
        let (mut b, j) = board();
        b.propose(&j, "h", "claim", json!({}));
        run(&mut b, &j, "h", &[true; 10]);
        let before = get(&b, "h").strength;
        for _ in 0..5 {
            b.tick();
        }
        assert!(get(&b, "h").strength < before * 0.25);
    }

    #[test]
    fn test_doubt_lapses_after_its_window() {
        let (mut b, j) = board();
        b.propose(&j, "h", "claim", json!({}));
        run(&mut b, &j, "h", &[true; 10]);
        let clear = get(&b, "h").confidence;
        b.dampen(&j, "h", 0.5, "small sample");
        b.dampen(&j, "h", 0.9, "also small");
        assert!((get(&b, "h").confidence - clear * 0.5).abs() < 1e-12);
        b.tick();
        assert!((get(&b, "h").confidence - clear).abs() < 1e-12);
    }
}
