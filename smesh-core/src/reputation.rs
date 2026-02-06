//! Reputation system - advanced reputation tracking
//!
//! Implements Bayesian reputation with decay and behavior fingerprinting.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{NodeId, TrustModel, DEFAULT_TRUST};

/// A single reputation observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationObservation {
    /// The node being observed
    pub node_id: NodeId,
    /// Type of observation
    pub observation_type: ObservationType,
    /// Value of the observation (0.0 - 1.0)
    pub value: f64,
    /// When the observation was made
    pub timestamp: DateTime<Utc>,
    /// Confidence in this observation
    pub confidence: f64,
}

/// Types of reputation observations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationType {
    /// Signal was reinforced by others
    SignalReinforced,
    /// Signal was not reinforced
    SignalIgnored,
    /// Node successfully completed a task
    TaskCompleted,
    /// Node failed to complete a task
    TaskFailed,
    /// Node's signal was validated as correct
    SignalValidated,
    /// Node's signal was invalidated
    SignalInvalidated,
    /// Node exhibited spam behavior
    SpamDetected,
    /// Node cooperated well
    CooperationGood,
}

/// Reputation entry for a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationEntry {
    /// Current reputation score
    pub score: f64,
    /// Number of observations
    pub observation_count: u64,
    /// Last interaction time
    pub last_interaction: DateTime<Utc>,
    /// Bayesian prior (alpha, beta for Beta distribution)
    pub alpha: f64,
    pub beta: f64,
    /// Behavior fingerprint metrics
    pub behavior: BehaviorFingerprint,
}

impl Default for ReputationEntry {
    fn default() -> Self {
        Self {
            score: DEFAULT_TRUST,
            observation_count: 0,
            last_interaction: Utc::now(),
            alpha: 1.0, // Uniform prior
            beta: 1.0,
            behavior: BehaviorFingerprint::default(),
        }
    }
}

impl ReputationEntry {
    /// Get the mean of the Beta distribution
    pub fn bayesian_mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Get the variance of the Beta distribution (uncertainty)
    pub fn bayesian_variance(&self) -> f64 {
        let n = self.alpha + self.beta;
        (self.alpha * self.beta) / (n * n * (n + 1.0))
    }

    /// Update with a new observation using Bayesian update
    pub fn update(&mut self, positive: bool, weight: f64) {
        let w = weight.clamp(0.1, 1.0);

        if positive {
            self.alpha += w;
        } else {
            self.beta += w;
        }

        self.score = self.bayesian_mean();
        self.observation_count += 1;
        self.last_interaction = Utc::now();
    }

    /// Apply time decay to the reputation
    pub fn apply_decay(&mut self, decay_rate: f64, current_time: DateTime<Utc>) {
        let time_since = (current_time - self.last_interaction).num_seconds() as f64;
        let decay_factor = (-decay_rate * time_since / 86400.0).exp(); // Daily decay

        // Decay alpha and beta toward uniform prior
        self.alpha = 1.0 + (self.alpha - 1.0) * decay_factor;
        self.beta = 1.0 + (self.beta - 1.0) * decay_factor;
        self.score = self.bayesian_mean();
    }
}

/// Behavior fingerprint for anomaly detection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BehaviorFingerprint {
    /// Average signals emitted per time unit
    pub emission_rate: f64,
    /// Ratio of reinforcements to sensed signals
    pub reinforcement_ratio: f64,
    /// Average trust given to others
    pub trust_given_mean: f64,
    /// Variance in trust given
    pub trust_given_variance: f64,
    /// Ratio of signals emitted to sensed
    pub emit_to_sense_ratio: f64,
}

impl BehaviorFingerprint {
    /// Check if behavior is anomalous compared to baseline
    pub fn is_anomalous(&self, baseline: &BehaviorFingerprint, threshold: f64) -> bool {
        let emission_diff = (self.emission_rate - baseline.emission_rate).abs();
        let reinforce_diff = (self.reinforcement_ratio - baseline.reinforcement_ratio).abs();
        let trust_diff = (self.trust_given_mean - baseline.trust_given_mean).abs();

        // Simple anomaly score
        let anomaly_score = emission_diff + reinforce_diff + trust_diff;
        anomaly_score > threshold
    }
}

/// Reputation system managing all node reputations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationSystem {
    /// Reputation entries for each node
    pub entries: HashMap<NodeId, ReputationEntry>,
    /// Trust model for updates
    #[serde(skip)]
    pub trust_model: TrustModel,
    /// Decay rate per day
    pub decay_rate: f64,
    /// Baseline behavior for anomaly detection
    pub baseline_behavior: BehaviorFingerprint,
    /// Anomaly threshold
    pub anomaly_threshold: f64,
}

impl Default for ReputationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ReputationSystem {
    /// Create a new reputation system
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            trust_model: TrustModel::default(),
            decay_rate: 0.01,
            baseline_behavior: BehaviorFingerprint::default(),
            anomaly_threshold: 2.0,
        }
    }

    /// Get or create reputation entry for a node
    pub fn get_entry(&mut self, node_id: &str) -> &mut ReputationEntry {
        self.entries.entry(node_id.to_string()).or_default()
    }

    /// Get reputation score for a node
    pub fn get_score(&self, node_id: &str) -> f64 {
        self.entries
            .get(node_id)
            .map(|e| e.score)
            .unwrap_or(DEFAULT_TRUST)
    }

    /// Record an observation about a node
    pub fn record_observation(&mut self, observation: ReputationObservation) {
        let entry = self.get_entry(&observation.node_id);

        let is_positive = match observation.observation_type {
            ObservationType::SignalReinforced => true,
            ObservationType::TaskCompleted => true,
            ObservationType::SignalValidated => true,
            ObservationType::CooperationGood => true,
            ObservationType::SignalIgnored => false,
            ObservationType::TaskFailed => false,
            ObservationType::SignalInvalidated => false,
            ObservationType::SpamDetected => false,
        };

        entry.update(is_positive, observation.confidence);
    }

    /// Update behavior fingerprint for a node
    pub fn update_behavior(
        &mut self,
        node_id: &str,
        signals_emitted: u64,
        signals_sensed: u64,
        signals_reinforced: u64,
        trust_values: &[f64],
        time_elapsed: f64,
    ) {
        let entry = self.get_entry(node_id);

        let time_factor = time_elapsed.max(1.0);

        entry.behavior.emission_rate = signals_emitted as f64 / time_factor;
        entry.behavior.reinforcement_ratio = if signals_sensed > 0 {
            signals_reinforced as f64 / signals_sensed as f64
        } else {
            0.0
        };
        entry.behavior.emit_to_sense_ratio = if signals_sensed > 0 {
            signals_emitted as f64 / signals_sensed as f64
        } else {
            0.0
        };

        if !trust_values.is_empty() {
            let mean: f64 = trust_values.iter().sum::<f64>() / trust_values.len() as f64;
            let variance: f64 = trust_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / trust_values.len() as f64;

            entry.behavior.trust_given_mean = mean;
            entry.behavior.trust_given_variance = variance;
        }
    }

    /// Check if a node's behavior is anomalous
    pub fn is_anomalous(&self, node_id: &str) -> bool {
        if let Some(entry) = self.entries.get(node_id) {
            entry
                .behavior
                .is_anomalous(&self.baseline_behavior, self.anomaly_threshold)
        } else {
            false
        }
    }

    /// Apply decay to all reputation entries
    pub fn apply_global_decay(&mut self) {
        let now = Utc::now();
        for entry in self.entries.values_mut() {
            entry.apply_decay(self.decay_rate, now);
        }
    }

    /// Get top N nodes by reputation
    pub fn top_nodes(&self, n: usize) -> Vec<(&NodeId, f64)> {
        let mut sorted: Vec<_> = self
            .entries
            .iter()
            .map(|(id, entry)| (id, entry.score))
            .collect();

        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Get bottom N nodes by reputation (potential bad actors)
    pub fn bottom_nodes(&self, n: usize) -> Vec<(&NodeId, f64)> {
        let mut sorted: Vec<_> = self
            .entries
            .iter()
            .map(|(id, entry)| (id, entry.score))
            .collect();

        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Compute network-wide reputation statistics
    pub fn stats(&self) -> ReputationStats {
        if self.entries.is_empty() {
            return ReputationStats::default();
        }

        let scores: Vec<f64> = self.entries.values().map(|e| e.score).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;

        ReputationStats {
            node_count: self.entries.len(),
            mean_score: mean,
            variance,
            min_score: scores.iter().cloned().fold(f64::INFINITY, f64::min),
            max_score: scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// Statistics about the reputation system
#[derive(Debug, Clone, Default)]
pub struct ReputationStats {
    pub node_count: usize,
    pub mean_score: f64,
    pub variance: f64,
    pub min_score: f64,
    pub max_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reputation_entry_update() {
        let mut entry = ReputationEntry::default();

        // Initial score should be around 0.5 (uniform prior)
        assert!((entry.score - 0.5).abs() < 0.01);

        // Add positive observations
        entry.update(true, 1.0);
        entry.update(true, 1.0);

        // Score should increase
        assert!(entry.score > 0.5);
    }

    #[test]
    fn test_reputation_system_observations() {
        let mut system = ReputationSystem::new();

        // Add positive observation
        system.record_observation(ReputationObservation {
            node_id: "node1".to_string(),
            observation_type: ObservationType::SignalReinforced,
            value: 1.0,
            timestamp: Utc::now(),
            confidence: 0.8,
        });

        let score = system.get_score("node1");
        assert!(score > DEFAULT_TRUST);

        // Add negative observation
        system.record_observation(ReputationObservation {
            node_id: "node1".to_string(),
            observation_type: ObservationType::SpamDetected,
            value: 0.0,
            timestamp: Utc::now(),
            confidence: 0.9,
        });

        let new_score = system.get_score("node1");
        assert!(new_score < score);
    }

    #[test]
    fn test_top_bottom_nodes() {
        let mut system = ReputationSystem::new();

        // Add nodes with different reputations
        for i in 0..5 {
            let node_id = format!("node{}", i);
            for _ in 0..i {
                system.record_observation(ReputationObservation {
                    node_id: node_id.clone(),
                    observation_type: ObservationType::TaskCompleted,
                    value: 1.0,
                    timestamp: Utc::now(),
                    confidence: 1.0,
                });
            }
        }

        let top = system.top_nodes(2);
        let bottom = system.bottom_nodes(2);

        assert_eq!(top.len(), 2);
        assert_eq!(bottom.len(), 2);
        assert!(top[0].1 >= top[1].1);
        assert!(bottom[0].1 <= bottom[1].1);
    }
}
