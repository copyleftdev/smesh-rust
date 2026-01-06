//! Trust model - Bayesian trust updates
//!
//! Implements the trust dynamics validated in the Python hypothesis.

use serde::{Deserialize, Serialize};

use crate::{MIN_TRUST, MAX_TRUST, DEFAULT_TRUST};

/// Trust model for updating trust scores between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustModel {
    /// Weight for positive evidence
    pub positive_weight: f64,
    /// Weight for negative evidence
    pub negative_weight: f64,
    /// Learning rate for trust updates
    pub learning_rate: f64,
    /// Decay rate for trust scores over time
    pub decay_rate: f64,
}

impl Default for TrustModel {
    fn default() -> Self {
        Self {
            positive_weight: 0.1,
            negative_weight: 0.2,  // Negative evidence weighted more (asymmetric)
            learning_rate: 0.1,
            decay_rate: 0.01,
        }
    }
}

impl TrustModel {
    /// Create a new trust model with custom parameters
    pub fn new(positive_weight: f64, negative_weight: f64) -> Self {
        Self {
            positive_weight,
            negative_weight,
            ..Default::default()
        }
    }
    
    /// Bayesian trust update
    ///
    /// Updates prior belief based on evidence using Bayes' rule:
    /// P(trustworthy | evidence) = P(evidence | trustworthy) * P(trustworthy) / P(evidence)
    ///
    /// # Arguments
    /// * `prior` - Current trust score (0.0 - 1.0)
    /// * `evidence` - Observed evidence (0.0 = negative, 1.0 = positive)
    /// * `weight` - Confidence in the evidence (0.0 - 1.0)
    pub fn bayesian_update(&self, prior: f64, evidence: f64, weight: f64) -> f64 {
        let prior = prior.clamp(MIN_TRUST, MAX_TRUST);
        let evidence = evidence.clamp(0.0, 1.0);
        let weight = weight.clamp(0.0, 1.0);
        
        // Likelihood: P(evidence | trustworthy)
        let likelihood = evidence * weight + (1.0 - evidence) * (1.0 - weight);
        
        // Marginal: P(evidence)
        let marginal = prior * likelihood + (1.0 - prior) * (1.0 - likelihood);
        
        // Posterior: P(trustworthy | evidence)
        if marginal > 0.0 {
            ((prior * likelihood) / marginal).clamp(MIN_TRUST, MAX_TRUST)
        } else {
            prior
        }
    }
    
    /// Simple additive trust update
    pub fn additive_update(&self, current: f64, positive: bool) -> f64 {
        let delta = if positive {
            self.positive_weight * self.learning_rate
        } else {
            -self.negative_weight * self.learning_rate
        };
        
        (current + delta).clamp(MIN_TRUST, MAX_TRUST)
    }
    
    /// Update trust based on signal reinforcement
    ///
    /// If a node's signal gets reinforced, increase trust in that node.
    pub fn update_on_reinforcement(&self, current_trust: f64, reinforcement_count: u32) -> f64 {
        if reinforcement_count == 0 {
            return current_trust;
        }
        
        // Diminishing returns for more reinforcements
        let boost = (reinforcement_count as f64).ln() * self.positive_weight;
        (current_trust + boost * self.learning_rate).clamp(MIN_TRUST, MAX_TRUST)
    }
    
    /// Update trust based on signal validity
    ///
    /// If a signal was later proven correct/incorrect, update trust.
    pub fn update_on_validation(&self, current_trust: f64, was_valid: bool, confidence: f64) -> f64 {
        let evidence = if was_valid { 1.0 } else { 0.0 };
        self.bayesian_update(current_trust, evidence, confidence)
    }
    
    /// Decay trust over time (for inactive relationships)
    pub fn decay(&self, current_trust: f64, time_since_interaction: f64) -> f64 {
        // Trust decays toward default over time
        let decay_factor = (-self.decay_rate * time_since_interaction).exp();
        let decayed = current_trust * decay_factor + DEFAULT_TRUST * (1.0 - decay_factor);
        decayed.clamp(MIN_TRUST, MAX_TRUST)
    }
    
    /// Compute aggregate trust from multiple sources
    ///
    /// Uses weighted average based on each source's own trust.
    pub fn aggregate_trust(&self, observations: &[(f64, f64)]) -> f64 {
        // observations: [(trust_score, observer_trust)]
        if observations.is_empty() {
            return DEFAULT_TRUST;
        }
        
        let total_weight: f64 = observations.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            return DEFAULT_TRUST;
        }
        
        let weighted_sum: f64 = observations
            .iter()
            .map(|(trust, weight)| trust * weight)
            .sum();
        
        (weighted_sum / total_weight).clamp(MIN_TRUST, MAX_TRUST)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bayesian_update_positive() {
        let model = TrustModel::default();
        
        let prior = 0.5;
        let posterior = model.bayesian_update(prior, 1.0, 0.8);
        
        // Positive evidence should increase trust
        assert!(posterior > prior);
    }
    
    #[test]
    fn test_bayesian_update_negative() {
        let model = TrustModel::default();
        
        let prior = 0.5;
        let posterior = model.bayesian_update(prior, 0.0, 0.8);
        
        // Negative evidence should decrease trust
        assert!(posterior < prior);
    }
    
    #[test]
    fn test_trust_bounds() {
        let model = TrustModel::default();
        
        // Even with extreme evidence, trust should stay bounded
        let very_high = model.bayesian_update(0.99, 1.0, 1.0);
        assert!(very_high <= MAX_TRUST);
        
        let very_low = model.bayesian_update(0.01, 0.0, 1.0);
        assert!(very_low >= MIN_TRUST);
    }
    
    #[test]
    fn test_decay() {
        let model = TrustModel::default();
        
        let high_trust = 0.9;
        let decayed = model.decay(high_trust, 100.0);
        
        // Trust should decay toward default
        assert!(decayed < high_trust);
        assert!(decayed > DEFAULT_TRUST);
    }
    
    #[test]
    fn test_aggregate_trust() {
        let model = TrustModel::default();
        
        // Three observers: high trust from trusted observer, low from untrusted
        let observations = vec![
            (0.9, 0.8),  // High trust, trusted observer
            (0.2, 0.3),  // Low trust, less trusted observer
        ];
        
        let aggregate = model.aggregate_trust(&observations);
        
        // Should be weighted toward high trust
        assert!(aggregate > 0.5);
    }
}
