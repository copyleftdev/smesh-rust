//! Node primitive - entities in the SMESH network
//!
//! Nodes can emit signals, sense the field, and maintain trust relationships.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::{Signal, DEFAULT_TRUST, MAX_TRUST, MIN_TRUST};

/// Unique identifier for a node
pub type NodeId = String;

/// Configuration for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Signal emission rate (signals per tick capacity)
    pub emission_rate: f64,
    /// Minimum intensity to perceive a signal
    pub sensing_threshold: f64,
    /// Minimum confidence to reinforce a signal
    pub reinforcement_threshold: f64,
    /// When to trigger SMESH+ escalation
    pub escalation_threshold: f64,
    /// Maximum concurrent tasks (for agent nodes)
    pub max_concurrent_tasks: usize,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            emission_rate: 1.0,
            sensing_threshold: 0.1,
            reinforcement_threshold: 0.5,
            escalation_threshold: 0.8,
            max_concurrent_tasks: 3,
        }
    }
}

/// A node in the SMESH network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier
    pub id: NodeId,

    /// Public key for identity verification
    pub public_key: String,

    /// Relative compute capacity
    pub compute_capacity: f64,

    /// Relative bandwidth capacity
    pub bandwidth_capacity: f64,

    /// Trust scores for other nodes (node_id -> trust_score)
    pub trust_scores: HashMap<NodeId, f64>,

    /// Node configuration
    pub config: NodeConfig,

    /// Whether this node is malicious (for simulation)
    pub is_malicious: bool,

    /// Type of malicious behavior (for simulation)
    pub malicious_behavior: MaliciousBehavior,

    /// Statistics
    pub stats: NodeStats,
}

/// Types of malicious behavior for simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MaliciousBehavior {
    #[default]
    None,
    /// Emits excessive signals
    Spam,
    /// Reinforces signals without verification
    FalseReinforce,
    /// Creates fake identities
    Sybil,
    /// Selectively drops signals
    Eclipse,
}

/// Statistics tracked by a node
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeStats {
    pub signals_emitted: u64,
    pub signals_sensed: u64,
    pub signals_reinforced: u64,
    pub signals_relayed: u64,
    pub escalations_triggered: u64,
}

impl Node {
    /// Create a new node with default configuration
    pub fn new() -> Self {
        Self::with_config(NodeConfig::default())
    }

    /// Create a new node with custom configuration
    pub fn with_config(config: NodeConfig) -> Self {
        let id = Uuid::new_v4().to_string()[..8].to_string();
        Self {
            id: id.clone(),
            public_key: Uuid::new_v4().to_string(),
            compute_capacity: 1.0,
            bandwidth_capacity: 1.0,
            trust_scores: HashMap::new(),
            config,
            is_malicious: false,
            malicious_behavior: MaliciousBehavior::None,
            stats: NodeStats::default(),
        }
    }

    /// Create a node with a specific ID
    pub fn with_id(id: &str) -> Self {
        let mut node = Self::new();
        node.id = id.to_string();
        node
    }

    /// Get trust score for another node
    pub fn get_trust(&self, other_node_id: &str) -> f64 {
        *self
            .trust_scores
            .get(other_node_id)
            .unwrap_or(&DEFAULT_TRUST)
    }

    /// Update trust score for another node
    pub fn update_trust(&mut self, other_node_id: &str, delta: f64) -> f64 {
        let current = self.get_trust(other_node_id);
        let new_trust = (current + delta).clamp(MIN_TRUST, MAX_TRUST);
        self.trust_scores
            .insert(other_node_id.to_string(), new_trust);
        new_trust
    }

    /// Set trust score for another node directly
    pub fn set_trust(&mut self, other_node_id: &str, trust: f64) {
        let clamped = trust.clamp(MIN_TRUST, MAX_TRUST);
        self.trust_scores.insert(other_node_id.to_string(), clamped);
    }

    /// Decide whether to reinforce a signal
    pub fn should_reinforce(&self, signal: &Signal, local_evidence: f64) -> bool {
        if self.is_malicious && self.malicious_behavior == MaliciousBehavior::FalseReinforce {
            return true; // Malicious nodes reinforce everything
        }

        let origin_trust = self.get_trust(&signal.origin_node_id);
        let evidence_score = local_evidence + origin_trust * signal.confidence;
        evidence_score >= self.config.reinforcement_threshold
    }

    /// Decide whether to relay a signal and with what dampening
    pub fn should_relay(&self, signal: &Signal, remaining_hops: u32) -> (bool, f64) {
        if remaining_hops == 0 {
            return (false, 0.0);
        }

        let origin_trust = self.get_trust(&signal.origin_node_id);
        let effective = signal.confidence * signal.current_intensity;

        // Propagation score
        let prop_score = effective * origin_trust * (remaining_hops as f64 / signal.radius as f64);

        // Probabilistic relay decision
        let should_relay = rand::random::<f64>() < prop_score;
        let dampening = if origin_trust > 0.7 { 0.9 } else { 0.7 };

        (should_relay, dampening)
    }

    /// Decide whether to trigger SMESH+ escalation
    pub fn should_escalate(&self, signals: &[Signal]) -> bool {
        if signals.is_empty() {
            return false;
        }

        let max_confidence = signals.iter().map(|s| s.confidence).fold(0.0, f64::max);
        let max_reinforcements = signals
            .iter()
            .map(|s| s.reinforcement_count)
            .max()
            .unwrap_or(0);

        max_confidence >= self.config.escalation_threshold && max_reinforcements >= 2
    }

    /// Check if this node can sense a signal (above threshold)
    pub fn can_sense(&self, signal: &Signal) -> bool {
        signal.current_intensity >= self.config.sensing_threshold
    }

    /// Mark this node as malicious (for simulation)
    pub fn make_malicious(&mut self, behavior: MaliciousBehavior) {
        self.is_malicious = true;
        self.malicious_behavior = behavior;
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SignalType;

    #[test]
    fn test_node_creation() {
        let node = Node::new();
        assert!(!node.id.is_empty());
        assert!(!node.public_key.is_empty());
        assert_eq!(node.compute_capacity, 1.0);
    }

    #[test]
    fn test_trust_management() {
        let mut node = Node::new();

        // Default trust
        assert_eq!(node.get_trust("unknown"), DEFAULT_TRUST);

        // Update trust
        node.update_trust("peer1", 0.2);
        assert!((node.get_trust("peer1") - 0.7).abs() < 0.01);

        // Trust bounds
        node.update_trust("peer1", 1.0);
        assert!(node.get_trust("peer1") <= MAX_TRUST);

        node.update_trust("peer2", -1.0);
        node.update_trust("peer2", -1.0);
        assert!(node.get_trust("peer2") >= MIN_TRUST);
    }

    #[test]
    fn test_malicious_behavior() {
        let mut node = Node::new();
        node.make_malicious(MaliciousBehavior::FalseReinforce);

        let signal = Signal::builder(SignalType::Data).confidence(0.1).build();

        // Malicious node should reinforce even low-confidence signals
        assert!(node.should_reinforce(&signal, 0.0));
    }
}
