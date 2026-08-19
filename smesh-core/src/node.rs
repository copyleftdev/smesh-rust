//! Node primitive - entities in the SMESH network
//!
//! Nodes can emit signals, sense the field, and maintain trust relationships.

use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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

/// The full reasoning behind a relay choice.
///
/// Relaying is probabilistic: `relay` is `roll < propagation_score`. Recording
/// both makes an otherwise unreproducible decision auditable after the fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayDecision {
    /// Whether the signal is forwarded.
    pub relay: bool,
    /// Intensity multiplier applied to the forwarded copy.
    pub dampening: f64,
    /// Probability the relay was granted with.
    pub propagation_score: f64,
    /// This node's trust in the signal's origin.
    pub origin_trust: f64,
    /// The draw that resolved the decision.
    pub roll: f64,
    /// Hops left in the signal's budget when it was considered.
    pub remaining_hops: u32,
    /// Set when the signal was refused outright, before any roll.
    pub veto: Option<String>,
}

impl Node {
    /// Create a new node with default configuration
    pub fn new() -> Self {
        Self::with_config(NodeConfig::default())
    }

    /// Create a new node with custom configuration
    pub fn with_config(config: NodeConfig) -> Self {
        let id = Uuid::new_v4().to_string()[..8].to_string();

        // Generate cryptographic public key using SHA256 hash of random bytes
        // In production, this should be replaced with proper asymmetric key generation (e.g., Ed25519)
        let mut rng = rand::thread_rng();
        let random_bytes: [u8; 32] = rng.gen();
        let mut hasher = Sha256::new();
        hasher.update(random_bytes);
        let public_key = format!("{:x}", hasher.finalize());

        Self {
            id: id.clone(),
            public_key,
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
        let decision = self.relay_decision(signal, remaining_hops);
        (decision.relay, decision.dampening)
    }

    /// Decide whether to relay a signal, returning the full reasoning.
    ///
    /// [`Node::should_relay`] is the terse form. This one exposes the score,
    /// the trust that fed it and the die roll that resolved it, so a relay
    /// choice can be journalled and replayed rather than merely observed.
    pub fn relay_decision(&self, signal: &Signal, remaining_hops: u32) -> RelayDecision {
        let origin_trust = self.get_trust(&signal.origin_node_id);
        let dampening = if origin_trust > 0.7 { 0.9 } else { 0.7 };

        let vetoed = |reason: &str| RelayDecision {
            relay: false,
            dampening: 0.0,
            propagation_score: 0.0,
            origin_trust,
            roll: 0.0,
            remaining_hops,
            veto: Some(reason.to_string()),
        };

        if remaining_hops == 0 {
            return vetoed("hop budget exhausted");
        }

        // Eclipse attackers black-hole traffic: they accept signals but never
        // forward them, blocking diffusion paths that route through them.
        if self.is_malicious && self.malicious_behavior == MaliciousBehavior::Eclipse {
            return vetoed("eclipse node black-holes traffic");
        }

        let effective = signal.confidence * signal.current_intensity;

        // Propagation score
        let propagation_score =
            effective * origin_trust * (remaining_hops as f64 / signal.radius as f64);

        // Probabilistic relay decision using cryptographically secure RNG
        let mut rng = rand::thread_rng();
        let roll = rng.gen::<f64>();

        RelayDecision {
            relay: roll < propagation_score,
            dampening,
            propagation_score,
            origin_trust,
            roll,
            remaining_hops,
            veto: None,
        }
    }

    /// Everyone who attests to a signal: its origin plus every reinforcer.
    ///
    /// Reinforcement is an *independent* attestation to the same claim, so the
    /// size of this set is how many parties corroborate it. Relaying a signal
    /// does not put you in it — only asserting it does.
    pub fn attesters(signal: &Signal) -> Vec<String> {
        let mut out = Vec::with_capacity(signal.reinforced_by.len() + 1);
        if !signal.origin_node_id.is_empty() {
            out.push(signal.origin_node_id.clone());
        }
        for id in &signal.reinforced_by {
            if !out.contains(id) {
                out.push(id.clone());
            }
        }
        out
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

    #[test]
    fn test_eclipse_drops_signals() {
        let mut node = Node::new();
        // A strong signal an honest node would relay with high probability.
        let mut signal = Signal::builder(SignalType::Data)
            .confidence(1.0)
            .intensity(1.0)
            .build();
        signal.current_intensity = 1.0;

        node.make_malicious(MaliciousBehavior::Eclipse);
        // Eclipse attackers never relay, no matter how strong the signal.
        for _ in 0..50 {
            assert_eq!(node.should_relay(&signal, 5), (false, 0.0));
        }
    }
}
