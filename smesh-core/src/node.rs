//! Node primitive - entities in the SMESH network
//!
//! Nodes can emit signals, sense the field, and maintain trust relationships.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use std::sync::Arc;

use crate::identity::NodeIdentity;
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

    /// Public key for identity verification, hex encoded.
    pub public_key: String,

    /// The private half, present only for a node this process actually is.
    ///
    /// Skipped by serde in both directions: a secret key must never reach a
    /// journal, a snapshot or the wire. A node decoded from any of those is a
    /// *view* of a peer and cannot sign, which is exactly right.
    #[serde(skip)]
    pub identity: Option<Arc<NodeIdentity>>,

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

        // A real Ed25519 keypair. This used to be the SHA-256 of some random
        // bytes, which looked like a key and could not verify anything: there
        // was no private half, so nothing could ever be signed with it.
        let identity = NodeIdentity::generate_named(id.clone());
        let public_key = identity.public_key_hex();

        Self {
            id: id.clone(),
            identity: Some(Arc::new(identity)),
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
        let roll = rand::thread_rng().gen::<f64>();
        self.relay_decision_with(signal, remaining_hops, roll)
    }

    /// The relay decision with the draw supplied by the caller.
    ///
    /// Relaying is the protocol's only genuine coin flip, and hiding the draw
    /// inside this function made the whole diffusion path impossible to replay.
    /// Taking it as an argument makes the decision a pure function of state: a
    /// simulation can sweep seeds, and a failing schedule can be reproduced
    /// exactly rather than described.
    pub fn relay_decision_with(
        &self,
        signal: &Signal,
        remaining_hops: u32,
        roll: f64,
    ) -> RelayDecision {
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

    /// A node with a chosen name and a keypair that signs under that name.
    ///
    /// Prefer this over assigning to `id` after construction: the identity is
    /// generated with the name baked into it, so renaming the node afterwards
    /// leaves it signing under a name it no longer presents, and its
    /// attestations stop counting toward the name anyone else sees.
    pub fn named(id: impl Into<NodeId>) -> Self {
        Self::new().with_identity(NodeIdentity::generate_named(id))
    }

    /// Whether this node's signing key matches the name it presents.
    ///
    /// False after `node.id` has been reassigned without the identity, which is
    /// the one way to end up signing under the wrong name.
    pub fn identity_matches_name(&self) -> bool {
        self.identity
            .as_ref()
            .is_some_and(|identity| identity.node_id() == self.id)
    }

    /// Adopt a specific identity, replacing the generated one.
    ///
    /// Use when a node's name is chosen rather than derived, so that its
    /// signatures are made under the name it presents.
    pub fn with_identity(mut self, identity: NodeIdentity) -> Self {
        self.id = identity.node_id().to_string();
        self.public_key = identity.public_key_hex();
        self.identity = Some(Arc::new(identity));
        self
    }

    /// Whether this node would relay, for a given draw. Pure.
    pub fn would_relay(&self, signal: &Signal, remaining_hops: u32, roll: f64) -> bool {
        self.relay_decision_with(signal, remaining_hops, roll).relay
    }

    /// Sign a signal on this node's behalf, if it holds a private key.
    ///
    /// Refuses when the key signs under a different name than the node
    /// presents, because such a signature verifies but attributes the claim to
    /// a name nobody is listening for.
    pub fn attest(&self, signal: &mut Signal) {
        if !self.identity_matches_name() {
            return;
        }
        if let Some(identity) = &self.identity {
            signal.attest(identity);
        }
    }

    /// Everyone who attests to a signal: its origin plus every reinforcer.
    ///
    /// This is the *local* view, and it trusts the names it is given. It is
    /// correct for a single-process simulation, where nothing is adversarial.
    /// Anything that came off a network should be counted with
    /// [`Signal::verified_attesters`] instead, which counts signatures.
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

    // ---- should_reinforce: the evidence threshold ---------------------------

    #[test]
    fn should_reinforce_weighs_evidence_against_the_threshold() {
        // Honest node, unknown origin => origin_trust is DEFAULT_TRUST (0.5),
        // reinforcement_threshold is 0.5.
        let node = Node::new();
        let signal = Signal::builder(SignalType::Data).confidence(0.8).build();

        // 0.0 + 0.5*0.8 = 0.4 < 0.5 -> no.
        assert!(!node.should_reinforce(&signal, 0.0));
        // 0.2 + 0.5*0.8 = 0.6 >= 0.5 -> yes. This case also separates `+` from
        // `-` (0.2-0.4=-0.2, no) and `+` from `*` (0.2*0.4=0.08, no).
        assert!(node.should_reinforce(&signal, 0.2));

        // Exact boundary: confidence 1.0 => 0.5*1.0 = 0.5 >= 0.5 -> yes.
        // Kills `>=`→`<`, which would flip this to no.
        let strong = Signal::builder(SignalType::Data).confidence(1.0).build();
        assert!(node.should_reinforce(&strong, 0.0));

        // origin_trust * confidence, not +/÷: a weak signal (0.5*0.1 = 0.05)
        // stays under threshold, where `+` (0.6) or `/` (5.0) would clear it.
        let weak = Signal::builder(SignalType::Data).confidence(0.1).build();
        assert!(!node.should_reinforce(&weak, 0.0));
    }

    #[test]
    fn malicious_false_reinforce_is_an_or_not_reached_by_honest_nodes() {
        // A malicious node whose behavior is NOT FalseReinforce must still fall
        // through to the evidence test. If the guard's `&&` became `||`, an
        // Eclipse node would suddenly reinforce everything.
        let mut node = Node::new();
        node.make_malicious(MaliciousBehavior::Eclipse);
        let weak = Signal::builder(SignalType::Data).confidence(0.1).build();
        assert!(
            !node.should_reinforce(&weak, 0.0),
            "only FalseReinforce shortcuts the threshold, not any malice"
        );
    }

    // ---- relay_decision_with: a pure function of state ----------------------

    #[test]
    fn relay_score_and_roll_boundary_are_exact() {
        let mut node = Node::new();
        node.set_trust("origin", 0.8);
        let mut signal = Signal::builder(SignalType::Data)
            .confidence(1.0)
            .radius(4)
            .build();
        signal.origin_node_id = "origin".to_string();
        signal.current_intensity = 1.0;

        // effective(1.0) * trust(0.8) * (hops 2 / radius 4 = 0.5) = 0.4.
        // Pinning the score kills every arithmetic mutant in it at once.
        let d = node.relay_decision_with(&signal, 2, 0.3);
        assert!((d.propagation_score - 0.4).abs() < 1e-9);
        assert!(d.relay, "roll 0.3 < score 0.4 relays");

        // Roll exactly equal to the score must NOT relay (`<`, not `<=`).
        let edge = node.relay_decision_with(&signal, 2, 0.4);
        assert!(!edge.relay, "roll == score does not relay");

        let over = node.relay_decision_with(&signal, 2, 0.5);
        assert!(!over.relay);
    }

    #[test]
    fn relay_dampening_switches_on_high_trust() {
        let mut node = Node::new();
        let mut signal = Signal::builder(SignalType::Data).confidence(1.0).build();
        signal.current_intensity = 1.0;
        signal.origin_node_id = "o".to_string();

        // trust > 0.7 -> 0.9, else 0.7. Test both sides and the boundary.
        node.set_trust("o", 0.8);
        assert_eq!(node.relay_decision_with(&signal, 5, 0.99).dampening, 0.9);
        node.set_trust("o", 0.5);
        assert_eq!(node.relay_decision_with(&signal, 5, 0.99).dampening, 0.7);
        // Exactly 0.7 is NOT greater than 0.7 -> low dampening (kills `>`→`>=`).
        node.set_trust("o", 0.7);
        assert_eq!(node.relay_decision_with(&signal, 5, 0.99).dampening, 0.7);
    }

    #[test]
    fn relay_vetoes_when_the_hop_budget_is_gone() {
        let node = Node::new();
        let signal = Signal::builder(SignalType::Data).confidence(1.0).build();
        // Zero remaining hops is a hard veto regardless of the roll.
        let d = node.relay_decision_with(&signal, 0, 0.0);
        assert!(!d.relay && d.veto.is_some());
    }

    // ---- attesters: origin plus unique reinforcers --------------------------

    #[test]
    fn attesters_lists_origin_first_then_dedups_reinforcers() {
        let mut signal = Signal::builder(SignalType::Data).build();
        signal.origin_node_id = "origin".to_string();
        signal.reinforced_by = vec!["r1".into(), "r2".into(), "r1".into()];

        let a = Node::attesters(&signal);
        assert_eq!(
            a,
            vec!["origin", "r1", "r2"],
            "origin leads, duplicates drop"
        );

        // An empty origin is not listed (the `!is_empty` guard).
        let mut anon = Signal::builder(SignalType::Data).build();
        anon.origin_node_id = String::new();
        anon.reinforced_by = vec!["r1".into()];
        assert_eq!(Node::attesters(&anon), vec!["r1"]);
    }

    // ---- can_sense: intensity above the sensing threshold -------------------

    #[test]
    fn can_sense_respects_the_threshold_boundary() {
        let node = Node::new(); // sensing_threshold 0.1
        let mut signal = Signal::builder(SignalType::Data).build();

        signal.current_intensity = 0.0;
        assert!(!node.can_sense(&signal), "silence is not sensed");
        signal.current_intensity = 0.1;
        assert!(node.can_sense(&signal), "exactly at threshold is sensed");
        signal.current_intensity = 0.05;
        assert!(!node.can_sense(&signal), "below threshold is not sensed");
    }

    // ---- set_trust: direct assignment, clamped ------------------------------

    #[test]
    fn set_trust_assigns_and_clamps() {
        let mut node = Node::new();
        node.set_trust("peer", 0.42);
        assert!(
            (node.get_trust("peer") - 0.42).abs() < 1e-9,
            "value is stored"
        );
        node.set_trust("peer", 5.0);
        assert_eq!(node.get_trust("peer"), MAX_TRUST, "clamped to the ceiling");
        node.set_trust("peer", -5.0);
        assert_eq!(node.get_trust("peer"), MIN_TRUST, "clamped to the floor");
    }

    // ---- identity binding ---------------------------------------------------

    #[test]
    fn named_node_matches_its_name_and_a_reassigned_id_does_not() {
        let node = Node::named("alice");
        assert_eq!(node.id, "alice", "the name is the node id");
        assert!(!node.public_key.is_empty(), "a keypair was generated");
        assert!(
            node.identity_matches_name(),
            "a named node signs under the name it presents"
        );

        // Renaming after the fact breaks the binding (the `==` in the check).
        let mut renamed = Node::named("alice");
        renamed.id = "bob".to_string();
        assert!(
            !renamed.identity_matches_name(),
            "the key still signs for 'alice', so the binding is broken"
        );
    }
}
