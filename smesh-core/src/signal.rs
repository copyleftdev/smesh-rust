//! Signal primitive - the core message type in SMESH
//!
//! Signals are environmental messages that:
//! - Have intensity that decays over time
//! - Can be reinforced by multiple observers
//! - Carry confidence scores
//! - Propagate through the field with dampening

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::identity::{Attestation, NodeIdentity};
use crate::{compute_signal_genome, NodeId, DEFAULT_DECAY_RATE, DEFAULT_TTL};

/// Types of signals in SMESH
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalType {
    /// General data signal
    Data,
    /// Query/request signal
    Query,
    /// Response to a query
    Response,
    /// Coordination signal
    Coordination,
    /// Heartbeat/presence signal
    Heartbeat,
    /// Alert/warning signal
    Alert,
    /// Custom application signal
    Custom,
}

/// Decay functions for signal intensity over time
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DecayFunction {
    /// Exponential decay: I(t) = I₀ * e^(-λt)
    #[default]
    Exponential,
    /// Linear decay: I(t) = I₀ * (1 - t/TTL)
    Linear,
    /// Sigmoid decay: smooth S-curve
    Sigmoid,
    /// Step function: full intensity until TTL, then zero
    Step,
}

/// Most attesters a single signal will carry.
///
/// Attestations arrive from peers and are relayed onward, so an unbounded list
/// is something one peer can grow at everyone else's expense.
pub const MAX_ATTESTATIONS: usize = 64;

/// A signal in the SMESH field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Unique identifier for this signal instance
    pub id: Uuid,

    /// Content-based hash for deduplication and reinforcement matching
    pub origin_hash: String,

    /// Type of signal
    pub signal_type: SignalType,

    /// Binary payload
    pub payload: Vec<u8>,

    /// Initial intensity (0.0 - 1.0)
    pub intensity: f64,

    /// Current intensity after decay
    pub current_intensity: f64,

    /// Time to live in seconds
    pub ttl: f64,

    /// Decay rate parameter
    pub decay_rate: f64,

    /// Decay function to use
    pub decay_function: DecayFunction,

    /// Maximum propagation hops
    pub radius: u32,

    /// Sender's confidence in this signal (0.0 - 1.0)
    pub confidence: f64,

    /// ID of the originating node
    pub origin_node_id: String,

    /// When the signal was created
    pub created_at: DateTime<Utc>,

    /// Number of times this signal has been reinforced
    pub reinforcement_count: u32,

    /// Node IDs that have reinforced this signal
    pub reinforced_by: Vec<String>,

    /// Current hop count
    pub hops: u32,

    /// Nodes this signal has diffused to through the network.
    ///
    /// This is the spatial frontier of the signal: it grows outward one
    /// network hop per tick as the signal spreads. An empty set means the
    /// signal has not yet entered network diffusion and is treated as
    /// *ambient* (sensable everywhere) for field-only use.
    #[serde(default)]
    pub reached_nodes: Vec<String>,

    /// Protocol checksum (carries build DNA for attribution)
    #[serde(default)]
    pub protocol_checksum: String,

    /// Signed statements that a node stands behind this claim.
    ///
    /// This is the trustworthy counterpart to `reinforced_by`. That field is a
    /// list of names anyone can write; these are signatures over
    /// [`Signal::origin_hash`], so they cannot be fabricated for a key the
    /// sender does not hold, nor lifted from a different claim. Anything
    /// arriving over a network should be counted from here.
    #[serde(default)]
    pub attestations: Vec<Attestation>,
}

impl Signal {
    /// Create a new signal builder
    pub fn builder(signal_type: SignalType) -> SignalBuilder {
        SignalBuilder::new(signal_type)
    }

    /// Compute current intensity based on decay function and elapsed time
    pub fn compute_intensity(&self, current_time: DateTime<Utc>) -> f64 {
        let age = (current_time - self.created_at).num_milliseconds() as f64 / 1000.0;

        if age < 0.0 {
            return self.intensity;
        }

        if age >= self.ttl {
            return 0.0;
        }

        match self.decay_function {
            DecayFunction::Exponential => self.intensity * (-self.decay_rate * age).exp(),
            DecayFunction::Linear => (self.intensity * (1.0 - age / self.ttl)).max(0.0),
            DecayFunction::Sigmoid => {
                let midpoint = self.ttl / 2.0;
                self.intensity / (1.0 + ((age - midpoint) * self.decay_rate).exp())
            }
            // Full intensity until expiry. The `age >= ttl` case already
            // returned 0.0 above, so no inner check is needed or reachable.
            DecayFunction::Step => self.intensity,
        }
    }

    /// Get effective intensity (current * confidence * reinforcement boost)
    pub fn effective_intensity(&self, current_time: DateTime<Utc>) -> f64 {
        let base = self.compute_intensity(current_time);
        let reinforcement_boost = 1.0 + (self.reinforcement_count as f64 * 0.1).min(0.5);
        (base * self.confidence * reinforcement_boost).min(1.0)
    }

    /// Check if signal has expired
    pub fn is_expired(&self, current_time: DateTime<Utc>) -> bool {
        let age = (current_time - self.created_at).num_milliseconds() as f64 / 1000.0;
        age >= self.ttl || self.compute_intensity(current_time) < 0.01
    }

    /// Reinforce this signal (increases confidence and count)
    pub fn reinforce(&mut self, reinforcer_id: &str) {
        if !self.reinforced_by.contains(&reinforcer_id.to_string()) {
            self.reinforced_by.push(reinforcer_id.to_string());
            self.reinforcement_count += 1;

            // Boost confidence with diminishing returns
            let boost = 0.1 / (1.0 + self.reinforcement_count as f64 * 0.5);
            self.confidence = (self.confidence + boost).min(1.0);
        }
    }

    /// Whether this signal has diffused to the given node.
    ///
    /// A signal that has not yet entered diffusion (empty reached set) is
    /// ambient and considered to have reached every node.
    pub fn has_reached(&self, node_id: &str) -> bool {
        self.reached_nodes.is_empty() || self.reached_nodes.iter().any(|n| n == node_id)
    }

    /// Record that the signal has diffused to a node (idempotent).
    pub fn mark_reached(&mut self, node_id: &str) {
        if !self.reached_nodes.iter().any(|n| n == node_id) {
            self.reached_nodes.push(node_id.to_string());
        }
    }

    /// Sign this signal, recording that `identity` stands behind the claim.
    ///
    /// Idempotent: a node attesting twice adds nothing, which is what makes
    /// re-assertion safe to do on a timer.
    pub fn attest(&mut self, identity: &NodeIdentity) {
        let mine = identity.public_key_hex();

        // Skip only if *we* already attested. Matching on name alone let
        // someone who squatted the name in first block the real holder from
        // ever signing its own claim.
        if self
            .attestations
            .iter()
            .any(|a| a.node_id == identity.node_id() && a.public_key == mine)
        {
            return;
        }

        // Drop any impostor entry for this name: we hold the key, they do not.
        self.attestations
            .retain(|a| a.node_id != identity.node_id());

        self.attestations.push(identity.attest(&self.origin_hash));
    }

    /// Everyone whose signature over this claim actually checks out.
    ///
    /// The count of this is the protocol's central measurement: how many
    /// independent parties assert the same thing. Unverifiable entries are
    /// dropped silently rather than counted.
    pub fn verified_attesters(&self) -> Vec<NodeId> {
        let mut attesters = Vec::with_capacity(self.attestations.len());
        for attestation in &self.attestations {
            if attestation.verify(&self.origin_hash) && !attesters.contains(&attestation.node_id) {
                attesters.push(attestation.node_id.clone());
            }
        }
        attesters
    }

    /// Merge attestations from a peer's copy, keeping only what verifies.
    ///
    /// Returns the names newly added, which is what tells a gossip layer
    /// whether its knowledge grew and therefore whether to pass the message on.
    ///
    /// An attestation is refused when its signature does not check out, or when
    /// it claims a name this signal already has under a *different* key. The
    /// second case is a same-claim impersonation attempt, and the first
    /// attestation seen wins.
    pub fn merge_attestations(&mut self, incoming: &[Attestation]) -> Vec<NodeId> {
        let mut added = Vec::new();

        for attestation in incoming {
            // A peer controls how many of these it sends, and each one costs a
            // signature verification and a slot in memory that then travels on
            // to everyone else. Cap it: no real claim needs more attesters than
            // there are nodes worth listening to.
            if self.attestations.len() >= MAX_ATTESTATIONS {
                break;
            }

            if !attestation.verify(&self.origin_hash) {
                continue;
            }

            match self
                .attestations
                .iter()
                .find(|a| a.node_id == attestation.node_id)
            {
                Some(existing) if existing.public_key == attestation.public_key => continue,
                Some(_) => continue, // name already bound to a different key
                None => {}
            }

            added.push(attestation.node_id.clone());
            self.attestations.push(attestation.clone());
        }

        added
    }

    /// Create a propagated copy with dampening
    pub fn propagate(&self, dampening: f64) -> Signal {
        let mut propagated = self.clone();
        propagated.id = Uuid::new_v4();
        propagated.intensity *= dampening;
        propagated.current_intensity *= dampening;
        propagated.hops += 1;
        propagated
    }

    /// Get payload as UTF-8 string
    pub fn payload_as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.payload).ok()
    }

    /// Compute the origin hash for deduplication
    fn compute_origin_hash(
        signal_type: SignalType,
        payload: &[u8],
        origin_node_id: &str,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!("{:?}", signal_type).as_bytes());
        hasher.update(payload);
        hasher.update(origin_node_id.as_bytes());
        // 128 bits. The previous 64 was fine against accident but thin against
        // an adversary hunting collisions, which now matters: attestations are
        // signatures over this hash, so two claims sharing one would let
        // agreement on the first be presented as agreement on the second.
        format!("{:x}", hasher.finalize())[..32].to_string()
    }
}

/// Builder for creating signals
#[derive(Debug)]
pub struct SignalBuilder {
    signal_type: SignalType,
    payload: Vec<u8>,
    intensity: f64,
    ttl: f64,
    decay_rate: f64,
    decay_function: DecayFunction,
    radius: u32,
    confidence: f64,
    origin_node_id: String,
}

impl SignalBuilder {
    /// Create a new signal builder
    pub fn new(signal_type: SignalType) -> Self {
        Self {
            signal_type,
            payload: Vec::new(),
            intensity: 1.0,
            ttl: DEFAULT_TTL,
            decay_rate: DEFAULT_DECAY_RATE,
            decay_function: DecayFunction::default(),
            radius: 5,
            confidence: 1.0,
            origin_node_id: String::new(),
        }
    }

    /// Set the payload
    pub fn payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }

    /// Set payload from JSON-serializable value
    pub fn payload_json<T: Serialize>(mut self, value: &T) -> Self {
        self.payload = serde_json::to_vec(value).unwrap_or_default();
        self
    }

    /// Set payload from TOON-serializable value (token-efficient format)
    ///
    /// TOON (Token-Oriented Object Notation) reduces token costs by ~20% vs JSON.
    /// Use this for payloads that will be processed by LLMs.
    pub fn payload_toon<T: Serialize>(mut self, value: &T) -> Self {
        // Convert to serde_json::Value first, then encode to TOON
        if let Ok(json_value) = serde_json::to_value(value) {
            self.payload = toon::encode(&json_value, None).into_bytes();
        }
        self
    }

    /// Set the initial intensity
    pub fn intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set the TTL in seconds
    pub fn ttl(mut self, ttl: f64) -> Self {
        self.ttl = ttl.max(0.0);
        self
    }

    /// Set the decay rate
    pub fn decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate.max(0.0);
        self
    }

    /// Set the decay function
    pub fn decay_function(mut self, func: DecayFunction) -> Self {
        self.decay_function = func;
        self
    }

    /// Set the propagation radius
    pub fn radius(mut self, radius: u32) -> Self {
        self.radius = radius;
        self
    }

    /// Set the confidence
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Address this signal by its content alone.
    ///
    /// The content hash normally covers the origin, which makes a signal
    /// *mine*: two nodes saying the same thing produce two distinct signals.
    /// A correlatable signal drops the origin from the address, so independent
    /// emitters of the same claim converge on one signal and are recorded as
    /// corroborating each other.
    ///
    /// This is the difference between an utterance and an assertion about the
    /// world. Use it whenever the point is that several parties agree. The
    /// origin is still stamped on the signal for attribution; it just stays out
    /// of the address, because the address is the claim, not the claimant.
    ///
    /// Anything correlatable must therefore keep evidence out of the payload —
    /// evidence differs per node and would make every address unique again.
    pub fn correlatable(mut self) -> Self {
        self.origin_node_id = String::new();
        self
    }

    /// Set the origin node ID
    ///
    /// This folds the origin into the content hash, so the resulting signal is
    /// unique to this node even if another node says exactly the same thing.
    /// For a claim meant to accumulate corroboration, use
    /// [`SignalBuilder::correlatable`] instead.
    pub fn origin(mut self, node_id: &str) -> Self {
        self.origin_node_id = node_id.to_string();
        self
    }

    /// Build the signal
    pub fn build(self) -> Signal {
        let now = Utc::now();
        let origin_hash =
            Signal::compute_origin_hash(self.signal_type, &self.payload, &self.origin_node_id);

        // Compute protocol checksum (carries build DNA for attribution)
        let protocol_checksum = compute_signal_genome(&origin_hash);

        Signal {
            id: Uuid::new_v4(),
            origin_hash,
            signal_type: self.signal_type,
            payload: self.payload,
            intensity: self.intensity,
            current_intensity: self.intensity,
            ttl: self.ttl,
            decay_rate: self.decay_rate,
            decay_function: self.decay_function,
            radius: self.radius,
            confidence: self.confidence,
            origin_node_id: self.origin_node_id,
            created_at: now,
            reinforcement_count: 0,
            reinforced_by: Vec::new(),
            hops: 0,
            reached_nodes: Vec::new(),
            protocol_checksum,
            attestations: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PROTOCOL_DNA;

    #[test]
    fn a_peer_cannot_grow_the_attestation_list_without_bound() {
        let mut signal = Signal::builder(SignalType::Data)
            .payload(b"claim".to_vec())
            .build();

        let flood: Vec<Attestation> = (0..MAX_ATTESTATIONS * 2)
            .map(|i| NodeIdentity::generate_named(format!("n{i}")).attest(&signal.origin_hash))
            .collect();

        signal.merge_attestations(&flood);
        assert_eq!(signal.attestations.len(), MAX_ATTESTATIONS);
    }

    #[test]
    fn squatting_a_name_does_not_stop_the_real_holder_signing() {
        let real = NodeIdentity::generate_named("latency");
        let impostor = NodeIdentity::generate_named("latency");

        let mut signal = Signal::builder(SignalType::Data)
            .payload(b"claim".to_vec())
            .build();

        // The impostor gets there first under the same name.
        signal.merge_attestations(&[impostor.attest(&signal.origin_hash)]);
        real.attest(&signal.origin_hash);
        signal.attest(&real);

        let keys: Vec<&str> = signal
            .attestations
            .iter()
            .map(|a| a.public_key.as_str())
            .collect();
        assert!(
            keys.contains(&real.public_key_hex().as_str()),
            "the real holder must still be able to sign its own claim"
        );
        assert_eq!(signal.verified_attesters(), vec!["latency".to_string()]);
    }

    #[test]
    fn test_signal_dna_fingerprint() {
        // Every signal carries the protocol DNA fingerprint
        let signal = Signal::builder(SignalType::Data)
            .payload(b"test".to_vec())
            .build();

        // The checksum is always 8 hex chars
        assert_eq!(signal.protocol_checksum.len(), 8);

        // Same payload = same checksum (deterministic)
        let signal2 = Signal::builder(SignalType::Data)
            .payload(b"test".to_vec())
            .build();
        assert_eq!(signal.protocol_checksum, signal2.protocol_checksum);

        // Different payload = different checksum
        let signal3 = Signal::builder(SignalType::Data)
            .payload(b"different".to_vec())
            .build();
        assert_ne!(signal.protocol_checksum, signal3.protocol_checksum);

        // DNA constant is embedded
        assert!(PROTOCOL_DNA.contains("sm3sh"));
        assert!(PROTOCOL_DNA.contains("ops"));
    }

    #[test]
    fn test_signal_creation() {
        let signal = Signal::builder(SignalType::Data)
            .payload(b"test".to_vec())
            .intensity(0.8)
            .ttl(30.0)
            .build();

        assert_eq!(signal.signal_type, SignalType::Data);
        assert_eq!(signal.intensity, 0.8);
        assert_eq!(signal.ttl, 30.0);
        assert_eq!(signal.payload, b"test".to_vec());
    }

    #[test]
    fn test_exponential_decay() {
        let signal = Signal::builder(SignalType::Data)
            .intensity(1.0)
            .decay_rate(0.1)
            .decay_function(DecayFunction::Exponential)
            .build();

        let now = signal.created_at;
        let later = now + chrono::Duration::seconds(10);

        let intensity = signal.compute_intensity(later);
        // e^(-0.1 * 10) ≈ 0.368
        assert!((intensity - 0.368).abs() < 0.01);
    }

    #[test]
    fn test_linear_decay() {
        let signal = Signal::builder(SignalType::Data)
            .intensity(1.0)
            .ttl(100.0)
            .decay_function(DecayFunction::Linear)
            .build();

        let now = signal.created_at;
        let later = now + chrono::Duration::seconds(50);

        let intensity = signal.compute_intensity(later);
        // 1.0 * (1 - 50/100) = 0.5
        assert!((intensity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_reinforcement() {
        let mut signal = Signal::builder(SignalType::Data).confidence(0.5).build();

        signal.reinforce("node1");
        signal.reinforce("node2");
        signal.reinforce("node1"); // Duplicate, should not increase

        assert_eq!(signal.reinforcement_count, 2);
        assert_eq!(signal.reinforced_by.len(), 2);
        assert!(signal.confidence > 0.5);
    }

    #[test]
    fn test_propagation() {
        let signal = Signal::builder(SignalType::Data).intensity(1.0).build();

        let propagated = signal.propagate(0.9);

        assert_eq!(propagated.intensity, 0.9);
        assert_eq!(propagated.hops, 1);
        assert_ne!(propagated.id, signal.id);
        assert_eq!(propagated.origin_hash, signal.origin_hash);
    }

    // ---- decay math, pinned to exact values --------------------------------
    //
    // The older decay tests assert within a wide tolerance and cover only two
    // of the four curves. That leaves the arithmetic free to be wrong: a `/`
    // flipped to `*`, a `-` to `+`, the Sigmoid and Step arms untouched. These
    // pin each curve to a value computed independently of the code under test,
    // so a mutated operator lands outside the tolerance rather than inside it.

    fn at(signal: &Signal, secs: i64) -> f64 {
        signal.compute_intensity(signal.created_at + chrono::Duration::seconds(secs))
    }

    #[test]
    fn exponential_curve_is_pinned_at_several_ages() {
        let s = Signal::builder(SignalType::Data)
            .intensity(1.0)
            .decay_rate(0.2)
            .ttl(1000.0)
            .decay_function(DecayFunction::Exponential)
            .build();
        // e^(-0.2 * age). Independent reference values.
        assert!((at(&s, 0) - 1.0).abs() < 1e-9, "age 0 is undecayed");
        assert!((at(&s, 5) - (-1.0f64).exp()).abs() < 1e-9, "0.2*5 = 1.0");
        assert!((at(&s, 10) - (-2.0f64).exp()).abs() < 1e-9, "0.2*10 = 2.0");
        // A larger rate must decay faster at the same age — kills `*`↔`/` and
        // `-`-sign mutants on the exponent that a single-point test misses.
        let faster = Signal::builder(SignalType::Data)
            .intensity(1.0)
            .decay_rate(0.4)
            .ttl(1000.0)
            .decay_function(DecayFunction::Exponential)
            .build();
        assert!(at(&faster, 10) < at(&s, 10));
    }

    #[test]
    fn linear_curve_is_pinned_and_reaches_zero_at_ttl() {
        let s = Signal::builder(SignalType::Data)
            .intensity(1.0)
            .ttl(100.0)
            .decay_function(DecayFunction::Linear)
            .build();
        assert!((at(&s, 0) - 1.0).abs() < 1e-9);
        assert!((at(&s, 25) - 0.75).abs() < 1e-9, "1 - 25/100");
        assert!((at(&s, 50) - 0.5).abs() < 1e-9, "1 - 50/100");
        assert!((at(&s, 75) - 0.25).abs() < 1e-9, "1 - 75/100");
    }

    #[test]
    fn sigmoid_curve_is_pinned_at_and_around_its_midpoint() {
        let s = Signal::builder(SignalType::Data)
            .intensity(1.0)
            .decay_rate(0.1)
            .ttl(100.0)
            .decay_function(DecayFunction::Sigmoid)
            .build();
        // 1 / (1 + e^((age - 50) * 0.1)). At the midpoint the exponent is 0,
        // so the value is exactly 1/2 — the one point that fixes the whole arm.
        assert!((at(&s, 50) - 0.5).abs() < 1e-9, "midpoint is exactly half");
        // Independent references either side. `*`↔`/`, `-`↔`+` on the exponent
        // all move these off their marks.
        let lo = 1.0 / (1.0 + ((40.0 - 50.0) * 0.1f64).exp());
        let hi = 1.0 / (1.0 + ((60.0 - 50.0) * 0.1f64).exp());
        assert!((at(&s, 40) - lo).abs() < 1e-9);
        assert!((at(&s, 60) - hi).abs() < 1e-9);
        // Monotone decreasing through the midpoint.
        assert!(at(&s, 40) > at(&s, 50) && at(&s, 50) > at(&s, 60));
    }

    #[test]
    fn step_curve_holds_full_intensity_then_drops_to_zero() {
        let s = Signal::builder(SignalType::Data)
            .intensity(0.8)
            .ttl(30.0)
            .decay_function(DecayFunction::Step)
            .build();
        assert!((at(&s, 0) - 0.8).abs() < 1e-9);
        assert!(
            (at(&s, 29) - 0.8).abs() < 1e-9,
            "full intensity right up to ttl"
        );
        assert_eq!(at(&s, 30), 0.0, "exactly at ttl is expired");
        assert_eq!(at(&s, 31), 0.0);
    }

    #[test]
    fn intensity_boundaries_hold_before_zero_and_at_ttl() {
        let s = Signal::builder(SignalType::Data)
            .intensity(0.9)
            .ttl(60.0)
            .decay_function(DecayFunction::Linear)
            .build();
        // age < 0 returns the undecayed intensity, not a decayed one.
        let before = s.compute_intensity(s.created_at - chrono::Duration::seconds(5));
        assert_eq!(before, 0.9, "a signal from the future has not decayed");
        // age == ttl returns exactly 0 (the `>=` boundary).
        assert_eq!(at(&s, 60), 0.0);
        // age just under ttl is still positive (kills `>=`→`>` shifting the edge).
        assert!(at(&s, 59) > 0.0);
    }

    #[test]
    fn effective_intensity_applies_confidence_and_reinforcement_boost() {
        let mut s = Signal::builder(SignalType::Data)
            .intensity(1.0)
            .confidence(0.5)
            .ttl(1000.0)
            .decay_function(DecayFunction::Step)
            .build();
        // base 1.0 * confidence 0.5 * boost (1 + 0*0.1) = 0.5.
        assert!((s.effective_intensity(s.created_at) - 0.5).abs() < 1e-9);
        // Two reinforcements: boost = 1 + min(2*0.1, 0.5) = 1.2, so 0.6.
        s.reinforcement_count = 2;
        assert!((s.effective_intensity(s.created_at) - 0.6).abs() < 1e-9);
        // The boost saturates at +0.5 and the whole thing is capped at 1.0.
        s.reinforcement_count = 100;
        s.confidence = 1.0;
        assert_eq!(
            s.effective_intensity(s.created_at),
            1.0,
            "capped at 1.0, and boost never exceeds 1.5"
        );
    }

    #[test]
    fn is_expired_fires_on_ttl_and_on_faded_intensity_separately() {
        // First clause: age >= ttl, independent of intensity.
        let a = Signal::builder(SignalType::Data)
            .intensity(1.0)
            .ttl(10.0)
            .decay_function(DecayFunction::Step)
            .build();
        assert!(!a.is_expired(a.created_at));
        assert!(a.is_expired(a.created_at + chrono::Duration::seconds(10)));

        // Second clause: still within ttl, but intensity has fallen below 0.01.
        let b = Signal::builder(SignalType::Data)
            .intensity(1.0)
            .decay_rate(1.0)
            .ttl(1000.0)
            .decay_function(DecayFunction::Exponential)
            .build();
        assert!(!b.is_expired(b.created_at), "fresh signal is live");
        // e^(-1.0 * 10) ≈ 4.5e-5 < 0.01, but age (10) is far below ttl (1000),
        // so only the intensity clause can be catching this.
        assert!(b.is_expired(b.created_at + chrono::Duration::seconds(10)));
    }

    #[test]
    fn reinforce_boost_has_diminishing_returns() {
        let mut s = Signal::builder(SignalType::Data).confidence(0.5).build();
        s.reinforce("a");
        // count is now 1: boost = 0.1 / (1 + 1*0.5) = 0.0666..., conf = 0.5666...
        assert!((s.confidence - (0.5 + 0.1 / 1.5)).abs() < 1e-9);
        let after_one = s.confidence;
        s.reinforce("b");
        // count 2: boost = 0.1 / (1 + 2*0.5) = 0.05 — strictly smaller step.
        let step_two = s.confidence - after_one;
        assert!((step_two - 0.05).abs() < 1e-9);
        assert!(
            step_two < 0.1 / 1.5,
            "each reinforcement adds less than the last"
        );
    }

    #[test]
    fn has_reached_treats_empty_as_ambient() {
        let mut s = Signal::builder(SignalType::Data).build();
        assert!(s.has_reached("anyone"), "empty reached-set is ambient");
        s.mark_reached("n1");
        assert!(s.has_reached("n1"));
        assert!(
            !s.has_reached("n2"),
            "a non-empty set excludes unreached nodes"
        );
    }

    #[test]
    fn propagate_dampens_both_intensity_fields() {
        let mut s = Signal::builder(SignalType::Data).intensity(1.0).build();
        s.current_intensity = 0.8;
        let p = s.propagate(0.5);
        assert!((p.intensity - 0.5).abs() < 1e-9);
        assert!(
            (p.current_intensity - 0.4).abs() < 1e-9,
            "current_intensity is dampened too, not left untouched or added to"
        );
    }
}
