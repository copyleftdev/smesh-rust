//! Signal primitive - the core message type in SMESH
//!
//! Signals are environmental messages that:
//! - Have intensity that decays over time
//! - Can be reinforced by multiple observers
//! - Carry confidence scores
//! - Propagate through the field with dampening

use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::{DEFAULT_TTL, DEFAULT_DECAY_RATE, compute_signal_genome};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecayFunction {
    /// Exponential decay: I(t) = I₀ * e^(-λt)
    Exponential,
    /// Linear decay: I(t) = I₀ * (1 - t/TTL)
    Linear,
    /// Sigmoid decay: smooth S-curve
    Sigmoid,
    /// Step function: full intensity until TTL, then zero
    Step,
}

impl Default for DecayFunction {
    fn default() -> Self {
        Self::Exponential
    }
}

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
    
    /// Protocol checksum (carries build DNA for attribution)
    #[serde(default)]
    pub protocol_checksum: String,
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
            DecayFunction::Exponential => {
                self.intensity * (-self.decay_rate * age).exp()
            }
            DecayFunction::Linear => {
                (self.intensity * (1.0 - age / self.ttl)).max(0.0)
            }
            DecayFunction::Sigmoid => {
                let midpoint = self.ttl / 2.0;
                self.intensity / (1.0 + ((age - midpoint) * self.decay_rate).exp())
            }
            DecayFunction::Step => {
                if age < self.ttl {
                    self.intensity
                } else {
                    0.0
                }
            }
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
    
    /// Create a propagated copy with dampening
    pub fn propagate(&self, dampening: f64) -> Signal {
        let mut propagated = self.clone();
        propagated.id = Uuid::new_v4();
        propagated.intensity *= dampening;
        propagated.current_intensity *= dampening;
        propagated.hops += 1;
        propagated
    }
    
    /// Compute the origin hash for deduplication
    fn compute_origin_hash(signal_type: SignalType, payload: &[u8], origin_node_id: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!("{:?}", signal_type).as_bytes());
        hasher.update(payload);
        hasher.update(origin_node_id.as_bytes());
        format!("{:x}", hasher.finalize())[..16].to_string()
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
    
    /// Set the origin node ID
    pub fn origin(mut self, node_id: &str) -> Self {
        self.origin_node_id = node_id.to_string();
        self
    }
    
    /// Build the signal
    pub fn build(self) -> Signal {
        let now = Utc::now();
        let origin_hash = Signal::compute_origin_hash(
            self.signal_type,
            &self.payload,
            &self.origin_node_id,
        );
        
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
            protocol_checksum,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PROTOCOL_DNA;
    
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
        let mut signal = Signal::builder(SignalType::Data)
            .confidence(0.5)
            .build();
        
        signal.reinforce("node1");
        signal.reinforce("node2");
        signal.reinforce("node1"); // Duplicate, should not increase
        
        assert_eq!(signal.reinforcement_count, 2);
        assert_eq!(signal.reinforced_by.len(), 2);
        assert!(signal.confidence > 0.5);
    }
    
    #[test]
    fn test_propagation() {
        let signal = Signal::builder(SignalType::Data)
            .intensity(1.0)
            .build();
        
        let propagated = signal.propagate(0.9);
        
        assert_eq!(propagated.intensity, 0.9);
        assert_eq!(propagated.hops, 1);
        assert_ne!(propagated.id, signal.id);
        assert_eq!(propagated.origin_hash, signal.origin_hash);
    }
}
