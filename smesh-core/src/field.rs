//! Field - the shared space where signals propagate
//!
//! The field manages signal storage, decay, and expiration.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Node, Signal};

/// The shared field where signals exist and propagate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    /// Active signals in the field (origin_hash -> Signal)
    pub signals: HashMap<String, Signal>,

    /// Signal history for analysis (time, signal)
    pub signal_history: Vec<(DateTime<Utc>, Signal)>,

    /// Current simulation time
    pub current_time: DateTime<Utc>,

    /// Decay rate per tick
    pub decay_tick_rate: f64,

    /// Maximum signals to store in history
    pub max_history: usize,
}

impl Field {
    /// Create a new field
    pub fn new() -> Self {
        Self {
            signals: HashMap::new(),
            signal_history: Vec::new(),
            current_time: Utc::now(),
            decay_tick_rate: 1.0,
            max_history: 10000,
        }
    }

    /// Create a field with custom decay rate
    pub fn with_decay_rate(decay_tick_rate: f64) -> Self {
        let mut field = Self::new();
        field.decay_tick_rate = decay_tick_rate;
        field
    }

    /// Emit a signal from a node into the field
    pub fn emit(&mut self, mut signal: Signal, source_node: &mut Node) -> String {
        signal.origin_node_id = source_node.id.clone();
        signal.created_at = self.current_time;

        // Check for existing signal to reinforce
        if let Some(existing) = self.signals.get_mut(&signal.origin_hash) {
            existing.reinforce(&source_node.id);
            source_node.stats.signals_reinforced += 1;
            return existing.origin_hash.clone();
        }

        let hash = signal.origin_hash.clone();
        self.signals.insert(hash.clone(), signal);
        source_node.stats.signals_emitted += 1;

        hash
    }

    /// Emit a signal without a node reference
    pub fn emit_anonymous(&mut self, signal: Signal) -> String {
        let hash = signal.origin_hash.clone();

        if let Some(existing) = self.signals.get_mut(&hash) {
            existing.reinforce("anonymous");
            return hash;
        }

        self.signals.insert(hash.clone(), signal);
        hash
    }

    /// Advance time and decay/expire signals
    pub fn tick(&mut self, dt: f64) -> usize {
        self.current_time += chrono::Duration::milliseconds((dt * 1000.0) as i64);

        let mut expired = Vec::new();

        for (hash, signal) in &mut self.signals {
            signal.current_intensity = signal.compute_intensity(self.current_time);
            if signal.is_expired(self.current_time) {
                expired.push(hash.clone());
            }
        }

        // Move expired signals to history
        for hash in &expired {
            if let Some(signal) = self.signals.remove(hash) {
                if self.signal_history.len() < self.max_history {
                    self.signal_history.push((self.current_time, signal));
                }
            }
        }

        expired.len()
    }

    /// Sense signals at a node's location (all signals above threshold)
    pub fn sense(&self, node: &Node) -> Vec<&Signal> {
        self.signals
            .values()
            .filter(|s| node.can_sense(s))
            .collect()
    }

    /// Get all active signals
    pub fn active_signals(&self) -> impl Iterator<Item = &Signal> {
        self.signals.values()
    }

    /// Get signal by hash
    pub fn get_signal(&self, hash: &str) -> Option<&Signal> {
        self.signals.get(hash)
    }

    /// Get mutable signal by hash
    pub fn get_signal_mut(&mut self, hash: &str) -> Option<&mut Signal> {
        self.signals.get_mut(hash)
    }

    /// Clear all signals
    pub fn clear(&mut self) {
        self.signals.clear();
    }

    /// Get statistics about the field
    pub fn stats(&self) -> FieldStats {
        let total_intensity: f64 = self.signals.values().map(|s| s.current_intensity).sum();
        let avg_intensity = if self.signals.is_empty() {
            0.0
        } else {
            total_intensity / self.signals.len() as f64
        };

        let total_reinforcements: u32 = self.signals.values().map(|s| s.reinforcement_count).sum();

        FieldStats {
            active_signals: self.signals.len(),
            total_intensity,
            avg_intensity,
            total_reinforcements,
            history_size: self.signal_history.len(),
        }
    }
}

impl Default for Field {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the field state
#[derive(Debug, Clone)]
pub struct FieldStats {
    pub active_signals: usize,
    pub total_intensity: f64,
    pub avg_intensity: f64,
    pub total_reinforcements: u32,
    pub history_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SignalType;

    #[test]
    fn test_field_emit() {
        let mut field = Field::new();
        let mut node = Node::new();

        let signal = Signal::builder(SignalType::Data)
            .payload(b"test".to_vec())
            .build();

        let hash = field.emit(signal, &mut node);

        assert!(!hash.is_empty());
        assert_eq!(field.signals.len(), 1);
        assert_eq!(node.stats.signals_emitted, 1);
    }

    #[test]
    fn test_signal_expiration() {
        let mut field = Field::new();

        let signal = Signal::builder(SignalType::Data).ttl(1.0).build();

        field.emit_anonymous(signal);
        assert_eq!(field.signals.len(), 1);

        // Advance time past TTL
        field.tick(2.0);
        assert_eq!(field.signals.len(), 0);
        assert_eq!(field.signal_history.len(), 1);
    }

    #[test]
    fn test_signal_reinforcement() {
        let mut field = Field::new();
        let mut node1 = Node::with_id("node1");
        let mut node2 = Node::with_id("node2");

        let signal1 = Signal::builder(SignalType::Data)
            .payload(b"same content".to_vec())
            .origin("node1")
            .build();

        let signal2 = Signal::builder(SignalType::Data)
            .payload(b"same content".to_vec())
            .origin("node1")
            .build();

        field.emit(signal1, &mut node1);
        field.emit(signal2, &mut node2);

        // Should have only one signal (reinforced)
        assert_eq!(field.signals.len(), 1);

        let signal = field.signals.values().next().unwrap();
        assert_eq!(signal.reinforcement_count, 1);
    }
}
