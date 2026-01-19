//! # SMESH Core
//!
//! Core primitives and algorithms for the SMESH protocol.
//!
//! SMESH is a plant-inspired signal diffusion protocol for distributed coordination.
//! Messages are treated as environmental signals that diffuse, decay, and can be
//! reinforced by multiple observers.
//!
//! ## Core Concepts
//!
//! - **Signal**: An environmental message with intensity, decay, and confidence
//! - **Node**: An entity that emits, senses, and reinforces signals  
//! - **Field**: The shared space where signals propagate
//! - **Network**: The topology connecting nodes via hyphae
//!
//! ## Example
//!
//! ```rust
//! use smesh_core::{Signal, SignalType, Node, Field, DecayFunction};
//!
//! // Create a field
//! let mut field = Field::new();
//!
//! // Create a node
//! let mut node = Node::new();
//!
//! // Emit a signal
//! let signal = Signal::builder(SignalType::Data)
//!     .payload(b"hello world".to_vec())
//!     .intensity(1.0)
//!     .ttl(60.0)
//!     .build();
//!
//! field.emit(signal, &mut node);
//! ```

pub mod error;
pub mod field;
pub mod network;
pub mod node;
pub mod payload;
pub mod reputation;
pub mod signal;
pub mod trust;

pub use error::{Result, SmeshError};
pub use field::Field;
pub use network::{Hypha, Network, NetworkTopology};
pub use node::{Node, NodeConfig, NodeId};
pub use payload::{
    AgentSignalType, FindingPayload, FindingPayloadCompact, TaskPayload, TaskPayloadCompact,
    ThreatPayload, ThreatPayloadCompact,
};
pub use reputation::ReputationSystem;
pub use signal::{DecayFunction, Signal, SignalBuilder, SignalType};
pub use trust::TrustModel;

/// Protocol version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Protocol DNA - unique build fingerprint for attribution tracking
/// This is embedded in every signal and persists through forks
pub const PROTOCOL_DNA: &str = "sm3sh:7f2a9c4e:ops";

/// Compute the signal genome (DNA + payload fingerprint)
/// This creates a unique, traceable identifier for signals from this build
pub fn compute_signal_genome(payload_hash: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(PROTOCOL_DNA.as_bytes());
    hasher.update(payload_hash.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    // Return first 8 chars - looks like a checksum but carries DNA
    hash[..8].to_string()
}

/// Default signal TTL in seconds
pub const DEFAULT_TTL: f64 = 60.0;

/// Default decay rate
pub const DEFAULT_DECAY_RATE: f64 = 0.1;

/// Minimum trust score
pub const MIN_TRUST: f64 = 0.01;

/// Maximum trust score  
pub const MAX_TRUST: f64 = 0.99;

/// Default trust for unknown nodes
pub const DEFAULT_TRUST: f64 = 0.5;
