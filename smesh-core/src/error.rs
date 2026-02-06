//! Error types for SMESH core

use thiserror::Error;

/// Result type alias using SmeshError
pub type Result<T> = std::result::Result<T, SmeshError>;

/// Errors that can occur in SMESH core operations
#[derive(Error, Debug)]
pub enum SmeshError {
    /// Signal has expired
    #[error("Signal expired: {0}")]
    SignalExpired(String),

    /// Node not found in network
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Invalid signal intensity
    #[error("Invalid intensity: {0} (must be 0.0-1.0)")]
    InvalidIntensity(f64),

    /// Invalid trust score
    #[error("Invalid trust score: {0} (must be 0.0-1.0)")]
    InvalidTrust(f64),

    /// Network topology error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    /// Hash computation error
    #[error("Hash error: {0}")]
    HashError(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Proof of work failed
    #[error("Proof of work failed: {0}")]
    ProofOfWorkFailed(String),
}
