//! # SMESH Runtime
//!
//! Async runtime and networking layer for SMESH protocol.
//!
//! Provides:
//! - Async event loop for signal processing
//! - P2P networking via QUIC
//! - Peer discovery and management

pub mod peer;
pub mod transport;
pub mod runtime;

pub use peer::{Peer, PeerManager, PeerId};
pub use transport::{Transport, QuicTransport, TransportConfig, TransportMessage, TransportError};
pub use runtime::{SmeshRuntime, RuntimeConfig, RuntimeEvent, RuntimeStats};
