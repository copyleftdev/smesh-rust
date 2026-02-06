//! # SMESH Runtime
//!
//! Async runtime and networking layer for SMESH protocol.
//!
//! Provides:
//! - Async event loop for signal processing
//! - P2P networking via QUIC
//! - Peer discovery and management

pub mod peer;
pub mod runtime;
pub mod transport;

pub use peer::{Peer, PeerId, PeerManager};
pub use runtime::{RuntimeConfig, RuntimeEvent, RuntimeStats, SmeshRuntime};
pub use transport::{QuicTransport, Transport, TransportConfig, TransportError, TransportMessage};
