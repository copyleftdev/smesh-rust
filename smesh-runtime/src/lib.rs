//! # SMESH Runtime
//!
//! Async runtime and networking layer for SMESH protocol.
//!
//! Provides:
//! - Async event loop for signal processing
//! - P2P networking via QUIC
//! - Peer discovery and management

pub mod journal;
pub mod mesh;
pub mod peer;
pub mod runtime;
pub mod transport;

pub use journal::{Journal, JournalEvent};
pub use mesh::{MeshConfig, MeshHandle};
pub use peer::{Peer, PeerId, PeerManager, PeerState};
pub use runtime::{RuntimeConfig, RuntimeEvent, RuntimeStats, SmeshRuntime};
pub use transport::{
    QuicTransport, TransportConfig, TransportError, TransportMessage, DEFAULT_CONNECT_TIMEOUT_MS,
    DEFAULT_IDLE_TIMEOUT_MS, DEFAULT_KEEPALIVE_MS,
};
