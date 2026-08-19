//! SMESH Runtime - async event loop for signal processing

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, info};

use serde_json::json;

use crate::journal::{payload_preview, Journal};
use crate::mesh::{self, MeshConfig, MeshHandle};
use crate::peer::{PeerId, PeerManager};
use crate::transport::{QuicTransport, TransportConfig, TransportError, TransportMessage};
use smesh_core::{Network, Node, NodeId, Signal};

/// How often `tick` writes a full field snapshot to the journal.
const SNAPSHOT_EVERY_TICKS: u64 = 5;

/// Configuration for the SMESH runtime
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Tick interval in milliseconds
    pub tick_interval_ms: u64,
    /// Maximum signals to process per tick
    pub max_signals_per_tick: usize,
    /// Enable signal propagation
    pub enable_propagation: bool,
    /// Transport configuration
    pub transport: TransportConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            tick_interval_ms: 100,
            max_signals_per_tick: 1000,
            enable_propagation: true,
            transport: TransportConfig::default(),
        }
    }
}

/// Events emitted by the runtime
#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    /// A new signal was emitted
    SignalEmitted { hash: String },
    /// A signal was reinforced
    SignalReinforced { hash: String, count: u32 },
    /// A signal arrived from a peer and was accepted by the local node
    SignalReceived {
        hash: String,
        from: PeerId,
        hops: u32,
    },
    /// A signal expired
    SignalExpired { hash: String },
    /// Network tick completed
    TickCompleted {
        tick: u64,
        active_signals: usize,
        expired: usize,
    },
    /// Peer connected
    PeerConnected { peer_id: PeerId },
    /// Peer disconnected
    PeerDisconnected { peer_id: PeerId },
}

/// The SMESH async runtime
pub struct SmeshRuntime {
    /// Network state
    network: Arc<RwLock<Network>>,
    /// Runtime configuration
    config: RuntimeConfig,
    /// Peer manager
    peers: Arc<PeerManager>,
    /// Event sender
    event_tx: mpsc::Sender<RuntimeEvent>,
    /// Event receiver (for consumers)
    event_rx: Option<mpsc::Receiver<RuntimeEvent>>,
    /// Current tick count
    tick_count: Arc<RwLock<u64>>,
    /// Shutdown signal
    shutdown: Arc<RwLock<bool>>,
    /// Transport, once this runtime has joined a mesh
    transport: Arc<RwLock<Option<Arc<QuicTransport>>>>,
    /// Event journal; disabled unless a run is being recorded
    journal: Arc<Journal>,
    /// Connection address -> peer node id, learned from the mesh handshake.
    ///
    /// Owned here rather than in the mesh so that journal lines written on the
    /// emit path can name the peer they went to. An accepted connection's
    /// address is ephemeral, so without this map a recorded send resolves to
    /// nothing when the run is replayed.
    peer_names: Arc<RwLock<std::collections::HashMap<std::net::SocketAddr, NodeId>>>,
}

impl SmeshRuntime {
    /// Create a new runtime
    pub fn new(config: RuntimeConfig) -> Self {
        let (event_tx, event_rx) = mpsc::channel(10000);
        let local_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

        Self {
            network: Arc::new(RwLock::new(Network::new())),
            config,
            peers: Arc::new(PeerManager::new(local_id, 100)),
            event_tx,
            event_rx: Some(event_rx),
            tick_count: Arc::new(RwLock::new(0)),
            shutdown: Arc::new(RwLock::new(false)),
            transport: Arc::new(RwLock::new(None)),
            journal: Journal::disabled(),
            peer_names: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Record this runtime's protocol events to `journal`.
    pub fn with_journal(mut self, journal: Arc<Journal>) -> Self {
        self.journal = journal;
        self
    }

    /// The journal this runtime records to.
    pub fn journal(&self) -> Arc<Journal> {
        Arc::clone(&self.journal)
    }

    /// Join a mesh, presenting `local_node_id` as this process's node.
    ///
    /// Brings up the QUIC endpoint, starts the accept and gossip loops, and
    /// dials the bootstrap peers. After this, [`SmeshRuntime::emit`] also
    /// broadcasts to connected peers, and signals arriving from peers are
    /// admitted by the local node's own sensing and relay policy (see
    /// [`crate::mesh`]).
    pub async fn join_mesh(
        &self,
        config: MeshConfig,
        local_node_id: &str,
    ) -> Result<MeshHandle, TransportError> {
        let (local_public_key, identity_pkcs8_der) = {
            let network = self.network.read().await;
            let Some(node) = network.nodes.get(local_node_id) else {
                return Err(TransportError::ConnectionFailed(format!(
                    "node {local_node_id} is not in this runtime's network"
                )));
            };
            let Some(identity) = node.identity.as_ref() else {
                return Err(TransportError::ConnectionFailed(format!(
                    "node {local_node_id} holds no signing key, so it cannot attest to anything"
                )));
            };
            (node.public_key.clone(), identity.to_pkcs8_der())
        };

        let (handle, transport) = mesh::start(mesh::MeshStartup {
            config,
            local_node_id: local_node_id.to_string() as NodeId,
            local_public_key,
            identity_pkcs8_der,
            network: Arc::clone(&self.network),
            peers: Arc::clone(&self.peers),
            event_tx: self.event_tx.clone(),
            journal: Arc::clone(&self.journal),
            conn_ids: Arc::clone(&self.peer_names),
        })
        .await?;

        *self.transport.write().await = Some(transport);

        Ok(handle)
    }

    /// Create runtime with existing network
    pub fn with_network(network: Network, config: RuntimeConfig) -> Self {
        let mut runtime = Self::new(config);
        runtime.network = Arc::new(RwLock::new(network));
        runtime
    }

    /// Take the event receiver (can only be called once)
    pub fn take_events(&mut self) -> Option<mpsc::Receiver<RuntimeEvent>> {
        self.event_rx.take()
    }

    /// Get a clone of the network (for external access)
    pub fn network(&self) -> Arc<RwLock<Network>> {
        Arc::clone(&self.network)
    }

    /// Get the peer manager
    pub fn peers(&self) -> Arc<PeerManager> {
        Arc::clone(&self.peers)
    }

    /// Emit a signal into the network, and onto the mesh if we are on one.
    ///
    /// Signals are content-addressed, so a node that independently reaches a
    /// conclusion another node already published lands on the same hash. That
    /// is treated as *corroboration*: this node is added as an attester and the
    /// merged claim still goes out, because our agreement is news to everyone
    /// who has not heard it. Swallowing it as a duplicate would silently
    /// discard the only evidence that two parties concur.
    pub async fn emit(&self, mut signal: Signal, node_id: &str) -> Option<String> {
        let mut network = self.network.write().await;

        // Check if node exists first
        if !network.nodes.contains_key(node_id) {
            return None;
        }

        // Stamp the emitting node as the signal's origin and seed its diffusion
        // frontier there, so the signal can spread outward from this node.
        //
        // Only stamp when the builder left it unset. Overwriting an origin that
        // is already part of the content hash would leave the signal claiming
        // one origin in its address and a different one in its field, and the
        // two would disagree forever after.
        if signal.origin_node_id.is_empty() {
            signal.origin_node_id = node_id.to_string();
        }
        signal.mark_reached(node_id);

        let hash = signal.origin_hash.clone();
        let first_assertion = !network.field.signals.contains_key(&hash);

        // Sign the claim. This is what makes our agreement countable by anyone
        // else: without it we are just another name in a list.
        let attestation = network
            .nodes
            .get(node_id)
            .and_then(|n| n.identity.as_ref().map(|identity| identity.attest(&hash)));

        if first_assertion {
            if let Some(attestation) = attestation {
                signal.merge_attestations(&[attestation]);
            }
            network.field.signals.insert(hash.clone(), signal);
        } else if let Some(existing) = network.field.signals.get_mut(&hash) {
            existing.reinforce(node_id);
            existing.mark_reached(node_id);
            if let Some(attestation) = attestation {
                existing.merge_attestations(&[attestation]);
            }
        }

        // Update node stats
        if let Some(node) = network.nodes.get_mut(node_id) {
            node.stats.signals_emitted += 1;
        }

        // Take a wire copy before releasing the lock. reached_nodes is local
        // knowledge and never crosses the wire; the receiver rewrites it. The
        // copy carries our merged attester set, which is what makes gossip
        // converge instead of each node broadcasting only its own view.
        let stored = network.field.signals.get(&hash);
        let wire_copy = stored.map(|s| {
            let mut s = s.clone();
            s.reached_nodes.clear();
            s
        });
        let attesters = stored.map(|s| s.verified_attesters()).unwrap_or_default();
        let payload = stored
            .map(|s| payload_preview(&s.payload, 512))
            .unwrap_or(serde_json::Value::Null);
        let confidence = stored.map(|s| s.confidence).unwrap_or(0.0);
        let field_time = network.field.current_time;
        drop(network);

        self.journal.record(
            "signal_emitted",
            json!({
                "hash": hash,
                "origin": node_id,
                "first_assertion": first_assertion,
                "attesters": attesters,
                "attester_count": attesters.len(),
                "confidence": confidence,
                "payload": payload,
            }),
        );

        let _ = self
            .event_tx
            .send(RuntimeEvent::SignalEmitted { hash: hash.clone() })
            .await;

        // If we are on a mesh, the signal goes out to peers as well as into
        // the local field.
        if let (Some(transport), Some(signal)) = (self.transport.read().await.clone(), wire_copy) {
            let hops = signal.hops;
            let intensity = signal.current_intensity;
            let msg = TransportMessage::signal(signal, field_time);
            let reached = transport.broadcast_all(&msg, None).await;

            let names = self.peer_names.read().await;
            for addr in &reached {
                self.journal.record(
                    "signal_sent",
                    json!({
                        "hash": hash,
                        "to": names.get(addr).cloned().unwrap_or_else(|| addr.to_string()),
                        "to_addr": addr.to_string(),
                        "hops": hops,
                        "intensity": intensity,
                        "kind": "origin",
                    }),
                );
            }
            drop(names);

            debug!("emitted {} to {} peer(s)", hash, reached.len());
        }

        Some(hash)
    }

    /// Add a node to the network
    pub async fn add_node(&self, node: Node) {
        let mut network = self.network.write().await;
        network.add_node(node);
    }

    /// Run a single tick
    pub async fn tick(&self) -> RuntimeEvent {
        let dt = self.config.tick_interval_ms as f64 / 1000.0;

        let mut network = self.network.write().await;
        let result = network.tick(dt);

        let mut tick_count = self.tick_count.write().await;
        *tick_count += 1;
        let tick = *tick_count;

        // Sample the whole field periodically. Decay is continuous, so a
        // replay that only saw emissions and arrivals would have to guess the
        // curve between them; these snapshots make it observed instead.
        if self.journal.is_enabled() && tick % SNAPSHOT_EVERY_TICKS == 0 {
            let signals: Vec<serde_json::Value> = network
                .field
                .signals
                .values()
                .map(|s| {
                    json!({
                        "hash": s.origin_hash,
                        "origin": s.origin_node_id,
                        "intensity": s.current_intensity,
                        "confidence": s.confidence,
                        "effective": s.effective_intensity(network.field.current_time),
                        "attesters": s.verified_attesters(),
                        "hops": s.hops,
                        "age_secs": (network.field.current_time - s.created_at)
                            .num_milliseconds() as f64
                            / 1000.0,
                    })
                })
                .collect();

            self.journal.record(
                "field_snapshot",
                json!({
                    "tick": tick,
                    "field_time": network.field.current_time.to_rfc3339(),
                    "active_signals": result.active_signals,
                    "expired_this_tick": result.expired_signals,
                    "signals": signals,
                }),
            );
        }

        let event = RuntimeEvent::TickCompleted {
            tick,
            active_signals: result.active_signals,
            expired: result.expired_signals,
        };

        let _ = self.event_tx.send(event.clone()).await;

        event
    }

    /// Run the event loop
    pub async fn run(&self) {
        let mut ticker = interval(Duration::from_millis(self.config.tick_interval_ms));

        info!(
            "SMESH runtime starting with {}ms tick interval",
            self.config.tick_interval_ms
        );

        loop {
            ticker.tick().await;

            // Check shutdown
            if *self.shutdown.read().await {
                info!("SMESH runtime shutting down");
                break;
            }

            // Run tick
            let event = self.tick().await;

            if let RuntimeEvent::TickCompleted {
                tick,
                active_signals,
                expired,
            } = &event
            {
                if *tick % 100 == 0 {
                    debug!(
                        "Tick {}: {} active signals, {} expired",
                        tick, active_signals, expired
                    );
                }
            }
        }
    }

    /// Run for a specific number of ticks (for testing)
    pub async fn run_ticks(&self, n: u64) -> Vec<RuntimeEvent> {
        let mut events = Vec::new();

        for _ in 0..n {
            let event = self.tick().await;
            events.push(event);
            tokio::time::sleep(Duration::from_millis(self.config.tick_interval_ms)).await;
        }

        events
    }

    /// Signal shutdown
    pub async fn shutdown(&self) {
        let mut shutdown = self.shutdown.write().await;
        *shutdown = true;
    }

    /// Get current tick count
    pub async fn current_tick(&self) -> u64 {
        *self.tick_count.read().await
    }

    /// Get network statistics
    pub async fn stats(&self) -> RuntimeStats {
        let network = self.network.read().await;
        let net_stats = network.stats();

        RuntimeStats {
            tick_count: *self.tick_count.read().await,
            node_count: net_stats.node_count,
            connection_count: net_stats.connection_count,
            active_signals: net_stats.field_stats.active_signals,
            total_reinforcements: net_stats.field_stats.total_reinforcements,
            peer_count: self.peers.peer_count().await,
            connected_peers: self.peers.connected_count().await,
        }
    }
}

/// Runtime statistics
#[derive(Debug, Clone)]
pub struct RuntimeStats {
    pub tick_count: u64,
    pub node_count: usize,
    pub connection_count: usize,
    pub active_signals: usize,
    pub total_reinforcements: u32,
    pub peer_count: usize,
    pub connected_peers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use smesh_core::{NetworkTopology, SignalType};

    #[tokio::test]
    async fn test_runtime_creation() {
        let runtime = SmeshRuntime::new(RuntimeConfig::default());
        let stats = runtime.stats().await;

        assert_eq!(stats.tick_count, 0);
        assert_eq!(stats.node_count, 0);
    }

    #[tokio::test]
    async fn test_runtime_with_network() {
        let network = Network::with_topology(5, NetworkTopology::Ring);
        let runtime = SmeshRuntime::with_network(network, RuntimeConfig::default());

        let stats = runtime.stats().await;
        assert_eq!(stats.node_count, 5);
    }

    #[tokio::test]
    async fn test_runtime_ticks() {
        let network = Network::with_topology(3, NetworkTopology::FullMesh);
        let runtime = SmeshRuntime::with_network(
            network,
            RuntimeConfig {
                tick_interval_ms: 10,
                ..Default::default()
            },
        );

        let events = runtime.run_ticks(5).await;
        assert_eq!(events.len(), 5);

        let stats = runtime.stats().await;
        assert_eq!(stats.tick_count, 5);
    }

    #[tokio::test]
    async fn test_signal_emission() {
        let mut network = Network::new();
        let node = Node::new();
        let node_id = node.id.clone();
        network.add_node(node);

        let runtime = SmeshRuntime::with_network(network, RuntimeConfig::default());

        let signal = Signal::builder(SignalType::Data)
            .payload(b"test".to_vec())
            .build();

        let hash = runtime.emit(signal, &node_id).await;
        assert!(hash.is_some());

        let stats = runtime.stats().await;
        assert_eq!(stats.active_signals, 1);
    }
}
