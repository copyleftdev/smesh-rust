//! SMESH Runtime - async event loop for signal processing

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, info};

use crate::peer::{PeerId, PeerManager};
use crate::transport::TransportConfig;
use smesh_core::{Network, Node, Signal};

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
        }
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

    /// Emit a signal into the network
    pub async fn emit(&self, signal: Signal, node_id: &str) -> Option<String> {
        let mut network = self.network.write().await;

        // Check if node exists first
        if !network.nodes.contains_key(node_id) {
            return None;
        }

        // Emit signal anonymously first, then update node stats
        let hash = network.field.emit_anonymous(signal);

        // Update node stats
        if let Some(node) = network.nodes.get_mut(node_id) {
            node.stats.signals_emitted += 1;
        }

        let _ = self
            .event_tx
            .send(RuntimeEvent::SignalEmitted { hash: hash.clone() })
            .await;

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
