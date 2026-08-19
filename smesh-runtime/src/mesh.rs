//! Mesh layer - gossip diffusion over the QUIC transport.
//!
//! [`crate::SmeshRuntime`] on its own runs a single-process simulation:
//! [`smesh_core::Network::tick`] expands a signal's frontier by walking every
//! node's relay policy from a god's-eye view of the whole graph. No node in a
//! real mesh can see that graph.
//!
//! This module is the local-knowledge counterpart. Each process owns exactly
//! one node on the wire, and every decision it makes is one that node could
//! make alone:
//!
//! - **Dedup** is by `origin_hash`, so a signal arriving twice by different
//!   routes reinforces instead of duplicating, and loops die on arrival.
//! - **Acceptance** is [`smesh_core::Node::can_sense`] against the local
//!   sensing threshold.
//! - **Forwarding** is [`smesh_core::Node::should_relay`], scored on the
//!   receiving node's own trust in the origin and the signal's remaining hop
//!   budget. A declined relay still keeps the signal locally.
//! - **`reached_nodes` never crosses the wire.** It is one node's private
//!   record of local diffusion; it is cleared on send and rewritten to the
//!   receiving node on arrival.
//!
//! Decay is rebased against the receiver's field clock from the age stamped at
//! send time, so two hosts with skewed wall clocks still agree on how old a
//! signal is.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use serde_json::json;
use smesh_core::{Attestation, Network, NodeId, Signal};

use crate::journal::Journal;

use crate::peer::{Peer, PeerManager, PeerState};
use crate::runtime::RuntimeEvent;
use crate::transport::{QuicTransport, TransportConfig, TransportError, TransportMessage};

/// Configuration for joining a mesh.
#[derive(Debug, Clone)]
pub struct MeshConfig {
    /// Address to listen on. Port 0 asks the OS to choose one.
    pub bind_addr: SocketAddr,
    /// Peers to dial on startup. May be empty for the first node.
    pub bootstrap: Vec<SocketAddr>,
    /// How often to ping peers, in milliseconds.
    pub keepalive_interval_ms: u64,
    /// Maximum peers to disclose in a `PeerResponse`.
    pub max_peers_shared: usize,
    /// Maximum accepted frame size, in bytes.
    pub max_message_size: usize,
    /// Arbitrary node description written into the journal's opening line.
    ///
    /// Recorded here, before any loop starts, because a peer can complete the
    /// handshake the instant the endpoint binds — journalling identity from the
    /// caller afterwards races with it and can land second.
    pub node_metadata: serde_json::Value,
    /// Whether to dial peers learned second-hand from a `PeerResponse`.
    ///
    /// Off pins the topology to exactly what `bootstrap` describes, which is
    /// what you want when the shape of the mesh is the thing under study —
    /// discovery quietly converts any topology into a full mesh.
    pub peer_discovery: bool,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:0".parse().unwrap(),
            bootstrap: Vec::new(),
            keepalive_interval_ms: 5_000,
            max_peers_shared: 32,
            max_message_size: 1024 * 1024,
            node_metadata: serde_json::Value::Null,
            peer_discovery: true,
        }
    }
}

/// Shared state for the mesh tasks.
struct MeshCtx {
    transport: Arc<QuicTransport>,
    network: Arc<RwLock<Network>>,
    peers: Arc<PeerManager>,
    event_tx: mpsc::Sender<RuntimeEvent>,
    /// The one node this process presents to the mesh.
    local_node_id: NodeId,
    /// Our dialable address, advertised in `Hello`.
    listen_addr: SocketAddr,
    /// Our own public key, advertised in `Hello`.
    local_public_key: String,
    /// Socket address -> node id, learned from `Hello`.
    ///
    /// An accepted connection's source address is ephemeral, so this is the
    /// only way to attribute an inbound frame to a SMESH node.
    conn_ids: Arc<RwLock<HashMap<SocketAddr, NodeId>>>,
    /// Addresses we dialled ourselves.
    ///
    /// Whoever dials sends the first `Hello`; the other side answers. Without
    /// recording that we initiated, the answer looks like a fresh introduction
    /// and we answer the answer, registering the peer twice.
    dialed: RwLock<HashSet<SocketAddr>>,
    /// Node name -> the public key that first presented it.
    ///
    /// Signatures prove key ownership, not name ownership, so on their own they
    /// do not stop a peer calling itself `latency`. Pinning the name to the key
    /// seen first, and refusing later keys for that name, closes it. This is
    /// trust-on-first-use: it cannot help if the impostor arrives first, but it
    /// makes a name unstealable for the rest of the run.
    pinned_keys: RwLock<HashMap<NodeId, String>>,
    max_peers_shared: usize,
    peer_discovery: bool,
    journal: Arc<Journal>,
}

/// A running mesh membership. Dropping it does not stop the tasks; call
/// [`MeshHandle::shutdown`].
pub struct MeshHandle {
    transport: Arc<QuicTransport>,
    listen_addr: SocketAddr,
    tasks: Vec<JoinHandle<()>>,
}

impl MeshHandle {
    /// The address peers should dial to reach us.
    pub fn listen_addr(&self) -> SocketAddr {
        self.listen_addr
    }

    /// The underlying transport.
    pub fn transport(&self) -> Arc<QuicTransport> {
        Arc::clone(&self.transport)
    }

    /// Number of live connections.
    pub async fn connection_count(&self) -> usize {
        self.transport.peer_count().await
    }

    /// Close the endpoint and stop the mesh tasks.
    pub async fn shutdown(self) {
        self.transport.shutdown().await;
        for task in self.tasks {
            task.abort();
        }
    }
}

/// Bring up the transport, join the mesh, and start the gossip tasks.
/// Everything the mesh needs from the runtime that owns it.
pub(crate) struct MeshStartup {
    /// How to join.
    pub config: MeshConfig,
    /// The node this process presents on the wire.
    pub local_node_id: NodeId,
    /// Its public key, advertised so peers can bind the name to it.
    pub local_public_key: String,
    /// Shared field and node state.
    pub network: Arc<RwLock<Network>>,
    /// Shared peer table.
    pub peers: Arc<PeerManager>,
    /// Where runtime events go.
    pub event_tx: mpsc::Sender<RuntimeEvent>,
    /// Where the run is recorded.
    pub journal: Arc<Journal>,
    /// Shared connection address to node id map.
    pub conn_ids: Arc<RwLock<HashMap<SocketAddr, NodeId>>>,
}

pub(crate) async fn start(
    startup: MeshStartup,
) -> Result<(MeshHandle, Arc<QuicTransport>), TransportError> {
    let MeshStartup {
        config,
        local_node_id,
        local_public_key,
        network,
        peers,
        event_tx,
        journal,
        conn_ids,
    } = startup;
    let mut transport = QuicTransport::new(TransportConfig {
        bind_addr: config.bind_addr,
        max_message_size: config.max_message_size,
        keepalive_interval_ms: config.keepalive_interval_ms,
        ..Default::default()
    })
    .await?;

    let incoming = transport.take_incoming().ok_or_else(|| {
        TransportError::ConnectionFailed("incoming receiver already taken".into())
    })?;

    let listen_addr = transport.local_addr()?;
    let transport = Arc::new(transport);

    // The opening line of this node's log, written before anything can arrive.
    journal.node_started(
        &listen_addr.to_string(),
        &config
            .bootstrap
            .iter()
            .map(|addr| addr.to_string())
            .collect::<Vec<_>>(),
        config.node_metadata.clone(),
    );

    info!("mesh node {} listening on {}", local_node_id, listen_addr);

    let ctx = Arc::new(MeshCtx {
        transport: Arc::clone(&transport),
        network,
        peers,
        event_tx,
        local_node_id,
        listen_addr,
        local_public_key,
        conn_ids,
        dialed: RwLock::new(HashSet::new()),
        pinned_keys: RwLock::new(HashMap::new()),
        max_peers_shared: config.max_peers_shared,
        peer_discovery: config.peer_discovery,
        journal,
    });

    let mut tasks = Vec::new();

    // Accept inbound connections.
    {
        let transport = Arc::clone(&transport);
        tasks.push(tokio::spawn(async move {
            transport.run_accept_loop().await;
        }));
    }

    // Process decoded frames.
    {
        let ctx = Arc::clone(&ctx);
        tasks.push(tokio::spawn(async move {
            inbound_loop(ctx, incoming).await;
        }));
    }

    // Keepalive / latency probing.
    {
        let ctx = Arc::clone(&ctx);
        let interval_ms = config.keepalive_interval_ms;
        tasks.push(tokio::spawn(async move {
            keepalive_loop(ctx, interval_ms).await;
        }));
    }

    // Dial bootstrap peers and introduce ourselves.
    for addr in &config.bootstrap {
        if *addr == listen_addr {
            continue;
        }
        match dial(&ctx, *addr).await {
            Ok(()) => info!("dialled bootstrap peer {}", addr),
            Err(e) => warn!("bootstrap peer {} unreachable: {}", addr, e),
        }
    }

    Ok((
        MeshHandle {
            transport: Arc::clone(&transport),
            listen_addr,
            tasks,
        },
        transport,
    ))
}

/// Connect to a peer and send our `Hello`.
async fn dial(ctx: &MeshCtx, addr: SocketAddr) -> Result<(), TransportError> {
    ctx.transport.connect(addr).await?;
    ctx.dialed.write().await.insert(addr);
    ctx.transport.send(addr, hello(ctx)).await
}

fn hello(ctx: &MeshCtx) -> TransportMessage {
    TransportMessage::Hello {
        node_id: ctx.local_node_id.clone(),
        public_key: ctx.local_public_key.clone(),
        listen_addr: ctx.listen_addr,
    }
}

async fn inbound_loop(
    ctx: Arc<MeshCtx>,
    mut incoming: mpsc::Receiver<(SocketAddr, TransportMessage)>,
) {
    while let Some((src, msg)) = incoming.recv().await {
        let ctx = Arc::clone(&ctx);
        match msg {
            TransportMessage::Hello {
                node_id,
                public_key,
                listen_addr,
            } => on_hello(&ctx, src, node_id, public_key, listen_addr).await,

            TransportMessage::Signal { signal, age_secs } => {
                on_signal(&ctx, src, signal, age_secs).await
            }

            TransportMessage::PeerRequest { max_peers } => {
                let peers = gossip_peers(&ctx, max_peers).await;
                let _ = ctx
                    .transport
                    .send(src, TransportMessage::PeerResponse { peers })
                    .await;
            }

            TransportMessage::PeerResponse { peers } => on_peer_response(&ctx, peers).await,

            TransportMessage::Ping { timestamp } => {
                let _ = ctx
                    .transport
                    .send(src, TransportMessage::Pong { timestamp })
                    .await;
            }

            TransportMessage::Pong { timestamp } => {
                let rtt = now_millis().saturating_sub(timestamp);
                if let Some(node_id) = ctx.conn_ids.read().await.get(&src).cloned() {
                    ctx.peers.record_latency(&node_id, rtt).await;
                }
            }
        }
    }

    debug!("inbound loop ended");
}

/// Register a peer that introduced itself, and answer in kind.
async fn on_hello(
    ctx: &MeshCtx,
    src: SocketAddr,
    node_id: NodeId,
    public_key: String,
    listen_addr: SocketAddr,
) {
    if node_id == ctx.local_node_id {
        return;
    }

    // Bind this name to this key, or refuse the peer if the name is already
    // spoken for by a different one.
    if !public_key.is_empty() {
        let mut pinned = ctx.pinned_keys.write().await;
        match pinned.get(&node_id) {
            Some(known) if known != &public_key => {
                drop(pinned);
                warn!(
                    "refusing {} from {}: name already pinned to a different key",
                    node_id, src
                );
                ctx.journal.record(
                    "identity_rejected",
                    json!({
                        "peer": node_id,
                        "source_addr": src.to_string(),
                        "reason": "name already pinned to a different public key",
                    }),
                );
                return;
            }
            Some(_) => {}
            None => {
                pinned.insert(node_id.clone(), public_key.clone());
            }
        }
    }

    let first_contact = {
        let mut ids = ctx.conn_ids.write().await;
        ids.insert(src, node_id.clone()).is_none()
    };
    let already_known = ctx.peers.get_peer(&node_id).await.is_some();

    let mut peer = Peer::new(node_id.clone(), listen_addr, node_id.clone());
    peer.state = PeerState::Connected;
    peer.touch();

    if !ctx.peers.add_peer(peer).await {
        debug!("peer table full, refused {}", node_id);
        return;
    }

    if !already_known {
        ctx.journal.record(
            "peer_connected",
            json!({
                "peer": node_id,
                "public_key": public_key,
                "peer_listen_addr": listen_addr.to_string(),
                "source_addr": src.to_string(),
                "we_dialled": ctx.dialed.read().await.contains(&src),
            }),
        );

        let _ = ctx
            .event_tx
            .send(RuntimeEvent::PeerConnected {
                peer_id: node_id.clone(),
            })
            .await;
    }

    // Exactly one side answers: the dialler already introduced itself, so it
    // must not treat the reply as a fresh introduction and answer again.
    let we_dialled = ctx.dialed.read().await.contains(&src);
    if first_contact && !we_dialled {
        let _ = ctx.transport.send(src, hello(ctx)).await;

        let peers = gossip_peers(ctx, ctx.max_peers_shared).await;
        if !peers.is_empty() {
            let _ = ctx
                .transport
                .send(src, TransportMessage::PeerResponse { peers })
                .await;
        }
    }
}

/// Dial peers we were told about but have not met.
async fn on_peer_response(ctx: &MeshCtx, peers: Vec<(String, SocketAddr)>) {
    if !ctx.peer_discovery {
        return;
    }
    for (node_id, addr) in peers {
        if node_id == ctx.local_node_id || addr == ctx.listen_addr {
            continue;
        }
        if ctx.peers.get_peer(&node_id).await.is_some() {
            continue;
        }
        debug!("learned about {} at {}, dialling", node_id, addr);
        if let Err(e) = dial(ctx, addr).await {
            debug!("could not reach learned peer {}: {}", addr, e);
        }
    }
}

async fn gossip_peers(ctx: &MeshCtx, max: usize) -> Vec<(String, SocketAddr)> {
    ctx.peers
        .connected_peers()
        .await
        .into_iter()
        .take(max)
        .map(|p| (p.node_id, p.addr))
        .collect()
}

/// What the local node decided to do with an arriving signal.
enum Outcome {
    /// Already known, but the message carried attesters we had not seen.
    ///
    /// The attester set is a grow-only set, so merging is what makes gossip
    /// converge: a node forwards whenever its own knowledge grew, and stops
    /// when it did not. That is also the loop breaker — a message that teaches
    /// us nothing goes no further.
    Merged {
        hash: String,
        new_attesters: Vec<String>,
        attesters: Vec<String>,
        confidence: f64,
        forward: Option<Box<Signal>>,
    },
    /// New and sensable. `forward` is set if the relay policy said yes.
    ///
    /// The signal is boxed so this variant does not inflate the whole enum.
    Accepted {
        hash: String,
        hops: u32,
        attesters: Vec<String>,
        forward: Option<Box<Signal>>,
    },
    /// Not taken up, with the reason.
    Dropped { hash: String, reason: String },
}

/// Reject a signal whose attestations use a name pinned to another key.
///
/// A valid signature proves you hold *a* key, not that you are entitled to the
/// name you signed under. Returns the offending names, or `None` if all is
/// well.
async fn reject_unpinned(ctx: &MeshCtx, attestations: &[Attestation]) -> Option<Vec<NodeId>> {
    let pinned = ctx.pinned_keys.read().await;
    let offenders: Vec<NodeId> = attestations
        .iter()
        .filter(|a| {
            pinned
                .get(&a.node_id)
                .is_some_and(|known| known != &a.public_key)
        })
        .map(|a| a.node_id.clone())
        .collect();

    (!offenders.is_empty()).then_some(offenders)
}

/// Apply the local node's own policy to a signal that arrived over the wire.
async fn on_signal(ctx: &MeshCtx, src: SocketAddr, mut signal: Signal, age_secs: f64) {
    let relayed_by = ctx
        .conn_ids
        .read()
        .await
        .get(&src)
        .cloned()
        .unwrap_or_else(|| src.to_string());

    let hash = signal.origin_hash.clone();

    // Count signatures, not names. `reinforced_by` arriving off a socket is
    // just a list the sender wrote; only attestations carry proof.
    let incoming_attestations = signal.attestations.clone();
    let incoming_attesters = signal.verified_attesters();
    let unverifiable = signal.attestations.len() - incoming_attesters.len();

    if let Some(rejected) = reject_unpinned(ctx, &incoming_attestations).await {
        ctx.journal.record(
            "identity_rejected",
            json!({
                "hash": hash,
                "relayed_by": relayed_by,
                "attesters": rejected,
                "reason": "attestation used a name pinned to a different public key",
            }),
        );
        return;
    }

    ctx.journal.record(
        "signal_received",
        json!({
            "hash": hash,
            "relayed_by": relayed_by,
            "origin": signal.origin_node_id,
            "hops": signal.hops,
            "age_secs": age_secs,
            "intensity": signal.current_intensity,
            "confidence": signal.confidence,
            "attesters": incoming_attesters,
            "unverifiable_attestations": unverifiable,
        }),
    );

    let outcome = {
        let mut network = ctx.network.write().await;
        let now = network.field.current_time;

        // Rebase decay onto our field clock: the sender told us how old the
        // signal was when it left, not when it was born by their wall clock.
        signal.created_at = now - chrono::Duration::milliseconds((age_secs * 1000.0) as i64);
        signal.current_intensity = signal.compute_intensity(now);

        if network.field.signals.contains_key(&hash) {
            // Merge the two attester sets. Anything the sender knew that we did
            // not is new information, and new information is worth passing on.
            let existing = network.field.signals.get_mut(&hash).expect("checked above");
            let new_attesters = existing.merge_attestations(&incoming_attestations);
            // Keep the local view in step with what actually verified.
            for attester in &new_attesters {
                existing.reinforce(attester);
            }
            let attesters = existing.verified_attesters();

            if new_attesters.is_empty() {
                Outcome::Dropped {
                    hash,
                    reason: "no new attesters".to_string(),
                }
            } else {
                let merged = existing.clone();
                let confidence = merged.confidence;

                // Gossip the merged view onward, subject to the same relay
                // policy a fresh signal would face.
                let forward = relay_forward(ctx, &network, &merged, &hash);

                if let Some(node) = network.nodes.get_mut(&ctx.local_node_id) {
                    node.stats.signals_reinforced += 1;
                    if forward.is_some() {
                        node.stats.signals_relayed += 1;
                    }
                }

                Outcome::Merged {
                    hash,
                    new_attesters,
                    attesters,
                    confidence,
                    forward,
                }
            }
        } else if incoming_attesters.is_empty() {
            // Nobody provably stands behind this. Before signatures existed the
            // origin was a bare string, so this is the case that used to let
            // anyone speak as anyone.
            Outcome::Dropped {
                hash,
                reason: "no verifiable attestation".to_string(),
            }
        } else if signal.is_expired(now) {
            Outcome::Dropped {
                hash,
                reason: "expired in flight".to_string(),
            }
        } else if signal.hops > signal.radius {
            Outcome::Dropped {
                hash,
                reason: "hop budget exhausted".to_string(),
            }
        } else {
            let Some(local) = network.nodes.get(&ctx.local_node_id) else {
                return;
            };

            if !local.can_sense(&signal) {
                Outcome::Dropped {
                    hash,
                    reason: "below sensing threshold".to_string(),
                }
            } else {
                // reached_nodes is this node's private view of local diffusion.
                // Whatever the sender knew about its own graph is meaningless
                // here, so replace it outright.
                signal.reached_nodes = vec![ctx.local_node_id.clone()];
                // Drop the sender's unsigned name list; only signatures survive
                // the trip, and they are already on the signal.
                signal.reinforced_by = incoming_attesters.clone();

                let hops = signal.hops;
                let attesters = incoming_attesters.clone();
                let forward = relay_forward(ctx, &network, &signal, &hash);

                network.field.signals.insert(hash.clone(), signal);

                if let Some(node) = network.nodes.get_mut(&ctx.local_node_id) {
                    node.stats.signals_sensed += 1;
                    if forward.is_some() {
                        node.stats.signals_relayed += 1;
                    }
                }

                Outcome::Accepted {
                    hash,
                    hops,
                    attesters,
                    forward,
                }
            }
        }
    };

    match outcome {
        Outcome::Merged {
            hash,
            new_attesters,
            attesters,
            confidence,
            forward,
        } => {
            ctx.journal.record(
                "signal_reinforced",
                json!({
                    "hash": hash,
                    "relayed_by": relayed_by,
                    "new_attesters": new_attesters,
                    "attesters": attesters,
                    "attester_count": attesters.len(),
                    "confidence": confidence,
                }),
            );

            let _ = ctx
                .event_tx
                .send(RuntimeEvent::SignalReinforced {
                    hash: hash.clone(),
                    count: attesters.len() as u32,
                })
                .await;

            forward_signal(ctx, forward, &hash, Some(src)).await;
        }

        Outcome::Accepted {
            hash,
            hops,
            attesters,
            forward,
        } => {
            ctx.journal.record(
                "signal_accepted",
                json!({
                    "hash": hash,
                    "relayed_by": relayed_by,
                    "hops": hops,
                    "attesters": attesters,
                    "attester_count": attesters.len(),
                }),
            );

            let _ = ctx
                .event_tx
                .send(RuntimeEvent::SignalReceived {
                    hash: hash.clone(),
                    from: relayed_by,
                    hops,
                })
                .await;

            forward_signal(ctx, forward, &hash, Some(src)).await;
        }

        Outcome::Dropped { hash, reason } => {
            ctx.journal.record(
                "signal_dropped",
                json!({ "hash": hash, "relayed_by": relayed_by, "reason": reason }),
            );
            debug!("dropped signal from {}: {}", relayed_by, reason);
        }
    }
}

/// Ask the local node whether to forward `signal`, journalling the reasoning.
///
/// Returns the dampened copy to send, or `None` if the policy declined. Takes
/// the network by reference so the caller keeps the lock across the decision.
fn relay_forward(
    ctx: &MeshCtx,
    network: &Network,
    signal: &Signal,
    hash: &str,
) -> Option<Box<Signal>> {
    let local = network.nodes.get(&ctx.local_node_id)?;
    let remaining = signal.radius.saturating_sub(signal.hops);
    let decision = local.relay_decision(signal, remaining);

    ctx.journal.record(
        "relay_decision",
        json!({
            "hash": hash,
            "relay": decision.relay,
            "propagation_score": decision.propagation_score,
            "origin_trust": decision.origin_trust,
            "roll": decision.roll,
            "remaining_hops": decision.remaining_hops,
            "veto": decision.veto,
        }),
    );

    decision.relay.then(|| {
        let mut fwd = signal.propagate(decision.dampening);
        fwd.reached_nodes.clear();
        Box::new(fwd)
    })
}

/// Send a forwarded copy to every peer but the one it came from.
async fn forward_signal(
    ctx: &MeshCtx,
    forward: Option<Box<Signal>>,
    hash: &str,
    except: Option<SocketAddr>,
) {
    let Some(fwd) = forward else {
        return;
    };

    let hops = fwd.hops;
    let intensity = fwd.current_intensity;
    let msg = TransportMessage::signal(*fwd, chrono::Utc::now());
    let reached = ctx.transport.broadcast_all(&msg, except).await;

    for addr in reached {
        let peer = ctx
            .conn_ids
            .read()
            .await
            .get(&addr)
            .cloned()
            .unwrap_or_else(|| addr.to_string());

        ctx.journal.record(
            "signal_sent",
            json!({
                "hash": hash,
                "to": peer,
                "to_addr": addr.to_string(),
                "hops": hops,
                "intensity": intensity,
                "kind": "relay",
            }),
        );
    }
}

async fn keepalive_loop(ctx: Arc<MeshCtx>, interval_ms: u64) {
    let mut ticker = tokio::time::interval(Duration::from_millis(interval_ms.max(100)));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        ticker.tick().await;

        let msg = TransportMessage::Ping {
            timestamp: now_millis(),
        };
        ctx.transport.broadcast_all(&msg, None).await;

        reap_dead_peers(&ctx).await;
    }
}

/// Retire peers whose connection the transport has dropped.
///
/// The transport removes a connection once its stream reader ends, which is
/// the only place a hangup is observable. Reconciling here keeps the peer
/// table from reporting a peer as connected after its socket is gone.
async fn reap_dead_peers(ctx: &MeshCtx) {
    let live: HashSet<SocketAddr> = ctx.transport.connected_addrs().await.into_iter().collect();

    let gone: Vec<(SocketAddr, NodeId)> = {
        let ids = ctx.conn_ids.read().await;
        ids.iter()
            .filter(|(addr, _)| !live.contains(*addr))
            .map(|(addr, node_id)| (*addr, node_id.clone()))
            .collect()
    };

    if gone.is_empty() {
        return;
    }

    {
        let mut ids = ctx.conn_ids.write().await;
        for (addr, _) in &gone {
            ids.remove(addr);
        }
    }

    for (addr, node_id) in gone {
        ctx.peers
            .update_state(&node_id, PeerState::Disconnected)
            .await;
        // Forget that we dialled this address, so a later reconnect performs a
        // full handshake instead of assuming it is still our own outbound leg.
        ctx.dialed.write().await.remove(&addr);

        ctx.journal.record(
            "peer_disconnected",
            json!({ "peer": node_id, "source_addr": addr.to_string() }),
        );

        let _ = ctx
            .event_tx
            .send(RuntimeEvent::PeerDisconnected { peer_id: node_id })
            .await;
    }
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
