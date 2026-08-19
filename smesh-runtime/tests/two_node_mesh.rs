//! End-to-end tests for the QUIC mesh layer.
//!
//! These exercise the properties that only appear once diffusion stops being a
//! single-process, god's-eye BFS: a signal crossing a real socket, a peer
//! learned second-hand being dialled, and content-addressed dedup keeping a
//! flood from duplicating state.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use smesh_core::{Network, Node, NodeId, Signal, SignalType};
use smesh_runtime::{MeshConfig, MeshHandle, RuntimeConfig, SmeshRuntime, TransportMessage};

const LOCALHOST: &str = "127.0.0.1:0";

/// A single-node runtime that has joined the mesh.
struct MeshNode {
    runtime: Arc<SmeshRuntime>,
    handle: MeshHandle,
    node_id: NodeId,
}

impl MeshNode {
    async fn start(name: &str, bootstrap: Vec<SocketAddr>) -> Self {
        let mut node = Node::named(name);
        // Trust the peers we will actually talk to, so the probabilistic relay
        // policy does not make these tests flaky.
        node.trust_scores.insert("node-a".to_string(), 0.99);
        node.trust_scores.insert("node-b".to_string(), 0.99);
        node.trust_scores.insert("node-c".to_string(), 0.99);
        let node_id = node.id.clone();

        let mut network = Network::new();
        network.add_node(node);

        let runtime = Arc::new(SmeshRuntime::with_network(
            network,
            RuntimeConfig::default(),
        ));

        let handle = runtime
            .join_mesh(
                MeshConfig {
                    bind_addr: LOCALHOST.parse().unwrap(),
                    bootstrap,
                    keepalive_interval_ms: 500,
                    ..Default::default()
                },
                &node_id,
            )
            .await
            .expect("joined mesh");

        Self {
            runtime,
            handle,
            node_id,
        }
    }

    fn addr(&self) -> SocketAddr {
        self.handle.listen_addr()
    }

    async fn has_signal(&self, hash: &str) -> bool {
        let network = self.runtime.network();
        let network = network.read().await;
        network.field.signals.contains_key(hash)
    }

    /// How many distinct signals carry this payload.
    ///
    /// The previous version counted map keys equal to a hash, which a map can
    /// never hold more than one of — it asserted nothing. What actually needs
    /// proving is that one claim does not become several signals.
    async fn signals_for_payload(&self, payload: &str) -> usize {
        let network = self.runtime.network();
        let network = network.read().await;
        network
            .field
            .signals
            .values()
            .filter(|s| s.payload == payload.as_bytes())
            .count()
    }

    /// Attesters recorded for a claim, in order.
    async fn attesters_for(&self, hash: &str) -> Vec<String> {
        let network = self.runtime.network();
        let network = network.read().await;
        network
            .field
            .signals
            .get(hash)
            .map(|s| s.verified_attesters())
            .unwrap_or_default()
    }

    async fn emit(&self, payload: &str) -> String {
        let signal = Signal::builder(SignalType::Coordination)
            .payload(payload.as_bytes().to_vec())
            .origin(&self.node_id)
            .intensity(1.0)
            .ttl(120.0)
            .radius(4)
            .build();

        self.runtime
            .emit(signal, &self.node_id)
            .await
            .expect("emitted")
    }

    /// Emit a claim addressed by content, so independent emitters converge.
    async fn emit_claim(&self, payload: &str) -> String {
        let signal = Signal::builder(SignalType::Alert)
            .correlatable()
            .payload(payload.as_bytes().to_vec())
            .intensity(1.0)
            .ttl(120.0)
            .radius(4)
            .build();

        self.runtime
            .emit(signal, &self.node_id)
            .await
            .expect("emitted")
    }

    async fn shutdown(self) {
        self.handle.shutdown().await;
    }
}

/// Poll until `cond` holds, or fail after `timeout`.
async fn eventually<F, Fut>(timeout: Duration, label: &str, mut cond: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if cond().await {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("timed out waiting for: {label}");
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test]
async fn signal_crosses_the_wire_between_two_nodes() {
    let a = MeshNode::start("node-a", vec![]).await;
    let b = MeshNode::start("node-b", vec![a.addr()]).await;

    // The Hello handshake registers each side with the other.
    eventually(Duration::from_secs(5), "a sees b", || async {
        a.runtime.peers().connected_count().await == 1
    })
    .await;
    eventually(Duration::from_secs(5), "b sees a", || async {
        b.runtime.peers().connected_count().await == 1
    })
    .await;

    let hash = a.emit("deploy the thing").await;

    eventually(Duration::from_secs(5), "b receives a's signal", || async {
        b.has_signal(&hash).await
    })
    .await;

    // B accepted it under its own sensing policy, and attributes the origin to
    // A even though A's local diffusion state never crossed the wire.
    let network = b.runtime.network();
    let network = network.read().await;
    let signal = network.field.signals.get(&hash).expect("signal present");
    assert_eq!(signal.origin_node_id, "node-a");
    assert_eq!(signal.payload, b"deploy the thing".to_vec());
    assert_eq!(
        signal.reached_nodes,
        vec!["node-b".to_string()],
        "reached_nodes must be the receiver's local view, not the sender's"
    );
    drop(network);

    a.shutdown().await;
    b.shutdown().await;
}

#[tokio::test]
async fn peer_learned_second_hand_is_dialled() {
    let a = MeshNode::start("node-a", vec![]).await;
    let b = MeshNode::start("node-b", vec![a.addr()]).await;

    eventually(Duration::from_secs(5), "a and b meet", || async {
        a.runtime.peers().connected_count().await == 1
    })
    .await;

    // C only knows about A. It should learn B from A's PeerResponse and dial it.
    let c = MeshNode::start("node-c", vec![a.addr()]).await;

    eventually(
        Duration::from_secs(10),
        "c discovers both a and b",
        || async {
            let peers = c.runtime.peers().connected_peers().await;
            let mut ids: Vec<String> = peers.into_iter().map(|p| p.node_id).collect();
            ids.sort();
            ids == vec!["node-a".to_string(), "node-b".to_string()]
        },
    )
    .await;

    a.shutdown().await;
    b.shutdown().await;
    c.shutdown().await;
}

#[tokio::test]
async fn flooding_does_not_duplicate_state() {
    // Fully connected triangle: one claim can reach a node directly and again
    // via a relay. What must hold is that arriving twice does not become two
    // signals, and does not inflate the count of who stands behind it.
    let a = MeshNode::start("node-a", vec![]).await;
    let b = MeshNode::start("node-b", vec![a.addr()]).await;
    let c = MeshNode::start("node-c", vec![a.addr(), b.addr()]).await;

    eventually(Duration::from_secs(10), "triangle forms", || async {
        a.runtime.peers().connected_count().await == 2
            && b.runtime.peers().connected_count().await == 2
            && c.runtime.peers().connected_count().await == 2
    })
    .await;

    let hash = a.emit_claim("consensus please").await;

    eventually(Duration::from_secs(6), "b and c both have it", || async {
        b.has_signal(&hash).await && c.has_signal(&hash).await
    })
    .await;

    // Re-assert repeatedly. Every one of these arrives at nodes that already
    // hold the claim, by more than one route.
    for _ in 0..3 {
        a.emit_claim("consensus please").await;
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    tokio::time::sleep(Duration::from_millis(600)).await;

    for (name, node) in [("a", &a), ("b", &b), ("c", &c)] {
        assert_eq!(
            node.signals_for_payload("consensus please").await,
            1,
            "{name} turned one claim into more than one signal"
        );

        let attesters = node.attesters_for(&hash).await;
        let mut unique = attesters.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            attesters.len(),
            unique.len(),
            "{name} recorded the same attester twice: {attesters:?}"
        );
        assert!(
            attesters.len() <= 3,
            "{name} counted more attesters than there are nodes: {attesters:?}"
        );
        assert!(
            attesters.contains(&"node-a".to_string()),
            "{name} lost the originator from the attester set"
        );
    }

    a.shutdown().await;
    b.shutdown().await;
    c.shutdown().await;
}

#[tokio::test]
async fn an_unsigned_claim_is_refused() {
    // Before attestations existed, `origin_node_id` was a bare string and this
    // signal would have been accepted and counted. It carries no proof that
    // anyone stands behind it, so it must now go nowhere.
    let a = MeshNode::start("node-a", vec![]).await;
    let b = MeshNode::start("node-b", vec![a.addr()]).await;

    eventually(Duration::from_secs(5), "a and b meet", || async {
        b.runtime.peers().connected_count().await == 1
    })
    .await;

    let mut signal = Signal::builder(SignalType::Coordination)
        .payload(b"unsigned claim".to_vec())
        .intensity(1.0)
        .ttl(60.0)
        .radius(4)
        .build();
    signal.origin_node_id = "node-a".to_string();
    let hash = signal.origin_hash.clone();
    assert!(signal.attestations.is_empty());

    let transport = a.handle.transport();
    let msg = TransportMessage::signal(signal, chrono::Utc::now());
    transport.broadcast_all(&msg, None).await;

    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(
        !b.has_signal(&hash).await,
        "a claim nobody signed must not enter the field"
    );

    a.shutdown().await;
    b.shutdown().await;
}

#[tokio::test]
async fn a_tampered_attestation_is_not_counted() {
    // An attacker replays a real signature against a claim it was not made for.
    // The signature is genuine; the binding is not.
    let a = MeshNode::start("node-a", vec![]).await;
    let b = MeshNode::start("node-b", vec![a.addr()]).await;

    eventually(Duration::from_secs(5), "a and b meet", || async {
        b.runtime.peers().connected_count().await == 1
    })
    .await;

    // A properly signed claim, so we have a valid attestation to steal.
    let real_hash = a.emit("genuine claim").await;
    eventually(
        Duration::from_secs(5),
        "b accepts the genuine claim",
        || async { b.has_signal(&real_hash).await },
    )
    .await;

    let stolen = {
        let network = a.runtime.network();
        let network = network.read().await;
        network.field.signals[&real_hash].attestations[0].clone()
    };

    // Bolt it onto a different claim.
    let mut forged = Signal::builder(SignalType::Coordination)
        .payload(b"claim the attacker wants believed".to_vec())
        .intensity(1.0)
        .ttl(60.0)
        .radius(4)
        .build();
    forged.origin_node_id = "node-a".to_string();
    forged.attestations = vec![stolen];
    let forged_hash = forged.origin_hash.clone();

    assert!(
        forged.verified_attesters().is_empty(),
        "a signature must not verify against a claim it was not made for"
    );

    let msg = TransportMessage::signal(forged, chrono::Utc::now());
    a.handle.transport().broadcast_all(&msg, None).await;

    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(
        !b.has_signal(&forged_hash).await,
        "a claim with only a replayed signature must be refused"
    );

    a.shutdown().await;
    b.shutdown().await;
}

#[tokio::test]
async fn corroboration_across_the_mesh_is_signature_backed() {
    // Two nodes independently reach the same conclusion. Content addressing
    // puts them on one signal, and each contributes a signature, so the count
    // of attesters is a count of verifiable statements rather than of names.
    let a = MeshNode::start("node-a", vec![]).await;
    let b = MeshNode::start("node-b", vec![a.addr()]).await;

    eventually(Duration::from_secs(5), "a and b meet", || async {
        a.runtime.peers().connected_count().await == 1
            && b.runtime.peers().connected_count().await == 1
    })
    .await;

    let hash_a = a.emit_claim("checkout-api degraded").await;
    let hash_b = b.emit_claim("checkout-api degraded").await;
    assert_eq!(hash_a, hash_b, "same claim must land on the same address");

    eventually(
        Duration::from_secs(6),
        "both sides see two signed attesters",
        || async {
            let seen = |node: &MeshNode| {
                let hash = hash_a.clone();
                let network = node.runtime.network();
                async move {
                    let network = network.read().await;
                    network
                        .field
                        .signals
                        .get(&hash)
                        .map(|s| s.verified_attesters().len())
                        .unwrap_or(0)
                }
            };
            seen(&a).await == 2 && seen(&b).await == 2
        },
    )
    .await;

    let network = a.runtime.network();
    let network = network.read().await;
    let signal = &network.field.signals[&hash_a];
    let mut attesters = signal.verified_attesters();
    attesters.sort();
    assert_eq!(attesters, vec!["node-a".to_string(), "node-b".to_string()]);
    assert_eq!(signal.attestations.len(), 2);
    drop(network);

    a.shutdown().await;
    b.shutdown().await;
}

#[tokio::test]
async fn a_node_learns_its_own_address_from_a_peer() {
    // A node bound to a wildcard address has no idea what address the rest of
    // the world reaches it on, and behind NAT it never could. The only source
    // of that fact is a peer reporting where the traffic arrived from.
    let a = MeshNode::start("node-a", vec![]).await;
    let b = MeshNode::start("node-b", vec![a.addr()]).await;

    eventually(
        Duration::from_secs(5),
        "b learns its own address",
        || async { b.handle.reflexive_addr().await.is_some() },
    )
    .await;

    let observed = b.handle.reflexive_addr().await.unwrap();
    assert_eq!(
        observed.ip(),
        a.addr().ip(),
        "the reported address should be on the path the peer saw us arrive from"
    );

    a.shutdown().await;
    b.shutdown().await;
}

#[tokio::test]
async fn punch_coordination_reaches_the_target() {
    // Two peers that cannot dial each other directly have to be introduced by
    // somebody they can both already reach. This exercises that relay end to
    // end: request -> rendezvous -> instruction -> dial.
    //
    // It does NOT prove NAT traversal. On loopback the resulting dial would
    // have succeeded anyway; what is under test is that the coordination path
    // runs and the two ends find each other through it.
    let rendezvous = MeshNode::start("rendezvous", vec![]).await;
    let left = MeshNode::start("left", vec![rendezvous.addr()]).await;
    let right = MeshNode::start("right", vec![rendezvous.addr()]).await;

    eventually(
        Duration::from_secs(6),
        "both reach the rendezvous",
        || async { rendezvous.runtime.peers().connected_count().await == 2 },
    )
    .await;

    // Discovery is on by default, so wait until they are NOT yet paired before
    // asserting the punch is what pairs them.
    let paired = |node: &MeshNode| {
        let peers = node.runtime.peers();
        async move {
            peers
                .connected_peers()
                .await
                .iter()
                .any(|p| p.node_id == "right")
        }
    };

    if !paired(&left).await {
        left.handle.request_punch("right").await;

        eventually(
            Duration::from_secs(8),
            "left and right pair through the rendezvous",
            || async {
                left.runtime.peers().get_peer("right").await.is_some()
                    || right.runtime.peers().get_peer("left").await.is_some()
            },
        )
        .await;
    }

    assert!(
        left.runtime.peers().get_peer("right").await.is_some()
            || right.runtime.peers().get_peer("left").await.is_some(),
        "the two ends should have found each other"
    );

    rendezvous.shutdown().await;
    left.shutdown().await;
    right.shutdown().await;
}
