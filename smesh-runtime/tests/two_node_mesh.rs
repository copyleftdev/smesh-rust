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
use smesh_runtime::{MeshConfig, MeshHandle, RuntimeConfig, SmeshRuntime};

const LOCALHOST: &str = "127.0.0.1:0";

/// A single-node runtime that has joined the mesh.
struct MeshNode {
    runtime: Arc<SmeshRuntime>,
    handle: MeshHandle,
    node_id: NodeId,
}

impl MeshNode {
    async fn start(name: &str, bootstrap: Vec<SocketAddr>) -> Self {
        let mut node = Node::new();
        node.id = name.to_string();
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

    async fn signal_count(&self, hash: &str) -> usize {
        let network = self.runtime.network();
        let network = network.read().await;
        network
            .field
            .signals
            .keys()
            .filter(|k| k.as_str() == hash)
            .count()
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
    // Fully connected triangle: A's signal can reach C directly and via B.
    let a = MeshNode::start("node-a", vec![]).await;
    let b = MeshNode::start("node-b", vec![a.addr()]).await;
    let c = MeshNode::start("node-c", vec![a.addr(), b.addr()]).await;

    eventually(Duration::from_secs(10), "triangle forms", || async {
        a.runtime.peers().connected_count().await == 2
            && b.runtime.peers().connected_count().await == 2
            && c.runtime.peers().connected_count().await == 2
    })
    .await;

    let hash = a.emit("consensus please").await;

    eventually(Duration::from_secs(5), "b and c both have it", || async {
        b.has_signal(&hash).await && c.has_signal(&hash).await
    })
    .await;

    // Give any relayed copies time to arrive and be deduped.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Content addressing is what stops a flood from becoming duplicate state:
    // a second arrival reinforces the existing signal rather than adding one.
    assert_eq!(c.signal_count(&hash).await, 1);
    assert_eq!(b.signal_count(&hash).await, 1);

    // And the origin never loops back to itself as a new signal.
    assert_eq!(a.signal_count(&hash).await, 1);

    a.shutdown().await;
    b.shutdown().await;
    c.shutdown().await;
}
