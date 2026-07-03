//! Network topology - how nodes are connected
//!
//! Defines network structures and routing through hyphae.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{Field, Node, NodeId};

/// Network topology types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NetworkTopology {
    /// Fully connected mesh
    FullMesh,
    /// Random graph with given edge probability
    Random,
    /// Small world network (Watts-Strogatz)
    #[default]
    SmallWorld,
    /// Scale-free network (Barabasi-Albert)
    ScaleFree,
    /// Ring topology
    Ring,
    /// Grid topology
    Grid,
}

/// A hypha - a directed connection between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypha {
    /// Source node
    pub from: NodeId,
    /// Target node
    pub to: NodeId,
    /// Connection strength (affects signal dampening)
    pub strength: f64,
    /// Latency in milliseconds
    pub latency: f64,
    /// Whether this connection is active
    pub active: bool,
}

impl Hypha {
    pub fn new(from: &str, to: &str) -> Self {
        Self {
            from: from.to_string(),
            to: to.to_string(),
            strength: 1.0,
            latency: 0.0,
            active: true,
        }
    }

    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    pub fn with_latency(mut self, latency: f64) -> Self {
        self.latency = latency.max(0.0);
        self
    }
}

/// The network connecting nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    /// All nodes in the network
    pub nodes: HashMap<NodeId, Node>,

    /// Connections between nodes (from_id -> [Hypha])
    pub hyphae: HashMap<NodeId, Vec<Hypha>>,

    /// The shared signal field
    pub field: Field,

    /// Network topology type
    pub topology: NetworkTopology,
}

impl Network {
    /// Create a new empty network
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            hyphae: HashMap::new(),
            field: Field::new(),
            topology: NetworkTopology::default(),
        }
    }

    /// Create a network with a specific topology
    pub fn with_topology(n_nodes: usize, topology: NetworkTopology) -> Self {
        let mut network = Self::new();
        network.topology = topology;

        // Create nodes
        for _ in 0..n_nodes {
            let node = Node::new();
            network.add_node(node);
        }

        // Build topology
        network.build_topology();

        network
    }

    /// Add a node to the network
    pub fn add_node(&mut self, node: Node) {
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        self.hyphae.insert(id, Vec::new());
    }

    /// Remove a node from the network
    pub fn remove_node(&mut self, node_id: &str) -> Option<Node> {
        self.hyphae.remove(node_id);

        // Remove hyphae pointing to this node
        for connections in self.hyphae.values_mut() {
            connections.retain(|h| h.to != node_id);
        }

        self.nodes.remove(node_id)
    }

    /// Connect two nodes with a hypha
    pub fn connect(&mut self, from: &str, to: &str) {
        if self.nodes.contains_key(from) && self.nodes.contains_key(to) {
            let hypha = Hypha::new(from, to);
            self.hyphae.entry(from.to_string()).or_default().push(hypha);
        }
    }

    /// Connect two nodes bidirectionally
    pub fn connect_bidirectional(&mut self, a: &str, b: &str) {
        self.connect(a, b);
        self.connect(b, a);
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: &str) -> Vec<&NodeId> {
        self.hyphae
            .get(node_id)
            .map(|connections| connections.iter().map(|h| &h.to).collect())
            .unwrap_or_default()
    }

    /// Get a node by ID
    pub fn get_node(&self, node_id: &str) -> Option<&Node> {
        self.nodes.get(node_id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, node_id: &str) -> Option<&mut Node> {
        self.nodes.get_mut(node_id)
    }

    /// Build the network topology
    fn build_topology(&mut self) {
        let node_ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        let n = node_ids.len();

        if n < 2 {
            return;
        }

        match self.topology {
            NetworkTopology::FullMesh => {
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            self.connect(&node_ids[i], &node_ids[j]);
                        }
                    }
                }
            }
            NetworkTopology::Random => {
                let mut rng = rand::thread_rng();
                let edge_prob = 0.3;

                for i in 0..n {
                    for j in (i + 1)..n {
                        if rng.gen::<f64>() < edge_prob {
                            self.connect_bidirectional(&node_ids[i], &node_ids[j]);
                        }
                    }
                }
            }
            NetworkTopology::SmallWorld => {
                // Watts-Strogatz: start with ring, rewire with probability
                let k = 4.min(n - 1); // Each node connects to k nearest neighbors
                let rewire_prob = 0.3;
                let mut rng = rand::thread_rng();

                // Create ring with k nearest neighbors
                for i in 0..n {
                    for j in 1..=k / 2 {
                        let neighbor = (i + j) % n;
                        self.connect_bidirectional(&node_ids[i], &node_ids[neighbor]);
                    }
                }

                // Rewire edges
                for i in 0..n {
                    for j in 1..=k / 2 {
                        if rng.gen::<f64>() < rewire_prob {
                            let old_neighbor = (i + j) % n;
                            let new_neighbor = rng.gen_range(0..n);

                            if new_neighbor != i && new_neighbor != old_neighbor {
                                // Remove old connection, add new
                                if let Some(connections) = self.hyphae.get_mut(&node_ids[i]) {
                                    connections.retain(|h| h.to != node_ids[old_neighbor]);
                                }
                                self.connect(&node_ids[i], &node_ids[new_neighbor]);
                            }
                        }
                    }
                }
            }
            NetworkTopology::ScaleFree => {
                // Barabasi-Albert preferential attachment
                let m = 2; // Edges to add for each new node
                let mut rng = rand::thread_rng();
                let mut degrees: Vec<usize> = vec![0; n];

                // Start with a small connected core
                for i in 0..m.min(n) {
                    for j in (i + 1)..m.min(n) {
                        self.connect_bidirectional(&node_ids[i], &node_ids[j]);
                        degrees[i] += 1;
                        degrees[j] += 1;
                    }
                }

                // Add remaining nodes with preferential attachment
                for i in m..n {
                    let total_degree: usize = degrees.iter().take(i).sum();
                    let mut connected = HashSet::new();

                    while connected.len() < m && connected.len() < i {
                        let target = (0..i).find(|&j| {
                            !connected.contains(&j)
                                && rng.gen::<f64>() < degrees[j] as f64 / total_degree.max(1) as f64
                        });

                        if let Some(j) = target {
                            self.connect_bidirectional(&node_ids[i], &node_ids[j]);
                            degrees[i] += 1;
                            degrees[j] += 1;
                            connected.insert(j);
                        }
                    }
                }
            }
            NetworkTopology::Ring => {
                for i in 0..n {
                    let next = (i + 1) % n;
                    self.connect_bidirectional(&node_ids[i], &node_ids[next]);
                }
            }
            NetworkTopology::Grid => {
                let side = (n as f64).sqrt().ceil() as usize;

                for i in 0..n {
                    let _row = i / side;
                    let col = i % side;

                    // Connect to right neighbor
                    if col + 1 < side && i + 1 < n {
                        self.connect_bidirectional(&node_ids[i], &node_ids[i + 1]);
                    }

                    // Connect to bottom neighbor
                    if i + side < n {
                        self.connect_bidirectional(&node_ids[i], &node_ids[i + side]);
                    }
                }
            }
        }
    }

    /// Run one simulation tick.
    ///
    /// Signals decay in the field and then *diffuse* one network hop outward.
    /// Each signal carries the set of nodes it has already reached
    /// ([`Signal::reached_nodes`]); every tick it spreads from that frontier to
    /// the not-yet-reached neighbours of every reached node, gated by each
    /// target's [`Node::should_relay`] policy and dampened by hypha strength.
    /// A signal therefore takes N ticks to travel N hops, spreading through the
    /// topology until it reaches its `radius` or decays away.
    ///
    /// The returned `propagated_signals` is the number of *new node arrivals*
    /// this tick (how much wider every signal's reach grew combined).
    pub fn tick(&mut self, dt: f64) -> NetworkTickResult {
        // Decay and expire signals first.
        let expired = self.field.tick(dt);

        let hashes: Vec<String> = self.field.signals.keys().cloned().collect();
        let mut newly_reached = 0;

        for hash in hashes {
            // Snapshot so we can consult node relay policies without holding a
            // borrow on the field while we later mutate it.
            let signal = match self.field.signals.get(&hash) {
                Some(s) => s.clone(),
                None => continue,
            };

            // Seed the frontier at the origin the first time we see this
            // signal. If the origin is not a node in this network (e.g. an
            // anonymous emit with no matching node), there is nothing to
            // diffuse from.
            let mut reached: Vec<NodeId> = signal.reached_nodes.clone();
            if reached.is_empty() {
                if self.nodes.contains_key(&signal.origin_node_id) {
                    reached.push(signal.origin_node_id.clone());
                } else {
                    continue;
                }
            }

            // Expand exactly one BFS layer: not-yet-reached neighbours of any
            // reached node whose target agrees to relay. We only expand while
            // the signal still has hop budget left.
            let mut frontier: Vec<NodeId> = Vec::new();
            if signal.hops < signal.radius {
                let reached_set: HashSet<&str> = reached.iter().map(|s| s.as_str()).collect();
                let remaining_hops = signal.radius - signal.hops;

                for node_id in &reached {
                    if let Some(hyphae) = self.hyphae.get(node_id) {
                        for hypha in hyphae {
                            if !hypha.active
                                || reached_set.contains(hypha.to.as_str())
                                || frontier.iter().any(|f| f == &hypha.to)
                            {
                                continue;
                            }
                            if let Some(target_node) = self.nodes.get(&hypha.to) {
                                let (should_relay, _dampening) =
                                    target_node.should_relay(&signal, remaining_hops);
                                if should_relay {
                                    frontier.push(hypha.to.clone());
                                }
                            }
                        }
                    }
                }
            }

            // Commit the (seed + newly reached) frontier back onto the signal.
            if let Some(s) = self.field.signals.get_mut(&hash) {
                for node_id in reached.iter().chain(frontier.iter()) {
                    s.mark_reached(node_id);
                }
                if !frontier.is_empty() {
                    s.hops += 1;
                }
            }

            // Record arrivals as node-level stats so spread is observable.
            for target in &frontier {
                if let Some(node) = self.nodes.get_mut(target) {
                    node.stats.signals_relayed += 1;
                    node.stats.signals_sensed += 1;
                }
            }

            newly_reached += frontier.len();
        }

        NetworkTickResult {
            expired_signals: expired,
            propagated_signals: newly_reached,
            active_signals: self.field.signals.len(),
        }
    }

    /// Get network statistics
    pub fn stats(&self) -> NetworkStats {
        let total_connections: usize = self.hyphae.values().map(|h| h.len()).sum();
        let node_degrees: Vec<usize> = self
            .nodes
            .keys()
            .map(|id| self.get_neighbors(id).len())
            .collect();

        let avg_degree = if node_degrees.is_empty() {
            0.0
        } else {
            node_degrees.iter().sum::<usize>() as f64 / node_degrees.len() as f64
        };

        NetworkStats {
            node_count: self.nodes.len(),
            connection_count: total_connections,
            avg_degree,
            field_stats: self.field.stats(),
        }
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a network tick
#[derive(Debug, Clone)]
pub struct NetworkTickResult {
    pub expired_signals: usize,
    pub propagated_signals: usize,
    pub active_signals: usize,
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub node_count: usize,
    pub connection_count: usize,
    pub avg_degree: f64,
    pub field_stats: crate::field::FieldStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = Network::with_topology(10, NetworkTopology::SmallWorld);

        assert_eq!(network.nodes.len(), 10);
        assert!(network.hyphae.values().any(|h| !h.is_empty()));
    }

    #[test]
    fn test_full_mesh() {
        let network = Network::with_topology(5, NetworkTopology::FullMesh);

        // Each node should connect to all others
        for (_id, connections) in &network.hyphae {
            assert_eq!(connections.len(), 4); // n-1 connections
        }
    }

    #[test]
    fn test_ring_topology() {
        let network = Network::with_topology(5, NetworkTopology::Ring);

        // Each node should have exactly 2 connections
        for connections in network.hyphae.values() {
            assert_eq!(connections.len(), 2);
        }
    }

    #[test]
    fn test_signal_diffusion_spreads_multi_hop() {
        use crate::{Signal, SignalType};

        // A 6-node ring guarantees a node three hops from the origin.
        let mut network = Network::with_topology(6, NetworkTopology::Ring);
        let origin = network.nodes.keys().next().unwrap().clone();

        // Strong signal + large radius so diffusion isn't cut short by decay
        // or the hop limit before it can traverse the ring.
        let signal = Signal::builder(SignalType::Data)
            .payload(b"spread me".to_vec())
            .intensity(1.0)
            .confidence(1.0)
            .ttl(10_000.0)
            .radius(50)
            .origin(&origin)
            .build();
        let hash = network.field.emit_anonymous(signal);

        // Before any tick the signal has not diffused anywhere.
        assert!(network
            .field
            .get_signal(&hash)
            .unwrap()
            .reached_nodes
            .is_empty());

        // Run until the signal covers the whole ring. Each active hypha relays
        // with probability ~0.5/tick, so non-coverage within 300 ticks is
        // astronomically unlikely (< 0.5^300). dt=0 keeps time-decay out of it.
        let mut prev_reach = 0;
        for _ in 0..300 {
            network.tick(0.0);
            let reach = network.field.get_signal(&hash).unwrap().reached_nodes.len();
            assert!(reach >= prev_reach, "reach must be monotonically non-decreasing");
            prev_reach = reach;
            if reach == network.nodes.len() {
                break;
            }
        }

        let sig = network.field.get_signal(&hash).unwrap();
        // The signal reached every node in the connected ring...
        assert_eq!(sig.reached_nodes.len(), network.nodes.len());
        assert!(sig.reached_nodes.contains(&origin));
        // ...which is only possible via multi-hop spreading: the origin has two
        // ring neighbours, so the far side is three hops away. The old code
        // never advanced past the origin's direct neighbours (hop 1).
        assert!(
            sig.hops >= 3,
            "far side of a 6-ring is 3 hops from origin, got {} hops",
            sig.hops
        );
    }

    #[test]
    fn test_diffusion_does_not_collapse_into_reinforcement() {
        use crate::{Signal, SignalType};

        // Regression guard for the original bug: propagated signals reused the
        // origin's content hash, so `emit_anonymous` merged them back into the
        // source instead of spreading. That produced phantom "reinforcements"
        // and zero real spread. Diffusion must grow reach, NOT reinforcement.
        let mut network = Network::with_topology(5, NetworkTopology::Ring);
        let origin = network.nodes.keys().next().unwrap().clone();

        let signal = Signal::builder(SignalType::Data)
            .payload(b"unique payload".to_vec())
            .intensity(1.0)
            .confidence(1.0)
            .ttl(10_000.0)
            .radius(50)
            .origin(&origin)
            .build();
        let hash = network.field.emit_anonymous(signal);

        for _ in 0..100 {
            network.tick(0.0);
        }

        let sig = network.field.get_signal(&hash).unwrap();
        // Reach grew (real diffusion)...
        assert!(sig.reached_nodes.len() > 1);
        // ...but the lone signal was never reinforced (nothing emitted the same
        // content), and it stayed a single field entry rather than fanning out
        // into collapsing copies.
        assert_eq!(sig.reinforcement_count, 0);
        assert_eq!(network.field.signals.len(), 1);
    }

    #[test]
    fn test_node_removal() {
        let mut network = Network::with_topology(5, NetworkTopology::FullMesh);
        let node_id = network.nodes.keys().next().unwrap().clone();

        network.remove_node(&node_id);

        assert_eq!(network.nodes.len(), 4);
        assert!(!network.hyphae.contains_key(&node_id));

        // No hyphae should point to removed node
        for connections in network.hyphae.values() {
            assert!(connections.iter().all(|h| h.to != node_id));
        }
    }
}
