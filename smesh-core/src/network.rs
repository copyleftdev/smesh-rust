//! Network topology - how nodes are connected
//!
//! Defines network structures and routing through hyphae.

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use rand::Rng;

use crate::{Node, NodeId, Field, Signal};

/// Network topology types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkTopology {
    /// Fully connected mesh
    FullMesh,
    /// Random graph with given edge probability
    Random,
    /// Small world network (Watts-Strogatz)
    SmallWorld,
    /// Scale-free network (Barabasi-Albert)
    ScaleFree,
    /// Ring topology
    Ring,
    /// Grid topology
    Grid,
}

impl Default for NetworkTopology {
    fn default() -> Self {
        Self::SmallWorld
    }
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
            self.hyphae
                .entry(from.to_string())
                .or_default()
                .push(hypha);
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
                    for j in 1..=k/2 {
                        let neighbor = (i + j) % n;
                        self.connect_bidirectional(&node_ids[i], &node_ids[neighbor]);
                    }
                }
                
                // Rewire edges
                for i in 0..n {
                    for j in 1..=k/2 {
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
                        let target = (0..i)
                            .find(|&j| {
                                !connected.contains(&j) &&
                                rng.gen::<f64>() < degrees[j] as f64 / total_degree.max(1) as f64
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
    
    /// Run one simulation tick
    pub fn tick(&mut self, dt: f64) -> NetworkTickResult {
        // Update field
        let expired = self.field.tick(dt);
        
        // Propagate signals through hyphae
        let mut propagated = 0;
        let signals: Vec<Signal> = self.field.signals.values().cloned().collect();
        
        for signal in signals {
            if signal.hops < signal.radius {
                // Get neighbors of origin node
                if let Some(neighbors) = self.hyphae.get(&signal.origin_node_id) {
                    for hypha in neighbors {
                        if hypha.active {
                            if let Some(target_node) = self.nodes.get(&hypha.to) {
                                let (should_relay, dampening) = target_node.should_relay(
                                    &signal,
                                    signal.radius - signal.hops,
                                );
                                
                                if should_relay {
                                    let propagated_signal = signal.propagate(
                                        dampening * hypha.strength
                                    );
                                    self.field.emit_anonymous(propagated_signal);
                                    propagated += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        NetworkTickResult {
            expired_signals: expired,
            propagated_signals: propagated,
            active_signals: self.field.signals.len(),
        }
    }
    
    /// Get network statistics
    pub fn stats(&self) -> NetworkStats {
        let total_connections: usize = self.hyphae.values().map(|h| h.len()).sum();
        let node_degrees: Vec<usize> = self.nodes
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
