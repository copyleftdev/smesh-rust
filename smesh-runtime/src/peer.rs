//! Peer management for SMESH networking

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use smesh_core::NodeId;

/// Unique identifier for a peer
pub type PeerId = String;

/// Information about a connected peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peer {
    /// Peer identifier (maps to NodeId)
    pub id: PeerId,
    /// Network address
    pub addr: SocketAddr,
    /// Associated SMESH node
    pub node_id: NodeId,
    /// Connection state
    pub state: PeerState,
    /// Last seen timestamp (unix millis)
    pub last_seen: u64,
    /// Round-trip latency in milliseconds
    pub latency_ms: u64,
}

/// State of a peer connection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerState {
    /// Discovered but not connected
    Discovered,
    /// Connection in progress
    Connecting,
    /// Fully connected
    Connected,
    /// Disconnected (may reconnect)
    Disconnected,
    /// Banned (will not reconnect)
    Banned,
}

impl Peer {
    /// Create a new peer
    pub fn new(id: PeerId, addr: SocketAddr, node_id: NodeId) -> Self {
        Self {
            id,
            addr,
            node_id,
            state: PeerState::Discovered,
            last_seen: 0,
            latency_ms: 0,
        }
    }
    
    /// Check if peer is connected
    pub fn is_connected(&self) -> bool {
        self.state == PeerState::Connected
    }
    
    /// Update last seen timestamp
    pub fn touch(&mut self) {
        self.last_seen = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
    }
}

/// Manages peer connections
#[derive(Debug)]
pub struct PeerManager {
    /// Known peers
    peers: Arc<RwLock<HashMap<PeerId, Peer>>>,
    /// Maximum peers to maintain
    max_peers: usize,
    /// Our own peer ID
    #[allow(dead_code)]
    local_id: PeerId,
}

impl PeerManager {
    /// Create a new peer manager
    pub fn new(local_id: PeerId, max_peers: usize) -> Self {
        Self {
            peers: Arc::new(RwLock::new(HashMap::new())),
            max_peers,
            local_id,
        }
    }
    
    /// Add or update a peer
    pub async fn add_peer(&self, peer: Peer) -> bool {
        let mut peers = self.peers.write().await;
        
        if peers.len() >= self.max_peers && !peers.contains_key(&peer.id) {
            return false;
        }
        
        peers.insert(peer.id.clone(), peer);
        true
    }
    
    /// Remove a peer
    pub async fn remove_peer(&self, peer_id: &str) -> Option<Peer> {
        let mut peers = self.peers.write().await;
        peers.remove(peer_id)
    }
    
    /// Get a peer by ID
    pub async fn get_peer(&self, peer_id: &str) -> Option<Peer> {
        let peers = self.peers.read().await;
        peers.get(peer_id).cloned()
    }
    
    /// Get all connected peers
    pub async fn connected_peers(&self) -> Vec<Peer> {
        let peers = self.peers.read().await;
        peers.values()
            .filter(|p| p.is_connected())
            .cloned()
            .collect()
    }
    
    /// Update peer state
    pub async fn update_state(&self, peer_id: &str, state: PeerState) {
        let mut peers = self.peers.write().await;
        if let Some(peer) = peers.get_mut(peer_id) {
            peer.state = state;
            peer.touch();
        }
    }
    
    /// Get peer count
    pub async fn peer_count(&self) -> usize {
        let peers = self.peers.read().await;
        peers.len()
    }
    
    /// Get connected peer count
    pub async fn connected_count(&self) -> usize {
        let peers = self.peers.read().await;
        peers.values().filter(|p| p.is_connected()).count()
    }
    
    /// Ban a peer
    pub async fn ban_peer(&self, peer_id: &str) {
        self.update_state(peer_id, PeerState::Banned).await;
    }
    
    /// Prune stale peers (not seen in timeout_ms)
    pub async fn prune_stale(&self, timeout_ms: u64) -> usize {
        let mut peers = self.peers.write().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let stale: Vec<PeerId> = peers
            .iter()
            .filter(|(_, p)| {
                p.state != PeerState::Banned && 
                p.last_seen > 0 && 
                now - p.last_seen > timeout_ms
            })
            .map(|(id, _)| id.clone())
            .collect();
        
        for id in &stale {
            peers.remove(id);
        }
        
        stale.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_peer_manager() {
        let manager = PeerManager::new("local".to_string(), 10);
        
        let peer = Peer::new(
            "peer1".to_string(),
            "127.0.0.1:8000".parse().unwrap(),
            "node1".to_string(),
        );
        
        assert!(manager.add_peer(peer).await);
        assert_eq!(manager.peer_count().await, 1);
        
        manager.update_state("peer1", PeerState::Connected).await;
        assert_eq!(manager.connected_count().await, 1);
        
        manager.remove_peer("peer1").await;
        assert_eq!(manager.peer_count().await, 0);
    }
}
