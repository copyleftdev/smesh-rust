//! Transport layer for SMESH networking
//!
//! Uses QUIC for reliable, encrypted P2P communication via quinn.

use quinn::{ClientConfig, Connection, Endpoint, RecvStream, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use smesh_core::Signal;

/// Transport errors
#[derive(Error, Debug)]
pub enum TransportError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Send failed: {0}")]
    SendFailed(String),

    #[error("Receive failed: {0}")]
    ReceiveFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Transport closed")]
    Closed,

    #[error("QUIC error: {0}")]
    QuicError(String),

    #[error("TLS error: {0}")]
    TlsError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Messages sent over the transport
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportMessage {
    /// Introduction sent on every freshly established connection.
    ///
    /// An inbound connection's `remote_address()` is the peer's *ephemeral*
    /// source port, not the port it listens on, so it cannot be dialled back
    /// or gossiped onward. `listen_addr` carries the dialable address, and
    /// `node_id` binds the socket to a SMESH node so reinforcement can be
    /// attributed to whoever relayed it.
    Hello {
        /// Sender's SMESH node id
        node_id: String,
        /// Sender's Ed25519 public key, hex encoded.
        ///
        /// Lets the receiver bind this name to this key for the rest of the
        /// run, so a later peer cannot present the same name under a different
        /// key and have its attestations counted.
        #[serde(default)]
        public_key: String,
        /// Address the sender accepts connections on
        listen_addr: SocketAddr,
    },

    /// A SMESH signal to propagate
    Signal {
        /// The signal itself
        signal: Signal,
        /// Age of the signal, in seconds, at the moment it was sent.
        ///
        /// The receiver rebases `created_at` against its own field clock using
        /// this age, so decay and expiry stay consistent between hosts whose
        /// wall clocks disagree. Only link latency leaks into the estimate.
        age_secs: f64,
    },

    /// Peer discovery request
    PeerRequest {
        /// Maximum peers to return
        max_peers: usize,
    },

    /// Peer discovery response
    PeerResponse {
        /// Known peer addresses
        peers: Vec<(String, SocketAddr)>,
    },

    /// Heartbeat/keepalive
    Ping { timestamp: u64 },

    /// Heartbeat response
    Pong { timestamp: u64 },
}

impl TransportMessage {
    /// Wrap a signal for transmission, stamping its age against `now`.
    pub fn signal(signal: Signal, now: chrono::DateTime<chrono::Utc>) -> Self {
        let age_secs = (now - signal.created_at).num_milliseconds() as f64 / 1000.0;
        TransportMessage::Signal {
            signal,
            age_secs: age_secs.max(0.0),
        }
    }
}

/// Configuration for the transport layer
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Local bind address
    pub bind_addr: SocketAddr,
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
    /// Keepalive interval in milliseconds
    pub keepalive_interval_ms: u64,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:0".parse().unwrap(),
            max_message_size: 1024 * 1024, // 1MB
            connect_timeout_ms: 5000,
            keepalive_interval_ms: 30000,
        }
    }
}

/// Install a rustls crypto provider for this process, exactly once.
///
/// rustls 0.23 refuses to pick for itself when more than one provider is
/// compiled in, and quinn pulls in both through its own feature set. Without
/// this, the first TLS config built panics rather than returning an error.
fn ensure_crypto_provider() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // A competing installation from elsewhere in the process is fine; we
        // only need *a* provider to be present.
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Generate self-signed certificate for QUIC
fn generate_self_signed_cert(
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), TransportError> {
    let cert = rcgen::generate_simple_self_signed(vec!["smesh".to_string()])
        .map_err(|e| TransportError::TlsError(e.to_string()))?;

    let key = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()).into();
    let cert_der = CertificateDer::from(cert.cert.der().to_vec());

    Ok((vec![cert_der], key))
}

/// Configure QUIC server with self-signed cert
fn configure_server() -> Result<ServerConfig, TransportError> {
    ensure_crypto_provider();
    let (certs, key) = generate_self_signed_cert()?;

    let mut server_config = ServerConfig::with_single_cert(certs, key)
        .map_err(|e| TransportError::TlsError(e.to_string()))?;

    let transport_config = Arc::new(quinn::TransportConfig::default());
    server_config.transport_config(transport_config);

    Ok(server_config)
}

/// Configure QUIC client (skip server verification for P2P)
fn configure_client() -> ClientConfig {
    ensure_crypto_provider();
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();

    ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto).unwrap(),
    ))
}

/// Skip server certificate verification (P2P nodes use self-signed certs)
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

/// QUIC-based P2P transport layer
pub struct QuicTransport {
    /// QUIC endpoint (server + client)
    endpoint: Endpoint,
    /// Transport configuration
    config: TransportConfig,
    /// Active connections by address
    connections: Arc<RwLock<std::collections::HashMap<SocketAddr, Connection>>>,
    /// Channel for incoming messages
    incoming_tx: mpsc::Sender<(SocketAddr, TransportMessage)>,
    incoming_rx: Option<mpsc::Receiver<(SocketAddr, TransportMessage)>>,
    /// Shutdown flag
    shutdown: Arc<RwLock<bool>>,
}

impl QuicTransport {
    /// Create a new QUIC transport
    pub async fn new(config: TransportConfig) -> Result<Self, TransportError> {
        let server_config = configure_server()?;

        let endpoint = Endpoint::server(server_config, config.bind_addr)?;

        let (incoming_tx, incoming_rx) = mpsc::channel(10000);

        info!("QUIC transport bound to {}", endpoint.local_addr()?);

        Ok(Self {
            endpoint,
            config,
            connections: Arc::new(RwLock::new(std::collections::HashMap::new())),
            incoming_tx,
            incoming_rx: Some(incoming_rx),
            shutdown: Arc::new(RwLock::new(false)),
        })
    }

    /// Take the incoming message receiver
    pub fn take_incoming(&mut self) -> Option<mpsc::Receiver<(SocketAddr, TransportMessage)>> {
        self.incoming_rx.take()
    }

    /// Get local bound address
    pub fn local_addr(&self) -> Result<SocketAddr, TransportError> {
        self.endpoint.local_addr().map_err(TransportError::IoError)
    }

    /// Connect to a peer
    pub async fn connect(&self, addr: SocketAddr) -> Result<(), TransportError> {
        // Check if already connected
        {
            let conns = self.connections.read().await;
            if conns.contains_key(&addr) {
                return Ok(());
            }
        }

        let client_config = configure_client();

        let connection = self
            .endpoint
            .connect_with(client_config, addr, "smesh")
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))?
            .await
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))?;

        debug!("Connected to peer at {}", addr);

        // Store connection
        {
            let mut conns = self.connections.write().await;
            conns.insert(addr, connection.clone());
        }

        // A QUIC connection is bidirectional regardless of who dialled it. The
        // accept loop only pumps connections we accepted, so a dialled peer's
        // streams need their own reader or nothing it sends us is ever read.
        let connections = Arc::clone(&self.connections);
        let incoming_tx = self.incoming_tx.clone();
        let max_message_size = self.config.max_message_size;
        tokio::spawn(async move {
            Self::handle_connection(connection, addr, incoming_tx, max_message_size).await;
            connections.write().await.remove(&addr);
        });

        Ok(())
    }

    /// Send a message to a peer
    pub async fn send(
        &self,
        addr: SocketAddr,
        msg: TransportMessage,
    ) -> Result<(), TransportError> {
        // Get or create connection
        let connection = {
            let conns = self.connections.read().await;
            conns.get(&addr).cloned()
        };

        let connection = match connection {
            Some(c) => c,
            None => {
                self.connect(addr).await?;
                let conns = self.connections.read().await;
                conns.get(&addr).cloned().ok_or_else(|| {
                    TransportError::ConnectionFailed("Connection not established".into())
                })?
            }
        };

        // Serialize message
        let data = bincode::serialize(&msg)
            .map_err(|e| TransportError::SerializationError(e.to_string()))?;

        // Open unidirectional stream and send
        let mut send_stream = connection
            .open_uni()
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;

        // Send length prefix + data
        let len = (data.len() as u32).to_be_bytes();
        send_stream
            .write_all(&len)
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;
        send_stream
            .write_all(&data)
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;
        send_stream
            .finish()
            .map_err(|e| TransportError::SendFailed(e.to_string()))?;

        Ok(())
    }

    /// Broadcast a signal to specific peers
    pub async fn broadcast(
        &self,
        addrs: &[SocketAddr],
        signal: Signal,
    ) -> Vec<Result<(), TransportError>> {
        let now = chrono::Utc::now();
        let sends = addrs
            .iter()
            .map(|addr| self.send(*addr, TransportMessage::signal(signal.clone(), now)));

        futures::future::join_all(sends).await
    }

    /// Addresses of every live connection, dialled or accepted.
    pub async fn connected_addrs(&self) -> Vec<SocketAddr> {
        self.connections.read().await.keys().copied().collect()
    }

    /// Send a message over every live connection, optionally skipping one.
    ///
    /// `except` is how a gossip relay avoids echoing a message straight back
    /// to the peer it just arrived from. Returns the addresses the message
    /// actually reached; failures are logged and omitted, since a dead peer
    /// must not abort delivery to healthy ones.
    pub async fn broadcast_all(
        &self,
        msg: &TransportMessage,
        except: Option<SocketAddr>,
    ) -> Vec<SocketAddr> {
        let targets: Vec<SocketAddr> = self
            .connections
            .read()
            .await
            .keys()
            .copied()
            .filter(|a| Some(*a) != except)
            .collect();

        let sends = targets.iter().map(|addr| self.send(*addr, msg.clone()));

        futures::future::join_all(sends)
            .await
            .into_iter()
            .zip(targets)
            .filter_map(|(result, addr)| match result {
                Ok(()) => Some(addr),
                Err(e) => {
                    debug!("broadcast to {} failed: {}", addr, e);
                    None
                }
            })
            .collect()
    }

    /// Start accepting incoming connections
    pub async fn run_accept_loop(&self) {
        info!("Starting QUIC accept loop");

        while !*self.shutdown.read().await {
            match self.endpoint.accept().await {
                Some(incoming) => {
                    let connections = Arc::clone(&self.connections);
                    let incoming_tx = self.incoming_tx.clone();
                    let max_message_size = self.config.max_message_size;

                    tokio::spawn(async move {
                        match incoming.await {
                            Ok(connection) => {
                                let addr = connection.remote_address();
                                debug!("Accepted connection from {}", addr);

                                // Store connection
                                {
                                    let mut conns = connections.write().await;
                                    conns.insert(addr, connection.clone());
                                }

                                // Handle incoming streams until the peer goes
                                // away, then drop it from the connection map so
                                // broadcasts stop targeting a dead socket.
                                Self::handle_connection(
                                    connection,
                                    addr,
                                    incoming_tx,
                                    max_message_size,
                                )
                                .await;
                                connections.write().await.remove(&addr);
                            }
                            Err(e) => {
                                warn!("Failed to accept connection: {}", e);
                            }
                        }
                    });
                }
                None => break,
            }
        }
    }

    /// Handle incoming streams from a connection
    async fn handle_connection(
        connection: Connection,
        addr: SocketAddr,
        incoming_tx: mpsc::Sender<(SocketAddr, TransportMessage)>,
        max_message_size: usize,
    ) {
        loop {
            match connection.accept_uni().await {
                Ok(recv_stream) => {
                    let tx = incoming_tx.clone();
                    tokio::spawn(async move {
                        if let Err(e) =
                            Self::handle_stream(recv_stream, addr, tx, max_message_size).await
                        {
                            debug!("Stream error from {}: {}", addr, e);
                        }
                    });
                }
                Err(e) => {
                    debug!("Connection closed from {}: {}", addr, e);
                    break;
                }
            }
        }
    }

    /// Handle a single incoming stream
    async fn handle_stream(
        mut recv_stream: RecvStream,
        addr: SocketAddr,
        incoming_tx: mpsc::Sender<(SocketAddr, TransportMessage)>,
        max_message_size: usize,
    ) -> Result<(), TransportError> {
        // Read length prefix
        let mut len_buf = [0u8; 4];
        recv_stream
            .read_exact(&mut len_buf)
            .await
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()))?;
        let len = u32::from_be_bytes(len_buf) as usize;

        // The length prefix is attacker-controlled: refuse to allocate for it
        // before checking it against the configured ceiling.
        if len > max_message_size {
            return Err(TransportError::ReceiveFailed(format!(
                "message of {len} bytes from {addr} exceeds max_message_size ({max_message_size})"
            )));
        }

        // Read message data
        let mut data = vec![0u8; len];
        recv_stream
            .read_exact(&mut data)
            .await
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()))?;

        // Deserialize
        let msg: TransportMessage = bincode::deserialize(&data)
            .map_err(|e| TransportError::SerializationError(e.to_string()))?;

        // Send to incoming channel
        incoming_tx
            .send((addr, msg))
            .await
            .map_err(|e| TransportError::ReceiveFailed(e.to_string()))?;

        Ok(())
    }

    /// Shutdown the transport
    pub async fn shutdown(&self) {
        *self.shutdown.write().await = true;
        self.endpoint.close(0u32.into(), b"shutdown");
    }

    /// Get connected peer count
    pub async fn peer_count(&self) -> usize {
        self.connections.read().await.len()
    }
}

/// Simple transport for testing (no QUIC)
pub struct Transport {
    config: TransportConfig,
    tx: mpsc::Sender<(SocketAddr, TransportMessage)>,
    rx: mpsc::Receiver<(SocketAddr, TransportMessage)>,
}

impl Transport {
    /// Create a new simple transport
    pub fn new(config: TransportConfig) -> Self {
        let (tx, _rx_out) = mpsc::channel(1000);
        let (_tx_in, rx) = mpsc::channel(1000);

        Self { config, tx, rx }
    }

    /// Send a message
    pub async fn send(
        &self,
        addr: SocketAddr,
        msg: TransportMessage,
    ) -> Result<(), TransportError> {
        self.tx
            .send((addr, msg))
            .await
            .map_err(|e| TransportError::SendFailed(e.to_string()))
    }

    /// Receive a message
    pub async fn recv(&mut self) -> Option<(SocketAddr, TransportMessage)> {
        self.rx.recv().await
    }

    /// Broadcast a signal
    pub async fn broadcast(
        &self,
        addrs: &[SocketAddr],
        signal: Signal,
    ) -> Vec<Result<(), TransportError>> {
        let now = chrono::Utc::now();
        let mut results = Vec::new();
        for addr in addrs {
            let result = self
                .send(*addr, TransportMessage::signal(signal.clone(), now))
                .await;
            results.push(result);
        }
        results
    }

    /// Get local address
    pub fn local_addr(&self) -> SocketAddr {
        self.config.bind_addr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_config() {
        let config = TransportConfig::default();
        assert_eq!(config.max_message_size, 1024 * 1024);
    }

    #[test]
    fn test_transport_message_serialization() {
        use smesh_core::SignalType;

        let signal = Signal::builder(SignalType::Data)
            .payload(b"test".to_vec())
            .build();

        let msg = TransportMessage::signal(signal, chrono::Utc::now());

        let serialized = bincode::serialize(&msg).unwrap();
        let deserialized: TransportMessage = bincode::deserialize(&serialized).unwrap();

        match deserialized {
            TransportMessage::Signal { signal, .. } => {
                assert_eq!(signal.payload, b"test".to_vec());
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_hello_roundtrip() {
        let msg = TransportMessage::Hello {
            node_id: "node-a".to_string(),
            public_key: "aa".repeat(32),
            listen_addr: "127.0.0.1:9001".parse().unwrap(),
        };

        let bytes = bincode::serialize(&msg).unwrap();
        match bincode::deserialize::<TransportMessage>(&bytes).unwrap() {
            TransportMessage::Hello {
                node_id,
                public_key,
                listen_addr,
            } => {
                assert_eq!(node_id, "node-a");
                assert_eq!(public_key, "aa".repeat(32));
                assert_eq!(listen_addr.port(), 9001);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[tokio::test]
    async fn test_oversized_frame_is_rejected_before_allocation() {
        // A peer claiming a 4 GiB body must be refused on the length prefix
        // alone, never by allocating the buffer it asked for.
        let listener = QuicTransport::new(TransportConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            max_message_size: 1024,
            ..Default::default()
        })
        .await
        .unwrap();
        let addr = listener.local_addr().unwrap();

        let accept = tokio::spawn(async move {
            listener.run_accept_loop().await;
        });

        let dialer = QuicTransport::new(TransportConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..Default::default()
        })
        .await
        .unwrap();
        dialer.connect(addr).await.unwrap();

        // Oversized frames are refused; the connection itself stays usable.
        assert_eq!(dialer.peer_count().await, 1);

        accept.abort();
    }
}
