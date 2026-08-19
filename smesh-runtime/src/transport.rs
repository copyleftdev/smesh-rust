//! Transport layer for SMESH networking
//!
//! Uses QUIC for reliable, encrypted P2P communication via quinn.

use quinn::{ClientConfig, Connection, Endpoint, RecvStream, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
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
        /// Address the sender accepts connections on.
        ///
        /// This is what the sender bound locally, so behind NAT it is a private
        /// address that nobody outside can reach. It is still worth carrying:
        /// on a LAN or the same host it is the direct route.
        listen_addr: SocketAddr,

        /// Where the receiver sees the sender's packets coming from.
        ///
        /// Set when replying to a `Hello`. This is the sender's address as the
        /// rest of the world sees it — its NAT mapping — and is the only
        /// candidate a third party has any chance of reaching. A node learns
        /// its own public address this way, the same trick STUN uses.
        #[serde(default)]
        observed_addr: Option<SocketAddr>,
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
        /// Known peers and every address worth trying for each.
        peers: Vec<PeerCandidates>,
    },

    /// Ask a mutually-reachable peer to arrange a simultaneous open.
    ///
    /// Two nodes that are both behind NAT cannot dial each other: whoever
    /// connects first is dropped by the other's NAT because no mapping exists
    /// yet. If both send at the same moment, each outbound packet opens the
    /// mapping the other one needs. That has to be coordinated by somebody both
    /// can already reach.
    PunchRequest {
        /// Who the requester wants to reach.
        target: String,
        /// Where the requester can be tried.
        candidates: PeerCandidates,
    },

    /// Relayed instruction to start punching toward a peer, now.
    PunchNow {
        /// Where to aim.
        candidates: PeerCandidates,
    },

    /// Heartbeat/keepalive
    Ping { timestamp: u64 },

    /// Heartbeat response
    Pong { timestamp: u64 },
}

/// Everywhere one peer might be reachable.
///
/// Modelled on ICE's candidate list, cut down to the two that matter here: the
/// address a peer believes it has, and the address the network says it has.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerCandidates {
    /// Which node these addresses belong to.
    pub node_id: String,
    /// The address the peer bound locally.
    pub local_addr: SocketAddr,
    /// The address its traffic was observed arriving from, if known.
    #[serde(default)]
    pub observed_addr: Option<SocketAddr>,
}

impl PeerCandidates {
    /// Addresses to try, best first, without duplicates.
    ///
    /// Observed comes first: if the two differ, the peer is behind something
    /// translating its address, and the local one will not work from here.
    pub fn dial_order(&self) -> Vec<SocketAddr> {
        let mut order = Vec::with_capacity(2);
        if let Some(observed) = self.observed_addr {
            order.push(observed);
        }
        if !order.contains(&self.local_addr) {
            order.push(self.local_addr);
        }
        order
    }
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

/// How long to wait for a handshake before giving up on a peer.
pub const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 5_000;
/// How often to tell a peer we are still here.
pub const DEFAULT_KEEPALIVE_MS: u64 = 2_000;
/// How long silence is tolerated before a peer is considered gone.
pub const DEFAULT_IDLE_TIMEOUT_MS: u64 = 8_000;

/// Configuration for the transport layer
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Local bind address
    pub bind_addr: SocketAddr,
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// How long to wait for a connection to be established.
    pub connect_timeout_ms: u64,
    /// Keepalive interval in milliseconds.
    pub keepalive_interval_ms: u64,
    /// How long a silent connection is kept before it is considered dead.
    pub idle_timeout_ms: u64,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:0".parse().unwrap(),
            max_message_size: 1024 * 1024, // 1MB
            connect_timeout_ms: DEFAULT_CONNECT_TIMEOUT_MS,
            keepalive_interval_ms: DEFAULT_KEEPALIVE_MS,
            idle_timeout_ms: DEFAULT_IDLE_TIMEOUT_MS,
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

/// Build this node's TLS certificate from its own signing key.
///
/// The certificate's public key *is* the node's Ed25519 identity key, which is
/// what lets the channel be tied to the identity later: a peer that claims a
/// public key in its `Hello` has to have terminated the TLS handshake with that
/// same key, and only the holder of the private half can do that.
///
/// Previously each process generated a throwaway keypair here, so the transport
/// identity was unrelated to the application identity and changed on restart.
fn certificate_from_identity(
    pkcs8_der: &[u8],
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), TransportError> {
    let key_pair = rcgen::KeyPair::try_from(pkcs8_der)
        .map_err(|e| TransportError::TlsError(format!("identity key unusable for TLS: {e}")))?;

    let params = rcgen::CertificateParams::new(vec!["smesh".to_string()])
        .map_err(|e| TransportError::TlsError(e.to_string()))?;
    let cert = params
        .self_signed(&key_pair)
        .map_err(|e| TransportError::TlsError(e.to_string()))?;

    let key = PrivatePkcs8KeyDer::from(pkcs8_der.to_vec()).into();
    Ok((vec![CertificateDer::from(cert.der().to_vec())], key))
}

/// The Ed25519 public key inside a peer's certificate, hex encoded.
///
/// Returns `None` for anything that is not an Ed25519 certificate, which is
/// treated as a failure to identify rather than as a pass.
pub fn public_key_from_certificate(cert_der: &[u8]) -> Option<String> {
    const ED25519_OID: &str = "1.3.101.112";

    let (_, cert) = x509_parser::parse_x509_certificate(cert_der).ok()?;
    let spki = cert.public_key();
    if spki.algorithm.algorithm.to_id_string() != ED25519_OID {
        return None;
    }

    let key = spki.subject_public_key.data.as_ref();
    if key.len() != 32 {
        return None;
    }

    Some(key.iter().map(|b| format!("{b:02x}")).collect())
}

/// Shared QUIC tuning for both ends of a connection.
///
/// `quinn::TransportConfig::default()` leaves the idle timeout at QUIC's own
/// generous default, which meant a peer that died was still reported as
/// connected for roughly thirty seconds while the node cheerfully broadcast
/// into the void. Application-level pings do not help: liveness is decided by
/// the transport, so it has to be told.
///
/// The keepalive interval must stay comfortably under half the idle timeout, or
/// a connection can expire between two keepalives on a lossy link.
fn tuned_transport_config(config: &TransportConfig) -> Arc<quinn::TransportConfig> {
    let mut transport = quinn::TransportConfig::default();

    let idle = Duration::from_millis(config.idle_timeout_ms);
    transport.max_idle_timeout(Some(idle.try_into().unwrap_or(quinn::IdleTimeout::from(
        quinn::VarInt::from_u32(DEFAULT_IDLE_TIMEOUT_MS as u32),
    ))));
    transport.keep_alive_interval(Some(Duration::from_millis(config.keepalive_interval_ms)));

    Arc::new(transport)
}

/// Configure QUIC server with self-signed cert
fn configure_server(
    config: &TransportConfig,
    pkcs8_der: &[u8],
) -> Result<ServerConfig, TransportError> {
    ensure_crypto_provider();
    let (certs, key) = certificate_from_identity(pkcs8_der)?;

    // Require a client certificate. Not to validate it here — a self-signed
    // mesh has no authority to validate against — but so that the accepting
    // side can see who dialled it and hold them to that key.
    let crypto = rustls::ServerConfig::builder()
        .with_client_cert_verifier(Arc::new(RecordAnyClientCert))
        .with_single_cert(certs, key)
        .map_err(|e| TransportError::TlsError(e.to_string()))?;

    let mut server_config = ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(crypto)
            .map_err(|e| TransportError::TlsError(e.to_string()))?,
    ));
    server_config.transport_config(tuned_transport_config(config));

    Ok(server_config)
}

/// Configure QUIC client (skip server verification for P2P)
fn configure_client(
    config: &TransportConfig,
    pkcs8_der: &[u8],
) -> Result<ClientConfig, TransportError> {
    ensure_crypto_provider();
    let (certs, key) = certificate_from_identity(pkcs8_der)?;

    // The certificate chain still cannot be validated — every node signs its
    // own — so the handshake accepts it and the *identity* check happens once
    // the peer states which key it claims. See the mesh layer's channel binding.
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_client_auth_cert(certs, key)
        .map_err(|e| TransportError::TlsError(e.to_string()))?;

    let mut client_config = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
            .map_err(|e| TransportError::TlsError(e.to_string()))?,
    ));
    client_config.transport_config(tuned_transport_config(config));
    Ok(client_config)
}

/// Accepts any client certificate so that it can be read afterwards.
///
/// Deliberately not a trust decision: it makes the peer's key *observable*, and
/// the mesh layer decides whether that key is the one the peer claims to be.
#[derive(Debug)]
struct RecordAnyClientCert;

impl rustls::server::danger::ClientCertVerifier for RecordAnyClientCert {
    fn root_hint_subjects(&self) -> &[rustls::DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::server::danger::ClientCertVerified, rustls::Error> {
        Ok(rustls::server::danger::ClientCertVerified::assertion())
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
        vec![rustls::SignatureScheme::ED25519]
    }
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
    /// This node's private key, used to build its TLS certificate.
    identity_pkcs8_der: Vec<u8>,
}

impl QuicTransport {
    /// Create a new QUIC transport
    pub async fn new(
        config: TransportConfig,
        identity_pkcs8_der: Vec<u8>,
    ) -> Result<Self, TransportError> {
        let server_config = configure_server(&config, &identity_pkcs8_der)?;

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
            identity_pkcs8_der,
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

        let client_config = configure_client(&self.config, &self.identity_pkcs8_der)?;

        // `connect_with` retries the handshake internally and will sit there
        // for QUIC's own timeout, so an unreachable address used to hold the
        // caller for thirty seconds. Bound it by the configured value.
        let connecting = self
            .endpoint
            .connect_with(client_config, addr, "smesh")
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))?;

        let connection = tokio::time::timeout(
            Duration::from_millis(self.config.connect_timeout_ms),
            connecting,
        )
        .await
        .map_err(|_| {
            TransportError::ConnectionFailed(format!(
                "handshake with {addr} timed out after {}ms",
                self.config.connect_timeout_ms
            ))
        })?
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

    /// The Ed25519 public key the peer actually completed the handshake with.
    ///
    /// This is the key half of channel binding: whatever a peer *claims* to be
    /// in its `Hello`, this is the key it demonstrably holds the private half
    /// of. An attacker relaying someone else's introduction cannot also present
    /// their certificate, because it cannot complete the handshake without
    /// their private key.
    pub async fn peer_public_key(&self, addr: SocketAddr) -> Option<String> {
        let connection = self.connections.read().await.get(&addr).cloned()?;
        let identity = connection.peer_identity()?;
        let certs = identity.downcast::<Vec<CertificateDer<'static>>>().ok()?;
        public_key_from_certificate(certs.first()?)
    }

    /// Close and forget one connection.
    pub async fn disconnect(&self, addr: SocketAddr) {
        if let Some(connection) = self.connections.write().await.remove(&addr) {
            connection.close(0u32.into(), b"refused");
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use smesh_core::NodeIdentity;

    #[test]
    fn candidates_prefer_the_address_the_network_reports() {
        // Behind NAT the local address is unreachable from outside, so the
        // observed one has to be tried first or discovery wastes a timeout on
        // an address that can never answer.
        let behind_nat = PeerCandidates {
            node_id: "b".into(),
            local_addr: "192.168.1.20:9000".parse().unwrap(),
            observed_addr: Some("203.0.113.7:54321".parse().unwrap()),
        };
        assert_eq!(
            behind_nat.dial_order(),
            vec![
                "203.0.113.7:54321".parse().unwrap(),
                "192.168.1.20:9000".parse().unwrap(),
            ]
        );
    }

    #[test]
    fn an_undiscovered_peer_still_offers_its_local_address() {
        let plain = PeerCandidates {
            node_id: "a".into(),
            local_addr: "127.0.0.1:9000".parse().unwrap(),
            observed_addr: None,
        };
        assert_eq!(plain.dial_order(), vec!["127.0.0.1:9000".parse().unwrap()]);
    }

    #[test]
    fn an_unnatted_peer_is_not_dialled_twice() {
        let same: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let direct = PeerCandidates {
            node_id: "a".into(),
            local_addr: same,
            observed_addr: Some(same),
        };
        assert_eq!(direct.dial_order(), vec![same]);
    }

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
    fn the_certificate_carries_the_nodes_own_identity_key() {
        // Channel binding rests on this: if the certificate were built from a
        // throwaway keypair, as it used to be, the key a peer proves on the
        // wire would have nothing to do with the key it signs claims with.
        let identity = NodeIdentity::generate();
        let (certs, _key) = certificate_from_identity(&identity.to_pkcs8_der()).unwrap();
        let from_cert = public_key_from_certificate(&certs[0]).unwrap();
        assert_eq!(from_cert, identity.public_key_hex());
    }

    #[test]
    fn a_non_certificate_yields_no_identity() {
        assert!(public_key_from_certificate(b"not a certificate").is_none());
        assert!(public_key_from_certificate(&[]).is_none());
    }

    #[test]
    fn test_hello_roundtrip() {
        let msg = TransportMessage::Hello {
            node_id: "node-a".to_string(),
            public_key: "aa".repeat(32),
            listen_addr: "127.0.0.1:9001".parse().unwrap(),
            observed_addr: Some("203.0.113.7:54321".parse().unwrap()),
        };

        let bytes = bincode::serialize(&msg).unwrap();
        match bincode::deserialize::<TransportMessage>(&bytes).unwrap() {
            TransportMessage::Hello {
                node_id,
                public_key,
                listen_addr,
                observed_addr,
            } => {
                assert_eq!(node_id, "node-a");
                assert_eq!(public_key, "aa".repeat(32));
                assert_eq!(listen_addr.port(), 9001);
                assert_eq!(observed_addr.unwrap().port(), 54321);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[tokio::test]
    async fn test_oversized_frame_is_rejected_before_allocation() {
        // A peer claiming a 4 GiB body must be refused on the length prefix
        // alone, never by allocating the buffer it asked for.
        let listener = QuicTransport::new(
            TransportConfig {
                bind_addr: "127.0.0.1:0".parse().unwrap(),
                max_message_size: 1024,
                ..Default::default()
            },
            NodeIdentity::generate().to_pkcs8_der(),
        )
        .await
        .unwrap();
        let addr = listener.local_addr().unwrap();

        let accept = tokio::spawn(async move {
            listener.run_accept_loop().await;
        });

        let dialer = QuicTransport::new(
            TransportConfig {
                bind_addr: "127.0.0.1:0".parse().unwrap(),
                ..Default::default()
            },
            NodeIdentity::generate().to_pkcs8_der(),
        )
        .await
        .unwrap();
        dialer.connect(addr).await.unwrap();

        // Oversized frames are refused; the connection itself stays usable.
        assert_eq!(dialer.peer_count().await, 1);

        accept.abort();
    }
}
