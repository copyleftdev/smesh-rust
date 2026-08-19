//! Cryptographic identity and verifiable attestation.
//!
//! Before this existed, a signal carried `origin_node_id` as a bare string and
//! the trust model gated relay probability on it. Anyone could claim to be
//! anyone, and — worse — a node could append arbitrary names to a signal's
//! reinforcement list, manufacturing corroboration for a claim nobody else had
//! ever seen. The protocol's central measurement, *how many independent parties
//! attest to this*, could be forged by a single participant.
//!
//! An [`Attestation`] is a signature over the claim being attested to, bound to
//! the attester's own name. It cannot be fabricated for a key you do not hold,
//! and it cannot be lifted off one claim and attached to another. Counting
//! attesters is therefore counting signatures.
//!
//! What this does *not* do on its own is stop someone picking a name that is
//! already taken. Signatures prove key ownership, not name ownership. The mesh
//! layer closes that by pinning a name to the key that first used it.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::NodeId;

/// Bytes an attestation signs over.
///
/// Binding the attester's name into the signed message is what stops an
/// attestation being replayed under a different name: the signature only
/// verifies for the exact pair it was produced for.
fn attestation_message(claim_hash: &str, node_id: &str) -> Vec<u8> {
    let mut message = Vec::with_capacity(claim_hash.len() + node_id.len() + 1);
    message.extend_from_slice(claim_hash.as_bytes());
    message.push(0x1f); // separator, so (a,bc) and (ab,c) cannot collide
    message.extend_from_slice(node_id.as_bytes());
    message
}

/// A node's private signing identity.
///
/// Deliberately not `Clone`, `Serialize` or `Debug`-revealing: the secret half
/// should not be duplicated casually, written to a journal, or sent anywhere.
pub struct NodeIdentity {
    signing_key: SigningKey,
    node_id: NodeId,
}

impl NodeIdentity {
    /// Generate a fresh identity with a random keypair.
    ///
    /// The node id is derived from the public key, so an identity generated
    /// this way is self-certifying: the name cannot be claimed by anyone who
    /// does not hold the key.
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let node_id = derive_node_id(&signing_key.verifying_key());
        Self {
            signing_key,
            node_id,
        }
    }

    /// Generate an identity that presents a chosen name.
    ///
    /// The keypair is still real and its signatures still verify, but the name
    /// is no longer derived from it, so it is only as trustworthy as whatever
    /// binds the name to the key — see the mesh layer's first-use pinning.
    /// Intended for readable node names in demos and tests.
    pub fn generate_named(node_id: impl Into<NodeId>) -> Self {
        Self {
            signing_key: SigningKey::generate(&mut OsRng),
            node_id: node_id.into(),
        }
    }

    /// This identity's node id.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// This identity's public key, hex encoded.
    pub fn public_key_hex(&self) -> String {
        hex(self.signing_key.verifying_key().as_bytes())
    }

    /// Attest to a claim, by its content hash.
    pub fn attest(&self, claim_hash: &str) -> Attestation {
        let message = attestation_message(claim_hash, &self.node_id);
        let signature: Signature = self.signing_key.sign(&message);
        Attestation {
            node_id: self.node_id.clone(),
            public_key: self.public_key_hex(),
            signature: hex(&signature.to_bytes()),
        }
    }
}

impl std::fmt::Debug for NodeIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeIdentity")
            .field("node_id", &self.node_id)
            .field("public_key", &self.public_key_hex())
            .finish_non_exhaustive()
    }
}

/// A signed statement that one node stands behind a claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attestation {
    /// Who is attesting.
    pub node_id: NodeId,
    /// Their public key, hex encoded.
    pub public_key: String,
    /// Signature over the claim hash bound to `node_id`, hex encoded.
    pub signature: String,
}

impl Attestation {
    /// Whether this attestation really covers `claim_hash`.
    ///
    /// Verifies the signature against the public key carried alongside it. That
    /// proves the holder of that key signed this exact claim under this exact
    /// name; it says nothing about whether the key is one you should trust.
    pub fn verify(&self, claim_hash: &str) -> bool {
        let (Some(key), Some(sig)) = (self.verifying_key(), self.signature_bytes()) else {
            return false;
        };
        key.verify(&attestation_message(claim_hash, &self.node_id), &sig)
            .is_ok()
    }

    /// Whether the node id is the one derived from this public key.
    ///
    /// True only for identities generated without a chosen name. A named
    /// identity fails this and must be bound to its key some other way.
    pub fn is_self_certifying(&self) -> bool {
        self.verifying_key()
            .map(|key| derive_node_id(&key) == self.node_id)
            .unwrap_or(false)
    }

    fn verifying_key(&self) -> Option<VerifyingKey> {
        let bytes: [u8; 32] = unhex(&self.public_key)?.try_into().ok()?;
        VerifyingKey::from_bytes(&bytes).ok()
    }

    fn signature_bytes(&self) -> Option<Signature> {
        let bytes: [u8; 64] = unhex(&self.signature)?.try_into().ok()?;
        Some(Signature::from_bytes(&bytes))
    }
}

/// Node id derived from a public key.
pub fn derive_node_id(key: &VerifyingKey) -> NodeId {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_attestation_verifies_for_the_claim_it_was_made_on() {
        let id = NodeIdentity::generate();
        let att = id.attest("abc123");
        assert!(att.verify("abc123"));
        assert!(att.is_self_certifying());
    }

    #[test]
    fn an_attestation_cannot_be_moved_to_another_claim() {
        // The whole point: agreement on one claim must not become agreement on
        // a different one.
        let id = NodeIdentity::generate();
        let att = id.attest("claim-one");
        assert!(!att.verify("claim-two"));
    }

    #[test]
    fn an_attestation_cannot_be_reused_under_another_name() {
        let id = NodeIdentity::generate_named("latency");
        let mut att = id.attest("abc123");
        assert!(att.verify("abc123"));

        att.node_id = "errors".to_string();
        assert!(
            !att.verify("abc123"),
            "renaming must invalidate the signature"
        );
    }

    #[test]
    fn attestations_cannot_be_forged_for_someone_elses_key() {
        let honest = NodeIdentity::generate_named("latency");
        let attacker = NodeIdentity::generate_named("errors");

        // The attacker signs, then swaps in the honest node's public key to
        // pass the signature off as theirs.
        let mut forged = attacker.attest("abc123");
        forged.node_id = "latency".to_string();
        forged.public_key = honest.public_key_hex();

        assert!(!forged.verify("abc123"));
    }

    #[test]
    fn a_named_identity_is_not_self_certifying() {
        let id = NodeIdentity::generate_named("latency");
        let att = id.attest("abc123");
        assert!(att.verify("abc123"), "the signature is still real");
        assert!(
            !att.is_self_certifying(),
            "but the name is not backed by the key"
        );
    }

    #[test]
    fn malformed_attestations_are_rejected_rather_than_panicking() {
        let att = Attestation {
            node_id: "latency".to_string(),
            public_key: "not hex".to_string(),
            signature: "also not hex".to_string(),
        };
        assert!(!att.verify("abc123"));
        assert!(!att.is_self_certifying());
    }

    #[test]
    fn two_identities_do_not_collide() {
        let a = NodeIdentity::generate();
        let b = NodeIdentity::generate();
        assert_ne!(a.node_id(), b.node_id());
        assert_ne!(a.public_key_hex(), b.public_key_hex());
    }
}
