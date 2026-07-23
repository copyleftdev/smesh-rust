//! Reviewer identity: a persistent Ed25519 keypair. Ratification is a
//! signed act — "which human approved this and when" must verify offline.

use crate::RatifyError;
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use smesh_world::{RatificationRecord, ReviewerId, Signature};
use std::path::Path;

pub struct ReviewerKey {
    pub reviewer: ReviewerId,
    signing: SigningKey,
}

impl ReviewerKey {
    /// Load the keypair at `path`, or generate and persist one (0600).
    pub fn load_or_generate(path: &Path, reviewer: ReviewerId) -> Result<Self, RatifyError> {
        let signing = if path.exists() {
            let bytes = std::fs::read(path)?;
            let key: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
                RatifyError::Key(format!("{} is not a 32-byte seed", path.display()))
            })?;
            SigningKey::from_bytes(&key)
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let key = SigningKey::generate(&mut rand::rngs::OsRng);
            std::fs::write(path, key.to_bytes())?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
            }
            key
        };
        Ok(Self { reviewer, signing })
    }

    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing.verifying_key()
    }

    /// Canonical bytes the signature covers: base rev + reviewer + the full
    /// decision map, serialized deterministically (BTreeMap ordering).
    pub fn ratification_message(
        base_rev: &str,
        reviewer: &ReviewerId,
        decisions: &std::collections::BTreeMap<String, smesh_world::ReviewDecision>,
    ) -> Vec<u8> {
        let payload = serde_json::json!({
            "base_rev": base_rev,
            "reviewer": reviewer.0,
            "decisions": decisions,
        });
        serde_json::to_vec(&payload).expect("canonical ratification payload serializes")
    }

    /// Produce a signed `RatificationRecord` over the decision map.
    pub fn sign_ratification(
        &self,
        base_rev: &str,
        decisions: std::collections::BTreeMap<String, smesh_world::ReviewDecision>,
    ) -> RatificationRecord {
        let message = Self::ratification_message(base_rev, &self.reviewer, &decisions);
        let signature = self.signing.sign(&message);
        RatificationRecord {
            reviewer: self.reviewer.clone(),
            decisions,
            signature: Signature(signature.to_bytes().to_vec()),
        }
    }

    /// Offline verification: does this record carry a valid signature from
    /// `key` over `base_rev`?
    pub fn verify_ratification(
        key: &VerifyingKey,
        base_rev: &str,
        record: &RatificationRecord,
    ) -> bool {
        let message = Self::ratification_message(base_rev, &record.reviewer, &record.decisions);
        ed25519_dalek::Signature::from_slice(&record.signature.0)
            .map(|sig| key.verify(&message, &sig).is_ok())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smesh_world::ReviewDecision;
    use std::collections::BTreeMap;

    fn temp_key_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("smesh-ratify-test-{name}-{}", std::process::id()))
    }

    fn decisions() -> BTreeMap<String, ReviewDecision> {
        [("edge-a".to_owned(), ReviewDecision::Approve)]
            .into_iter()
            .collect()
    }

    #[test]
    fn keypair_persists_across_loads() {
        let path = temp_key_path("persist");
        let a = ReviewerKey::load_or_generate(&path, ReviewerId("dj".into())).unwrap();
        let b = ReviewerKey::load_or_generate(&path, ReviewerId("dj".into())).unwrap();
        assert_eq!(a.verifying_key(), b.verifying_key());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn signed_ratification_verifies_and_tampering_fails() {
        let path = temp_key_path("verify");
        let key = ReviewerKey::load_or_generate(&path, ReviewerId("dj".into())).unwrap();
        let record = key.sign_ratification("rev0", decisions());
        assert!(ReviewerKey::verify_ratification(
            &key.verifying_key(),
            "rev0",
            &record
        ));
        assert!(
            !ReviewerKey::verify_ratification(&key.verifying_key(), "rev1", &record),
            "signature must bind the base revision"
        );
        let mut tampered = record.clone();
        tampered
            .decisions
            .insert("edge-b".into(), ReviewDecision::Approve);
        assert!(!ReviewerKey::verify_ratification(
            &key.verifying_key(),
            "rev0",
            &tampered
        ));
        std::fs::remove_file(&path).unwrap();
    }
}
