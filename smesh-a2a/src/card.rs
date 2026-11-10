//! The mesh's Agent Card, and signing and verifying Agent Cards.
//!
//! To an A2A client the whole mesh is one agent. Its card advertises one skill
//! per agent in the mesh, described by that agent's own capability, so the
//! card says what this particular mesh can do in whatever domains its agents
//! cover. Any task can be sent; one no agent suits comes back `REJECTED`.
//!
//! # Signatures
//!
//! A2A v1.0 signs a card with a detached JWS over its RFC 8785 (JCS)
//! canonical form, with the `signatures` field removed. SMESH nodes already
//! hold Ed25519 keys, so cards are signed with `alg: EdDSA`, and the `kid` is
//! the public key itself (`ed25519:<base64url>`).
//!
//! A `kid` that carries its own key proves only that *someone* holding that key
//! signed the card. That is why [`verify_card`] takes the set of keys the
//! caller already trusts and refuses any other: checking a signature against
//! the key the card hands you would accept a card anyone made.

use a2a::{
    AgentCapabilities, AgentCard, AgentCardSignature, AgentInterface, AgentSkill,
    TRANSPORT_PROTOCOL_HTTP_JSON, TRANSPORT_PROTOCOL_JSONRPC,
};
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use base64::Engine;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde_json::Value;
use smesh_agent::AgentInfo;

/// Settings for the card the gateway serves.
#[derive(Debug, Clone)]
pub struct MeshCardConfig {
    /// Agent name shown to clients.
    pub name: String,
    /// What the mesh does.
    pub description: String,
    /// Version of this agent (not of the protocol).
    pub version: String,
    /// Base URL the gateway is reachable at, e.g. `http://127.0.0.1:7410`.
    pub base_url: String,
}

impl MeshCardConfig {
    /// Defaults for a gateway at `base_url`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            name: "SMESH mesh".into(),
            description: "A mesh of agents that settle who takes each task through \
                          decaying, reinforced signals rather than a central dispatcher. \
                          Each skill below is one agent. Send any task in plain language; \
                          the agents judge for themselves whether they suit it, and a task \
                          none of them suits is rejected."
                .into(),
            version: env!("CARGO_PKG_VERSION").into(),
            base_url: base_url.into(),
        }
    }
}

/// Build the card for a mesh whose agents are `roster`. Unsigned; see
/// [`sign_card`].
pub fn mesh_card(config: &MeshCardConfig, roster: &[AgentInfo]) -> AgentCard {
    let base = config.base_url.trim_end_matches('/').to_string();
    AgentCard {
        name: config.name.clone(),
        description: config.description.clone(),
        version: config.version.clone(),
        supported_interfaces: vec![
            AgentInterface::new(base.clone(), TRANSPORT_PROTOCOL_JSONRPC),
            AgentInterface::new(base, TRANSPORT_PROTOCOL_HTTP_JSON),
        ],
        capabilities: AgentCapabilities {
            streaming: Some(true),
            push_notifications: Some(false),
            extensions: None,
            extended_agent_card: Some(false),
        },
        default_input_modes: vec!["text/plain".into(), "application/json".into()],
        default_output_modes: vec!["text/plain".into()],
        skills: unique_ids(roster.iter().map(skill_for).collect()),
        provider: None,
        documentation_url: None,
        icon_url: None,
        security_schemes: None,
        security_requirements: None,
        signatures: None,
    }
}

/// Make skill IDs unique, suffixing a repeat with its position (and further
/// if that is taken too). Different names can still slug to the same ID, as
/// `a2a:Foo` and `a2a-Foo` do, and a client that keys skills by ID would see
/// only one of those agents.
fn unique_ids(mut skills: Vec<AgentSkill>) -> Vec<AgentSkill> {
    let mut seen = std::collections::HashSet::new();
    for (i, skill) in skills.iter_mut().enumerate() {
        let base = skill.id.clone();
        let mut n = i + 1;
        while !seen.insert(skill.id.clone()) {
            skill.id = format!("{base}-{n}");
            n += 1;
        }
    }
    skills
}

fn skill_for(agent: &AgentInfo) -> AgentSkill {
    let id: String = agent
        .name
        .to_lowercase()
        .chars()
        // Letters and digits in any script: an ASCII-only rule turned every
        // name written in, say, Japanese into the same run of dashes.
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    AgentSkill {
        id,
        name: agent.name.clone(),
        description: agent.capability.clone(),
        // Specialties as tags let a peer with no affinity source still match
        // this agent by built-in task type (see `smesh_a2a::remote`).
        tags: agent
            .specialties
            .iter()
            .cloned()
            .chain([agent.role.to_lowercase(), "smesh".into()])
            .collect(),
        examples: None,
        input_modes: None,
        output_modes: None,
        security_requirements: None,
    }
}

/// Why a card signature did not verify.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CardSignatureError {
    /// The card carries no signatures.
    #[error("card is unsigned")]
    Unsigned,
    /// A signature or header is not valid base64url / JSON / Ed25519.
    #[error("malformed signature: {0}")]
    Malformed(String),
    /// Signed with an algorithm other than EdDSA.
    #[error("unsupported algorithm {0:?}")]
    UnsupportedAlgorithm(String),
    /// No signature was made by a trusted key.
    #[error("no signature from a trusted key")]
    UntrustedKey,
    /// A trusted key's signature does not match the card.
    #[error("signature does not match the card")]
    BadSignature,
}

/// The `kid` used for an Ed25519 key.
pub fn key_id(key: &VerifyingKey) -> String {
    format!("ed25519:{}", B64.encode(key.as_bytes()))
}

/// Sign `card` with `key`, appending to any signatures it already has.
pub fn sign_card(card: &mut AgentCard, key: &SigningKey) {
    let header = serde_json::json!({
        "alg": "EdDSA",
        "kid": key_id(&key.verifying_key()),
        "typ": "JOSE",
    });
    let protected = B64.encode(canonical_json(&header));
    let signing_input = signing_input(&protected, card);
    let signature = key.sign(signing_input.as_bytes());
    card.signatures
        .get_or_insert_with(Vec::new)
        .push(AgentCardSignature {
            protected,
            signature: B64.encode(signature.to_bytes()),
            header: None,
        });
}

/// Verify that `card` carries a valid signature from one of `trusted`.
///
/// Returns the key that signed it. Signatures from keys not in `trusted` are
/// skipped, not accepted.
///
/// A card may carry several signatures, some made by other implementations
/// with other algorithms. One this function cannot use (malformed, or not
/// EdDSA) is skipped too, so it cannot veto a valid trusted signature beside
/// it. When nothing verifies, the error reported is the most telling one: a
/// trusted key's signature that does not match, then the first unusable
/// signature, then "no trusted key".
pub fn verify_card(
    card: &AgentCard,
    trusted: &[VerifyingKey],
) -> Result<VerifyingKey, CardSignatureError> {
    let signatures = card
        .signatures
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or(CardSignatureError::Unsigned)?;

    let mut bad_trusted = false;
    let mut first_unusable = None;
    for sig in signatures {
        match check_signature(sig, card, trusted) {
            Ok(Some(key)) => return Ok(key),
            Ok(None) => {}
            Err(CardSignatureError::BadSignature) => bad_trusted = true,
            Err(e) => {
                first_unusable.get_or_insert(e);
            }
        }
    }
    Err(if bad_trusted {
        CardSignatureError::BadSignature
    } else {
        first_unusable.unwrap_or(CardSignatureError::UntrustedKey)
    })
}

/// Check one signature. `Ok(None)` means it was made by a key not in
/// `trusted`.
fn check_signature(
    sig: &AgentCardSignature,
    card: &AgentCard,
    trusted: &[VerifyingKey],
) -> Result<Option<VerifyingKey>, CardSignatureError> {
    let header: Value = B64
        .decode(&sig.protected)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .ok_or_else(|| CardSignatureError::Malformed("protected header".into()))?;
    let alg = header.get("alg").and_then(Value::as_str).unwrap_or("");
    if alg != "EdDSA" {
        return Err(CardSignatureError::UnsupportedAlgorithm(alg.into()));
    }
    let kid = header.get("kid").and_then(Value::as_str).unwrap_or("");
    let Some(key) = trusted.iter().find(|k| key_id(k) == kid) else {
        return Ok(None);
    };

    let bytes = B64
        .decode(&sig.signature)
        .map_err(|e| CardSignatureError::Malformed(e.to_string()))?;
    let signature =
        Signature::from_slice(&bytes).map_err(|e| CardSignatureError::Malformed(e.to_string()))?;
    key.verify(signing_input(&sig.protected, card).as_bytes(), &signature)
        .map(|()| Some(*key))
        .map_err(|_| CardSignatureError::BadSignature)
}

/// JWS signing input for a detached payload: `protected "." payload`, where the
/// payload is the card's canonical form without `signatures`.
fn signing_input(protected: &str, card: &AgentCard) -> String {
    let mut value = serde_json::to_value(card).expect("AgentCard serializes");
    if let Value::Object(map) = &mut value {
        map.remove("signatures");
    }
    format!("{protected}.{}", B64.encode(canonical_json(&value)))
}

/// RFC 8785 canonical JSON: object keys sorted by UTF-16 code units, no
/// whitespace.
///
/// Numbers are written as serde_json writes them, which matches RFC 8785 for
/// integers and ordinary decimals but not for every float (`1e21`, say).
/// Agent Cards carry no floats, so this does not arise for them.
pub fn canonical_json(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
            out.push(b'{');
            for (i, (k, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend(serde_json::to_vec(k).expect("string serializes"));
                out.push(b':');
                write_canonical(v, out);
            }
            out.push(b'}');
        }
        Value::Array(items) => {
            out.push(b'[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(v, out);
            }
            out.push(b']');
        }
        scalar => out.extend(serde_json::to_vec(scalar).expect("scalar serializes")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    pub(crate) fn roster() -> Vec<AgentInfo> {
        vec![
            AgentInfo {
                id: "n1".into(),
                name: "Translator".into(),
                role: "General".into(),
                capability: "a translator between English and Portuguese".into(),
                specialties: vec![],
            },
            AgentInfo {
                id: "n2".into(),
                name: "Coder-2".into(),
                role: "Coder".into(),
                capability: "an expert software developer".into(),
                specialties: vec!["code_write".into(), "testing".into()],
            },
        ]
    }

    fn agent(name: &str) -> AgentInfo {
        AgentInfo {
            id: name.into(),
            name: name.into(),
            role: "General".into(),
            capability: "x".into(),
            specialties: vec![],
        }
    }

    #[test]
    fn test_skill_ids_are_unique_even_when_names_slug_alike() {
        // ":" and "-" both slug to "-"; "a2a-foo-2" is taken by a real name.
        let names = ["a2a:Foo", "a2a-Foo", "a2a-foo-2", "翻訳", "契約"];
        let roster: Vec<_> = names.iter().map(|n| agent(n)).collect();
        let card = mesh_card(&MeshCardConfig::new("http://x"), &roster);
        let ids: Vec<_> = card.skills.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["a2a-foo", "a2a-foo-2", "a2a-foo-2-3", "翻訳", "契約"]);
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len());
    }

    fn card() -> AgentCard {
        mesh_card(&MeshCardConfig::new("http://127.0.0.1:7410/"), &roster())
    }

    #[test]
    fn test_mesh_card_has_one_skill_per_agent_in_its_own_words() {
        let card = card();
        let skills: Vec<_> = card
            .skills
            .iter()
            .map(|s| (s.id.as_str(), s.description.as_str()))
            .collect();
        assert_eq!(
            skills,
            [
                ("translator", "a translator between English and Portuguese"),
                ("coder-2", "an expert software developer"),
            ]
        );
        assert_eq!(card.supported_interfaces[0].url, "http://127.0.0.1:7410");
        assert_eq!(card.supported_interfaces[0].protocol_version, "1.0");
    }

    #[test]
    fn test_mesh_card_round_trips_through_json() {
        let card = card();
        let json = serde_json::to_string(&card).unwrap();
        assert!(json.contains("\"supportedInterfaces\""));
        let back: AgentCard = serde_json::from_str(&json).unwrap();
        assert_eq!(back, card);
    }

    #[test]
    fn test_canonical_json_sorts_keys_and_drops_whitespace() {
        let v = serde_json::json!({"b": [2, {"z": 1, "a": "x"}], "a": true});
        assert_eq!(
            canonical_json(&v),
            br#"{"a":true,"b":[2,{"a":"x","z":1}]}"#.to_vec()
        );
    }

    #[test]
    fn test_signed_card_verifies_with_trusted_key() {
        let mut card = card();
        sign_card(&mut card, &key(1));
        let signer = verify_card(&card, &[key(1).verifying_key()]).unwrap();
        assert_eq!(signer, key(1).verifying_key());
    }

    #[test]
    fn test_signature_survives_json_round_trip() {
        let mut card = card();
        sign_card(&mut card, &key(1));
        let back: AgentCard = serde_json::from_str(&serde_json::to_string(&card).unwrap()).unwrap();
        assert!(verify_card(&back, &[key(1).verifying_key()]).is_ok());
    }

    #[test]
    fn test_tampered_card_fails() {
        let mut card = card();
        sign_card(&mut card, &key(1));
        card.supported_interfaces[0].url = "http://evil.example".into();
        assert_eq!(
            verify_card(&card, &[key(1).verifying_key()]),
            Err(CardSignatureError::BadSignature)
        );
    }

    #[test]
    fn test_card_signed_by_untrusted_key_is_refused() {
        let mut card = card();
        sign_card(&mut card, &key(2));
        assert_eq!(
            verify_card(&card, &[key(1).verifying_key()]),
            Err(CardSignatureError::UntrustedKey)
        );
    }

    #[test]
    fn test_unsigned_card_is_refused() {
        assert_eq!(
            verify_card(&card(), &[key(1).verifying_key()]),
            Err(CardSignatureError::Unsigned)
        );
    }

    /// A signature another implementation might add: a different algorithm.
    fn foreign_signature(alg: &str) -> AgentCardSignature {
        AgentCardSignature {
            protected: B64.encode(format!(r#"{{"alg":"{alg}","kid":"other"}}"#)),
            signature: B64.encode([0u8; 64]),
            header: None,
        }
    }

    fn garbage_signature() -> AgentCardSignature {
        AgentCardSignature {
            protected: "!!not base64!!".into(),
            signature: String::new(),
            header: None,
        }
    }

    #[test]
    fn test_foreign_algorithm_signature_does_not_veto_a_trusted_one() {
        let mut card = card();
        card.signatures = Some(vec![foreign_signature("ES256")]);
        sign_card(&mut card, &key(1));
        assert_eq!(
            verify_card(&card, &[key(1).verifying_key()]).unwrap(),
            key(1).verifying_key()
        );
    }

    #[test]
    fn test_malformed_signature_does_not_veto_a_trusted_one() {
        let mut card = card();
        card.signatures = Some(vec![garbage_signature()]);
        sign_card(&mut card, &key(1));
        assert!(verify_card(&card, &[key(1).verifying_key()]).is_ok());
    }

    #[test]
    fn test_only_unusable_signatures_report_the_first_problem() {
        let mut card = card();
        card.signatures = Some(vec![foreign_signature("ES256"), garbage_signature()]);
        assert_eq!(
            verify_card(&card, &[key(1).verifying_key()]),
            Err(CardSignatureError::UnsupportedAlgorithm("ES256".into()))
        );
    }

    #[test]
    fn test_bad_trusted_signature_outranks_unusable_ones() {
        let mut card = card();
        sign_card(&mut card, &key(1));
        card.signatures
            .as_mut()
            .unwrap()
            .push(foreign_signature("ES256"));
        card.name = "tampered".into();
        assert_eq!(
            verify_card(&card, &[key(1).verifying_key()]),
            Err(CardSignatureError::BadSignature)
        );
    }

    #[test]
    fn test_any_trusted_signature_suffices() {
        let mut card = card();
        sign_card(&mut card, &key(2));
        sign_card(&mut card, &key(1));
        assert_eq!(
            verify_card(&card, &[key(1).verifying_key()]).unwrap(),
            key(1).verifying_key()
        );
    }
}
