//! # SMESH Refinery
//!
//! Tier 1/2 of the world-model pipeline (`WORLD-MODEL.md`): expert roles run
//! over LLM backends, a corpus goes in, a staged changeset comes out — ready
//! for human ratification. The refinery never signs anything.
//!
//! The confabulation firewall lives in [`extract`]: an extractor's emission
//! only becomes a candidate if its quote is found verbatim in the cited
//! document and `Citation::grounded` succeeds. Everything else is rejected
//! and counted, never silently dropped.

pub mod extract;
pub mod packet;
pub mod roster;
pub mod run;
pub mod verify;

use async_trait::async_trait;
use smesh_agent::OpenRouterClient;
use smesh_world::WorldError;

#[derive(Debug, thiserror::Error)]
pub enum RefineryError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("unparseable expert response: {0}")]
    Parse(String),
    #[error(transparent)]
    World(#[from] WorldError),
}

/// Minimal completion interface the refinery needs from any LLM provider.
/// Production uses OpenRouter; tests use scripted oracles.
#[async_trait]
pub trait Oracle: Send + Sync {
    async fn complete(
        &self,
        model: &str,
        system: &str,
        prompt: &str,
    ) -> Result<String, RefineryError>;
}

#[async_trait]
impl Oracle for OpenRouterClient {
    async fn complete(
        &self,
        model: &str,
        system: &str,
        prompt: &str,
    ) -> Result<String, RefineryError> {
        self.generate_with_model(model, prompt, Some(system))
            .await
            .map_err(|e| RefineryError::Backend(e.to_string()))
    }
}

/// Extract the first JSON value from an LLM response, tolerating code
/// fences and prose around it.
pub(crate) fn extract_json(text: &str) -> Result<serde_json::Value, RefineryError> {
    let start = text
        .find(['[', '{'])
        .ok_or_else(|| RefineryError::Parse(format!("no JSON in response: {text:.100}")))?;
    let candidate = &text[start..];
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in candidate.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '[' | '{' if !in_string => depth += 1,
            ']' | '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&candidate[..=i])
                        .map_err(|e| RefineryError::Parse(e.to_string()));
                }
            }
            _ => {}
        }
    }
    Err(RefineryError::Parse("unterminated JSON in response".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_tolerates_fences_and_prose() {
        let v = extract_json("Here you go:\n```json\n[{\"a\": 1}]\n```\nDone.").unwrap();
        assert_eq!(v[0]["a"], 1);
    }

    #[test]
    fn extract_json_handles_brackets_inside_strings() {
        let v = extract_json("{\"q\": \"a ] tricky [ one\"}").unwrap();
        assert_eq!(v["q"], "a ] tricky [ one");
    }

    #[test]
    fn extract_json_rejects_json_free_prose() {
        assert!(extract_json("I could not find any candidates.").is_err());
    }
}
