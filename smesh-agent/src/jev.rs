//! Client for TypeSafe's System One API (Jev), and an affinity source on it.
//!
//! Jev is a decision model: it does not write text, it answers typed
//! questions about a `state` with calibrated probabilities, in roughly
//! 100-150 ms. SMESH uses it where it needs a fast judgement from plain
//! language, chiefly whether an agent suits a task ([`JevAffinity`]).
//!
//! API: `POST {base}/v1/systemone` with `{model, state, questions}` and a
//! bearer key; answers come back keyed by question name. See
//! <https://docs.typesafe.ai/api/system-one>.
//!
//! Credentials resolve like OpenRouter's: `TYPESAFE_API_KEY` from the
//! environment, else from `$TYPESAFE_ENV_FILE`, else `~/.creds/typesafe.env`.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use thiserror::Error;
use tracing::{debug, warn};

use crate::affinity::AffinitySource;
use crate::coordinator::TaskDefinition;
use crate::openrouter::read_key_from_file;

/// Production API base.
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
/// Model alias for the latest stable Jev.
pub const DEFAULT_MODEL: &str = "jev-latest";

/// Settings for a [`JevClient`].
#[derive(Debug, Clone)]
pub struct JevConfig {
    /// Bearer key.
    pub api_key: String,
    /// API base URL, without a trailing slash.
    pub base_url: String,
    /// Model name or alias.
    pub model: String,
    /// Per-request timeout.
    pub timeout: Duration,
}

impl JevConfig {
    /// A config for `api_key` with the default base URL, model and timeout.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            timeout: Duration::from_secs(10),
        }
    }

    /// Resolve from the environment and credentials files. `None` without a key.
    ///
    /// `TYPESAFE_BASE_URL` and `TYPESAFE_MODEL` override the defaults.
    pub fn from_env() -> Option<Self> {
        let key = resolve("TYPESAFE_API_KEY")?;
        let mut config = Self::new(key);
        if let Some(base) = resolve("TYPESAFE_BASE_URL") {
            config.base_url = base.trim_end_matches('/').to_string();
        }
        if let Some(model) = resolve("TYPESAFE_MODEL") {
            config.model = model;
        }
        Some(config)
    }
}

fn resolve(key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(key) {
        if !v.is_empty() {
            return Some(v);
        }
    }
    let mut files = Vec::new();
    if let Ok(explicit) = std::env::var("TYPESAFE_ENV_FILE") {
        files.push(std::path::PathBuf::from(explicit));
    }
    if let Some(home) = std::env::var_os("HOME") {
        files.push(std::path::Path::new(&home).join(".creds/typesafe.env"));
    }
    files
        .iter()
        .find_map(|f| read_key_from_file(f, key))
        .filter(|v| !v.is_empty())
}

/// Errors from the System One API.
#[derive(Debug, Error)]
pub enum JevError {
    /// The request did not complete.
    #[error("request failed: {0}")]
    Http(String),
    /// The API answered with an error status.
    #[error("HTTP {status}: {body}")]
    Status {
        /// HTTP status code.
        status: u16,
        /// Response body, truncated.
        body: String,
    },
    /// The response was not the documented shape.
    #[error("unexpected response: {0}")]
    Parse(String),
}

/// One answer, by question type. Box answers and any future types are
/// accepted and ignored.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// Yes/no: the probability of yes.
    Noul {
        /// Probability of yes, 0 to 1.
        noul: f64,
    },
    /// One option from a set.
    Choice {
        /// The most probable option.
        choice: String,
        /// How concentrated the distribution is, 0 to 1.
        confidence: f64,
        /// Probability of each option.
        probabilities: HashMap<String, f64>,
    },
    /// A level on an ordered rubric.
    Score {
        /// Probability-weighted level, from 0.
        score: f64,
        /// How concentrated the distribution is, 0 to 1.
        confidence: f64,
        /// Probability of each level.
        probabilities: HashMap<String, f64>,
    },
    /// A question type this client does not model.
    #[serde(other)]
    Other,
}

/// Token counts for one request.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct Usage {
    /// Billable input tokens.
    pub input_tokens: u64,
    /// Output tokens (currently free).
    pub output_tokens: u64,
}

/// A System One response.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SystemOneResponse {
    /// The model that answered (an alias resolves to a version).
    pub model: String,
    /// Answers keyed by question name.
    pub answers: HashMap<String, Answer>,
    /// Token usage.
    pub usage: Usage,
}

impl SystemOneResponse {
    /// The probability of yes for Noul question `name`.
    pub fn noul(&self, name: &str) -> Option<f64> {
        match self.answers.get(name)? {
            Answer::Noul { noul } => Some(*noul),
            _ => None,
        }
    }
}

/// A System One API client.
#[derive(Clone)]
pub struct JevClient {
    http: reqwest::Client,
    config: JevConfig,
}

impl JevClient {
    /// A client with `config`.
    pub fn new(config: JevConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_default();
        Self { http, config }
    }

    /// A client from the environment and credentials files, if a key is found.
    pub fn from_env() -> Option<Self> {
        JevConfig::from_env().map(Self::new)
    }

    /// The model this client asks for.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Ask `questions` (name to question object) about `state`.
    ///
    /// Retries twice on 429 and 529, which the API documents as transient.
    pub async fn system_one(
        &self,
        state: Value,
        questions: Map<String, Value>,
    ) -> Result<SystemOneResponse, JevError> {
        let body = json!({
            "model": self.config.model,
            "state": state,
            "questions": questions,
        });
        let url = format!("{}/v1/systemone", self.config.base_url);

        let mut delay = Duration::from_millis(250);
        for attempt in 0.. {
            let response = self
                .http
                .post(&url)
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send()
                .await
                .map_err(|e| JevError::Http(e.to_string()))?;
            let status = response.status().as_u16();
            if (status == 429 || status == 529) && attempt < 2 {
                debug!(status, attempt, "System One busy; retrying");
                tokio::time::sleep(delay).await;
                delay *= 2;
                continue;
            }
            let text = response
                .text()
                .await
                .map_err(|e| JevError::Http(e.to_string()))?;
            if !(200..300).contains(&status) {
                return Err(JevError::Status {
                    status,
                    body: text.chars().take(300).collect(),
                });
            }
            return serde_json::from_str(&text).map_err(|e| JevError::Parse(e.to_string()));
        }
        unreachable!("the retry loop returns")
    }
}

/// The question [`JevAffinity`] asks about one agent and one task.
pub fn affinity_question(task: &TaskDefinition, capability: &str) -> (Value, Map<String, Value>) {
    let state = json!({ "task": task.description, "agent": capability });
    let mut questions = Map::new();
    questions.insert(
        "suits".into(),
        json!({
            "type": "noul",
            "instructions": "Can the agent described in `agent` carry out `task` well, \
                             based on its description?",
        }),
    );
    (state, questions)
}

/// An [`AffinitySource`] that asks Jev whether each agent can do each task.
///
/// The answer is the probability that the agent described can carry the task
/// out well, judged from the two descriptions alone. On any API failure it
/// gives no judgement, and the agent falls back to its role table.
pub struct JevAffinity {
    client: JevClient,
}

impl JevAffinity {
    /// An affinity source on `client`.
    pub fn new(client: JevClient) -> Self {
        Self { client }
    }

    /// From the environment and credentials files, if a key is found.
    pub fn from_env() -> Option<Self> {
        JevClient::from_env().map(Self::new)
    }
}

#[async_trait]
impl AffinitySource for JevAffinity {
    async fn assess(&self, task: &TaskDefinition, capability: &str) -> Option<f64> {
        let (state, questions) = affinity_question(task, capability);
        match self.client.system_one(state, questions).await {
            Ok(response) => {
                let p = response.noul("suits");
                if p.is_none() {
                    warn!("Jev response had no `suits` answer");
                }
                p
            }
            Err(e) => {
                warn!(error = %e, "Jev affinity failed; falling back to the role table");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parses_documented_response_shape() {
        // Shape from https://api.typesafe.ai/openapi.json, values from a live call.
        let raw = r#"{
            "model": "jev-1.14.0",
            "answers": {
                "suits": {"type": "noul", "noul": 0.95},
                "kind": {"type": "choice", "choice": "a", "confidence": 1.0,
                         "probabilities": {"a": 1.0, "b": 0.0}},
                "level": {"type": "score", "score": 1.25, "confidence": 0.6,
                          "legend": {"0": "x", "1": "y"}, "probabilities": {"0": 0.2, "1": 0.8}},
                "where": {"type": "box", "top": 0.1, "left": 0.1, "bottom": 0.5, "right": 0.5}
            },
            "usage": {"input_tokens": 505, "output_tokens": 87}
        }"#;
        let r: SystemOneResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(r.model, "jev-1.14.0");
        assert_eq!(r.noul("suits"), Some(0.95));
        assert_eq!(r.noul("kind"), None);
        assert!(matches!(r.answers["level"], Answer::Score { score, .. } if score == 1.25));
        assert_eq!(r.answers["where"], Answer::Other);
        assert_eq!(r.usage.input_tokens, 505);
    }

    #[test]
    fn test_affinity_question_judges_one_agent_alone() {
        let task = TaskDefinition::untyped("Translate the refund policy into Portuguese", 0.5);
        let (state, questions) = affinity_question(&task, "a translator");
        assert_eq!(state["task"], "Translate the refund policy into Portuguese");
        assert_eq!(state["agent"], "a translator");
        assert_eq!(questions["suits"]["type"], "noul");
        // Only this agent is in the state: no other agent to rank against.
        assert_eq!(state.as_object().unwrap().len(), 2);
    }
}
