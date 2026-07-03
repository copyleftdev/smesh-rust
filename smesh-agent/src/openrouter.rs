//! OpenRouter client for LLM inference
//!
//! OpenRouter exposes an OpenAI-compatible Chat Completions API, so this client
//! talks to `{base_url}/chat/completions` with a Bearer token. Credentials are
//! resolved from the environment, falling back to a dotenv-style credentials
//! file (e.g. `~/.creds/openrouter.env`) so local testing works without having
//! to export variables into every shell.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::backend::{
    GenerateRequest as BackendRequest, GenerateResponse as BackendResponse, LlmBackend, LlmError,
    LlmProvider,
};

/// Default OpenRouter API base URL (OpenAI-compatible).
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Default model slug used for local testing.
pub const DEFAULT_MODEL: &str = "google/gemini-2.5-flash-lite";

/// OpenRouter client errors
#[derive(Error, Debug)]
pub enum OpenRouterError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("Missing credentials: set OPENROUTER_API_KEY (or add it to ~/.creds/openrouter.env)")]
    MissingCredentials,

    #[error("Authentication failed (check OPENROUTER_API_KEY)")]
    AuthFailed,

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

/// Configuration for the OpenRouter client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    /// Base URL for the OpenRouter API (OpenAI-compatible)
    pub base_url: String,
    /// API key (Bearer token)
    pub api_key: String,
    /// Model slug (e.g. "google/gemini-2.5-flash-lite")
    pub model: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Temperature for generation
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: String::new(),
            model: DEFAULT_MODEL.to_string(),
            timeout_secs: 120,
            temperature: 0.7,
            max_tokens: 2048,
        }
    }
}

impl OpenRouterConfig {
    /// Build config from the environment.
    ///
    /// Reads `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `OPENROUTER_MODEL`,
    /// falling back to a dotenv-style credentials file (see
    /// [`resolve_env_var`]). Returns `None` if no API key can be found.
    pub fn from_env() -> Option<Self> {
        let api_key = resolve_env_var("OPENROUTER_API_KEY")?;
        Some(Self {
            base_url: resolve_env_var("OPENROUTER_BASE_URL")
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            api_key,
            model: resolve_env_var("OPENROUTER_MODEL").unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            timeout_secs: 120,
            temperature: 0.7,
            max_tokens: 2048,
        })
    }

    /// Create a config with a specific API key (base URL and model default).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    /// Use a specific model slug.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

/// OpenAI-compatible chat message.
#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Request payload for the chat completions endpoint.
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    model: String,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// Response from the models endpoint.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
}

/// Client for the OpenRouter API.
#[derive(Debug, Clone)]
pub struct OpenRouterClient {
    config: OpenRouterConfig,
    client: Client,
}

impl OpenRouterClient {
    /// Create a new client from an explicit config.
    pub fn new(config: OpenRouterConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Create a client from the environment / credentials file.
    ///
    /// Returns `None` if no API key is configured.
    pub fn from_env() -> Option<Self> {
        OpenRouterConfig::from_env().map(Self::new)
    }

    /// Masked API key for display.
    pub fn api_key_masked(&self) -> String {
        let key = &self.config.api_key;
        if key.len() > 8 {
            format!("{}...{}", &key[..4], &key[key.len() - 4..])
        } else {
            "****".to_string()
        }
    }

    /// Check that the API is reachable and the key is accepted.
    pub async fn is_available(&self) -> bool {
        if self.config.api_key.is_empty() {
            return false;
        }
        let url = format!("{}/models", self.config.base_url);
        match self
            .client
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .timeout(Duration::from_secs(10))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// List available model slugs.
    pub async fn list_models(&self) -> Result<Vec<String>, OpenRouterError> {
        let url = format!("{}/models", self.config.base_url);
        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .send()
            .await?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(OpenRouterError::AuthFailed);
        }
        if !resp.status().is_success() {
            return Err(OpenRouterError::ConnectionFailed(format!(
                "Status: {}",
                resp.status()
            )));
        }

        let models: ModelsResponse = resp.json().await?;
        Ok(models.data.into_iter().map(|m| m.id).collect())
    }

    /// Generate a completion for a prompt with an optional system message.
    pub async fn generate(
        &self,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<String, OpenRouterError> {
        self.generate_with_model(&self.config.model, prompt, system)
            .await
    }

    /// Generate using a specific model, overriding the configured default.
    pub async fn generate_with_model(
        &self,
        model: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<String, OpenRouterError> {
        if self.config.api_key.is_empty() {
            return Err(OpenRouterError::MissingCredentials);
        }

        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        });

        let request = ChatRequest {
            model: model.to_string(),
            messages,
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
        };

        let url = format!("{}/chat/completions", self.config.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            // Optional attribution headers recommended by OpenRouter.
            .header("HTTP-Referer", "https://github.com/smesh-protocol/smesh-rust")
            .header("X-Title", "SMESH")
            .json(&request)
            .send()
            .await?;

        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(OpenRouterError::AuthFailed);
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(OpenRouterError::GenerationFailed(format!(
                "Status {}: {}",
                status, body
            )));
        }

        let chat: ChatResponse = resp
            .json()
            .await
            .map_err(|e| OpenRouterError::InvalidResponse(e.to_string()))?;

        chat.choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| OpenRouterError::InvalidResponse("no choices returned".to_string()))
    }

    /// Configured model slug.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Override the configured model.
    pub fn set_model(&mut self, model: &str) {
        self.config.model = model.to_string();
    }
}

#[async_trait]
impl LlmBackend for OpenRouterClient {
    fn provider(&self) -> LlmProvider {
        LlmProvider::OpenRouter
    }

    fn model(&self) -> &str {
        &self.config.model
    }

    async fn is_available(&self) -> bool {
        OpenRouterClient::is_available(self).await
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        OpenRouterClient::list_models(self)
            .await
            .map_err(|e| LlmError::RequestFailed(e.to_string()))
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendResponse, LlmError> {
        let start = Instant::now();

        // Per-request overrides on top of the configured client.
        let mut messages = Vec::new();
        if let Some(sys) = request.system.as_deref() {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: request.prompt.clone(),
        });

        let payload = ChatRequest {
            model: self.config.model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens.min(self.config.max_tokens),
        };

        if self.config.api_key.is_empty() {
            return Err(LlmError::AuthFailed);
        }

        let url = format!("{}/chat/completions", self.config.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .header("HTTP-Referer", "https://github.com/smesh-protocol/smesh-rust")
            .header("X-Title", "SMESH")
            .json(&payload)
            .send()
            .await
            .map_err(|e| LlmError::RequestFailed(e.to_string()))?;

        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(LlmError::AuthFailed);
        }
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(60);
            return Err(LlmError::RateLimited(Duration::from_secs(retry_after)));
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::GenerationFailed(format!(
                "Status {}: {}",
                status, body
            )));
        }

        let chat: ChatResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let content = chat
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();
        let stop_reason = chat.choices.first().and_then(|c| c.finish_reason.clone());
        let model = if chat.model.is_empty() {
            self.config.model.clone()
        } else {
            chat.model
        };
        let (input_tokens, output_tokens) = chat
            .usage
            .map(|u| (Some(u.prompt_tokens), Some(u.completion_tokens)))
            .unwrap_or((None, None));

        Ok(BackendResponse {
            content,
            provider: LlmProvider::OpenRouter,
            model,
            latency: start.elapsed(),
            input_tokens,
            output_tokens,
            stop_reason,
        })
    }
}

/// Resolve an environment variable, falling back to dotenv-style files.
///
/// Lookup order:
/// 1. The process environment.
/// 2. `$OPENROUTER_ENV_FILE`, if set.
/// 3. `~/.creds/openrouter.env`
/// 4. `./openrouter.env`
/// 5. `./.env.local`
///
/// This keeps secrets out of the repo while letting local testing work without
/// manually exporting variables.
pub fn resolve_env_var(key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(key) {
        if !v.is_empty() {
            return Some(v);
        }
    }

    for path in candidate_env_files() {
        if let Some(v) = read_key_from_file(&path, key) {
            if !v.is_empty() {
                return Some(v);
            }
        }
    }

    None
}

fn candidate_env_files() -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    if let Ok(explicit) = std::env::var("OPENROUTER_ENV_FILE") {
        paths.push(std::path::PathBuf::from(explicit));
    }
    if let Some(home) = std::env::var_os("HOME") {
        paths.push(std::path::Path::new(&home).join(".creds/openrouter.env"));
    }
    paths.push(std::path::PathBuf::from("openrouter.env"));
    paths.push(std::path::PathBuf::from(".env.local"));
    paths
}

fn read_key_from_file(path: &std::path::Path, key: &str) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == key {
                return Some(unquote(v.trim()));
            }
        }
    }
    None
}

fn unquote(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
    {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OpenRouterConfig::default();
        assert_eq!(config.base_url, DEFAULT_BASE_URL);
        assert_eq!(config.model, DEFAULT_MODEL);
        assert!(config.api_key.is_empty());
    }

    #[test]
    fn test_config_builder() {
        let config = OpenRouterConfig::new("sk-or-test").with_model("anthropic/claude-3.5-sonnet");
        assert_eq!(config.api_key, "sk-or-test");
        assert_eq!(config.model, "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn test_api_key_masked() {
        let client = OpenRouterClient::new(OpenRouterConfig::new("sk-or-abcdefghijklmnop"));
        let masked = client.api_key_masked();
        assert!(masked.contains("..."));
        assert!(!masked.contains("abcdefghijklmnop"));
    }

    #[test]
    fn test_unquote() {
        assert_eq!(unquote("\"quoted\""), "quoted");
        assert_eq!(unquote("'single'"), "single");
        assert_eq!(unquote("bare"), "bare");
    }

    #[test]
    fn test_read_key_from_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("smesh_openrouter_test.env");
        std::fs::write(
            &path,
            "# comment\nOPENROUTER_API_KEY=sk-or-xyz\nexport OPENROUTER_MODEL=google/gemini-2.5-flash-lite\n",
        )
        .unwrap();

        assert_eq!(
            read_key_from_file(&path, "OPENROUTER_API_KEY"),
            Some("sk-or-xyz".to_string())
        );
        assert_eq!(
            read_key_from_file(&path, "OPENROUTER_MODEL"),
            Some("google/gemini-2.5-flash-lite".to_string())
        );
        assert_eq!(read_key_from_file(&path, "NOT_PRESENT"), None);

        let _ = std::fs::remove_file(&path);
    }
}
