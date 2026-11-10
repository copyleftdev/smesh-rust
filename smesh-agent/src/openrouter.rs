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
    ContentBlock, GenerateRequest as BackendRequest, GenerateRequestV2,
    GenerateResponse as BackendResponse, GenerateResponseV2, LlmBackend, LlmError, LlmProvider,
    MessageRole, StopReason, ToolChoice,
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
    /// `null` when the model only calls tools.
    #[serde(default, deserialize_with = "null_as_empty")]
    content: String,
}

/// Read a JSON `null` string as empty: OpenAI-compatible APIs send
/// `"content": null` on a message that only calls tools.
fn null_as_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
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
            .header("HTTP-Referer", "https://github.com/copyleftdev/smesh-rust")
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
            .header("HTTP-Referer", "https://github.com/copyleftdev/smesh-rust")
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

    async fn generate_v2(
        &self,
        request: GenerateRequestV2,
    ) -> Result<GenerateResponseV2, LlmError> {
        self.chat_v2(request).await
    }

    /// Tool use depends on the model; most current OpenRouter models support it.
    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_multi_turn(&self) -> bool {
        true
    }
}

// ============================================================================
// Tool use over the OpenAI-compatible API
// ============================================================================

/// The OpenAI-compatible `messages` for a V2 request.
///
/// A [`ContentBlock::ToolUse`] becomes an assistant `tool_calls` entry, and a
/// [`ContentBlock::ToolResult`] becomes its own `tool` message, which is how
/// the Chat Completions API pairs a call with its result.
fn chat_messages(request: &GenerateRequestV2) -> Vec<serde_json::Value> {
    use serde_json::json;
    let mut out = Vec::new();
    if let Some(sys) = request.system.as_deref() {
        out.push(json!({ "role": "system", "content": sys }));
    }
    for message in &request.messages {
        let text: String = message
            .content
            .iter()
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("");
        match message.role {
            MessageRole::Assistant => {
                let calls: Vec<serde_json::Value> = message
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        ContentBlock::ToolUse { id, name, input } => Some(json!({
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": input.to_string() },
                        })),
                        _ => None,
                    })
                    .collect();
                if calls.is_empty() {
                    out.push(json!({ "role": "assistant", "content": text }));
                } else {
                    let content = if text.is_empty() {
                        serde_json::Value::Null
                    } else {
                        json!(text)
                    };
                    out.push(
                        json!({ "role": "assistant", "content": content, "tool_calls": calls }),
                    );
                }
            }
            MessageRole::User => {
                for block in &message.content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } = block
                    {
                        let content = if *is_error {
                            format!("ERROR: {content}")
                        } else {
                            content.clone()
                        };
                        out.push(json!({ "role": "tool", "tool_call_id": tool_use_id, "content": content }));
                    }
                }
                if !text.is_empty() {
                    out.push(json!({ "role": "user", "content": text }));
                }
            }
        }
    }
    out
}

/// The full Chat Completions payload for a V2 request.
fn chat_payload(model: &str, max_tokens: u32, request: &GenerateRequestV2) -> serde_json::Value {
    use serde_json::json;
    let mut payload = json!({
        "model": model,
        "messages": chat_messages(request),
        "temperature": request.temperature,
        "max_tokens": request.max_tokens.min(max_tokens),
    });
    if !request.stop_sequences.is_empty() {
        payload["stop"] = json!(request.stop_sequences);
    }
    if !request.tools.is_empty() {
        payload["tools"] = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                })
            })
            .collect();
        payload["tool_choice"] = match &request.tool_choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::Any => json!("required"),
            ToolChoice::None => json!("none"),
            ToolChoice::Tool { name } => {
                json!({ "type": "function", "function": { "name": name } })
            }
        };
    }
    payload
}

/// Content blocks and stop reason from a Chat Completions response.
fn parse_chat_v2(body: &serde_json::Value) -> Result<(Vec<ContentBlock>, StopReason), LlmError> {
    let choice = body
        .get("choices")
        .and_then(|c| c.get(0))
        .ok_or_else(|| LlmError::InvalidResponse("no choices returned".to_string()))?;
    let message = &choice["message"];
    let mut content = Vec::new();
    if let Some(text) = message["content"].as_str().filter(|t| !t.is_empty()) {
        content.push(ContentBlock::Text {
            text: text.to_string(),
        });
    }
    for call in message["tool_calls"].as_array().into_iter().flatten() {
        let name = call["function"]["name"].as_str().unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let raw = call["function"]["arguments"].as_str().unwrap_or("{}");
        // Arguments arrive as a JSON string. One that does not parse is
        // passed through as a string, and the tool reports it as bad input
        // so the model can correct itself.
        let input = if raw.trim().is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.into()))
        };
        content.push(ContentBlock::ToolUse {
            id: call["id"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple())),
            name: name.to_string(),
            input,
        });
    }
    // Some providers finish a tool call with "stop"; the calls decide it.
    let stop = if content.iter().any(ContentBlock::is_tool_use) {
        StopReason::ToolUse
    } else {
        match choice["finish_reason"].as_str() {
            Some("length") => StopReason::MaxTokens,
            Some("tool_calls") => StopReason::ToolUse,
            _ => StopReason::EndTurn,
        }
    };
    Ok((content, stop))
}

impl OpenRouterClient {
    /// POST a Chat Completions payload and return the JSON body.
    async fn post_chat(&self, payload: &serde_json::Value) -> Result<serde_json::Value, LlmError> {
        if self.config.api_key.is_empty() {
            return Err(LlmError::AuthFailed);
        }
        let url = format!("{}/chat/completions", self.config.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .header("HTTP-Referer", "https://github.com/copyleftdev/smesh-rust")
            .header("X-Title", "SMESH")
            .json(payload)
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
        resp.json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))
    }

    /// Multi-turn generation with tools, over the OpenAI-compatible API.
    async fn chat_v2(&self, request: GenerateRequestV2) -> Result<GenerateResponseV2, LlmError> {
        let start = Instant::now();
        let payload = chat_payload(&self.config.model, self.config.max_tokens, &request);
        let body = self.post_chat(&payload).await?;
        // OpenRouter reports some upstream failures as a 200 with an error body.
        if let Some(err) = body.get("error") {
            return Err(LlmError::GenerationFailed(err.to_string()));
        }
        let (content, stop_reason) = parse_chat_v2(&body)?;
        Ok(GenerateResponseV2 {
            content,
            stop_reason,
            model: body["model"]
                .as_str()
                .filter(|m| !m.is_empty())
                .unwrap_or(&self.config.model)
                .to_string(),
            id: body["id"].as_str().unwrap_or_default().to_string(),
            input_tokens: body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            latency: start.elapsed(),
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

/// The value of `key` in a dotenv file. A key assigned more than once takes
/// its last value, as `source` and dotenv loaders do: a file with an old key
/// above a new one means the new one.
pub(crate) fn read_key_from_file(path: &std::path::Path, key: &str) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    let mut found = None;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == key {
                found = Some(unquote(v.trim()));
            }
        }
    }
    found
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

    fn tool_request() -> GenerateRequestV2 {
        use crate::backend::{Message, ToolDefinition};
        let mut request = GenerateRequestV2::new(vec![
            Message::user("What is in notes.txt?"),
            Message {
                role: MessageRole::Assistant,
                content: vec![
                    ContentBlock::Text {
                        text: "Let me look.".into(),
                    },
                    ContentBlock::ToolUse {
                        id: "call_1".into(),
                        name: "files__read".into(),
                        input: serde_json::json!({ "path": "notes.txt" }),
                    },
                ],
            },
            Message::tool_result("call_1", "no such file", true),
        ]);
        request.system = Some("You are careful.".into());
        request.tools = vec![ToolDefinition::new(
            "files__read",
            "Read a file",
            serde_json::json!({ "type": "object", "properties": { "path": { "type": "string" } } }),
        )];
        request
    }

    #[test]
    fn a_tool_conversation_maps_onto_chat_completions() {
        let payload = chat_payload("m", 1000, &tool_request());
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "Let me look.");
        let call = &msgs[2]["tool_calls"][0];
        assert_eq!(call["id"], "call_1");
        assert_eq!(call["function"]["name"], "files__read");
        // Arguments travel as a JSON string, as the API requires.
        let args: serde_json::Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["path"], "notes.txt");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "call_1");
        assert_eq!(msgs[3]["content"], "ERROR: no such file");
        assert_eq!(
            payload["tools"][0]["function"]["parameters"]["type"],
            "object"
        );
        assert_eq!(payload["tool_choice"], "auto");
    }

    #[test]
    fn no_tools_sends_no_tool_fields() {
        let request = GenerateRequestV2::new(vec![crate::backend::Message::user("hi")]);
        let payload = chat_payload("m", 1000, &request);
        assert!(payload.get("tools").is_none());
        assert!(payload.get("tool_choice").is_none());
    }

    #[test]
    fn tool_calls_in_a_response_become_tool_use_blocks() {
        let body = serde_json::json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": null,
                    "tool_calls": [
                        { "id": "a", "type": "function",
                          "function": { "name": "files__read", "arguments": "{\"path\":\"x\"}" } },
                        { "id": "b", "type": "function",
                          "function": { "name": "files__read", "arguments": "not json" } }
                    ]
                }
            }]
        });
        let (content, stop) = parse_chat_v2(&body).unwrap();
        // "stop" with tool calls still means the model wants its tools run.
        assert_eq!(stop, StopReason::ToolUse);
        assert_eq!(content.len(), 2);
        match &content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!((id.as_str(), name.as_str()), ("a", "files__read"));
                assert_eq!(input["path"], "x");
            }
            other => panic!("expected a tool use, got {other:?}"),
        }
        match &content[1] {
            ContentBlock::ToolUse { input, .. } => assert_eq!(input, "not json"),
            other => panic!("expected a tool use, got {other:?}"),
        }
    }

    #[test]
    fn a_plain_answer_ends_the_turn() {
        let body = serde_json::json!({
            "choices": [{ "finish_reason": "stop", "message": { "content": "done" } }]
        });
        let (content, stop) = parse_chat_v2(&body).unwrap();
        assert_eq!(stop, StopReason::EndTurn);
        assert_eq!(content[0].as_text(), Some("done"));
        let null_content: ChatResponseMessage =
            serde_json::from_value(serde_json::json!({ "content": null })).unwrap();
        assert_eq!(null_content.content, "");
    }

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

    #[test]
    fn test_read_key_from_file_last_assignment_wins() {
        let path = std::env::temp_dir().join("smesh_openrouter_last_wins_test.env");
        std::fs::write(
            &path,
            "OPENROUTER_API_KEY=sk-or-old\nOTHER=x\nOPENROUTER_API_KEY=sk-or-new\n",
        )
        .unwrap();

        assert_eq!(
            read_key_from_file(&path, "OPENROUTER_API_KEY"),
            Some("sk-or-new".to_string())
        );

        let _ = std::fs::remove_file(&path);
    }
}
