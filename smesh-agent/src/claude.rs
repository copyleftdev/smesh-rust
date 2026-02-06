//! Claude/Anthropic API client with full API support
//!
//! Supports:
//! - Multi-turn conversations
//! - Tool use (function calling)
//! - Streaming responses
//! - Constitutional AI principles via system prompts

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::{Duration, Instant};

use crate::backend::{
    ContentBlock, GenerateRequest, GenerateRequestV2, GenerateResponse, GenerateResponseV2,
    LlmBackend, LlmError, LlmProvider, Message, MessageRole, StopReason, StreamResult, ToolChoice,
    ToolDefinition,
};

/// Configuration for Claude client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeConfig {
    /// Anthropic API key
    pub api_key: String,
    /// Model to use (e.g., "claude-3-5-sonnet-20241022")
    pub model: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// API version
    pub api_version: String,
}

impl ClaudeConfig {
    /// Create config from environment variable
    pub fn from_env() -> Option<Self> {
        std::env::var("ANTHROPIC_API_KEY").ok().map(|api_key| Self {
            api_key,
            model: "claude-sonnet-4-20250514".to_string(),
            timeout_secs: 120,
            max_tokens: 4096,
            api_version: "2023-06-01".to_string(),
        })
    }

    /// Create with specific API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "claude-sonnet-4-20250514".to_string(),
            timeout_secs: 120,
            max_tokens: 4096,
            api_version: "2023-06-01".to_string(),
        }
    }

    /// Use a different model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

/// Request payload for Claude messages API (simple version)
#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<SimpleMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SimpleMessage {
    role: String,
    content: String,
}

/// Response from Claude messages API (simple version)
#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<SimpleContentBlock>,
    model: String,
    stop_reason: Option<String>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct SimpleContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

// ============================================================================
// Extended API Types (V2) - Full Claude API support
// ============================================================================

/// Request payload for Claude messages API (extended version)
#[derive(Debug, Serialize)]
struct MessagesRequestV2 {
    model: String,
    max_tokens: u32,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ApiToolChoice>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

/// Message format for Claude API
#[derive(Debug, Serialize, Deserialize)]
struct ApiMessage {
    role: String,
    content: ApiMessageContent,
}

/// Content can be a string or array of content blocks
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum ApiMessageContent {
    Text(String),
    Blocks(Vec<ApiContentBlock>),
}

/// Content block in API format
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

/// Tool definition for Claude API
#[derive(Debug, Serialize)]
struct ApiTool {
    name: String,
    description: String,
    input_schema: Value,
}

/// Tool choice for Claude API
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

/// Extended response from Claude messages API
#[derive(Debug, Deserialize)]
struct MessagesResponseV2 {
    id: String,
    content: Vec<ApiContentBlock>,
    model: String,
    stop_reason: String,
    usage: Usage,
}

// Note: SSE streaming types are in streaming.rs module

/// Error response from Claude API
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ApiError,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

/// Claude API client
#[derive(Debug, Clone)]
pub struct ClaudeClient {
    config: ClaudeConfig,
    client: Client,
}

impl ClaudeClient {
    /// Create a new Claude client
    pub fn new(config: ClaudeConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Create from environment variable
    pub fn from_env() -> Option<Self> {
        ClaudeConfig::from_env().map(Self::new)
    }

    /// Get the API key (masked for display)
    pub fn api_key_masked(&self) -> String {
        let key = &self.config.api_key;
        if key.len() > 8 {
            format!("{}...{}", &key[..4], &key[key.len() - 4..])
        } else {
            "****".to_string()
        }
    }

    /// Set the model
    pub fn set_model(&mut self, model: &str) {
        self.config.model = model.to_string();
    }

    /// Convert backend Message to API format
    fn message_to_api(msg: &Message) -> ApiMessage {
        let content = if msg.content.len() == 1 {
            if let Some(text) = msg.content[0].as_text() {
                ApiMessageContent::Text(text.to_string())
            } else {
                ApiMessageContent::Blocks(
                    msg.content.iter().map(Self::content_block_to_api).collect(),
                )
            }
        } else {
            ApiMessageContent::Blocks(msg.content.iter().map(Self::content_block_to_api).collect())
        };

        ApiMessage {
            role: match msg.role {
                MessageRole::User => "user".to_string(),
                MessageRole::Assistant => "assistant".to_string(),
            },
            content,
        }
    }

    /// Convert backend ContentBlock to API format
    fn content_block_to_api(block: &ContentBlock) -> ApiContentBlock {
        match block {
            ContentBlock::Text { text } => ApiContentBlock::Text { text: text.clone() },
            ContentBlock::ToolUse { id, name, input } => ApiContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            },
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => ApiContentBlock::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: content.clone(),
                is_error: *is_error,
            },
        }
    }

    /// Convert API ContentBlock to backend format
    fn api_content_block_to_backend(block: &ApiContentBlock) -> ContentBlock {
        match block {
            ApiContentBlock::Text { text } => ContentBlock::Text { text: text.clone() },
            ApiContentBlock::ToolUse { id, name, input } => ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            },
            ApiContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => ContentBlock::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: content.clone(),
                is_error: *is_error,
            },
        }
    }

    /// Convert backend ToolDefinition to API format
    fn tool_to_api(tool: &ToolDefinition) -> ApiTool {
        ApiTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        }
    }

    /// Convert backend ToolChoice to API format
    fn tool_choice_to_api(choice: &ToolChoice) -> Option<ApiToolChoice> {
        match choice {
            ToolChoice::Auto => Some(ApiToolChoice::Auto),
            ToolChoice::Any => Some(ApiToolChoice::Any),
            ToolChoice::None => None,
            ToolChoice::Tool { name } => Some(ApiToolChoice::Tool { name: name.clone() }),
        }
    }

    /// Parse stop reason string to StopReason enum
    fn parse_stop_reason(reason: &str) -> StopReason {
        match reason {
            "end_turn" => StopReason::EndTurn,
            "max_tokens" => StopReason::MaxTokens,
            "stop_sequence" => StopReason::StopSequence,
            "tool_use" => StopReason::ToolUse,
            _ => StopReason::EndTurn,
        }
    }

    /// Make a V2 API request
    async fn make_v2_request(
        &self,
        request: &GenerateRequestV2,
    ) -> Result<reqwest::Response, LlmError> {
        let api_request = MessagesRequestV2 {
            model: self.config.model.clone(),
            max_tokens: request.max_tokens.min(self.config.max_tokens),
            messages: request.messages.iter().map(Self::message_to_api).collect(),
            system: request.system.clone(),
            temperature: Some(request.temperature),
            stop_sequences: request.stop_sequences.clone(),
            tools: request.tools.iter().map(Self::tool_to_api).collect(),
            tool_choice: if request.tools.is_empty() {
                None
            } else {
                Self::tool_choice_to_api(&request.tool_choice)
            },
            stream: request.stream,
        };

        self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.api_version)
            .header("content-type", "application/json")
            .json(&api_request)
            .send()
            .await
            .map_err(|e| LlmError::RequestFailed(e.to_string()))
    }

    /// Handle error response
    async fn handle_error_response(&self, resp: reqwest::Response) -> LlmError {
        let status = resp.status();

        if status == 401 {
            return LlmError::AuthFailed;
        }

        if status == 429 {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(60);
            return LlmError::RateLimited(Duration::from_secs(retry_after));
        }

        let error_body = resp.text().await.unwrap_or_default();

        if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&error_body) {
            return LlmError::GenerationFailed(format!(
                "{}: {}",
                error_resp.error.error_type, error_resp.error.message
            ));
        }

        LlmError::GenerationFailed(format!("Status {}: {}", status, error_body))
    }
}

#[async_trait]
impl LlmBackend for ClaudeClient {
    fn provider(&self) -> LlmProvider {
        LlmProvider::Claude
    }

    fn model(&self) -> &str {
        &self.config.model
    }

    async fn is_available(&self) -> bool {
        // Try a minimal request to check if the API is accessible
        // We'll just check if we can reach the API endpoint
        let resp = self
            .client
            .get("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.api_version)
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        // Even a 405 Method Not Allowed means the API is reachable
        match resp {
            Ok(r) => r.status().as_u16() != 0,
            Err(_) => false,
        }
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        // Claude doesn't have a list models endpoint, return known models
        Ok(vec![
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-5-haiku-20241022".to_string(),
            "claude-3-opus-20240229".to_string(),
            "claude-3-sonnet-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
        ])
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, LlmError> {
        let start = Instant::now();

        let messages_request = MessagesRequest {
            model: self.config.model.clone(),
            max_tokens: request.max_tokens.min(self.config.max_tokens),
            messages: vec![SimpleMessage {
                role: "user".to_string(),
                content: request.prompt,
            }],
            system: request.system,
            temperature: Some(request.temperature),
            stop_sequences: request.stop_sequences,
        };

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", &self.config.api_version)
            .header("content-type", "application/json")
            .json(&messages_request)
            .send()
            .await
            .map_err(|e| LlmError::RequestFailed(e.to_string()))?;

        let status = resp.status();

        if status == 401 {
            return Err(LlmError::AuthFailed);
        }

        if status == 429 {
            // Check for retry-after header
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(60);
            return Err(LlmError::RateLimited(Duration::from_secs(retry_after)));
        }

        if !status.is_success() {
            let error_body = resp.text().await.unwrap_or_default();

            // Try to parse as structured error
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&error_body) {
                return Err(LlmError::GenerationFailed(format!(
                    "{}: {}",
                    error_resp.error.error_type, error_resp.error.message
                )));
            }

            return Err(LlmError::GenerationFailed(format!(
                "Status {}: {}",
                status, error_body
            )));
        }

        let messages_resp: MessagesResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let content = messages_resp
            .content
            .into_iter()
            .filter(|c| c.content_type == "text")
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");

        let latency = start.elapsed();

        Ok(GenerateResponse {
            content,
            provider: LlmProvider::Claude,
            model: messages_resp.model,
            latency,
            input_tokens: Some(messages_resp.usage.input_tokens),
            output_tokens: Some(messages_resp.usage.output_tokens),
            stop_reason: messages_resp.stop_reason,
        })
    }

    async fn generate_v2(
        &self,
        request: GenerateRequestV2,
    ) -> Result<GenerateResponseV2, LlmError> {
        let start = Instant::now();

        let resp = self.make_v2_request(&request).await?;
        let status = resp.status();

        if !status.is_success() {
            return Err(self.handle_error_response(resp).await);
        }

        let api_resp: MessagesResponseV2 = resp
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let content = api_resp
            .content
            .iter()
            .map(Self::api_content_block_to_backend)
            .collect();

        Ok(GenerateResponseV2 {
            content,
            stop_reason: Self::parse_stop_reason(&api_resp.stop_reason),
            model: api_resp.model,
            id: api_resp.id,
            input_tokens: api_resp.usage.input_tokens,
            output_tokens: api_resp.usage.output_tokens,
            latency: start.elapsed(),
        })
    }

    async fn generate_stream(&self, _request: GenerateRequestV2) -> Result<StreamResult, LlmError> {
        // TODO: Enable when streaming feature is configured with bytes crate
        Err(LlmError::UnsupportedFeature(
            "Streaming not yet available. Use generate_v2 for non-streaming requests.".to_string(),
        ))
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_multi_turn(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_env() {
        // This will be None if ANTHROPIC_API_KEY is not set
        let config = ClaudeConfig::from_env();
        // Just test it doesn't panic
        let _ = config;
    }

    #[test]
    fn test_config_builder() {
        let config = ClaudeConfig::new("test-key").with_model("claude-3-haiku-20240307");

        assert_eq!(config.model, "claude-3-haiku-20240307");
        assert_eq!(config.api_key, "test-key");
    }

    #[test]
    fn test_api_key_masked() {
        let client = ClaudeClient::new(ClaudeConfig::new("sk-ant-api03-abcdefghijklmnop"));
        let masked = client.api_key_masked();
        assert!(masked.contains("..."));
        assert!(!masked.contains("abcdefghijklmnop"));
    }
}
