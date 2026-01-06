//! Claude/Anthropic API client

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::backend::{
    GenerateRequest, GenerateResponse, LlmBackend, LlmError, LlmProvider,
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
        std::env::var("ANTHROPIC_API_KEY").ok().map(|api_key| {
            Self {
                api_key,
                model: "claude-sonnet-4-20250514".to_string(),
                timeout_secs: 120,
                max_tokens: 4096,
                api_version: "2023-06-01".to_string(),
            }
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

/// Request payload for Claude messages API
#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

/// Response from Claude messages API
#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: Option<String>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

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
            format!("{}...{}", &key[..4], &key[key.len()-4..])
        } else {
            "****".to_string()
        }
    }
    
    /// Set the model
    pub fn set_model(&mut self, model: &str) {
        self.config.model = model.to_string();
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
        let resp = self.client
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
            messages: vec![Message {
                role: "user".to_string(),
                content: request.prompt,
            }],
            system: request.system,
            temperature: Some(request.temperature),
            stop_sequences: request.stop_sequences,
        };
        
        let resp = self.client
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
            let retry_after = resp.headers()
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
                    "{}: {}", error_resp.error.error_type, error_resp.error.message
                )));
            }
            
            return Err(LlmError::GenerationFailed(format!(
                "Status {}: {}", status, error_body
            )));
        }
        
        let messages_resp: MessagesResponse = resp.json().await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;
        
        let content = messages_resp.content
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
        let config = ClaudeConfig::new("test-key")
            .with_model("claude-3-haiku-20240307");
        
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
