//! Ollama client for local LLM inference

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::backend::{
    GenerateRequest as BackendRequest, GenerateResponse as BackendResponse, LlmBackend, LlmError,
    LlmProvider,
};

/// Ollama client errors
#[derive(Error, Debug)]
pub enum OllamaError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Timeout")]
    Timeout,
}

/// Configuration for Ollama client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Base URL for Ollama API
    pub base_url: String,
    /// Model to use
    pub model: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Temperature for generation
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "deepseek-coder-v2:16b".to_string(),
            timeout_secs: 120,
            temperature: 0.7,
            max_tokens: 2048,
        }
    }
}

/// Request payload for Ollama generate endpoint
#[derive(Debug, Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    options: GenerateOptions,
}

#[derive(Debug, Serialize)]
struct GenerateOptions {
    temperature: f32,
    num_predict: u32,
}

/// Response from Ollama generate endpoint
#[derive(Debug, Deserialize)]
struct GenerateResponse {
    response: String,
    #[serde(default)]
    #[allow(dead_code)]
    done: bool,
}

/// Response from Ollama tags endpoint (list models)
#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    name: String,
    #[serde(default)]
    #[allow(dead_code)]
    size: u64,
}

/// Client for interacting with Ollama API
#[derive(Debug, Clone)]
pub struct OllamaClient {
    config: OllamaConfig,
    client: Client,
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(config: OllamaConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Create with default configuration
    pub fn default_client() -> Self {
        Self::new(OllamaConfig::default())
    }

    /// Check if Ollama is available
    pub async fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.config.base_url);

        match self
            .client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// List available models
    pub async fn list_models(&self) -> Result<Vec<String>, OllamaError> {
        let url = format!("{}/api/tags", self.config.base_url);

        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            return Err(OllamaError::ConnectionFailed(format!(
                "Status: {}",
                resp.status()
            )));
        }

        let tags: TagsResponse = resp.json().await?;
        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }

    /// Generate a response from the model
    pub async fn generate(
        &self,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<String, OllamaError> {
        let url = format!("{}/api/generate", self.config.base_url);

        let request = GenerateRequest {
            model: self.config.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
            system: system.map(|s| s.to_string()),
            options: GenerateOptions {
                temperature: self.config.temperature,
                num_predict: self.config.max_tokens,
            },
        };

        let resp = self.client.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(OllamaError::GenerationFailed(format!(
                "Status {}: {}",
                status, body
            )));
        }

        let gen_resp: GenerateResponse = resp.json().await?;
        Ok(gen_resp.response)
    }

    /// Generate with a specific model (override config)
    pub async fn generate_with_model(
        &self,
        model: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<String, OllamaError> {
        let url = format!("{}/api/generate", self.config.base_url);

        let request = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: false,
            system: system.map(|s| s.to_string()),
            options: GenerateOptions {
                temperature: self.config.temperature,
                num_predict: self.config.max_tokens,
            },
        };

        let resp = self.client.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            return Err(OllamaError::GenerationFailed(format!(
                "Status: {}",
                resp.status()
            )));
        }

        let gen_resp: GenerateResponse = resp.json().await?;
        Ok(gen_resp.response)
    }

    /// Get the configured model name
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Update the model
    pub fn set_model(&mut self, model: &str) {
        self.config.model = model.to_string();
    }
}

#[async_trait]
impl LlmBackend for OllamaClient {
    fn provider(&self) -> LlmProvider {
        LlmProvider::Ollama
    }

    fn model(&self) -> &str {
        &self.config.model
    }

    async fn is_available(&self) -> bool {
        OllamaClient::is_available(self).await
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        OllamaClient::list_models(self)
            .await
            .map_err(|e| LlmError::RequestFailed(e.to_string()))
    }

    async fn generate(&self, request: BackendRequest) -> Result<BackendResponse, LlmError> {
        let start = Instant::now();

        let result = OllamaClient::generate(self, &request.prompt, request.system.as_deref())
            .await
            .map_err(|e| LlmError::GenerationFailed(e.to_string()))?;

        let latency = start.elapsed();

        Ok(BackendResponse {
            content: result,
            provider: LlmProvider::Ollama,
            model: self.config.model.clone(),
            latency,
            input_tokens: None, // Ollama doesn't always report tokens
            output_tokens: None,
            stop_reason: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert!(config.model.contains("deepseek"));
    }

    #[tokio::test]
    async fn test_client_creation() {
        let client = OllamaClient::default_client();
        assert_eq!(client.model(), "deepseek-coder-v2:16b");
    }
}
