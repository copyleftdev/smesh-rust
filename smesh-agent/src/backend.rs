//! Unified LLM backend trait for multiple providers

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors from LLM backends
#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Request failed: {0}")]
    RequestFailed(String),
    
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Generation failed: {0}")]
    GenerationFailed(String),
    
    #[error("Rate limited: retry after {0:?}")]
    RateLimited(Duration),
    
    #[error("Authentication failed")]
    AuthFailed,
    
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
    
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

/// LLM provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmProvider {
    /// Local Ollama instance
    Ollama,
    /// Anthropic Claude API
    Claude,
    /// OpenAI API (future)
    OpenAI,
}

impl std::fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmProvider::Ollama => write!(f, "Ollama"),
            LlmProvider::Claude => write!(f, "Claude"),
            LlmProvider::OpenAI => write!(f, "OpenAI"),
        }
    }
}

/// Generation request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// The prompt/user message
    pub prompt: String,
    /// Optional system message
    pub system: Option<String>,
    /// Temperature (0.0 - 1.0)
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

impl GenerateRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            system: None,
            temperature: 0.7,
            max_tokens: 2048,
            stop_sequences: Vec::new(),
        }
    }
    
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }
    
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
    
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }
}

/// Generation response with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// The generated text
    pub content: String,
    /// Provider that generated the response
    pub provider: LlmProvider,
    /// Model used
    pub model: String,
    /// Time taken for generation
    pub latency: Duration,
    /// Input tokens (if available)
    pub input_tokens: Option<u32>,
    /// Output tokens (if available)
    pub output_tokens: Option<u32>,
    /// Stop reason
    pub stop_reason: Option<String>,
}

impl GenerateResponse {
    /// Calculate tokens per second (if output_tokens available)
    pub fn tokens_per_second(&self) -> Option<f64> {
        self.output_tokens.map(|tokens| {
            tokens as f64 / self.latency.as_secs_f64()
        })
    }
}

/// Unified trait for LLM backends
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Get the provider type
    fn provider(&self) -> LlmProvider;
    
    /// Get the current model name
    fn model(&self) -> &str;
    
    /// Check if the backend is available
    async fn is_available(&self) -> bool;
    
    /// List available models (if supported)
    async fn list_models(&self) -> Result<Vec<String>, LlmError>;
    
    /// Generate a response
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, LlmError>;
    
    /// Simple generate with just a prompt
    async fn generate_simple(&self, prompt: &str) -> Result<String, LlmError> {
        let request = GenerateRequest::new(prompt);
        let response = self.generate(request).await?;
        Ok(response.content)
    }
    
    /// Generate with system prompt
    async fn generate_with_system(&self, prompt: &str, system: &str) -> Result<String, LlmError> {
        let request = GenerateRequest::new(prompt).with_system(system);
        let response = self.generate(request).await?;
        Ok(response.content)
    }
}

/// Benchmark results for comparing backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Provider tested
    pub provider: LlmProvider,
    /// Model tested
    pub model: String,
    /// Prompt used
    pub prompt: String,
    /// Response received
    pub response: String,
    /// Time to first byte (if streaming)
    pub ttfb: Option<Duration>,
    /// Total latency
    pub total_latency: Duration,
    /// Input tokens
    pub input_tokens: Option<u32>,
    /// Output tokens  
    pub output_tokens: Option<u32>,
    /// Tokens per second
    pub tokens_per_second: Option<f64>,
    /// Success
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Run a benchmark against a backend
pub async fn benchmark_backend(
    backend: &dyn LlmBackend,
    prompt: &str,
    system: Option<&str>,
) -> BenchmarkResult {
    let start = Instant::now();
    
    let request = GenerateRequest::new(prompt)
        .with_temperature(0.7)
        .with_max_tokens(512);
    
    let request = if let Some(sys) = system {
        request.with_system(sys)
    } else {
        request
    };
    
    match backend.generate(request).await {
        Ok(response) => {
            let tps = response.tokens_per_second();
            BenchmarkResult {
                provider: backend.provider(),
                model: backend.model().to_string(),
                prompt: prompt.to_string(),
                response: response.content,
                ttfb: None,
                total_latency: start.elapsed(),
                input_tokens: response.input_tokens,
                output_tokens: response.output_tokens,
                tokens_per_second: tps,
                success: true,
                error: None,
            }
        }
        Err(e) => BenchmarkResult {
            provider: backend.provider(),
            model: backend.model().to_string(),
            prompt: prompt.to_string(),
            response: String::new(),
            ttfb: None,
            total_latency: start.elapsed(),
            input_tokens: None,
            output_tokens: None,
            tokens_per_second: None,
            success: false,
            error: Some(e.to_string()),
        },
    }
}

/// Compare multiple backends with the same prompts
pub async fn compare_backends(
    backends: &[&dyn LlmBackend],
    prompts: &[(&str, Option<&str>)], // (prompt, system)
) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    
    for (prompt, system) in prompts {
        for backend in backends {
            let result = benchmark_backend(*backend, prompt, *system).await;
            results.push(result);
        }
    }
    
    results
}

/// Print benchmark comparison table
pub fn print_comparison(results: &[BenchmarkResult]) {
    println!("\n{:=<80}", "");
    println!("LLM Backend Comparison");
    println!("{:=<80}\n", "");
    
    println!("{:<12} {:<20} {:>10} {:>10} {:>12}", 
        "Provider", "Model", "Latency", "Tokens", "Tok/sec");
    println!("{:-<66}", "");
    
    for result in results {
        let latency = format!("{:.2}s", result.total_latency.as_secs_f64());
        let tokens = result.output_tokens
            .map(|t| t.to_string())
            .unwrap_or_else(|| "-".to_string());
        let tps = result.tokens_per_second
            .map(|t| format!("{:.1}", t))
            .unwrap_or_else(|| "-".to_string());
        
        let status = if result.success { "" } else { " [FAILED]" };
        
        println!("{:<12} {:<20} {:>10} {:>10} {:>12}{}", 
            result.provider.to_string(),
            truncate(&result.model, 20),
            latency,
            tokens,
            tps,
            status,
        );
    }
    println!();
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len-3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_request_builder() {
        let req = GenerateRequest::new("Hello")
            .with_system("You are helpful")
            .with_temperature(0.5)
            .with_max_tokens(100);
        
        assert_eq!(req.prompt, "Hello");
        assert_eq!(req.system, Some("You are helpful".to_string()));
        assert_eq!(req.temperature, 0.5);
        assert_eq!(req.max_tokens, 100);
    }
}
