//! Unified LLM backend trait for multiple providers

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
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

    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
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

// ============================================================================
// Multi-turn Message Types
// ============================================================================

/// Role of a message in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
}

/// A message in a multi-turn conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Create a simple text message from a user
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Create a simple text message from the assistant
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    /// Create a message with a tool result
    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
                is_error,
            }],
        }
    }

    /// Get text content if this is a simple text message
    pub fn text(&self) -> Option<&str> {
        if self.content.len() == 1 {
            if let ContentBlock::Text { text } = &self.content[0] {
                return Some(text);
            }
        }
        None
    }
}

// ============================================================================
// Tool Use Types
// ============================================================================

/// Definition of a tool the model can use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (must match function call pattern)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// JSON Schema for the input parameters
    pub input_schema: Value,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Content block in a message (text, tool use, or tool result)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content
    Text { text: String },
    /// Tool use request from the model
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// Result of a tool execution
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

impl ContentBlock {
    /// Check if this is a text block
    pub fn is_text(&self) -> bool {
        matches!(self, ContentBlock::Text { .. })
    }

    /// Check if this is a tool use block
    pub fn is_tool_use(&self) -> bool {
        matches!(self, ContentBlock::ToolUse { .. })
    }

    /// Get text content if this is a text block
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            _ => None,
        }
    }
}

/// How the model should choose which tool to use
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    /// Model decides whether to use a tool
    #[default]
    Auto,
    /// Model must use a tool
    Any,
    /// Model cannot use tools
    None,
    /// Model must use the specified tool
    Tool { name: String },
}

/// Reason the model stopped generating
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Normal end of response
    EndTurn,
    /// Hit max tokens limit
    MaxTokens,
    /// Hit a stop sequence
    StopSequence,
    /// Model wants to use a tool
    ToolUse,
}

// ============================================================================
// Streaming Types
// ============================================================================

/// Events emitted during streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Message generation started
    MessageStart { message: StreamMessageStart },
    /// A content block started
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    /// Delta for a content block
    ContentBlockDelta { index: usize, delta: ContentDelta },
    /// A content block finished
    ContentBlockStop { index: usize },
    /// Message-level delta (stop reason, usage)
    MessageDelta {
        delta: MessageDeltaData,
        usage: Option<StreamUsage>,
    },
    /// Message generation complete
    MessageStop,
    /// Error during streaming
    Error { error: StreamError },
    /// Ping to keep connection alive
    Ping,
}

/// Initial message data in MessageStart event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMessageStart {
    pub id: String,
    pub model: String,
    pub role: String,
}

/// Delta content during streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// Text being generated
    TextDelta { text: String },
    /// Partial JSON for tool input
    InputJsonDelta { partial_json: String },
}

/// Message-level delta data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaData {
    pub stop_reason: Option<StopReason>,
}

/// Usage info during streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamUsage {
    pub output_tokens: u32,
}

/// Error during streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamError {
    pub message: String,
}

// ============================================================================
// Extended Request/Response Types
// ============================================================================

/// Extended generation request with full Claude API support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequestV2 {
    /// Conversation messages
    pub messages: Vec<Message>,
    /// System prompt
    pub system: Option<String>,
    /// Available tools
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    /// Tool choice strategy
    #[serde(default)]
    pub tool_choice: ToolChoice,
    /// Enable streaming
    #[serde(default)]
    pub stream: bool,
    /// Temperature (0.0 - 1.0)
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Stop sequences
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

impl GenerateRequestV2 {
    /// Create a new request from messages
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            system: None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            stream: false,
            temperature: 0.7,
            max_tokens: 4096,
            stop_sequences: Vec::new(),
        }
    }

    /// Create a simple single-turn request
    pub fn simple(prompt: impl Into<String>) -> Self {
        Self::new(vec![Message::user(prompt)])
    }

    /// Add system prompt
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Add tools
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set tool choice
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Enable streaming
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }
}

/// Extended generation response with full Claude API support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponseV2 {
    /// Response content blocks
    pub content: Vec<ContentBlock>,
    /// Why generation stopped
    pub stop_reason: StopReason,
    /// Model that generated the response
    pub model: String,
    /// Request ID
    pub id: String,
    /// Input tokens used
    pub input_tokens: u32,
    /// Output tokens generated
    pub output_tokens: u32,
    /// Time taken for generation
    #[serde(skip)]
    pub latency: Duration,
}

impl GenerateResponseV2 {
    /// Get all text content concatenated
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get all tool use blocks
    pub fn tool_uses(&self) -> Vec<(&str, &str, &Value)> {
        self.content
            .iter()
            .filter_map(|c| match c {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.as_str(), name.as_str(), input))
                }
                _ => None,
            })
            .collect()
    }

    /// Check if the model wants to use tools
    pub fn wants_tool_use(&self) -> bool {
        self.stop_reason == StopReason::ToolUse
    }
}

/// Type alias for streaming response
pub type StreamResult = Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>;

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
        self.output_tokens
            .map(|tokens| tokens as f64 / self.latency.as_secs_f64())
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

    // ========================================================================
    // Extended API (V2) - Tool use, multi-turn, streaming
    // ========================================================================

    /// Generate with full V2 API support (multi-turn, tools)
    async fn generate_v2(
        &self,
        _request: GenerateRequestV2,
    ) -> Result<GenerateResponseV2, LlmError> {
        Err(LlmError::UnsupportedFeature(
            "generate_v2 not supported by this backend".to_string(),
        ))
    }

    /// Generate with streaming support
    async fn generate_stream(&self, _request: GenerateRequestV2) -> Result<StreamResult, LlmError> {
        Err(LlmError::UnsupportedFeature(
            "streaming not supported by this backend".to_string(),
        ))
    }

    /// Check if this backend supports tool use
    fn supports_tools(&self) -> bool {
        false
    }

    /// Check if this backend supports streaming
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Check if this backend supports multi-turn conversations
    fn supports_multi_turn(&self) -> bool {
        false
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

    println!(
        "{:<12} {:<20} {:>10} {:>10} {:>12}",
        "Provider", "Model", "Latency", "Tokens", "Tok/sec"
    );
    println!("{:-<66}", "");

    for result in results {
        let latency = format!("{:.2}s", result.total_latency.as_secs_f64());
        let tokens = result
            .output_tokens
            .map(|t| t.to_string())
            .unwrap_or_else(|| "-".to_string());
        let tps = result
            .tokens_per_second
            .map(|t| format!("{:.1}", t))
            .unwrap_or_else(|| "-".to_string());

        let status = if result.success { "" } else { " [FAILED]" };

        println!(
            "{:<12} {:<20} {:>10} {:>10} {:>12}{}",
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
        format!("{}...", &s[..max_len - 3])
    }
}

// ============================================================================
// Conversation Manager
// ============================================================================

/// Manages a multi-turn conversation with tool support
#[derive(Debug, Clone)]
pub struct Conversation {
    /// Messages in the conversation
    pub messages: Vec<Message>,
    /// System prompt
    pub system: Option<String>,
    /// Available tools
    pub tools: Vec<ToolDefinition>,
}

impl Conversation {
    /// Create a new empty conversation
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            system: None,
            tools: Vec::new(),
        }
    }

    /// Create a conversation with a system prompt
    pub fn with_system(system: impl Into<String>) -> Self {
        Self {
            messages: Vec::new(),
            system: Some(system.into()),
            tools: Vec::new(),
        }
    }

    /// Add tools to the conversation
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Add a user message
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(Message::user(content));
    }

    /// Add an assistant message
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(Message::assistant(content));
    }

    /// Add a tool result
    pub fn add_tool_result(
        &mut self,
        tool_use_id: impl Into<String>,
        result: impl Into<String>,
        is_error: bool,
    ) {
        self.messages
            .push(Message::tool_result(tool_use_id, result, is_error));
    }

    /// Add a raw message
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Convert to a V2 request
    pub fn to_request(&self) -> GenerateRequestV2 {
        GenerateRequestV2 {
            messages: self.messages.clone(),
            system: self.system.clone(),
            tools: self.tools.clone(),
            tool_choice: if self.tools.is_empty() {
                ToolChoice::None
            } else {
                ToolChoice::Auto
            },
            stream: false,
            temperature: 0.7,
            max_tokens: 4096,
            stop_sequences: Vec::new(),
        }
    }

    /// Clear all messages but keep system prompt and tools
    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }

    /// Get the last assistant message text
    pub fn last_assistant_text(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::Assistant)
            .and_then(|m| m.text())
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn test_message_creation() {
        let user_msg = Message::user("Hello");
        assert_eq!(user_msg.role, MessageRole::User);
        assert_eq!(user_msg.text(), Some("Hello"));

        let assistant_msg = Message::assistant("Hi there!");
        assert_eq!(assistant_msg.role, MessageRole::Assistant);
        assert_eq!(assistant_msg.text(), Some("Hi there!"));
    }

    #[test]
    fn test_tool_result_message() {
        let msg = Message::tool_result("tool-123", "result data", false);
        assert_eq!(msg.role, MessageRole::User);
        assert!(
            matches!(&msg.content[0], ContentBlock::ToolResult { tool_use_id, is_error, .. }
            if tool_use_id == "tool-123" && !is_error)
        );
    }

    #[test]
    fn test_content_block_methods() {
        let text_block = ContentBlock::Text {
            text: "hello".to_string(),
        };
        assert!(text_block.is_text());
        assert!(!text_block.is_tool_use());
        assert_eq!(text_block.as_text(), Some("hello"));

        let tool_block = ContentBlock::ToolUse {
            id: "id".to_string(),
            name: "tool".to_string(),
            input: serde_json::json!({}),
        };
        assert!(!tool_block.is_text());
        assert!(tool_block.is_tool_use());
    }

    #[test]
    fn test_generate_request_v2() {
        let req = GenerateRequestV2::simple("Hello")
            .with_system("Be helpful")
            .with_temperature(0.5)
            .with_max_tokens(1000)
            .with_stream(true);

        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.system, Some("Be helpful".to_string()));
        assert_eq!(req.temperature, 0.5);
        assert_eq!(req.max_tokens, 1000);
        assert!(req.stream);
    }

    #[test]
    fn test_conversation() {
        let mut conv = Conversation::with_system("You are helpful");
        conv.add_user_message("Hello");
        conv.add_assistant_message("Hi!");
        conv.add_user_message("How are you?");

        assert_eq!(conv.messages.len(), 3);
        assert_eq!(conv.system, Some("You are helpful".to_string()));
        assert_eq!(conv.last_assistant_text(), Some("Hi!"));

        let request = conv.to_request();
        assert_eq!(request.messages.len(), 3);
    }

    #[test]
    fn test_tool_definition() {
        let tool = ToolDefinition::new(
            "get_weather",
            "Get the weather for a location",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }),
        );

        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.description, "Get the weather for a location");
    }

    #[test]
    fn test_content_block_serialization() {
        let text = ContentBlock::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("\"type\":\"text\""));

        let tool_use = ContentBlock::ToolUse {
            id: "123".to_string(),
            name: "test".to_string(),
            input: serde_json::json!({}),
        };
        let json = serde_json::to_string(&tool_use).unwrap();
        assert!(json.contains("\"type\":\"tool_use\""));
    }
}
