//! # SMESH Agent
//!
//! LLM agent integration for SMESH protocol.
//!
//! Provides:
//! - Unified LLM backend trait for multiple providers
//! - Ollama client for local LLM inference
//! - Claude/Anthropic client for cloud LLM
//! - Streaming support for real-time responses
//! - Constitutional AI steering via principles
//! - Agent nodes that use LLM for decision-making
//! - Task coordination via SMESH signals

pub mod backend;
pub mod ollama;
pub mod claude;
pub mod streaming;
pub mod constitutional;
pub mod agent;
pub mod coordinator;

// Core backend types
pub use backend::{
    LlmBackend, LlmError, LlmProvider,
    GenerateRequest, GenerateResponse,
    // V2 types for full API support
    GenerateRequestV2, GenerateResponseV2,
    Message, MessageRole, ContentBlock, ToolDefinition, ToolChoice, StopReason,
    Conversation,
    // Streaming types
    StreamEvent, StreamResult, ContentDelta,
    // Benchmarking
    BenchmarkResult, benchmark_backend, compare_backends, print_comparison,
};

// LLM Backends
pub use ollama::{OllamaClient, OllamaConfig};
pub use claude::{ClaudeClient, ClaudeConfig};

// Streaming utilities
pub use streaming::{parse_sse_event, parse_stop_reason, collect_text};

// Constitutional AI
pub use constitutional::{
    ConstitutionalPrinciple, PrinciplePreset,
    security_principles, quality_principles, smesh_principles,
    apply_principles, security_system_prompt, comprehensive_system_prompt,
};

// Agent system
pub use agent::{LlmAgent, AgentConfig, AgentRole, TaskType, AgentTask};
pub use coordinator::{
    AgentCoordinator, CoordinatorConfig, TaskDefinition, CoordinatorResult,
    BackendFactory,
};
