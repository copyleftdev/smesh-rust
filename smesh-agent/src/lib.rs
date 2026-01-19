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

pub mod agent;
pub mod backend;
pub mod claude;
pub mod constitutional;
pub mod coordinator;
pub mod ollama;
pub mod streaming;

// Core backend types
pub use backend::{
    benchmark_backend,
    compare_backends,
    print_comparison,
    // Benchmarking
    BenchmarkResult,
    ContentBlock,
    ContentDelta,
    Conversation,
    GenerateRequest,
    // V2 types for full API support
    GenerateRequestV2,
    GenerateResponse,
    GenerateResponseV2,
    LlmBackend,
    LlmError,
    LlmProvider,
    Message,
    MessageRole,
    StopReason,
    // Streaming types
    StreamEvent,
    StreamResult,
    ToolChoice,
    ToolDefinition,
};

// LLM Backends
pub use claude::{ClaudeClient, ClaudeConfig};
pub use ollama::{OllamaClient, OllamaConfig};

// Streaming utilities
pub use streaming::{collect_text, parse_sse_event, parse_stop_reason};

// Constitutional AI
pub use constitutional::{
    apply_principles, comprehensive_system_prompt, quality_principles, security_principles,
    security_system_prompt, smesh_principles, ConstitutionalPrinciple, PrinciplePreset,
};

// Agent system
pub use agent::{AgentConfig, AgentRole, AgentTask, LlmAgent, TaskType};
pub use coordinator::{
    AgentCoordinator, BackendFactory, CoordinatorConfig, CoordinatorResult, TaskDefinition,
};
