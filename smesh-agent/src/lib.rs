//! # SMESH Agent
//!
//! LLM agent integration for SMESH protocol.
//!
//! Provides:
//! - Unified LLM backend trait for multiple providers
//! - Ollama client for local LLM inference
//! - Claude/Anthropic client for cloud LLM
//! - Agent nodes that use LLM for decision-making
//! - Task coordination via SMESH signals

pub mod backend;
pub mod ollama;
pub mod claude;
pub mod agent;
pub mod coordinator;

pub use backend::{
    LlmBackend, LlmError, LlmProvider,
    GenerateRequest, GenerateResponse,
    BenchmarkResult, benchmark_backend, compare_backends, print_comparison,
};
pub use ollama::{OllamaClient, OllamaConfig};
pub use claude::{ClaudeClient, ClaudeConfig};
pub use agent::{LlmAgent, AgentConfig, AgentRole, TaskType, AgentTask};
pub use coordinator::{AgentCoordinator, CoordinatorConfig, TaskDefinition, CoordinatorResult};
