//! Vuln Swarm - Multi-agent vulnerability scanner using SMESH signal coordination
//!
//! This module implements a Claude-focused swarm of 10 specialized vulnerability
//! agents that analyze code and achieve consensus through signal reinforcement.
//!
//! # Architecture
//!
//! - **VulnAgent**: Wrapper around a SMESH Node + ClaudeClient with a specialization
//! - **VulnSpecialization**: 7 vulnerability categories (Injection, XSS, Auth, etc.)
//! - **VulnSwarmCoordinator**: Orchestrates parallel analysis with rate limiting
//! - **SwarmMetrics**: Tracks token usage, consensus timing, and TOON savings
//!
//! # Consensus Model
//!
//! Findings are emitted as SMESH signals. When multiple agents discover the same
//! vulnerability, they reinforce the signal. Consensus is reached when:
//! - 3+ agents reinforce a finding, OR
//! - 2+ agents reinforce with confidence >= 0.8

// Allow dead code for public API items that may be used externally
#![allow(dead_code)]

mod agent;
mod config;
mod coordinator;
mod findings;
mod metrics;

pub use agent::VulnAgent;
pub use config::{OutputFormat, VulnSwarmConfig};
pub use coordinator::{print_results, results_to_json, VulnSwarmCoordinator};
pub use findings::{ConsensusFinding, Severity};
