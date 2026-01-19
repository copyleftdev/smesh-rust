//! Swarm modules - Multi-agent coordination using SMESH signal diffusion
//!
//! This module contains different swarm implementations that demonstrate
//! SMESH's core concepts: signal-based coordination, emergent consensus,
//! and trust evolution.
//!
//! # Swarm Types
//!
//! - **CodingSwarm**: Multi-agent collaborative coding (ARCHITECT, CODER, TESTER, REVIEWER)
//! - **VulnSwarm**: Vulnerability scanning with consensus-based finding validation
//!
//! # SMESH Concepts Demonstrated
//!
//! - **Signal-based coordination**: Agents communicate via signals, not direct calls
//! - **Emergent consensus**: Multiple agents reinforcing signals to reach agreement
//! - **Trust evolution**: Nodes update trust based on outcomes (code accepted, tests pass)
//! - **Decentralized task claiming**: Agents sense and respond to signals

// Allow dead code for public API items that may be used externally
#![allow(dead_code)]

mod agent;
mod coding_swarm;
mod config;
mod coordinator;
mod findings;
mod metrics;

// Vuln Swarm exports
pub use agent::VulnAgent;
pub use config::{OutputFormat, VulnSwarmConfig};
pub use coordinator::{print_results, results_to_json, VulnSwarmCoordinator};
pub use findings::{ConsensusFinding, Severity};

// Coding Swarm exports
pub use coding_swarm::{
    CodingSwarmConfig, CodingSwarmCoordinator,
    print_coding_results, results_to_json as coding_results_to_json,
};
