//! # SMESH Bounty
//!
//! Security bounty hunting swarm combining SMESH signal-based coordination
//! with agentic tool execution.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────┐  signals  ┌──────────┐
//! │ Recon    │◄────────►│ Source   │
//! │ Scout    │           │ Auditor  │
//! └────┬─────┘           └────┬─────┘
//!      │                      │
//! SMESH Field (signal diffusion, trust, decay)
//!      │                      │
//! ┌────┴─────┐           ┌────┴─────┐
//! │ Triager  │◄────────►│ Report   │
//! │          │           │ Writer   │
//! └──────────┘           └──────────┘
//! ```
//!
//! Each agent wraps:
//! - **SMESH Node** for coordination (signals, trust, reputation)
//! - **LLM Backend** for intelligence (Claude API with tool use)
//! - **ToolExecutor** for action (file I/O, bash, grep, web fetch)
//!
//! ## Quick Start
//!
//! ```no_run
//! use smesh_bounty::{BountyConfig, BountyCoordinator};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = BountyConfig::new("./target-repo");
//!     let mut coordinator = BountyCoordinator::new(config).unwrap();
//!     let result = coordinator.run().await.unwrap();
//!     smesh_bounty::print_results(&result);
//! }
//! ```

pub mod analyst;
pub mod coordinator;
pub mod hunter;
pub mod specialization;
pub mod tools;
pub mod web_deep;
pub mod web_recon;
pub mod web_swarm;

pub use coordinator::{
    BountyConfig, BountyCoordinator, BountyResult, ConsensusFinding, OutputFormat,
    print_results, results_to_json,
};
pub use hunter::{BountyHunter, HunterConfig, HunterFinding, HunterMetrics};
pub use specialization::BountySpecialization;
pub use tools::{
    BashTool, FileReadTool, GlobTool, GrepTool, SandboxConfig, Tool, ToolError, ToolExecutor,
    ToolResult, TrustLevel, WebFetchTool,
};
pub use web_recon::{run_web_redteam, Arsenal, WebRedTeamResult};
pub use web_deep::WebFinding;
pub use web_swarm::{run_full_spectrum, CorrelatedFinding, FullSpectrumConfig, FullSpectrumResult};
pub use analyst::{analyze_exploitability, generate_report, AnalyzedFinding};
