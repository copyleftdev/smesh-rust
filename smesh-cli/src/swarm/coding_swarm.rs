//! CodingSwarm - Multi-agent collaborative coding via SMESH signal coordination
//!
//! Demonstrates true SMESH concepts:
//! - Signal-based coordination (no direct calls between agents)
//! - Emergent consensus (multiple agents reinforcing decisions)
//! - Trust evolution (based on code acceptance, test results)
//! - Decentralized task claiming (agents sense and respond to signals)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use smesh_agent::{ClaudeClient, ClaudeConfig, GenerateRequestV2, LlmBackend};
use smesh_core::{Field, Node, Signal, SignalType};

/// Agent roles in the coding swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodingRole {
    /// Designs module structure, defines interfaces
    Architect,
    /// Implements code based on specs
    Coder,
    /// Writes and runs tests
    Tester,
    /// Reviews code quality and correctness
    Reviewer,
}

impl CodingRole {
    fn system_prompt(&self) -> &'static str {
        match self {
            CodingRole::Architect => {
                r#"You are a software architect designing a modular Python todo application.

Your responsibilities:
1. Break down the todo app into clear modules (models, storage, api, cli)
2. Define interfaces between modules
3. Emit task specifications for coders

When given a project description, output a MODULE SPEC:
MODULE: <name>
PURPOSE: <one sentence>
INTERFACE:
- <function signature>
- <function signature>
DEPENDENCIES: <comma-separated module names or "none">

Be concise. Focus on clean separation of concerns."#
            }
            CodingRole::Coder => {
                r#"You are a Python developer implementing modules for a todo application.

Your responsibilities:
1. Implement the module according to the specification
2. Follow the interface exactly
3. Write clean, documented code

When given a module spec, output:
```python
# <module_name>.py
<implementation>
```

Keep implementations focused and correct. Use type hints."#
            }
            CodingRole::Tester => {
                r#"You are a test engineer writing tests for a todo application.

Your responsibilities:
1. Write pytest tests for the given module
2. Cover edge cases
3. Report test results

When given code to test, output:
```python
# test_<module_name>.py
<test implementation>
```

Then output:
TEST_RESULT: PASS|FAIL
COVERAGE: <brief summary of what's tested>"#
            }
            CodingRole::Reviewer => {
                r#"You are a code reviewer for a todo application.

Your responsibilities:
1. Check code correctness
2. Verify interface compliance
3. Identify potential issues

When reviewing code, output:
REVIEW: APPROVE|REQUEST_CHANGES
ISSUES:
- <issue or "none">
SUGGESTION: <one improvement or "none">"#
            }
        }
    }

    fn color(&self) -> &'static str {
        match self {
            CodingRole::Architect => "\x1b[95m", // Magenta
            CodingRole::Coder => "\x1b[96m",     // Cyan
            CodingRole::Tester => "\x1b[93m",    // Yellow
            CodingRole::Reviewer => "\x1b[92m",  // Green
        }
    }
}

/// Signal types specific to coding coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodingSignalType {
    /// Architect emits a task for coders
    Task { module: String, spec: String },
    /// Coder claims a task
    Claim { module: String, coder_id: String },
    /// Coder submits implementation
    Code { module: String, code: String },
    /// Reviewer provides feedback
    Review { module: String, approved: bool, feedback: String },
    /// Tester reports results
    TestResult { module: String, passed: bool, details: String },
    /// Consensus reached on a module
    Consensus { module: String, status: String },
}

impl CodingSignalType {
    fn to_payload(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    fn from_payload(payload: &[u8]) -> Option<Self> {
        serde_json::from_slice(payload).ok()
    }

    fn signal_type(&self) -> SignalType {
        match self {
            CodingSignalType::Task { .. } => SignalType::Coordination,
            CodingSignalType::Claim { .. } => SignalType::Response,
            CodingSignalType::Code { .. } => SignalType::Data,
            CodingSignalType::Review { .. } => SignalType::Response,
            CodingSignalType::TestResult { .. } => SignalType::Data,
            CodingSignalType::Consensus { .. } => SignalType::Alert,
        }
    }
}

/// A coding agent in the swarm
pub struct CodingAgent {
    pub node: Node,
    pub role: CodingRole,
    pub client: ClaudeClient,
    pub claimed_tasks: Vec<String>,
    pub completed_tasks: Vec<String>,
    pub stats: AgentStats,
}

#[derive(Debug, Clone, Default)]
pub struct AgentStats {
    pub signals_emitted: u32,
    pub signals_sensed: u32,
    pub tasks_claimed: u32,
    pub tasks_completed: u32,
    pub code_approved: u32,
    pub code_rejected: u32,
    pub tests_passed: u32,
    pub tests_failed: u32,
    pub tokens_used: u64,
}

impl CodingAgent {
    pub fn new(role: CodingRole, client: ClaudeClient) -> Self {
        let mut node = Node::new();
        node.id = format!("{:?}-{}", role, &node.id[..4]);

        Self {
            node,
            role,
            client,
            claimed_tasks: Vec::new(),
            completed_tasks: Vec::new(),
            stats: AgentStats::default(),
        }
    }

    pub fn id(&self) -> &str {
        &self.node.id
    }

    /// Generate a response using Claude
    pub async fn generate(&mut self, prompt: &str) -> Result<String> {
        let system = self.role.system_prompt();
        let request = GenerateRequestV2::simple(prompt)
            .with_system(system);
        let response = self.client.generate_v2(request).await?;
        self.stats.tokens_used += response.input_tokens as u64;
        self.stats.tokens_used += response.output_tokens as u64;
        Ok(response.text())
    }
}

/// Configuration for the coding swarm
#[derive(Debug, Clone)]
pub struct CodingSwarmConfig {
    /// Project description
    pub project_description: String,
    /// Number of coder agents
    pub num_coders: usize,
    /// Claude model to use
    pub model: String,
    /// Consensus threshold (agents that must agree)
    pub consensus_threshold: u32,
    /// Maximum ticks to run
    pub max_ticks: u32,
    /// Tick interval in milliseconds
    pub tick_interval_ms: u64,
}

impl Default for CodingSwarmConfig {
    fn default() -> Self {
        Self {
            project_description: "A modular Python todo application with CLI interface".to_string(),
            num_coders: 2,
            model: "claude-sonnet-4-20250514".to_string(),
            consensus_threshold: 2,
            max_ticks: 50,
            tick_interval_ms: 500,
        }
    }
}

/// Module state tracking
#[derive(Debug, Clone, Default)]
pub struct ModuleState {
    pub name: String,
    pub spec: Option<String>,
    pub claimed_by: Option<String>,
    pub code: Option<String>,
    pub review_approved: bool,
    pub review_count: u32,
    pub tests_passed: bool,
    pub test_count: u32,
    pub consensus_reached: bool,
    pub time_to_consensus: Option<u64>, // milliseconds
}

impl ModuleState {
    fn has_consensus(&self, threshold: u32) -> bool {
        self.review_approved && self.review_count >= threshold
            && self.tests_passed && self.test_count >= 1
    }
}

/// Metrics collected during the swarm run
#[derive(Debug, Clone, Default)]
pub struct SwarmMetrics {
    pub total_signals_emitted: u32,
    pub total_reinforcements: u32,
    pub modules_completed: u32,
    pub total_tokens: u64,
    pub trust_evolution: Vec<TrustSnapshot>,
    pub consensus_times: Vec<u64>,
    pub start_time: Option<Instant>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone)]
pub struct TrustSnapshot {
    pub tick: u32,
    pub trust_scores: HashMap<String, f64>,
}

/// Results from a coding swarm run
#[derive(Debug, Clone)]
pub struct CodingSwarmResult {
    pub modules: Vec<ModuleState>,
    pub metrics: SwarmMetrics,
    pub agent_stats: HashMap<String, AgentStats>,
    pub generated_code: HashMap<String, String>,
}

/// The main swarm coordinator
pub struct CodingSwarmCoordinator {
    pub config: CodingSwarmConfig,
    pub field: Field,
    pub agents: Vec<CodingAgent>,
    pub modules: HashMap<String, ModuleState>,
    pub metrics: SwarmMetrics,
    pub tick_count: u32,
}

impl CodingSwarmCoordinator {
    pub fn new(config: CodingSwarmConfig) -> Result<Self> {
        // Load API key from environment or .env file
        if let Ok(contents) = std::fs::read_to_string(".env") {
            for line in contents.lines() {
                if let Some((key, value)) = line.split_once('=') {
                    if key.trim() == "ANTHROPIC_API_KEY" {
                        std::env::set_var("ANTHROPIC_API_KEY", value.trim());
                    }
                }
            }
        }

        let mut claude_config = ClaudeConfig::from_env()
            .ok_or_else(|| anyhow::anyhow!("ANTHROPIC_API_KEY not set"))?
            .with_model(&config.model);
        claude_config.max_tokens = 2048;

        // Create agents
        let mut agents = Vec::new();

        // One architect
        agents.push(CodingAgent::new(
            CodingRole::Architect,
            ClaudeClient::new(claude_config.clone()),
        ));

        // Multiple coders
        for _ in 0..config.num_coders {
            agents.push(CodingAgent::new(
                CodingRole::Coder,
                ClaudeClient::new(claude_config.clone()),
            ));
        }

        // One tester
        agents.push(CodingAgent::new(
            CodingRole::Tester,
            ClaudeClient::new(claude_config.clone()),
        ));

        // One reviewer
        agents.push(CodingAgent::new(
            CodingRole::Reviewer,
            ClaudeClient::new(claude_config),
        ));

        Ok(Self {
            config,
            field: Field::new(),
            agents,
            modules: HashMap::new(),
            metrics: SwarmMetrics::default(),
            tick_count: 0,
        })
    }

    /// Run the coding swarm
    pub async fn run(&mut self) -> Result<CodingSwarmResult> {
        self.metrics.start_time = Some(Instant::now());
        let reset = "\x1b[0m";

        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║           SMESH Coding Swarm - Signal Coordination        ║");
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        println!("Project: {}", self.config.project_description);
        println!("Agents: {} total", self.agents.len());
        for agent in &self.agents {
            println!("  {} • {}{:?}{} ({})",
                agent.role.color(), agent.role.color(), agent.role, reset, agent.id());
        }
        println!();

        // Phase 1: Architect designs modules
        println!("{}━━━ Phase 1: Architecture Design ━━━{}\n", "\x1b[95m", reset);
        self.phase_architecture().await?;

        // Phase 2: Coders claim and implement tasks
        println!("\n{}━━━ Phase 2: Implementation ━━━{}\n", "\x1b[96m", reset);
        self.phase_implementation().await?;

        // Phase 3: Review and testing cycle
        println!("\n{}━━━ Phase 3: Review & Testing ━━━{}\n", "\x1b[93m", reset);
        self.phase_review_testing().await?;

        // Finalize metrics
        if let Some(start) = self.metrics.start_time {
            self.metrics.elapsed_ms = start.elapsed().as_millis() as u64;
        }

        // Collect results
        let mut agent_stats = HashMap::new();
        let mut total_tokens = 0u64;
        for agent in &self.agents {
            agent_stats.insert(agent.id().to_string(), agent.stats.clone());
            total_tokens += agent.stats.tokens_used;
        }
        self.metrics.total_tokens = total_tokens;

        let generated_code: HashMap<String, String> = self.modules
            .iter()
            .filter_map(|(name, state)| {
                state.code.as_ref().map(|code| (name.clone(), code.clone()))
            })
            .collect();

        self.metrics.modules_completed = self.modules.values()
            .filter(|m| m.consensus_reached)
            .count() as u32;

        Ok(CodingSwarmResult {
            modules: self.modules.values().cloned().collect(),
            metrics: self.metrics.clone(),
            agent_stats,
            generated_code,
        })
    }

    /// Phase 1: Architect designs module structure
    async fn phase_architecture(&mut self) -> Result<()> {
        let reset = "\x1b[0m";
        let architect_idx = self.agents.iter().position(|a| a.role == CodingRole::Architect)
            .ok_or_else(|| anyhow::anyhow!("No architect agent"))?;

        let prompt = format!(
            "Design a modular Python todo application.\n\n\
            Requirements:\n\
            - Todo model with id, title, completed, created_at\n\
            - JSON file storage\n\
            - CLI interface for add, list, complete, delete\n\n\
            Output 3-4 module specs (models, storage, cli). \
            Each module should be self-contained."
        );

        print!("{}ARCHITECT{} designing modules... ", "\x1b[95m", reset);
        std::io::Write::flush(&mut std::io::stdout())?;

        let response = self.agents[architect_idx].generate(&prompt).await?;
        println!("✓");

        // Parse module specs from response
        let modules = self.parse_module_specs(&response);

        for module in &modules {
            println!("  📦 Module: {} - {}", module.0, module.1.lines().next().unwrap_or(""));

            // Emit task signal
            let signal_payload = CodingSignalType::Task {
                module: module.0.clone(),
                spec: module.1.clone(),
            };

            let signal = Signal::builder(signal_payload.signal_type())
                .payload(signal_payload.to_payload())
                .intensity(1.0)
                .confidence(0.9)
                .ttl(120.0)
                .origin(&self.agents[architect_idx].node.id)
                .build();

            self.field.emit_anonymous(signal);
            self.agents[architect_idx].stats.signals_emitted += 1;
            self.metrics.total_signals_emitted += 1;

            // Track module state
            self.modules.insert(module.0.clone(), ModuleState {
                name: module.0.clone(),
                spec: Some(module.1.clone()),
                ..Default::default()
            });
        }

        self.tick_count += 1;
        self.field.tick(1.0);

        Ok(())
    }

    /// Phase 2: Coders claim tasks and implement
    async fn phase_implementation(&mut self) -> Result<()> {
        let reset = "\x1b[0m";

        // Get list of modules that need implementation
        let modules_to_implement: Vec<String> = self.modules.keys().cloned().collect();

        for module_name in modules_to_implement {
            // Find a coder to claim this task
            let coder_idx = self.find_available_coder();
            if coder_idx.is_none() {
                continue;
            }
            let coder_idx = coder_idx.unwrap();

            let spec = match self.modules.get(&module_name) {
                Some(m) => m.spec.clone().unwrap_or_default(),
                None => continue,
            };

            // Emit claim signal
            let coder_id = self.agents[coder_idx].id().to_string();
            let claim_signal = CodingSignalType::Claim {
                module: module_name.clone(),
                coder_id: coder_id.clone(),
            };

            let signal = Signal::builder(claim_signal.signal_type())
                .payload(claim_signal.to_payload())
                .intensity(0.9)
                .confidence(0.85)
                .origin(&coder_id)
                .build();

            self.field.emit_anonymous(signal);
            self.agents[coder_idx].stats.signals_emitted += 1;
            self.agents[coder_idx].claimed_tasks.push(module_name.clone());
            self.metrics.total_signals_emitted += 1;

            // Update module state
            if let Some(state) = self.modules.get_mut(&module_name) {
                state.claimed_by = Some(coder_id.clone());
            }

            print!("{}CODER{} {} implementing {}... ",
                "\x1b[96m", reset, coder_id, module_name);
            std::io::Write::flush(&mut std::io::stdout())?;

            // Generate implementation
            let prompt = format!(
                "Implement this Python module:\n\n{}\n\n\
                Output only the Python code wrapped in ```python```.",
                spec
            );

            let response = self.agents[coder_idx].generate(&prompt).await?;
            println!("✓");

            // Extract code from response
            let code = self.extract_code(&response);

            // Emit code signal
            let code_signal = CodingSignalType::Code {
                module: module_name.clone(),
                code: code.clone(),
            };

            let signal = Signal::builder(code_signal.signal_type())
                .payload(code_signal.to_payload())
                .intensity(0.95)
                .confidence(0.8)
                .origin(&coder_id)
                .build();

            let _hash = self.field.emit_anonymous(signal);
            self.agents[coder_idx].stats.signals_emitted += 1;
            self.agents[coder_idx].stats.tasks_completed += 1;
            self.metrics.total_signals_emitted += 1;

            // Update module state
            if let Some(state) = self.modules.get_mut(&module_name) {
                state.code = Some(code.clone());
            }

            // Show code preview
            let preview: String = code.lines().take(3).collect::<Vec<_>>().join("\n");
            println!("    └─ {}", preview.replace('\n', "\n       "));

            self.tick_count += 1;
            self.field.tick(1.0);

            // Rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        Ok(())
    }

    /// Phase 3: Review and testing cycle
    async fn phase_review_testing(&mut self) -> Result<()> {
        let reset = "\x1b[0m";

        // Get modules with code that need review
        let modules_to_review: Vec<(String, String)> = self.modules
            .iter()
            .filter_map(|(name, state)| {
                state.code.as_ref().map(|code| (name.clone(), code.clone()))
            })
            .collect();

        for (module_name, code) in modules_to_review {
            let start_time = Instant::now();

            // Reviewer reviews the code
            let reviewer_idx = self.agents.iter()
                .position(|a| a.role == CodingRole::Reviewer)
                .ok_or_else(|| anyhow::anyhow!("No reviewer agent"))?;

            let spec = self.modules.get(&module_name)
                .and_then(|m| m.spec.clone())
                .unwrap_or_default();

            print!("{}REVIEWER{} reviewing {}... ", "\x1b[92m", reset, module_name);
            std::io::Write::flush(&mut std::io::stdout())?;

            let prompt = format!(
                "Review this Python module implementation:\n\n\
                SPECIFICATION:\n{}\n\n\
                CODE:\n```python\n{}\n```",
                spec, code
            );

            let response = self.agents[reviewer_idx].generate(&prompt).await?;
            let approved = response.contains("REVIEW: APPROVE");
            println!("{}", if approved { "✓ APPROVED" } else { "✗ CHANGES REQUESTED" });

            // Emit review signal
            let review_signal = CodingSignalType::Review {
                module: module_name.clone(),
                approved,
                feedback: response.clone(),
            };

            let signal = Signal::builder(review_signal.signal_type())
                .payload(review_signal.to_payload())
                .intensity(0.9)
                .confidence(if approved { 0.9 } else { 0.6 })
                .origin(&self.agents[reviewer_idx].node.id)
                .build();

            self.field.emit_anonymous(signal);
            self.agents[reviewer_idx].stats.signals_emitted += 1;
            self.metrics.total_signals_emitted += 1;

            // Update module state and trust
            if let Some(state) = self.modules.get_mut(&module_name) {
                state.review_approved = approved;
                state.review_count += 1;

                // Trust evolution: update trust based on review
                if let Some(coder_id) = &state.claimed_by {
                    let delta = if approved { 0.1 } else { -0.1 };
                    self.agents[reviewer_idx].node.update_trust(coder_id, delta);

                    // Find coder and update their stats
                    for agent in &mut self.agents {
                        if agent.id() == coder_id {
                            if approved {
                                agent.stats.code_approved += 1;
                            } else {
                                agent.stats.code_rejected += 1;
                            }
                        }
                    }
                }
            }

            self.tick_count += 1;
            self.field.tick(1.0);

            // Tester tests the code
            let tester_idx = self.agents.iter()
                .position(|a| a.role == CodingRole::Tester)
                .ok_or_else(|| anyhow::anyhow!("No tester agent"))?;

            print!("{}TESTER{} testing {}... ", "\x1b[93m", reset, module_name);
            std::io::Write::flush(&mut std::io::stdout())?;

            let prompt = format!(
                "Write pytest tests for this module and report if they would pass:\n\n\
                ```python\n{}\n```\n\n\
                Output the tests, then TEST_RESULT: PASS or FAIL",
                code
            );

            let response = self.agents[tester_idx].generate(&prompt).await?;
            let passed = response.contains("TEST_RESULT: PASS");
            println!("{}", if passed { "✓ PASSED" } else { "✗ FAILED" });

            // Emit test result signal
            let test_signal = CodingSignalType::TestResult {
                module: module_name.clone(),
                passed,
                details: response.clone(),
            };

            let signal = Signal::builder(test_signal.signal_type())
                .payload(test_signal.to_payload())
                .intensity(0.9)
                .confidence(if passed { 0.9 } else { 0.5 })
                .origin(&self.agents[tester_idx].node.id)
                .build();

            self.field.emit_anonymous(signal);
            self.agents[tester_idx].stats.signals_emitted += 1;
            self.metrics.total_signals_emitted += 1;

            // Update module state and trust
            if let Some(state) = self.modules.get_mut(&module_name) {
                state.tests_passed = passed;
                state.test_count += 1;

                // Trust evolution based on test results
                if let Some(coder_id) = &state.claimed_by {
                    let delta = if passed { 0.1 } else { -0.15 };
                    self.agents[tester_idx].node.update_trust(coder_id, delta);

                    // Update coder stats
                    for agent in &mut self.agents {
                        if agent.id() == coder_id {
                            if passed {
                                agent.stats.tests_passed += 1;
                            } else {
                                agent.stats.tests_failed += 1;
                            }
                        }
                    }
                }

                // Check for consensus
                if state.has_consensus(self.config.consensus_threshold) {
                    state.consensus_reached = true;
                    state.time_to_consensus = Some(start_time.elapsed().as_millis() as u64);
                    self.metrics.consensus_times.push(state.time_to_consensus.unwrap());

                    // Emit consensus signal
                    let consensus_signal = CodingSignalType::Consensus {
                        module: module_name.clone(),
                        status: "COMPLETE".to_string(),
                    };

                    let signal = Signal::builder(consensus_signal.signal_type())
                        .payload(consensus_signal.to_payload())
                        .intensity(1.0)
                        .confidence(0.95)
                        .build();

                    self.field.emit_anonymous(signal);
                    self.metrics.total_signals_emitted += 1;

                    println!("    └─ {}✓ CONSENSUS REACHED{} ({}ms)",
                        "\x1b[92m", reset, state.time_to_consensus.unwrap());
                }
            }

            // Record trust snapshot
            let mut trust_snapshot = HashMap::new();
            for agent in &self.agents {
                for (other_id, &trust) in &agent.node.trust_scores {
                    trust_snapshot.insert(
                        format!("{}→{}", agent.id(), other_id),
                        trust,
                    );
                }
            }
            self.metrics.trust_evolution.push(TrustSnapshot {
                tick: self.tick_count,
                trust_scores: trust_snapshot,
            });

            self.tick_count += 1;
            self.field.tick(1.0);

            // Rate limiting
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        // Count total reinforcements from field
        self.metrics.total_reinforcements = self.field.stats().total_reinforcements;

        Ok(())
    }

    fn find_available_coder(&self) -> Option<usize> {
        self.agents.iter()
            .position(|a| a.role == CodingRole::Coder && a.claimed_tasks.len() < 2)
    }

    fn parse_module_specs(&self, response: &str) -> Vec<(String, String)> {
        let mut modules = Vec::new();
        let mut current_module = String::new();
        let mut current_spec = String::new();
        let mut in_module = false;

        for line in response.lines() {
            if line.starts_with("MODULE:") {
                if in_module && !current_module.is_empty() {
                    modules.push((current_module.clone(), current_spec.clone()));
                }
                current_module = line.replace("MODULE:", "").trim().to_string();
                current_spec = String::new();
                in_module = true;
            } else if in_module {
                current_spec.push_str(line);
                current_spec.push('\n');
            }
        }

        if in_module && !current_module.is_empty() {
            modules.push((current_module, current_spec));
        }

        // If no modules parsed, create defaults
        if modules.is_empty() {
            modules = vec![
                ("models".to_string(), "Todo model with id, title, completed, created_at".to_string()),
                ("storage".to_string(), "JSON file storage for todos".to_string()),
                ("cli".to_string(), "CLI interface for add, list, complete, delete".to_string()),
            ];
        }

        modules
    }

    fn extract_code(&self, response: &str) -> String {
        let mut code = String::new();
        let mut in_code_block = false;

        for line in response.lines() {
            if line.starts_with("```python") {
                in_code_block = true;
                continue;
            }
            if line.starts_with("```") && in_code_block {
                break;
            }
            if in_code_block {
                code.push_str(line);
                code.push('\n');
            }
        }

        if code.is_empty() {
            // Return the whole response if no code block found
            response.to_string()
        } else {
            code
        }
    }
}

/// Print results to console
pub fn print_coding_results(result: &CodingSwarmResult) {
    let reset = "\x1b[0m";
    let cyan = "\x1b[96m";
    let green = "\x1b[92m";
    let yellow = "\x1b[93m";

    println!("\n{}═══════════════════════════════════════════════════════════════{}", cyan, reset);
    println!("{}                    CODING SWARM REPORT                        {}", cyan, reset);
    println!("{}═══════════════════════════════════════════════════════════════{}\n", cyan, reset);

    // Summary
    println!("{}Summary:{}", green, reset);
    println!("  Duration: {:.1}s", result.metrics.elapsed_ms as f64 / 1000.0);
    println!("  Modules: {} completed / {} total",
        result.metrics.modules_completed,
        result.modules.len());
    println!("  Total signals: {}", result.metrics.total_signals_emitted);
    println!("  Reinforcements: {}", result.metrics.total_reinforcements);
    println!("  Total tokens: {}", result.metrics.total_tokens);

    // Consensus timing
    if !result.metrics.consensus_times.is_empty() {
        let avg_time: f64 = result.metrics.consensus_times.iter()
            .map(|&t| t as f64)
            .sum::<f64>() / result.metrics.consensus_times.len() as f64;
        println!("  Avg time to consensus: {:.0}ms", avg_time);
    }

    // Module status
    println!("\n{}Modules:{}", yellow, reset);
    for module in &result.modules {
        let status = if module.consensus_reached {
            format!("{}✓ COMPLETE{}", green, reset)
        } else if module.code.is_some() {
            format!("{}◐ IN PROGRESS{}", yellow, reset)
        } else {
            format!("○ PENDING")
        };

        println!("  {} {} (reviews: {}, tests: {})",
            status, module.name, module.review_count, module.test_count);
    }

    // Agent performance
    println!("\n{}Agent Performance:{}", cyan, reset);
    for (agent_id, stats) in &result.agent_stats {
        println!("  {}:", agent_id);
        println!("    Signals: {} emitted", stats.signals_emitted);
        if stats.code_approved > 0 || stats.code_rejected > 0 {
            println!("    Code: {} approved, {} rejected",
                stats.code_approved, stats.code_rejected);
        }
        if stats.tests_passed > 0 || stats.tests_failed > 0 {
            println!("    Tests: {} passed, {} failed",
                stats.tests_passed, stats.tests_failed);
        }
        println!("    Tokens: {}", stats.tokens_used);
    }

    // Trust evolution
    if !result.metrics.trust_evolution.is_empty() {
        println!("\n{}Trust Evolution:{}", green, reset);
        if let Some(final_snapshot) = result.metrics.trust_evolution.last() {
            for (pair, trust) in &final_snapshot.trust_scores {
                let bar_len = (trust * 10.0) as usize;
                let bar = "█".repeat(bar_len);
                println!("  {} [{:<10}] {:.2}", pair, bar, trust);
            }
        }
    }

    println!("\n{}═══════════════════════════════════════════════════════════════{}", cyan, reset);
    println!("{}🌿 Swarm coordination complete via SMESH signal diffusion{}", green, reset);
    println!("{}═══════════════════════════════════════════════════════════════{}\n", cyan, reset);
}

/// Convert results to JSON
pub fn results_to_json(result: &CodingSwarmResult) -> String {
    serde_json::to_string_pretty(&serde_json::json!({
        "summary": {
            "elapsed_ms": result.metrics.elapsed_ms,
            "modules_completed": result.metrics.modules_completed,
            "total_signals": result.metrics.total_signals_emitted,
            "total_reinforcements": result.metrics.total_reinforcements,
            "total_tokens": result.metrics.total_tokens,
        },
        "modules": result.modules.iter().map(|m| {
            serde_json::json!({
                "name": m.name,
                "consensus_reached": m.consensus_reached,
                "review_count": m.review_count,
                "test_count": m.test_count,
                "time_to_consensus_ms": m.time_to_consensus,
            })
        }).collect::<Vec<_>>(),
        "generated_code": result.generated_code,
    })).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_consensus() {
        let state = ModuleState {
            name: "test".to_string(),
            spec: Some("test spec".to_string()),
            code: Some("def test(): pass".to_string()),
            review_approved: true,
            review_count: 2,
            tests_passed: true,
            test_count: 1,
            ..Default::default()
        };

        assert!(state.has_consensus(2));
        assert!(!state.has_consensus(3));
    }

    #[test]
    fn test_coding_signal_serialization() {
        let signal = CodingSignalType::Task {
            module: "models".to_string(),
            spec: "Todo model".to_string(),
        };

        let payload = signal.to_payload();
        let decoded = CodingSignalType::from_payload(&payload);

        assert!(decoded.is_some());
    }
}
