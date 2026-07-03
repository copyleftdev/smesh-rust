//! Agent coordinator - manages multiple LLM agents via SMESH
//!
//! Provides backend factory pattern for creating agents with different
//! LLM backends (OpenRouter, Claude, etc.).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::agent::{
    AgentAction, AgentConfig, AgentRole, AgentTask, LlmAgent, TaskStatus, TaskType,
};
use crate::backend::LlmBackend;
use crate::claude::{ClaudeClient, ClaudeConfig};
use crate::openrouter::{OpenRouterClient, OpenRouterConfig};
use smesh_core::{Field, NodeId, Signal, SignalType};

/// Factory function type for creating backends
pub type BackendFactory = Box<dyn Fn() -> Arc<dyn LlmBackend> + Send + Sync>;

/// Configuration for the agent coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Number of agents to create
    pub n_agents: usize,
    /// Model to use for all agents
    pub model: String,
    /// Agent roles to use (cycled if fewer than n_agents)
    pub roles: Vec<AgentRole>,
    /// Maximum ticks to run
    pub max_ticks: u64,
    /// Tick interval in milliseconds
    pub tick_interval_ms: u64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            n_agents: 3,
            model: crate::openrouter::DEFAULT_MODEL.to_string(),
            roles: vec![AgentRole::Coder, AgentRole::Reviewer, AgentRole::Analyst],
            max_ticks: 100,
            tick_interval_ms: 100,
        }
    }
}

/// Task definition for the coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDefinition {
    pub task_type: String,
    pub description: String,
    pub priority: f64,
}

/// Result of coordinator execution
#[derive(Debug, Clone)]
pub struct CoordinatorResult {
    pub tasks_total: usize,
    pub tasks_completed: usize,
    pub total_llm_calls: u64,
    pub elapsed_secs: f64,
    pub agent_results: Vec<(String, Vec<AgentTask>)>,
}

/// Coordinates multiple LLM agents using SMESH signals
pub struct AgentCoordinator {
    /// Configuration
    config: CoordinatorConfig,
    /// Agents by node ID
    agents: HashMap<NodeId, LlmAgent>,
    /// Signal field for coordination
    field: Field,
    /// Tasks by ID
    tasks: HashMap<String, TaskDefinition>,
    /// Completed task IDs
    completed: Vec<String>,
    /// Current tick
    tick: u64,
}

impl AgentCoordinator {
    /// Create a new coordinator with a custom backend factory
    pub fn new(config: CoordinatorConfig, backend_factory: BackendFactory) -> Self {
        let mut agents = HashMap::new();

        // Create agents
        for i in 0..config.n_agents {
            let role = config.roles[i % config.roles.len()];
            let name = format!("{:?}-{}", role, i + 1);

            let agent_config = AgentConfig {
                name: name.clone(),
                role,
                openrouter: None,
                ..Default::default()
            };

            let backend = backend_factory();
            let agent = LlmAgent::new(agent_config, backend);
            let node_id = agent.node_id().to_string();

            info!("Created agent: {} ({})", name, node_id);
            agents.insert(node_id, agent);
        }

        Self {
            config,
            agents,
            field: Field::new(),
            tasks: HashMap::new(),
            completed: Vec::new(),
            tick: 0,
        }
    }

    /// Create a coordinator with an OpenRouter backend.
    ///
    /// Credentials are resolved from the environment / credentials file; the
    /// given model slug overrides the configured default.
    pub fn with_openrouter(config: CoordinatorConfig, model: &str) -> Self {
        let model = model.to_string();
        let factory: BackendFactory = Box::new(move || {
            let or_config = OpenRouterConfig::from_env()
                .unwrap_or_default()
                .with_model(model.clone());
            Arc::new(OpenRouterClient::new(or_config))
        });
        Self::new(config, factory)
    }

    /// Create a coordinator with Claude backend
    pub fn with_claude(config: CoordinatorConfig, claude_config: ClaudeConfig) -> Self {
        let factory: BackendFactory =
            Box::new(move || Arc::new(ClaudeClient::new(claude_config.clone())));
        Self::new(config, factory)
    }

    /// Add a task to be coordinated.
    ///
    /// Emits a `task_available` signal into the shared field. Agents sense it
    /// independently and decide whether to claim — there is no central dispatch.
    pub fn add_task(&mut self, task: TaskDefinition) -> String {
        let task_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

        // The payload must be JSON with an `agent_signal_type` discriminator so
        // that `LlmAgent::process_signal` can parse and act on it. (The previous
        // TOON encoding was unparseable by the agent, so no task was ever
        // claimed.)
        let payload = serde_json::json!({
            "agent_signal_type": "task_available",
            "task_id": task_id,
            "task_type": task.task_type,
            "priority": task.priority,
        });

        let signal = Signal::builder(SignalType::Coordination)
            .payload(serde_json::to_vec(&payload).unwrap_or_default())
            .intensity(0.5 + task.priority * 0.5)
            .ttl(120.0)
            .build();

        self.field.emit_anonymous(signal);
        self.tasks.insert(task_id.clone(), task);

        debug!("Added task: {}", task_id);
        task_id
    }

    /// Emit (or refresh) this agent's claim on a task as a signal in the shared
    /// field, so competing agents can sense it and decide whether to yield.
    fn emit_claim_signal(&mut self, task_id: &str, claimer: &str, affinity: f64, task_type: &str) {
        let payload = serde_json::json!({
            "agent_signal_type": "task_claimed",
            "task_id": task_id,
            "claimer": claimer,
            "affinity": affinity,
            "task_type": task_type,
        });

        let signal = Signal::builder(SignalType::Coordination)
            .payload(serde_json::to_vec(&payload).unwrap_or_default())
            .intensity(affinity)
            .confidence(affinity)
            .ttl(120.0)
            .origin(claimer)
            .build();

        self.field.emit_anonymous(signal);
    }

    /// Withdraw an agent's claim on a task by removing its claim signal from the
    /// field (the agent backed off in favour of a stronger claimant).
    fn withdraw_claim_signal(&mut self, task_id: &str, claimer: &str) {
        let to_remove: Vec<String> = self
            .field
            .signals
            .iter()
            .filter(|(_, s)| {
                parse_claim(s)
                    .map(|(tid, c, _)| tid == task_id && c == claimer)
                    .unwrap_or(false)
            })
            .map(|(hash, _)| hash.clone())
            .collect();

        for hash in to_remove {
            self.field.signals.remove(&hash);
        }
    }

    /// Run one decentralized coordination round.
    ///
    /// Every agent independently senses the shared field and reacts with purely
    /// local rules: claim a task it has affinity for, or back off from a task
    /// whose field carries a stronger competing claim. Each decision is written
    /// back into the field as a `task_claimed` signal (or the withdrawal of
    /// one). No agent — and not the coordinator — has a global view that ranks
    /// the claimants. Returns `true` if the field changed this round.
    ///
    /// Because weaker claimants withdraw their signals, repeated rounds converge
    /// on exactly one surviving claim per task: consensus *emerges* from local
    /// back-off rather than a central `max`.
    pub fn coordination_round(&mut self) -> bool {
        // Snapshot so every agent senses the same field state this round.
        let signals: Vec<Signal> = self.field.signals.values().cloned().collect();

        // Each agent decides locally first (mutating only its own state).
        let mut claims: Vec<(NodeId, String, f64)> = Vec::new();
        let mut backoffs: Vec<(NodeId, String)> = Vec::new();

        for (node_id, agent) in &mut self.agents {
            for action in agent.process_signals(&signals) {
                match action {
                    AgentAction::ClaimTask { task_id, affinity } => {
                        claims.push((node_id.clone(), task_id, affinity));
                    }
                    AgentAction::BackOff { task_id } => {
                        backoffs.push((node_id.clone(), task_id));
                    }
                    AgentAction::CompleteTask { .. } => {}
                }
            }
        }

        let changed = !claims.is_empty() || !backoffs.is_empty();

        // Apply decisions to the shared field so they are sensable next round.
        for (node_id, task_id, affinity) in claims {
            let task_type = self
                .tasks
                .get(&task_id)
                .map(|t| t.task_type.clone())
                .unwrap_or_default();
            info!(
                "{} claims {} (affinity {:.0}%)",
                node_id,
                task_id,
                affinity * 100.0
            );
            self.emit_claim_signal(&task_id, &node_id, affinity, &task_type);
        }
        for (node_id, task_id) in backoffs {
            debug!("{} backs off {}", node_id, task_id);
            self.withdraw_claim_signal(&task_id, &node_id);
        }

        changed
    }

    /// Read the emergent assignment from the settled field: for each task, the
    /// agent whose claim signal survived the back-off process.
    ///
    /// Decentralized back-off does the winnowing — normally a single claim per
    /// task remains here. If two claims survive (their affinities were within
    /// the back-off margin, so neither node yielded), the residual tie is broken
    /// deterministically by higher affinity then node id. That tie-break only
    /// ever ranks co-survivors, never the full claimant set.
    pub fn converged_winners(&self) -> Vec<(NodeId, String)> {
        let mut winners: HashMap<String, (NodeId, f64)> = HashMap::new();

        for signal in self.field.signals.values() {
            if let Some((task_id, claimer, affinity)) = parse_claim(signal) {
                winners
                    .entry(task_id)
                    .and_modify(|best| {
                        if affinity > best.1 || (affinity == best.1 && claimer < best.0) {
                            *best = (claimer.clone(), affinity);
                        }
                    })
                    .or_insert((claimer, affinity));
            }
        }

        winners
            .into_iter()
            .map(|(task_id, (node_id, _))| (node_id, task_id))
            .collect()
    }

    /// Execute a task on its assigned agent via the LLM backend.
    /// Returns true if the task completed successfully.
    async fn assign_and_execute(&mut self, node_id: &str, task_id: &str) -> bool {
        let task_def = match self.tasks.get(task_id) {
            Some(t) => t.clone(),
            None => return false,
        };
        let agent = match self.agents.get_mut(node_id) {
            Some(a) => a,
            None => return false,
        };

        let task_type = task_def
            .task_type
            .parse::<TaskType>()
            .unwrap_or(TaskType::Analysis);

        let mut task = AgentTask {
            id: task_id.to_string(),
            task_type,
            description: task_def.description.clone(),
            priority: task_def.priority,
            status: TaskStatus::InProgress,
            result: None,
        };

        info!("{} executing task {}", agent.name(), task_id);

        match agent.execute_task(&mut task).await {
            Ok(_) => {
                agent.results.push(task);
                self.completed.push(task_id.to_string());
                true
            }
            Err(e) => {
                warn!("Task {} failed: {}", task_id, e);
                false
            }
        }
    }

    /// Run the coordinator: settle assignments through decentralized signal
    /// exchange, then execute the emergent winners.
    pub async fn run(&mut self, tasks: Vec<TaskDefinition>) -> CoordinatorResult {
        let start = std::time::Instant::now();

        // Add all tasks (emits task_available signals).
        for task in tasks {
            self.add_task(task);
        }

        let total_tasks = self.tasks.len();
        info!(
            "Starting coordinator with {} tasks, {} agents",
            total_tasks,
            self.agents.len()
        );

        // Phase 1 — decentralized settlement. Run local claim/back-off rounds
        // until the field stops changing (or a generous round budget is hit).
        let max_rounds = self.config.max_ticks.max(self.agents.len() as u64 * 2 + 4);
        for round in 0..max_rounds {
            self.tick = round + 1;
            if !self.coordination_round() {
                info!("Claims settled after {} round(s)", self.tick);
                break;
            }
        }

        // Phase 2 — execute the emergent winners (one LLM call per assignment).
        for (node_id, task_id) in self.converged_winners() {
            if self.completed.contains(&task_id) {
                continue;
            }
            self.assign_and_execute(&node_id, &task_id).await;
        }

        let elapsed = start.elapsed().as_secs_f64();
        let total_llm_calls: u64 = self.agents.values().map(|a| a.llm_calls).sum();

        let agent_results: Vec<_> = self
            .agents
            .values()
            .map(|a| (a.name().to_string(), a.results.clone()))
            .collect();

        CoordinatorResult {
            tasks_total: total_tasks,
            tasks_completed: self.completed.len(),
            total_llm_calls,
            elapsed_secs: elapsed,
            agent_results,
        }
    }
}

/// Parse a `task_claimed` signal into (task_id, claimer, affinity).
/// Returns `None` for any other signal type or malformed payload.
fn parse_claim(signal: &Signal) -> Option<(String, NodeId, f64)> {
    let payload: serde_json::Value = serde_json::from_slice(&signal.payload).ok()?;
    if payload.get("agent_signal_type")?.as_str()? != "task_claimed" {
        return None;
    }
    let task_id = payload.get("task_id")?.as_str()?.to_string();
    let claimer = payload.get("claimer")?.as_str()?.to_string();
    let affinity = payload.get("affinity")?.as_f64().unwrap_or(0.0);
    Some((task_id, claimer, affinity))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let config = CoordinatorConfig {
            n_agents: 3,
            ..Default::default()
        };

        let coordinator = AgentCoordinator::with_openrouter(config, "test-model");
        assert_eq!(coordinator.agents.len(), 3);
    }

    #[test]
    fn test_add_task() {
        let mut coordinator =
            AgentCoordinator::with_openrouter(CoordinatorConfig::default(), "test-model");

        let task_id = coordinator.add_task(TaskDefinition {
            task_type: "code_review".to_string(),
            description: "Review code".to_string(),
            priority: 0.8,
        });

        assert!(!task_id.is_empty());
        assert_eq!(coordinator.tasks.len(), 1);
        assert_eq!(coordinator.field.signals.len(), 1);
    }

    /// Settlement uses only the field + local agent rules, so it runs with no
    /// LLM backend involved. Drives `coordination_round`/`converged_winners`
    /// directly to prove assignment *emerges* rather than being centrally sorted.
    fn settle(coord: &mut AgentCoordinator) {
        let mut rounds = 0;
        while coord.coordination_round() {
            rounds += 1;
            assert!(rounds < 25, "decentralized settlement should converge");
        }
    }

    #[test]
    fn test_emergent_assignment_via_signal_backoff() {
        let config = CoordinatorConfig {
            n_agents: 3,
            roles: vec![AgentRole::Coder, AgentRole::Reviewer, AgentRole::Analyst],
            ..Default::default()
        };
        let mut coord = AgentCoordinator::with_openrouter(config, "test-model");

        let cw = coord.add_task(TaskDefinition {
            task_type: "code_write".into(),
            description: "implement".into(),
            priority: 0.8,
        });
        let cr = coord.add_task(TaskDefinition {
            task_type: "code_review".into(),
            description: "review".into(),
            priority: 0.8,
        });
        let an = coord.add_task(TaskDefinition {
            task_type: "analysis".into(),
            description: "analyze".into(),
            priority: 0.8,
        });

        settle(&mut coord);

        let winners: HashMap<String, NodeId> = coord
            .converged_winners()
            .into_iter()
            .map(|(node, task)| (task, node))
            .collect();

        let role_of = |node: &str| coord.agents.get(node).unwrap().config.role;

        // Each task emerged to the role with the dominant skill — no central
        // arbiter ranked the claims; weaker claimants withdrew locally.
        assert_eq!(winners.len(), 3, "every task assigned to some agent");
        assert_eq!(role_of(&winners[&cw]), AgentRole::Coder);
        assert_eq!(role_of(&winners[&cr]), AgentRole::Reviewer);
        assert_eq!(role_of(&winners[&an]), AgentRole::Analyst);
    }

    #[test]
    fn test_weaker_claims_withdraw_from_field() {
        // Regression guard: the losers must actually *remove* their claim
        // signals (decentralized back-off), leaving exactly one surviving claim
        // per task in the field — not a coordinator-side `max_by` over a pile of
        // competing claims that all linger.
        let config = CoordinatorConfig {
            n_agents: 3,
            roles: vec![AgentRole::Coder, AgentRole::Reviewer, AgentRole::Analyst],
            ..Default::default()
        };
        let mut coord = AgentCoordinator::with_openrouter(config, "test-model");

        coord.add_task(TaskDefinition {
            task_type: "code_write".into(),
            description: "implement".into(),
            priority: 0.9,
        });
        coord.add_task(TaskDefinition {
            task_type: "analysis".into(),
            description: "analyze".into(),
            priority: 0.9,
        });

        // Round 1: multiple agents claim each task (Coder/Reviewer both eye
        // code_write; Analyst/Reviewer both eye analysis).
        coord.coordination_round();
        let claims_after_first: usize = coord
            .field
            .signals
            .values()
            .filter(|s| parse_claim(s).is_some())
            .count();
        assert!(
            claims_after_first > 2,
            "expected contested claims in round 1, got {claims_after_first}"
        );

        settle(&mut coord);

        let surviving: usize = coord
            .field
            .signals
            .values()
            .filter(|s| parse_claim(s).is_some())
            .count();
        assert_eq!(surviving, 2, "one surviving claim per task after back-off");
    }
}
