//! Agent coordinator - manages multiple LLM agents via SMESH
//!
//! Provides backend factory pattern for creating agents with different
//! LLM backends (OpenRouter, Claude, etc.).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::affinity::AffinitySource;
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
    /// An optional label. One of [`TaskType`]'s names lets agents claim the
    /// task from their role table; any other label, or an empty one, means
    /// only an [`AffinitySource`] assessment can make an agent claim it.
    #[serde(default)]
    pub task_type: String,
    /// The task, in plain language.
    pub description: String,
    /// Priority from 0.0 to 1.0.
    pub priority: f64,
}

impl TaskDefinition {
    /// A task with no type label: any domain, routed by assessment alone.
    pub fn untyped(description: impl Into<String>, priority: f64) -> Self {
        Self {
            task_type: String::new(),
            description: description.into(),
            priority,
        }
    }
}

/// The claim and back-off events from one settlement round (for tracing/viz).
#[derive(Debug, Clone, Default)]
pub struct RoundTrace {
    /// (agent node id, task id, affinity) for each claim emitted this round.
    pub claims: Vec<(NodeId, String, f64)>,
    /// (agent node id, task id) for each claim withdrawn this round.
    pub backoffs: Vec<(NodeId, String)>,
}

/// A stable summary of one agent for layout/labelling.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub id: NodeId,
    pub name: String,
    pub role: String,
    /// What the agent does, in plain language.
    pub capability: String,
    /// Built-in task types this agent's role table rates at
    /// [`SPECIALTY_LEVEL`] or above. Lets a peer without an affinity source
    /// still match it by type; empty for agents described in words alone.
    pub specialties: Vec<String>,
}

/// The role-table level at which a task type counts as an agent's specialty.
pub const SPECIALTY_LEVEL: f64 = 0.8;

/// One task carried out by the agent that won it.
#[derive(Debug, Clone)]
pub struct ExecutedTask {
    /// Task ID in the coordinator's field.
    pub task_id: String,
    /// Node ID of the agent that executed it.
    pub node_id: NodeId,
    /// Display name of that agent.
    pub agent_name: String,
    /// What the agent produced.
    pub result: String,
    /// Tools the agent called on the way, in order.
    pub tool_calls: Vec<crate::tools::ToolCall>,
}

/// Why a task could not be executed.
#[derive(Debug, thiserror::Error)]
pub enum ExecuteError {
    /// No such task was ever added.
    #[error("unknown task {0}")]
    UnknownTask(String),
    /// Settlement ended with no claim on the task: no agent had the affinity.
    #[error("no agent claimed task {0}")]
    Unclaimed(String),
    /// The winning agent's backend failed.
    #[error("agent {agent} failed task {task_id}: {source}")]
    Backend {
        /// Task that failed.
        task_id: String,
        /// Agent that was executing it.
        agent: String,
        /// The backend's error.
        #[source]
        source: crate::backend::LlmError,
    },
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
    /// Judges how well each agent suits each task. `None` uses role tables.
    affinity: Option<Arc<dyn AffinitySource>>,
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
            affinity: None,
        }
    }

    /// Use `source` to judge how well each agent suits each task, instead of
    /// (or before falling back to) the role tables. See [`assess_task`].
    ///
    /// [`assess_task`]: Self::assess_task
    pub fn with_affinity_source(mut self, source: Arc<dyn AffinitySource>) -> Self {
        self.affinity = Some(source);
        self
    }

    /// Whether agents assess tasks with an affinity source.
    pub fn has_affinity_source(&self) -> bool {
        self.affinity.is_some()
    }

    /// Have every agent assess how well it suits `task_id`.
    ///
    /// Each agent is judged on its own description alone, never against the
    /// others, so the judgement is still each agent's own: ranking is left to
    /// claims and back-offs in the field. The judgements run concurrently and
    /// are cached on each agent, so settlement afterwards stays synchronous,
    /// local and replayable. An agent the source gives no judgement for falls
    /// back to its role table. Does nothing without an affinity source.
    pub async fn assess_task(&mut self, task_id: &str) {
        let (Some(source), Some(task)) = (self.affinity.clone(), self.tasks.get(task_id).cloned())
        else {
            return;
        };
        let agents: Vec<(NodeId, String)> = self
            .agents
            .iter()
            .map(|(id, a)| (id.clone(), a.profile()))
            .collect();
        let judgements = futures::future::join_all(
            agents
                .iter()
                .map(|(_, capability)| source.assess(&task, capability)),
        )
        .await;
        for ((node_id, _), judgement) in agents.into_iter().zip(judgements) {
            if let (Some(level), Some(agent)) = (judgement, self.agents.get_mut(&node_id)) {
                debug!("{} assesses task {} at {:.2}", agent.name(), task_id, level);
                agent
                    .assessed
                    .insert(task_id.to_string(), level.clamp(0.0, 1.0));
            }
        }
    }

    /// [`add_task`](Self::add_task), then [`assess_task`](Self::assess_task).
    pub async fn add_task_assessed(&mut self, task: TaskDefinition) -> String {
        let task_id = self.add_task(task);
        self.assess_task(&task_id).await;
        task_id
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

    /// Add an already-built agent to the mesh.
    ///
    /// Use this for an agent whose backend the factory does not make — a
    /// remote A2A agent, say. It takes part in settlement exactly like the
    /// agents the coordinator created. Returns its node ID.
    pub fn add_agent(&mut self, agent: LlmAgent) -> NodeId {
        let node_id = agent.node_id().to_string();
        info!("Added agent: {} ({})", agent.name(), node_id);
        self.agents.insert(node_id.clone(), agent);
        node_id
    }

    /// Every agent in the mesh.
    pub fn agents(&self) -> impl Iterator<Item = &LlmAgent> {
        self.agents.values()
    }

    /// One agent, by node ID.
    pub fn agent(&self, node_id: &str) -> Option<&LlmAgent> {
        self.agents.get(node_id)
    }

    /// The agent whose claim on `task_id` survived settlement, if any.
    pub fn winner_for(&self, task_id: &str) -> Option<NodeId> {
        self.converged_winners()
            .into_iter()
            .find(|(_, t)| t == task_id)
            .map(|(node, _)| node)
    }

    /// Execute one task on the agent that won it in settlement.
    ///
    /// Call after [`settle_traced`](Self::settle_traced) (or rounds of
    /// [`coordination_round`](Self::coordination_round)) have run.
    pub async fn execute_settled(&mut self, task_id: &str) -> Result<ExecutedTask, ExecuteError> {
        let task_def = self
            .tasks
            .get(task_id)
            .cloned()
            .ok_or_else(|| ExecuteError::UnknownTask(task_id.to_string()))?;
        let node_id = self
            .winner_for(task_id)
            .ok_or_else(|| ExecuteError::Unclaimed(task_id.to_string()))?;
        let agent = self
            .agents
            .get_mut(&node_id)
            .ok_or_else(|| ExecuteError::Unclaimed(task_id.to_string()))?;

        let mut task = AgentTask {
            id: task_id.to_string(),
            task_type: task_def.task_type.parse::<TaskType>().ok(),
            description: task_def.description,
            priority: task_def.priority,
            status: TaskStatus::InProgress,
            result: None,
        };

        let agent_name = agent.name().to_string();
        let result =
            agent
                .execute_task(&mut task)
                .await
                .map_err(|source| ExecuteError::Backend {
                    task_id: task_id.to_string(),
                    agent: agent_name.clone(),
                    source,
                })?;
        let tool_calls = std::mem::take(&mut agent.last_tool_calls);
        agent.results.push(task);
        self.completed.push(task_id.to_string());

        Ok(ExecutedTask {
            task_id: task_id.to_string(),
            node_id,
            agent_name,
            result,
            tool_calls,
        })
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
        self.coordination_round_traced().1
    }

    /// Like [`coordination_round`], but also returns the per-agent claim and
    /// back-off events that occurred this round (for visualization / tracing).
    /// The second tuple element is `true` if the field changed.
    pub fn coordination_round_traced(&mut self) -> (RoundTrace, bool) {
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
        let trace = RoundTrace {
            claims: claims.clone(),
            backoffs: backoffs.clone(),
        };

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

        (trace, changed)
    }

    /// Run decentralized settlement to convergence, capturing each round's
    /// claim/back-off events. Purely local + LLM-free — suitable for driving a
    /// live visualization of how consensus emerges.
    pub fn settle_traced(&mut self) -> Vec<RoundTrace> {
        let max_rounds = self.config.max_ticks.max(self.agents.len() as u64 * 2 + 4);
        let mut rounds = Vec::new();
        for _ in 0..max_rounds {
            let (trace, changed) = self.coordination_round_traced();
            rounds.push(trace);
            if !changed {
                break;
            }
        }
        rounds
    }

    /// A stable summary of every agent (node id, display name, role) for
    /// laying out and labelling a visualization.
    pub fn agent_roster(&self) -> Vec<AgentInfo> {
        let mut roster: Vec<AgentInfo> = self
            .agents
            .iter()
            .map(|(id, agent)| AgentInfo {
                id: id.clone(),
                name: agent.name().to_string(),
                role: format!("{:?}", agent.config.role),
                capability: agent.profile(),
                specialties: TaskType::ALL
                    .iter()
                    .filter(|t| agent.skill(**t) >= SPECIALTY_LEVEL)
                    .map(|t| t.as_str().to_string())
                    .collect(),
            })
            .collect();
        roster.sort_by(|a, b| a.name.cmp(&b.name));
        roster
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

        let task_type = task_def.task_type.parse::<TaskType>().ok();

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
            self.add_task_assessed(task).await;
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

    // ---- assessment: agents described in plain language, any domain ----

    /// Says an agent suits a task when the task mentions a word from the
    /// agent's capability, with a level set per capability word.
    struct KeywordAffinity(Vec<(&'static str, &'static str, f64)>);

    #[async_trait::async_trait]
    impl AffinitySource for KeywordAffinity {
        async fn assess(&self, task: &TaskDefinition, capability: &str) -> Option<f64> {
            let task = task.description.to_lowercase();
            self.0
                .iter()
                .find(|(cap, word, _)| capability.contains(cap) && task.contains(word))
                .map(|(_, _, level)| *level)
                .or(Some(0.05))
        }
    }

    /// Answers nothing, as a failing service would.
    struct Silent;

    #[async_trait::async_trait]
    impl AffinitySource for Silent {
        async fn assess(&self, _: &TaskDefinition, _: &str) -> Option<f64> {
            None
        }
    }

    fn described(name: &str, capability: &str) -> LlmAgent {
        let config = AgentConfig::new(name, AgentRole::General).with_capability(capability);
        LlmAgent::new(
            config,
            Arc::new(OpenRouterClient::new(OpenRouterConfig::default())),
        )
    }

    fn empty_mesh() -> AgentCoordinator {
        AgentCoordinator::with_openrouter(
            CoordinatorConfig {
                n_agents: 0,
                ..Default::default()
            },
            "test-model",
        )
    }

    fn winner_name(coord: &AgentCoordinator, task_id: &str) -> Option<String> {
        let node = coord.winner_for(task_id)?;
        Some(coord.agents[&node].name().to_string())
    }

    #[tokio::test]
    async fn test_untyped_task_goes_to_the_agent_whose_description_fits() {
        let mut coord = empty_mesh().with_affinity_source(Arc::new(KeywordAffinity(vec![
            ("translator", "translate", 0.95),
            ("contracts", "clause", 0.92),
        ])));
        coord.add_agent(described(
            "Translator",
            "a translator between English and Portuguese",
        ));
        coord.add_agent(described("Contracts", "a reviewer of commercial contracts"));

        let t = coord
            .add_task_assessed(TaskDefinition::untyped("Translate the refund policy", 0.5))
            .await;
        let c = coord
            .add_task_assessed(TaskDefinition::untyped(
                "Is this renewal clause risky?",
                0.5,
            ))
            .await;
        settle(&mut coord);

        assert_eq!(winner_name(&coord, &t).as_deref(), Some("Translator"));
        assert_eq!(winner_name(&coord, &c).as_deref(), Some("Contracts"));
    }

    #[tokio::test]
    async fn test_the_agent_holding_the_right_tool_claims_the_task() {
        use crate::tools::{Effect, Tool, ToolError, ToolOutput, Toolbelt};

        struct Named(&'static str, &'static str, Effect);
        #[async_trait::async_trait]
        impl Tool for Named {
            fn name(&self) -> &str {
                self.0
            }
            fn description(&self) -> &str {
                self.1
            }
            fn input_schema(&self) -> serde_json::Value {
                serde_json::json!({ "type": "object" })
            }
            fn effect(&self) -> Effect {
                self.2
            }
            async fn call(&self, _: serde_json::Value) -> Result<ToolOutput, ToolError> {
                Ok(ToolOutput::text(""))
            }
        }

        // Two agents described identically; only one holds the issue tracker,
        // and the source rewards the tool's presence in the profile.
        let mut coord = empty_mesh().with_affinity_source(Arc::new(KeywordAffinity(vec![
            ("tracker__open_issue", "file a bug", 0.9),
            ("an engineer", "file a bug", 0.4),
        ])));
        let tracker = Toolbelt::new(Effect::Write)
            .with_tool(Arc::new(Named(
                "tracker__open_issue",
                "Open an issue in the tracker.",
                Effect::Write,
            )))
            .unwrap();
        coord.add_agent(described("Plain", "an engineer"));
        coord.add_agent(described("Equipped", "an engineer").with_toolbelt(tracker));

        let equipped = coord
            .agents
            .values()
            .find(|a| a.name() == "Equipped")
            .unwrap();
        assert!(
            equipped
                .profile()
                .contains("tracker__open_issue (Open an issue in the tracker.)"),
            "{}",
            equipped.profile()
        );
        assert_eq!(
            coord
                .agent_roster()
                .into_iter()
                .find(|i| i.name == "Equipped")
                .unwrap()
                .capability,
            equipped.profile(),
            "peers and Agent Cards see the tools too"
        );

        let id = coord
            .add_task_assessed(TaskDefinition::untyped(
                "Please file a bug about the crash",
                0.5,
            ))
            .await;
        settle(&mut coord);
        assert_eq!(winner_name(&coord, &id).as_deref(), Some("Equipped"));
    }

    #[tokio::test]
    async fn test_tools_above_clearance_do_not_count_toward_routing() {
        use crate::tools::{builtin::Now, Effect, Toolbelt};
        // `Now` is a read tool, so a read clearance shows it ...
        let shown = described("A", "an assistant").with_toolbelt(
            Toolbelt::new(Effect::Read)
                .with_tool(Arc::new(Now))
                .unwrap(),
        );
        assert!(shown.profile().contains("builtin__now"));
        // ... and an agent with no usable tools reads exactly as before.
        let none = described("B", "an assistant").with_toolbelt(Toolbelt::new(Effect::Read));
        assert_eq!(none.profile(), "an assistant");
    }

    #[tokio::test]
    async fn test_task_no_agent_suits_is_left_unclaimed() {
        let mut coord = empty_mesh().with_affinity_source(Arc::new(KeywordAffinity(vec![(
            "translator",
            "translate",
            0.95,
        )])));
        coord.add_agent(described("Translator", "a translator"));
        let id = coord
            .add_task_assessed(TaskDefinition::untyped("Bake a sourdough loaf", 0.5))
            .await;
        settle(&mut coord);
        assert_eq!(coord.winner_for(&id), None);
    }

    #[tokio::test]
    async fn test_untyped_task_without_a_source_is_unclaimed() {
        let mut coord = empty_mesh();
        coord.add_agent(described("Translator", "a translator"));
        let id = coord
            .add_task_assessed(TaskDefinition::untyped("Translate this", 0.5))
            .await;
        settle(&mut coord);
        assert_eq!(coord.winner_for(&id), None);
    }

    #[tokio::test]
    async fn test_assessment_overrides_the_role_table() {
        // By the table, a Reviewer (0.95) beats a Writer (0.5) on code_review.
        // The task is really about prose, and the assessment says so.
        let mut coord = AgentCoordinator::with_openrouter(
            CoordinatorConfig {
                n_agents: 2,
                roles: vec![AgentRole::Reviewer, AgentRole::Writer],
                ..Default::default()
            },
            "test-model",
        )
        .with_affinity_source(Arc::new(KeywordAffinity(vec![
            ("technical writer", "wording", 0.98),
            ("code reviewer", "wording", 0.55),
        ])));
        let id = coord
            .add_task_assessed(TaskDefinition {
                task_type: "code_review".into(),
                description: "Review the wording of our error messages".into(),
                priority: 0.5,
            })
            .await;
        settle(&mut coord);
        assert_eq!(winner_name(&coord, &id).as_deref(), Some("Writer-2"));
    }

    #[tokio::test]
    async fn test_failed_assessment_falls_back_to_the_role_table() {
        let mut coord = AgentCoordinator::with_openrouter(
            CoordinatorConfig {
                n_agents: 2,
                roles: vec![AgentRole::Reviewer, AgentRole::Writer],
                ..Default::default()
            },
            "test-model",
        )
        .with_affinity_source(Arc::new(Silent));
        let id = coord
            .add_task_assessed(TaskDefinition {
                task_type: "code_review".into(),
                description: "Review the wording of our error messages".into(),
                priority: 0.5,
            })
            .await;
        settle(&mut coord);
        assert_eq!(winner_name(&coord, &id).as_deref(), Some("Reviewer-1"));
    }

    #[test]
    fn test_roster_carries_each_agents_capability() {
        let mut coord = empty_mesh();
        coord.add_agent(described("Tutor", "a maths tutor for school students"));
        let roster = coord.agent_roster();
        assert_eq!(roster[0].capability, "a maths tutor for school students");
    }
}
