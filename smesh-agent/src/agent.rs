//! LLM-powered agent for SMESH
//!
//! Provides generic backend support for LLM agents, allowing use of
//! OpenRouter, Claude, or other LLM backends.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use crate::backend::LlmProvider;
use crate::backend::{LlmBackend, LlmError};
use crate::openrouter::{OpenRouterClient, OpenRouterConfig};
use crate::tools::{privilege_rank, run_with_tools, Effect, ToolCall, Toolbelt};
use smesh_core::{Node, Signal};

/// Agent roles that determine skills and behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    /// Code writing and implementation
    Coder,
    /// Code review and quality
    Reviewer,
    /// Analysis and research
    Analyst,
    /// Documentation and writing
    Writer,
    /// General purpose
    General,
}

impl AgentRole {
    /// Get skill levels for this role
    pub fn skills(&self) -> HashMap<TaskType, f64> {
        match self {
            AgentRole::Coder => [
                (TaskType::CodeWrite, 0.95),
                (TaskType::CodeReview, 0.7),
                (TaskType::Testing, 0.8),
                (TaskType::Documentation, 0.5),
                (TaskType::Analysis, 0.6),
            ]
            .into_iter()
            .collect(),
            AgentRole::Reviewer => [
                (TaskType::CodeReview, 0.95),
                (TaskType::CodeWrite, 0.6),
                (TaskType::Testing, 0.7),
                (TaskType::Documentation, 0.6),
                (TaskType::Analysis, 0.8),
            ]
            .into_iter()
            .collect(),
            AgentRole::Analyst => [
                (TaskType::Analysis, 0.95),
                (TaskType::Documentation, 0.7),
                (TaskType::CodeReview, 0.5),
                (TaskType::CodeWrite, 0.4),
                (TaskType::Testing, 0.5),
            ]
            .into_iter()
            .collect(),
            AgentRole::Writer => [
                (TaskType::Documentation, 0.95),
                (TaskType::Analysis, 0.7),
                (TaskType::CodeReview, 0.5),
                (TaskType::CodeWrite, 0.4),
                (TaskType::Testing, 0.3),
            ]
            .into_iter()
            .collect(),
            AgentRole::General => [
                (TaskType::CodeWrite, 0.7),
                (TaskType::CodeReview, 0.7),
                (TaskType::Testing, 0.7),
                (TaskType::Documentation, 0.7),
                (TaskType::Analysis, 0.7),
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Get system prompt for this role
    ///
    /// The prompt deliberately does not mention SMESH or signals. Coordination
    /// happens before the model is called; the model only does the work. Told
    /// it "coordinates through a signal-based system (SMESH)", models invented
    /// a protocol to perform in every reply ("SMESH: Starting code_write
    /// task.", "SMESH_START ... SMESH_END", JSON envelopes), and that reached
    /// whoever asked, A2A clients included.
    pub fn system_prompt(&self, name: &str) -> String {
        system_prompt_for(name, self.capability())
    }

    /// This role as a plain-language capability: what an agent with no
    /// description of its own says it does.
    pub fn capability(&self) -> &'static str {
        match self {
            AgentRole::Coder => "an expert software developer",
            AgentRole::Reviewer => "a code reviewer and quality specialist",
            AgentRole::Analyst => "a technical analyst and researcher",
            AgentRole::Writer => "a technical writer and documentation specialist",
            AgentRole::General => "a general-purpose AI assistant",
        }
    }
}

/// The system prompt for an agent called `name` that does `capability`.
///
/// See [`AgentRole::system_prompt`] for why it never mentions SMESH.
pub fn system_prompt_for(name: &str, capability: &str) -> String {
    format!(
        "You are {name}, {capability}. Your reply is the finished work itself \
         and goes straight to the person who asked for it. Write only the \
         result: no status lines, headers, protocol markers or notes about \
         the task. Be concise and focused."
    )
}

/// Types of tasks agents can handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    CodeWrite,
    CodeReview,
    Testing,
    Documentation,
    Analysis,
}

impl TaskType {
    /// Every task type, in a fixed order.
    pub const ALL: [TaskType; 5] = [
        TaskType::CodeWrite,
        TaskType::CodeReview,
        TaskType::Testing,
        TaskType::Documentation,
        TaskType::Analysis,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::CodeWrite => "code_write",
            TaskType::CodeReview => "code_review",
            TaskType::Testing => "testing",
            TaskType::Documentation => "documentation",
            TaskType::Analysis => "analysis",
        }
    }
}

impl std::str::FromStr for TaskType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "code_write" => Ok(TaskType::CodeWrite),
            "code_review" => Ok(TaskType::CodeReview),
            "testing" => Ok(TaskType::Testing),
            "documentation" => Ok(TaskType::Documentation),
            "analysis" => Ok(TaskType::Analysis),
            _ => Err(()),
        }
    }
}

/// Configuration for an LLM agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent name
    pub name: String,
    /// Agent role
    pub role: AgentRole,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Minimum affinity to claim a task
    pub affinity_threshold: f64,
    /// OpenRouter configuration (used when the agent builds its own backend)
    #[serde(default)]
    pub openrouter: Option<OpenRouterConfig>,
    /// Per-task-type skill levels that replace the role's table.
    ///
    /// A role's skills are a fixed table. An agent whose abilities come from
    /// somewhere else — a remote A2A agent, whose card advertises its skills —
    /// sets them here instead. A task type missing from the map falls back to
    /// the role's level for it.
    #[serde(default)]
    pub skill_overrides: Option<HashMap<TaskType, f64>>,
    /// What this agent does, in plain language. Overrides the role's preset.
    ///
    /// This is what lets an agent be anything: a translator, a contract
    /// reviewer, a tutor. An affinity source reads it to judge whether the
    /// agent suits a task, and it becomes the agent's system prompt. Phrase it
    /// to follow "You are <name>, ...", e.g. "a translator between English and
    /// Portuguese".
    #[serde(default)]
    pub capability: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "Agent".to_string(),
            role: AgentRole::General,
            max_concurrent_tasks: 3,
            affinity_threshold: 0.5,
            openrouter: Some(OpenRouterConfig::default()),
            skill_overrides: None,
            capability: None,
        }
    }
}

impl AgentConfig {
    /// Create a new agent config with a name and role
    pub fn new(name: impl Into<String>, role: AgentRole) -> Self {
        Self {
            name: name.into(),
            role,
            ..Default::default()
        }
    }

    /// Set maximum concurrent tasks
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent_tasks = max;
        self
    }

    /// Set affinity threshold
    pub fn with_affinity_threshold(mut self, threshold: f64) -> Self {
        self.affinity_threshold = threshold;
        self
    }

    /// Describe what this agent does, in place of its role's preset.
    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.capability = Some(capability.into());
        self
    }

    /// What this agent does: its own description, or its role's preset.
    pub fn capability_text(&self) -> &str {
        self.capability
            .as_deref()
            .unwrap_or_else(|| self.role.capability())
    }

    /// Replace the role's skill levels with these, per task type.
    pub fn with_skill_overrides(mut self, skills: HashMap<TaskType, f64>) -> Self {
        self.skill_overrides = Some(skills);
        self
    }
}

/// A task being worked on
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    /// The task's type, when it has one of the built-in ones.
    pub task_type: Option<TaskType>,
    pub description: String,
    pub priority: f64,
    pub status: TaskStatus,
    pub result: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// An LLM-powered agent in the SMESH network
pub struct LlmAgent {
    /// Underlying SMESH node
    pub node: Node,
    /// Agent configuration
    pub config: AgentConfig,
    /// LLM backend (generic)
    backend: Arc<dyn LlmBackend>,
    /// Current tasks
    pub current_tasks: Vec<AgentTask>,
    /// Task queue (claimed but not started)
    pub task_queue: Vec<String>,
    /// Tasks this agent has yielded to a stronger claimant and will not
    /// re-claim. This makes decentralized back-off *stable*: once an agent
    /// concedes a task to a higher-affinity peer it stays conceded, so the
    /// field settles instead of oscillating between claim and back-off.
    pub conceded: Vec<String>,
    /// Completed task results
    pub results: Vec<AgentTask>,
    /// Metrics
    pub llm_calls: u64,
    /// This agent's own judgement of how well it suits each task, by task ID,
    /// from an affinity source. It takes precedence over the role table, and
    /// is what lets an agent claim a task that has no built-in type.
    pub assessed: HashMap<String, f64>,
    /// Tools this agent may call, and how far it may go with them.
    toolbelt: Option<Toolbelt>,
    /// Tool calls made while executing the most recent task.
    pub last_tool_calls: Vec<ToolCall>,
}

/// Most model turns an agent spends calling tools on one task.
pub const MAX_TOOL_TURNS: usize = 8;

/// Claims within this much affinity of each other count as a near-tie.
///
/// Outside it the stronger claim wins, as it always has. Inside it the claim
/// from the less privileged agent wins (least privilege): when two agents
/// suit a task about equally, the one that can do less harm takes it.
pub const CLAIM_MARGIN: f64 = 0.1;

/// Whether a claim (`their` affinity and privilege) beats ours on a task.
///
/// Every agent applies the same rule to what it senses, so settlement stays
/// decentralized: a much stronger claim wins; in a near-tie
/// ([`CLAIM_MARGIN`]) the less privileged claim wins; otherwise neither
/// yields.
pub fn claim_outranks(
    their_affinity: f64,
    their_privilege: Option<Effect>,
    our_affinity: f64,
    our_privilege: Option<Effect>,
) -> bool {
    if their_affinity > our_affinity + CLAIM_MARGIN {
        return true;
    }
    (their_affinity - our_affinity).abs() <= CLAIM_MARGIN
        && privilege_rank(their_privilege) < privilege_rank(our_privilege)
}

impl LlmAgent {
    /// Create a new LLM agent with a specific backend
    pub fn new(config: AgentConfig, backend: Arc<dyn LlmBackend>) -> Self {
        let node = Node::new();

        Self {
            node,
            config,
            backend,
            current_tasks: Vec::new(),
            task_queue: Vec::new(),
            conceded: Vec::new(),
            results: Vec::new(),
            llm_calls: 0,
            assessed: HashMap::new(),
            toolbelt: None,
            last_tool_calls: Vec::new(),
        }
    }

    /// Give this agent tools. They count toward which tasks it suits (see
    /// [`profile`](Self::profile)) and it calls them while executing.
    pub fn with_toolbelt(mut self, toolbelt: Toolbelt) -> Self {
        self.toolbelt = Some(toolbelt.for_agent(&self.config.name));
        self
    }

    /// This agent's tools, if it has any.
    pub fn toolbelt(&self) -> Option<&Toolbelt> {
        self.toolbelt.as_ref()
    }

    /// The most this agent could do to the world, for least-privilege
    /// tie-breaks: the strongest effect among its usable tools, or
    /// [`Effect::Read`] for a model with no tools (it can only answer). `None`
    /// for a remote A2A agent, whose powers this mesh cannot see.
    pub fn privilege(&self) -> Option<Effect> {
        if self.backend.provider() == LlmProvider::A2a {
            return None;
        }
        Some(
            self.toolbelt
                .as_ref()
                .and_then(|b| b.usable().map(|t| t.effect()).max())
                .unwrap_or(Effect::Read),
        )
    }

    /// What this agent does and the tools it can use, in plain language.
    /// This is what an affinity source judges it on, so an agent holding the
    /// right tools for a task is the one that claims it.
    pub fn profile(&self) -> String {
        let capability = self.config.capability_text();
        match self.toolbelt.as_ref().map(Toolbelt::describe) {
            Some(tools) if !tools.is_empty() => format!("{capability}. {tools}"),
            _ => capability.to_string(),
        }
    }

    /// Create a new LLM agent with an OpenRouter backend.
    ///
    /// Uses the config's `openrouter` settings if present, otherwise resolves
    /// credentials from the environment / credentials file.
    pub fn with_openrouter(config: AgentConfig) -> Self {
        let or_config = config
            .openrouter
            .as_ref()
            .cloned()
            .or_else(OpenRouterConfig::from_env)
            .unwrap_or_default();
        let backend = Arc::new(OpenRouterClient::new(or_config));
        Self::new(config, backend)
    }

    /// Create a new LLM agent with a specific OpenRouter configuration.
    pub fn with_openrouter_config(config: AgentConfig, or_config: OpenRouterConfig) -> Self {
        let backend = Arc::new(OpenRouterClient::new(or_config));
        Self::new(config, backend)
    }

    /// Get a reference to the backend
    pub fn backend(&self) -> &dyn LlmBackend {
        self.backend.as_ref()
    }

    /// Get agent's node ID
    pub fn node_id(&self) -> &str {
        &self.node.id
    }

    /// Get agent's name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get skill level for a task type
    pub fn skill(&self, task_type: TaskType) -> f64 {
        if let Some(level) = self
            .config
            .skill_overrides
            .as_ref()
            .and_then(|skills| skills.get(&task_type))
        {
            return *level;
        }
        self.config
            .role
            .skills()
            .get(&task_type)
            .copied()
            .unwrap_or(0.5)
    }

    /// Check if agent can take more tasks
    pub fn has_capacity(&self) -> bool {
        self.current_tasks.len() < self.config.max_concurrent_tasks
    }

    /// Decide whether to claim a task based on skills
    pub fn should_claim(&self, task_type: TaskType, _priority: f64) -> Option<f64> {
        if !self.has_capacity() {
            return None;
        }

        let affinity = self.skill(task_type);
        if affinity >= self.config.affinity_threshold {
            Some(affinity)
        } else {
            None
        }
    }

    /// How strongly this agent would claim a task.
    ///
    /// Its own assessment for the task if it has one; otherwise the role table
    /// for the task's type, if the type is a built-in one; otherwise `None`,
    /// meaning the agent has no basis to claim it.
    pub fn affinity_for(&self, task_id: &str, task_type: Option<&str>) -> Option<f64> {
        if let Some(level) = self.assessed.get(task_id) {
            return Some(*level);
        }
        task_type
            .and_then(|t| t.parse::<TaskType>().ok())
            .map(|t| self.skill(t))
    }

    /// The affinity this agent claims `task_id` with, if it would claim it at
    /// all: it has capacity and its affinity reaches its threshold.
    pub fn claim_level(&self, task_id: &str, task_type: Option<&str>) -> Option<f64> {
        if !self.has_capacity() {
            return None;
        }
        self.affinity_for(task_id, task_type)
            .filter(|a| *a >= self.config.affinity_threshold)
    }

    /// Execute a task using the LLM
    pub async fn execute_task(&mut self, task: &mut AgentTask) -> Result<String, LlmError> {
        let type_line = task
            .task_type
            .map(|t| format!("Type: {}\n", t.as_str()))
            .unwrap_or_default();
        let prompt = format!(
            "Execute this task:\n\n\
             {type_line}\
             Description: {}\n\
             Priority: {:.1}\n\n\
             Provide your work result. Be concise but complete.",
            task.description, task.priority
        );

        let system = system_prompt_for(&self.config.name, self.config.capability_text());

        debug!("Agent {} executing task {}", self.config.name, task.id);

        self.last_tool_calls.clear();
        let tools = self
            .toolbelt
            .as_ref()
            .filter(|b| b.has_usable() && self.backend.supports_tools());
        let result = match tools {
            Some(belt) => {
                let run = run_with_tools(
                    self.backend.as_ref(),
                    &system,
                    &prompt,
                    belt,
                    MAX_TOOL_TURNS,
                )
                .await?;
                self.llm_calls += run.turns as u64;
                self.last_tool_calls = run.calls;
                run.text
            }
            None => {
                self.llm_calls += 1;
                self.backend.generate_with_system(&prompt, &system).await?
            }
        };

        task.status = TaskStatus::Completed;
        task.result = Some(result.clone());

        info!("Agent {} completed task {}", self.config.name, task.id);

        Ok(result)
    }

    /// Process signals and decide on actions
    pub fn process_signals(&mut self, signals: &[Signal]) -> Vec<AgentAction> {
        let mut actions = Vec::new();

        for signal in signals {
            if let Some(action) = self.process_signal(signal) {
                actions.push(action);
            }
        }

        actions
    }

    fn process_signal(&mut self, signal: &Signal) -> Option<AgentAction> {
        // Parse signal payload
        let payload: serde_json::Value = serde_json::from_slice(&signal.payload).ok()?;

        let signal_type = payload.get("agent_signal_type")?.as_str()?;

        match signal_type {
            "task_available" => {
                let task_id = payload.get("task_id")?.as_str()?;
                let task_type = payload.get("task_type").and_then(|v| v.as_str());

                // Skip if already in queue, or if we already yielded this task
                // to a stronger claimant (stable back-off).
                if self.task_queue.contains(&task_id.to_string())
                    || self.conceded.iter().any(|t| t == task_id)
                {
                    return None;
                }

                // Check if we should claim
                if let Some(affinity) = self.claim_level(task_id, task_type) {
                    self.task_queue.push(task_id.to_string());
                    return Some(AgentAction::ClaimTask {
                        task_id: task_id.to_string(),
                        affinity,
                    });
                }
            }
            "task_claimed" => {
                let task_id = payload.get("task_id")?.as_str()?;
                let claimer = payload.get("claimer")?.as_str()?;
                let their_affinity = payload.get("affinity")?.as_f64().unwrap_or(0.0);
                let their_privilege = claim_privilege(&payload);

                // Check for conflict
                if self.task_queue.contains(&task_id.to_string()) && claimer != self.node.id {
                    // Back off from a much stronger claim, or from a near-tie
                    // with a less privileged agent.
                    let task_type = payload.get("task_type").and_then(|v| v.as_str());
                    let our_affinity = self.affinity_for(task_id, task_type).unwrap_or(0.5);

                    if claim_outranks(
                        their_affinity,
                        their_privilege,
                        our_affinity,
                        self.privilege(),
                    ) {
                        self.task_queue.retain(|id| id != task_id);
                        // Remember the concession so we don't re-claim next round.
                        if !self.conceded.iter().any(|t| t == task_id) {
                            self.conceded.push(task_id.to_string());
                        }
                        return Some(AgentAction::BackOff {
                            task_id: task_id.to_string(),
                        });
                    }
                }
            }
            _ => {}
        }

        None
    }
}

/// The claimer's privilege carried in a claim signal (`None` when absent or
/// unknown, which ranks highest).
pub fn claim_privilege(payload: &serde_json::Value) -> Option<Effect> {
    payload
        .get("privilege")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse().ok())
}

/// Actions an agent can take
#[derive(Debug, Clone)]
pub enum AgentAction {
    /// Claim a task
    ClaimTask { task_id: String, affinity: f64 },
    /// Back off from a claimed task
    BackOff { task_id: String },
    /// Emit task completion
    CompleteTask { task_id: String, result: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_prompt_does_not_invite_protocol_framing() {
        for role in [
            AgentRole::Coder,
            AgentRole::Reviewer,
            AgentRole::Analyst,
            AgentRole::Writer,
            AgentRole::General,
        ] {
            let prompt = role.system_prompt("Agent-1");
            assert!(prompt.starts_with("You are Agent-1, a"), "{prompt}");
            // Naming the coordination protocol made models perform it in
            // their replies; see `AgentRole::system_prompt`.
            assert!(!prompt.contains("SMESH"), "{prompt}");
            assert!(!prompt.to_lowercase().contains("signal"), "{prompt}");
        }
    }

    #[test]
    fn test_agent_creation() {
        let config = AgentConfig {
            name: "TestAgent".to_string(),
            role: AgentRole::Coder,
            ..Default::default()
        };

        let agent = LlmAgent::with_openrouter(config);

        assert_eq!(agent.name(), "TestAgent");
        assert!(agent.skill(TaskType::CodeWrite) > 0.9);
        assert!(agent.has_capacity());
    }

    #[test]
    fn test_should_claim() {
        let config = AgentConfig {
            name: "Coder".to_string(),
            role: AgentRole::Coder,
            affinity_threshold: 0.6, // Higher threshold
            ..Default::default()
        };

        let agent = LlmAgent::with_openrouter(config);

        // Coder should claim code tasks (skill 0.95 > 0.6)
        assert!(agent.should_claim(TaskType::CodeWrite, 0.5).is_some());

        // Coder should not claim documentation (skill 0.5 < 0.6 threshold)
        assert!(agent.should_claim(TaskType::Documentation, 0.5).is_none());
    }

    #[test]
    fn test_role_skills() {
        let coder_skills = AgentRole::Coder.skills();
        let analyst_skills = AgentRole::Analyst.skills();

        assert!(coder_skills[&TaskType::CodeWrite] > analyst_skills[&TaskType::CodeWrite]);
        assert!(analyst_skills[&TaskType::Analysis] > coder_skills[&TaskType::Analysis]);
    }

    #[test]
    fn test_agent_config_builder() {
        let config = AgentConfig::new("TestAgent", AgentRole::Analyst)
            .with_max_concurrent(5)
            .with_affinity_threshold(0.8);

        assert_eq!(config.name, "TestAgent");
        assert_eq!(config.role, AgentRole::Analyst);
        assert_eq!(config.max_concurrent_tasks, 5);
        assert_eq!(config.affinity_threshold, 0.8);
    }
}
