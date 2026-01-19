//! Agent coordinator - manages multiple LLM agents via SMESH
//!
//! Provides backend factory pattern for creating agents with different
//! LLM backends (Ollama, Claude, etc.).

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn};

use smesh_core::{Signal, SignalType, Field, NodeId};
use crate::agent::{LlmAgent, AgentConfig, AgentRole, AgentTask, TaskType, TaskStatus, AgentAction};
use crate::backend::LlmBackend;
use crate::ollama::{OllamaClient, OllamaConfig};
use crate::claude::{ClaudeClient, ClaudeConfig};

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
            model: "deepseek-coder-v2:16b".to_string(),
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
                ollama: None,
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

    /// Create a coordinator with Ollama backend (backward compatibility)
    pub fn with_ollama(config: CoordinatorConfig, model: &str) -> Self {
        let model = model.to_string();
        let factory: BackendFactory = Box::new(move || {
            let ollama_config = OllamaConfig {
                model: model.clone(),
                ..Default::default()
            };
            Arc::new(OllamaClient::new(ollama_config))
        });
        Self::new(config, factory)
    }

    /// Create a coordinator with Claude backend
    pub fn with_claude(config: CoordinatorConfig, claude_config: ClaudeConfig) -> Self {
        let factory: BackendFactory = Box::new(move || {
            Arc::new(ClaudeClient::new(claude_config.clone()))
        });
        Self::new(config, factory)
    }
    
    /// Add a task to be coordinated
    pub fn add_task(&mut self, task: TaskDefinition) -> String {
        let task_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        
        // Emit task available signal
        let payload = serde_json::json!({
            "agent_signal_type": "task_available",
            "task_id": task_id,
            "task_type": task.task_type,
            "priority": task.priority,
            "description": task.description,
        });
        
        let signal = Signal::builder(SignalType::Custom)
            .payload(serde_json::to_vec(&payload).unwrap())
            .intensity(0.5 + task.priority * 0.5)
            .ttl(120.0)
            .build();
        
        self.field.emit_anonymous(signal);
        self.tasks.insert(task_id.clone(), task);
        
        debug!("Added task: {}", task_id);
        task_id
    }
    
    /// Run one coordination tick
    pub async fn tick(&mut self) -> TickResult {
        self.tick += 1;
        let mut claims: HashMap<String, Vec<(NodeId, f64)>> = HashMap::new();
        let mut completions = Vec::new();
        
        // Each agent senses and decides
        let signals: Vec<Signal> = self.field.signals.values().cloned().collect();
        
        for (node_id, agent) in &mut self.agents {
            let actions = agent.process_signals(&signals);
            
            for action in actions {
                match action {
                    AgentAction::ClaimTask { task_id, affinity } => {
                        claims.entry(task_id)
                            .or_default()
                            .push((node_id.clone(), affinity));
                    }
                    AgentAction::CompleteTask { task_id, result } => {
                        completions.push((node_id.clone(), task_id, result));
                    }
                    _ => {}
                }
            }
        }
        
        // Resolve claims - highest affinity wins
        for (task_id, task_claims) in claims {
            if self.completed.contains(&task_id) {
                continue;
            }
            
            if let Some((winner_id, affinity)) = task_claims.into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            {
                // Assign task to winner
                if let Some(agent) = self.agents.get_mut(&winner_id) {
                    if let Some(task_def) = self.tasks.get(&task_id) {
                        let task_type = TaskType::from_str(&task_def.task_type)
                            .unwrap_or(TaskType::Analysis);
                        
                        let task = AgentTask {
                            id: task_id.clone(),
                            task_type,
                            description: task_def.description.clone(),
                            priority: task_def.priority,
                            status: TaskStatus::InProgress,
                            result: None,
                        };
                        
                        agent.current_tasks.push(task);
                        info!("{} claimed task {} (affinity: {:.0}%)", 
                              agent.name(), task_id, affinity * 100.0);
                    }
                }
            }
        }
        
        // Execute tasks - collect task indices first to avoid borrow issues
        let mut executed = 0;
        for agent in self.agents.values_mut() {
            // Find tasks that need execution
            let task_indices: Vec<usize> = agent.current_tasks
                .iter()
                .enumerate()
                .filter(|(_, t)| t.status == TaskStatus::InProgress)
                .map(|(i, _)| i)
                .collect();
            
            // Execute each task
            for idx in task_indices {
                let task = &mut agent.current_tasks[idx];
                let task_id = task.id.clone();

                // Execute task using the backend
                let prompt = format!(
                    "Execute this task:\n\nType: {}\nDescription: {}\nPriority: {:.1}\n\nProvide your work result. Be concise but complete.",
                    task.task_type.as_str(),
                    task.description,
                    task.priority
                );

                let system = agent.config.role.system_prompt(&agent.config.name);

                match agent.backend().generate_with_system(&prompt, &system).await {
                    Ok(result) => {
                        agent.llm_calls += 1;
                        agent.current_tasks[idx].status = TaskStatus::Completed;
                        agent.current_tasks[idx].result = Some(result);
                        executed += 1;
                        self.completed.push(task_id);
                    }
                    Err(e) => {
                        warn!("Task {} failed: {}", task_id, e);
                        agent.current_tasks[idx].status = TaskStatus::Failed;
                    }
                }
            }
            
            // Move completed tasks to results
            let completed: Vec<_> = agent.current_tasks
                .drain(..)
                .filter(|t| t.status == TaskStatus::Completed)
                .collect();
            agent.results.extend(completed);
        }
        
        // Tick the field
        self.field.tick(self.config.tick_interval_ms as f64 / 1000.0);
        
        TickResult {
            tick: self.tick,
            pending_tasks: self.tasks.len() - self.completed.len(),
            completed_tasks: self.completed.len(),
            tasks_executed: executed,
        }
    }
    
    /// Run the coordinator until all tasks complete or max ticks
    pub async fn run(&mut self, tasks: Vec<TaskDefinition>) -> CoordinatorResult {
        let start = std::time::Instant::now();
        
        // Add all tasks
        for task in tasks {
            self.add_task(task);
        }
        
        let total_tasks = self.tasks.len();
        info!("Starting coordinator with {} tasks, {} agents", 
              total_tasks, self.agents.len());
        
        // Run until done
        for _ in 0..self.config.max_ticks {
            let result = self.tick().await;
            
            if result.tick % 10 == 0 {
                debug!("Tick {}: {} pending, {} completed", 
                       result.tick, result.pending_tasks, result.completed_tasks);
            }
            
            // Check if done
            if result.pending_tasks == 0 && 
               self.agents.values().all(|a| a.current_tasks.is_empty()) {
                info!("All tasks completed at tick {}", result.tick);
                break;
            }
            
            tokio::time::sleep(std::time::Duration::from_millis(
                self.config.tick_interval_ms
            )).await;
        }
        
        let elapsed = start.elapsed().as_secs_f64();
        let total_llm_calls: u64 = self.agents.values().map(|a| a.llm_calls).sum();
        
        let agent_results: Vec<_> = self.agents
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

/// Result of a single tick
#[derive(Debug, Clone)]
pub struct TickResult {
    pub tick: u64,
    pub pending_tasks: usize,
    pub completed_tasks: usize,
    pub tasks_executed: usize,
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

        let coordinator = AgentCoordinator::with_ollama(config, "test-model");
        assert_eq!(coordinator.agents.len(), 3);
    }

    #[test]
    fn test_add_task() {
        let mut coordinator =
            AgentCoordinator::with_ollama(CoordinatorConfig::default(), "test-model");

        let task_id = coordinator.add_task(TaskDefinition {
            task_type: "code_review".to_string(),
            description: "Review code".to_string(),
            priority: 0.8,
        });

        assert!(!task_id.is_empty());
        assert_eq!(coordinator.tasks.len(), 1);
        assert_eq!(coordinator.field.signals.len(), 1);
    }
}
