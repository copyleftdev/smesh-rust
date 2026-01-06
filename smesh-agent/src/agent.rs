//! LLM-powered agent for SMESH

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tracing::{info, debug};

use smesh_core::{Node, Signal};
use crate::ollama::{OllamaClient, OllamaConfig, OllamaError};

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
            ].into_iter().collect(),
            AgentRole::Reviewer => [
                (TaskType::CodeReview, 0.95),
                (TaskType::CodeWrite, 0.6),
                (TaskType::Testing, 0.7),
                (TaskType::Documentation, 0.6),
                (TaskType::Analysis, 0.8),
            ].into_iter().collect(),
            AgentRole::Analyst => [
                (TaskType::Analysis, 0.95),
                (TaskType::Documentation, 0.7),
                (TaskType::CodeReview, 0.5),
                (TaskType::CodeWrite, 0.4),
                (TaskType::Testing, 0.5),
            ].into_iter().collect(),
            AgentRole::Writer => [
                (TaskType::Documentation, 0.95),
                (TaskType::Analysis, 0.7),
                (TaskType::CodeReview, 0.5),
                (TaskType::CodeWrite, 0.4),
                (TaskType::Testing, 0.3),
            ].into_iter().collect(),
            AgentRole::General => [
                (TaskType::CodeWrite, 0.7),
                (TaskType::CodeReview, 0.7),
                (TaskType::Testing, 0.7),
                (TaskType::Documentation, 0.7),
                (TaskType::Analysis, 0.7),
            ].into_iter().collect(),
        }
    }
    
    /// Get system prompt for this role
    pub fn system_prompt(&self, name: &str) -> String {
        let role_desc = match self {
            AgentRole::Coder => "expert software developer",
            AgentRole::Reviewer => "code reviewer and quality specialist",
            AgentRole::Analyst => "technical analyst and researcher",
            AgentRole::Writer => "technical writer and documentation specialist",
            AgentRole::General => "general-purpose AI assistant",
        };
        
        format!(
            "You are {}, a {}. You coordinate with other agents through \
             a signal-based system (SMESH). Be concise and focused on the task.",
            name, role_desc
        )
    }
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
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::CodeWrite => "code_write",
            TaskType::CodeReview => "code_review",
            TaskType::Testing => "testing",
            TaskType::Documentation => "documentation",
            TaskType::Analysis => "analysis",
        }
    }
    
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "code_write" => Some(TaskType::CodeWrite),
            "code_review" => Some(TaskType::CodeReview),
            "testing" => Some(TaskType::Testing),
            "documentation" => Some(TaskType::Documentation),
            "analysis" => Some(TaskType::Analysis),
            _ => None,
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
    /// Ollama configuration
    pub ollama: OllamaConfig,
    /// Maximum concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Minimum affinity to claim a task
    pub affinity_threshold: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "Agent".to_string(),
            role: AgentRole::General,
            ollama: OllamaConfig::default(),
            max_concurrent_tasks: 3,
            affinity_threshold: 0.5,
        }
    }
}

/// A task being worked on
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub task_type: TaskType,
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
    /// Ollama client
    pub ollama: OllamaClient,
    /// Current tasks
    pub current_tasks: Vec<AgentTask>,
    /// Task queue (claimed but not started)
    pub task_queue: Vec<String>,
    /// Completed task results
    pub results: Vec<AgentTask>,
    /// Metrics
    pub llm_calls: u64,
}

impl LlmAgent {
    /// Create a new LLM agent
    pub fn new(config: AgentConfig) -> Self {
        let node = Node::new();
        let ollama = OllamaClient::new(config.ollama.clone());
        
        Self {
            node,
            config,
            ollama,
            current_tasks: Vec::new(),
            task_queue: Vec::new(),
            results: Vec::new(),
            llm_calls: 0,
        }
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
        self.config.role.skills().get(&task_type).copied().unwrap_or(0.5)
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
    
    /// Execute a task using the LLM
    pub async fn execute_task(&mut self, task: &mut AgentTask) -> Result<String, OllamaError> {
        let prompt = format!(
            "Execute this task:\n\n\
             Type: {}\n\
             Description: {}\n\
             Priority: {:.1}\n\n\
             Provide your work result. Be concise but complete.",
            task.task_type.as_str(),
            task.description,
            task.priority
        );
        
        let system = self.config.role.system_prompt(&self.config.name);
        
        debug!("Agent {} executing task {}", self.config.name, task.id);
        
        let result = self.ollama.generate(&prompt, Some(&system)).await?;
        self.llm_calls += 1;
        
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
                let task_type_str = payload.get("task_type")?.as_str()?;
                let task_type = TaskType::from_str(task_type_str)?;
                let priority = payload.get("priority")?.as_f64().unwrap_or(0.5);
                
                // Skip if already in queue
                if self.task_queue.contains(&task_id.to_string()) {
                    return None;
                }
                
                // Check if we should claim
                if let Some(affinity) = self.should_claim(task_type, priority) {
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
                
                // Check for conflict
                if self.task_queue.contains(&task_id.to_string()) && claimer != self.node.id {
                    // Back off if they have higher affinity
                    let task_type_str = payload.get("task_type").and_then(|v| v.as_str());
                    let our_affinity = task_type_str
                        .and_then(TaskType::from_str)
                        .map(|tt| self.skill(tt))
                        .unwrap_or(0.5);
                    
                    if their_affinity > our_affinity + 0.1 {
                        self.task_queue.retain(|id| id != task_id);
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

/// Actions an agent can take
#[derive(Debug, Clone)]
pub enum AgentAction {
    /// Claim a task
    ClaimTask {
        task_id: String,
        affinity: f64,
    },
    /// Back off from a claimed task
    BackOff {
        task_id: String,
    },
    /// Emit task completion
    CompleteTask {
        task_id: String,
        result: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_agent_creation() {
        let config = AgentConfig {
            name: "TestAgent".to_string(),
            role: AgentRole::Coder,
            ..Default::default()
        };
        
        let agent = LlmAgent::new(config);
        
        assert_eq!(agent.name(), "TestAgent");
        assert!(agent.skill(TaskType::CodeWrite) > 0.9);
        assert!(agent.has_capacity());
    }
    
    #[test]
    fn test_should_claim() {
        let config = AgentConfig {
            role: AgentRole::Coder,
            affinity_threshold: 0.6,  // Higher threshold
            ..Default::default()
        };
        
        let agent = LlmAgent::new(config);
        
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
}
