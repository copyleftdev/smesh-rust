//! How well an agent suits a task, judged from what the agent says it does.
//!
//! The built-in role tables only know five software task types. An
//! [`AffinitySource`] judges any agent against any task from plain language:
//! the agent's capability ("a translator between English and Portuguese") and
//! the task's description. That is what lets a mesh hold agents of any kind.
//!
//! A source is asked about one agent at a time and never sees the others, so
//! the answer is that agent's own claim strength. Choosing between agents is
//! still left to claims and back-offs in the field.

use async_trait::async_trait;

use crate::coordinator::TaskDefinition;

/// Judges how well an agent suits a task.
#[async_trait]
pub trait AffinitySource: Send + Sync {
    /// How well an agent that does `capability` suits `task`, from 0 to 1.
    ///
    /// `None` means no judgement (the service failed, say); the agent then
    /// falls back to its role table.
    async fn assess(&self, task: &TaskDefinition, capability: &str) -> Option<f64>;
}
