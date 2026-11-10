//! External A2A agents as mesh participants.
//!
//! A remote agent joins the mesh as an ordinary [`LlmAgent`] whose backend is
//! an A2A client. Two things come from its Agent Card:
//!
//! - **Affinity.** Each card skill whose id or tags name a task type gives the
//!   agent a skill level for that type. The agent then claims, competes and
//!   backs off in settlement exactly like a local agent. A task type no skill
//!   names gets no level, so the agent never claims it.
//! - **Transport.** The SDK's client negotiates a binding from the card's
//!   `supportedInterfaces`.
//!
//! When the remote agent wins a task, the task prompt is sent as an A2A
//! message. The mesh's role system prompt is not sent: the remote agent has
//! its own instructions, and the A2A message has nowhere to put another's.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use a2a::{
    AgentCard, GetTaskRequest, Message, Part, Role, SendMessageConfiguration, SendMessageRequest,
    SendMessageResponse, Task, TaskState,
};
use a2a_client::agent_card::AgentCardResolver;
use a2a_client::transport::Transport;
use a2a_client::{A2AClient, A2AClientFactory};
use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;
use smesh_agent::{
    AgentConfig, AgentRole, GenerateRequest, GenerateResponse, LlmAgent, LlmBackend, LlmError,
    LlmProvider, TaskType,
};
use tracing::{debug, warn};

use crate::card::{verify_card, CardSignatureError};

/// Words that, as a card skill id or tag, mean a skill covers a task type.
/// Compared after lowercasing and turning `-` and spaces into `_`.
fn synonyms(task_type: TaskType) -> &'static [&'static str] {
    match task_type {
        TaskType::CodeWrite => &[
            "code_write",
            "coding",
            "code",
            "programming",
            "implementation",
            "code_generation",
        ],
        TaskType::CodeReview => &["code_review", "review", "audit", "security_review"],
        TaskType::Testing => &["testing", "test", "tests", "qa", "test_generation"],
        TaskType::Documentation => &["documentation", "docs", "writing", "technical_writing"],
        TaskType::Analysis => &[
            "analysis",
            "research",
            "analytics",
            "investigation",
            "summarization",
        ],
    }
}

fn normalize(word: &str) -> String {
    word.trim().to_lowercase().replace(['-', ' '], "_")
}

/// The skill levels a card earns: `level` for every task type some skill's id
/// or tags name, nothing for the rest.
pub fn affinities_from_card(card: &AgentCard, level: f64) -> HashMap<TaskType, f64> {
    let words: Vec<String> = card
        .skills
        .iter()
        .flat_map(|s| std::iter::once(&s.id).chain(s.tags.iter()))
        .map(|w| normalize(w))
        .collect();

    TaskType::ALL
        .iter()
        .filter(|t| {
            synonyms(**t)
                .iter()
                .any(|syn| words.iter().any(|w| w == syn))
        })
        .map(|t| (*t, level))
        .collect()
}

/// Which cards to accept on discovery.
#[derive(Debug, Clone)]
pub enum CardTrust {
    /// Accept any card, signed or not.
    AnyCard,
    /// Accept only cards signed by one of these keys.
    SignedBy(Vec<VerifyingKey>),
}

/// Settings for a remote agent.
#[derive(Debug, Clone)]
pub struct RemoteAgentOptions {
    /// Skill level for each task type the card covers. Local roles range
    /// 0.3 to 0.95; 0.9 lets a specialist compete with the best local agent
    /// without always beating it.
    pub affinity: f64,
    /// How long to wait for a remote task to finish.
    pub timeout: Duration,
    /// Delay between polls while a remote task is still running.
    pub poll_interval: Duration,
    /// Which cards to accept.
    pub trust: CardTrust,
}

impl Default for RemoteAgentOptions {
    fn default() -> Self {
        Self {
            affinity: 0.9,
            timeout: Duration::from_secs(300),
            poll_interval: Duration::from_millis(500),
            trust: CardTrust::AnyCard,
        }
    }
}

/// Why a remote agent could not be added.
#[derive(Debug, thiserror::Error)]
pub enum RemoteError {
    /// Fetching or using the card failed.
    #[error("A2A: {0}")]
    A2a(#[from] a2a::A2AError),
    /// The card's signature was required and did not check out.
    #[error("agent card rejected: {0}")]
    Card(#[from] CardSignatureError),
}

/// An [`LlmBackend`] that hands each request to a remote A2A agent.
pub struct A2aBackend {
    client: A2AClient<Box<dyn Transport>>,
    name: String,
    timeout: Duration,
    poll_interval: Duration,
}

impl A2aBackend {
    /// A backend for the agent described by `card`, negotiating a transport
    /// from its interfaces.
    pub async fn connect(
        card: &AgentCard,
        options: &RemoteAgentOptions,
    ) -> Result<Self, RemoteError> {
        let client = A2AClientFactory::builder()
            .build()
            .create_from_card(card)
            .await?;
        Ok(Self {
            client,
            name: card.name.clone(),
            timeout: options.timeout,
            poll_interval: options.poll_interval,
        })
    }

    async fn wait_for(&self, mut task: Task, started: Instant) -> Result<Task, LlmError> {
        while !task.status.state.is_terminal() && !is_interrupted(&task.status.state) {
            if started.elapsed() >= self.timeout {
                return Err(LlmError::Timeout(self.timeout));
            }
            tokio::time::sleep(self.poll_interval).await;
            task = self
                .client
                .get_task(&GetTaskRequest {
                    id: task.id.clone(),
                    history_length: Some(0),
                    tenant: None,
                })
                .await
                .map_err(|e| LlmError::RequestFailed(e.to_string()))?;
            debug!(agent = %self.name, task = %task.id, state = ?task.status.state, "polled remote task");
        }
        Ok(task)
    }
}

fn is_interrupted(state: &TaskState) -> bool {
    matches!(state, TaskState::InputRequired | TaskState::AuthRequired)
}

fn texts(parts: &[Part]) -> Vec<&str> {
    parts.iter().filter_map(|p| p.as_text()).collect()
}

/// The text a finished remote task produced: its artifacts' text, or failing
/// that its final status message.
pub fn task_output(task: &Task) -> Option<String> {
    let from_artifacts: Vec<&str> = task
        .artifacts
        .iter()
        .flatten()
        .flat_map(|a| texts(&a.parts))
        .collect();
    let chosen = if from_artifacts.is_empty() {
        task.status
            .message
            .as_ref()
            .map(|m| texts(&m.parts))
            .unwrap_or_default()
    } else {
        from_artifacts
    };
    (!chosen.is_empty()).then(|| chosen.join("\n\n"))
}

/// Turn a remote task's end state into the backend's answer.
pub fn outcome(task: &Task) -> Result<String, LlmError> {
    let reason = || {
        task.status
            .message
            .as_ref()
            .map(|m| texts(&m.parts).join(" "))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "no reason given".into())
    };
    match task.status.state {
        TaskState::Completed => task_output(task)
            .ok_or_else(|| LlmError::InvalidResponse("completed task has no text output".into())),
        TaskState::InputRequired | TaskState::AuthRequired => {
            Err(LlmError::UnsupportedFeature(format!(
                "remote agent asked for more input ({:?}), which a mesh task cannot give: {}",
                task.status.state,
                reason()
            )))
        }
        ref state => Err(LlmError::GenerationFailed(format!(
            "remote task ended {state:?}: {}",
            reason()
        ))),
    }
}

#[async_trait]
impl LlmBackend for A2aBackend {
    fn provider(&self) -> LlmProvider {
        LlmProvider::A2a
    }

    fn model(&self) -> &str {
        &self.name
    }

    async fn is_available(&self) -> bool {
        true
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec![self.name.clone()])
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, LlmError> {
        let started = Instant::now();
        let req = SendMessageRequest {
            message: Message::new(Role::User, vec![Part::text(request.prompt)]),
            configuration: Some(SendMessageConfiguration {
                accepted_output_modes: Some(vec!["text/plain".into()]),
                task_push_notification_config: None,
                history_length: Some(0),
                return_immediately: None,
            }),
            metadata: None,
            tenant: None,
        };
        let response = tokio::time::timeout(self.timeout, self.client.send_message(&req))
            .await
            .map_err(|_| LlmError::Timeout(self.timeout))?
            .map_err(|e| LlmError::RequestFailed(e.to_string()))?;

        let content = match response {
            SendMessageResponse::Message(message) => {
                let text = texts(&message.parts).join("\n\n");
                if text.is_empty() {
                    return Err(LlmError::InvalidResponse("reply has no text".into()));
                }
                text
            }
            SendMessageResponse::Task(task) => outcome(&self.wait_for(task, started).await?)?,
        };

        Ok(GenerateResponse {
            content,
            provider: LlmProvider::A2a,
            model: self.name.clone(),
            latency: started.elapsed(),
            input_tokens: None,
            output_tokens: None,
            stop_reason: None,
        })
    }
}

/// A remote A2A agent, discovered and ready to join a mesh.
pub struct RemoteAgent {
    /// The card it was discovered with.
    pub card: AgentCard,
    /// Skill levels derived from the card.
    pub skills: HashMap<TaskType, f64>,
    backend: Arc<A2aBackend>,
}

impl RemoteAgent {
    /// Fetch the card at `{base_url}/.well-known/agent-card.json`, check it
    /// against `options.trust`, and connect.
    pub async fn discover(
        base_url: &str,
        options: RemoteAgentOptions,
    ) -> Result<Self, RemoteError> {
        let card = AgentCardResolver::new(None).resolve(base_url).await?;
        Self::from_card(card, options).await
    }

    /// Use a card already in hand.
    pub async fn from_card(
        card: AgentCard,
        options: RemoteAgentOptions,
    ) -> Result<Self, RemoteError> {
        if let CardTrust::SignedBy(keys) = &options.trust {
            verify_card(&card, keys)?;
        }
        let skills = affinities_from_card(&card, options.affinity);
        if skills.is_empty() {
            warn!(agent = %card.name, "no card skill names a mesh task type; it will never claim");
        }
        let backend = Arc::new(A2aBackend::connect(&card, &options).await?);
        Ok(Self {
            card,
            skills,
            backend,
        })
    }

    /// A mesh agent for it: a general-role agent whose skills are the card's.
    ///
    /// Each call makes a new agent over the same connection, so one discovered
    /// remote can sit in every coordinator a gateway builds.
    pub fn to_llm_agent(&self) -> LlmAgent {
        let config = AgentConfig {
            name: format!("a2a:{}", self.card.name),
            role: AgentRole::General,
            openrouter: None,
            ..AgentConfig::default()
        }
        .with_skill_overrides(
            // Every type is listed so the role's defaults never leak in: a type
            // the card does not cover gets 0.0, below any claim threshold.
            TaskType::ALL
                .iter()
                .map(|t| (*t, self.skills.get(t).copied().unwrap_or(0.0)))
                .collect(),
        );
        LlmAgent::new(config, Arc::clone(&self.backend) as Arc<dyn LlmBackend>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use a2a::{Artifact, TaskStatus};

    fn card_with(skills: &[(&str, &[&str])]) -> AgentCard {
        let mut card = crate::card::mesh_card(&crate::card::MeshCardConfig::new("http://x"));
        card.skills = skills
            .iter()
            .map(|(id, tags)| a2a::AgentSkill {
                id: id.to_string(),
                name: id.to_string(),
                description: String::new(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                examples: None,
                input_modes: None,
                output_modes: None,
                security_requirements: None,
            })
            .collect();
        card
    }

    fn task(state: TaskState, artifact: Option<&str>, status: Option<&str>) -> Task {
        Task {
            id: "t".into(),
            context_id: "c".into(),
            status: TaskStatus {
                state,
                message: status.map(|s| Message::new(Role::Agent, vec![Part::text(s)])),
                timestamp: None,
            },
            artifacts: artifact.map(|a| {
                vec![Artifact {
                    artifact_id: "a".into(),
                    name: None,
                    description: None,
                    parts: vec![Part::text(a)],
                    metadata: None,
                    extensions: None,
                }]
            }),
            history: None,
            metadata: None,
        }
    }

    #[test]
    fn test_skills_map_by_id_and_tag() {
        let card = card_with(&[("security-review", &[]), ("x", &["Technical Writing"])]);
        let skills = affinities_from_card(&card, 0.8);
        assert_eq!(
            skills,
            HashMap::from([(TaskType::CodeReview, 0.8), (TaskType::Documentation, 0.8)])
        );
    }

    #[test]
    fn test_unrelated_skills_give_no_affinity() {
        let card = card_with(&[("weather", &["forecast"])]);
        assert!(affinities_from_card(&card, 0.9).is_empty());
    }

    #[test]
    fn test_substrings_do_not_count() {
        // "test" is a synonym; "latest" contains it but is not it.
        let card = card_with(&[("latest-news", &["latest"])]);
        assert!(affinities_from_card(&card, 0.9).is_empty());
    }

    #[test]
    fn test_completed_task_returns_artifact_text() {
        let t = task(TaskState::Completed, Some("result"), Some("done"));
        assert_eq!(outcome(&t).unwrap(), "result");
    }

    #[test]
    fn test_completed_task_without_artifact_falls_back_to_status() {
        let t = task(TaskState::Completed, None, Some("inline answer"));
        assert_eq!(outcome(&t).unwrap(), "inline answer");
    }

    #[test]
    fn test_failed_task_is_an_error_with_reason() {
        let t = task(TaskState::Failed, None, Some("quota exceeded"));
        let err = outcome(&t).unwrap_err().to_string();
        assert!(
            err.contains("Failed") && err.contains("quota exceeded"),
            "{err}"
        );
    }

    #[test]
    fn test_input_required_is_not_treated_as_success() {
        let t = task(TaskState::InputRequired, None, Some("which repo?"));
        assert!(matches!(outcome(&t), Err(LlmError::UnsupportedFeature(_))));
    }
}
