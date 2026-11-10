//! End to end over real HTTP: the gateway served with axum, driven by the A2A
//! project's own client, and a mesh that recruits a remote A2A agent.
//!
//! Backends are deterministic fakes, so no API key or network beyond loopback
//! is needed.

use std::sync::Arc;
use std::time::Duration;

use a2a::{
    AgentCard, Message, Part, Role, SendMessageRequest, SendMessageResponse, StreamResponse, Task,
    TaskState,
};
use a2a_client::agent_card::AgentCardResolver;
use a2a_client::A2AClientFactory;
use async_trait::async_trait;
use ed25519_dalek::SigningKey;
use futures::StreamExt;
use smesh_a2a::{
    mesh_card, router, sign_card, CardTrust, CoordinatorFactory, MeshCardConfig, RemoteAgent,
    RemoteAgentOptions, RemoteError,
};
use smesh_agent::{
    AgentCoordinator, AgentRole, CoordinatorConfig, GenerateRequest, GenerateResponse, LlmBackend,
    LlmError, LlmProvider, TaskDefinition,
};

/// Answers every prompt with `"<tag>: <first line of the task description>"`,
/// or fails if `fail` is set.
struct Fake {
    tag: &'static str,
    fail: bool,
}

/// Never answers within a test's lifetime.
struct Stalled;

#[async_trait]
impl LlmBackend for Stalled {
    fn provider(&self) -> LlmProvider {
        LlmProvider::OpenRouter
    }
    fn model(&self) -> &str {
        "stalled"
    }
    async fn is_available(&self) -> bool {
        true
    }
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec![])
    }
    async fn generate(&self, _request: GenerateRequest) -> Result<GenerateResponse, LlmError> {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        Err(LlmError::Timeout(Duration::from_secs(3600)))
    }
}

#[async_trait]
impl LlmBackend for Fake {
    fn provider(&self) -> LlmProvider {
        LlmProvider::OpenRouter
    }
    fn model(&self) -> &str {
        self.tag
    }
    async fn is_available(&self) -> bool {
        true
    }
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec![])
    }
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, LlmError> {
        if self.fail {
            return Err(LlmError::GenerationFailed("fake backend down".into()));
        }
        let description = request
            .prompt
            .lines()
            .find_map(|l| l.strip_prefix("Description: "))
            .unwrap_or(&request.prompt)
            .to_string();
        Ok(GenerateResponse {
            content: format!("{}: {}", self.tag, description),
            provider: LlmProvider::OpenRouter,
            model: self.tag.into(),
            latency: Duration::ZERO,
            input_tokens: None,
            output_tokens: None,
            stop_reason: None,
        })
    }
}

fn factory(roles: Vec<AgentRole>, tag: &'static str, fail: bool) -> CoordinatorFactory {
    Arc::new(move || {
        let config = CoordinatorConfig {
            n_agents: roles.len(),
            roles: roles.clone(),
            max_ticks: 20,
            ..CoordinatorConfig::default()
        };
        AgentCoordinator::new(config, Box::new(move || Arc::new(Fake { tag, fail })))
    })
}

/// Serve a gateway on a free loopback port; returns its base URL.
async fn spawn_gateway(factory: CoordinatorFactory, sign_with: Option<&SigningKey>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let mut card = mesh_card(&MeshCardConfig::new(base.clone()));
    if let Some(key) = sign_with {
        sign_card(&mut card, key);
    }
    tokio::spawn(async move {
        axum::serve(listener, router(factory, card)).await.unwrap();
    });
    base
}

async fn client_for(base: &str) -> a2a_client::A2AClient<Box<dyn a2a_client::Transport>> {
    let card: AgentCard = AgentCardResolver::new(None).resolve(base).await.unwrap();
    A2AClientFactory::builder()
        .build()
        .create_from_card(&card)
        .await
        .unwrap()
}

fn request(text: &str, task_type: Option<&str>) -> SendMessageRequest {
    let mut message = Message::new(Role::User, vec![Part::text(text)]);
    if let Some(t) = task_type {
        message.metadata = Some([("taskType".to_string(), serde_json::json!(t))].into());
    }
    SendMessageRequest {
        message,
        configuration: None,
        metadata: None,
        tenant: None,
    }
}

fn artifact_text(task: &Task) -> String {
    task.artifacts
        .iter()
        .flatten()
        .flat_map(|a| a.parts.iter().filter_map(|p| p.as_text()))
        .collect::<Vec<_>>()
        .join("\n")
}

async fn send(base: &str, text: &str, task_type: Option<&str>) -> Task {
    match client_for(base)
        .await
        .send_message(&request(text, task_type))
        .await
        .unwrap()
    {
        SendMessageResponse::Task(task) => task,
        SendMessageResponse::Message(m) => panic!("expected a task, got message {m:?}"),
    }
}

#[tokio::test]
async fn test_gateway_serves_a_v1_agent_card() {
    let base = spawn_gateway(factory(vec![AgentRole::Coder], "local", false), None).await;
    let card = AgentCardResolver::new(None).resolve(&base).await.unwrap();
    assert_eq!(card.name, "SMESH mesh");
    assert_eq!(card.supported_interfaces.len(), 2);
    assert!(card.skills.iter().any(|s| s.id == "code_review"));
}

#[tokio::test]
async fn test_send_message_completes_on_the_settled_winner() {
    let base = spawn_gateway(
        factory(
            vec![AgentRole::Coder, AgentRole::Reviewer, AgentRole::Writer],
            "local",
            false,
        ),
        None,
    )
    .await;

    let task = send(&base, "look over the retry loop", Some("code_review")).await;

    assert_eq!(task.status.state, TaskState::Completed);
    assert_eq!(artifact_text(&task), "local: look over the retry loop");
    // The reviewer has the highest code_review affinity, so it must win.
    let status = task.status.message.as_ref().and_then(|m| m.text()).unwrap();
    assert!(status.contains("Reviewer"), "status was {status:?}");
}

#[tokio::test]
async fn test_streaming_reports_settlement_rounds_then_completion() {
    let base = spawn_gateway(
        factory(vec![AgentRole::Coder, AgentRole::Reviewer], "local", false),
        None,
    )
    .await;
    let client = client_for(&base).await;
    let events: Vec<StreamResponse> = client
        .send_streaming_message(&request("Review this diff", None))
        .await
        .unwrap()
        .map(|e| e.unwrap())
        .collect()
        .await;

    let smesh_events: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            StreamResponse::StatusUpdate(u) => u
                .metadata
                .as_ref()?
                .get("smesh")?
                .get("event")?
                .as_str()
                .map(String::from),
            _ => None,
        })
        .collect();
    assert_eq!(
        smesh_events.first().map(String::as_str),
        Some("task_available")
    );
    assert!(
        smesh_events.iter().any(|e| e == "round"),
        "{smesh_events:?}"
    );
    assert!(
        smesh_events.iter().any(|e| e == "settled"),
        "{smesh_events:?}"
    );
    assert_eq!(smesh_events.last().map(String::as_str), Some("completed"));
    assert!(events
        .iter()
        .any(|e| matches!(e, StreamResponse::ArtifactUpdate(_))));
}

#[tokio::test]
async fn test_task_no_agent_can_claim_is_rejected_not_failed() {
    // A Writer's testing skill is 0.3, under the default 0.5 claim threshold.
    let base = spawn_gateway(factory(vec![AgentRole::Writer], "local", false), None).await;
    let task = send(&base, "anything", Some("testing")).await;
    assert_eq!(task.status.state, TaskState::Rejected);
    assert!(task.artifacts.is_none());
}

#[tokio::test]
async fn test_backend_failure_fails_the_task() {
    let base = spawn_gateway(factory(vec![AgentRole::Coder], "local", true), None).await;
    let task = send(&base, "x", Some("code_write")).await;
    assert_eq!(task.status.state, TaskState::Failed);
    let status = task.status.message.as_ref().and_then(|m| m.text()).unwrap();
    assert!(status.contains("fake backend down"), "{status}");
}

#[tokio::test]
async fn test_unknown_task_type_fails_with_a_reason() {
    let base = spawn_gateway(factory(vec![AgentRole::Coder], "local", false), None).await;
    let task = send(&base, "x", Some("poetry")).await;
    assert_eq!(task.status.state, TaskState::Failed);
}

/// Run one task on a coordinator holding `local` agents plus a remote agent.
async fn mesh_with_remote(local: Vec<AgentRole>, remote_base: &str, task_type: &str) -> String {
    let mut coordinator = AgentCoordinator::new(
        CoordinatorConfig {
            n_agents: local.len(),
            roles: if local.is_empty() {
                vec![AgentRole::General]
            } else {
                local.clone()
            },
            max_ticks: 20,
            ..CoordinatorConfig::default()
        },
        Box::new(|| {
            Arc::new(Fake {
                tag: "local",
                fail: false,
            })
        }),
    );
    let remote = RemoteAgent::discover(remote_base, RemoteAgentOptions::default())
        .await
        .unwrap();
    coordinator.add_agent(remote.to_llm_agent());

    let id = coordinator.add_task(TaskDefinition {
        task_type: task_type.into(),
        description: "check the auth flow".into(),
        priority: 0.5,
    });
    coordinator.settle_traced();
    coordinator.execute_settled(&id).await.unwrap().result
}

#[tokio::test]
async fn test_remote_agent_wins_when_it_is_the_strongest_claimant() {
    // The remote "agent" is itself a mesh gateway; its card covers every type.
    let remote = spawn_gateway(factory(vec![AgentRole::Reviewer], "remote", false), None).await;
    // Local Writer's code_review is 0.5; the remote's 0.9 beats it by more
    // than the back-off margin, so the Writer yields.
    let result = mesh_with_remote(vec![AgentRole::Writer], &remote, "code_review").await;
    assert!(result.starts_with("remote: "), "{result}");
}

#[tokio::test]
async fn test_strong_local_specialist_still_beats_remote() {
    let remote = spawn_gateway(factory(vec![AgentRole::Reviewer], "remote", false), None).await;
    // Local Reviewer's code_review is 0.95 against the remote's 0.9.
    let result = mesh_with_remote(vec![AgentRole::Reviewer], &remote, "code_review").await;
    assert!(result.starts_with("local: "), "{result}");
}

#[tokio::test]
async fn test_signed_card_is_required_when_trust_is_keyed() {
    let trusted = SigningKey::from_bytes(&[7; 32]);
    let stranger = SigningKey::from_bytes(&[8; 32]);
    let keyed = |k: &SigningKey| RemoteAgentOptions {
        trust: CardTrust::SignedBy(vec![k.verifying_key()]),
        ..RemoteAgentOptions::default()
    };

    let signed = spawn_gateway(
        factory(vec![AgentRole::Coder], "remote", false),
        Some(&trusted),
    )
    .await;
    assert!(RemoteAgent::discover(&signed, keyed(&trusted))
        .await
        .is_ok());
    assert!(matches!(
        RemoteAgent::discover(&signed, keyed(&stranger)).await,
        Err(RemoteError::Card(_))
    ));

    let unsigned = spawn_gateway(factory(vec![AgentRole::Coder], "remote", false), None).await;
    assert!(matches!(
        RemoteAgent::discover(&unsigned, keyed(&trusted)).await,
        Err(RemoteError::Card(_))
    ));
}

#[tokio::test]
async fn test_cancel_aborts_a_running_task() {
    let stalled: CoordinatorFactory = Arc::new(|| {
        AgentCoordinator::new(
            CoordinatorConfig {
                n_agents: 1,
                roles: vec![AgentRole::Coder],
                max_ticks: 20,
                ..CoordinatorConfig::default()
            },
            Box::new(|| Arc::new(Stalled)),
        )
    });
    let base = spawn_gateway(stalled, None).await;
    let client = client_for(&base).await;

    let mut req = request("implement it", Some("code_write"));
    req.configuration = Some(a2a::SendMessageConfiguration {
        accepted_output_modes: None,
        task_push_notification_config: None,
        history_length: None,
        return_immediately: Some(true),
    });
    let SendMessageResponse::Task(task) = client.send_message(&req).await.unwrap() else {
        panic!("expected a task");
    };
    assert!(!task.status.state.is_terminal(), "{:?}", task.status.state);

    let canceled = client
        .cancel_task(&a2a::CancelTaskRequest {
            id: task.id.clone(),
            metadata: None,
            tenant: None,
        })
        .await
        .unwrap();
    assert_eq!(canceled.status.state, TaskState::Canceled);

    // The aborted job must not later overwrite CANCELED with another state.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let later = client
        .get_task(&a2a::GetTaskRequest {
            id: task.id,
            history_length: None,
            tenant: None,
        })
        .await
        .unwrap();
    assert_eq!(later.status.state, TaskState::Canceled);
}
