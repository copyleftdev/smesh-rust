//! The mesh as an A2A server.
//!
//! Each A2A task becomes one mesh task. Its lifecycle maps onto the signal
//! lifecycle:
//!
//! | A2A state   | Mesh event                                                   |
//! |-------------|--------------------------------------------------------------|
//! | `SUBMITTED` | Request accepted (set by the SDK before the executor runs)   |
//! | `WORKING`   | `task_available` emitted; one update per settlement round     |
//! | `WORKING`   | Settled: the surviving claimant starts executing             |
//! | `COMPLETED` | The winner's output is attached as an artifact               |
//! | `REJECTED`  | Settlement ended with no claim: no agent had the affinity     |
//! | `FAILED`    | The winner's backend failed, or the message was unusable     |
//! | `CANCELED`  | The client cancelled; the running job is aborted             |
//!
//! `REJECTED` rather than `FAILED` for an unclaimed task matters: nothing broke,
//! the mesh declined, which is what a stale `task_available` signal fading
//! unanswered means.
//!
//! Every status update carries an `smesh` metadata object with the structured
//! event (round number, claims, back-offs, winner), so a client can follow the
//! settlement rather than only read the prose in the status message.
//!
//! Each task gets its own coordinator from the factory, so concurrent tasks do
//! not share a field and cannot claim on each other's signals.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use a2a::{
    A2AError, Artifact, Message, Part, Role, StreamResponse, TaskArtifactUpdateEvent, TaskState,
    TaskStatus, TaskStatusUpdateEvent,
};
use a2a_server::{
    AgentExecutor, DefaultRequestHandler, ExecutorContext, InMemoryTaskStore, StaticAgentCard,
};
use futures::channel::mpsc;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value};
use smesh_agent::{AgentCoordinator, ExecuteError, RoundTrace};
use tokio::task::AbortHandle;
use tracing::{info, warn};

use crate::classify::task_from_message;

/// Makes a fresh coordinator (with its agents) for one A2A task.
pub type CoordinatorFactory = Arc<dyn Fn() -> AgentCoordinator + Send + Sync>;

/// Metadata key under which status updates carry the structured mesh event.
pub const SMESH_METADATA_KEY: &str = "smesh";

/// [`AgentExecutor`] that settles each A2A task on a SMESH mesh.
pub struct MeshExecutor {
    factory: CoordinatorFactory,
    running: Arc<Mutex<HashMap<String, AbortHandle>>>,
}

impl MeshExecutor {
    /// An executor that builds a coordinator per task with `factory`.
    pub fn new(factory: CoordinatorFactory) -> Self {
        Self {
            factory,
            running: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// Sends events for one task.
struct Events {
    tx: mpsc::UnboundedSender<Result<StreamResponse, A2AError>>,
    task_id: String,
    context_id: String,
}

impl Events {
    fn status(&self, state: TaskState, text: impl Into<String>, smesh: Value) {
        let mut message = Message::new(Role::Agent, vec![Part::text(text)]);
        message.task_id = Some(self.task_id.clone());
        message.context_id = Some(self.context_id.clone());
        let event = StreamResponse::StatusUpdate(TaskStatusUpdateEvent {
            task_id: self.task_id.clone(),
            context_id: self.context_id.clone(),
            status: TaskStatus {
                state,
                message: Some(message),
                timestamp: Some(chrono_now()),
            },
            metadata: Some(HashMap::from([(SMESH_METADATA_KEY.to_string(), smesh)])),
        });
        // A closed receiver means the client went away; nothing to do.
        let _ = self.tx.unbounded_send(Ok(event));
    }

    fn artifact(&self, name: &str, text: String, smesh: Value) {
        let event = StreamResponse::ArtifactUpdate(TaskArtifactUpdateEvent {
            task_id: self.task_id.clone(),
            context_id: self.context_id.clone(),
            artifact: Artifact {
                artifact_id: a2a::new_artifact_id(),
                name: Some(name.to_string()),
                description: None,
                parts: vec![Part::text(text).with_media_type("text/plain")],
                metadata: Some(HashMap::from([(SMESH_METADATA_KEY.to_string(), smesh)])),
                extensions: None,
            },
            append: None,
            last_chunk: Some(true),
            metadata: None,
        });
        let _ = self.tx.unbounded_send(Ok(event));
    }
}

fn chrono_now() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now()
}

fn round_json(round: usize, trace: &RoundTrace) -> Value {
    json!({
        "event": "round",
        "round": round,
        "claims": trace.claims.iter().map(|(node, _, affinity)| {
            json!({"node": node, "affinity": affinity})
        }).collect::<Vec<_>>(),
        "backoffs": trace.backoffs.iter().map(|(node, _)| node).collect::<Vec<_>>(),
    })
}

/// Run one A2A task on a fresh coordinator, reporting through `events`.
async fn run_task(mut coordinator: AgentCoordinator, message: Option<Message>, events: Events) {
    let Some(message) = message else {
        events.status(
            TaskState::Failed,
            "no message to execute",
            json!({"event": "error"}),
        );
        return;
    };
    let task = match task_from_message(&message) {
        Ok(task) => task,
        Err(e) => {
            events.status(
                TaskState::Failed,
                format!("cannot make a mesh task from this message: {e}"),
                json!({"event": "error", "reason": e.to_string()}),
            );
            return;
        }
    };

    let task_type = task.task_type.clone();
    let mesh_task_id = coordinator.add_task(task);
    events.status(
        TaskState::Working,
        format!("task_available emitted as {task_type}"),
        json!({"event": "task_available", "meshTaskId": mesh_task_id, "taskType": task_type}),
    );

    let roster: HashMap<String, String> = coordinator
        .agent_roster()
        .into_iter()
        .map(|a| (a.id, a.name))
        .collect();
    let name = |node: &str| {
        roster
            .get(node)
            .cloned()
            .unwrap_or_else(|| node.to_string())
    };

    for (i, trace) in coordinator.settle_traced().iter().enumerate() {
        if trace.claims.is_empty() && trace.backoffs.is_empty() {
            continue;
        }
        let mut parts: Vec<String> = trace
            .claims
            .iter()
            .map(|(n, _, a)| format!("{} claims ({:.0}%)", name(n), a * 100.0))
            .collect();
        parts.extend(
            trace
                .backoffs
                .iter()
                .map(|(n, _)| format!("{} backs off", name(n))),
        );
        events.status(
            TaskState::Working,
            format!("round {}: {}", i + 1, parts.join(", ")),
            round_json(i + 1, trace),
        );
    }

    let Some(winner) = coordinator.winner_for(&mesh_task_id) else {
        events.status(
            TaskState::Rejected,
            format!("no agent claimed this {task_type} task"),
            json!({"event": "unclaimed", "meshTaskId": mesh_task_id}),
        );
        return;
    };
    events.status(
        TaskState::Working,
        format!("settled on {}; executing", name(&winner)),
        json!({"event": "settled", "winner": winner, "winnerName": name(&winner)}),
    );

    match coordinator.execute_settled(&mesh_task_id).await {
        Ok(done) => {
            info!(task = %events.task_id, agent = %done.agent_name, "mesh task completed");
            let smesh =
                json!({"event": "completed", "agent": done.node_id, "agentName": done.agent_name});
            events.artifact("result", done.result, smesh.clone());
            events.status(
                TaskState::Completed,
                format!("completed by {}", done.agent_name),
                smesh,
            );
        }
        Err(ExecuteError::Unclaimed(_)) => events.status(
            TaskState::Rejected,
            format!("no agent claimed this {task_type} task"),
            json!({"event": "unclaimed", "meshTaskId": mesh_task_id}),
        ),
        Err(e) => {
            warn!(task = %events.task_id, error = %e, "mesh task failed");
            events.status(
                TaskState::Failed,
                e.to_string(),
                json!({"event": "error", "reason": e.to_string()}),
            );
        }
    }
}

impl AgentExecutor for MeshExecutor {
    fn execute(
        &self,
        ctx: ExecutorContext,
    ) -> BoxStream<'static, Result<StreamResponse, A2AError>> {
        let (tx, rx) = mpsc::unbounded();
        let events = Events {
            tx,
            task_id: ctx.task_id.clone(),
            context_id: ctx.context_id.clone(),
        };
        let coordinator = (self.factory)();
        let running = Arc::clone(&self.running);
        let task_id = ctx.task_id.clone();

        let job = tokio::spawn(async move {
            run_task(coordinator, ctx.message, events).await;
        });
        running
            .lock()
            .expect("running-task map poisoned")
            .insert(task_id.clone(), job.abort_handle());

        // Forget the handle once the stream ends, whichever way it ends.
        let cleanup = Arc::clone(&self.running);
        let forget = futures::stream::once(async move {
            cleanup
                .lock()
                .expect("running-task map poisoned")
                .remove(&task_id);
        })
        .filter_map(|()| futures::future::ready(None));
        rx.chain(forget).boxed()
    }

    fn cancel(&self, ctx: ExecutorContext) -> BoxStream<'static, Result<StreamResponse, A2AError>> {
        if let Some(handle) = self
            .running
            .lock()
            .expect("running-task map poisoned")
            .remove(&ctx.task_id)
        {
            handle.abort();
        }
        let event = StreamResponse::StatusUpdate(TaskStatusUpdateEvent {
            task_id: ctx.task_id,
            context_id: ctx.context_id,
            status: TaskStatus {
                state: TaskState::Canceled,
                message: None,
                timestamp: Some(chrono_now()),
            },
            metadata: Some(HashMap::from([(
                SMESH_METADATA_KEY.to_string(),
                json!({"event": "canceled"}),
            )])),
        });
        futures::stream::once(async move { Ok(event) }).boxed()
    }
}

/// An axum router serving the mesh over A2A: the Agent Card at
/// `/.well-known/agent-card.json`, JSON-RPC at `POST /`, and the HTTP+JSON
/// binding (`/message:send`, `/tasks/{id}`, ...) beside it.
pub fn router(factory: CoordinatorFactory, card: a2a::AgentCard) -> axum::Router {
    let handler = Arc::new(
        DefaultRequestHandler::new(MeshExecutor::new(factory), InMemoryTaskStore::new())
            .with_capabilities(card.capabilities.clone()),
    );
    a2a_server::agent_card::agent_card_router(Arc::new(StaticAgentCard::new(card)))
        .merge(a2a_server::jsonrpc::jsonrpc_router(Arc::clone(&handler)))
        .merge(a2a_server::rest::rest_router(handler))
}

/// Serve the mesh over A2A on `listener` until the process stops.
pub async fn serve(
    listener: tokio::net::TcpListener,
    factory: CoordinatorFactory,
    card: a2a::AgentCard,
) -> std::io::Result<()> {
    axum::serve(listener, router(factory, card)).await
}
