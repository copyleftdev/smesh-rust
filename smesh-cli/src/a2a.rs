//! `smesh a2a`: serve the mesh over A2A, and talk to A2A agents.

use std::sync::Arc;

use a2a::{Message, Part, Role, SendMessageRequest, StreamResponse};
use a2a_client::agent_card::AgentCardResolver;
use a2a_client::A2AClientFactory;
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use ed25519_dalek::SigningKey;
use futures::StreamExt;
use smesh_a2a::{
    key_id, mesh_card, sign_card, CoordinatorFactory, MeshCardConfig, RemoteAgent,
    RemoteAgentOptions,
};
use smesh_agent::{AgentCoordinator, AgentRole, CoordinatorConfig, OpenRouterConfig};

/// `smesh a2a` subcommands.
#[derive(Subcommand)]
pub enum A2aCommand {
    /// Serve the mesh as an A2A v1.0 agent
    Serve {
        /// Address to listen on
        #[arg(long, default_value = "127.0.0.1:7410")]
        listen: String,

        /// URL clients reach the gateway at (default: http://<listen>)
        #[arg(long)]
        public_url: Option<String>,

        /// Number of local agents (roles cycle Coder, Reviewer, Analyst, Writer)
        #[arg(short, long, default_value = "4")]
        agents: usize,

        /// OpenRouter model for local agents
        #[arg(short, long, default_value = "google/gemini-2.5-flash-lite")]
        model: String,

        /// Base URL of a remote A2A agent to recruit into the mesh (repeatable)
        #[arg(long)]
        recruit: Vec<String>,

        /// File holding a hex Ed25519 seed (32 bytes) to sign the Agent Card with
        #[arg(long)]
        sign_key: Option<std::path::PathBuf>,
    },

    /// Send a task to an A2A agent and stream what happens
    Send {
        /// Base URL of the agent
        #[arg(long, default_value = "http://127.0.0.1:7410")]
        url: String,

        /// Task type (code_write, code_review, testing, documentation, analysis);
        /// inferred from the text if omitted
        #[arg(short, long)]
        task_type: Option<String>,

        /// The task
        text: String,
    },

    /// Fetch and print an agent's card
    Card {
        /// Base URL of the agent
        #[arg(default_value = "http://127.0.0.1:7410")]
        url: String,
    },
}

/// Run an `smesh a2a` subcommand.
pub async fn run(command: A2aCommand) -> Result<()> {
    match command {
        A2aCommand::Serve {
            listen,
            public_url,
            agents,
            model,
            recruit,
            sign_key,
        } => serve(listen, public_url, agents, model, recruit, sign_key).await,
        A2aCommand::Send {
            url,
            task_type,
            text,
        } => send(&url, task_type, text).await,
        A2aCommand::Card { url } => {
            let card = AgentCardResolver::new(None)
                .resolve(&url)
                .await
                .context("fetching agent card")?;
            println!("{}", serde_json::to_string_pretty(&card)?);
            Ok(())
        }
    }
}

fn read_signing_key(path: &std::path::Path) -> Result<SigningKey> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading signing key {}", path.display()))?;
    let hex = text.trim();
    if hex.len() != 64 {
        bail!("signing key must be 64 hex characters (a 32-byte Ed25519 seed)");
    }
    let mut seed = [0u8; 32];
    for (i, byte) in seed.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).context("signing key is not hex")?;
    }
    Ok(SigningKey::from_bytes(&seed))
}

async fn serve(
    listen: String,
    public_url: Option<String>,
    agents: usize,
    model: String,
    recruit: Vec<String>,
    sign_key: Option<std::path::PathBuf>,
) -> Result<()> {
    if agents > 0 && OpenRouterConfig::from_env().is_none() {
        bail!("OpenRouter not configured. Set OPENROUTER_API_KEY (or add it to ~/.creds/openrouter.env)");
    }
    if agents == 0 && recruit.is_empty() {
        bail!("a mesh with no local agents needs at least one --recruit");
    }

    let mut remotes = Vec::new();
    for url in &recruit {
        let remote = RemoteAgent::discover(url, RemoteAgentOptions::default())
            .await
            .with_context(|| format!("recruiting {url}"))?;
        let mut skills: Vec<_> = remote.skills.keys().map(|t| t.as_str()).collect();
        skills.sort();
        println!(
            "✓ recruited {} ({url}) for {}",
            remote.card.name,
            skills.join(", ")
        );
        remotes.push(Arc::new(remote));
    }

    let config = CoordinatorConfig {
        n_agents: agents,
        model: model.clone(),
        roles: vec![
            AgentRole::Coder,
            AgentRole::Reviewer,
            AgentRole::Analyst,
            AgentRole::Writer,
        ],
        ..CoordinatorConfig::default()
    };
    let factory: CoordinatorFactory = Arc::new(move || {
        let mut coordinator = AgentCoordinator::with_openrouter(config.clone(), &model);
        for remote in &remotes {
            coordinator.add_agent(remote.to_llm_agent());
        }
        coordinator
    });

    let listener = tokio::net::TcpListener::bind(&listen)
        .await
        .with_context(|| format!("binding {listen}"))?;
    let base = public_url.unwrap_or_else(|| format!("http://{}", listener.local_addr().unwrap()));
    let mut card = mesh_card(&MeshCardConfig::new(base.clone()));
    if let Some(path) = sign_key {
        let key = read_signing_key(&path)?;
        sign_card(&mut card, &key);
        println!("✓ card signed by {}", key_id(&key.verifying_key()));
    }

    println!("SMESH A2A gateway");
    println!("  card     {base}/.well-known/agent-card.json");
    println!("  JSON-RPC POST {base}/");
    println!("  REST     POST {base}/message:send");
    smesh_a2a::serve(listener, factory, card).await?;
    Ok(())
}

async fn send(url: &str, task_type: Option<String>, text: String) -> Result<()> {
    let card = AgentCardResolver::new(None)
        .resolve(url)
        .await
        .context("fetching agent card")?;
    let client = A2AClientFactory::builder()
        .build()
        .create_from_card(&card)
        .await
        .context("connecting")?;

    let mut message = Message::new(Role::User, vec![Part::text(text)]);
    if let Some(t) = task_type {
        message.metadata = Some([("taskType".to_string(), serde_json::json!(t))].into());
    }
    let req = SendMessageRequest {
        message,
        configuration: None,
        metadata: None,
        tenant: None,
    };

    let mut events = client.send_streaming_message(&req).await?;
    while let Some(event) = events.next().await {
        match event? {
            StreamResponse::StatusUpdate(update) => {
                let text = update
                    .status
                    .message
                    .as_ref()
                    .and_then(|m| m.text())
                    .unwrap_or("");
                println!("[{:?}] {text}", update.status.state);
            }
            StreamResponse::ArtifactUpdate(update) => {
                for part in &update.artifact.parts {
                    if let Some(text) = part.as_text() {
                        println!("\n{text}\n");
                    }
                }
            }
            StreamResponse::Task(task) => println!("[{:?}] task {}", task.status.state, task.id),
            StreamResponse::Message(m) => println!("{}", m.text().unwrap_or("")),
        }
    }
    Ok(())
}
