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
use smesh_agent::{
    AffinitySource, AgentConfig, AgentCoordinator, AgentRole, CoordinatorConfig, JevAffinity,
    JevClient, LlmAgent, OpenRouterClient, OpenRouterConfig,
};

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

        /// Number of role agents (cycling Coder, Reviewer, Analyst, Writer).
        /// Default 4, or 0 when --agent is given.
        #[arg(short, long)]
        agents: Option<usize>,

        /// An agent described in plain language, as NAME=CAPABILITY, phrased to
        /// follow "You are NAME, ...". Repeatable. Needs --jev.
        /// Example: --agent "Translator=a translator between English and Portuguese"
        #[arg(long = "agent", value_name = "NAME=CAPABILITY")]
        described: Vec<String>,

        /// Have each agent judge whether it suits each task with Jev
        /// (TypeSafe), instead of the built-in software role tables. Reads
        /// TYPESAFE_API_KEY or ~/.creds/typesafe.env.
        #[arg(long)]
        jev: bool,

        /// Add a research agent ("Quant") that answers from a finished
        /// `smesh quant run` directory. Jev picks the relevant findings.
        /// Needs --jev.
        #[arg(long)]
        quant_run: Option<std::path::PathBuf>,

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

        /// Optional task type label, e.g. code_review. Only agents without an
        /// assessment use it; any label is accepted
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
            described,
            jev,
            quant_run,
            model,
            recruit,
            sign_key,
        } => {
            serve(ServeOptions {
                listen,
                public_url,
                agents,
                described,
                jev,
                quant_run,
                model,
                recruit,
                sign_key,
            })
            .await
        }
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
    parse_signing_key(&text)
}

/// Parse a 32-byte Ed25519 seed written as 64 hex characters.
///
/// Every byte is checked to be an ASCII hex digit before any slicing: the
/// length alone counts bytes, so non-ASCII text of the right length would
/// otherwise be sliced inside a character and panic, and `from_str_radix`
/// would accept a `+` sign.
fn parse_signing_key(text: &str) -> Result<SigningKey> {
    let hex = text.trim().as_bytes();
    if hex.len() != 64 || !hex.iter().all(u8::is_ascii_hexdigit) {
        bail!("signing key must be 64 hex characters (a 32-byte Ed25519 seed)");
    }
    let digit = |b: u8| (b as char).to_digit(16).expect("checked hex digit") as u8;
    let mut seed = [0u8; 32];
    for (byte, pair) in seed.iter_mut().zip(hex.chunks_exact(2)) {
        *byte = digit(pair[0]) << 4 | digit(pair[1]);
    }
    Ok(SigningKey::from_bytes(&seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_described_splits_name_and_capability() {
        assert_eq!(
            parse_described("Translator = a translator, EN to PT").unwrap(),
            ("Translator".into(), "a translator, EN to PT".into())
        );
        // Only the first '=' separates; the capability may contain more.
        assert_eq!(parse_described("A=x=y").unwrap().1, "x=y");
        assert!(parse_described("no-equals").is_err());
        assert!(parse_described("=empty name").is_err());
        assert!(parse_described("Empty=").is_err());
    }

    #[test]
    fn test_parse_signing_key_reads_hex_seed() {
        let key = parse_signing_key(&format!("{}\n", "0a".repeat(32))).unwrap();
        assert_eq!(key.to_bytes(), [0x0a; 32]);
        let upper = parse_signing_key(&"FF".repeat(32)).unwrap();
        assert_eq!(upper.to_bytes(), [0xff; 32]);
    }

    #[test]
    fn test_parse_signing_key_rejects_non_ascii_of_the_right_length() {
        // "a" + 31 two-byte "é" + "b" = 64 bytes. The first byte pair ends
        // inside the first "é", so slicing it by byte pairs used to panic.
        let text = format!("a{}b", "é".repeat(31));
        assert_eq!(text.len(), 64);
        assert!(parse_signing_key(&text).is_err());
    }

    #[test]
    fn test_parse_signing_key_rejects_signs_and_wrong_lengths() {
        assert!(parse_signing_key(&format!("+1{}", "00".repeat(31))).is_err());
        assert!(parse_signing_key(&"00".repeat(31)).is_err());
        assert!(parse_signing_key(&"zz".repeat(32)).is_err());
    }
}

/// Everything `smesh a2a serve` was given.
struct ServeOptions {
    listen: String,
    public_url: Option<String>,
    agents: Option<usize>,
    described: Vec<String>,
    jev: bool,
    quant_run: Option<std::path::PathBuf>,
    model: String,
    recruit: Vec<String>,
    sign_key: Option<std::path::PathBuf>,
}

/// Split `NAME=CAPABILITY`.
fn parse_described(spec: &str) -> Result<(String, String)> {
    let (name, capability) = spec
        .split_once('=')
        .map(|(n, c)| (n.trim(), c.trim()))
        .filter(|(n, c)| !n.is_empty() && !c.is_empty())
        .with_context(|| format!("--agent {spec:?} must be NAME=CAPABILITY"))?;
    Ok((name.to_string(), capability.to_string()))
}

async fn serve(opts: ServeOptions) -> Result<()> {
    let described: Vec<(String, String)> = opts
        .described
        .iter()
        .map(|s| parse_described(s))
        .collect::<Result<_>>()?;
    let role_agents = opts
        .agents
        .unwrap_or(if described.is_empty() { 4 } else { 0 });

    if opts.quant_run.is_some() && !opts.jev {
        bail!("--quant-run needs --jev: the research agent is routed to, and picks its findings, by Jev");
    }
    let quant = match &opts.quant_run {
        Some(dir) => {
            let backend = smesh_quant::agent::FindingsBackend::load(dir, JevClient::from_env())
                .with_context(|| format!("loading research run {}", dir.display()))?;
            println!(
                "✓ research agent Quant answers from {} ({} findings)",
                dir.display(),
                backend.findings().len()
            );
            Some(Arc::new(backend))
        }
        None => None,
    };
    if !described.is_empty() && !opts.jev {
        bail!(
            "--agent needs --jev: an agent described in words can only be matched to tasks \
             by assessment; without it, it would claim like a general-purpose role"
        );
    }
    let affinity = if opts.jev {
        let source = JevAffinity::from_env().context(
            "Jev not configured. Set TYPESAFE_API_KEY (or add it to ~/.creds/typesafe.env)",
        )?;
        println!("✓ agents assess tasks with Jev");
        Some(Arc::new(source) as Arc<dyn AffinitySource>)
    } else {
        None
    };
    let local = role_agents + described.len();
    if local > 0 && OpenRouterConfig::from_env().is_none() {
        bail!("OpenRouter not configured. Set OPENROUTER_API_KEY (or add it to ~/.creds/openrouter.env)");
    }
    if local == 0 && quant.is_none() && opts.recruit.is_empty() {
        bail!("a mesh with no local agents needs at least one --recruit");
    }

    let mut remotes = Vec::new();
    for url in &opts.recruit {
        let remote = RemoteAgent::discover(url, RemoteAgentOptions::default())
            .await
            .with_context(|| format!("recruiting {url}"))?;
        println!("✓ recruited {} ({url})", remote.card.name);
        remotes.push(Arc::new(remote));
    }

    let config = CoordinatorConfig {
        n_agents: role_agents,
        model: opts.model.clone(),
        roles: vec![
            AgentRole::Coder,
            AgentRole::Reviewer,
            AgentRole::Analyst,
            AgentRole::Writer,
        ],
        ..CoordinatorConfig::default()
    };
    let model = opts.model.clone();
    let factory: CoordinatorFactory = Arc::new(move || {
        let mut coordinator = AgentCoordinator::with_openrouter(config.clone(), &model);
        if let Some(source) = &affinity {
            coordinator = coordinator.with_affinity_source(Arc::clone(source));
        }
        for (name, capability) in &described {
            let backend = OpenRouterClient::new(
                OpenRouterConfig::from_env()
                    .unwrap_or_default()
                    .with_model(model.clone()),
            );
            coordinator.add_agent(LlmAgent::new(
                AgentConfig::new(name.clone(), AgentRole::General)
                    .with_capability(capability.clone()),
                Arc::new(backend),
            ));
        }
        if let Some(backend) = &quant {
            coordinator.add_agent(LlmAgent::new(
                AgentConfig::new("Quant", AgentRole::Analyst)
                    .with_capability(smesh_quant::agent::CAPABILITY),
                Arc::clone(backend) as Arc<dyn smesh_agent::LlmBackend>,
            ));
        }
        for remote in &remotes {
            coordinator.add_agent(remote.to_llm_agent());
        }
        coordinator
    });

    let listener = tokio::net::TcpListener::bind(&opts.listen)
        .await
        .with_context(|| format!("binding {}", opts.listen))?;
    let base = opts
        .public_url
        .unwrap_or_else(|| format!("http://{}", listener.local_addr().unwrap()));
    let roster = factory().agent_roster();
    let mut card = mesh_card(&MeshCardConfig::new(base.clone()), &roster);
    if let Some(path) = opts.sign_key {
        let key = read_signing_key(&path)?;
        sign_card(&mut card, &key);
        println!("✓ card signed by {}", key_id(&key.verifying_key()));
    }

    println!("SMESH A2A gateway");
    for agent in &roster {
        println!(
            "  agent    {}: {}",
            agent.name,
            agent.capability.lines().next().unwrap_or("")
        );
    }
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
