//! `smesh tools`: give a mesh's agents tools, and try them.
//!
//! Tools come from MCP servers declared in a config file (see
//! `smesh tools example`) and from tools written in Rust (`builtin.*`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::Subcommand;
use smesh_agent::tools::config::{AgentTools, Catalog, ToolsConfig};
use smesh_agent::{
    AffinitySource, AgentConfig, AgentCoordinator, AgentRole, CallOutcome, CoordinatorConfig,
    Effect, JevAffinity, LlmAgent, OpenRouterClient, OpenRouterConfig, TaskDefinition, ToolCall,
    Toolbelt,
};

/// Default config file name.
pub const DEFAULT_CONFIG: &str = "smesh.tools.toml";

/// Model for an agent with tools when its config names none. Chosen from live
/// runs: it lists before it reads and reads before it writes, where lighter
/// models guessed file names or wrote summaries of files they never opened.
pub const DEFAULT_TOOL_MODEL: &str = "openai/gpt-5-mini";

/// A starting config, printed by `smesh tools example`.
pub const EXAMPLE: &str = r#"# SMESH tools: which MCP servers a mesh can reach, and which agent may use what.
#
#   smesh tools list                 # check what each agent gets
#   smesh tools ask "..."            # give the mesh a task
#   smesh a2a serve --jev --tools smesh.tools.toml

# An MCP server run as a local process, speaking MCP over stdio.
[servers.files]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]

# An MCP server over streamable HTTP. A value starting with `$` is read from
# the environment, so secrets stay out of this file.
# [servers.tracker]
# url = "https://mcp.example.com/mcp"
# bearer = "$TRACKER_TOKEN"
#
# Correct what a tool may do when the server's annotations are missing or wrong.
# Unannotated tools count as "write" until you say otherwise.
# [servers.tracker.effects]
# search_issues = "read"

# Each agent says what it does in plain language (phrased to follow "You are
# NAME, ..."), lists tool patterns over <server>.<tool> (`*` matches anything),
# and has a clearance: read (the default), write or execute. Tools above an
# agent's clearance are never offered to it and never run for it. `model` is
# optional (default openai/gpt-5-mini) and must support tool calling.
[[agents]]
name = "Librarian"
capability = "a librarian who answers questions from the files in this directory"
tools = ["files.read_*", "files.list_*", "files.search_*", "files.directory_tree", "builtin.now"]
clearance = "read"

[[agents]]
name = "Scribe"
capability = "a writer who drafts and saves notes and documents as files in this directory"
tools = ["files.*", "builtin.now"]
clearance = "write"
"#;

/// `smesh tools` subcommands.
#[derive(Subcommand)]
pub enum ToolsCommand {
    /// Print a starting config (save it as smesh.tools.toml)
    Example,

    /// Connect to the configured servers and show what each agent may use
    List {
        /// Tools config file
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,
    },

    /// Call one tool directly, as the operator, and print what it returns
    Call {
        /// Tools config file
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,

        /// Tool as <server>.<tool>, e.g. files.read_text_file
        tool: String,

        /// Arguments as a JSON object
        #[arg(default_value = "{}")]
        input: String,

        /// Allow a tool whose effect is write or execute
        #[arg(long)]
        allow_write: bool,
    },

    /// Give the mesh a task: agents judge their fit (with their tools), the
    /// field settles who takes it, and the winner works with its tools
    Ask {
        /// Tools config file
        #[arg(long, default_value = DEFAULT_CONFIG)]
        config: PathBuf,

        /// Send the task to this agent instead of letting the mesh settle it
        #[arg(long)]
        agent: Option<String>,

        /// OpenRouter model for every agent, overriding the config. It must
        /// support tool calling. Default: each agent's `model`, else
        /// openai/gpt-5-mini
        #[arg(short, long)]
        model: Option<String>,

        /// The task
        task: String,
    },
}

/// Run an `smesh tools` subcommand.
pub async fn run(command: ToolsCommand) -> Result<()> {
    match command {
        ToolsCommand::Example => {
            print!("{EXAMPLE}");
            Ok(())
        }
        ToolsCommand::List { config } => list(&config).await,
        ToolsCommand::Call {
            config,
            tool,
            input,
            allow_write,
        } => call(&config, &tool, &input, allow_write).await,
        ToolsCommand::Ask {
            config,
            agent,
            model,
            task,
        } => ask(&config, agent.as_deref(), model.as_deref(), &task).await,
    }
}

/// Load a config, explaining how to start one when it is missing.
pub fn load(path: &Path) -> Result<ToolsConfig> {
    if !path.exists() {
        bail!(
            "no tools config at {}. Start one with: smesh tools example > {DEFAULT_CONFIG}",
            path.display()
        );
    }
    ToolsConfig::load(path).with_context(|| format!("in {}", path.display()))
}

/// A connected config and every agent's toolbelt.
pub struct Equipped {
    /// Every tool, keeping the connections open.
    pub catalog: Catalog,
    /// Each agent with its toolbelt, in config order.
    pub agents: Vec<(AgentTools, Toolbelt)>,
}

/// Load a config, connect to its servers, and build every agent's toolbelt,
/// warning about patterns that match nothing.
pub async fn equip(path: &Path) -> Result<Equipped> {
    let config = load(path)?;
    let catalog = config
        .connect()
        .await
        .with_context(|| format!("connecting the servers in {}", path.display()))?;
    let mut agents = Vec::new();
    for agent in &config.agents {
        let (belt, unmatched) = catalog.toolbelt_for(agent)?;
        for pattern in unmatched {
            eprintln!(
                "⚠ {}: tool pattern {pattern:?} matches no tool (see `smesh tools list`)",
                agent.name
            );
        }
        agents.push((agent.clone(), belt));
    }
    Ok(Equipped { catalog, agents })
}

/// An agent from the config, on OpenRouter, holding its toolbelt. `model`
/// overrides the agent's own; without either it gets [`DEFAULT_TOOL_MODEL`].
pub fn llm_agent(agent: &AgentTools, belt: &Toolbelt, model: Option<&str>) -> LlmAgent {
    let model = model
        .or(agent.model.as_deref())
        .unwrap_or(DEFAULT_TOOL_MODEL);
    let backend = OpenRouterClient::new(
        OpenRouterConfig::from_env()
            .unwrap_or_default()
            .with_model(model.to_string()),
    );
    LlmAgent::new(
        AgentConfig::new(agent.name.clone(), AgentRole::General)
            .with_capability(agent.capability.clone()),
        Arc::new(backend),
    )
    .with_toolbelt(belt.clone())
}

async fn list(path: &Path) -> Result<()> {
    let equipped = equip(path).await?;
    println!("Tools ({}):", equipped.catalog.entries().len());
    for (qualified, tool) in equipped.catalog.entries() {
        println!(
            "  {qualified:<36} {:<8} {}",
            tool.effect(),
            summary(tool.description())
        );
    }
    for (agent, belt) in &equipped.agents {
        println!(
            "\n{} (clearance {}): {}",
            agent.name,
            belt.clearance(),
            agent.capability
        );
        let mut any = false;
        for tool in belt.tools() {
            any = true;
            if tool.effect() <= belt.clearance() {
                println!("  ✓ {}", tool.name());
            } else {
                println!(
                    "  ✗ {}  (needs {} clearance; hidden from this agent)",
                    tool.name(),
                    tool.effect()
                );
            }
        }
        if !any {
            println!("  (no tools)");
        }
    }
    Ok(())
}

async fn call(path: &Path, tool: &str, input: &str, allow_write: bool) -> Result<()> {
    let config = load(path)?;
    let catalog = config.connect().await?;
    let Some((_, implementation)) = catalog.entries().iter().find(|(q, _)| q == tool) else {
        bail!("no tool {tool:?}; `smesh tools list` shows them all");
    };
    if implementation.effect() > Effect::Read && !allow_write {
        bail!(
            "{tool} is a {} tool; pass --allow-write to run it",
            implementation.effect()
        );
    }
    let input: serde_json::Value =
        serde_json::from_str(input).context("arguments must be a JSON object")?;
    let belt = Toolbelt::new(Effect::Execute)
        .with_tool(Arc::clone(implementation))?
        .for_agent("operator");
    let result = belt.call(implementation.name(), input).await;
    println!("{}", result.output);
    if result.outcome != CallOutcome::Ok {
        bail!("{tool}: {:?}", result.outcome);
    }
    Ok(())
}

async fn ask(path: &Path, only: Option<&str>, model: Option<&str>, task: &str) -> Result<()> {
    if OpenRouterConfig::from_env().is_none() {
        bail!("OpenRouter not configured. Set OPENROUTER_API_KEY (or add it to ~/.creds/openrouter.env)");
    }
    let equipped = equip(path).await?;
    let agents: Vec<&(AgentTools, Toolbelt)> = match only {
        Some(name) => {
            let found: Vec<_> = equipped
                .agents
                .iter()
                .filter(|(a, _)| a.name == name)
                .collect();
            if found.is_empty() {
                bail!("no agent named {name:?} in {}", path.display());
            }
            found
        }
        None => equipped.agents.iter().collect(),
    };
    if agents.is_empty() {
        bail!("{} declares no agents", path.display());
    }

    let mut coordinator = AgentCoordinator::with_openrouter(
        CoordinatorConfig {
            n_agents: 0,
            ..CoordinatorConfig::default()
        },
        DEFAULT_TOOL_MODEL,
    );
    // One agent needs no routing; several are settled by assessment.
    let assessed = agents.len() > 1;
    if assessed {
        let source = JevAffinity::from_env().context(
            "routing between agents uses Jev; set TYPESAFE_API_KEY, or pick one with --agent",
        )?;
        coordinator = coordinator.with_affinity_source(Arc::new(source) as Arc<dyn AffinitySource>);
    } else {
        // A lone agent claims anything put to it.
        coordinator = coordinator.with_affinity_source(Arc::new(Always));
    }
    for (agent, belt) in &agents {
        coordinator.add_agent(llm_agent(agent, belt, model));
    }

    let id = coordinator
        .add_task_assessed(TaskDefinition::untyped(task, 0.5))
        .await;
    let fits: Vec<(String, f64, Option<Effect>)> = coordinator
        .agents()
        .filter_map(|a| {
            a.assessed
                .get(&id)
                .map(|l| (a.name().to_string(), *l, a.privilege()))
        })
        .collect();
    if assessed {
        for (name, level, privilege) in &fits {
            let privilege = privilege.map(|p| p.as_str()).unwrap_or("unknown");
            println!(
                "· {name} ({privilege}) judges its fit at {:.0}%",
                level * 100.0
            );
        }
    }
    coordinator.settle_traced();
    let Some(winner) = coordinator.winner_for(&id) else {
        let best = coordinator
            .agents()
            .filter_map(|a| a.assessed.get(&id).map(|l| (a, *l)))
            .max_by(|x, y| x.1.total_cmp(&y.1));
        match best {
            Some((agent, level)) => bail!(
                "no agent suits this task well enough to claim it: the best fit, {}, judged \
                 itself at {:.0}%, under its {:.0}% claim threshold. Rephrase the task, or \
                 send it to one agent with --agent NAME",
                agent.name(),
                level * 100.0,
                agent.config.affinity_threshold * 100.0
            ),
            None => bail!("no agent suits this task well enough to claim it"),
        }
    };
    let winner_name = coordinator
        .agent(&winner)
        .map(|a| a.name().to_string())
        .unwrap_or(winner);
    let best = fits.iter().max_by(|a, b| a.1.total_cmp(&b.1));
    match best {
        Some((top, level, _)) if assessed && *top != winner_name => println!(
            "→ {winner_name} takes it: within {:.0} points of {top}'s {:.0}%, and less privileged\n",
            smesh_agent::agent::CLAIM_MARGIN * 100.0,
            level * 100.0
        ),
        _ => println!("→ {winner_name} takes it\n"),
    }

    let done = coordinator
        .execute_settled(&id)
        .await
        .context("executing the task")?;
    for c in &done.tool_calls {
        print_call(c);
    }
    if !done.tool_calls.is_empty() {
        println!();
    }
    println!("{}", done.result.trim());
    Ok(())
}

fn print_call(c: &ToolCall) {
    let mark = match c.outcome {
        CallOutcome::Ok => "✓",
        CallOutcome::Error => "!",
        CallOutcome::Refused => "✗",
    };
    let input = c.input.to_string();
    let input = if input.chars().count() > 100 {
        format!("{}…", input.chars().take(100).collect::<String>())
    } else {
        input
    };
    println!("  {mark} {} {input}  ({} ms)", c.tool, c.duration_ms);
    if c.outcome != CallOutcome::Ok {
        println!("      {}", first_line(&c.output));
    }
}

/// A description's first sentence, cut to fit one terminal line.
fn summary(text: &str) -> String {
    let line = first_line(text);
    let sentence = line.find(". ").map(|i| &line[..=i]).unwrap_or(line);
    if sentence.chars().count() > 72 {
        format!("{}…", sentence.chars().take(71).collect::<String>())
    } else {
        sentence.to_string()
    }
}

fn first_line(text: &str) -> &str {
    text.lines().find(|l| !l.trim().is_empty()).unwrap_or("")
}

/// Claims every task: a mesh of one needs no judgement.
struct Always;

#[async_trait::async_trait]
impl AffinitySource for Always {
    async fn assess(&self, _: &TaskDefinition, _: &str) -> Option<f64> {
        Some(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_printed_example_is_a_valid_config() {
        let config = ToolsConfig::parse(EXAMPLE).expect("the example must parse");
        assert_eq!(config.agents.len(), 2);
        assert_eq!(config.agents[0].clearance, Effect::Read);
        assert_eq!(config.agents[1].clearance, Effect::Write);
    }

    #[test]
    fn an_agents_model_comes_from_the_flag_then_the_config_then_the_default() {
        let config = ToolsConfig::parse(
            "[[agents]]\nname = \"A\"\ncapability = \"c\"\nmodel = \"x/own\"\n\
             [[agents]]\nname = \"B\"\ncapability = \"c\"\n",
        )
        .unwrap();
        let belt = Toolbelt::new(Effect::Read);
        let model = |a: &AgentTools, m| llm_agent(a, &belt, m).backend().model().to_string();
        assert_eq!(model(&config.agents[0], Some("x/flag")), "x/flag");
        assert_eq!(model(&config.agents[0], None), "x/own");
        assert_eq!(model(&config.agents[1], None), DEFAULT_TOOL_MODEL);
    }
}
