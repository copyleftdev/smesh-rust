//! SMESH CLI - Command line tools for testing and running SMESH

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use smesh_agent::{
    benchmark_backend, print_comparison, AgentCoordinator, AgentRole, ClaudeClient, ClaudeConfig,
    CoordinatorConfig, OllamaClient, OllamaConfig, TaskDefinition,
};
use smesh_core::{Network, NetworkTopology, Signal, SignalType};
use smesh_runtime::{RuntimeConfig, SmeshRuntime};

mod review;
mod threat;

#[derive(Parser)]
#[command(name = "smesh")]
#[command(about = "SMESH - Plant-inspired signal diffusion protocol", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, default_value = "false")]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Check system status and dependencies
    Status,

    /// Run a simulation
    Sim {
        /// Number of nodes
        #[arg(short, long, default_value = "10")]
        nodes: usize,

        /// Network topology (ring, mesh, small_world, scale_free)
        #[arg(short, long, default_value = "small_world")]
        topology: String,

        /// Number of ticks to run
        #[arg(short = 'k', long, default_value = "100")]
        ticks: u64,
    },

    /// Run LLM agent coordination demo
    Agents {
        /// Number of agents
        #[arg(short, long, default_value = "3")]
        agents: usize,

        /// Ollama model to use
        #[arg(short, long, default_value = "deepseek-coder-v2:16b")]
        model: String,

        /// Run demo tasks
        #[arg(long, default_value = "true")]
        demo: bool,
    },

    /// Test Ollama connection
    Ollama {
        /// Model to test
        #[arg(short, long, default_value = "deepseek-coder-v2:16b")]
        model: String,
    },

    /// Benchmark signal processing
    Bench {
        /// Number of signals
        #[arg(short, long, default_value = "10000")]
        signals: usize,

        /// Number of nodes
        #[arg(short, long, default_value = "100")]
        nodes: usize,
    },

    /// Compare LLM backends (Ollama vs Claude)
    Compare {
        /// Ollama model to use
        #[arg(long, default_value = "qwen2.5-coder:7b")]
        ollama_model: String,

        /// Claude model to use
        #[arg(long, default_value = "claude-sonnet-4-20250514")]
        claude_model: String,

        /// Custom prompt to test
        #[arg(short, long)]
        prompt: Option<String>,
    },

    /// Run SMESH-coordinated code review on a repository
    Review {
        /// Path to the repository to review
        #[arg(short, long)]
        path: PathBuf,

        /// Ollama model to use for review
        #[arg(short, long, default_value = "qwen2.5-coder:7b")]
        model: String,
    },

    /// Analyze threat patterns from security payload repositories
    Threat {
        /// Path to the payload repository (e.g., PayloadsAllTheThings)
        #[arg(short, long)]
        path: PathBuf,

        /// Ollama model to use
        #[arg(short, long, default_value = "qwen2.5-coder:7b")]
        model: String,

        /// Maximum files to analyze (to limit token usage)
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
}

fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::INFO };

    FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .compact()
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    match cli.command {
        Commands::Status => cmd_status().await,
        Commands::Sim {
            nodes,
            topology,
            ticks,
        } => cmd_sim(nodes, &topology, ticks).await,
        Commands::Agents {
            agents,
            model,
            demo,
        } => cmd_agents(agents, &model, demo).await,
        Commands::Ollama { model } => cmd_ollama(&model).await,
        Commands::Bench { signals, nodes } => cmd_bench(signals, nodes).await,
        Commands::Compare {
            ollama_model,
            claude_model,
            prompt,
        } => cmd_compare(&ollama_model, &claude_model, prompt.as_deref()).await,
        Commands::Review { path, model } => review::run_review(&path, &model).await.map(|_| ()),
        Commands::Threat { path, model, limit } => {
            threat::run_threat_analysis(&path, &model, limit)
                .await
                .map(|_| ())
        }
    }
}

async fn cmd_status() -> Result<()> {
    println!("╔═══════════════════════════════════════╗");
    println!("║         SMESH Status Check            ║");
    println!("╚═══════════════════════════════════════╝");
    println!();

    // Check core
    println!("✓ smesh-core: OK");
    println!("  Version: {}", smesh_core::VERSION);

    // Check runtime
    println!("✓ smesh-runtime: OK");

    // Check Ollama
    print!("  Ollama: ");
    let client = OllamaClient::default_client();
    if client.is_available().await {
        println!("✓ Connected");

        match client.list_models().await {
            Ok(models) => {
                println!("  Models available:");
                for model in models.iter().take(5) {
                    println!("    - {}", model);
                }
                if models.len() > 5 {
                    println!("    ... and {} more", models.len() - 5);
                }
            }
            Err(e) => println!("  Could not list models: {}", e),
        }
    } else {
        println!("✗ Not available (start with: ollama serve)");
    }

    println!();
    println!("Ready to run SMESH!");

    Ok(())
}

async fn cmd_sim(n_nodes: usize, topology_str: &str, ticks: u64) -> Result<()> {
    let topology = match topology_str {
        "ring" => NetworkTopology::Ring,
        "mesh" | "full" => NetworkTopology::FullMesh,
        "small_world" | "sw" => NetworkTopology::SmallWorld,
        "scale_free" | "sf" => NetworkTopology::ScaleFree,
        "random" => NetworkTopology::Random,
        "grid" => NetworkTopology::Grid,
        _ => {
            println!("Unknown topology '{}', using small_world", topology_str);
            NetworkTopology::SmallWorld
        }
    };

    println!("╔═══════════════════════════════════════╗");
    println!("║         SMESH Simulation              ║");
    println!("╚═══════════════════════════════════════╝");
    println!();
    println!("Nodes: {}", n_nodes);
    println!("Topology: {:?}", topology);
    println!("Ticks: {}", ticks);
    println!();

    // Create network
    let network = Network::with_topology(n_nodes, topology);
    let stats = network.stats();

    println!("Network created:");
    println!("  Nodes: {}", stats.node_count);
    println!("  Connections: {}", stats.connection_count);
    println!("  Avg degree: {:.1}", stats.avg_degree);
    println!();

    // Create runtime
    let runtime = SmeshRuntime::with_network(
        network,
        RuntimeConfig {
            tick_interval_ms: 10,
            ..Default::default()
        },
    );

    // Emit some test signals
    let node_ids: Vec<String> = {
        let net = runtime.network();
        let net_guard = net.read().await;
        net_guard.nodes.keys().take(5).cloned().collect()
    };

    println!("Emitting test signals...");
    for (i, node_id) in node_ids.iter().enumerate() {
        let signal = Signal::builder(SignalType::Data)
            .payload(format!("Test signal {}", i).into_bytes())
            .intensity(0.8)
            .ttl(50.0)
            .build();

        runtime.emit(signal, node_id).await;
    }

    // Run simulation
    println!("Running {} ticks...", ticks);
    let start = std::time::Instant::now();

    let _events = runtime.run_ticks(ticks).await;

    let elapsed = start.elapsed();
    let final_stats = runtime.stats().await;

    println!();
    println!("Simulation complete:");
    println!("  Duration: {:.2}ms", elapsed.as_millis());
    println!("  Ticks/sec: {:.0}", ticks as f64 / elapsed.as_secs_f64());
    println!("  Final active signals: {}", final_stats.active_signals);
    println!(
        "  Total reinforcements: {}",
        final_stats.total_reinforcements
    );

    Ok(())
}

async fn cmd_agents(n_agents: usize, model: &str, demo: bool) -> Result<()> {
    println!("╔═══════════════════════════════════════╗");
    println!("║      SMESH Agent Coordination         ║");
    println!("╚═══════════════════════════════════════╝");
    println!();

    // Check Ollama first
    let client = OllamaClient::new(OllamaConfig {
        model: model.to_string(),
        ..Default::default()
    });

    if !client.is_available().await {
        println!("✗ Ollama not available. Start with: ollama serve");
        return Ok(());
    }

    println!("✓ Ollama connected");
    println!("  Model: {}", model);
    println!("  Agents: {}", n_agents);
    println!();

    // Create coordinator
    let config = CoordinatorConfig {
        n_agents,
        model: model.to_string(),
        roles: vec![AgentRole::Coder, AgentRole::Reviewer, AgentRole::Analyst],
        max_ticks: 50,
        tick_interval_ms: 100,
    };

    let mut coordinator = AgentCoordinator::with_ollama(config, model);

    // Define demo tasks
    let tasks = if demo {
        vec![
            TaskDefinition {
                task_type: "code_review".to_string(),
                description: "Review this function: def add(a, b): return a + b".to_string(),
                priority: 0.8,
            },
            TaskDefinition {
                task_type: "analysis".to_string(),
                description: "Analyze the complexity of binary search".to_string(),
                priority: 0.7,
            },
            TaskDefinition {
                task_type: "documentation".to_string(),
                description: "Write a docstring for a merge sort function".to_string(),
                priority: 0.6,
            },
        ]
    } else {
        println!("No tasks specified. Use --demo for demo tasks.");
        return Ok(());
    };

    println!("Tasks: {}", tasks.len());
    for task in &tasks {
        println!("  - {} (priority: {:.1})", task.task_type, task.priority);
    }
    println!();

    // Run coordination
    println!("Running coordination...");
    let result = coordinator.run(tasks).await;

    println!();
    println!("╔═══════════════════════════════════════╗");
    println!("║             Results                   ║");
    println!("╚═══════════════════════════════════════╝");
    println!();
    println!(
        "Tasks completed: {}/{}",
        result.tasks_completed, result.tasks_total
    );
    println!("Total LLM calls: {}", result.total_llm_calls);
    println!("Elapsed: {:.1}s", result.elapsed_secs);
    println!();

    // Show agent outputs
    println!("Agent Outputs:");
    for (agent_name, tasks) in &result.agent_results {
        if !tasks.is_empty() {
            println!();
            println!("  {}:", agent_name);
            for task in tasks {
                if let Some(ref result) = task.result {
                    let preview: String = result.chars().take(100).collect();
                    println!("    [{}] {}...", task.task_type.as_str(), preview);
                }
            }
        }
    }

    Ok(())
}

async fn cmd_ollama(model: &str) -> Result<()> {
    println!("Testing Ollama connection...");
    println!("Model: {}", model);
    println!();

    let client = OllamaClient::new(OllamaConfig {
        model: model.to_string(),
        ..Default::default()
    });

    if !client.is_available().await {
        println!("✗ Ollama not available");
        println!("  Start with: ollama serve");
        return Ok(());
    }

    println!("✓ Ollama connected");

    // List models
    match client.list_models().await {
        Ok(models) => {
            println!("✓ Models available: {}", models.len());

            let has_model = models
                .iter()
                .any(|m| m.contains(model.split(':').next().unwrap_or(model)));
            if has_model {
                println!("✓ Model '{}' found", model);
            } else {
                println!("✗ Model '{}' not found", model);
                println!("  Available models:");
                for m in models.iter().take(5) {
                    println!("    - {}", m);
                }
                return Ok(());
            }
        }
        Err(e) => {
            println!("✗ Could not list models: {}", e);
            return Ok(());
        }
    }

    // Test generation
    println!();
    println!("Testing generation...");

    match client
        .generate(
            "Say 'SMESH ready' if you can read this.",
            Some("Be very brief."),
        )
        .await
    {
        Ok(response) => {
            println!("✓ Generation successful");
            println!("  Response: {}", response.trim());
        }
        Err(e) => {
            println!("✗ Generation failed: {}", e);
        }
    }

    Ok(())
}

async fn cmd_bench(n_signals: usize, n_nodes: usize) -> Result<()> {
    println!("╔═══════════════════════════════════════╗");
    println!("║         SMESH Benchmark               ║");
    println!("╚═══════════════════════════════════════╝");
    println!();
    println!("Signals: {}", n_signals);
    println!("Nodes: {}", n_nodes);
    println!();

    // Create network
    let mut network = Network::with_topology(n_nodes, NetworkTopology::SmallWorld);
    let node_ids: Vec<String> = network.nodes.keys().cloned().collect();

    // Benchmark signal emission
    println!("Benchmarking signal emission...");
    let start = std::time::Instant::now();

    for i in 0..n_signals {
        let node_id = &node_ids[i % node_ids.len()];
        let signal = Signal::builder(SignalType::Data)
            .payload(format!("Benchmark signal {}", i).into_bytes())
            .intensity(0.8)
            .ttl(100.0)
            .origin(node_id)
            .build();

        network.field.emit_anonymous(signal);
    }

    let emit_elapsed = start.elapsed();
    println!(
        "  Emission: {:.2}ms ({:.0} signals/sec)",
        emit_elapsed.as_millis(),
        n_signals as f64 / emit_elapsed.as_secs_f64()
    );

    // Benchmark tick processing
    println!("Benchmarking tick processing...");
    let start = std::time::Instant::now();
    let ticks = 100;

    for _ in 0..ticks {
        network.tick(0.1);
    }

    let tick_elapsed = start.elapsed();
    println!(
        "  {} ticks: {:.2}ms ({:.0} ticks/sec)",
        ticks,
        tick_elapsed.as_millis(),
        ticks as f64 / tick_elapsed.as_secs_f64()
    );

    // Summary
    let final_stats = network.stats();
    println!();
    println!("Final state:");
    println!(
        "  Active signals: {}",
        final_stats.field_stats.active_signals
    );
    println!("  History size: {}", final_stats.field_stats.history_size);
    println!(
        "  Total reinforcements: {}",
        final_stats.field_stats.total_reinforcements
    );

    Ok(())
}

async fn cmd_compare(
    ollama_model: &str,
    claude_model: &str,
    custom_prompt: Option<&str>,
) -> Result<()> {
    println!("╔═══════════════════════════════════════╗");
    println!("║      LLM Backend Comparison           ║");
    println!("╚═══════════════════════════════════════╝");
    println!();

    // Test prompts
    let prompts = if let Some(p) = custom_prompt {
        vec![(p, None)]
    } else {
        vec![
            (
                "Write a short Rust function that calculates fibonacci numbers.",
                Some("You are a helpful coding assistant. Be concise."),
            ),
            (
                "Explain in 2-3 sentences how signal propagation works in a mesh network.",
                None,
            ),
            ("What is 2 + 2? Answer with just the number.", None),
        ]
    };

    let mut results = Vec::new();

    // Check Ollama
    print!("Checking Ollama... ");
    let ollama_config = OllamaConfig {
        model: ollama_model.to_string(),
        ..OllamaConfig::default()
    };
    let ollama = OllamaClient::new(ollama_config);

    let ollama_available = ollama.is_available().await;
    if ollama_available {
        println!("✓ Available (model: {})", ollama_model);
    } else {
        println!("✗ Not available (start with: ollama serve)");
    }

    // Check Claude
    print!("Checking Claude... ");
    let claude_available = if let Some(config) = ClaudeConfig::from_env() {
        let claude = ClaudeClient::new(config.with_model(claude_model));
        println!("✓ API key found (model: {})", claude_model);
        Some(claude)
    } else {
        println!("✗ No API key (set ANTHROPIC_API_KEY)");
        None
    };

    println!();

    if !ollama_available && claude_available.is_none() {
        println!("No backends available. Please:");
        println!("  - Start Ollama: ollama serve");
        println!("  - Or set ANTHROPIC_API_KEY environment variable");
        return Ok(());
    }

    // Run benchmarks
    for (i, (prompt, system)) in prompts.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Prompt {}: {}", i + 1, truncate_str(prompt, 50));
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        // Test Ollama
        if ollama_available {
            print!("  Ollama: ");
            let result = benchmark_backend(&ollama, prompt, *system).await;
            if result.success {
                println!("✓ {:.2}s", result.total_latency.as_secs_f64());
                println!("    Response: {}", truncate_str(&result.response, 80));
            } else {
                println!("✗ {}", result.error.as_deref().unwrap_or("unknown error"));
            }
            results.push(result);
        }

        // Test Claude
        if let Some(ref claude) = claude_available {
            print!("  Claude: ");
            let result = benchmark_backend(claude, prompt, *system).await;
            if result.success {
                let tps = result
                    .tokens_per_second
                    .map(|t| format!(" ({:.1} tok/s)", t))
                    .unwrap_or_default();
                println!("✓ {:.2}s{}", result.total_latency.as_secs_f64(), tps);
                println!("    Response: {}", truncate_str(&result.response, 80));
            } else {
                println!("✗ {}", result.error.as_deref().unwrap_or("unknown error"));
            }
            results.push(result);
        }

        println!();
    }

    // Summary table
    if !results.is_empty() {
        print_comparison(&results);
    }

    Ok(())
}

fn truncate_str(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
