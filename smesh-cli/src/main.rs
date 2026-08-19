//! SMESH CLI - Command line tools for testing and running SMESH

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use smesh_agent::{
    benchmark_backend, print_comparison, AgentCoordinator, AgentRole, ClaudeClient, ClaudeConfig,
    CoordinatorConfig, OpenRouterClient, TaskDefinition,
};
use smesh_core::{Network, NetworkTopology, Node, Signal, SignalType};
use smesh_runtime::{MeshConfig, RuntimeConfig, RuntimeEvent, SmeshRuntime};

mod adjudicate;
mod analysis;
mod owasp;
mod resilience;
mod review;
mod showcase;
mod swarm;
mod viz;

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

        /// OpenRouter model to use
        #[arg(short, long, default_value = "google/gemini-2.5-flash-lite")]
        model: String,

        /// Run demo tasks
        #[arg(long, default_value = "true")]
        demo: bool,
    },

    /// Test OpenRouter connection
    Openrouter {
        /// Model to test
        #[arg(short, long, default_value = "google/gemini-2.5-flash-lite")]
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

    /// Compare LLM backends (OpenRouter vs Claude)
    Compare {
        /// OpenRouter model to use
        #[arg(long, default_value = "google/gemini-2.5-flash-lite")]
        openrouter_model: String,

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

        /// OpenRouter model to use for review
        #[arg(short, long, default_value = "google/gemini-2.5-flash-lite")]
        model: String,
    },

    /// Run multi-agent coding swarm (Claude-powered) - demonstrates SMESH coordination
    Code {
        /// Number of coder agents
        #[arg(long, default_value = "2")]
        coders: usize,

        /// Consensus threshold (agents that must agree)
        #[arg(long, default_value = "2")]
        consensus: u32,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Serve the SMESH signal field visualization (single binary, zero deps)
    Viz {
        /// Port to serve on
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },

    /// Serve the two-tab kinetic showcase (OWASP benchmark + live mesh)
    Showcase {
        /// Port to serve on
        #[arg(short, long, default_value = "8090")]
        port: u16,
    },

    /// Adjudicate pharmaceutical claims against signed AION formulary policies
    Adjudicate {
        /// Write a self-contained HTML decision report to this path
        #[arg(long, default_value = "adjudication-report.html")]
        html: PathBuf,

        /// Also write results as JSON to this path
        #[arg(long)]
        json: Option<PathBuf>,
    },

    /// Benchmark mesh resilience under failure, eclipse, and Byzantine attacks
    Resilience {
        /// Number of nodes in the mesh
        #[arg(long, default_value = "60")]
        nodes: usize,

        /// Topology (small_world, scale_free, ring, mesh, grid, random)
        #[arg(short, long, default_value = "small_world")]
        topology: String,

        /// Trials averaged per sweep point
        #[arg(long, default_value = "8")]
        trials: usize,

        /// Write a self-contained HTML scorecard to this path
        #[arg(long, default_value = "resilience-scorecard.html")]
        html: PathBuf,

        /// Also write raw results as JSON to this path
        #[arg(long)]
        json: Option<PathBuf>,
    },

    /// Run web red team mission against a live target (authorized testing only)
    Redteam {
        /// Target domain (e.g., example.com)
        #[arg(short, long)]
        target: String,
    },

    /// Full spectrum SMESH-coordinated red team (all tiers + Claude analysis)
    Fullscan {
        /// Target domain (e.g., example.com)
        #[arg(short, long)]
        target: String,

        /// Skip JavaScript static analysis (faster)
        #[arg(long, default_value = "false")]
        no_js: bool,

        /// Run Claude exploitability analysis on findings
        #[arg(long, default_value = "false")]
        analyze: bool,

        /// Generate full report via Claude
        #[arg(long, default_value = "false")]
        report: bool,
    },

    /// Run security bounty hunting swarm (SMESH + Claude + Tools)
    Bounty {
        /// Path to scan for vulnerabilities
        #[arg(short, long, default_value = ".")]
        path: PathBuf,

        /// Maximum files to analyze
        #[arg(long, default_value = "50")]
        max_files: usize,

        /// Output format (text, json, markdown)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Consensus threshold (agents that must agree)
        #[arg(long, default_value = "2")]
        consensus: u32,

        /// Quick scan mode (fewer agents, faster)
        #[arg(long, default_value = "false")]
        quick: bool,

        /// Claude model to use
        #[arg(long, default_value = "claude-sonnet-4-20250514")]
        model: String,
    },

    /// Run multi-agent vulnerability swarm scan (Claude-powered)
    Swarm {
        /// Path to scan for vulnerabilities
        #[arg(short, long, default_value = ".")]
        path: PathBuf,

        /// Maximum files to analyze
        #[arg(long, default_value = "50")]
        max_files: usize,

        /// Output format (text, json, sarif)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Consensus threshold (agents that must agree)
        #[arg(long, default_value = "3")]
        consensus: u32,
    },

    /// Orchestrate the multi-concern analysis mesh as real processes
    Orchestrate {
        /// Directory for journals and the merged timeline
        #[arg(long, default_value = "runs/latest")]
        out: PathBuf,

        /// First port; concerns take consecutive ports from here
        #[arg(long, default_value = "9301")]
        base_port: u16,

        /// Corpus seed, shared by every analyst
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Wall-clock milliseconds per corpus minute
        #[arg(long, default_value = "700")]
        bucket_ms: u64,

        /// Distinct concerns that must concur for consensus
        #[arg(long, default_value = "4")]
        consensus_threshold: usize,

        /// Extra gossip time after the corpus ends, in milliseconds
        #[arg(long, default_value = "6000")]
        settle_ms: u64,
    },

    /// Run one analyst node of the analysis mesh (usually spawned by orchestrate)
    Analyze {
        /// Which concern this analyst owns
        #[arg(long)]
        concern: String,

        /// Address to listen on
        #[arg(long, default_value = "127.0.0.1:0")]
        bind: String,

        /// Peer to dial; repeat for several
        #[arg(long = "peer")]
        peers: Vec<String>,

        /// Where to write this node's journal
        #[arg(long)]
        journal: Option<PathBuf>,

        /// Shared run epoch in unix milliseconds
        #[arg(long)]
        run_epoch: Option<i64>,

        /// Corpus seed
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Wall-clock milliseconds per corpus minute
        #[arg(long, default_value = "700")]
        bucket_ms: u64,

        /// Distinct concerns that must concur for consensus
        #[arg(long, default_value = "4")]
        consensus_threshold: usize,

        /// Peers to wait for before starting the timeline
        #[arg(long, default_value = "0")]
        expect_peers: usize,

        /// Extra gossip time after the corpus ends, in milliseconds
        #[arg(long, default_value = "6000")]
        settle_ms: u64,
    },

    /// Run one node on a live P2P mesh over QUIC
    Mesh {
        /// Address to listen on (port 0 lets the OS pick)
        #[arg(long, default_value = "127.0.0.1:0")]
        bind: String,

        /// Peer to dial on startup; repeat for several
        #[arg(long = "peer")]
        peers: Vec<String>,

        /// Readable node id (defaults to a random one)
        #[arg(long)]
        name: Option<String>,

        /// File holding this node's signing key, created if absent.
        ///
        /// Without one the key is new on every start, and peers that pinned the
        /// old key will refuse the restarted node as an impostor.
        #[arg(long)]
        identity: Option<PathBuf>,

        /// Payload to emit onto the mesh once connected
        #[arg(long)]
        emit: Option<String>,

        /// Seconds to wait before emitting
        #[arg(long, default_value = "2")]
        emit_after: u64,

        /// Seconds to stay on the mesh (0 = until Ctrl-C)
        #[arg(long, default_value = "20")]
        duration: u64,
    },

    /// Benchmark the SMESH mesh against the OWASP Benchmark corpus
    Owasp {
        /// Path to a BenchmarkJava checkout
        #[arg(long, default_value = ".owasp-benchmark/BenchmarkJava")]
        path: PathBuf,

        /// Test cases to sample per category (0 = full corpus)
        #[arg(long, default_value = "25")]
        per_category: usize,

        /// OpenRouter model to use
        #[arg(short, long, default_value = "google/gemini-2.5-flash-lite")]
        model: String,

        /// Detector agents per test case (1-4)
        #[arg(long, default_value = "3")]
        agents: usize,

        /// Agents that must agree to flag a category
        #[arg(long, default_value = "2")]
        consensus: u32,

        /// Files analyzed concurrently
        #[arg(long, default_value = "6")]
        concurrency: usize,

        /// Write a self-contained HTML scorecard to this path
        #[arg(long, default_value = "owasp-scorecard.html")]
        html: PathBuf,

        /// Also write raw results as JSON to this path
        #[arg(long)]
        json: Option<PathBuf>,

        /// Re-render the scorecard from a previous results JSON (skips the mesh run)
        #[arg(long)]
        from_json: Option<PathBuf>,
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
        Commands::Openrouter { model } => cmd_openrouter(&model).await,
        Commands::Bench { signals, nodes } => cmd_bench(signals, nodes).await,
        Commands::Compare {
            openrouter_model,
            claude_model,
            prompt,
        } => cmd_compare(&openrouter_model, &claude_model, prompt.as_deref()).await,
        Commands::Review { path, model } => review::run_review(&path, &model).await.map(|_| ()),
        Commands::Code {
            coders,
            consensus,
            format,
        } => cmd_code(coders, consensus, &format).await,
        Commands::Viz { port } => {
            viz::serve(port)?;
            Ok(())
        }
        Commands::Showcase { port } => {
            showcase::serve(port)?;
            Ok(())
        }
        Commands::Adjudicate { html, json } => cmd_adjudicate(html, json).await,
        Commands::Resilience {
            nodes,
            topology,
            trials,
            html,
            json,
        } => cmd_resilience(nodes, &topology, trials, html, json).await,
        Commands::Redteam { target } => cmd_redteam(&target).await,
        Commands::Mesh {
            bind,
            peers,
            name,
            identity,
            emit,
            emit_after,
            duration,
        } => cmd_mesh(&bind, &peers, name, identity, emit, emit_after, duration).await,
        Commands::Orchestrate {
            out,
            base_port,
            seed,
            bucket_ms,
            consensus_threshold,
            settle_ms,
        } => analysis::orchestrate::run(analysis::orchestrate::RunConfig {
            out_dir: out,
            base_port,
            seed,
            bucket_ms,
            consensus_threshold,
            settle_ms,
        })
        .await
        .map(|_| ()),
        Commands::Analyze {
            concern,
            bind,
            peers,
            journal,
            run_epoch,
            seed,
            bucket_ms,
            consensus_threshold,
            expect_peers,
            settle_ms,
        } => {
            let concern = analysis::concern::Concern::parse(&concern).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown concern '{concern}'; expected one of: {}",
                    analysis::concern::Concern::all()
                        .iter()
                        .map(|c| c.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;

            analysis::node::run(analysis::node::AnalystConfig {
                concern,
                bind: bind.parse()?,
                peers: peers
                    .iter()
                    .map(|p| p.parse::<std::net::SocketAddr>())
                    .collect::<Result<Vec<_>, _>>()?,
                journal,
                run_epoch_ms: run_epoch.unwrap_or_else(|| chrono::Utc::now().timestamp_millis()),
                seed,
                bucket_ms,
                consensus_threshold,
                expect_peers,
                settle_ms,
                verbose: true,
            })
            .await
        }
        Commands::Fullscan {
            target,
            no_js,
            analyze,
            report,
        } => cmd_fullscan(&target, no_js, analyze, report).await,
        Commands::Bounty {
            path,
            max_files,
            format,
            consensus,
            quick,
            model,
        } => cmd_bounty(&path, max_files, &format, consensus, quick, &model).await,
        Commands::Swarm {
            path,
            max_files,
            format,
            consensus,
        } => cmd_swarm(&path, max_files, &format, consensus).await,
        Commands::Owasp {
            path,
            per_category,
            model,
            agents,
            consensus,
            concurrency,
            html,
            json,
            from_json,
        } => {
            cmd_owasp(
                path,
                per_category,
                model,
                agents,
                consensus,
                concurrency,
                html,
                json,
                from_json,
            )
            .await
        }
    }
}

async fn cmd_adjudicate(html: PathBuf, json: Option<PathBuf>) -> Result<()> {
    use adjudicate::report;

    println!("╔═══════════════════════════════════════╗");
    println!("║   SMESH × AION Claim Adjudication      ║");
    println!("╚═══════════════════════════════════════╝");

    let results = adjudicate::adjudicate_samples();
    report::print_report(&results);

    std::fs::write(&html, report::render_html(&results))
        .map_err(|e| anyhow::anyhow!("failed to write HTML: {e}"))?;
    println!("\n📊 Decision report written to: {}", html.display());

    if let Some(json_path) = json {
        std::fs::write(&json_path, report::to_json(&results))
            .map_err(|e| anyhow::anyhow!("failed to write JSON: {e}"))?;
        println!("📄 JSON written to: {}", json_path.display());
    }
    Ok(())
}

async fn cmd_resilience(
    nodes: usize,
    topology: &str,
    trials: usize,
    html: PathBuf,
    json: Option<PathBuf>,
) -> Result<()> {
    use resilience::{report, ResilienceConfig};

    println!("╔═══════════════════════════════════════╗");
    println!("║     SMESH Resilience Benchmark        ║");
    println!("╚═══════════════════════════════════════╝\n");

    let topo = match topology {
        "ring" => NetworkTopology::Ring,
        "mesh" | "full" => NetworkTopology::FullMesh,
        "scale_free" | "sf" => NetworkTopology::ScaleFree,
        "random" => NetworkTopology::Random,
        "grid" => NetworkTopology::Grid,
        _ => NetworkTopology::SmallWorld,
    };

    let cfg = ResilienceConfig {
        nodes,
        topology: topo,
        trials,
        ..Default::default()
    };

    println!(
        "Sweeping attacks over the real mesh ({nodes} nodes, {topology}, {trials} trials/point)…"
    );
    let report_data = resilience::run_benchmark(cfg);

    report::print_report(&report_data);

    std::fs::write(&html, report::render_html(&report_data))
        .map_err(|e| anyhow::anyhow!("failed to write HTML: {e}"))?;
    println!("📊 Scorecard written to: {}", html.display());

    if let Some(json_path) = json {
        std::fs::write(&json_path, report::to_json(&report_data))
            .map_err(|e| anyhow::anyhow!("failed to write JSON: {e}"))?;
        println!("📄 JSON results written to: {}", json_path.display());
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn cmd_owasp(
    path: PathBuf,
    per_category: usize,
    model: String,
    agents: usize,
    consensus: u32,
    concurrency: usize,
    html: PathBuf,
    json: Option<PathBuf>,
    from_json: Option<PathBuf>,
) -> Result<()> {
    use owasp::{report, score, OwaspConfig};

    println!("╔═══════════════════════════════════════╗");
    println!("║     SMESH × OWASP Benchmark           ║");
    println!("╚═══════════════════════════════════════╝");
    println!();

    // Re-render path: load a previous results JSON and skip the mesh entirely.
    let scorecard = if let Some(src) = from_json {
        let raw = std::fs::read_to_string(&src)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", src.display()))?;
        score::from_json(&raw)
            .ok_or_else(|| anyhow::anyhow!("could not parse results JSON at {}", src.display()))?
    } else {
        let config = OwaspConfig {
            benchmark_root: path,
            per_category,
            model,
            agents,
            consensus_threshold: consensus,
            high_conf_threshold: 0.8,
            concurrency,
        };
        owasp::run_benchmark(config)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?
    };

    report::print_scorecard(&scorecard);

    std::fs::write(&html, report::render_html(&scorecard))
        .map_err(|e| anyhow::anyhow!("failed to write HTML: {e}"))?;
    println!("\n📊 Scorecard written to: {}", html.display());

    if let Some(json_path) = json {
        std::fs::write(&json_path, report::to_json(&scorecard))
            .map_err(|e| anyhow::anyhow!("failed to write JSON: {e}"))?;
        println!("📄 JSON results written to: {}", json_path.display());
    }

    Ok(())
}

async fn cmd_fullscan(target: &str, no_js: bool, analyze: bool, report: bool) -> Result<()> {
    use smesh_bounty::{
        analyze_exploitability, generate_report, run_full_spectrum, FullSpectrumConfig,
    };

    let mut config = FullSpectrumConfig::full(target);
    if no_js {
        config.js_analysis = false;
    }

    let result = run_full_spectrum(config)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // Claude analysis if requested
    if analyze || report {
        let analyzed = analyze_exploitability(&result.correlated_findings, target)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        if report {
            let report_text = generate_report(
                &result.correlated_findings,
                &analyzed,
                target,
                &result.subdomains,
                result.endpoints.len(),
                result.duration.as_secs_f64(),
            )
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;

            // Save report
            let report_path = result.work_dir.join("report.md");
            std::fs::write(&report_path, &report_text)
                .map_err(|e| anyhow::anyhow!("Failed to write report: {}", e))?;
            println!("\n  Report saved to: {}\n", report_path.display());
        }
    }

    Ok(())
}

async fn cmd_redteam(target: &str) -> Result<()> {
    smesh_bounty::run_web_redteam(target)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(())
}

async fn cmd_bounty(
    path: &Path,
    max_files: usize,
    format: &str,
    consensus: u32,
    quick: bool,
    model: &str,
) -> Result<()> {
    use smesh_bounty::{BountyConfig, BountyCoordinator, OutputFormat as BountyOutputFormat};

    let config = if quick {
        BountyConfig::quick(path)
            .with_model(model)
            .with_consensus(consensus)
    } else {
        BountyConfig::new(path)
            .with_model(model)
            .with_consensus(consensus)
    };

    // Override max_files if specified
    let config = BountyConfig {
        max_files,
        output_format: match format {
            "json" => BountyOutputFormat::Json,
            "markdown" | "md" => BountyOutputFormat::Markdown,
            _ => BountyOutputFormat::Text,
        },
        ..config
    };

    let mut coordinator = BountyCoordinator::new(config).map_err(|e| anyhow::anyhow!("{}", e))?;

    let result = coordinator
        .run()
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    match format {
        "json" => {
            println!("{}", smesh_bounty::results_to_json(&result));
        }
        _ => {
            smesh_bounty::print_results(&result);
        }
    }

    Ok(())
}

async fn cmd_swarm(path: &Path, max_files: usize, format: &str, consensus: u32) -> Result<()> {
    use swarm::{
        print_results, results_to_json, OutputFormat, VulnSwarmConfig, VulnSwarmCoordinator,
    };

    println!("╔═══════════════════════════════════════╗");
    println!("║      SMESH Vulnerability Swarm        ║");
    println!("╚═══════════════════════════════════════╝");
    println!();

    let config = VulnSwarmConfig::new(path.to_path_buf())
        .with_max_files(max_files)
        .with_output_format(OutputFormat::from_str(format))
        .with_consensus_threshold(consensus);

    let mut coordinator =
        VulnSwarmCoordinator::new(config).map_err(|e| anyhow::anyhow!("{}", e))?;

    let result = coordinator
        .run()
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    match OutputFormat::from_str(format) {
        OutputFormat::Json => {
            println!("{}", results_to_json(&result));
        }
        OutputFormat::Sarif => {
            // TODO: Implement SARIF output
            println!("SARIF output not yet implemented. Using JSON:");
            println!("{}", results_to_json(&result));
        }
        OutputFormat::Text => {
            print_results(&result, false);
        }
    }

    Ok(())
}

async fn cmd_code(num_coders: usize, consensus: u32, format: &str) -> Result<()> {
    use swarm::{
        coding_results_to_json, print_coding_results, CodingSwarmConfig, CodingSwarmCoordinator,
    };

    let mut config = CodingSwarmConfig::default();
    config.num_coders = num_coders;
    config.consensus_threshold = consensus;

    let mut coordinator =
        CodingSwarmCoordinator::new(config).map_err(|e| anyhow::anyhow!("{}", e))?;

    let result = coordinator
        .run()
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    match format {
        "json" => {
            println!("{}", coding_results_to_json(&result));
        }
        _ => {
            print_coding_results(&result);
        }
    }

    Ok(())
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

    // Check OpenRouter
    print!("  OpenRouter: ");
    match OpenRouterClient::from_env() {
        Some(client) => {
            println!("✓ Credentials found (key {})", client.api_key_masked());
            println!("  Model: {}", client.model());
            if client.is_available().await {
                println!("  API: ✓ Reachable");
            } else {
                println!("  API: ✗ Not reachable (check key/network)");
            }
        }
        None => {
            println!("✗ Not configured");
            println!("  Set OPENROUTER_API_KEY (or add it to ~/.creds/openrouter.env)");
        }
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

    // Report how far signals actually diffused through the network.
    {
        let net = runtime.network();
        let net_guard = net.read().await;
        let n_nodes = net_guard.nodes.len().max(1);
        let reaches: Vec<usize> = net_guard
            .field
            .active_signals()
            .map(|s| s.reached_nodes.len())
            .collect();

        if !reaches.is_empty() {
            let max_reach = reaches.iter().copied().max().unwrap_or(0);
            let avg_reach = reaches.iter().sum::<usize>() as f64 / reaches.len() as f64;
            let max_hops = net_guard
                .field
                .active_signals()
                .map(|s| s.hops)
                .max()
                .unwrap_or(0);
            println!();
            println!("Signal diffusion:");
            println!(
                "  Avg nodes reached: {:.1}/{} ({:.0}% coverage)",
                avg_reach,
                n_nodes,
                (avg_reach / n_nodes as f64) * 100.0
            );
            println!("  Max nodes reached: {}/{}", max_reach, n_nodes);
            println!("  Max hops travelled: {}", max_hops);
        }
    }

    Ok(())
}

async fn cmd_agents(n_agents: usize, model: &str, demo: bool) -> Result<()> {
    println!("╔═══════════════════════════════════════╗");
    println!("║      SMESH Agent Coordination         ║");
    println!("╚═══════════════════════════════════════╝");
    println!();

    // Check OpenRouter credentials first
    let client = match OpenRouterClient::from_env() {
        Some(mut c) => {
            c.set_model(model);
            c
        }
        None => {
            println!(
                "✗ OpenRouter not configured. Set OPENROUTER_API_KEY (or add it to ~/.creds/openrouter.env)"
            );
            return Ok(());
        }
    };

    if !client.is_available().await {
        println!("✗ OpenRouter not reachable (check OPENROUTER_API_KEY and network)");
        return Ok(());
    }

    println!("✓ OpenRouter connected");
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

    let mut coordinator = AgentCoordinator::with_openrouter(config, model);

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

async fn cmd_openrouter(model: &str) -> Result<()> {
    println!("Testing OpenRouter connection...");
    println!("Model: {}", model);
    println!();

    let client = match OpenRouterClient::from_env() {
        Some(mut c) => {
            c.set_model(model);
            c
        }
        None => {
            println!("✗ OpenRouter not configured");
            println!("  Set OPENROUTER_API_KEY (or add it to ~/.creds/openrouter.env)");
            return Ok(());
        }
    };

    println!("✓ Credentials found (key {})", client.api_key_masked());

    if !client.is_available().await {
        println!("✗ OpenRouter not reachable (check OPENROUTER_API_KEY and network)");
        return Ok(());
    }
    println!("✓ API reachable");

    // Confirm the requested model is offered.
    match client.list_models().await {
        Ok(models) => {
            println!("✓ Models available: {}", models.len());
            if models.iter().any(|m| m == model) {
                println!("✓ Model '{}' found", model);
            } else {
                println!("⚠ Model '{}' not in catalog (it may still work)", model);
            }
        }
        Err(e) => {
            println!("⚠ Could not list models: {}", e);
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
    openrouter_model: &str,
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

    // Check OpenRouter
    print!("Checking OpenRouter... ");
    let openrouter = OpenRouterClient::from_env().map(|mut c| {
        c.set_model(openrouter_model);
        c
    });
    let openrouter_available = match &openrouter {
        Some(c) => {
            let ok = c.is_available().await;
            if ok {
                println!("✓ Available (model: {})", openrouter_model);
            } else {
                println!("✗ Not reachable (check OPENROUTER_API_KEY)");
            }
            ok
        }
        None => {
            println!("✗ Not configured (set OPENROUTER_API_KEY)");
            false
        }
    };

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

    if !openrouter_available && claude_available.is_none() {
        println!("No backends available. Please:");
        println!("  - Set OPENROUTER_API_KEY (or add it to ~/.creds/openrouter.env)");
        println!("  - Or set ANTHROPIC_API_KEY environment variable");
        return Ok(());
    }

    // Run benchmarks
    for (i, (prompt, system)) in prompts.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Prompt {}: {}", i + 1, truncate_str(prompt, 50));
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        // Test OpenRouter
        if let (true, Some(or)) = (openrouter_available, &openrouter) {
            print!("  OpenRouter: ");
            let result = benchmark_backend(or, prompt, *system).await;
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

/// Run one node on a live P2P mesh over QUIC.
///
/// Each process owns a single SMESH node. Signals emitted here go out to every
/// connected peer; signals arriving from peers are admitted by this node's own
/// sensing threshold and forwarded by its own relay policy.
async fn cmd_mesh(
    bind: &str,
    peers: &[String],
    name: Option<String>,
    identity: Option<PathBuf>,
    emit: Option<String>,
    emit_after: u64,
    duration: u64,
) -> Result<()> {
    use std::sync::Arc;
    use std::time::Duration;

    let bind_addr: std::net::SocketAddr = bind
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid --bind {bind}: {e}"))?;

    let bootstrap: Vec<std::net::SocketAddr> = peers
        .iter()
        .map(|p| {
            p.parse::<std::net::SocketAddr>()
                .map_err(|e| anyhow::anyhow!("invalid --peer {p}: {e}"))
        })
        .collect::<Result<_>>()?;

    // One node per process: this is the identity we present on the wire.
    let mut node = match (name, identity) {
        // A durable key is what lets this node restart and be recognised.
        (Some(name), Some(path)) => Node::new().with_identity(
            smesh_core::NodeIdentity::load_or_create(&path, name)
                .with_context(|| format!("loading identity from {}", path.display()))?,
        ),
        (Some(name), None) => Node::named(name),
        (None, Some(path)) => {
            let generated = Node::new().id;
            Node::new().with_identity(
                smesh_core::NodeIdentity::load_or_create(&path, generated)
                    .with_context(|| format!("loading identity from {}", path.display()))?,
            )
        }
        (None, None) => Node::new(),
    };
    let node_id = node.id.clone();

    let mut network = Network::new();
    network.add_node(node);

    let mut runtime = SmeshRuntime::with_network(
        network,
        RuntimeConfig {
            tick_interval_ms: 100,
            ..Default::default()
        },
    );
    let mut events = runtime
        .take_events()
        .ok_or_else(|| anyhow::anyhow!("event receiver already taken"))?;
    let runtime = Arc::new(runtime);

    let mesh = runtime
        .join_mesh(
            MeshConfig {
                bind_addr,
                bootstrap: bootstrap.clone(),
                keepalive_interval_ms: 3_000,
                ..Default::default()
            },
            &node_id,
        )
        .await?;

    println!("╭─ SMESH mesh node");
    println!("│  node id   {node_id}");
    println!("│  listening {}", mesh.listen_addr());
    if bootstrap.is_empty() {
        println!("│  bootstrap (none — waiting for peers to dial in)");
    } else {
        for addr in &bootstrap {
            println!("│  bootstrap {addr}");
        }
    }
    println!("╰─");

    // Decay and local diffusion keep running while we are on the mesh.
    {
        let runtime = Arc::clone(&runtime);
        tokio::spawn(async move { runtime.run().await });
    }

    // Report everything except the per-tick chatter.
    let events_task = tokio::spawn(async move {
        while let Some(event) = events.recv().await {
            match event {
                RuntimeEvent::TickCompleted { .. } => {}
                RuntimeEvent::PeerConnected { peer_id } => {
                    println!("  [peer]   connected {peer_id}");
                }
                RuntimeEvent::PeerDisconnected { peer_id } => {
                    println!("  [peer]   disconnected {peer_id}");
                }
                RuntimeEvent::SignalEmitted { hash } => {
                    println!("  [emit]   {hash}");
                }
                RuntimeEvent::SignalReceived { hash, from, hops } => {
                    println!("  [recv]   {hash} from {from} (hop {hops})");
                }
                RuntimeEvent::SignalReinforced { hash, count } => {
                    println!("  [reinf]  {hash} ×{count}");
                }
                RuntimeEvent::SignalExpired { hash } => {
                    println!("  [expire] {hash}");
                }
            }
        }
    });

    if let Some(payload) = emit {
        let runtime = Arc::clone(&runtime);
        let node_id = node_id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(emit_after)).await;
            let signal = Signal::builder(SignalType::Coordination)
                .payload(payload.as_bytes().to_vec())
                .origin(&node_id)
                .intensity(1.0)
                .ttl(120.0)
                .radius(4)
                .build();
            runtime.emit(signal, &node_id).await;
        });
    }

    if duration == 0 {
        println!("\nrunning until Ctrl-C…\n");
        tokio::signal::ctrl_c().await?;
    } else {
        println!("\nrunning for {duration}s…\n");
        tokio::time::sleep(Duration::from_secs(duration)).await;
    }

    let stats = runtime.stats().await;
    println!("\n╭─ final state");
    println!("│  ticks            {}", stats.tick_count);
    println!("│  peers known      {}", stats.peer_count);
    println!("│  peers connected  {}", stats.connected_peers);
    println!("│  live connections {}", mesh.connection_count().await);
    println!("│  active signals   {}", stats.active_signals);
    println!("│  reinforcements   {}", stats.total_reinforcements);

    for peer in runtime.peers().connected_peers().await {
        println!(
            "│  peer {} at {} rtt {}ms",
            peer.id, peer.addr, peer.latency_ms
        );
    }

    let network = runtime.network();
    let network = network.read().await;
    if let Some(node) = network.get_node(&node_id) {
        println!(
            "│  emitted {} · sensed {} · relayed {} · reinforced {}",
            node.stats.signals_emitted,
            node.stats.signals_sensed,
            node.stats.signals_relayed,
            node.stats.signals_reinforced
        );
    }
    println!("╰─");

    runtime.shutdown().await;
    mesh.shutdown().await;
    events_task.abort();

    Ok(())
}
