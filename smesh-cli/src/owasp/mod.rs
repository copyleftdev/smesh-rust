//! OWASP Benchmark harness: run the SMESH detection mesh over the corpus and
//! score it against ground truth.

pub mod corpus;
pub mod detector;
pub mod report;
pub mod score;

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use smesh_agent::OpenRouterClient;

use detector::{default_agents, detect_file, FileVerdict};
use score::Scorecard;

/// Configuration for an OWASP Benchmark run.
#[derive(Debug, Clone)]
pub struct OwaspConfig {
    /// Path to a BenchmarkJava checkout.
    pub benchmark_root: PathBuf,
    /// Test cases to sample per category (0 = all).
    pub per_category: usize,
    /// OpenRouter model slug.
    pub model: String,
    /// Detector agents per test case (1..=4).
    pub agents: usize,
    /// Agents that must agree for a category to be flagged.
    pub consensus_threshold: u32,
    /// Single-agent confidence that flags a category on its own.
    pub high_conf_threshold: f64,
    /// Files analysed concurrently.
    pub concurrency: usize,
}

impl Default for OwaspConfig {
    fn default() -> Self {
        Self {
            benchmark_root: PathBuf::from(".owasp-benchmark/BenchmarkJava"),
            per_category: 25,
            model: smesh_agent::openrouter::DEFAULT_MODEL.to_string(),
            agents: 3,
            consensus_threshold: 2,
            high_conf_threshold: 0.8,
            concurrency: 6,
        }
    }
}

/// Run the benchmark end to end and return a scored scorecard.
pub async fn run_benchmark(config: OwaspConfig) -> Result<Scorecard, String> {
    // 1. Load + sample the corpus.
    let all = corpus::load(&config.benchmark_root).map_err(|e| e.to_string())?;
    let sampled = if config.per_category == 0 {
        all
    } else {
        corpus::sample_balanced(&all, config.per_category)
    };
    let total = sampled.len();
    if total == 0 {
        return Err("no test cases selected".to_string());
    }

    // 2. Build the OpenRouter backend (credentials from env / creds file).
    let mut client = OpenRouterClient::from_env().ok_or_else(|| {
        "OpenRouter not configured. Set OPENROUTER_API_KEY (or add it to ~/.creds/openrouter.env)"
            .to_string()
    })?;
    client.set_model(&config.model);
    if !client.is_available().await {
        return Err("OpenRouter not reachable (check OPENROUTER_API_KEY and network)".to_string());
    }

    let backend = Arc::new(client);
    let agents = Arc::new(default_agents(config.agents));

    println!(
        "Running SMESH mesh over {} OWASP Benchmark cases ({} agents/case, model {})...",
        total, agents.len(), config.model
    );

    // 3. Fan out detection across files with bounded concurrency.
    let done = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let results: Vec<(corpus::TestCase, FileVerdict)> = futures::stream::iter(sampled)
        .map(|case| {
            let backend = Arc::clone(&backend);
            let agents = Arc::clone(&agents);
            let done = Arc::clone(&done);
            let consensus = config.consensus_threshold;
            let high_conf = config.high_conf_threshold;
            async move {
                let content = std::fs::read_to_string(&case.file_path).unwrap_or_default();
                let verdict = if content.trim().is_empty() {
                    FileVerdict::default()
                } else {
                    detect_file(&content, backend.as_ref(), &agents, consensus, high_conf).await
                };
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(10) || n == total {
                    println!("  scanned {n}/{total}");
                }
                (case, verdict)
            }
        })
        .buffer_unordered(config.concurrency)
        .collect()
        .await;

    let elapsed = start.elapsed().as_secs_f64();

    // 4. Score against ground truth.
    Ok(score::score(&results, &config.model, agents.len(), elapsed))
}
