//! Vuln Swarm Coordinator - Orchestrates multi-agent vulnerability scanning

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Semaphore;
use tokio::time::{sleep, Duration};

use smesh_agent::{ClaudeClient, ClaudeConfig};
use smesh_core::{Field, Signal, SignalType};

use super::agent::{VulnAgent, VulnSpecialization};
use super::config::VulnSwarmConfig;
use super::findings::{parse_findings_from_response, ConsensusFinding, Severity, VulnFinding};
use super::metrics::SwarmMetrics;

/// Result of a swarm scan
#[derive(Debug)]
pub struct SwarmResult {
    /// Findings that reached consensus
    pub consensus_findings: Vec<ConsensusFinding>,
    /// All raw findings (for debugging)
    pub raw_findings: Vec<VulnFinding>,
    /// Scan metrics
    pub metrics: SwarmMetrics,
}

/// Rate limiter for API requests
struct RateLimiter {
    /// Semaphore for concurrent request limiting
    semaphore: Arc<Semaphore>,
    /// Requests made in current minute window
    requests_this_minute: u32,
    /// Start of current minute window
    minute_window_start: Instant,
    /// Max requests per minute
    requests_per_minute: u32,
}

impl RateLimiter {
    fn new(max_concurrent: usize, requests_per_minute: u32) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            requests_this_minute: 0,
            minute_window_start: Instant::now(),
            requests_per_minute,
        }
    }

    async fn acquire(&mut self) {
        // Check minute window
        if self.minute_window_start.elapsed() >= Duration::from_secs(60) {
            self.requests_this_minute = 0;
            self.minute_window_start = Instant::now();
        }

        // Wait if at rate limit
        if self.requests_this_minute >= self.requests_per_minute {
            let wait_time = Duration::from_secs(60) - self.minute_window_start.elapsed();
            if !wait_time.is_zero() {
                sleep(wait_time).await;
            }
            self.requests_this_minute = 0;
            self.minute_window_start = Instant::now();
        }

        // Acquire semaphore permit
        let _permit = self.semaphore.acquire().await.unwrap();
        self.requests_this_minute += 1;
    }
}

/// Coordinator for the vulnerability swarm
pub struct VulnSwarmCoordinator {
    /// Configuration
    config: VulnSwarmConfig,
    /// Vulnerability agents
    agents: Vec<VulnAgent>,
    /// SMESH signal field
    field: Field,
    /// Active findings (dedup_id -> finding)
    findings: HashMap<String, VulnFinding>,
    /// Consensus findings
    consensus_findings: Vec<ConsensusFinding>,
    /// Metrics
    metrics: SwarmMetrics,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// Tick counter
    tick_count: u32,
    /// Scan start time
    start_time: Option<Instant>,
}

impl VulnSwarmCoordinator {
    /// Create a new coordinator with the given configuration
    pub fn new(config: VulnSwarmConfig) -> Result<Self, String> {
        // Get Claude client
        let claude_config = ClaudeConfig::from_env().ok_or("ANTHROPIC_API_KEY not set")?;
        let claude_config = claude_config.with_model(&config.model);

        // Create 10 agents: some specializations get 2 instances
        let mut agents = Vec::new();
        let specializations = VulnSpecialization::all();

        // Create 2 agents for Injection, XSS, Auth (most common vulns)
        // and 1 agent for each other specialization
        for spec in &specializations {
            let instances = match spec {
                VulnSpecialization::Injection
                | VulnSpecialization::Xss
                | VulnSpecialization::Auth => 2,
                _ => 1,
            };

            for i in 1..=instances {
                let client = ClaudeClient::new(claude_config.clone());
                agents.push(VulnAgent::new(*spec, i, client));
            }
        }

        let rate_limiter =
            RateLimiter::new(config.max_concurrent_requests, config.requests_per_minute);

        Ok(Self {
            config,
            agents,
            field: Field::new(),
            findings: HashMap::new(),
            consensus_findings: Vec::new(),
            metrics: SwarmMetrics::new(),
            rate_limiter,
            tick_count: 0,
            start_time: None,
        })
    }

    /// Run the vulnerability scan
    pub async fn run(&mut self) -> Result<SwarmResult, String> {
        self.start_time = Some(Instant::now());

        println!("\nStarting Vuln Swarm scan...");
        println!("Created {} vulnerability agents\n", self.agents.len());

        // Collect files to scan
        let files = self.collect_files()?;
        let total_files = files.len().min(self.config.max_files);
        println!("Found {} files to analyze\n", total_files);

        // Analyze each file
        for (idx, file_path) in files.iter().take(total_files).enumerate() {
            self.analyze_file(idx + 1, total_files, file_path).await?;

            // Tick the field after each file
            self.tick_field();
        }

        // Post-analysis phase: additional ticks for consensus
        println!("\nRunning consensus phase (20 ticks)...");
        for _ in 0..20 {
            self.tick_field();
            sleep(Duration::from_millis(self.config.tick_interval_ms / 10)).await;
        }

        // Finalize metrics
        self.metrics.total_duration = self.start_time.unwrap().elapsed();
        self.metrics.aggregate_from_agents(&self.agents);

        // Collect raw findings
        let raw_findings: Vec<VulnFinding> = self.findings.values().cloned().collect();

        Ok(SwarmResult {
            consensus_findings: self.consensus_findings.clone(),
            raw_findings,
            metrics: self.metrics.clone(),
        })
    }

    /// Collect files to scan from target path
    fn collect_files(&self) -> Result<Vec<PathBuf>, String> {
        let target = &self.config.target_path;
        let extensions = self.config.scannable_extensions();

        if target.is_file() {
            return Ok(vec![target.clone()]);
        }

        let mut files = Vec::new();
        self.collect_files_recursive(target, extensions, &mut files)?;

        // Sort by modification time (most recent first)
        files.sort_by(|a, b| {
            let a_time = fs::metadata(a).and_then(|m| m.modified()).ok();
            let b_time = fs::metadata(b).and_then(|m| m.modified()).ok();
            b_time.cmp(&a_time)
        });

        Ok(files)
    }

    fn collect_files_recursive(
        &self,
        dir: &Path,
        extensions: &[&str],
        files: &mut Vec<PathBuf>,
    ) -> Result<(), String> {
        if !dir.is_dir() {
            return Ok(());
        }

        let entries = fs::read_dir(dir).map_err(|e| e.to_string())?;

        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            // Skip hidden directories and common non-source dirs
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.starts_with('.')
                    || name == "target"
                    || name == "node_modules"
                    || name == "vendor"
                    || name == "dist"
                    || name == "build"
                    || name == "__pycache__"
                {
                    continue;
                }
                self.collect_files_recursive(&path, extensions, files)?;
            } else if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if extensions.contains(&ext_str.as_str()) {
                    files.push(path);
                }
            }
        }

        Ok(())
    }

    /// Analyze a single file
    async fn analyze_file(
        &mut self,
        idx: usize,
        total: usize,
        file_path: &Path,
    ) -> Result<(), String> {
        let relative_path = file_path
            .strip_prefix(&self.config.target_path)
            .unwrap_or(file_path)
            .display()
            .to_string();

        println!("[{}/{}] Analyzing: {}", idx, total, relative_path);

        // Read file content
        let content = fs::read_to_string(file_path).map_err(|e| e.to_string())?;

        // Skip small files
        if content.lines().count() < 5 {
            println!("  Skipped (too small)\n");
            self.metrics.record_file_skipped();
            return Ok(());
        }

        self.metrics.record_file_scanned();

        // Truncate content if too large
        let max_chars = self.config.max_tokens_per_file * 4; // ~4 chars per token
        let truncated = if content.len() > max_chars {
            &content[..max_chars]
        } else {
            &content
        };

        // Score agent relevance and select top 5
        let mut agent_scores: Vec<(usize, f64)> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, agent)| (i, agent.relevance_to_file(truncated)))
            .collect();

        agent_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_agents: Vec<usize> = agent_scores.iter().take(5).map(|(i, _)| *i).collect();

        // Analyze with top agents
        for agent_idx in top_agents {
            self.rate_limiter.acquire().await;

            // Need to split borrow to avoid holding mutable ref across await
            let agent_id = self.agents[agent_idx].id.clone();
            let spec = self.agents[agent_idx].specialization;

            let response = self.agents[agent_idx]
                .analyze(&relative_path, truncated)
                .await;

            match response {
                Ok(response_text) => {
                    // Parse findings from response
                    let new_findings = parse_findings_from_response(
                        &response_text,
                        &relative_path,
                        &agent_id,
                        spec.name(),
                    );

                    for finding in new_findings {
                        self.process_finding(finding, &agent_id);
                    }
                }
                Err(e) => {
                    eprintln!("  {} error: {}", agent_id, e);
                }
            }
        }

        println!();
        Ok(())
    }

    /// Process a new finding (emit or reinforce)
    fn process_finding(&mut self, finding: VulnFinding, agent_id: &str) {
        let dedup_id = finding.dedup_id.clone();

        if let Some(existing) = self.findings.get_mut(&dedup_id) {
            // Reinforce existing finding
            existing.reinforce(agent_id);
            self.metrics.record_signal_reinforced();

            // Update agent metrics
            for agent in &mut self.agents {
                if agent.id == agent_id {
                    agent.record_reinforcement();
                    break;
                }
            }

            let sev = existing.severity;
            let conf = existing.confidence;
            let count = existing.reinforcement_count;
            let short_id = &existing.dedup_id[..8];

            println!(
                "  {}[{}]{} {}-{} reinforced {} ({}x, {:.0}%)",
                sev.color(),
                sev.name(),
                Severity::reset(),
                agent_id,
                sev.name(),
                short_id,
                count + 1,
                conf * 100.0
            );

            // Check for consensus
            if existing.has_consensus(
                self.config.consensus_threshold,
                self.config.high_confidence_threshold,
            ) {
                // Check if not already in consensus list
                if !self
                    .consensus_findings
                    .iter()
                    .any(|cf| cf.finding.dedup_id == dedup_id)
                {
                    let consensus = ConsensusFinding::from_finding(
                        existing.clone(),
                        existing.found_at,
                        self.tick_count,
                    );
                    self.metrics.record_consensus(&consensus);
                    self.consensus_findings.push(consensus);

                    println!(
                        "\n  \x1b[1;32mCONSENSUS: {} - {} ({}x reinforced)\x1b[0m\n",
                        existing.vuln_type,
                        existing.file_path,
                        existing.reinforcement_count + 1
                    );
                }
            }
        } else {
            // Emit new finding as signal
            let sev = finding.severity;
            let conf = finding.confidence;
            let short_id = &finding.dedup_id[..8.min(finding.dedup_id.len())];

            // Create TOON payload for signal
            let payload = finding.to_compact_payload();
            let toon_bytes = serde_json::to_vec(&payload).unwrap_or_default();

            // Record TOON savings (compare to full JSON)
            let full_json = serde_json::to_vec(&finding).unwrap_or_default();
            self.metrics
                .record_toon_savings(full_json.len(), toon_bytes.len());

            // Emit signal to field
            let signal = Signal::builder(SignalType::Alert)
                .payload_toon(&payload)
                .intensity(severity_to_intensity(sev))
                .confidence(conf)
                .ttl(self.config.signal_ttl_secs)
                .origin(agent_id)
                .build();

            self.field.emit_anonymous(signal);
            self.metrics.record_signal_emitted();
            self.metrics.record_raw_finding(sev);

            // Update agent metrics
            for agent in &mut self.agents {
                if agent.id == agent_id {
                    agent.record_finding();
                    break;
                }
            }

            println!(
                "  {}[{}]{} {} found: {} at {} (conf: {:.0}%)",
                sev.color(),
                sev.name(),
                Severity::reset(),
                agent_id,
                finding.vuln_type,
                short_id,
                conf * 100.0
            );

            self.findings.insert(dedup_id, finding);
        }
    }

    /// Tick the signal field (decay, expiration)
    fn tick_field(&mut self) {
        let expired = self
            .field
            .tick(self.config.tick_interval_ms as f64 / 1000.0);
        self.tick_count += 1;
        self.metrics.record_tick();

        for _ in 0..expired {
            self.metrics.record_signal_expired();
        }
    }

    /// Get the metrics
    pub fn metrics(&self) -> &SwarmMetrics {
        &self.metrics
    }
}

/// Map severity to signal intensity
fn severity_to_intensity(severity: Severity) -> f64 {
    match severity {
        Severity::Critical => 1.0,
        Severity::High => 0.9,
        Severity::Medium => 0.7,
        Severity::Low => 0.5,
        Severity::Info => 0.3,
    }
}

/// Print the scan results
pub fn print_results(result: &SwarmResult, verbose: bool) {
    println!("\n{}", result.metrics.summary_report());

    if result.consensus_findings.is_empty() {
        println!("No vulnerability findings reached consensus.\n");
        return;
    }

    println!("\nConsensus Findings:\n");

    // Sort by severity
    let mut sorted = result.consensus_findings.clone();
    sorted.sort_by(|a, b| b.finding.severity.cmp(&a.finding.severity));

    for cf in &sorted {
        let f = &cf.finding;
        println!(
            "{}[{}]{} {} - {}",
            f.severity.color(),
            f.severity.name(),
            Severity::reset(),
            f.vuln_type,
            f.file_path
        );

        if let Some(line) = f.line_number {
            println!("  Line: {}", line);
        }

        println!("  Confidence: {:.0}%", f.confidence * 100.0);
        println!("  Reinforced by: {} agents", f.reinforcement_count + 1);
        println!("  Time to consensus: {}ms", cf.time_to_consensus_ms);

        if verbose {
            println!("  Description: {}", f.description);
            println!("  Reinforcers: {}", f.reinforced_by.join(", "));
        }

        println!();
    }
}

/// Generate JSON output
pub fn results_to_json(result: &SwarmResult) -> String {
    #[derive(serde::Serialize)]
    struct JsonOutput {
        findings: Vec<JsonFinding>,
        metrics: JsonMetrics,
    }

    #[derive(serde::Serialize)]
    struct JsonFinding {
        severity: String,
        vuln_type: String,
        file_path: String,
        line_number: Option<u32>,
        confidence: f64,
        description: String,
        reinforcement_count: u32,
        time_to_consensus_ms: u64,
    }

    #[derive(serde::Serialize)]
    struct JsonMetrics {
        files_scanned: u32,
        raw_findings: u32,
        consensus_findings: u32,
        total_tokens: u32,
        toon_savings_percent: f64,
        duration_secs: f64,
    }

    let findings: Vec<JsonFinding> = result
        .consensus_findings
        .iter()
        .map(|cf| JsonFinding {
            severity: cf.finding.severity.name().to_string(),
            vuln_type: cf.finding.vuln_type.clone(),
            file_path: cf.finding.file_path.clone(),
            line_number: cf.finding.line_number,
            confidence: cf.finding.confidence,
            description: cf.finding.description.clone(),
            reinforcement_count: cf.finding.reinforcement_count,
            time_to_consensus_ms: cf.time_to_consensus_ms,
        })
        .collect();

    let output = JsonOutput {
        findings,
        metrics: JsonMetrics {
            files_scanned: result.metrics.files_scanned,
            raw_findings: result.metrics.raw_findings,
            consensus_findings: result.metrics.consensus_findings,
            total_tokens: result.metrics.total_tokens(),
            toon_savings_percent: result.metrics.toon_savings_percentage(),
            duration_secs: result.metrics.total_duration.as_secs_f64(),
        },
    };

    serde_json::to_string_pretty(&output).unwrap_or_default()
}
