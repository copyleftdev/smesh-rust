//! BountyCoordinator - orchestrates the bounty hunting swarm
//!
//! Combines SMESH signal-based coordination with tool-equipped agents.
//!
//! Flow:
//! 1. Create hunters based on team composition
//! 2. Recon agent maps attack surface, emits target signals
//! 3. Source/Config/Dep agents claim files based on specialization relevance
//! 4. Agents execute tasks with agentic tool-use loops
//! 5. Findings are emitted as SMESH signals, reinforced by other agents
//! 6. Triager deduplicates and correlates findings
//! 7. Report writer produces final output

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, info, warn};

use smesh_agent::{ClaudeClient, ClaudeConfig, LlmBackend};
use smesh_core::{Field, Network, NetworkTopology};

use crate::hunter::{BountyHunter, HunterConfig, HunterFinding, HunterMetrics};
use crate::specialization::BountySpecialization;
use crate::tools::SandboxConfig;

/// Configuration for a bounty scan
#[derive(Debug, Clone)]
pub struct BountyConfig {
    /// Target directory to scan
    pub target_path: PathBuf,
    /// Team composition: (specialization, count)
    pub team: Vec<(BountySpecialization, u32)>,
    /// Claude model to use
    pub model: String,
    /// Maximum files to analyze
    pub max_files: usize,
    /// Consensus threshold (agents that must agree on a finding)
    pub consensus_threshold: u32,
    /// Max concurrent API requests
    pub max_concurrent: usize,
    /// Output format
    pub output_format: OutputFormat,
}

/// Output format for results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Sarif,
    Markdown,
}

impl Default for BountyConfig {
    fn default() -> Self {
        Self {
            target_path: PathBuf::from("."),
            team: BountySpecialization::default_team(),
            model: "claude-sonnet-4-20250514".to_string(),
            max_files: 50,
            consensus_threshold: 2,
            max_concurrent: 5,
            output_format: OutputFormat::Text,
        }
    }
}

impl BountyConfig {
    /// Create config for a target path
    pub fn new(target_path: impl Into<PathBuf>) -> Self {
        Self {
            target_path: target_path.into(),
            ..Default::default()
        }
    }

    /// Set the model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set team composition
    pub fn with_team(mut self, team: Vec<(BountySpecialization, u32)>) -> Self {
        self.team = team;
        self
    }

    /// Set output format
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set consensus threshold
    pub fn with_consensus(mut self, threshold: u32) -> Self {
        self.consensus_threshold = threshold;
        self
    }

    /// Quick scan: fewer agents, lower consensus
    pub fn quick(target_path: impl Into<PathBuf>) -> Self {
        Self {
            target_path: target_path.into(),
            team: vec![
                (BountySpecialization::SourceAudit, 2),
                (BountySpecialization::ConfigAudit, 1),
                (BountySpecialization::DependencyHunter, 1),
            ],
            max_files: 20,
            consensus_threshold: 1,
            ..Default::default()
        }
    }
}

/// A finding that reached consensus across agents
#[derive(Debug, Clone)]
pub struct ConsensusFinding {
    /// The core finding
    pub finding: HunterFinding,
    /// Number of agents that confirmed it
    pub confirmations: u32,
    /// Agent IDs that found/confirmed it
    pub confirmed_by: Vec<String>,
    /// Time to consensus in ms
    pub time_to_consensus_ms: u64,
}

/// Result of a bounty scan
#[derive(Debug)]
pub struct BountyResult {
    /// Findings that reached consensus
    pub consensus_findings: Vec<ConsensusFinding>,
    /// All raw findings
    pub raw_findings: Vec<HunterFinding>,
    /// Per-agent metrics
    pub agent_metrics: Vec<(String, BountySpecialization, HunterMetrics)>,
    /// Total scan duration
    pub duration: std::time::Duration,
    /// Files scanned
    pub files_scanned: usize,
    /// Target path
    pub target_path: PathBuf,
}

/// The bounty hunting swarm coordinator
#[allow(dead_code)]
pub struct BountyCoordinator {
    config: BountyConfig,
    hunters: Vec<BountyHunter>,
    field: Field,
    network: Network,
    raw_findings: Vec<HunterFinding>,
    consensus_findings: Vec<ConsensusFinding>,
    start_time: Option<Instant>,
}

impl BountyCoordinator {
    /// Create a new coordinator
    pub fn new(config: BountyConfig) -> Result<Self, String> {
        let claude_config = ClaudeConfig::from_env()
            .ok_or("ANTHROPIC_API_KEY not set. Export it to use Claude-powered bounty hunting.")?;
        let claude_config = claude_config.with_model(&config.model);

        let sandbox = SandboxConfig::rooted(&config.target_path);

        // Create hunters based on team composition
        let mut hunters = Vec::new();
        for (spec, count) in &config.team {
            for i in 1..=*count {
                let hunter_config = HunterConfig::new(*spec, i);
                let backend: Arc<dyn LlmBackend> =
                    Arc::new(ClaudeClient::new(claude_config.clone()));
                let hunter = BountyHunter::new(hunter_config, backend, sandbox.clone());
                hunters.push(hunter);
            }
        }

        // Build SMESH network connecting all hunters
        let n_hunters = hunters.len();
        let mut network = Network::with_topology(0, NetworkTopology::FullMesh);

        // Add hunter nodes to network
        for hunter in &hunters {
            network.add_node(hunter.node.clone());
        }

        // Connect all hunters (full mesh for small teams)
        let ids: Vec<String> = hunters.iter().map(|h| h.node.id.clone()).collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                network.connect_bidirectional(&ids[i], &ids[j]);
            }
        }

        info!(
            "Created bounty swarm: {} hunters, {} connections",
            n_hunters,
            network.stats().connection_count
        );

        Ok(Self {
            config,
            hunters,
            field: Field::new(),
            network,
            raw_findings: Vec::new(),
            consensus_findings: Vec::new(),
            start_time: None,
        })
    }

    /// Run the bounty scan
    pub async fn run(&mut self) -> Result<BountyResult, String> {
        self.start_time = Some(Instant::now());

        println!("\n{}", BANNER);
        println!("Target: {}", self.config.target_path.display());
        println!("Agents: {}", self.hunters.len());
        for (spec, count) in &self.config.team {
            println!("  {} x{}", spec.name(), count);
        }
        println!();

        // Phase 1: Collect files
        let files = self.collect_files()?;
        let total_files = files.len().min(self.config.max_files);
        println!("Found {} scannable files\n", total_files);

        // Phase 2: Recon (if we have recon agents)
        self.run_recon_phase(&files).await;

        // Phase 3: Main analysis - route files to specialists
        self.run_analysis_phase(&files, total_files).await;

        // Phase 4: Dependency audit
        self.run_dependency_phase().await;

        // Phase 5: Triage and consensus
        self.run_triage_phase().await;

        // Phase 6: Report generation
        self.run_report_phase().await;

        let duration = self.start_time.unwrap().elapsed();

        // Collect metrics
        let agent_metrics: Vec<_> = self
            .hunters
            .iter()
            .map(|h| {
                (
                    h.name().to_string(),
                    h.specialization(),
                    h.metrics.clone(),
                )
            })
            .collect();

        Ok(BountyResult {
            consensus_findings: self.consensus_findings.clone(),
            raw_findings: self.raw_findings.clone(),
            agent_metrics,
            duration,
            files_scanned: total_files,
            target_path: self.config.target_path.clone(),
        })
    }

    /// Phase 1: Recon agents map the attack surface
    async fn run_recon_phase(&mut self, files: &[PathBuf]) {
        let recon_indices: Vec<usize> = self
            .hunters
            .iter()
            .enumerate()
            .filter(|(_, h)| h.specialization() == BountySpecialization::Recon)
            .map(|(i, _)| i)
            .collect();

        if recon_indices.is_empty() {
            return;
        }

        println!("--- Phase 1: Reconnaissance ---\n");

        let file_list: String = files
            .iter()
            .take(100)
            .map(|f| {
                f.strip_prefix(&self.config.target_path)
                    .unwrap_or(f)
                    .display()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n");

        let task = format!(
            "Map the attack surface of this project.\n\n\
             Project root: {}\n\n\
             Files in scope:\n{}\n\n\
             Identify: endpoints, services, entry points, technology stack, \
             and areas most likely to contain vulnerabilities.",
            self.config.target_path.display(),
            file_list,
        );

        for idx in recon_indices {
            println!("  {} scanning...", self.hunters[idx].name());
            match self.hunters[idx].execute_task(&task).await {
                Ok(findings) => {
                    for finding in &findings {
                        println!("  Found: {} - {}", finding.vuln_type, finding.description);
                        let signal = self.hunters[idx].finding_to_signal(finding);
                        self.field.emit_anonymous(signal);
                    }
                    self.raw_findings.extend(findings);
                }
                Err(e) => warn!("{} recon failed: {}", self.hunters[idx].name(), e),
            }
        }

        println!();
    }

    /// Phase 3: Route files to specialist agents based on relevance
    async fn run_analysis_phase(&mut self, files: &[PathBuf], max_files: usize) {
        let analysis_specs = [
            BountySpecialization::SourceAudit,
            BountySpecialization::ConfigAudit,
        ];

        let analyst_indices: Vec<usize> = self
            .hunters
            .iter()
            .enumerate()
            .filter(|(_, h)| analysis_specs.contains(&h.specialization()))
            .map(|(i, _)| i)
            .collect();

        if analyst_indices.is_empty() {
            return;
        }

        println!("--- Phase 2: Source Analysis ---\n");

        for (file_idx, file_path) in files.iter().take(max_files).enumerate() {
            let relative = file_path
                .strip_prefix(&self.config.target_path)
                .unwrap_or(file_path)
                .display()
                .to_string();

            // Read file for relevance scoring
            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if content.lines().count() < 5 {
                continue;
            }

            // Score each analyst's relevance to this file
            let mut scored: Vec<(usize, f64)> = analyst_indices
                .iter()
                .map(|&idx| {
                    let score = self.hunters[idx]
                        .specialization()
                        .relevance_score(&content);
                    (idx, score)
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Assign to top 2 most relevant agents
            let top_agents: Vec<usize> = scored.iter().take(2).map(|(idx, _)| *idx).collect();

            println!(
                "[{}/{}] {} -> {}",
                file_idx + 1,
                max_files,
                relative,
                top_agents
                    .iter()
                    .map(|i| self.hunters[*i].name())
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            for &agent_idx in &top_agents {
                let task = format!(
                    "Analyze this file for security vulnerabilities:\n\
                     File: {}\n\n\
                     Read the file and search for vulnerability patterns. \
                     Report any findings with severity, type, file path, line number, and description.",
                    relative,
                );

                match self.hunters[agent_idx].execute_task(&task).await {
                    Ok(findings) => {
                        for finding in &findings {
                            let sev_color = match finding.severity.as_str() {
                                "CRITICAL" => "\x1b[1;31m",
                                "HIGH" => "\x1b[31m",
                                "MEDIUM" => "\x1b[33m",
                                "LOW" => "\x1b[36m",
                                _ => "\x1b[37m",
                            };
                            println!(
                                "  {}[{}]\x1b[0m {} - {} (conf: {:.0}%)",
                                sev_color,
                                finding.severity,
                                finding.vuln_type,
                                finding.file_path,
                                finding.confidence * 100.0
                            );

                            let signal = self.hunters[agent_idx].finding_to_signal(finding);
                            self.field.emit_anonymous(signal);
                        }
                        self.raw_findings.extend(findings);
                    }
                    Err(e) => {
                        warn!("{} analysis failed: {}", self.hunters[agent_idx].name(), e);
                    }
                }
            }
        }

        println!();
    }

    /// Phase 4: Dependency audit
    async fn run_dependency_phase(&mut self) {
        let dep_indices: Vec<usize> = self
            .hunters
            .iter()
            .enumerate()
            .filter(|(_, h)| h.specialization() == BountySpecialization::DependencyHunter)
            .map(|(i, _)| i)
            .collect();

        if dep_indices.is_empty() {
            return;
        }

        println!("--- Phase 3: Dependency Audit ---\n");

        let task = format!(
            "Audit dependencies for known vulnerabilities in project at: {}\n\n\
             1. Find all dependency manifests (Cargo.toml, package.json, requirements.txt, etc.)\n\
             2. Read them and identify third-party packages\n\
             3. Run available audit tools (cargo audit, npm audit, etc.)\n\
             4. Report any known CVEs or security issues",
            self.config.target_path.display(),
        );

        for idx in dep_indices {
            println!("  {} auditing dependencies...", self.hunters[idx].name());
            match self.hunters[idx].execute_task(&task).await {
                Ok(findings) => {
                    for finding in &findings {
                        println!(
                            "  [{}] {} - {}",
                            finding.severity, finding.vuln_type, finding.description
                        );
                        let signal = self.hunters[idx].finding_to_signal(finding);
                        self.field.emit_anonymous(signal);
                    }
                    self.raw_findings.extend(findings);
                }
                Err(e) => warn!("{} dep audit failed: {}", self.hunters[idx].name(), e),
            }
        }

        println!();
    }

    /// Phase 5: Triage findings with consensus
    async fn run_triage_phase(&mut self) {
        println!("--- Phase 4: Triage & Consensus ---\n");

        // Build consensus from raw findings
        let mut finding_groups: HashMap<String, Vec<HunterFinding>> = HashMap::new();

        for finding in &self.raw_findings {
            // Group by (severity + vuln_type + file_path) as dedup key
            let key = format!(
                "{}:{}:{}",
                finding.severity, finding.vuln_type, finding.file_path
            );
            finding_groups.entry(key).or_default().push(finding.clone());
        }

        for (key, group) in &finding_groups {
            let confirmations = group.len() as u32;
            let best = group
                .iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                .unwrap();

            let confirmed_by: Vec<String> = group.iter().map(|f| f.found_by.clone()).collect();

            if confirmations >= self.config.consensus_threshold {
                println!(
                    "  CONSENSUS: [{}] {} at {} ({}x confirmed by {})",
                    best.severity,
                    best.vuln_type,
                    best.file_path,
                    confirmations,
                    confirmed_by.join(", ")
                );

                self.consensus_findings.push(ConsensusFinding {
                    finding: best.clone(),
                    confirmations,
                    confirmed_by,
                    time_to_consensus_ms: self
                        .start_time
                        .map(|s| s.elapsed().as_millis() as u64)
                        .unwrap_or(0),
                });
            } else {
                debug!(
                    "Below threshold: {} ({}/{} confirmations)",
                    key, confirmations, self.config.consensus_threshold
                );
            }
        }

        // Sort by severity
        self.consensus_findings.sort_by(|a, b| {
            severity_rank(&b.finding.severity).cmp(&severity_rank(&a.finding.severity))
        });

        println!(
            "\n  {} raw findings -> {} consensus findings\n",
            self.raw_findings.len(),
            self.consensus_findings.len()
        );
    }

    /// Phase 6: Report generation
    async fn run_report_phase(&mut self) {
        // Report writer agents are optional - if none, skip
        let report_indices: Vec<usize> = self
            .hunters
            .iter()
            .enumerate()
            .filter(|(_, h)| h.specialization() == BountySpecialization::ReportWriter)
            .map(|(i, _)| i)
            .collect();

        if report_indices.is_empty() || self.consensus_findings.is_empty() {
            return;
        }

        println!("--- Phase 5: Report Generation ---\n");

        // Build findings summary for report writer
        let findings_text: String = self
            .consensus_findings
            .iter()
            .enumerate()
            .map(|(i, cf)| {
                format!(
                    "{}. [{}] {} at {}{}\n   Confirmed by: {}\n   Description: {}",
                    i + 1,
                    cf.finding.severity,
                    cf.finding.vuln_type,
                    cf.finding.file_path,
                    cf.finding
                        .line_number
                        .map(|l| format!(":{}", l))
                        .unwrap_or_default(),
                    cf.confirmed_by.join(", "),
                    cf.finding.description,
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let task = format!(
            "Write a security assessment report for: {}\n\n\
             Consensus Findings:\n{}\n\n\
             Total files scanned: {}\n\
             Include: executive summary, finding details, remediation roadmap.",
            self.config.target_path.display(),
            findings_text,
            self.config.max_files,
        );

        for idx in report_indices {
            println!("  {} writing report...", self.hunters[idx].name());
            match self.hunters[idx].execute_task(&task).await {
                Ok(_) => {
                    info!("Report generated by {}", self.hunters[idx].name());
                }
                Err(e) => warn!("{} report failed: {}", self.hunters[idx].name(), e),
            }
        }

        println!();
    }

    /// Collect files to scan
    fn collect_files(&self) -> Result<Vec<PathBuf>, String> {
        let target = &self.config.target_path;
        let extensions = SCANNABLE_EXTENSIONS;

        if target.is_file() {
            return Ok(vec![target.clone()]);
        }

        let mut files = Vec::new();

        for entry in walkdir::WalkDir::new(target)
            .max_depth(10)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                !name.starts_with('.')
                    && name != "target"
                    && name != "node_modules"
                    && name != "vendor"
                    && name != "dist"
                    && name != "build"
                    && name != "__pycache__"
                    && name != ".git"
            })
        {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            if !entry.file_type().is_file() {
                continue;
            }

            if let Some(ext) = entry.path().extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if extensions.contains(&ext_str.as_str()) {
                    files.push(entry.path().to_path_buf());
                }
            }
        }

        // Sort by modification time (newest first)
        files.sort_by(|a, b| {
            let a_time = std::fs::metadata(a).and_then(|m| m.modified()).ok();
            let b_time = std::fs::metadata(b).and_then(|m| m.modified()).ok();
            b_time.cmp(&a_time)
        });

        Ok(files)
    }
}

fn severity_rank(sev: &str) -> u8 {
    match sev.to_uppercase().as_str() {
        "CRITICAL" => 5,
        "HIGH" => 4,
        "MEDIUM" => 3,
        "LOW" => 2,
        "INFO" => 1,
        _ => 0,
    }
}

const SCANNABLE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "rb", "php", "c", "cpp", "h", "hpp",
    "cs", "swift", "kt", "scala", "sql", "sh", "bash", "yaml", "yml", "json", "xml", "html",
    "htm", "toml", "cfg", "ini", "conf", "env", "dockerfile",
];

const BANNER: &str = r#"
 ____                    _           _   _             _
| __ )  ___  _   _ _ __ | |_ _   _  | | | |_   _ _ __ | |_ ___ _ __
|  _ \ / _ \| | | | '_ \| __| | | | | |_| | | | | '_ \| __/ _ \ '__|
| |_) | (_) | |_| | | | | |_| |_| | |  _  | |_| | | | | ||  __/ |
|____/ \___/ \__,_|_| |_|\__|\__, | |_| |_|\__,_|_| |_|\__\___|_|
                              |___/
  SMESH-Powered Security Bounty Swarm
"#;

/// Print results to terminal
pub fn print_results(result: &BountyResult) {
    println!("====================================");
    println!("  BOUNTY SCAN RESULTS");
    println!("====================================\n");

    println!(
        "Target: {}",
        result.target_path.display()
    );
    println!("Files scanned: {}", result.files_scanned);
    println!("Duration: {:.1}s", result.duration.as_secs_f64());
    println!(
        "Raw findings: {} -> Consensus: {}\n",
        result.raw_findings.len(),
        result.consensus_findings.len()
    );

    // Agent metrics
    println!("Agent Metrics:");
    for (name, spec, metrics) in &result.agent_metrics {
        if metrics.llm_calls > 0 {
            println!(
                "  {} ({}): {} LLM calls, {} tool calls, {} findings",
                name,
                spec.name(),
                metrics.llm_calls,
                metrics.tool_calls,
                metrics.findings_emitted
            );
        }
    }

    println!();

    if result.consensus_findings.is_empty() {
        println!("No vulnerability findings reached consensus.\n");
        return;
    }

    println!("Consensus Findings:\n");

    for cf in &result.consensus_findings {
        let f = &cf.finding;
        let sev_color = match f.severity.as_str() {
            "CRITICAL" => "\x1b[1;31m",
            "HIGH" => "\x1b[31m",
            "MEDIUM" => "\x1b[33m",
            "LOW" => "\x1b[36m",
            _ => "\x1b[37m",
        };

        println!(
            "{}[{}]\x1b[0m {} - {}{}",
            sev_color,
            f.severity,
            f.vuln_type,
            f.file_path,
            f.line_number
                .map(|l| format!(":{}", l))
                .unwrap_or_default()
        );
        println!("  Confidence: {:.0}%", f.confidence * 100.0);
        println!(
            "  Confirmed by: {} agents ({})",
            cf.confirmations,
            cf.confirmed_by.join(", ")
        );
        println!("  {}", f.description);
        println!();
    }
}

/// Generate JSON output
pub fn results_to_json(result: &BountyResult) -> String {
    #[derive(serde::Serialize)]
    struct JsonOutput {
        target: String,
        files_scanned: usize,
        duration_secs: f64,
        findings: Vec<JsonFinding>,
    }

    #[derive(serde::Serialize)]
    struct JsonFinding {
        severity: String,
        vuln_type: String,
        file_path: String,
        line_number: Option<u32>,
        confidence: f64,
        description: String,
        confirmations: u32,
        confirmed_by: Vec<String>,
    }

    let output = JsonOutput {
        target: result.target_path.display().to_string(),
        files_scanned: result.files_scanned,
        duration_secs: result.duration.as_secs_f64(),
        findings: result
            .consensus_findings
            .iter()
            .map(|cf| JsonFinding {
                severity: cf.finding.severity.clone(),
                vuln_type: cf.finding.vuln_type.clone(),
                file_path: cf.finding.file_path.clone(),
                line_number: cf.finding.line_number,
                confidence: cf.finding.confidence,
                description: cf.finding.description.clone(),
                confirmations: cf.confirmations,
                confirmed_by: cf.confirmed_by.clone(),
            })
            .collect(),
    };

    serde_json::to_string_pretty(&output).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounty_config_default() {
        let config = BountyConfig::default();
        let total_agents: u32 = config.team.iter().map(|(_, c)| c).sum();
        assert_eq!(total_agents, 9);
    }

    #[test]
    fn test_quick_config() {
        let config = BountyConfig::quick("/tmp/test");
        assert_eq!(config.consensus_threshold, 1);
        assert_eq!(config.max_files, 20);
    }

    #[test]
    fn test_severity_rank() {
        assert!(severity_rank("CRITICAL") > severity_rank("HIGH"));
        assert!(severity_rank("HIGH") > severity_rank("MEDIUM"));
        assert!(severity_rank("MEDIUM") > severity_rank("LOW"));
    }
}
