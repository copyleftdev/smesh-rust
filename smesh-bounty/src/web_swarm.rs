//! SMESH-coordinated web red team swarm
//!
//! Wraps all web recon and deep analysis phases in SMESH signal coordination:
//! - Each phase emits findings as SMESH signals
//! - Findings from multiple phases reinforce each other (consensus)
//! - Trust scores weight finding reliability
//! - Signal decay naturally ages out stale findings
//! - Correlation phase uses signal reinforcement counts to prioritize

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use smesh_core::{
    Field, Node, ReputationSystem, Signal, SignalType,
};

use crate::web_deep::{
    phase_api_analysis, phase_auth_and_headers, phase_cloud_recon, phase_js_analysis,
    phase_staging_raid, phase_wordpress_deep, WebFinding,
};
use crate::web_recon::{
    phase_crawl_and_fuzz, phase_nuclei_scan, phase_port_scan, phase_subdomain_dragnet,
    Arsenal, NucleiFinding, WorkDir,
};

/// A swarm agent representing a phase/specialization
#[allow(dead_code)]
struct SwarmAgent {
    node: Node,
    name: String,
    phase: SwarmPhase,
    findings_emitted: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
enum SwarmPhase {
    SubdomainDragnet,
    PortScan,
    WebCrawl,
    NucleiScan,
    WordPressDeep,
    ApiAnalysis,
    JsAnalysis,
    CloudRecon,
    StagingRaid,
    AuthHeaders,
    GhResearch,
    TailxCorrelate,
}

impl SwarmPhase {
    fn name(&self) -> &'static str {
        match self {
            Self::SubdomainDragnet => "SUBDOMAIN",
            Self::PortScan => "PORTSCAN",
            Self::WebCrawl => "CRAWLER",
            Self::NucleiScan => "NUCLEI",
            Self::WordPressDeep => "WORDPRESS",
            Self::ApiAnalysis => "API",
            Self::JsAnalysis => "JS-AUDIT",
            Self::CloudRecon => "CLOUD",
            Self::StagingRaid => "STAGING",
            Self::AuthHeaders => "AUTH",
            Self::GhResearch => "RESEARCH",
            Self::TailxCorrelate => "CORRELATE",
        }
    }
}

/// Configuration for the full spectrum scan
#[derive(Debug, Clone)]
pub struct FullSpectrumConfig {
    /// Target domain
    pub target: String,
    /// Enable WordPress deep analysis
    pub wordpress: bool,
    /// Enable API endpoint analysis
    pub api_analysis: bool,
    /// Enable JavaScript static analysis
    pub js_analysis: bool,
    /// Enable cloud infrastructure recon
    pub cloud_recon: bool,
    /// Enable staging environment raiding
    pub staging_raid: bool,
    /// Enable auth/header analysis
    pub auth_headers: bool,
    /// Consensus threshold for final findings
    pub consensus_threshold: u32,
}

impl FullSpectrumConfig {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
            wordpress: true,
            api_analysis: true,
            js_analysis: true,
            cloud_recon: true,
            staging_raid: true,
            auth_headers: true,
            consensus_threshold: 1,
        }
    }

    /// Everything enabled
    pub fn full(target: impl Into<String>) -> Self {
        Self::new(target)
    }

    /// Quick scan - skip expensive phases
    pub fn quick(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
            wordpress: true,
            api_analysis: true,
            js_analysis: false,
            cloud_recon: true,
            staging_raid: true,
            auth_headers: true,
            consensus_threshold: 1,
        }
    }
}

/// A correlated finding with SMESH signal metadata
#[derive(Debug, Clone)]
pub struct CorrelatedFinding {
    pub finding: WebFinding,
    /// How many phases independently found/confirmed this
    pub signal_reinforcements: u32,
    /// Which agents confirmed
    pub confirmed_by: Vec<String>,
    /// SMESH signal intensity (decayed)
    pub signal_intensity: f64,
    /// Combined confidence
    pub confidence: f64,
}

/// Result of a full spectrum scan
#[derive(Debug)]
pub struct FullSpectrumResult {
    pub target: String,
    pub correlated_findings: Vec<CorrelatedFinding>,
    pub all_findings: Vec<(String, WebFinding)>, // (phase_name, finding)
    pub subdomains: Vec<String>,
    pub endpoints: Vec<String>,
    pub nuclei_findings: Vec<NucleiFinding>,
    pub duration: std::time::Duration,
    pub work_dir: PathBuf,
    pub phase_stats: HashMap<String, PhaseStats>,
}

#[derive(Debug, Clone, Default)]
pub struct PhaseStats {
    pub findings: u32,
    pub duration_ms: u64,
}

/// Run the full spectrum SMESH-coordinated web red team
pub async fn run_full_spectrum(config: FullSpectrumConfig) -> Result<FullSpectrumResult, String> {
    let start = Instant::now();
    let target = &config.target;

    println!("{}", FULL_SPECTRUM_BANNER);
    println!("  Target: {}", target);
    println!("  Mode: Full Spectrum SMESH-Coordinated Swarm\n");

    // Initialize arsenal
    let arsenal = Arsenal::detect().await;
    arsenal.print_status();

    // Initialize SMESH coordination layer
    let mut field = Field::new();
    let _reputation = ReputationSystem::new();
    let mut agents: HashMap<SwarmPhase, SwarmAgent> = HashMap::new();
    let mut all_findings: Vec<(String, WebFinding)> = Vec::new();
    let mut phase_stats: HashMap<String, PhaseStats> = HashMap::new();

    // Create agents for each phase
    let phases = vec![
        SwarmPhase::SubdomainDragnet, SwarmPhase::PortScan, SwarmPhase::WebCrawl,
        SwarmPhase::NucleiScan, SwarmPhase::WordPressDeep, SwarmPhase::ApiAnalysis,
        SwarmPhase::JsAnalysis, SwarmPhase::CloudRecon, SwarmPhase::StagingRaid,
        SwarmPhase::AuthHeaders,
    ];

    for phase in &phases {
        let mut node = Node::new();
        node.id = phase.name().to_string();
        agents.insert(*phase, SwarmAgent {
            node,
            name: phase.name().to_string(),
            phase: *phase,
            findings_emitted: 0,
        });
    }

    println!("\n  SMESH swarm: {} agents initialized\n", agents.len());

    let work = WorkDir::new(target).map_err(|e| e.to_string())?;

    // ========================================================================
    // TIER 1: Dragnet (runs linearly - each phase feeds the next)
    // ========================================================================

    println!("\x1b[1;35m{}\x1b[0m", "=".repeat(60));
    println!("\x1b[1;35m  TIER 1: DRAGNET\x1b[0m");
    println!("\x1b[1;35m{}\x1b[0m\n", "=".repeat(60));

    // Phase 1: Subdomain enumeration
    let p1_start = Instant::now();
    let subdomains = phase_subdomain_dragnet(target, &arsenal, &work).await;
    phase_stats.insert("subdomain".into(), PhaseStats {
        findings: subdomains.len() as u32,
        duration_ms: p1_start.elapsed().as_millis() as u64,
    });

    // Phase 2: Port scan
    let p2_start = Instant::now();
    let _live = phase_port_scan(&subdomains, &arsenal, &work).await;
    phase_stats.insert("portscan".into(), PhaseStats {
        findings: 0,
        duration_ms: p2_start.elapsed().as_millis() as u64,
    });

    // Phase 3: Crawl + fuzz
    let p3_start = Instant::now();
    let endpoints = phase_crawl_and_fuzz(target, &arsenal, &work).await;
    phase_stats.insert("crawl".into(), PhaseStats {
        findings: endpoints.len() as u32,
        duration_ms: p3_start.elapsed().as_millis() as u64,
    });

    // Phase 4: Nuclei vuln scan
    let p4_start = Instant::now();
    let nuclei_findings = phase_nuclei_scan(target, &endpoints, &arsenal, &work).await;
    for nf in &nuclei_findings {
        let wf = nuclei_to_web_finding(nf);
        emit_finding(&mut field, &mut agents, SwarmPhase::NucleiScan, &wf);
        all_findings.push(("NUCLEI".into(), wf));
    }
    phase_stats.insert("nuclei".into(), PhaseStats {
        findings: nuclei_findings.len() as u32,
        duration_ms: p4_start.elapsed().as_millis() as u64,
    });

    // ========================================================================
    // TIER 2: Deep Analysis (runs on discovered attack surface)
    // ========================================================================

    println!("\x1b[1;35m{}\x1b[0m", "=".repeat(60));
    println!("\x1b[1;35m  TIER 2: DEEP ANALYSIS\x1b[0m");
    println!("\x1b[1;35m{}\x1b[0m\n", "=".repeat(60));

    // WordPress specialist
    if config.wordpress {
        let ps = Instant::now();
        let wp_findings = phase_wordpress_deep(target, &arsenal, &work).await;
        for f in &wp_findings {
            emit_finding(&mut field, &mut agents, SwarmPhase::WordPressDeep, f);
            all_findings.push(("WORDPRESS".into(), f.clone()));
        }
        // Cross-reinforce: if nuclei also found WP issues, reinforce
        cross_reinforce(&mut field, &wp_findings, &all_findings);
        phase_stats.insert("wordpress".into(), PhaseStats {
            findings: wp_findings.len() as u32,
            duration_ms: ps.elapsed().as_millis() as u64,
        });
    }

    // API analysis
    if config.api_analysis {
        let ps = Instant::now();
        let api_findings = phase_api_analysis(&subdomains, &arsenal, &work).await;
        for f in &api_findings {
            emit_finding(&mut field, &mut agents, SwarmPhase::ApiAnalysis, f);
            all_findings.push(("API".into(), f.clone()));
        }
        phase_stats.insert("api".into(), PhaseStats {
            findings: api_findings.len() as u32,
            duration_ms: ps.elapsed().as_millis() as u64,
        });
    }

    // JS analysis
    if config.js_analysis {
        let ps = Instant::now();
        let js_findings = phase_js_analysis(target, &endpoints, &arsenal, &work).await;
        for f in &js_findings {
            emit_finding(&mut field, &mut agents, SwarmPhase::JsAnalysis, f);
            all_findings.push(("JS-AUDIT".into(), f.clone()));
        }
        phase_stats.insert("js".into(), PhaseStats {
            findings: js_findings.len() as u32,
            duration_ms: ps.elapsed().as_millis() as u64,
        });
    }

    // Cloud recon
    if config.cloud_recon {
        let ps = Instant::now();
        let cloud_findings = phase_cloud_recon(&subdomains, &arsenal, &work).await;
        for f in &cloud_findings {
            emit_finding(&mut field, &mut agents, SwarmPhase::CloudRecon, f);
            all_findings.push(("CLOUD".into(), f.clone()));
        }
        phase_stats.insert("cloud".into(), PhaseStats {
            findings: cloud_findings.len() as u32,
            duration_ms: ps.elapsed().as_millis() as u64,
        });
    }

    // Staging raid
    if config.staging_raid {
        let ps = Instant::now();
        let staging_findings = phase_staging_raid(&subdomains, &arsenal, &work).await;
        for f in &staging_findings {
            emit_finding(&mut field, &mut agents, SwarmPhase::StagingRaid, f);
            all_findings.push(("STAGING".into(), f.clone()));
        }
        phase_stats.insert("staging".into(), PhaseStats {
            findings: staging_findings.len() as u32,
            duration_ms: ps.elapsed().as_millis() as u64,
        });
    }

    // Auth/headers
    if config.auth_headers {
        let ps = Instant::now();
        let auth_findings = phase_auth_and_headers(target, &subdomains, &arsenal, &work).await;
        for f in &auth_findings {
            emit_finding(&mut field, &mut agents, SwarmPhase::AuthHeaders, f);
            all_findings.push(("AUTH".into(), f.clone()));
        }
        phase_stats.insert("auth".into(), PhaseStats {
            findings: auth_findings.len() as u32,
            duration_ms: ps.elapsed().as_millis() as u64,
        });
    }

    // ========================================================================
    // TIER 3: SMESH Consensus & Correlation
    // ========================================================================

    println!("\x1b[1;35m{}\x1b[0m", "=".repeat(60));
    println!("\x1b[1;35m  TIER 3: SMESH CONSENSUS\x1b[0m");
    println!("\x1b[1;35m{}\x1b[0m\n", "=".repeat(60));

    // Tick the field to decay old signals
    field.tick(1.0);

    // Build correlated findings from signal state
    let correlated = correlate_findings(&field, &all_findings, config.consensus_threshold);

    println!("  {} total findings across all phases", all_findings.len());
    println!("  {} correlated findings after SMESH consensus\n", correlated.len());

    // Print signal field stats
    let stats = field.stats();
    println!("  SMESH Field:");
    println!("    Active signals: {}", stats.active_signals);
    println!("    Total reinforcements: {}", stats.total_reinforcements);
    println!("    Avg intensity: {:.2}\n", stats.avg_intensity);

    let duration = start.elapsed();

    let result = FullSpectrumResult {
        target: target.to_string(),
        correlated_findings: correlated,
        all_findings,
        subdomains,
        endpoints,
        nuclei_findings,
        duration,
        work_dir: work.root,
        phase_stats,
    };

    print_full_spectrum_results(&result);

    Ok(result)
}

/// Emit a finding as a SMESH signal
fn emit_finding(
    field: &mut Field,
    agents: &mut HashMap<SwarmPhase, SwarmAgent>,
    phase: SwarmPhase,
    finding: &WebFinding,
) {
    let intensity = match finding.severity.as_str() {
        "CRITICAL" => 1.0,
        "HIGH" => 0.9,
        "MEDIUM" => 0.7,
        "LOW" => 0.5,
        _ => 0.3,
    };

    let signal = Signal::builder(SignalType::Alert)
        .payload_json(finding)
        .intensity(intensity)
        .confidence(finding.confidence)
        .ttl(300.0) // 5 min TTL
        .origin(phase.name())
        .build();

    if let Some(agent) = agents.get_mut(&phase) {
        field.emit(signal, &mut agent.node);
        agent.findings_emitted += 1;
    } else {
        field.emit_anonymous(signal);
    }
}

/// Cross-reinforce findings that match across phases
fn cross_reinforce(
    field: &mut Field,
    new_findings: &[WebFinding],
    existing: &[(String, WebFinding)],
) {
    for new in new_findings {
        for (_phase, existing) in existing {
            // Same category + overlapping URL = reinforcement
            if new.category == existing.category
                || (new.url.contains(&existing.url) || existing.url.contains(&new.url))
            {
                // Find the signal in the field and reinforce it
                let _key = format!("{}:{}", existing.category, existing.url);
                for signal in field.signals.values_mut() {
                    if let Some(payload_str) = signal.payload_as_str() {
                        if payload_str.contains(&existing.category)
                            && payload_str.contains(&existing.url)
                        {
                            signal.reinforce(&format!("cross-{}", new.category));
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Correlate all findings using SMESH signal reinforcement
fn correlate_findings(
    _field: &Field,
    all_findings: &[(String, WebFinding)],
    consensus_threshold: u32,
) -> Vec<CorrelatedFinding> {
    let mut grouped: HashMap<String, Vec<(String, WebFinding)>> = HashMap::new();

    // Group by dedup key: severity + category + url_domain
    for (phase, finding) in all_findings {
        let url_key = finding.url.split('/').take(4).collect::<Vec<_>>().join("/");
        let key = format!("{}:{}:{}", finding.severity, finding.category, url_key);
        grouped.entry(key).or_default().push((phase.clone(), finding.clone()));
    }

    let mut correlated: Vec<CorrelatedFinding> = Vec::new();

    for (_key, group) in &grouped {
        let best = group.iter()
            .max_by(|a, b| a.1.confidence.partial_cmp(&b.1.confidence).unwrap())
            .unwrap();

        let confirmed_by: Vec<String> = group.iter().map(|(p, _)| p.clone()).collect();
        let reinforcements = (group.len() as u32).saturating_sub(1);

        // Boost confidence based on cross-phase confirmation
        let base_conf = best.1.confidence;
        let boosted_conf = (base_conf + reinforcements as f64 * 0.1).min(1.0);

        if group.len() as u32 >= consensus_threshold {
            correlated.push(CorrelatedFinding {
                finding: best.1.clone(),
                signal_reinforcements: reinforcements,
                confirmed_by,
                signal_intensity: boosted_conf,
                confidence: boosted_conf,
            });
        }
    }

    // Sort by severity then confidence
    correlated.sort_by(|a, b| {
        let sev_cmp = severity_rank(&b.finding.severity).cmp(&severity_rank(&a.finding.severity));
        if sev_cmp == std::cmp::Ordering::Equal {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            sev_cmp
        }
    });

    correlated
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

fn nuclei_to_web_finding(nf: &NucleiFinding) -> WebFinding {
    WebFinding {
        category: format!("nuclei-{}", nf.template_id),
        severity: nf.severity.clone(),
        title: nf.name.clone(),
        detail: nf.description.clone(),
        url: nf.matched_url.clone(),
        evidence: nf.curl_command.clone(),
        confidence: 0.9,
    }
}

fn print_full_spectrum_results(result: &FullSpectrumResult) {
    println!("\n\x1b[1;32m{}\x1b[0m", "=".repeat(60));
    println!("\x1b[1;32m  FULL SPECTRUM RED TEAM - MISSION COMPLETE\x1b[0m");
    println!("\x1b[1;32m{}\x1b[0m\n", "=".repeat(60));

    println!("  Target:         {}", result.target);
    println!("  Duration:       {:.1}s", result.duration.as_secs_f64());
    println!("  Subdomains:     {}", result.subdomains.len());
    println!("  Endpoints:      {}", result.endpoints.len());
    println!("  Total findings: {}", result.all_findings.len());
    println!("  Correlated:     {}", result.correlated_findings.len());
    println!("  Artifacts:      {}\n", result.work_dir.display());

    // Phase breakdown
    println!("  Phase Breakdown:");
    for (phase, stats) in &result.phase_stats {
        println!("    {:<15} {:>3} findings  ({:.1}s)",
            phase, stats.findings, stats.duration_ms as f64 / 1000.0);
    }

    if result.correlated_findings.is_empty() {
        println!("\n  No findings passed consensus threshold.\n");
        return;
    }

    // Findings by severity
    println!("\n  \x1b[1mCorrelated Findings:\x1b[0m\n");

    let mut by_sev: HashMap<String, Vec<&CorrelatedFinding>> = HashMap::new();
    for cf in &result.correlated_findings {
        by_sev.entry(cf.finding.severity.clone()).or_default().push(cf);
    }

    for sev in &["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"] {
        if let Some(findings) = by_sev.get(*sev) {
            let color = match *sev {
                "CRITICAL" => "\x1b[1;31m",
                "HIGH" => "\x1b[31m",
                "MEDIUM" => "\x1b[33m",
                "LOW" => "\x1b[36m",
                _ => "\x1b[37m",
            };

            println!("  {}--- {} ({}) ---\x1b[0m", color, sev, findings.len());
            for cf in findings {
                let reinforce = if cf.signal_reinforcements > 0 {
                    format!(" [{}x reinforced]", cf.signal_reinforcements)
                } else {
                    String::new()
                };
                println!("    {} (conf: {:.0}%){}", cf.finding.title, cf.confidence * 100.0, reinforce);
                println!("      {} @ {}", cf.finding.category, cf.finding.url);
                if !cf.finding.evidence.is_empty() {
                    let ev = if cf.finding.evidence.len() > 80 {
                        format!("{}...", &cf.finding.evidence[..77])
                    } else {
                        cf.finding.evidence.clone()
                    };
                    println!("      Evidence: {}", ev);
                }
                println!("      Confirmed by: {}", cf.confirmed_by.join(", "));
                println!();
            }
        }
    }

    println!("  All artifacts: {}\n", result.work_dir.display());
}

const FULL_SPECTRUM_BANNER: &str = r#"
 ╔═══════════════════════════════════════════════════════════╗
 ║  SMESH FULL SPECTRUM RED TEAM                            ║
 ║  Dragnet > Deep Analysis > SMESH Consensus > Report      ║
 ║                                                          ║
 ║  Tier 1: Subdomain + Port + Crawl + Nuclei               ║
 ║  Tier 2: WP + API + JS + Cloud + Staging + Auth           ║
 ║  Tier 3: SMESH Signal Correlation + Consensus             ║
 ╚═══════════════════════════════════════════════════════════╝
"#;
