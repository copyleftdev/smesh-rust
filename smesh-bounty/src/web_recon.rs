//! Web Red Team module - elite-tier web target reconnaissance and attack
//!
//! Unlike source code scanning, this module targets live web applications
//! using the full arsenal of installed tools:
//!
//! - **Project Discovery**: nuclei, httpx, subfinder, naabu, katana, dnsx, tlsx, asnmap
//! - **Zentinel**: static analysis on any scraped/downloaded source
//! - **Semgrep**: pattern-based vulnerability detection
//! - **Tailx**: log cognition for correlating findings
//! - **ffuf**: directory/endpoint fuzzing
//! - **nmap**: port/service scanning
//! - **gh CLI**: CVE/exploit research and correlation
//!
//! Philosophy: dragnet first (map everything), then narrow (correlate and exploit).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tracing::{debug, warn};

/// Arsenal: every tool available on this system, categorized
#[derive(Debug, Clone)]
pub struct Arsenal {
    pub tools: HashMap<String, ToolInfo>,
}

#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub name: String,
    pub path: String,
    pub category: ToolCategory,
    pub available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolCategory {
    SubdomainEnum,
    PortScan,
    WebProbe,
    Crawl,
    VulnScan,
    DirFuzz,
    DnsRecon,
    TlsAudit,
    StaticAnalysis,
    LogAnalysis,
    Research,
}

impl Arsenal {
    /// Detect all available tools on the system
    pub async fn detect() -> Self {
        let checks = vec![
            ("subfinder", "subfinder", ToolCategory::SubdomainEnum),
            ("dnsx", "dnsx", ToolCategory::DnsRecon),
            ("asnmap", "asnmap", ToolCategory::SubdomainEnum),
            ("alterx", "alterx", ToolCategory::SubdomainEnum),
            ("shuffledns", "shuffledns", ToolCategory::DnsRecon),
            ("naabu", "naabu", ToolCategory::PortScan),
            ("nmap", "nmap", ToolCategory::PortScan),
            ("httpx", "httpx", ToolCategory::WebProbe),
            ("tlsx", "tlsx", ToolCategory::TlsAudit),
            ("cdncheck", "cdncheck", ToolCategory::WebProbe),
            ("katana", "katana", ToolCategory::Crawl),
            ("ffuf", "ffuf", ToolCategory::DirFuzz),
            ("nuclei", "nuclei", ToolCategory::VulnScan),
            ("semgrep", "semgrep", ToolCategory::StaticAnalysis),
            ("zent", "zent", ToolCategory::StaticAnalysis),
            ("tailx", "tailx", ToolCategory::LogAnalysis),
            ("gitleaks", "gitleaks", ToolCategory::StaticAnalysis),
            ("gh", "gh", ToolCategory::Research),
            ("curl", "curl", ToolCategory::WebProbe),
            ("webanalyze", "webanalyze", ToolCategory::WebProbe),
        ];

        let mut tools = HashMap::new();

        for (name, cmd, category) in checks {
            let available = tokio::process::Command::new("which")
                .arg(cmd)
                .output()
                .await
                .map(|o| o.status.success())
                .unwrap_or(false);

            let path = if available {
                tokio::process::Command::new("which")
                    .arg(cmd)
                    .output()
                    .await
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .unwrap_or_default()
                    .trim()
                    .to_string()
            } else {
                String::new()
            };

            tools.insert(
                name.to_string(),
                ToolInfo {
                    name: name.to_string(),
                    path,
                    category,
                    available,
                },
            );
        }

        Self { tools }
    }

    /// Print arsenal status
    pub fn print_status(&self) {
        println!("  Arsenal:");
        let mut by_cat: HashMap<ToolCategory, Vec<&ToolInfo>> = HashMap::new();
        for tool in self.tools.values() {
            by_cat.entry(tool.category).or_default().push(tool);
        }

        let categories = [
            (ToolCategory::SubdomainEnum, "Subdomain Enum"),
            (ToolCategory::DnsRecon, "DNS Recon"),
            (ToolCategory::PortScan, "Port Scanning"),
            (ToolCategory::WebProbe, "Web Probing"),
            (ToolCategory::Crawl, "Crawling"),
            (ToolCategory::DirFuzz, "Dir Fuzzing"),
            (ToolCategory::VulnScan, "Vuln Scanning"),
            (ToolCategory::TlsAudit, "TLS Audit"),
            (ToolCategory::StaticAnalysis, "Static Analysis"),
            (ToolCategory::LogAnalysis, "Log Analysis"),
            (ToolCategory::Research, "Research"),
        ];

        for (cat, label) in &categories {
            if let Some(tools) = by_cat.get(cat) {
                let status: Vec<String> = tools
                    .iter()
                    .map(|t| {
                        if t.available {
                            format!("\x1b[32m{}\x1b[0m", t.name)
                        } else {
                            format!("\x1b[31m{}\x1b[0m", t.name)
                        }
                    })
                    .collect();
                println!("    {}: {}", label, status.join(", "));
            }
        }
    }

    /// Check if a specific tool is available
    pub fn has(&self, name: &str) -> bool {
        self.tools
            .get(name)
            .map(|t| t.available)
            .unwrap_or(false)
    }
}

// ============================================================================
// Web Red Team Phases
// ============================================================================

/// Output directory for all scan artifacts
pub struct WorkDir {
    pub root: PathBuf,
}

impl WorkDir {
    pub fn new(target: &str) -> std::io::Result<Self> {
        let sanitized = target.replace("://", "_").replace(['/', '.', ':'], "_");
        let root = PathBuf::from(format!("/tmp/smesh-bounty-{}", sanitized));
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn file(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    pub fn path_str(&self, name: &str) -> String {
        self.file(name).to_string_lossy().to_string()
    }
}

/// Run a command and return stdout, logging stderr
pub async fn run_tool(
    cmd: &str,
    args: &[&str],
    timeout_secs: u64,
) -> Result<String, String> {
    debug!("Running: {} {}", cmd, args.join(" "));

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        tokio::process::Command::new(cmd).args(args).output(),
    )
    .await
    .map_err(|_| format!("Timeout after {}s: {} {}", timeout_secs, cmd, args.join(" ")))?
    .map_err(|e| format!("Failed to run {} {}: {}", cmd, args.join(" "), e))?;

    let stdout = String::from_utf8_lossy(&result.stdout).to_string();
    let stderr = String::from_utf8_lossy(&result.stderr).to_string();

    if !stderr.is_empty() && !result.status.success() {
        debug!("STDERR from {} {}: {}", cmd, args.join(" "), stderr.chars().take(200).collect::<String>());
    }

    Ok(stdout)
}

// ============================================================================
// Phase 1: Subdomain & DNS Dragnet
// ============================================================================

/// Enumerate all subdomains and DNS records
pub async fn phase_subdomain_dragnet(
    target: &str,
    arsenal: &Arsenal,
    work: &WorkDir,
) -> Vec<String> {
    println!("\n\x1b[1;35m--- Phase 1: Subdomain & DNS Dragnet ---\x1b[0m\n");

    let mut all_subs: Vec<String> = Vec::new();
    let subs_file = work.path_str("subdomains.txt");

    // subfinder: passive subdomain enumeration
    if arsenal.has("subfinder") {
        println!("  [subfinder] Passive subdomain enumeration...");
        match run_tool("subfinder", &["-d", target, "-silent"], 120).await {
            Ok(output) => {
                let count = output.lines().count();
                println!("    Found {} subdomains", count);
                all_subs.extend(output.lines().map(|l| l.trim().to_string()));
            }
            Err(e) => warn!("subfinder failed: {}", e),
        }
    }

    // alterx: generate permutations from discovered subdomains
    if arsenal.has("alterx") && !all_subs.is_empty() {
        println!("  [alterx] Generating subdomain permutations...");
        // Write subs to file for piping
        let _ = std::fs::write(&subs_file, all_subs.join("\n"));
        match run_tool(
            "bash",
            &["-c", &format!("cat {} | alterx -silent 2>/dev/null | head -500", subs_file)],
            60,
        )
        .await
        {
            Ok(output) => {
                let count = output.lines().count();
                println!("    Generated {} permutations", count);
                all_subs.extend(output.lines().map(|l| l.trim().to_string()));
            }
            Err(e) => debug!("alterx: {}", e),
        }
    }

    // dnsx: resolve and verify
    if arsenal.has("dnsx") && !all_subs.is_empty() {
        let _ = std::fs::write(&subs_file, all_subs.join("\n"));
        println!("  [dnsx] Resolving {} subdomains...", all_subs.len());
        let resolved_file = work.path_str("resolved.txt");
        match run_tool(
            "bash",
            &[
                "-c",
                &format!(
                    "cat {} | dnsx -silent -a -resp 2>/dev/null | tee {}",
                    subs_file, resolved_file
                ),
            ],
            120,
        )
        .await
        {
            Ok(output) => {
                let count = output.lines().count();
                println!("    {} subdomains resolved", count);
                // Replace with only resolved ones
                all_subs = output
                    .lines()
                    .filter_map(|l| l.split_whitespace().next())
                    .map(|s| s.to_string())
                    .collect();
            }
            Err(e) => warn!("dnsx failed: {}", e),
        }
    }

    // asnmap: ASN information
    if arsenal.has("asnmap") {
        println!("  [asnmap] ASN lookup for {}...", target);
        match run_tool("asnmap", &["-d", target, "-silent"], 30).await {
            Ok(output) => {
                if !output.trim().is_empty() {
                    println!("    ASN ranges:\n{}", indent(&output, 6));
                    let _ = std::fs::write(work.path_str("asn.txt"), &output);
                }
            }
            Err(e) => debug!("asnmap: {}", e),
        }
    }

    // Deduplicate
    all_subs.sort();
    all_subs.dedup();

    let _ = std::fs::write(&subs_file, all_subs.join("\n"));
    println!("\n  Total unique subdomains: {}\n", all_subs.len());

    all_subs
}

// ============================================================================
// Phase 2: Port & Service Discovery
// ============================================================================

/// Scan ports on all discovered hosts
pub async fn phase_port_scan(
    subdomains: &[String],
    arsenal: &Arsenal,
    work: &WorkDir,
) -> Vec<String> {
    println!("\x1b[1;35m--- Phase 2: Port & Service Discovery ---\x1b[0m\n");

    let hosts_file = work.path_str("hosts.txt");
    let _ = std::fs::write(&hosts_file, subdomains.join("\n"));
    let mut live_endpoints: Vec<String> = Vec::new();

    // naabu: fast port scanning
    if arsenal.has("naabu") {
        println!("  [naabu] Fast port scan on {} hosts...", subdomains.len());
        let ports_file = work.path_str("ports.txt");
        match run_tool(
            "bash",
            &[
                "-c",
                &format!(
                    "naabu -list {} -top-ports 1000 -silent 2>/dev/null | tee {}",
                    hosts_file, ports_file
                ),
            ],
            300,
        )
        .await
        {
            Ok(output) => {
                let count = output.lines().count();
                println!("    {} open port:host combinations", count);
                live_endpoints.extend(output.lines().map(|l| l.trim().to_string()));
            }
            Err(e) => warn!("naabu failed: {}", e),
        }
    }

    // httpx: probe for live HTTP services
    if arsenal.has("httpx") {
        println!("  [httpx] Probing HTTP services...");
        let httpx_file = work.path_str("httpx.json");
        match run_tool(
            "bash",
            &[
                "-c",
                &format!(
                    "cat {} | httpx -silent -json -title -tech-detect -status-code -follow-redirects 2>/dev/null | tee {}",
                    hosts_file, httpx_file
                ),
            ],
            180,
        )
        .await
        {
            Ok(output) => {
                for line in output.lines() {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                        let url = v["url"].as_str().unwrap_or("?");
                        let status = v["status_code"].as_i64().unwrap_or(0);
                        let title = v["title"].as_str().unwrap_or("");
                        let tech = v["tech"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            })
                            .unwrap_or_default();

                        println!(
                            "    [{:>3}] {} {} {}",
                            status,
                            url,
                            if title.is_empty() {
                                String::new()
                            } else {
                                format!("({})", title)
                            },
                            if tech.is_empty() {
                                String::new()
                            } else {
                                format!("[{}]", tech)
                            }
                        );
                    }
                }
            }
            Err(e) => warn!("httpx failed: {}", e),
        }
    }

    // tlsx: TLS certificate analysis
    if arsenal.has("tlsx") {
        println!("  [tlsx] TLS certificate analysis...");
        let tls_file = work.path_str("tls.json");
        match run_tool(
            "bash",
            &[
                "-c",
                &format!(
                    "cat {} | tlsx -silent -json -san -cn -so 2>/dev/null | tee {}",
                    hosts_file, tls_file
                ),
            ],
            120,
        )
        .await
        {
            Ok(output) => {
                let count = output.lines().count();
                println!("    {} TLS certificates analyzed", count);
            }
            Err(e) => debug!("tlsx: {}", e),
        }
    }

    println!();
    live_endpoints
}

// ============================================================================
// Phase 3: Crawl & Endpoint Discovery
// ============================================================================

/// Deep crawl and directory fuzzing
pub async fn phase_crawl_and_fuzz(
    target: &str,
    arsenal: &Arsenal,
    work: &WorkDir,
) -> Vec<String> {
    println!("\x1b[1;35m--- Phase 3: Crawl & Endpoint Discovery ---\x1b[0m\n");

    let mut endpoints: Vec<String> = Vec::new();

    // katana: headless crawling
    if arsenal.has("katana") {
        println!("  [katana] Deep crawling https://{}...", target);
        let katana_file = work.path_str("katana.txt");
        match run_tool(
            "katana",
            &[
                "-u", &format!("https://{}", target),
                "-d", "3",
                "-jc",       // JS crawl
                "-kf", "all", // known files
                "-ef", "css,png,jpg,gif,svg,woff,woff2,ttf,eot,ico",
                "-silent",
                "-o", &katana_file,
            ],
            180,
        )
        .await
        {
            Ok(_) => {
                let crawled = std::fs::read_to_string(&katana_file)
                    .unwrap_or_default();
                let count = crawled.lines().count();
                println!("    {} URLs discovered", count);
                endpoints.extend(crawled.lines().map(|l| l.trim().to_string()));

                // Show interesting findings
                for line in crawled.lines() {
                    let lower = line.to_lowercase();
                    if lower.contains("api")
                        || lower.contains("admin")
                        || lower.contains("login")
                        || lower.contains("graphql")
                        || lower.contains("swagger")
                        || lower.contains(".json")
                        || lower.contains(".xml")
                        || lower.contains("wp-")
                    {
                        println!("    \x1b[33m>> {}\x1b[0m", line);
                    }
                }
            }
            Err(e) => warn!("katana failed: {}", e),
        }
    }

    // ffuf: directory fuzzing on main target
    if arsenal.has("ffuf") {
        println!("  [ffuf] Directory fuzzing https://{}...", target);
        let wordlist = if Path::new("/usr/share/seclists/Discovery/Web-Content/common.txt").exists()
        {
            "/usr/share/seclists/Discovery/Web-Content/common.txt"
        } else if Path::new("/usr/share/wordlists/dirb/common.txt").exists() {
            "/usr/share/wordlists/dirb/common.txt"
        } else {
            "" // Skip if no wordlist
        };

        if !wordlist.is_empty() {
            let ffuf_file = work.path_str("ffuf.json");
            match run_tool(
                "ffuf",
                &[
                    "-u", &format!("https://{}/FUZZ", target),
                    "-w", wordlist,
                    "-mc", "200,201,301,302,401,403,405,500",
                    "-t", "20",
                    "-timeout", "10",
                    "-o", &ffuf_file,
                    "-of", "json",
                    "-s",
                ],
                120,
            )
            .await
            {
                Ok(_) => {
                    if let Ok(content) = std::fs::read_to_string(&ffuf_file) {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                            if let Some(results) = v["results"].as_array() {
                                println!("    {} paths found", results.len());
                                for r in results.iter().take(30) {
                                    let url = r["url"].as_str().unwrap_or("?");
                                    let status = r["status"].as_i64().unwrap_or(0);
                                    let length = r["length"].as_i64().unwrap_or(0);
                                    println!("    [{:>3}] {} ({}b)", status, url, length);
                                    endpoints.push(url.to_string());
                                }
                            }
                        }
                    }
                }
                Err(e) => debug!("ffuf: {}", e),
            }
        } else {
            println!("    Skipped: no wordlist found (install seclists)");
        }
    }

    endpoints.sort();
    endpoints.dedup();
    let _ = std::fs::write(work.path_str("endpoints.txt"), endpoints.join("\n"));
    println!("\n  Total unique endpoints: {}\n", endpoints.len());

    endpoints
}

// ============================================================================
// Phase 4: Nuclei Vulnerability Scanning
// ============================================================================

/// Run nuclei templates against discovered targets
pub async fn phase_nuclei_scan(
    target: &str,
    endpoints: &[String],
    arsenal: &Arsenal,
    work: &WorkDir,
) -> Vec<NucleiFinding> {
    println!("\x1b[1;35m--- Phase 4: Nuclei Vulnerability Scan ---\x1b[0m\n");

    let mut findings: Vec<NucleiFinding> = Vec::new();

    if !arsenal.has("nuclei") {
        println!("  nuclei not available, skipping");
        return findings;
    }

    // Write all endpoints to file for nuclei
    let targets_file = work.path_str("nuclei-targets.txt");
    let mut all_targets = vec![format!("https://{}", target)];
    all_targets.extend(endpoints.iter().cloned());
    all_targets.sort();
    all_targets.dedup();
    let _ = std::fs::write(&targets_file, all_targets.join("\n"));

    let nuclei_file = work.path_str("nuclei.json");

    // Run nuclei with critical/high/medium templates
    println!(
        "  [nuclei] Scanning {} targets with vulnerability templates...",
        all_targets.len().min(200)
    );

    let template_tags = "cve,wordpress,wp-plugin,exposure,misconfig,tech,token,xss,sqli,ssrf,lfi,rce,default-login,takeover";

    match run_tool(
        "nuclei",
        &[
            "-list", &targets_file,
            "-tags", template_tags,
            "-severity", "critical,high,medium",
            "-json-export", &nuclei_file,
            "-silent",
            "-rate-limit", "50",
            "-bulk-size", "25",
            "-concurrency", "10",
            "-timeout", "10",
            "-retries", "1",
            "-no-update-templates",
        ],
        600,
    )
    .await
    {
        Ok(output) => {
            // Parse nuclei JSON output
            if let Ok(content) = std::fs::read_to_string(&nuclei_file) {
                for line in content.lines() {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                        let finding = NucleiFinding {
                            template_id: v["template-id"]
                                .as_str()
                                .unwrap_or("unknown")
                                .to_string(),
                            name: v["info"]["name"]
                                .as_str()
                                .unwrap_or("Unknown")
                                .to_string(),
                            severity: v["info"]["severity"]
                                .as_str()
                                .unwrap_or("info")
                                .to_uppercase(),
                            matched_url: v["matched-at"]
                                .as_str()
                                .or_else(|| v["host"].as_str())
                                .unwrap_or("?")
                                .to_string(),
                            description: v["info"]["description"]
                                .as_str()
                                .unwrap_or("")
                                .to_string(),
                            tags: v["info"]["tags"]
                                .as_array()
                                .map(|a| {
                                    a.iter()
                                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                        .collect()
                                })
                                .unwrap_or_default(),
                            curl_command: v["curl-command"]
                                .as_str()
                                .unwrap_or("")
                                .to_string(),
                        };

                        let sev_color = match finding.severity.as_str() {
                            "CRITICAL" => "\x1b[1;31m",
                            "HIGH" => "\x1b[31m",
                            "MEDIUM" => "\x1b[33m",
                            _ => "\x1b[36m",
                        };

                        println!(
                            "  {}[{}]\x1b[0m {} - {} @ {}",
                            sev_color,
                            finding.severity,
                            finding.template_id,
                            finding.name,
                            finding.matched_url,
                        );

                        findings.push(finding);
                    }
                }
            }

            if findings.is_empty() && !output.trim().is_empty() {
                // Sometimes nuclei outputs to stdout in non-json
                for line in output.lines().take(20) {
                    if !line.trim().is_empty() {
                        println!("  {}", line);
                    }
                }
            }
        }
        Err(e) => warn!("nuclei failed: {}", e),
    }

    println!("\n  {} vulnerabilities found\n", findings.len());
    findings
}

/// A finding from nuclei
#[derive(Debug, Clone)]
pub struct NucleiFinding {
    pub template_id: String,
    pub name: String,
    pub severity: String,
    pub matched_url: String,
    pub description: String,
    pub tags: Vec<String>,
    pub curl_command: String,
}

// ============================================================================
// Phase 5: GH CLI Research & Correlation
// ============================================================================

/// Use gh CLI to research CVEs, exploits, and correlate findings
pub async fn phase_gh_research(
    target: &str,
    findings: &[NucleiFinding],
    arsenal: &Arsenal,
    work: &WorkDir,
) -> Vec<String> {
    println!("\x1b[1;35m--- Phase 5: Research & Correlation (gh CLI) ---\x1b[0m\n");

    let mut intel: Vec<String> = Vec::new();

    if !arsenal.has("gh") {
        println!("  gh CLI not available, skipping");
        return intel;
    }

    // Search for exploits/PoCs related to technologies found
    let search_terms = vec![
        format!("WordPress 6.9 exploit"),
        format!("Elementor Pro vulnerability"),
        format!("{} CVE", target),
        format!("Angular SPA vulnerability 2025 2026"),
    ];

    for term in &search_terms {
        println!("  [gh] Searching: \"{}\"...", term);
        match run_tool(
            "gh",
            &[
                "search", "repos",
                "--match", "name,description",
                &term,
                "--limit", "5",
                "--json", "name,description,url,stargazersCount",
            ],
            30,
        )
        .await
        {
            Ok(output) => {
                if let Ok(repos) = serde_json::from_str::<Vec<serde_json::Value>>(&output) {
                    for repo in repos.iter().take(3) {
                        let name = repo["name"].as_str().unwrap_or("?");
                        let desc = repo["description"].as_str().unwrap_or("");
                        let url = repo["url"].as_str().unwrap_or("");
                        let stars = repo["stargazersCount"].as_i64().unwrap_or(0);
                        if stars > 0 {
                            println!("    {} ({} stars): {}", name, stars, url);
                            intel.push(format!("{}: {} - {}", name, desc, url));
                        }
                    }
                }
            }
            Err(e) => debug!("gh search failed: {}", e),
        }
    }

    // Search for CVEs from nuclei findings
    for finding in findings.iter().take(5) {
        if finding.template_id.starts_with("CVE-") || finding.template_id.contains("cve") {
            println!("  [gh] Researching {}...", finding.template_id);
            match run_tool(
                "gh",
                &[
                    "search", "repos",
                    &finding.template_id,
                    "--limit", "3",
                    "--json", "name,url,stargazersCount",
                ],
                15,
            )
            .await
            {
                Ok(output) => {
                    if let Ok(repos) = serde_json::from_str::<Vec<serde_json::Value>>(&output) {
                        for repo in &repos {
                            let url = repo["url"].as_str().unwrap_or("");
                            let stars = repo["stargazersCount"].as_i64().unwrap_or(0);
                            println!("    PoC: {} ({} stars)", url, stars);
                            intel.push(format!(
                                "PoC for {}: {}",
                                finding.template_id, url
                            ));
                        }
                    }
                }
                Err(_) => {}
            }
        }
    }

    let _ = std::fs::write(work.path_str("intel.txt"), intel.join("\n"));
    println!("\n  {} intelligence items gathered\n", intel.len());

    intel
}

// ============================================================================
// Phase 6: Tailx Correlation
// ============================================================================

/// Use tailx to correlate all scan outputs for patterns
pub async fn phase_tailx_correlate(
    arsenal: &Arsenal,
    work: &WorkDir,
) {
    println!("\x1b[1;35m--- Phase 6: Tailx Correlation ---\x1b[0m\n");

    if !arsenal.has("tailx") {
        println!("  tailx not available, skipping");
        return;
    }

    // Concatenate all scan output files
    let log_files: Vec<String> = ["nuclei.json", "httpx.json", "tls.json"]
        .iter()
        .map(|f| work.path_str(f))
        .filter(|p| Path::new(p).exists())
        .collect();

    if log_files.is_empty() {
        println!("  No scan logs to correlate");
        return;
    }

    println!("  [tailx] Correlating {} scan outputs...", log_files.len());

    let mut args = vec!["-s", "-n", "--json"];
    let file_refs: Vec<&str> = log_files.iter().map(|s| s.as_str()).collect();
    args.extend(file_refs);

    match run_tool("tailx", &args, 30).await {
        Ok(output) => {
            // Get the last line which is the triage summary
            if let Some(last_line) = output.lines().last() {
                if let Ok(triage) = serde_json::from_str::<serde_json::Value>(last_line) {
                    if let Some(groups) = triage["top_groups"].as_array() {
                        println!("  Patterns identified: {}", groups.len());
                        for g in groups.iter().take(5) {
                            let exemplar = g["exemplar"].as_str().unwrap_or("?");
                            let count = g["count"].as_i64().unwrap_or(0);
                            println!("    [{:>4}x] {}", count, exemplar);
                        }
                    }
                    if let Some(hypotheses) = triage["hypotheses"].as_array() {
                        if !hypotheses.is_empty() {
                            println!("  Correlation hypotheses:");
                            for h in hypotheses.iter().take(3) {
                                let conf = h["confidence"].as_f64().unwrap_or(0.0);
                                if let Some(causes) = h["causes"].as_array() {
                                    for c in causes {
                                        let label = c["label"].as_str().unwrap_or("?");
                                        println!("    [{:.0}%] {}", conf * 100.0, label);
                                    }
                                }
                            }
                        }
                    }

                    let _ = std::fs::write(
                        work.path_str("correlation.json"),
                        serde_json::to_string_pretty(&triage).unwrap_or_default(),
                    );
                }
            }
        }
        Err(e) => debug!("tailx failed: {}", e),
    }

    println!();
}

// ============================================================================
// Master Orchestrator
// ============================================================================

/// Run the full red team mission against a web target
pub async fn run_web_redteam(target: &str) -> Result<WebRedTeamResult, String> {
    let start = Instant::now();

    println!("{}", WEB_BANNER);
    println!("  Target: {}", target);

    // Detect arsenal
    let arsenal = Arsenal::detect().await;
    arsenal.print_status();

    // Create work directory
    let work = WorkDir::new(target).map_err(|e| e.to_string())?;
    println!("  Work dir: {}\n", work.root.display());

    // Phase 1: Subdomain dragnet
    let subdomains = phase_subdomain_dragnet(target, &arsenal, &work).await;

    // Phase 2: Port & service discovery
    let _live_endpoints = phase_port_scan(&subdomains, &arsenal, &work).await;

    // Phase 3: Crawl & fuzz
    let endpoints = phase_crawl_and_fuzz(target, &arsenal, &work).await;

    // Phase 4: Nuclei vuln scan
    let nuclei_findings = phase_nuclei_scan(target, &endpoints, &arsenal, &work).await;

    // Phase 5: GH research & correlation
    let intel = phase_gh_research(target, &nuclei_findings, &arsenal, &work).await;

    // Phase 6: Tailx correlation
    phase_tailx_correlate(&arsenal, &work).await;

    let duration = start.elapsed();

    let result = WebRedTeamResult {
        target: target.to_string(),
        subdomains_found: subdomains.len(),
        endpoints_found: endpoints.len(),
        nuclei_findings,
        intel_items: intel,
        duration,
        work_dir: work.root,
    };

    // Print summary
    print_web_results(&result);

    Ok(result)
}

/// Results of a web red team mission
#[derive(Debug)]
pub struct WebRedTeamResult {
    pub target: String,
    pub subdomains_found: usize,
    pub endpoints_found: usize,
    pub nuclei_findings: Vec<NucleiFinding>,
    pub intel_items: Vec<String>,
    pub duration: std::time::Duration,
    pub work_dir: PathBuf,
}

fn print_web_results(result: &WebRedTeamResult) {
    println!("\n\x1b[1;32m====================================\x1b[0m");
    println!("\x1b[1;32m  RED TEAM MISSION COMPLETE\x1b[0m");
    println!("\x1b[1;32m====================================\x1b[0m\n");
    println!("  Target:       {}", result.target);
    println!("  Duration:     {:.1}s", result.duration.as_secs_f64());
    println!("  Subdomains:   {}", result.subdomains_found);
    println!("  Endpoints:    {}", result.endpoints_found);
    println!("  Vulns found:  {}", result.nuclei_findings.len());
    println!("  Intel items:  {}", result.intel_items.len());
    println!("  Artifacts:    {}", result.work_dir.display());

    if !result.nuclei_findings.is_empty() {
        println!("\n  \x1b[1mTop Findings:\x1b[0m");
        for f in result.nuclei_findings.iter().take(10) {
            let sev_color = match f.severity.as_str() {
                "CRITICAL" => "\x1b[1;31m",
                "HIGH" => "\x1b[31m",
                "MEDIUM" => "\x1b[33m",
                _ => "\x1b[36m",
            };
            println!(
                "    {}[{}]\x1b[0m {} @ {}",
                sev_color, f.severity, f.name, f.matched_url
            );
        }
    }

    println!("\n  All artifacts saved to: {}\n", result.work_dir.display());
}

fn indent(s: &str, n: usize) -> String {
    let pad: String = " ".repeat(n);
    s.lines()
        .map(|l| format!("{}{}", pad, l))
        .collect::<Vec<_>>()
        .join("\n")
}

const WEB_BANNER: &str = r#"
 ╔══════════════════════════════════════════════════╗
 ║  SMESH RED TEAM - Web Target Mission             ║
 ║  Dragnet > Enumerate > Scan > Correlate > Report ║
 ╚══════════════════════════════════════════════════╝
"#;
