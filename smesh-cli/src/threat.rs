use anyhow::Result;
use smesh_agent::{OllamaClient, OllamaConfig};
use smesh_core::{Field, Node, Signal, SignalType};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokio::time::{sleep, Duration};

/// Threat categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThreatCategory {
    Injection,        // SQLi, Command Injection, LDAP, XPath
    XSS,              // Cross-Site Scripting variants
    Authentication,   // Auth bypass, session hijacking
    Traversal,        // Path traversal, LFI, RFI
    Deserialization,  // Insecure deserialization
    SSRF,             // Server-side request forgery
    XXE,              // XML External Entity
    Cryptographic,    // Crypto weaknesses
    Misconfiguration, // CORS, headers, etc.
    Other,            // Uncategorized
}

impl ThreatCategory {
    fn from_path(path: &str) -> Self {
        let lower = path.to_lowercase();
        
        if lower.contains("injection") || lower.contains("sqli") {
            ThreatCategory::Injection
        } else if lower.contains("xss") || lower.contains("cross-site scripting") {
            ThreatCategory::XSS
        } else if lower.contains("traversal") || lower.contains("inclusion") || lower.contains("lfi") || lower.contains("rfi") {
            ThreatCategory::Traversal
        } else if lower.contains("auth") || lower.contains("session") || lower.contains("takeover") {
            ThreatCategory::Authentication
        } else if lower.contains("deseriali") {
            ThreatCategory::Deserialization
        } else if lower.contains("ssrf") || lower.contains("request forgery") {
            ThreatCategory::SSRF
        } else if lower.contains("xxe") || lower.contains("xml") {
            ThreatCategory::XXE
        } else if lower.contains("crypto") || lower.contains("jwt") {
            ThreatCategory::Cryptographic
        } else if lower.contains("cors") || lower.contains("config") || lower.contains("header") {
            ThreatCategory::Misconfiguration
        } else {
            ThreatCategory::Other
        }
    }

    fn severity(&self) -> &'static str {
        match self {
            ThreatCategory::Injection => "CRITICAL",
            ThreatCategory::Deserialization => "CRITICAL",
            ThreatCategory::Traversal => "HIGH",
            ThreatCategory::SSRF => "HIGH",
            ThreatCategory::XXE => "HIGH",
            ThreatCategory::XSS => "MEDIUM",
            ThreatCategory::Authentication => "HIGH",
            ThreatCategory::Cryptographic => "MEDIUM",
            ThreatCategory::Misconfiguration => "LOW",
            ThreatCategory::Other => "INFO",
        }
    }

    fn color(&self) -> &'static str {
        match self {
            ThreatCategory::Injection | ThreatCategory::Deserialization => "\x1b[91m", // Red
            ThreatCategory::Traversal | ThreatCategory::SSRF | ThreatCategory::XXE | ThreatCategory::Authentication => "\x1b[93m", // Yellow
            ThreatCategory::XSS | ThreatCategory::Cryptographic => "\x1b[94m", // Blue
            _ => "\x1b[90m", // Gray
        }
    }
}

/// A threat pattern identified in the payloads
#[derive(Debug, Clone)]
pub struct ThreatPattern {
    pub category: ThreatCategory,
    pub source_file: String,
    pub pattern_name: String,
    pub description: String,
    pub example_payloads: Vec<String>,
    pub mitigations: Vec<String>,
    pub confidence: f32,
    pub reinforcements: u32,
}

/// Threat analyzer agent types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyzerType {
    PatternExtractor,   // Extracts payload patterns
    RiskAssessor,       // Assesses risk level
    MitigationAdvisor,  // Suggests mitigations
    CorrelationFinder,  // Finds related patterns
}

impl AnalyzerType {
    fn system_prompt(&self) -> &'static str {
        match self {
            AnalyzerType::PatternExtractor => {
                "You are a security payload pattern extractor. Given a markdown file about a security vulnerability:
1. Identify the main attack technique
2. Extract 3-5 representative payload examples (just the payloads, one per line)
3. Note any variations or bypass techniques

Format your response as:
TECHNIQUE: [name]
PAYLOADS:
- [payload1]
- [payload2]
- [payload3]
VARIATIONS: [brief note on variations]

Be concise. Focus on the most impactful patterns."
            }
            AnalyzerType::RiskAssessor => {
                "You are a security risk assessor. Given information about a vulnerability type:
1. Rate the severity (CRITICAL/HIGH/MEDIUM/LOW)
2. Explain the potential impact in one sentence
3. List affected contexts (web, API, mobile, etc.)

Format:
SEVERITY: [level]
IMPACT: [one sentence]
CONTEXTS: [comma-separated list]

Be concise and accurate."
            }
            AnalyzerType::MitigationAdvisor => {
                "You are a security mitigation advisor. Given a vulnerability type:
1. List 3 specific mitigation strategies
2. Include code-level and architecture-level mitigations

Format:
MITIGATIONS:
1. [mitigation with brief explanation]
2. [mitigation with brief explanation]
3. [mitigation with brief explanation]

Be specific and actionable."
            }
            AnalyzerType::CorrelationFinder => {
                "You are a threat correlation analyst. Given a vulnerability type:
1. Identify related attack chains
2. Note common combinations with other vulnerabilities
3. Identify reconnaissance indicators

Format:
RELATED_ATTACKS: [comma-separated]
CHAINS_WITH: [what it commonly chains with]
RECON_SIGNS: [indicators of this attack being attempted]

Be concise."
            }
        }
    }
}

/// Run SMESH-coordinated threat analysis
pub async fn run_threat_analysis(repo_path: &Path, model: &str, limit: usize) -> Result<Vec<ThreatPattern>> {
    println!("\n\x1b[91m🔥 SMESH Threat Intelligence - Signal Diffusion Mode\x1b[0m\n");
    println!("Target: {}", repo_path.display());
    println!("Model: {}", model);
    println!("File limit: {}\n", limit);

    // Initialize SMESH primitives
    let mut field = Field::new();

    // Create analyzer nodes
    let analyzers = vec![
        AnalyzerType::PatternExtractor,
        AnalyzerType::RiskAssessor,
        AnalyzerType::MitigationAdvisor,
    ];

    let mut nodes: HashMap<AnalyzerType, Node> = HashMap::new();
    for analyzer in &analyzers {
        let node = Node::new();
        nodes.insert(*analyzer, node);
    }

    println!("📡 Analyzer nodes: {}", nodes.len());

    // Collect markdown files (threat documentation)
    let md_files = collect_threat_files(repo_path, limit)?;
    println!("📂 Found {} threat documents to analyze\n", md_files.len());

    // Initialize Ollama client
    let mut config = OllamaConfig::default();
    config.model = model.to_string();
    config.max_tokens = 1024;
    let client = OllamaClient::new(config);

    // Check Ollama connection
    if !client.is_available().await {
        anyhow::bail!("Ollama not available. Run: ollama serve");
    }

    let mut all_patterns: Vec<ThreatPattern> = Vec::new();
    let mut category_counts: HashMap<ThreatCategory, usize> = HashMap::new();

    // Process each threat file
    for (idx, file_path) in md_files.iter().enumerate() {
        let relative_path = file_path
            .strip_prefix(repo_path)
            .unwrap_or(file_path)
            .display()
            .to_string();

        // Determine category from path
        let category = ThreatCategory::from_path(&relative_path);
        *category_counts.entry(category).or_insert(0) += 1;

        println!(
            "{}─────────────────────────────────────────────────────────\x1b[0m",
            category.color()
        );
        println!(
            "{}📄 [{}/{}] {} [{}]\x1b[0m",
            category.color(),
            idx + 1,
            md_files.len(),
            relative_path,
            category.severity()
        );

        let content = fs::read_to_string(file_path)?;
        
        // Skip very small files
        if content.lines().count() < 10 {
            println!("   ⏭️  Skipping (< 10 lines)\n");
            continue;
        }

        // Emit THREAT_DOC signal
        let doc_signal = Signal::builder(SignalType::Alert)
            .payload(relative_path.as_bytes().to_vec())
            .intensity(1.0)
            .confidence(0.9)
            .build();
        
        field.emit_anonymous(doc_signal);

        // Pattern Extractor analyzes the file
        let extractor_prompt = format!(
            "{}\n\n---\nAnalyze this threat documentation:\n\n{}",
            AnalyzerType::PatternExtractor.system_prompt(),
            truncate_content(&content, 3000)
        );

        print!("   🔍 Extracting patterns... ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let extraction = match client.generate(&extractor_prompt, None).await {
            Ok(resp) => {
                println!("✓");
                resp
            }
            Err(e) => {
                println!("✗ ({})", e);
                continue;
            }
        };

        // Emit pattern signal
        let pattern_signal = Signal::builder(SignalType::Data)
            .payload(extraction.as_bytes().to_vec())
            .intensity(0.9)
            .confidence(0.8)
            .build();
        field.emit_anonymous(pattern_signal);

        // Mitigation Advisor provides countermeasures
        let mitigation_prompt = format!(
            "{}\n\n---\nProvide mitigations for: {:?}\nContext from extraction:\n{}",
            AnalyzerType::MitigationAdvisor.system_prompt(),
            category,
            truncate_content(&extraction, 500)
        );

        print!("   🛡️  Generating mitigations... ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let mitigations = match client.generate(&mitigation_prompt, None).await {
            Ok(resp) => {
                println!("✓");
                resp
            }
            Err(e) => {
                println!("✗ ({})", e);
                String::new()
            }
        };

        // Parse payloads from extraction
        let payloads = extract_payloads(&extraction);
        let mitigation_list = extract_mitigations(&mitigations);

        // Calculate reinforcement based on severity
        let reinforcements = match category.severity() {
            "CRITICAL" => 3,
            "HIGH" => 2,
            "MEDIUM" => 1,
            _ => 0,
        };

        let pattern = ThreatPattern {
            category,
            source_file: relative_path.clone(),
            pattern_name: extract_technique(&extraction),
            description: extraction.lines().take(3).collect::<Vec<_>>().join(" "),
            example_payloads: payloads,
            mitigations: mitigation_list,
            confidence: 0.7 + (reinforcements as f32 * 0.1),
            reinforcements,
        };

        println!("      └─ Pattern: {} (confidence: {:.0}%)", 
            pattern.pattern_name, 
            pattern.confidence * 100.0
        );

        all_patterns.push(pattern);

        // Field tick
        field.tick(1.0);

        // Rate limiting
        sleep(Duration::from_millis(200)).await;
        println!();
    }

    // Print threat intelligence summary
    print_threat_summary(&all_patterns, &category_counts, &field);

    Ok(all_patterns)
}

fn collect_threat_files(dir: &Path, limit: usize) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    collect_md_recursive(dir, &mut files)?;
    
    // Sort by path for consistent ordering
    files.sort();
    
    // Apply limit
    files.truncate(limit);
    
    Ok(files)
}

fn collect_md_recursive(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            // Skip hidden dirs and specific folders
            if !name.starts_with('.') && name != "_template" {
                collect_md_recursive(&path, files)?;
            }
        } else if path.extension().map_or(false, |e| e == "md") {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            // Skip README, CONTRIBUTING, etc.
            if !name.starts_with("README") && !name.starts_with("CONTRIB") && !name.starts_with("DISCLAIM") {
                files.push(path);
            }
        }
    }

    Ok(())
}

fn truncate_content(content: &str, max_chars: usize) -> &str {
    if content.len() <= max_chars {
        content
    } else {
        &content[..max_chars]
    }
}

fn extract_payloads(extraction: &str) -> Vec<String> {
    let mut payloads = Vec::new();
    let mut in_payloads = false;

    for line in extraction.lines() {
        if line.contains("PAYLOADS:") || line.contains("PAYLOAD:") {
            in_payloads = true;
            continue;
        }
        if in_payloads {
            if line.starts_with("- ") || line.starts_with("* ") {
                let payload = line.trim_start_matches("- ").trim_start_matches("* ").trim();
                if !payload.is_empty() && payload.len() < 200 {
                    payloads.push(payload.to_string());
                }
            } else if line.starts_with("VARIATIONS") || line.starts_with("TECHNIQUE") || line.is_empty() {
                if !payloads.is_empty() {
                    break;
                }
            }
        }
    }

    // Limit to 5 payloads
    payloads.truncate(5);
    payloads
}

fn extract_mitigations(response: &str) -> Vec<String> {
    let mut mitigations = Vec::new();

    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("1.") || trimmed.starts_with("2.") || trimmed.starts_with("3.") {
            let mitigation = trimmed[2..].trim().to_string();
            if !mitigation.is_empty() {
                mitigations.push(mitigation);
            }
        }
    }

    mitigations
}

fn extract_technique(extraction: &str) -> String {
    for line in extraction.lines() {
        if line.starts_with("TECHNIQUE:") {
            return line.replace("TECHNIQUE:", "").trim().to_string();
        }
    }
    "Unknown".to_string()
}

fn print_threat_summary(
    patterns: &[ThreatPattern],
    category_counts: &HashMap<ThreatCategory, usize>,
    field: &Field,
) {
    let reset = "\x1b[0m";
    let red = "\x1b[91m";
    let yellow = "\x1b[93m";
    let green = "\x1b[92m";
    let cyan = "\x1b[96m";

    println!("\n{}═══════════════════════════════════════════════════════════════{}", red, reset);
    println!("{}                    THREAT INTELLIGENCE SUMMARY                 {}", red, reset);
    println!("{}═══════════════════════════════════════════════════════════════{}\n", red, reset);

    // Category breakdown
    println!("{}📊 Threat Categories:{}", cyan, reset);
    let mut categories: Vec<_> = category_counts.iter().collect();
    categories.sort_by(|a, b| b.1.cmp(a.1));
    
    for (cat, count) in categories {
        let color = cat.color();
        println!("   {} {:?}: {} files [{}]{}", color, cat, count, cat.severity(), reset);
    }

    // Signal field stats
    let stats = field.stats();
    println!("\n{}📡 Signal Field:{}", cyan, reset);
    println!("   Active signals: {}", stats.active_signals);
    println!("   Total reinforcements: {}", stats.total_reinforcements);

    // Critical patterns
    let critical: Vec<_> = patterns.iter()
        .filter(|p| p.category.severity() == "CRITICAL")
        .collect();
    
    if !critical.is_empty() {
        println!("\n{}🚨 CRITICAL Patterns ({}):{}", red, critical.len(), reset);
        for p in critical.iter().take(5) {
            println!("   {} • {} ({})", red, p.pattern_name, p.source_file);
            if !p.example_payloads.is_empty() {
                println!("     {}Example: {}{}", yellow, p.example_payloads[0], reset);
            }
        }
    }

    // Top mitigations
    let mut all_mitigations: Vec<&String> = patterns.iter()
        .flat_map(|p| p.mitigations.iter())
        .collect();
    all_mitigations.truncate(10);

    if !all_mitigations.is_empty() {
        println!("\n{}🛡️  Top Mitigations:{}", green, reset);
        for (i, m) in all_mitigations.iter().take(5).enumerate() {
            let truncated: String = m.chars().take(70).collect();
            println!("   {}. {}", i + 1, truncated);
        }
    }

    println!("\n{}═══════════════════════════════════════════════════════════════{}", red, reset);
    println!("{}🔥 Analysis complete. {} threat patterns identified.{}", red, patterns.len(), reset);
    println!("{}═══════════════════════════════════════════════════════════════{}\n", red, reset);
}
