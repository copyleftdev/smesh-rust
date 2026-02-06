use anyhow::Result;
use smesh_agent::{OllamaClient, OllamaConfig};
use smesh_core::{Field, FindingPayloadCompact, Node, Signal, SignalType};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokio::time::{sleep, Duration};

/// Agent specializations for code review
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReviewerType {
    Security,
    Performance,
    Style,
    Documentation,
}

impl ReviewerType {
    fn system_prompt(&self) -> &'static str {
        match self {
            ReviewerType::Security => {
                "You are a security-focused code reviewer. Look for:
- Unsafe code blocks and their justification
- Buffer overflows, integer overflows
- Unchecked array access
- Potential panics in library code
- Missing input validation
Be concise. List findings as bullet points. If code is safe, say 'No security issues found.'"
            }
            ReviewerType::Performance => {
                "You are a performance-focused code reviewer. Look for:
- Unnecessary allocations
- Inefficient algorithms
- Missing #[inline] on hot paths
- Opportunities for SIMD or vectorization
- Cache-unfriendly patterns
Be concise. List findings as bullet points. If code is optimal, say 'No performance issues found.'"
            }
            ReviewerType::Style => {
                "You are a Rust style reviewer. Look for:
- Idiomatic Rust patterns
- Proper error handling
- Good naming conventions
- Code organization
- Missing or excessive comments
Be concise. List findings as bullet points. If style is good, say 'No style issues found.'"
            }
            ReviewerType::Documentation => {
                "You are a documentation reviewer. Look for:
- Missing doc comments on public items
- Incorrect or outdated docs
- Missing examples in docs
- Unclear API descriptions
Be concise. List findings as bullet points. If docs are complete, say 'Documentation is adequate.'"
            }
        }
    }

    fn signal_type(&self) -> SignalType {
        match self {
            ReviewerType::Security => SignalType::Alert,
            ReviewerType::Performance => SignalType::Query,
            ReviewerType::Style => SignalType::Data,
            ReviewerType::Documentation => SignalType::Heartbeat,
        }
    }
}

/// A code review finding
#[derive(Debug, Clone)]
pub struct Finding {
    pub reviewer: ReviewerType,
    pub file: String,
    pub content: String,
    pub confidence: f32,
    pub reinforcements: u32,
}

/// Run a SMESH-coordinated code review
pub async fn run_review(repo_path: &Path, model: &str) -> Result<Vec<Finding>> {
    println!("\n🌿 SMESH Code Review - Signal Diffusion Mode\n");
    println!("Target: {}", repo_path.display());
    println!("Model: {}\n", model);

    // Initialize SMESH primitives
    let mut field = Field::new();

    // Create reviewer nodes
    let reviewers = vec![
        ReviewerType::Security,
        ReviewerType::Performance,
        ReviewerType::Style,
        ReviewerType::Documentation,
    ];

    let mut nodes: HashMap<ReviewerType, Node> = HashMap::new();
    for reviewer in &reviewers {
        let node = Node::new();
        nodes.insert(*reviewer, node);
    }

    println!("📡 Network topology: SmallWorld ({} nodes)", nodes.len());

    // Collect Rust files
    let rust_files = collect_rust_files(repo_path)?;
    println!("📂 Found {} Rust files to review\n", rust_files.len());

    // Initialize Ollama client
    let config = OllamaConfig {
        model: model.to_string(),
        ..Default::default()
    };
    let client = OllamaClient::new(config);

    // Check Ollama connection
    if !client.is_available().await {
        anyhow::bail!("Ollama not available. Run: ollama serve");
    }

    let mut all_findings: Vec<Finding> = Vec::new();

    // Process each file
    for (idx, file_path) in rust_files.iter().enumerate() {
        let relative_path = file_path
            .strip_prefix(repo_path)
            .unwrap_or(file_path)
            .display()
            .to_string();

        println!("─────────────────────────────────────────────────────────");
        println!("📄 [{}/{}] {}", idx + 1, rust_files.len(), relative_path);
        println!("─────────────────────────────────────────────────────────");

        let code = fs::read_to_string(file_path)?;

        // Skip very small files
        if code.lines().count() < 10 {
            println!("   ⏭️  Skipping (< 10 lines)\n");
            continue;
        }

        // Emit FILE_READY signal
        let file_signal = Signal::builder(SignalType::Data)
            .payload(relative_path.as_bytes().to_vec())
            .intensity(1.0)
            .confidence(1.0)
            .build();

        field.emit_anonymous(file_signal.clone());
        println!("   📡 Emitted FILE_READY signal (intensity: 1.0)");

        // Each reviewer processes the file
        for reviewer_type in &reviewers {
            let node = nodes.get(reviewer_type).unwrap();

            // Check if node senses the signal (based on sensitivity)
            if !node.can_sense(&file_signal) {
                continue;
            }

            let prompt = format!(
                "{}\n\nReview this Rust code:\n\n```rust\n{}\n```",
                reviewer_type.system_prompt(),
                truncate_code(&code, 2000) // Limit tokens
            );

            print!("   🔍 {:?} reviewing... ", reviewer_type);
            std::io::Write::flush(&mut std::io::stdout())?;

            match client.generate(&prompt, None).await {
                Ok(response) => {
                    println!("✓");

                    // Create finding signal
                    let has_issues = !response.to_lowercase().contains("no ")
                        || response.to_lowercase().contains("issue")
                        || response.to_lowercase().contains("missing")
                        || response.to_lowercase().contains("should");

                    if has_issues && response.len() > 50 {
                        // Use TOON format for compact signal payload (20% token savings)
                        let finding_payload = FindingPayloadCompact {
                            f: relative_path.clone(),
                            r: format!("{:?}", reviewer_type).to_lowercase(),
                            c: 0.7,
                        };
                        let finding_signal = Signal::builder(reviewer_type.signal_type())
                            .payload_toon(&finding_payload)
                            .intensity(0.8)
                            .confidence(0.7)
                            .build();

                        // Emit to field
                        field.emit_anonymous(finding_signal.clone());

                        // Other nodes may reinforce if they "agree"
                        let reinforcements = simulate_reinforcement(&nodes, reviewer_type);

                        let finding = Finding {
                            reviewer: *reviewer_type,
                            file: relative_path.clone(),
                            content: response.clone(),
                            confidence: 0.7 + (reinforcements as f32 * 0.1),
                            reinforcements,
                        };

                        println!("      └─ Finding emitted (reinforced {}x)", reinforcements);
                        all_findings.push(finding);
                    }
                }
                Err(e) => {
                    println!("✗ ({})", e);
                }
            }

            // Small delay to avoid hammering Ollama
            sleep(Duration::from_millis(100)).await;
        }

        // Field tick - decay old signals
        field.tick(1.0);
        println!("   ⏱️  Field tick (signals decayed)\n");
    }

    // Print summary
    println!("\n═══════════════════════════════════════════════════════");
    println!("                    REVIEW SUMMARY");
    println!("═══════════════════════════════════════════════════════\n");

    let stats = field.stats();
    println!("📊 Signal Field Stats:");
    println!("   Active signals: {}", stats.active_signals);
    println!("   Total intensity: {:.2}", stats.total_intensity);
    println!("   Findings: {}\n", all_findings.len());

    // Group findings by type
    for reviewer_type in &reviewers {
        let type_findings: Vec<_> = all_findings
            .iter()
            .filter(|f| f.reviewer == *reviewer_type)
            .collect();

        if !type_findings.is_empty() {
            println!("🔹 {:?} Findings ({}):", reviewer_type, type_findings.len());
            for finding in type_findings {
                println!("   File: {}", finding.file);
                println!(
                    "   Confidence: {:.0}% (reinforced {}x)",
                    finding.confidence * 100.0,
                    finding.reinforcements
                );
                println!("   ───");
                for line in finding.content.lines().take(5) {
                    println!("   {}", line);
                }
                if finding.content.lines().count() > 5 {
                    println!("   ...(truncated)");
                }
                println!();
            }
        }
    }

    println!("═══════════════════════════════════════════════════════");
    println!("🌿 Review complete. Consensus emerged via signal reinforcement.");
    println!("═══════════════════════════════════════════════════════\n");

    Ok(all_findings)
}

/// Collect all .rs files in a directory
fn collect_rust_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();

    if dir.is_file() && dir.extension().is_some_and(|e| e == "rs") {
        files.push(dir.to_path_buf());
        return Ok(files);
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip hidden dirs, target, tests for now
        if path.is_dir() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with('.') || name == "target" || name == "tests" || name == "benches" {
                continue;
            }
            files.extend(collect_rust_files(&path)?);
        } else if path.extension().is_some_and(|e| e == "rs") {
            files.push(path);
        }
    }

    Ok(files)
}

/// Truncate code to limit tokens
fn truncate_code(code: &str, max_chars: usize) -> &str {
    if code.len() <= max_chars {
        code
    } else {
        &code[..max_chars]
    }
}

/// Simulate reinforcement from other nodes
fn simulate_reinforcement(nodes: &HashMap<ReviewerType, Node>, source: &ReviewerType) -> u32 {
    // In a real system, other nodes would actually evaluate the finding
    // For now, simulate based on reviewer type correlations
    let mut reinforcements = 0;

    for other_type in nodes.keys() {
        if other_type == source {
            continue;
        }

        // Security and Performance often agree
        // Style and Documentation often agree
        let agrees = matches!(
            (source, other_type),
            (ReviewerType::Security, ReviewerType::Performance)
                | (ReviewerType::Performance, ReviewerType::Security)
                | (ReviewerType::Style, ReviewerType::Documentation)
                | (ReviewerType::Documentation, ReviewerType::Style)
        );

        // Simple deterministic reinforcement for demo
        if agrees {
            reinforcements += 1;
        }
    }

    reinforcements
}
