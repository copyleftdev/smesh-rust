//! Claude-powered analysis layer
//!
//! Takes raw findings from the web swarm and uses Claude with tool-use
//! to reason about exploitability, build attack chains, and generate reports.

use smesh_agent::{
    ClaudeClient, GenerateRequestV2, LlmBackend,
};

use crate::web_swarm::CorrelatedFinding;

/// Claude-powered exploitability analysis on correlated findings
pub async fn analyze_exploitability(
    findings: &[CorrelatedFinding],
    target: &str,
) -> Result<Vec<AnalyzedFinding>, String> {
    let client = ClaudeClient::from_env()
        .ok_or("ANTHROPIC_API_KEY not set")?;

    let mut analyzed = Vec::new();

    if findings.is_empty() {
        return Ok(analyzed);
    }

    println!("\n\x1b[1;35m--- Claude Analysis: Exploitability Assessment ---\x1b[0m\n");

    // Build a summary of all findings for Claude
    let findings_text = findings.iter().enumerate().map(|(i, cf)| {
        format!(
            "{}. [{}] {} ({})\n   URL: {}\n   Evidence: {}\n   Category: {}\n   Confirmed by: {}\n   Confidence: {:.0}%",
            i + 1,
            cf.finding.severity,
            cf.finding.title,
            cf.finding.category,
            cf.finding.url,
            cf.finding.evidence,
            cf.finding.category,
            cf.confirmed_by.join(", "),
            cf.confidence * 100.0,
        )
    }).collect::<Vec<_>>().join("\n\n");

    let prompt = format!(
        r#"You are an elite red team analyst assessing findings from an authorized security assessment of {}.

Here are the correlated findings from our multi-phase scan:

{}

For each finding, assess:
1. EXPLOITABILITY: Is this actually exploitable? (Easy/Medium/Hard/Theoretical/False Positive)
2. IMPACT: What could an attacker achieve? (RCE, Data Breach, Account Takeover, Info Disclosure, DoS)
3. ATTACK CHAIN: Can this be combined with other findings for a more severe attack?
4. CVSS ESTIMATE: Rough CVSS 3.1 score (0.0-10.0)
5. PRIORITY: Should this be fixed immediately, soon, or eventually?

Also identify:
- Any ATTACK CHAINS where multiple findings combine into a more severe vulnerability
- FALSE POSITIVES that should be filtered out
- The TOP 3 most impactful findings and why

Be specific, technical, and honest about false positives. This is HIPAA-regulated dental/healthcare data."#,
        target, findings_text
    );

    println!("  Sending {} findings to Claude for analysis...", findings.len());

    let request = GenerateRequestV2::simple(&prompt)
        .with_system("You are a senior penetration tester with expertise in web application security, cloud infrastructure, WordPress, and healthcare/HIPAA compliance. Be precise, technical, and actionable.")
        .with_temperature(0.3)
        .with_max_tokens(4096);

    match client.generate_v2(request).await {
        Ok(response) => {
            let analysis = response.text();
            println!("  Claude analysis complete ({} tokens)\n", response.output_tokens);

            // Parse the analysis into structured findings
            analyzed = parse_analysis(&analysis, findings);

            // Print the analysis
            println!("{}\n", analysis);
        }
        Err(e) => {
            println!("  Claude analysis failed: {}", e);
            return Err(e.to_string());
        }
    }

    Ok(analyzed)
}

/// Generate a full red team report using Claude
pub async fn generate_report(
    findings: &[CorrelatedFinding],
    analyzed: &[AnalyzedFinding],
    target: &str,
    subdomains: &[String],
    endpoints_count: usize,
    duration_secs: f64,
) -> Result<String, String> {
    let client = ClaudeClient::from_env()
        .ok_or("ANTHROPIC_API_KEY not set")?;

    println!("\n\x1b[1;35m--- Claude Analysis: Report Generation ---\x1b[0m\n");

    let findings_text = analyzed.iter().enumerate().map(|(i, af)| {
        format!(
            "{}. [{}] {} (CVSS: {:.1})\n   Exploitability: {}\n   Impact: {}\n   URL: {}\n   Attack Chain: {}\n   Priority: {}",
            i + 1,
            af.severity,
            af.title,
            af.cvss_estimate,
            af.exploitability,
            af.impact,
            af.url,
            af.attack_chain.as_deref().unwrap_or("None"),
            af.priority,
        )
    }).collect::<Vec<_>>().join("\n\n");

    // If no analyzed findings, use raw correlated
    let findings_for_report = if findings_text.is_empty() {
        findings.iter().enumerate().map(|(i, cf)| {
            format!(
                "{}. [{}] {}\n   URL: {}\n   Evidence: {}\n   Confidence: {:.0}%",
                i + 1, cf.finding.severity, cf.finding.title,
                cf.finding.url, cf.finding.evidence, cf.confidence * 100.0,
            )
        }).collect::<Vec<_>>().join("\n\n")
    } else {
        findings_text
    };

    let prompt = format!(
        r#"Write a professional security assessment report for an authorized red team engagement.

TARGET: {}
SCOPE: External web application security assessment
SUBDOMAINS DISCOVERED: {}
ENDPOINTS CRAWLED: {}
SCAN DURATION: {:.1} seconds

FINDINGS:
{}

Write the report in Markdown with these sections:
1. Executive Summary (2-3 sentences for C-level audience)
2. Scope & Methodology
3. Risk Summary (critical/high/medium/low counts)
4. Critical & High Findings (detailed writeup for each)
5. Medium & Low Findings (brief)
6. Attack Chain Analysis (if findings combine)
7. Remediation Roadmap (prioritized)
8. HIPAA Compliance Implications (this is healthcare/dental data)

For each critical/high finding include:
- Description
- Impact
- Reproduction steps
- Remediation
- CWE reference

Keep it professional, actionable, and suitable for both executives and engineers."#,
        target,
        subdomains.len(),
        endpoints_count,
        duration_secs,
        findings_for_report,
    );

    println!("  Generating report via Claude...");

    let request = GenerateRequestV2::simple(&prompt)
        .with_system("You are a senior security consultant writing a professional penetration test report. Use clear, precise language. The audience is both technical (engineers) and non-technical (executives, compliance officers).")
        .with_temperature(0.4)
        .with_max_tokens(8192);

    match client.generate_v2(request).await {
        Ok(response) => {
            let report = response.text();
            println!("  Report generated ({} tokens)\n", response.output_tokens);
            Ok(report)
        }
        Err(e) => {
            println!("  Report generation failed: {}", e);
            Err(e.to_string())
        }
    }
}

/// A finding with Claude's exploitability analysis
#[derive(Debug, Clone)]
pub struct AnalyzedFinding {
    pub title: String,
    pub severity: String,
    pub url: String,
    pub exploitability: String,
    pub impact: String,
    pub cvss_estimate: f64,
    pub priority: String,
    pub attack_chain: Option<String>,
    pub false_positive: bool,
}

/// Parse Claude's analysis text into structured findings
fn parse_analysis(analysis: &str, original: &[CorrelatedFinding]) -> Vec<AnalyzedFinding> {
    // Map original findings and try to extract Claude's assessment for each
    original.iter().map(|cf| {
        AnalyzedFinding {
            title: cf.finding.title.clone(),
            severity: cf.finding.severity.clone(),
            url: cf.finding.url.clone(),
            exploitability: extract_field(analysis, &cf.finding.title, "Exploitability")
                .unwrap_or_else(|| "Unknown".into()),
            impact: extract_field(analysis, &cf.finding.title, "Impact")
                .unwrap_or_else(|| "Unknown".into()),
            cvss_estimate: extract_cvss(analysis, &cf.finding.title).unwrap_or(5.0),
            priority: extract_field(analysis, &cf.finding.title, "Priority")
                .unwrap_or_else(|| "Medium".into()),
            attack_chain: extract_field(analysis, &cf.finding.title, "Attack Chain"),
            false_positive: analysis.to_lowercase().contains(&format!(
                "false positive"
            )) && analysis.contains(&cf.finding.title),
        }
    }).collect()
}

fn extract_field(text: &str, finding_title: &str, field: &str) -> Option<String> {
    // Simple extraction: look for the field name near the finding title
    let lower = text.to_lowercase();
    let title_lower = finding_title.to_lowercase();
    let field_lower = field.to_lowercase();

    if let Some(title_pos) = lower.find(&title_lower) {
        let region = &text[title_pos..text.len().min(title_pos + 500)];
        for line in region.lines() {
            let line_lower = line.to_lowercase();
            if line_lower.contains(&field_lower) {
                if let Some(colon_pos) = line.find(':') {
                    let value = line[colon_pos + 1..].trim();
                    if !value.is_empty() {
                        return Some(value.to_string());
                    }
                }
            }
        }
    }
    None
}

fn extract_cvss(text: &str, finding_title: &str) -> Option<f64> {
    let lower = text.to_lowercase();
    let title_lower = finding_title.to_lowercase();

    if let Some(title_pos) = lower.find(&title_lower) {
        let region = &lower[title_pos..lower.len().min(title_pos + 500)];
        // Look for patterns like "CVSS: 7.5" or "CVSS 3.1: 8.0"
        if let Some(cvss_pos) = region.find("cvss") {
            let after = &region[cvss_pos..];
            for word in after.split_whitespace().skip(1).take(5) {
                let cleaned = word.trim_matches(|c: char| !c.is_numeric() && c != '.');
                if let Ok(score) = cleaned.parse::<f64>() {
                    if (0.0..=10.0).contains(&score) {
                        return Some(score);
                    }
                }
            }
        }
    }
    None
}
