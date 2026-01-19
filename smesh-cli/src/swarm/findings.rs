//! Vulnerability findings and consensus types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Severity levels for vulnerabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Severity {
    /// Informational finding
    Info,
    /// Low severity - minimal impact
    Low,
    /// Medium severity - moderate impact
    Medium,
    /// High severity - significant impact
    High,
    /// Critical severity - severe impact, immediate action required
    Critical,
}

impl Severity {
    /// ANSI color code for terminal output
    pub fn color(&self) -> &'static str {
        match self {
            Severity::Critical => "\x1b[91m", // Bright red
            Severity::High => "\x1b[31m",     // Red
            Severity::Medium => "\x1b[33m",   // Yellow
            Severity::Low => "\x1b[36m",      // Cyan
            Severity::Info => "\x1b[37m",     // White
        }
    }

    /// Reset ANSI color
    pub fn reset() -> &'static str {
        "\x1b[0m"
    }

    /// Display name
    pub fn name(&self) -> &'static str {
        match self {
            Severity::Critical => "CRITICAL",
            Severity::High => "HIGH",
            Severity::Medium => "MEDIUM",
            Severity::Low => "LOW",
            Severity::Info => "INFO",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "CRITICAL" | "CRIT" | "C" => Severity::Critical,
            "HIGH" | "H" => Severity::High,
            "MEDIUM" | "MED" | "M" => Severity::Medium,
            "LOW" | "L" => Severity::Low,
            _ => Severity::Info,
        }
    }

    /// Compact code for TOON format
    pub fn code(&self) -> &'static str {
        match self {
            Severity::Critical => "C",
            Severity::High => "H",
            Severity::Medium => "M",
            Severity::Low => "L",
            Severity::Info => "I",
        }
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A vulnerability finding from an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnFinding {
    /// Unique deduplication ID based on content
    pub dedup_id: String,

    /// File where the vulnerability was found
    pub file_path: String,

    /// Line number (if identified)
    pub line_number: Option<u32>,

    /// Vulnerability type/category
    pub vuln_type: String,

    /// Severity level
    pub severity: Severity,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,

    /// Description of the vulnerability
    pub description: String,

    /// Agent ID that found this
    pub found_by: String,

    /// Timestamp when found
    pub found_at: DateTime<Utc>,

    /// Number of times this finding was reinforced
    pub reinforcement_count: u32,

    /// Agent IDs that reinforced this finding
    pub reinforced_by: Vec<String>,
}

impl VulnFinding {
    /// Create a new finding
    pub fn new(
        file_path: impl Into<String>,
        vuln_type: impl Into<String>,
        severity: Severity,
        confidence: f64,
        description: impl Into<String>,
        found_by: impl Into<String>,
    ) -> Self {
        let file_path = file_path.into();
        let vuln_type = vuln_type.into();
        let description = description.into();
        let found_by = found_by.into();

        // Compute dedup ID from file + vuln_type + first 100 chars of description
        let dedup_id = Self::compute_dedup_id(&file_path, &vuln_type, &description);

        Self {
            dedup_id,
            file_path,
            line_number: None,
            vuln_type,
            severity,
            confidence: confidence.clamp(0.0, 1.0),
            description,
            found_by,
            found_at: Utc::now(),
            reinforcement_count: 0,
            reinforced_by: Vec::new(),
        }
    }

    /// Set line number
    pub fn with_line(mut self, line: u32) -> Self {
        self.line_number = Some(line);
        self
    }

    /// Compute deduplication ID
    fn compute_dedup_id(file_path: &str, vuln_type: &str, description: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(file_path.as_bytes());
        hasher.update(vuln_type.as_bytes());
        // Use first 100 chars of description for dedup
        hasher.update(description.chars().take(100).collect::<String>().as_bytes());
        format!("{:x}", hasher.finalize())[..12].to_string()
    }

    /// Reinforce this finding from another agent
    pub fn reinforce(&mut self, agent_id: &str) {
        if !self.reinforced_by.contains(&agent_id.to_string()) {
            self.reinforced_by.push(agent_id.to_string());
            self.reinforcement_count += 1;

            // Boost confidence with diminishing returns
            let boost = 0.05 / (1.0 + self.reinforcement_count as f64 * 0.3);
            self.confidence = (self.confidence + boost).min(1.0);
        }
    }

    /// Check if consensus is reached
    ///
    /// Total agents = 1 (original finder) + reinforcement_count
    pub fn has_consensus(&self, threshold: u32, high_conf_threshold: f64) -> bool {
        let total_agents = 1 + self.reinforcement_count;

        // Standard consensus: N agents agree
        if total_agents >= threshold {
            return true;
        }

        // Fast-track: 2+ agents with high confidence
        total_agents >= 2 && self.confidence >= high_conf_threshold
    }

    /// Convert to compact payload for TOON encoding
    pub fn to_compact_payload(&self) -> VulnFindingPayload {
        VulnFindingPayload {
            f: self.file_path.clone(),
            l: self.line_number,
            t: compress_vuln_type(&self.vuln_type),
            s: self.severity.code().to_string(),
            c: (self.confidence * 100.0) as u8,
            d: self.description.chars().take(100).collect(),
        }
    }
}

/// Compact payload for TOON encoding (reduces token costs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnFindingPayload {
    /// File path
    pub f: String,
    /// Line number
    pub l: Option<u32>,
    /// Vulnerability type code
    pub t: String,
    /// Severity code (C/H/M/L/I)
    pub s: String,
    /// Confidence percentage (0-100)
    pub c: u8,
    /// Description (truncated)
    pub d: String,
}

/// A finding that has reached consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusFinding {
    /// The original finding
    pub finding: VulnFinding,

    /// Time taken to reach consensus (ms)
    pub time_to_consensus_ms: u64,

    /// Number of ticks to reach consensus
    pub ticks_to_consensus: u32,
}

impl ConsensusFinding {
    /// Create from a finding that reached consensus
    pub fn from_finding(finding: VulnFinding, start_time: DateTime<Utc>, ticks: u32) -> Self {
        let time_to_consensus_ms = (Utc::now() - start_time).num_milliseconds().max(0) as u64;

        Self {
            finding,
            time_to_consensus_ms,
            ticks_to_consensus: ticks,
        }
    }
}

/// Compress vulnerability type to short code
fn compress_vuln_type(vuln_type: &str) -> String {
    let lower = vuln_type.to_lowercase();
    if lower.contains("injection") || lower.contains("sqli") {
        "inj".to_string()
    } else if lower.contains("xss") || lower.contains("cross-site") {
        "xss".to_string()
    } else if lower.contains("auth") {
        "auth".to_string()
    } else if lower.contains("crypto") || lower.contains("encrypt") {
        "cryp".to_string()
    } else if lower.contains("deserial") {
        "deser".to_string()
    } else if lower.contains("path") || lower.contains("traversal") {
        "path".to_string()
    } else if lower.contains("ssrf") {
        "ssrf".to_string()
    } else {
        vuln_type.chars().take(4).collect()
    }
}

/// Parse findings from LLM response text
pub fn parse_findings_from_response(
    response: &str,
    file_path: &str,
    agent_id: &str,
    default_vuln_type: &str,
) -> Vec<VulnFinding> {
    let mut findings = Vec::new();

    // Look for structured finding markers
    // Expected format:
    // [SEVERITY] VulnType: Description (line X)
    // or
    // - [SEVERITY] Description

    for line in response.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Skip lines that indicate no findings
        if line.to_lowercase().contains("no vulnerabilit")
            || line.to_lowercase().contains("no issues")
            || line.to_lowercase().contains("code appears safe")
            || line.to_lowercase().contains("not found")
        {
            continue;
        }

        // Try to parse [SEVERITY] format
        if let Some(finding) = parse_severity_line(line, file_path, agent_id, default_vuln_type) {
            findings.push(finding);
            continue;
        }

        // Try to parse bullet point format
        if (line.starts_with('-') || line.starts_with('*') || line.starts_with("•"))
            && line.len() > 10
        {
            let content = line.trim_start_matches(['-', '*', '•']).trim();
            if !content.is_empty() && is_likely_finding(content) {
                let severity = infer_severity(content);
                let finding = VulnFinding::new(
                    file_path,
                    default_vuln_type,
                    severity,
                    0.6, // Default confidence for inferred findings
                    content,
                    agent_id,
                );
                findings.push(finding);
            }
        }
    }

    findings
}

/// Parse a line with [SEVERITY] format
fn parse_severity_line(
    line: &str,
    file_path: &str,
    agent_id: &str,
    default_vuln_type: &str,
) -> Option<VulnFinding> {
    // Match patterns like [CRITICAL], [HIGH], etc.
    let severities = ["CRITICAL", "HIGH", "MEDIUM", "MED", "LOW", "INFO"];

    for sev_str in severities {
        let pattern = format!("[{}]", sev_str);
        if line.to_uppercase().contains(&pattern) {
            let severity = Severity::from_str(sev_str);
            let description = line
                .replace(&format!("[{}]", sev_str), "")
                .replace(&format!("[{}]", sev_str.to_lowercase()), "")
                .trim()
                .to_string();

            if description.len() > 5 {
                // Extract line number if present (e.g., "line 42" or ":42")
                let line_num = extract_line_number(&description);

                // Try to extract vuln type from description
                let vuln_type =
                    extract_vuln_type(&description).unwrap_or(default_vuln_type.to_string());

                let mut finding = VulnFinding::new(
                    file_path,
                    vuln_type,
                    severity,
                    severity_to_confidence(severity),
                    &description,
                    agent_id,
                );

                if let Some(ln) = line_num {
                    finding = finding.with_line(ln);
                }

                return Some(finding);
            }
        }
    }

    None
}

/// Extract line number from description
fn extract_line_number(text: &str) -> Option<u32> {
    // Match "line 42", "Line: 42", ":42", "L42"
    let patterns = [r"[Ll]ine\s*:?\s*(\d+)", r":(\d+)", r"L(\d+)"];

    for pattern in patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if let Some(caps) = re.captures(text) {
                if let Some(m) = caps.get(1) {
                    if let Ok(num) = m.as_str().parse::<u32>() {
                        return Some(num);
                    }
                }
            }
        }
    }

    None
}

/// Extract vulnerability type from description
fn extract_vuln_type(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let types = [
        ("sql injection", "SQL Injection"),
        ("command injection", "Command Injection"),
        ("code injection", "Code Injection"),
        ("ldap injection", "LDAP Injection"),
        ("xpath injection", "XPath Injection"),
        ("xss", "Cross-Site Scripting"),
        ("cross-site scripting", "Cross-Site Scripting"),
        ("broken auth", "Broken Authentication"),
        ("authentication", "Authentication Issue"),
        ("authorization", "Authorization Issue"),
        ("crypto", "Cryptographic Issue"),
        ("weak encrypt", "Weak Encryption"),
        ("hardcoded", "Hardcoded Credentials"),
        ("deserializ", "Insecure Deserialization"),
        ("path traversal", "Path Traversal"),
        ("directory traversal", "Directory Traversal"),
        ("ssrf", "Server-Side Request Forgery"),
        ("open redirect", "Open Redirect"),
        ("race condition", "Race Condition"),
        ("buffer overflow", "Buffer Overflow"),
        ("memory leak", "Memory Leak"),
        ("use after free", "Use After Free"),
    ];

    for (pattern, vuln_type) in types {
        if lower.contains(pattern) {
            return Some(vuln_type.to_string());
        }
    }

    None
}

/// Check if text is likely a vulnerability finding
fn is_likely_finding(text: &str) -> bool {
    let lower = text.to_lowercase();
    let vuln_keywords = [
        "vulnerab",
        "inject",
        "xss",
        "auth",
        "crypto",
        "unsafe",
        "insecure",
        "hardcoded",
        "leak",
        "overflow",
        "traversal",
        "ssrf",
        "deserial",
        "password",
        "credential",
        "secret",
        "token",
        "sensitive",
        "exploit",
        "bypass",
        "privilege",
        "escalat",
    ];

    vuln_keywords.iter().any(|kw| lower.contains(kw))
}

/// Infer severity from finding text
fn infer_severity(text: &str) -> Severity {
    let lower = text.to_lowercase();

    if lower.contains("critical")
        || lower.contains("rce")
        || lower.contains("remote code")
        || lower.contains("sql injection")
        || lower.contains("command injection")
    {
        Severity::Critical
    } else if lower.contains("high")
        || lower.contains("authentication bypass")
        || lower.contains("privilege escalation")
        || lower.contains("hardcoded password")
        || lower.contains("hardcoded secret")
    {
        Severity::High
    } else if lower.contains("medium")
        || lower.contains("xss")
        || lower.contains("path traversal")
        || lower.contains("ssrf")
    {
        Severity::Medium
    } else if lower.contains("low") || lower.contains("information disclosure") {
        Severity::Low
    } else {
        Severity::Medium // Default to medium for unknown
    }
}

/// Map severity to default confidence
fn severity_to_confidence(severity: Severity) -> f64 {
    match severity {
        Severity::Critical => 0.85,
        Severity::High => 0.80,
        Severity::Medium => 0.70,
        Severity::Low => 0.60,
        Severity::Info => 0.50,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::High);
        assert!(Severity::High > Severity::Medium);
        assert!(Severity::Medium > Severity::Low);
        assert!(Severity::Low > Severity::Info);
    }

    #[test]
    fn test_finding_dedup() {
        let f1 = VulnFinding::new(
            "test.rs",
            "SQLi",
            Severity::High,
            0.8,
            "Found SQL injection",
            "agent1",
        );
        let f2 = VulnFinding::new(
            "test.rs",
            "SQLi",
            Severity::High,
            0.8,
            "Found SQL injection",
            "agent2",
        );

        assert_eq!(f1.dedup_id, f2.dedup_id);
    }

    #[test]
    fn test_finding_reinforcement() {
        let mut finding = VulnFinding::new(
            "test.rs",
            "XSS",
            Severity::Medium,
            0.7,
            "XSS found",
            "agent1",
        );

        finding.reinforce("agent2");
        assert_eq!(finding.reinforcement_count, 1);
        assert!(finding.confidence > 0.7);

        // Duplicate reinforcement should not increase count
        finding.reinforce("agent2");
        assert_eq!(finding.reinforcement_count, 1);
    }

    #[test]
    fn test_consensus_detection() {
        // Low confidence finding - won't trigger fast-track
        let mut finding =
            VulnFinding::new("test.rs", "XSS", Severity::Medium, 0.5, "XSS", "agent1");

        // 1 agent (original) - no consensus
        assert!(
            !finding.has_consensus(3, 0.8),
            "1 agent should not reach consensus"
        );

        // 2 agents (original + 1 reinforcement) with low confidence - no consensus
        finding.reinforce("agent2");
        assert!(
            !finding.has_consensus(3, 0.8),
            "2 agents with low confidence should not reach consensus"
        );

        // 3 agents (original + 2 reinforcements) - standard consensus reached
        finding.reinforce("agent3");
        assert!(
            finding.has_consensus(3, 0.8),
            "3 agents should reach standard consensus"
        );

        // High confidence fast-track test
        let mut high_conf_finding = VulnFinding::new(
            "test2.rs",
            "SQLi",
            Severity::Critical,
            0.85,
            "SQLi",
            "agent1",
        );

        // 1 agent with high confidence - no consensus (need 2+ for fast-track)
        assert!(
            !high_conf_finding.has_consensus(3, 0.8),
            "1 agent even with high confidence should not reach consensus"
        );

        // 2 agents with high confidence - fast-track consensus
        high_conf_finding.reinforce("agent2");
        assert!(
            high_conf_finding.has_consensus(3, 0.8),
            "2 agents with high confidence should fast-track consensus"
        );
    }

    #[test]
    fn test_parse_severity_line() {
        let line = "[HIGH] SQL Injection in query function at line 42";
        let finding = parse_severity_line(line, "db.rs", "agent1", "Unknown");

        assert!(finding.is_some());
        let f = finding.unwrap();
        assert_eq!(f.severity, Severity::High);
        assert_eq!(f.line_number, Some(42));
    }
}
