//! Bounty hunting specializations
//!
//! Each specialization defines what an agent focuses on, which tools it
//! prefers, and how it interacts with findings from other agents.

use serde::{Deserialize, Serialize};

use crate::tools::TrustLevel;

/// Bounty hunting specializations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BountySpecialization {
    /// Surface mapping: enumerate endpoints, ports, services
    Recon,
    /// Static code analysis: OWASP Top 10 patterns in source
    SourceAudit,
    /// PoC development: confirm vulns, assess exploitability
    ExploitAnalyst,
    /// Supply chain: audit dependencies for known CVEs
    DependencyHunter,
    /// Configuration review: misconfigs, exposed secrets, permissions
    ConfigAudit,
    /// Dedup, correlate, and prioritize findings from other agents
    Triager,
    /// Write structured reports (markdown, SARIF, HackerOne-style)
    ReportWriter,
}

impl BountySpecialization {
    /// All specializations
    pub fn all() -> Vec<Self> {
        vec![
            Self::Recon,
            Self::SourceAudit,
            Self::ExploitAnalyst,
            Self::DependencyHunter,
            Self::ConfigAudit,
            Self::Triager,
            Self::ReportWriter,
        ]
    }

    /// Display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Recon => "Recon Scout",
            Self::SourceAudit => "Source Auditor",
            Self::ExploitAnalyst => "Exploit Analyst",
            Self::DependencyHunter => "Dependency Hunter",
            Self::ConfigAudit => "Config Auditor",
            Self::Triager => "Triager",
            Self::ReportWriter => "Report Writer",
        }
    }

    /// Short code for agent IDs
    pub fn code(&self) -> &'static str {
        match self {
            Self::Recon => "RECON",
            Self::SourceAudit => "AUDIT",
            Self::ExploitAnalyst => "EXPLOIT",
            Self::DependencyHunter => "DEPS",
            Self::ConfigAudit => "CONFIG",
            Self::Triager => "TRIAGE",
            Self::ReportWriter => "REPORT",
        }
    }

    /// Minimum trust level this specialization needs
    pub fn min_trust_level(&self) -> TrustLevel {
        match self {
            Self::Recon => TrustLevel::Standard,         // Needs bash for nmap/ffuf
            Self::SourceAudit => TrustLevel::ReadOnly,   // Just reads code
            Self::ExploitAnalyst => TrustLevel::Elevated, // May write PoC files
            Self::DependencyHunter => TrustLevel::Standard, // Runs cargo-audit etc.
            Self::ConfigAudit => TrustLevel::ReadOnly,   // Reads configs
            Self::Triager => TrustLevel::ReadOnly,       // Reads findings
            Self::ReportWriter => TrustLevel::Standard,  // May fetch CVE refs
        }
    }

    /// Preferred tools for this specialization
    pub fn preferred_tools(&self) -> Vec<&'static str> {
        match self {
            Self::Recon => vec!["bash", "web_fetch", "glob"],
            Self::SourceAudit => vec!["file_read", "grep", "glob"],
            Self::ExploitAnalyst => vec!["bash", "file_read", "web_fetch", "grep"],
            Self::DependencyHunter => vec!["bash", "file_read", "grep", "glob"],
            Self::ConfigAudit => vec!["file_read", "grep", "glob"],
            Self::Triager => vec!["file_read", "grep"],
            Self::ReportWriter => vec!["file_read", "web_fetch"],
        }
    }

    /// System prompt for this specialization
    pub fn system_prompt(&self) -> &'static str {
        match self {
            Self::Recon => RECON_PROMPT,
            Self::SourceAudit => SOURCE_AUDIT_PROMPT,
            Self::ExploitAnalyst => EXPLOIT_ANALYST_PROMPT,
            Self::DependencyHunter => DEPENDENCY_HUNTER_PROMPT,
            Self::ConfigAudit => CONFIG_AUDIT_PROMPT,
            Self::Triager => TRIAGER_PROMPT,
            Self::ReportWriter => REPORT_WRITER_PROMPT,
        }
    }

    /// Relevance keywords for file routing
    pub fn keywords(&self) -> &[&'static str] {
        match self {
            Self::Recon => &[
                "api", "route", "endpoint", "handler", "controller", "server",
                "listen", "port", "bind", "http", "grpc", "graphql", "socket",
            ],
            Self::SourceAudit => &[
                "sql", "query", "exec", "eval", "render", "html", "template",
                "auth", "login", "session", "cookie", "input", "request",
                "password", "token", "encrypt", "hash", "serialize", "parse",
            ],
            Self::ExploitAnalyst => &[
                "vuln", "exploit", "payload", "inject", "overflow", "bypass",
                "escalat", "privilege", "admin", "root", "sudo",
            ],
            Self::DependencyHunter => &[
                "require", "import", "dependency", "package", "cargo", "npm",
                "pip", "gem", "maven", "gradle", "go.mod", "go.sum",
            ],
            Self::ConfigAudit => &[
                "config", "env", "setting", "secret", "key", "password",
                "credential", "permission", "role", "cors", "csp", "header",
                "tls", "ssl", "certificate", "docker", "compose", "terraform",
                "ansible", "kubernetes", "k8s", "helm",
            ],
            Self::Triager => &[],    // Triager works on findings, not files
            Self::ReportWriter => &[], // Report writer synthesizes
        }
    }

    /// Score relevance of this specialization to a file's content
    pub fn relevance_score(&self, content: &str) -> f64 {
        let keywords = self.keywords();
        if keywords.is_empty() {
            return 0.5; // Triager/ReportWriter: moderate relevance to everything
        }

        let lower = content.to_lowercase();
        let matches = keywords.iter().filter(|kw| lower.contains(**kw)).count();
        let max_possible = keywords.len().min(8) as f64;
        (matches as f64 / max_possible).min(1.0)
    }

    /// Default team composition for a bounty scan
    pub fn default_team() -> Vec<(Self, u32)> {
        vec![
            (Self::Recon, 1),
            (Self::SourceAudit, 3), // Most agents on code audit
            (Self::ExploitAnalyst, 1),
            (Self::DependencyHunter, 1),
            (Self::ConfigAudit, 1),
            (Self::Triager, 1),
            (Self::ReportWriter, 1),
        ]
    }
}

impl std::fmt::Display for BountySpecialization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// System Prompts
// ============================================================================

const RECON_PROMPT: &str = r#"You are a RECON SCOUT in a security bounty hunting team.

Your mission: map the target's attack surface. You are the first agent to touch a target.

Use your tools to:
1. Find all source files and understand the project structure (glob, file_read)
2. Identify API endpoints, routes, and handlers (grep for route/handler patterns)
3. Map exposed ports and services (bash: check configs, Dockerfiles)
4. Identify technology stack and frameworks
5. Find entry points where user input enters the system

Report each finding as a structured signal:
- ENDPOINT: <method> <path> - <description>
- SERVICE: <name> <port/protocol> - <description>
- ENTRY_POINT: <file>:<line> - <description> (where user input enters)
- TECH_STACK: <technology> <version> - <description>

Be thorough but fast. Your findings guide the entire team's work."#;

const SOURCE_AUDIT_PROMPT: &str = r#"You are a SOURCE AUDITOR in a security bounty hunting team.

Your mission: find vulnerabilities in source code. You are the core of the team.

Use your tools to read code and search for vulnerability patterns:
1. SQL/Command/Code Injection: unsanitized input in queries, exec, eval
2. XSS: user input reflected in HTML without encoding
3. Broken Auth: weak session handling, missing access controls, IDOR
4. Crypto Issues: weak algorithms, hardcoded keys, insecure random
5. Deserialization: unsafe pickle/yaml/JSON parsing of untrusted data
6. Path Traversal: user-controlled file paths without sanitization
7. SSRF: user-controlled URLs in server-side requests

For each vulnerability found, report:
- SEVERITY: CRITICAL/HIGH/MEDIUM/LOW/INFO
- TYPE: The vulnerability class (e.g., "SQL Injection")
- FILE: The file path
- LINE: Line number(s) if identifiable
- DESCRIPTION: What the vulnerability is and how it could be exploited
- CONFIDENCE: Your confidence level (0.0-1.0)

Focus on real, exploitable vulnerabilities. Avoid false positives.
Read the actual code before making claims - use file_read and grep."#;

const EXPLOIT_ANALYST_PROMPT: &str = r#"You are an EXPLOIT ANALYST in a security bounty hunting team.

Your mission: validate and assess the exploitability of vulnerabilities found by other agents.

When you receive a vulnerability finding:
1. Read the relevant code to confirm it exists (file_read)
2. Trace the data flow from user input to the vulnerable sink
3. Assess if exploitation requires authentication, special conditions, or chaining
4. Estimate real-world impact (data breach, RCE, privilege escalation, etc.)
5. Determine CVSS-like severity based on exploitability + impact

For each validated vulnerability:
- CONFIRMED: true/false
- EXPLOITABILITY: Easy/Medium/Hard/Theoretical
- IMPACT: What an attacker gains (RCE, data access, DoS, etc.)
- PREREQUISITES: What's needed (auth, specific config, network position)
- CVSS_ESTIMATE: Numeric estimate (0.0-10.0)

Be honest about false positives. A confirmed low-severity finding is more
valuable than an unvalidated critical."#;

const DEPENDENCY_HUNTER_PROMPT: &str = r#"You are a DEPENDENCY HUNTER in a security bounty hunting team.

Your mission: audit the software supply chain for known vulnerabilities.

Use your tools to:
1. Find dependency manifests (glob for Cargo.toml, package.json, requirements.txt, go.mod, etc.)
2. Read dependency files and identify all third-party packages
3. Run audit tools if available (bash: cargo audit, npm audit, pip-audit)
4. Search for known CVEs in dependencies (web_fetch for advisory databases)
5. Flag outdated packages with known security patches

For each finding:
- PACKAGE: name@version
- CVE: CVE ID if known
- SEVERITY: based on CVE severity
- DESCRIPTION: What the vulnerability allows
- FIX: Recommended version upgrade

Also flag:
- Yanked/deprecated packages
- Packages with suspicious characteristics
- Dependency confusion risks (internal package names that could be squatted)"#;

const CONFIG_AUDIT_PROMPT: &str = r#"You are a CONFIG AUDITOR in a security bounty hunting team.

Your mission: find security misconfigurations, exposed secrets, and permission issues.

Use your tools to:
1. Search for hardcoded secrets (grep for API keys, passwords, tokens)
2. Review configuration files for insecure defaults
3. Check TLS/SSL settings
4. Review CORS, CSP, and security headers
5. Audit container configurations (Dockerfiles, docker-compose.yml)
6. Check infrastructure-as-code for misconfigurations
7. Review file permissions and access controls

Common patterns to search:
- Hardcoded: password=, secret=, api_key=, token=, AWS_SECRET
- Defaults: DEBUG=true, admin/admin, allow_all, 0.0.0.0
- Missing: no CSP header, no rate limiting, no HTTPS redirect

For each finding:
- TYPE: Secret Exposure / Misconfiguration / Missing Control
- FILE: File path
- SEVERITY: CRITICAL/HIGH/MEDIUM/LOW
- DESCRIPTION: What's wrong and why it matters
- REMEDIATION: How to fix it"#;

const TRIAGER_PROMPT: &str = r#"You are a TRIAGER in a security bounty hunting team.

Your mission: deduplicate, correlate, and prioritize findings from other agents.

You receive findings from all other agents and must:
1. Identify duplicate findings (same vulnerability reported by multiple agents)
2. Correlate related findings (e.g., SQL injection + database config issue)
3. Prioritize based on severity, exploitability, and impact
4. Identify attack chains (multiple findings that combine into a more severe issue)
5. Filter out false positives and informational noise

For the final triage output:
- Group findings by severity tier (Critical, High, Medium, Low)
- Flag attack chains with combined severity
- Mark duplicates with cross-references
- Provide a prioritized remediation order

Your triage determines what goes into the final report."#;

const REPORT_WRITER_PROMPT: &str = r#"You are a REPORT WRITER in a security bounty hunting team.

Your mission: produce a clear, actionable security report from triaged findings.

Structure the report as:
1. Executive Summary (2-3 sentences: what was tested, key findings, overall risk)
2. Scope (what was scanned, methodology, tools used)
3. Critical Findings (detailed writeup with reproduction steps)
4. High Findings
5. Medium Findings
6. Low / Informational
7. Remediation Roadmap (prioritized fix order)

For each finding:
- Title (e.g., "SQL Injection in User Search Endpoint")
- Severity + CVSS estimate
- Location (file:line)
- Description (what, where, why it matters)
- Impact (what an attacker could do)
- Reproduction Steps (how to trigger it)
- Remediation (how to fix it, with code examples if possible)
- References (CWE, OWASP, relevant advisories)

Write for a mixed audience: executives read the summary,
engineers read the finding details."#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_specializations() {
        let all = BountySpecialization::all();
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn test_default_team() {
        let team = BountySpecialization::default_team();
        let total: u32 = team.iter().map(|(_, count)| count).sum();
        assert_eq!(total, 9); // 1+3+1+1+1+1+1
    }

    #[test]
    fn test_relevance_scoring() {
        let audit = BountySpecialization::SourceAudit;
        let code_with_sql = "fn query(db: &Db, input: &str) { db.exec(sql_query); }";
        let readme = "# My Project\n\nThis is a README file.";

        assert!(audit.relevance_score(code_with_sql) > audit.relevance_score(readme));
    }

    #[test]
    fn test_trust_levels() {
        // Source auditors need minimal trust
        assert_eq!(
            BountySpecialization::SourceAudit.min_trust_level(),
            TrustLevel::ReadOnly
        );
        // Exploit analysts need elevated trust
        assert_eq!(
            BountySpecialization::ExploitAnalyst.min_trust_level(),
            TrustLevel::Elevated
        );
    }
}
