//! Vulnerability scanning agents with specializations

use smesh_agent::{ClaudeClient, GenerateRequest, LlmBackend};
use smesh_core::Node;

/// Vulnerability specialization types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VulnSpecialization {
    /// SQL, Command, LDAP, XPath injection
    Injection,
    /// Cross-Site Scripting
    Xss,
    /// Authentication and authorization issues
    Auth,
    /// Cryptographic weaknesses
    Crypto,
    /// Insecure deserialization
    Deserialization,
    /// Path/directory traversal
    PathTraversal,
    /// Server-Side Request Forgery
    Ssrf,
}

impl VulnSpecialization {
    /// Get all specializations
    pub fn all() -> Vec<Self> {
        vec![
            Self::Injection,
            Self::Xss,
            Self::Auth,
            Self::Crypto,
            Self::Deserialization,
            Self::PathTraversal,
            Self::Ssrf,
        ]
    }

    /// Display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Injection => "Injection",
            Self::Xss => "XSS",
            Self::Auth => "Auth",
            Self::Crypto => "Crypto",
            Self::Deserialization => "Deserialization",
            Self::PathTraversal => "PathTraversal",
            Self::Ssrf => "SSRF",
        }
    }

    /// Short code for agent ID
    pub fn code(&self) -> &'static str {
        match self {
            Self::Injection => "INJECT",
            Self::Xss => "XSS",
            Self::Auth => "AUTH",
            Self::Crypto => "CRYPTO",
            Self::Deserialization => "DESER",
            Self::PathTraversal => "PATH",
            Self::Ssrf => "SSRF",
        }
    }

    /// System prompt for this specialization
    pub fn system_prompt(&self) -> &'static str {
        match self {
            Self::Injection => INJECTION_PROMPT,
            Self::Xss => XSS_PROMPT,
            Self::Auth => AUTH_PROMPT,
            Self::Crypto => CRYPTO_PROMPT,
            Self::Deserialization => DESER_PROMPT,
            Self::PathTraversal => PATH_PROMPT,
            Self::Ssrf => SSRF_PROMPT,
        }
    }

    /// Keywords relevant to this specialization (for file relevance scoring)
    pub fn keywords(&self) -> &[&'static str] {
        match self {
            Self::Injection => &[
                "sql",
                "query",
                "execute",
                "database",
                "db",
                "cursor",
                "statement",
                "command",
                "shell",
                "exec",
                "system",
                "popen",
                "subprocess",
                "ldap",
                "xpath",
                "eval",
                "compile",
            ],
            Self::Xss => &[
                "html",
                "template",
                "render",
                "response",
                "output",
                "write",
                "innerhtml",
                "document",
                "dom",
                "script",
                "sanitize",
                "escape",
                "encode",
                "innerHTML",
                "outerHTML",
            ],
            Self::Auth => &[
                "auth",
                "login",
                "password",
                "session",
                "token",
                "jwt",
                "oauth",
                "credential",
                "user",
                "permission",
                "role",
                "access",
                "verify",
                "authenticate",
                "authorize",
                "cookie",
                "bearer",
            ],
            Self::Crypto => &[
                "crypto", "encrypt", "decrypt", "hash", "sign", "verify", "key", "cipher", "aes",
                "rsa", "hmac", "secret", "salt", "iv", "random", "md5", "sha1", "sha256", "bcrypt",
                "argon",
            ],
            Self::Deserialization => &[
                "serial",
                "deserial",
                "pickle",
                "yaml",
                "json",
                "marshal",
                "unmarshal",
                "parse",
                "load",
                "loads",
                "decode",
                "fromjson",
                "objectinputstream",
                "readobject",
            ],
            Self::PathTraversal => &[
                "path",
                "file",
                "directory",
                "read",
                "write",
                "open",
                "load",
                "include",
                "require",
                "import",
                "fs",
                "io",
                "fopen",
                "readfile",
                "basename",
                "dirname",
                "realpath",
            ],
            Self::Ssrf => &[
                "url", "http", "https", "request", "fetch", "get", "post", "curl", "wget",
                "urllib", "requests", "client", "connect", "socket", "redirect", "proxy",
            ],
        }
    }

    /// Score relevance of this specialization to a file's content
    pub fn relevance_score(&self, content: &str) -> f64 {
        let lower = content.to_lowercase();
        let keywords = self.keywords();

        let mut matches = 0;
        for keyword in keywords {
            if lower.contains(keyword) {
                matches += 1;
            }
        }

        // Normalize to 0.0 - 1.0
        let max_possible = keywords.len().min(10) as f64;
        (matches as f64 / max_possible).min(1.0)
    }
}

impl std::fmt::Display for VulnSpecialization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// System prompts for each specialization
const INJECTION_PROMPT: &str = r#"You are a security expert specializing in injection vulnerabilities.

Analyze code for:
- SQL Injection: string concatenation in queries, unsanitized user input in SQL
- Command Injection: shell commands with user input, unsafe exec/system calls
- LDAP Injection: unsanitized input in LDAP queries
- XPath Injection: user input in XPath expressions
- Code Injection: eval(), exec() with untrusted input

Report findings in this format:
[SEVERITY] VulnType: Description (line X if identifiable)

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO
If no vulnerabilities found, respond: "No injection vulnerabilities found."
Be concise and specific. Focus only on injection issues."#;

const XSS_PROMPT: &str = r#"You are a security expert specializing in Cross-Site Scripting (XSS).

Analyze code for:
- Reflected XSS: user input echoed directly in responses
- Stored XSS: unsanitized data from storage rendered in HTML
- DOM XSS: client-side manipulation without encoding
- Missing output encoding/escaping
- Unsafe innerHTML usage
- Template injection

Report findings in this format:
[SEVERITY] VulnType: Description (line X if identifiable)

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO
If no vulnerabilities found, respond: "No XSS vulnerabilities found."
Be concise and specific. Focus only on XSS issues."#;

const AUTH_PROMPT: &str = r#"You are a security expert specializing in authentication and authorization.

Analyze code for:
- Broken Authentication: weak password policies, missing rate limiting
- Broken Access Control: missing authorization checks, IDOR
- Session Management: insecure session handling, fixation
- Credential Storage: plaintext passwords, weak hashing
- JWT Issues: weak algorithms, missing validation
- Hardcoded Credentials: secrets in source code

Report findings in this format:
[SEVERITY] VulnType: Description (line X if identifiable)

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO
If no vulnerabilities found, respond: "No authentication/authorization vulnerabilities found."
Be concise and specific. Focus only on auth issues."#;

const CRYPTO_PROMPT: &str = r#"You are a security expert specializing in cryptographic issues.

Analyze code for:
- Weak Algorithms: MD5, SHA1 for passwords, DES, RC4
- Hardcoded Keys: encryption keys in source code
- Insufficient Key Length: RSA < 2048, AES < 128
- Insecure Random: Math.random(), rand() for security
- Missing Encryption: sensitive data unencrypted
- IV Reuse: static or predictable IVs
- ECB Mode: using ECB for block ciphers

Report findings in this format:
[SEVERITY] VulnType: Description (line X if identifiable)

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO
If no vulnerabilities found, respond: "No cryptographic vulnerabilities found."
Be concise and specific. Focus only on crypto issues."#;

const DESER_PROMPT: &str = r#"You are a security expert specializing in deserialization vulnerabilities.

Analyze code for:
- Unsafe Deserialization: pickle, yaml.load, ObjectInputStream
- Type Confusion: missing type validation after deserialization
- Gadget Chains: exploitable class structures
- XML External Entities (XXE): when parsing XML
- JSON Injection: unsafe JSON parsing

Report findings in this format:
[SEVERITY] VulnType: Description (line X if identifiable)

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO
If no vulnerabilities found, respond: "No deserialization vulnerabilities found."
Be concise and specific. Focus only on deserialization issues."#;

const PATH_PROMPT: &str = r#"You are a security expert specializing in path traversal vulnerabilities.

Analyze code for:
- Directory Traversal: ../ sequences not filtered
- Arbitrary File Read: user-controlled file paths
- Arbitrary File Write: writing to user-controlled paths
- Path Manipulation: unsanitized path components
- Symlink Attacks: following symbolic links unsafely
- Missing Chroot/Sandboxing

Report findings in this format:
[SEVERITY] VulnType: Description (line X if identifiable)

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO
If no vulnerabilities found, respond: "No path traversal vulnerabilities found."
Be concise and specific. Focus only on path traversal issues."#;

const SSRF_PROMPT: &str = r#"You are a security expert specializing in SSRF vulnerabilities.

Analyze code for:
- Server-Side Request Forgery: user-controlled URLs in server requests
- Open Redirect: unvalidated redirect URLs
- DNS Rebinding: missing DNS pinning
- Protocol Smuggling: switching protocols via user input
- Cloud Metadata Access: requests to 169.254.169.254

Report findings in this format:
[SEVERITY] VulnType: Description (line X if identifiable)

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO
If no vulnerabilities found, respond: "No SSRF vulnerabilities found."
Be concise and specific. Focus only on SSRF issues."#;

/// Per-agent metrics
#[derive(Debug, Clone, Default)]
pub struct AgentMetrics {
    /// Number of files analyzed
    pub files_analyzed: u32,
    /// Number of findings emitted
    pub findings_emitted: u32,
    /// Number of reinforcements made
    pub reinforcements_made: u32,
    /// Total input tokens used
    pub input_tokens: u32,
    /// Total output tokens used
    pub output_tokens: u32,
    /// Total API calls made
    pub api_calls: u32,
}

/// A vulnerability scanning agent
#[derive(Debug)]
pub struct VulnAgent {
    /// Agent identifier (e.g., "INJECT-1")
    pub id: String,
    /// Specialization
    pub specialization: VulnSpecialization,
    /// SMESH node for signal emission
    pub node: Node,
    /// Claude client for analysis
    client: ClaudeClient,
    /// Agent metrics
    pub metrics: AgentMetrics,
}

impl VulnAgent {
    /// Create a new vulnerability agent
    pub fn new(specialization: VulnSpecialization, instance: u32, client: ClaudeClient) -> Self {
        let id = format!("{}-{}", specialization.code(), instance);
        let mut node = Node::new();
        node.id = id.clone();

        Self {
            id,
            specialization,
            node,
            client,
            metrics: AgentMetrics::default(),
        }
    }

    /// Get the system prompt for this agent
    pub fn system_prompt(&self) -> &'static str {
        self.specialization.system_prompt()
    }

    /// Score relevance of this agent to analyze a file
    pub fn relevance_to_file(&self, content: &str) -> f64 {
        self.specialization.relevance_score(content)
    }

    /// Analyze code for vulnerabilities
    pub async fn analyze(&mut self, file_path: &str, code: &str) -> Result<String, String> {
        let prompt = format!(
            "Analyze this code from file '{}' for {} vulnerabilities:\n\n```\n{}\n```",
            file_path,
            self.specialization.name(),
            code
        );

        let request = GenerateRequest::new(&prompt)
            .with_system(self.system_prompt())
            .with_temperature(0.3) // Lower temperature for more focused analysis
            .with_max_tokens(1024);

        self.metrics.api_calls += 1;
        self.metrics.files_analyzed += 1;

        match self.client.generate(request).await {
            Ok(response) => {
                if let Some(input) = response.input_tokens {
                    self.metrics.input_tokens += input;
                }
                if let Some(output) = response.output_tokens {
                    self.metrics.output_tokens += output;
                }
                Ok(response.content)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Record that this agent emitted a finding
    pub fn record_finding(&mut self) {
        self.metrics.findings_emitted += 1;
    }

    /// Record that this agent reinforced a finding
    pub fn record_reinforcement(&mut self) {
        self.metrics.reinforcements_made += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specialization_keywords() {
        let spec = VulnSpecialization::Injection;
        assert!(spec.keywords().contains(&"sql"));
        assert!(spec.keywords().contains(&"exec"));
    }

    #[test]
    fn test_relevance_scoring() {
        let spec = VulnSpecialization::Injection;

        // Code with SQL/database keywords should score higher
        let sql_code = "fn query_database(db: &Connection) { db.execute(sql); }";
        let other_code = "function render() { return div; }";

        let sql_score = spec.relevance_score(sql_code);
        let other_score = spec.relevance_score(other_code);

        assert!(
            sql_score > other_score,
            "sql_score={}, other_score={}",
            sql_score,
            other_score
        );
    }

    #[test]
    fn test_all_specializations() {
        let all = VulnSpecialization::all();
        assert_eq!(all.len(), 7);
    }
}
