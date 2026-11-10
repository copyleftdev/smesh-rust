//! Tool execution layer - gives bounty hunter agents hands
//!
//! Every tool here is a [`smesh_agent::Tool`], the same trait every mesh uses,
//! holding the [`SandboxConfig`] it runs in. A hunter's [`ToolExecutor`] is a
//! [`Toolbelt`]: its trust level decides which tools join the belt (as it
//! always has) and maps to the belt's clearance, so calls go through the
//! shared gate, timeout and journal. Because they are ordinary tools, any
//! mesh can hand them to its own agents with [`bounty_tools`].

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

pub use smesh_agent::tools::Tool;
use smesh_agent::tools::{CallOutcome, Effect, ToolCall, ToolOutput, Toolbelt};
use smesh_agent::ToolDefinition;

/// Errors from tool execution
#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Timeout after {0}ms")]
    Timeout(u64),
}

/// Trust level gates which tools an agent can use
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrustLevel {
    /// Read-only: FileRead, Grep, Glob
    ReadOnly = 0,
    /// Standard: adds WebFetch, limited Bash (no write)
    Standard = 1,
    /// Elevated: adds Bash with write, FileWrite
    Elevated = 2,
    /// Full: unrestricted tool access
    Full = 3,
}

/// Result of executing a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool name that was executed
    pub tool_name: String,
    /// Whether execution succeeded
    pub success: bool,
    /// Output content
    pub content: String,
    /// Whether this is an error response
    pub is_error: bool,
}

/// Sandbox configuration constraining tool execution
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Root directory tools can access (prevents path traversal)
    pub root_dir: PathBuf,
    /// Maximum output size in bytes
    pub max_output_bytes: usize,
    /// Command timeout in milliseconds
    pub command_timeout_ms: u64,
    /// Blocked commands (for Bash tool)
    pub blocked_commands: Vec<String>,
    /// Blocked file patterns (won't read/write)
    pub blocked_patterns: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("."),
            max_output_bytes: 64 * 1024, // 64KB
            command_timeout_ms: 30_000,  // 30s
            blocked_commands: vec![
                "rm -rf /".into(),
                "mkfs".into(),
                "dd if=/dev".into(),
                ":(){:|:&};:".into(),
                "chmod -R 777".into(),
            ],
            blocked_patterns: vec![
                "**/.env".into(),
                "**/.git/config".into(),
                "**/credentials*".into(),
                "**/*.pem".into(),
                "**/*.key".into(),
                "**/id_rsa*".into(),
            ],
        }
    }
}

impl SandboxConfig {
    /// Create sandbox rooted at a directory
    pub fn rooted(root: impl Into<PathBuf>) -> Self {
        Self {
            root_dir: root.into(),
            ..Default::default()
        }
    }

    /// Check if a path is within the sandbox
    pub fn is_allowed_path(&self, path: &Path) -> bool {
        // Resolve to absolute and check containment
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => return false,
        };
        let root = match self.root_dir.canonicalize() {
            Ok(p) => p,
            Err(_) => return false,
        };

        if !canonical.starts_with(&root) {
            return false;
        }

        // Check blocked patterns
        let path_str = canonical.to_string_lossy();
        for pattern in &self.blocked_patterns {
            if let Ok(glob) = glob::Pattern::new(pattern) {
                if glob.matches(&path_str) {
                    return false;
                }
            }
        }

        true
    }

    /// Check if a command is blocked
    pub fn is_blocked_command(&self, cmd: &str) -> bool {
        let lower = cmd.to_lowercase();
        self.blocked_commands
            .iter()
            .any(|blocked| lower.contains(&blocked.to_lowercase()))
    }
}

// ============================================================================
// Tool Implementations
// ============================================================================

/// Implement the shared [`Tool`] trait for a bounty tool whose inherent
/// `tool_name`, `about`, `schema` and `run` hold its behaviour.
macro_rules! shared_tool {
    ($t:ty, $effect:expr) => {
        #[async_trait]
        impl Tool for $t {
            fn name(&self) -> &str {
                self.tool_name()
            }
            fn description(&self) -> &str {
                self.about()
            }
            fn input_schema(&self) -> Value {
                self.schema()
            }
            fn effect(&self) -> Effect {
                $effect
            }
            async fn call(
                &self,
                input: Value,
            ) -> Result<ToolOutput, smesh_agent::tools::ToolError> {
                let result = self.run(input, &self.sandbox).await.map_err(|e| match e {
                    ToolError::InvalidInput(m) => smesh_agent::tools::ToolError::InvalidInput(m),
                    other => smesh_agent::tools::ToolError::Failed(other.to_string()),
                })?;
                Ok(if result.is_error {
                    ToolOutput::error(result.content)
                } else {
                    ToolOutput::text(result.content)
                })
            }
        }
    };
}

/// Read a file's contents
pub struct FileReadTool {
    sandbox: Arc<SandboxConfig>,
}

impl FileReadTool {
    /// This tool, confined to `sandbox`.
    pub fn new(sandbox: Arc<SandboxConfig>) -> Self {
        Self { sandbox }
    }
}

shared_tool!(FileReadTool, Effect::Read);

impl FileReadTool {
    fn tool_name(&self) -> &str {
        "file_read"
    }

    fn about(&self) -> &str {
        "Read a file's contents. Supports offset and limit for large files."
    }

    fn min_trust_level() -> TrustLevel {
        TrustLevel::ReadOnly
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read (relative to scan root)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum lines to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn run(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError> {
        let path_str = input["path"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("path is required".into()))?;

        let full_path = sandbox.root_dir.join(path_str);

        if !sandbox.is_allowed_path(&full_path) {
            return Err(ToolError::PermissionDenied(format!(
                "Path outside sandbox: {}",
                path_str
            )));
        }

        let content = tokio::fs::read_to_string(&full_path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to read {}: {}", path_str, e))
        })?;

        let offset = input["offset"].as_u64().unwrap_or(0) as usize;
        let limit = input["limit"].as_u64().unwrap_or(500) as usize;

        let lines: Vec<&str> = content.lines().skip(offset).take(limit).collect();
        let numbered: String = lines
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>4}\t{}", offset + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

        let output = if numbered.len() > sandbox.max_output_bytes {
            format!(
                "{}...\n[truncated at {} bytes]",
                &numbered[..sandbox.max_output_bytes],
                sandbox.max_output_bytes
            )
        } else {
            numbered
        };

        Ok(ToolResult {
            tool_name: self.tool_name().into(),
            success: true,
            content: output,
            is_error: false,
        })
    }
}

/// Search file contents with regex
pub struct GrepTool {
    sandbox: Arc<SandboxConfig>,
}

impl GrepTool {
    /// This tool, confined to `sandbox`.
    pub fn new(sandbox: Arc<SandboxConfig>) -> Self {
        Self { sandbox }
    }
}

shared_tool!(GrepTool, Effect::Read);

impl GrepTool {
    fn tool_name(&self) -> &str {
        "grep"
    }

    fn about(&self) -> &str {
        "Search file contents using regex. Returns matching lines with file paths and line numbers."
    }

    fn min_trust_level() -> TrustLevel {
        TrustLevel::ReadOnly
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (relative to scan root)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.rs', '*.py')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 50)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn run(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError> {
        let pattern_str = input["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("pattern is required".into()))?;

        let re = regex::Regex::new(pattern_str)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid regex: {}", e)))?;

        let search_path = if let Some(p) = input["path"].as_str() {
            sandbox.root_dir.join(p)
        } else {
            sandbox.root_dir.clone()
        };

        let glob_pattern = input["glob"].as_str();
        let max_results = input["max_results"].as_u64().unwrap_or(50) as usize;

        let mut matches = Vec::new();

        for entry in walkdir::WalkDir::new(&search_path)
            .max_depth(10)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                !name.starts_with('.')
                    && name != "target"
                    && name != "node_modules"
                    && name != "vendor"
            })
        {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();

            // Apply glob filter
            if let Some(glob_pat) = glob_pattern {
                if let Ok(g) = glob::Pattern::new(glob_pat) {
                    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
                    if !g.matches(&file_name) {
                        continue;
                    }
                }
            }

            if !sandbox.is_allowed_path(path) {
                continue;
            }

            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(_) => continue, // Skip binary/unreadable files
            };

            let relative = path
                .strip_prefix(&sandbox.root_dir)
                .unwrap_or(path)
                .display()
                .to_string();

            for (line_num, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    matches.push(format!("{}:{}:{}", relative, line_num + 1, line.trim()));
                    if matches.len() >= max_results {
                        break;
                    }
                }
            }

            if matches.len() >= max_results {
                break;
            }
        }

        let output = if matches.is_empty() {
            "No matches found.".to_string()
        } else {
            format!("{} matches:\n{}", matches.len(), matches.join("\n"))
        };

        Ok(ToolResult {
            tool_name: self.tool_name().into(),
            success: true,
            content: output,
            is_error: false,
        })
    }
}

/// Find files by glob pattern
pub struct GlobTool {
    sandbox: Arc<SandboxConfig>,
}

impl GlobTool {
    /// This tool, confined to `sandbox`.
    pub fn new(sandbox: Arc<SandboxConfig>) -> Self {
        Self { sandbox }
    }
}

shared_tool!(GlobTool, Effect::Read);

impl GlobTool {
    fn tool_name(&self) -> &str {
        "glob"
    }

    fn about(&self) -> &str {
        "Find files matching a glob pattern. Returns file paths sorted by modification time."
    }

    fn min_trust_level() -> TrustLevel {
        TrustLevel::ReadOnly
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '**/*.rs', 'src/**/*.py')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (relative to scan root)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn run(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError> {
        let pattern = input["pattern"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("pattern is required".into()))?;

        let base = if let Some(p) = input["path"].as_str() {
            sandbox.root_dir.join(p)
        } else {
            sandbox.root_dir.clone()
        };

        let full_pattern = base.join(pattern).to_string_lossy().to_string();

        let mut files: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();

        let matches = glob::glob(&full_pattern)
            .map_err(|e| ToolError::InvalidInput(format!("Invalid glob pattern: {}", e)))?;
        for path in matches.flatten() {
            if path.is_file() && sandbox.is_allowed_path(&path) {
                let mtime = std::fs::metadata(&path)
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                files.push((path, mtime));
            }
        }

        // Sort by modification time (newest first)
        files.sort_by_key(|f| std::cmp::Reverse(f.1));

        let output: Vec<String> = files
            .iter()
            .take(100)
            .map(|(p, _)| {
                p.strip_prefix(&sandbox.root_dir)
                    .unwrap_or(p)
                    .display()
                    .to_string()
            })
            .collect();

        let result = if output.is_empty() {
            "No files found.".to_string()
        } else {
            format!("{} files:\n{}", output.len(), output.join("\n"))
        };

        Ok(ToolResult {
            tool_name: self.tool_name().into(),
            success: true,
            content: result,
            is_error: false,
        })
    }
}

/// Execute shell commands (sandboxed)
pub struct BashTool {
    sandbox: Arc<SandboxConfig>,
}

impl BashTool {
    /// This tool, confined to `sandbox`.
    pub fn new(sandbox: Arc<SandboxConfig>) -> Self {
        Self { sandbox }
    }
}

shared_tool!(BashTool, Effect::Execute);

impl BashTool {
    fn tool_name(&self) -> &str {
        "bash"
    }

    fn about(&self) -> &str {
        "Execute a shell command. Use for running security tools (nmap, nikto, cargo-audit, npm audit, etc.) and system commands."
    }

    fn min_trust_level() -> TrustLevel {
        TrustLevel::Standard
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 30000)"
                }
            },
            "required": ["command"]
        })
    }

    async fn run(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError> {
        let command = input["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("command is required".into()))?;

        if sandbox.is_blocked_command(command) {
            return Err(ToolError::PermissionDenied(format!(
                "Command blocked by sandbox: {}",
                command
            )));
        }

        let timeout_ms = input["timeout_ms"]
            .as_u64()
            .unwrap_or(sandbox.command_timeout_ms);

        let output = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            tokio::process::Command::new("bash")
                .arg("-c")
                .arg(command)
                .current_dir(&sandbox.root_dir)
                .output(),
        )
        .await
        .map_err(|_| ToolError::Timeout(timeout_ms))?
        .map_err(|e| ToolError::ExecutionFailed(format!("Command failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut content = String::new();
        if !stdout.is_empty() {
            content.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !content.is_empty() {
                content.push_str("\nSTDERR:\n");
            }
            content.push_str(&stderr);
        }
        if content.is_empty() {
            content = format!("(exit code: {})", output.status.code().unwrap_or(-1));
        }

        // Truncate if too large
        if content.len() > sandbox.max_output_bytes {
            content = format!(
                "{}...\n[truncated at {} bytes]",
                &content[..sandbox.max_output_bytes],
                sandbox.max_output_bytes
            );
        }

        Ok(ToolResult {
            tool_name: self.tool_name().into(),
            success: output.status.success(),
            content,
            is_error: !output.status.success(),
        })
    }
}

/// Fetch a URL (for checking endpoints, downloading CVE data, etc.)
pub struct WebFetchTool {
    sandbox: Arc<SandboxConfig>,
}

impl WebFetchTool {
    /// This tool, confined to `sandbox`.
    pub fn new(sandbox: Arc<SandboxConfig>) -> Self {
        Self { sandbox }
    }
}

shared_tool!(WebFetchTool, Effect::Read);

impl WebFetchTool {
    fn tool_name(&self) -> &str {
        "web_fetch"
    }

    fn about(&self) -> &str {
        "Fetch content from a URL. Useful for checking endpoints, downloading CVE references, and verifying external resources."
    }

    fn min_trust_level() -> TrustLevel {
        TrustLevel::Standard
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (GET, HEAD)",
                    "enum": ["GET", "HEAD"]
                },
                "headers": {
                    "type": "object",
                    "description": "Additional HTTP headers"
                }
            },
            "required": ["url"]
        })
    }

    async fn run(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError> {
        let url = input["url"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidInput("url is required".into()))?;

        // Block internal/metadata endpoints
        let lower_url = url.to_lowercase();
        if lower_url.contains("169.254.169.254")
            || lower_url.contains("metadata.google")
            || lower_url.starts_with("file://")
        {
            return Err(ToolError::PermissionDenied(
                "Blocked: internal/metadata endpoint".into(),
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(sandbox.command_timeout_ms))
            .redirect(reqwest::redirect::Policy::limited(3))
            .build()
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let method = input["method"].as_str().unwrap_or("GET");
        let request = match method {
            "HEAD" => client.head(url),
            _ => client.get(url),
        };

        let resp = request
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Fetch failed: {}", e)))?;

        let status = resp.status();
        let headers: Vec<String> = resp
            .headers()
            .iter()
            .take(20)
            .map(|(k, v)| format!("{}: {}", k, v.to_str().unwrap_or("?")))
            .collect();

        let body = resp
            .text()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Read body failed: {}", e)))?;

        let truncated = if body.len() > sandbox.max_output_bytes {
            format!(
                "{}...\n[truncated at {} bytes]",
                &body[..sandbox.max_output_bytes],
                sandbox.max_output_bytes
            )
        } else {
            body
        };

        let content = format!(
            "HTTP {} {}\n{}\n\n{}",
            status.as_u16(),
            status.canonical_reason().unwrap_or(""),
            headers.join("\n"),
            truncated
        );

        Ok(ToolResult {
            tool_name: self.tool_name().into(),
            success: status.is_success(),
            content,
            is_error: status.is_server_error(),
        })
    }
}

// ============================================================================
// Tool Executor
// ============================================================================

/// The clearance a trust level grants on the shared toolbelt.
impl TrustLevel {
    /// The most this trust level may do. Which tools an agent holds is still
    /// decided per tool by [`bounty_tools`]; this is the gate behind it.
    pub fn clearance(self) -> Effect {
        match self {
            TrustLevel::ReadOnly => Effect::Read,
            TrustLevel::Standard | TrustLevel::Elevated | TrustLevel::Full => Effect::Execute,
        }
    }
}

/// Every bounty tool confined to `sandbox`, with the trust level it needs.
pub fn bounty_tools(sandbox: Arc<SandboxConfig>) -> Vec<(Arc<dyn Tool>, TrustLevel)> {
    vec![
        (
            Arc::new(FileReadTool::new(Arc::clone(&sandbox))),
            FileReadTool::min_trust_level(),
        ),
        (
            Arc::new(GrepTool::new(Arc::clone(&sandbox))),
            GrepTool::min_trust_level(),
        ),
        (
            Arc::new(GlobTool::new(Arc::clone(&sandbox))),
            GlobTool::min_trust_level(),
        ),
        (
            Arc::new(BashTool::new(Arc::clone(&sandbox))),
            BashTool::min_trust_level(),
        ),
        (
            Arc::new(WebFetchTool::new(sandbox)),
            WebFetchTool::min_trust_level(),
        ),
    ]
}

/// A hunter's tools: a shared [`Toolbelt`] holding the bounty tools its trust
/// level allows.
pub struct ToolExecutor {
    belt: Toolbelt,
    sandbox: Arc<SandboxConfig>,
    trust_level: TrustLevel,
}

impl ToolExecutor {
    /// The tools `trust_level` allows, confined to `sandbox`.
    pub fn new(sandbox: SandboxConfig, trust_level: TrustLevel) -> Self {
        let sandbox = Arc::new(sandbox);
        let mut executor = Self {
            belt: Toolbelt::new(trust_level.clearance()).with_timeout(
                std::time::Duration::from_millis(sandbox.command_timeout_ms.saturating_add(5_000)),
            ),
            sandbox,
            trust_level,
        };
        executor.add_allowed();
        executor
    }

    /// Add every tool the current trust level allows that is not yet held.
    fn add_allowed(&mut self) {
        for (tool, min) in bounty_tools(Arc::clone(&self.sandbox)) {
            if min <= self.trust_level && !self.has_tool(tool.name()) {
                // Names are fixed and distinct, so adding cannot fail.
                let _ = self.belt.add(tool);
            }
        }
    }

    /// Get tool definitions for LLM (only tools this executor can access)
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.belt.definitions()
    }

    /// Execute a tool by name, through the toolbelt's gate and journal.
    pub async fn execute(&self, tool_name: &str, input: Value) -> Result<ToolResult, ToolError> {
        let call = self.belt.call(tool_name, input).await;
        match call.outcome {
            CallOutcome::Refused if call.effect.is_none() => {
                Err(ToolError::NotFound(tool_name.into()))
            }
            CallOutcome::Refused => Err(ToolError::PermissionDenied(call.output)),
            CallOutcome::Error => Ok(ToolResult {
                tool_name: tool_name.into(),
                success: false,
                content: call.output,
                is_error: true,
            }),
            CallOutcome::Ok => Ok(ToolResult {
                tool_name: tool_name.into(),
                success: true,
                content: call.output,
                is_error: false,
            }),
        }
    }

    /// Check if a tool is available
    pub fn has_tool(&self, name: &str) -> bool {
        self.belt.tools().iter().any(|t| t.name() == name)
    }

    /// Get current trust level
    pub fn trust_level(&self) -> TrustLevel {
        self.trust_level
    }

    /// Every call made so far.
    pub fn journal(&self) -> Vec<ToolCall> {
        self.belt.journal()
    }

    /// The underlying toolbelt, to hand these tools to a mesh agent.
    pub fn toolbelt(&self) -> &Toolbelt {
        &self.belt
    }

    /// Upgrade trust level (enables more tools). The journal is kept.
    pub fn upgrade_trust(&mut self, new_level: TrustLevel) {
        if new_level > self.trust_level {
            self.trust_level = new_level;
            self.belt.set_clearance(new_level.clearance());
            self.add_allowed();
        }
    }
}

impl std::fmt::Debug for ToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutor")
            .field("trust_level", &self.trust_level)
            .field("toolbelt", &self.belt)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::ReadOnly < TrustLevel::Standard);
        assert!(TrustLevel::Standard < TrustLevel::Elevated);
        assert!(TrustLevel::Elevated < TrustLevel::Full);
    }

    #[test]
    fn test_tool_gating() {
        let sandbox = SandboxConfig::rooted("/tmp/test");
        let executor = ToolExecutor::new(sandbox, TrustLevel::ReadOnly);

        // Read-only should have file_read, grep, glob
        assert!(executor.has_tool("file_read"));
        assert!(executor.has_tool("grep"));
        assert!(executor.has_tool("glob"));

        // But not bash or web_fetch
        assert!(!executor.has_tool("bash"));
        assert!(!executor.has_tool("web_fetch"));
    }

    #[test]
    fn test_standard_trust_tools() {
        let sandbox = SandboxConfig::rooted("/tmp/test");
        let executor = ToolExecutor::new(sandbox, TrustLevel::Standard);

        assert!(executor.has_tool("file_read"));
        assert!(executor.has_tool("bash"));
        assert!(executor.has_tool("web_fetch"));
    }

    #[test]
    fn test_sandbox_blocked_commands() {
        let sandbox = SandboxConfig::default();
        assert!(sandbox.is_blocked_command("rm -rf /"));
        assert!(sandbox.is_blocked_command("sudo mkfs.ext4 /dev/sda"));
        assert!(!sandbox.is_blocked_command("cargo audit"));
        assert!(!sandbox.is_blocked_command("grep -r password ."));
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("smesh-bounty-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("app.py"), "password = 'hunter2'\nprint('ok')\n").unwrap();
        dir
    }

    #[test]
    fn bounty_tools_declare_what_they_can_do() {
        let effects: Vec<(String, Effect)> = bounty_tools(Arc::new(SandboxConfig::default()))
            .into_iter()
            .map(|(t, _)| (t.name().to_string(), t.effect()))
            .collect();
        assert!(effects.contains(&("file_read".into(), Effect::Read)));
        assert!(effects.contains(&("web_fetch".into(), Effect::Read)));
        assert!(effects.contains(&("bash".into(), Effect::Execute)));
        assert_eq!(TrustLevel::ReadOnly.clearance(), Effect::Read);
        assert_eq!(TrustLevel::Standard.clearance(), Effect::Execute);
    }

    #[tokio::test]
    async fn calls_run_through_the_shared_toolbelt_and_are_journaled() {
        let dir = scratch("journal");
        let executor = ToolExecutor::new(SandboxConfig::rooted(&dir), TrustLevel::ReadOnly);

        let read = executor
            .execute("file_read", serde_json::json!({ "path": "app.py" }))
            .await
            .unwrap();
        assert!(
            read.success && read.content.contains("hunter2"),
            "{}",
            read.content
        );

        let grep = executor
            .execute("grep", serde_json::json!({ "pattern": "password" }))
            .await
            .unwrap();
        assert!(grep.content.contains("app.py:1:"), "{}", grep.content);

        // The sandbox still confines paths; its refusal is now a tool error.
        let escape = executor
            .execute(
                "file_read",
                serde_json::json!({ "path": "../../etc/passwd" }),
            )
            .await
            .unwrap();
        assert!(
            escape.is_error && escape.content.contains("outside sandbox"),
            "{}",
            escape.content
        );

        // A tool the trust level never granted is unknown to this agent.
        assert!(matches!(
            executor
                .execute("bash", serde_json::json!({ "command": "id" }))
                .await,
            Err(ToolError::NotFound(_))
        ));

        let journal = executor.journal();
        assert_eq!(journal.len(), 4);
        assert_eq!(journal[0].outcome, CallOutcome::Ok);
        assert_eq!(journal[2].outcome, CallOutcome::Error);
        assert_eq!(journal[3].outcome, CallOutcome::Refused);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn earning_trust_adds_tools_and_keeps_the_journal() {
        let dir = scratch("upgrade");
        let mut executor = ToolExecutor::new(SandboxConfig::rooted(&dir), TrustLevel::ReadOnly);
        executor
            .execute("glob", serde_json::json!({ "pattern": "*.py" }))
            .await
            .unwrap();
        executor.upgrade_trust(TrustLevel::Standard);
        assert!(executor.has_tool("bash"));
        let ran = executor
            .execute("bash", serde_json::json!({ "command": "echo hi" }))
            .await
            .unwrap();
        assert_eq!(ran.content.trim(), "hi");
        let blocked = executor
            .execute("bash", serde_json::json!({ "command": "rm -rf /" }))
            .await
            .unwrap();
        assert!(
            blocked.is_error && blocked.content.contains("blocked"),
            "{}",
            blocked.content
        );
        assert_eq!(
            executor.journal().len(),
            3,
            "the journal survives the upgrade"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn any_mesh_agent_can_hold_bounty_tools_under_its_own_clearance() {
        let dir = scratch("mesh");
        let mut belt = Toolbelt::new(Effect::Read);
        for (tool, _) in bounty_tools(Arc::new(SandboxConfig::rooted(&dir))) {
            belt.add(tool).unwrap();
        }
        let offered: Vec<String> = belt.definitions().into_iter().map(|d| d.name).collect();
        assert!(offered.contains(&"grep".to_string()));
        assert!(
            !offered.contains(&"bash".to_string()),
            "execute is above read clearance"
        );
        let refused = belt
            .call("bash", serde_json::json!({ "command": "id" }))
            .await;
        assert_eq!(refused.outcome, CallOutcome::Refused);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tool_definitions() {
        let sandbox = SandboxConfig::rooted("/tmp/test");
        let executor = ToolExecutor::new(sandbox, TrustLevel::Full);

        let defs = executor.tool_definitions();
        assert_eq!(defs.len(), 5);

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"file_read"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"bash"));
    }
}
