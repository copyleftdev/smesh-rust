//! Tool execution layer - gives bounty hunter agents hands
//!
//! Inspired by Claude Code's tool dispatch system, each tool has:
//! - A name and description (for LLM tool-use)
//! - A JSON schema for input parameters
//! - An async execute method that returns structured output
//!
//! Tools are gated by trust level: low-trust agents get read-only tools,
//! high-trust agents get execution tools.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

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

/// Trait for executable tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (must match LLM tool_use name)
    fn name(&self) -> &str;

    /// Human-readable description
    fn description(&self) -> &str;

    /// Minimum trust level required
    fn min_trust_level(&self) -> TrustLevel;

    /// JSON Schema for input parameters
    fn input_schema(&self) -> Value;

    /// Execute the tool with given input
    async fn execute(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError>;

    /// Convert to LLM ToolDefinition
    fn to_definition(&self) -> ToolDefinition {
        ToolDefinition::new(self.name(), self.description(), self.input_schema())
    }
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
            command_timeout_ms: 30_000,   // 30s
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

/// Read a file's contents
pub struct FileReadTool;

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "file_read"
    }

    fn description(&self) -> &str {
        "Read a file's contents. Supports offset and limit for large files."
    }

    fn min_trust_level(&self) -> TrustLevel {
        TrustLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
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

    async fn execute(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError> {
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

        let content = tokio::fs::read_to_string(&full_path)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read {}: {}", path_str, e)))?;

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
            tool_name: self.name().into(),
            success: true,
            content: output,
            is_error: false,
        })
    }
}

/// Search file contents with regex
pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents using regex. Returns matching lines with file paths and line numbers."
    }

    fn min_trust_level(&self) -> TrustLevel {
        TrustLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
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

    async fn execute(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError> {
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
            tool_name: self.name().into(),
            success: true,
            content: output,
            is_error: false,
        })
    }
}

/// Find files by glob pattern
pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Returns file paths sorted by modification time."
    }

    fn min_trust_level(&self) -> TrustLevel {
        TrustLevel::ReadOnly
    }

    fn input_schema(&self) -> Value {
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

    async fn execute(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError> {
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

        for entry in glob::glob(&full_pattern).map_err(|e| {
            ToolError::InvalidInput(format!("Invalid glob pattern: {}", e))
        })? {
            if let Ok(path) = entry {
                if path.is_file() && sandbox.is_allowed_path(&path) {
                    let mtime = std::fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    files.push((path, mtime));
                }
            }
        }

        // Sort by modification time (newest first)
        files.sort_by(|a, b| b.1.cmp(&a.1));

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
            tool_name: self.name().into(),
            success: true,
            content: result,
            is_error: false,
        })
    }
}

/// Execute shell commands (sandboxed)
pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command. Use for running security tools (nmap, nikto, cargo-audit, npm audit, etc.) and system commands."
    }

    fn min_trust_level(&self) -> TrustLevel {
        TrustLevel::Standard
    }

    fn input_schema(&self) -> Value {
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

    async fn execute(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError> {
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
            tool_name: self.name().into(),
            success: output.status.success(),
            content,
            is_error: !output.status.success(),
        })
    }
}

/// Fetch a URL (for checking endpoints, downloading CVE data, etc.)
pub struct WebFetchTool;

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch content from a URL. Useful for checking endpoints, downloading CVE references, and verifying external resources."
    }

    fn min_trust_level(&self) -> TrustLevel {
        TrustLevel::Standard
    }

    fn input_schema(&self) -> Value {
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

    async fn execute(&self, input: Value, sandbox: &SandboxConfig) -> Result<ToolResult, ToolError> {
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
            tool_name: self.name().into(),
            success: status.is_success(),
            content,
            is_error: status.is_server_error(),
        })
    }
}

// ============================================================================
// Tool Executor
// ============================================================================

/// Manages available tools and executes them with trust gating
pub struct ToolExecutor {
    tools: HashMap<String, Arc<dyn Tool>>,
    sandbox: SandboxConfig,
    trust_level: TrustLevel,
}

impl ToolExecutor {
    /// Create a new tool executor with default tools
    pub fn new(sandbox: SandboxConfig, trust_level: TrustLevel) -> Self {
        let mut tools: HashMap<String, Arc<dyn Tool>> = HashMap::new();

        let all_tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(FileReadTool),
            Arc::new(GrepTool),
            Arc::new(GlobTool),
            Arc::new(BashTool),
            Arc::new(WebFetchTool),
        ];

        for tool in all_tools {
            if tool.min_trust_level() <= trust_level {
                tools.insert(tool.name().to_string(), tool);
            }
        }

        Self {
            tools,
            sandbox,
            trust_level,
        }
    }

    /// Get tool definitions for LLM (only tools this executor can access)
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.to_definition()).collect()
    }

    /// Execute a tool by name
    pub async fn execute(&self, tool_name: &str, input: Value) -> Result<ToolResult, ToolError> {
        let tool = self
            .tools
            .get(tool_name)
            .ok_or_else(|| ToolError::NotFound(tool_name.into()))?;

        if tool.min_trust_level() > self.trust_level {
            return Err(ToolError::PermissionDenied(format!(
                "Tool {} requires {:?} trust, agent has {:?}",
                tool_name,
                tool.min_trust_level(),
                self.trust_level
            )));
        }

        tool.execute(input, &self.sandbox).await
    }

    /// Check if a tool is available
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get current trust level
    pub fn trust_level(&self) -> TrustLevel {
        self.trust_level
    }

    /// Upgrade trust level (enables more tools)
    pub fn upgrade_trust(&mut self, new_level: TrustLevel) {
        if new_level > self.trust_level {
            self.trust_level = new_level;

            // Re-add tools that are now accessible
            let new_tools: Vec<Arc<dyn Tool>> = vec![
                Arc::new(FileReadTool),
                Arc::new(GrepTool),
                Arc::new(GlobTool),
                Arc::new(BashTool),
                Arc::new(WebFetchTool),
            ];

            for tool in new_tools {
                if tool.min_trust_level() <= self.trust_level {
                    self.tools
                        .entry(tool.name().to_string())
                        .or_insert(tool);
                }
            }
        }
    }
}

impl std::fmt::Debug for ToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutor")
            .field("trust_level", &self.trust_level)
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
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
