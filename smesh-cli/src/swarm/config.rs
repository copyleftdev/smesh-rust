//! Configuration for the Vuln Swarm scanner

use std::path::PathBuf;

/// Configuration for a vulnerability swarm scan
#[derive(Debug, Clone)]
pub struct VulnSwarmConfig {
    /// Path to scan for vulnerabilities
    pub target_path: PathBuf,

    /// Claude model to use (default: claude-sonnet-4-20250514)
    pub model: String,

    /// Maximum number of files to analyze
    pub max_files: usize,

    /// Maximum tokens to send per file (truncates code)
    pub max_tokens_per_file: usize,

    /// Number of agents that must agree for consensus (default: 3)
    pub consensus_threshold: u32,

    /// Confidence threshold for fast-track consensus with fewer agents (default: 0.8)
    pub high_confidence_threshold: f64,

    /// Interval between field ticks in milliseconds (default: 500)
    pub tick_interval_ms: u64,

    /// Maximum concurrent API requests (default: 5)
    pub max_concurrent_requests: usize,

    /// Rate limit: requests per minute (default: 50)
    pub requests_per_minute: u32,

    /// TTL for vulnerability signals in seconds (default: 60)
    pub signal_ttl_secs: f64,

    /// Output format (text, json, sarif)
    pub output_format: OutputFormat,
}

/// Output format for scan results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output
    Json,
    /// SARIF (Static Analysis Results Interchange Format)
    Sarif,
}

impl OutputFormat {
    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "json" => OutputFormat::Json,
            "sarif" => OutputFormat::Sarif,
            _ => OutputFormat::Text,
        }
    }
}

impl Default for VulnSwarmConfig {
    fn default() -> Self {
        Self {
            target_path: PathBuf::from("."),
            model: "claude-sonnet-4-20250514".to_string(),
            max_files: 50,
            max_tokens_per_file: 4000,
            consensus_threshold: 3,
            high_confidence_threshold: 0.8,
            tick_interval_ms: 500,
            max_concurrent_requests: 5,
            requests_per_minute: 50,
            signal_ttl_secs: 60.0,
            output_format: OutputFormat::Text,
        }
    }
}

impl VulnSwarmConfig {
    /// Create a new config with a target path
    pub fn new(target_path: PathBuf) -> Self {
        Self {
            target_path,
            ..Default::default()
        }
    }

    /// Set the model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set maximum files to analyze
    pub fn with_max_files(mut self, max: usize) -> Self {
        self.max_files = max;
        self
    }

    /// Set output format
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set consensus threshold
    pub fn with_consensus_threshold(mut self, threshold: u32) -> Self {
        self.consensus_threshold = threshold;
        self
    }

    /// Set rate limiting parameters
    pub fn with_rate_limits(mut self, concurrent: usize, per_minute: u32) -> Self {
        self.max_concurrent_requests = concurrent;
        self.requests_per_minute = per_minute;
        self
    }

    /// File extensions to scan
    pub fn scannable_extensions(&self) -> &[&str] {
        &[
            "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "rb", "php", "c", "cpp", "h",
            "hpp", "cs", "swift", "kt", "scala", "sql", "sh", "bash", "yaml", "yml", "json", "xml",
            "html", "htm",
        ]
    }
}
