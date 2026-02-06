//! Typed payload structures for SMESH signals
//!
//! These structs provide type-safe payloads that can be efficiently
//! serialized with TOON format for reduced token costs when processed by LLMs.
//!
//! # TOON vs JSON Token Savings
//!
//! TOON (Token-Oriented Object Notation) typically reduces tokens by 18-25%:
//!
//! ```text
//! JSON: {"task_id":"abc123","task_type":"review","priority":0.8}  (52 chars)
//! TOON: task_id:abc123|task_type:review|priority:0.8              (42 chars)
//! ```

use serde::{Deserialize, Serialize};

/// Task signal payload for agent coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPayload {
    /// Unique task identifier
    pub task_id: String,
    /// Type of task (e.g., "review", "analyze", "generate")
    pub task_type: String,
    /// Task priority (0.0 - 1.0)
    pub priority: f64,
    /// Task description
    pub description: String,
}

/// Compact task signal (minimal fields for lower token cost)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPayloadCompact {
    /// Task ID (shortened)
    pub id: String,
    /// Task type code (e.g., "rev", "ana", "gen")
    pub ty: String,
    /// Priority (0.0 - 1.0)
    pub p: f64,
}

impl TaskPayloadCompact {
    /// Create from full TaskPayload, discarding description
    pub fn from_full(full: &TaskPayload) -> Self {
        Self {
            id: full.task_id.clone(),
            ty: Self::compress_type(&full.task_type),
            p: full.priority,
        }
    }

    /// Compress task type to short code
    fn compress_type(task_type: &str) -> String {
        match task_type.to_lowercase().as_str() {
            "review" | "code_review" => "rev".to_string(),
            "analyze" | "analysis" => "ana".to_string(),
            "generate" | "generation" => "gen".to_string(),
            "threat" | "threat_analysis" => "thr".to_string(),
            "correlate" | "correlation" => "cor".to_string(),
            "mitigate" | "mitigation" => "mit".to_string(),
            other => other.chars().take(3).collect(),
        }
    }
}

/// Finding signal payload for code review results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingPayload {
    /// File path where finding was discovered
    pub file: String,
    /// Reviewer type (e.g., "security", "performance")
    pub reviewer: String,
    /// Finding content/description
    pub content: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Compact finding payload (for signal transmission)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingPayloadCompact {
    /// File path (relative)
    pub f: String,
    /// Reviewer code
    pub r: String,
    /// Confidence
    pub c: f32,
}

impl FindingPayloadCompact {
    /// Create from full FindingPayload
    pub fn from_full(full: &FindingPayload) -> Self {
        Self {
            f: full.file.clone(),
            r: Self::compress_reviewer(&full.reviewer),
            c: full.confidence,
        }
    }

    /// Compress reviewer type to short code
    fn compress_reviewer(reviewer: &str) -> String {
        match reviewer.to_lowercase().as_str() {
            "security" => "sec".to_string(),
            "performance" => "perf".to_string(),
            "style" => "sty".to_string(),
            "documentation" => "doc".to_string(),
            other => other.chars().take(4).collect(),
        }
    }
}

/// Threat pattern payload for threat analysis signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatPayload {
    /// Threat category
    pub category: String,
    /// Source file path
    pub source: String,
    /// Pattern name
    pub pattern: String,
    /// Severity level
    pub severity: String,
}

/// Compact threat payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatPayloadCompact {
    /// Category code
    pub cat: String,
    /// Source file
    pub src: String,
    /// Severity code (C/H/M/L/I)
    pub sev: String,
}

impl ThreatPayloadCompact {
    /// Create from full ThreatPayload
    pub fn from_full(full: &ThreatPayload) -> Self {
        Self {
            cat: Self::compress_category(&full.category),
            src: full.source.clone(),
            sev: Self::compress_severity(&full.severity),
        }
    }

    /// Compress category to code
    fn compress_category(category: &str) -> String {
        match category.to_lowercase().as_str() {
            "injection" => "inj".to_string(),
            "xss" => "xss".to_string(),
            "authentication" => "auth".to_string(),
            "traversal" => "trav".to_string(),
            "deserialization" => "deser".to_string(),
            "ssrf" => "ssrf".to_string(),
            "xxe" => "xxe".to_string(),
            "cryptographic" => "cryp".to_string(),
            "misconfiguration" => "misc".to_string(),
            other => other.chars().take(4).collect(),
        }
    }

    /// Compress severity to single char
    fn compress_severity(severity: &str) -> String {
        match severity.to_uppercase().as_str() {
            "CRITICAL" => "C".to_string(),
            "HIGH" => "H".to_string(),
            "MEDIUM" => "M".to_string(),
            "LOW" => "L".to_string(),
            "INFO" => "I".to_string(),
            other => other.chars().next().unwrap_or('?').to_string(),
        }
    }
}

/// Agent signal type indicator
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentSignalType {
    /// Task is available for claiming
    TaskAvailable,
    /// Task has been claimed by an agent
    TaskClaimed,
    /// Task has been completed
    TaskCompleted,
    /// Finding discovered
    Finding,
    /// Correlation identified
    Correlation,
}

impl AgentSignalType {
    /// Convert to compact string representation
    pub fn as_code(&self) -> &'static str {
        match self {
            AgentSignalType::TaskAvailable => "avail",
            AgentSignalType::TaskClaimed => "claim",
            AgentSignalType::TaskCompleted => "done",
            AgentSignalType::Finding => "find",
            AgentSignalType::Correlation => "corr",
        }
    }

    /// Parse from compact string
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "avail" => Some(AgentSignalType::TaskAvailable),
            "claim" => Some(AgentSignalType::TaskClaimed),
            "done" => Some(AgentSignalType::TaskCompleted),
            "find" => Some(AgentSignalType::Finding),
            "corr" => Some(AgentSignalType::Correlation),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to encode a value to TOON format
    fn encode_toon<T: Serialize>(value: &T) -> String {
        let json_value = serde_json::to_value(value).unwrap();
        toon::encode(&json_value, None)
    }

    #[test]
    fn test_task_payload_toon_encoding() {
        let payload = TaskPayload {
            task_id: "abc123".to_string(),
            task_type: "review".to_string(),
            priority: 0.8,
            description: "Review the auth module".to_string(),
        };

        // Encode as TOON
        let toon_str = encode_toon(&payload);

        // TOON is primarily for LLM consumption, not roundtrip
        // Verify it encodes without error and is non-empty
        assert!(!toon_str.is_empty());
        println!("TOON output: {}", toon_str);
    }

    #[test]
    fn test_compact_payload_savings() {
        let full = TaskPayload {
            task_id: "abc123".to_string(),
            task_type: "review".to_string(),
            priority: 0.8,
            description: "Review the auth module for security issues".to_string(),
        };

        let compact = TaskPayloadCompact::from_full(&full);

        let full_json = serde_json::to_string(&full).unwrap();
        let compact_toon = encode_toon(&compact);

        // Compact TOON should be significantly smaller
        assert!(compact_toon.len() < full_json.len());

        // Print for visibility
        println!("Full JSON: {} bytes - {}", full_json.len(), full_json);
        println!(
            "Compact TOON: {} bytes - {}",
            compact_toon.len(),
            compact_toon
        );
        println!(
            "Savings: {:.1}%",
            (1.0 - compact_toon.len() as f64 / full_json.len() as f64) * 100.0
        );
    }

    #[test]
    fn test_threat_payload_compact() {
        let full = ThreatPayload {
            category: "Injection".to_string(),
            source: "sql/queries.rs".to_string(),
            pattern: "SQL Injection via string concat".to_string(),
            severity: "CRITICAL".to_string(),
        };

        let compact = ThreatPayloadCompact::from_full(&full);

        assert_eq!(compact.cat, "inj");
        assert_eq!(compact.sev, "C");
    }

    #[test]
    fn test_agent_signal_type_roundtrip() {
        let types = [
            AgentSignalType::TaskAvailable,
            AgentSignalType::TaskClaimed,
            AgentSignalType::TaskCompleted,
            AgentSignalType::Finding,
            AgentSignalType::Correlation,
        ];

        for t in types {
            let code = t.as_code();
            let parsed = AgentSignalType::from_code(code).unwrap();
            assert_eq!(t, parsed);
        }
    }
}
