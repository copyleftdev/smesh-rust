//! BountyHunter agent - the core hybrid of SMESH Node + LLM + Tool execution
//!
//! This is the heart of the integration. Each BountyHunter is:
//! - A SMESH Node (emits, senses, reinforces signals)
//! - An LLM client (thinks via Claude/Ollama)
//! - A ToolExecutor (acts on the filesystem, network, etc.)
//!
//! The agentic loop:
//! 1. Receive task (via SMESH signal or direct assignment)
//! 2. Build prompt with system instructions + task + context
//! 3. Send to LLM with tool definitions
//! 4. If LLM requests tool_use: execute tool, feed result back
//! 5. Repeat 3-4 until LLM returns end_turn
//! 6. Emit findings as SMESH signals

use std::sync::Arc;
use tracing::{debug, info};

use smesh_agent::{
    ContentBlock, Conversation, GenerateRequestV2, LlmBackend, LlmError,
    Message, StopReason, ToolDefinition,
};
use smesh_core::{Node, Signal, SignalType, ThreatPayload};

use crate::specialization::BountySpecialization;
use crate::tools::{SandboxConfig, ToolExecutor, ToolResult, TrustLevel};

/// Configuration for a BountyHunter agent
#[derive(Debug, Clone)]
pub struct HunterConfig {
    /// Agent name (e.g., "AUDIT-1")
    pub name: String,
    /// Specialization
    pub specialization: BountySpecialization,
    /// Maximum tool-use turns per task
    pub max_turns: u32,
    /// LLM temperature (lower = more focused)
    pub temperature: f32,
    /// Maximum tokens per LLM response
    pub max_tokens: u32,
}

impl HunterConfig {
    /// Create config for a specialization
    pub fn new(specialization: BountySpecialization, instance: u32) -> Self {
        Self {
            name: format!("{}-{}", specialization.code(), instance),
            specialization,
            max_turns: 15,
            temperature: 0.3,
            max_tokens: 4096,
        }
    }
}

/// A security finding emitted by a hunter
#[derive(Debug, Clone)]
pub struct HunterFinding {
    /// Severity level
    pub severity: String,
    /// Vulnerability type
    pub vuln_type: String,
    /// File path where found
    pub file_path: String,
    /// Line number if known
    pub line_number: Option<u32>,
    /// Description
    pub description: String,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
    /// The agent that found it
    pub found_by: String,
}

/// Metrics tracked per hunter
#[derive(Debug, Clone, Default)]
pub struct HunterMetrics {
    pub tool_calls: u32,
    pub llm_calls: u32,
    pub findings_emitted: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub turns_used: u32,
}

/// A BountyHunter agent: SMESH Node + LLM + Tools
pub struct BountyHunter {
    /// SMESH node for coordination
    pub node: Node,
    /// Configuration
    pub config: HunterConfig,
    /// LLM backend
    backend: Arc<dyn LlmBackend>,
    /// Tool executor (sandboxed, trust-gated)
    tools: ToolExecutor,
    /// Collected findings
    pub findings: Vec<HunterFinding>,
    /// Metrics
    pub metrics: HunterMetrics,
}

impl BountyHunter {
    /// Create a new BountyHunter
    pub fn new(
        config: HunterConfig,
        backend: Arc<dyn LlmBackend>,
        sandbox: SandboxConfig,
    ) -> Self {
        let trust_level = config.specialization.min_trust_level();
        let tools = ToolExecutor::new(sandbox, trust_level);

        let mut node = Node::new();
        node.id = config.name.clone();

        Self {
            node,
            config,
            backend,
            tools,
            findings: Vec::new(),
            metrics: HunterMetrics::default(),
        }
    }

    /// Get agent name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get specialization
    pub fn specialization(&self) -> BountySpecialization {
        self.config.specialization
    }

    /// Run the agentic tool-use loop on a task
    ///
    /// This is the core integration: SMESH provides the task, Claude thinks,
    /// tools execute, findings become signals.
    pub async fn execute_task(&mut self, task_description: &str) -> Result<Vec<HunterFinding>, LlmError> {
        let system_prompt = format!(
            "{}\n\nYou are agent '{}' in the SMESH bounty hunting swarm.\n\
             Use the available tools to investigate. Be thorough but focused.\n\
             When you find a vulnerability, describe it clearly with severity, \
             type, file path, line number, and description.",
            self.config.specialization.system_prompt(),
            self.config.name,
        );

        let tool_defs = self.tools.tool_definitions();

        // Build initial conversation
        let mut conversation = Conversation::with_system(&system_prompt)
            .with_tools(tool_defs);
        conversation.add_user_message(task_description);

        let mut turn = 0;
        let mut findings = Vec::new();

        loop {
            if turn >= self.config.max_turns {
                info!("{}: hit max turns ({})", self.config.name, self.config.max_turns);
                break;
            }

            turn += 1;
            self.metrics.turns_used += 1;

            // Call LLM
            let request = GenerateRequestV2 {
                messages: conversation.messages.clone(),
                system: conversation.system.clone(),
                tools: conversation.tools.clone(),
                tool_choice: smesh_agent::ToolChoice::Auto,
                stream: false,
                temperature: self.config.temperature,
                max_tokens: self.config.max_tokens,
                stop_sequences: Vec::new(),
            };

            debug!("{}: turn {}, sending request", self.config.name, turn);
            self.metrics.llm_calls += 1;

            let response = self.backend.generate_v2(request).await?;
            self.metrics.input_tokens += response.input_tokens;
            self.metrics.output_tokens += response.output_tokens;

            // Add assistant response to conversation
            conversation.add_message(Message {
                role: smesh_agent::MessageRole::Assistant,
                content: response.content.clone(),
            });

            // Check if model wants to use tools
            if response.stop_reason == StopReason::ToolUse {
                // Execute each tool use request
                let tool_uses: Vec<(String, String, serde_json::Value)> = response
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolUse { id, name, input } => {
                            Some((id.clone(), name.clone(), input.clone()))
                        }
                        _ => None,
                    })
                    .collect();

                for (tool_id, tool_name, tool_input) in tool_uses {
                    debug!("{}: executing tool {} ({})", self.config.name, tool_name, tool_id);
                    self.metrics.tool_calls += 1;

                    let result = match self.tools.execute(&tool_name, tool_input).await {
                        Ok(result) => result,
                        Err(e) => ToolResult {
                            tool_name: tool_name.clone(),
                            success: false,
                            content: format!("Tool error: {}", e),
                            is_error: true,
                        },
                    };

                    // Feed tool result back into conversation
                    conversation.add_message(Message::tool_result(
                        &tool_id,
                        &result.content,
                        result.is_error,
                    ));
                }
            } else {
                // Model is done - extract findings from final response text
                let text = response.text();
                if !text.is_empty() {
                    let parsed = parse_findings_from_text(&text, &self.config.name);
                    findings.extend(parsed);
                }
                break;
            }
        }

        // Store findings
        self.metrics.findings_emitted += findings.len() as u32;
        self.findings.extend(findings.clone());

        info!(
            "{}: task complete - {} findings, {} tool calls, {} LLM calls",
            self.config.name,
            findings.len(),
            self.metrics.tool_calls,
            self.metrics.llm_calls,
        );

        Ok(findings)
    }

    /// Convert a finding into a SMESH signal for field emission
    pub fn finding_to_signal(&self, finding: &HunterFinding) -> Signal {
        let payload = ThreatPayload {
            category: finding.vuln_type.clone(),
            source: finding.file_path.clone(),
            pattern: finding.description.clone(),
            severity: finding.severity.clone(),
        };

        let intensity = match finding.severity.to_uppercase().as_str() {
            "CRITICAL" => 1.0,
            "HIGH" => 0.9,
            "MEDIUM" => 0.7,
            "LOW" => 0.5,
            _ => 0.3,
        };

        Signal::builder(SignalType::Alert)
            .payload_json(&payload)
            .intensity(intensity)
            .confidence(finding.confidence)
            .ttl(120.0)
            .origin(&self.config.name)
            .build()
    }

    /// Get tool definitions (for LLM prompting)
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.tool_definitions()
    }

    /// Upgrade the agent's trust level (e.g., after proving reliable)
    pub fn upgrade_trust(&mut self, level: TrustLevel) {
        self.tools.upgrade_trust(level);
    }
}

impl std::fmt::Debug for BountyHunter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BountyHunter")
            .field("name", &self.config.name)
            .field("specialization", &self.config.specialization)
            .field("findings", &self.findings.len())
            .field("metrics", &self.metrics)
            .finish()
    }
}

// ============================================================================
// Finding Parser
// ============================================================================

/// Parse structured findings from LLM text output
///
/// Looks for patterns like:
/// [CRITICAL] SQL Injection: Unsanitized input in query at src/db.rs:42
/// SEVERITY: HIGH
/// TYPE: XSS
/// etc.
fn parse_findings_from_text(text: &str, agent_name: &str) -> Vec<HunterFinding> {
    let mut findings = Vec::new();

    // Pattern 1: [SEVERITY] Type: Description (file:line)
    // Two passes: first try with file:line, then without
    let bracket_with_loc = regex::Regex::new(
        r"(?i)\[(CRITICAL|HIGH|MEDIUM|LOW|INFO)\]\s+([^:]+?):\s+(.+?)\(([^):]+):(\d+)\)"
    ).unwrap();
    let bracket_no_loc = regex::Regex::new(
        r"(?im)\[(CRITICAL|HIGH|MEDIUM|LOW|INFO)\]\s+([^:]+?):\s+(.+?)$"
    ).unwrap();

    // First pass: findings with file:line location
    for cap in bracket_with_loc.captures_iter(text) {
        let severity = cap[1].to_uppercase();
        let confidence = match severity.as_str() {
            "CRITICAL" => 0.9,
            "HIGH" => 0.8,
            "MEDIUM" => 0.7,
            "LOW" => 0.6,
            _ => 0.5,
        };
        findings.push(HunterFinding {
            severity,
            vuln_type: cap[2].trim().to_string(),
            description: cap[3].trim().to_string(),
            file_path: cap[4].trim().to_string(),
            line_number: cap[5].parse().ok(),
            confidence,
            found_by: agent_name.to_string(),
        });
    }

    // Second pass: findings without location (skip lines already matched)
    for cap in bracket_no_loc.captures_iter(text) {
        let severity = cap[1].to_uppercase();
        let vuln_type = cap[2].trim().to_string();
        let description = cap[3].trim().to_string();

        // Skip if we already captured this line via the with-location regex
        let already_found = findings.iter().any(|f| {
            f.severity == severity && f.vuln_type == vuln_type
        });
        if already_found {
            continue;
        }

        let confidence = match severity.as_str() {
            "CRITICAL" => 0.9,
            "HIGH" => 0.8,
            "MEDIUM" => 0.7,
            "LOW" => 0.6,
            _ => 0.5,
        };
        findings.push(HunterFinding {
            severity,
            vuln_type,
            file_path: String::new(),
            line_number: None,
            description,
            confidence,
            found_by: agent_name.to_string(),
        });
    }

    // Pattern 2: Structured SEVERITY: / TYPE: / FILE: blocks
    if findings.is_empty() {
        let mut current_severity = String::new();
        let mut current_type = String::new();
        let mut current_file = String::new();
        let mut current_line: Option<u32> = None;
        let mut current_desc = String::new();
        let mut current_conf = 0.7_f64;

        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(val) = trimmed.strip_prefix("SEVERITY:") {
                // Flush previous finding if we have one
                if !current_severity.is_empty() && !current_type.is_empty() {
                    findings.push(HunterFinding {
                        severity: current_severity.clone(),
                        vuln_type: current_type.clone(),
                        file_path: current_file.clone(),
                        line_number: current_line,
                        description: current_desc.clone(),
                        confidence: current_conf,
                        found_by: agent_name.to_string(),
                    });
                }
                current_severity = val.trim().to_uppercase();
                current_type.clear();
                current_file.clear();
                current_line = None;
                current_desc.clear();
                current_conf = 0.7;
            } else if let Some(val) = trimmed.strip_prefix("TYPE:") {
                current_type = val.trim().to_string();
            } else if let Some(val) = trimmed.strip_prefix("FILE:") {
                current_file = val.trim().to_string();
            } else if let Some(val) = trimmed.strip_prefix("LINE:") {
                current_line = val.trim().parse().ok();
            } else if let Some(val) = trimmed.strip_prefix("DESCRIPTION:") {
                current_desc = val.trim().to_string();
            } else if let Some(val) = trimmed.strip_prefix("CONFIDENCE:") {
                current_conf = val.trim().parse().unwrap_or(0.7);
            }
        }

        // Flush last finding
        if !current_severity.is_empty() && !current_type.is_empty() {
            findings.push(HunterFinding {
                severity: current_severity,
                vuln_type: current_type,
                file_path: current_file,
                line_number: current_line,
                description: current_desc,
                confidence: current_conf,
                found_by: agent_name.to_string(),
            });
        }
    }

    findings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bracket_findings() {
        let text = r#"
Analysis complete. Found the following:

[CRITICAL] SQL Injection: User input concatenated into query (src/db.rs:42)
[HIGH] XSS: Unescaped output in template (templates/user.html:15)
[LOW] Information Disclosure: Stack trace in error response
"#;
        let findings = parse_findings_from_text(text, "AUDIT-1");
        assert_eq!(findings.len(), 3);
        assert_eq!(findings[0].severity, "CRITICAL");
        assert_eq!(findings[0].vuln_type, "SQL Injection");
        assert_eq!(findings[0].file_path, "src/db.rs");
        assert_eq!(findings[0].line_number, Some(42));
        assert_eq!(findings[1].severity, "HIGH");
        assert_eq!(findings[2].severity, "LOW");
    }

    #[test]
    fn test_parse_structured_findings() {
        let text = r#"
SEVERITY: CRITICAL
TYPE: Command Injection
FILE: src/api/exec.rs
LINE: 87
DESCRIPTION: User input passed directly to std::process::Command
CONFIDENCE: 0.95
"#;
        let findings = parse_findings_from_text(text, "AUDIT-2");
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, "CRITICAL");
        assert_eq!(findings[0].vuln_type, "Command Injection");
        assert_eq!(findings[0].line_number, Some(87));
        assert!((findings[0].confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_hunter_config() {
        let config = HunterConfig::new(BountySpecialization::SourceAudit, 1);
        assert_eq!(config.name, "AUDIT-1");
        assert_eq!(config.specialization, BountySpecialization::SourceAudit);
    }
}
