//! Swarm metrics and trust tracking

use std::collections::HashMap;
use std::time::Duration;

use super::{ConsensusFinding, Severity, VulnAgent};

/// Trust snapshot at a point in time
#[derive(Debug, Clone)]
pub struct TrustSnapshot {
    /// Tick number
    pub tick: u32,
    /// Trust scores per agent
    pub agent_trust: HashMap<String, f64>,
}

/// Aggregated metrics for the swarm scan
#[derive(Debug, Clone, Default)]
pub struct SwarmMetrics {
    /// Total files scanned
    pub files_scanned: u32,

    /// Total files skipped (too small, binary, etc.)
    pub files_skipped: u32,

    /// Raw findings before consensus
    pub raw_findings: u32,

    /// Findings that reached consensus
    pub consensus_findings: u32,

    /// Findings by severity
    pub findings_by_severity: HashMap<Severity, u32>,

    /// Total signals emitted
    pub signals_emitted: u32,

    /// Total signal reinforcements
    pub signals_reinforced: u32,

    /// Signals that expired without consensus
    pub signals_expired: u32,

    /// Total input tokens used
    pub total_input_tokens: u32,

    /// Total output tokens used
    pub total_output_tokens: u32,

    /// Total API calls made
    pub total_api_calls: u32,

    /// Estimated tokens saved by TOON encoding
    pub toon_tokens_saved: u32,

    /// Average time to consensus (ms)
    pub avg_time_to_consensus_ms: f64,

    /// Total ticks executed
    pub total_ticks: u32,

    /// Total duration
    pub total_duration: Duration,

    /// Trust evolution snapshots
    pub trust_snapshots: Vec<TrustSnapshot>,
}

impl SwarmMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a file scan
    pub fn record_file_scanned(&mut self) {
        self.files_scanned += 1;
    }

    /// Record a skipped file
    pub fn record_file_skipped(&mut self) {
        self.files_skipped += 1;
    }

    /// Record a raw finding
    pub fn record_raw_finding(&mut self, severity: Severity) {
        self.raw_findings += 1;
        *self.findings_by_severity.entry(severity).or_insert(0) += 1;
    }

    /// Record a signal emission
    pub fn record_signal_emitted(&mut self) {
        self.signals_emitted += 1;
    }

    /// Record a signal reinforcement
    pub fn record_signal_reinforced(&mut self) {
        self.signals_reinforced += 1;
    }

    /// Record an expired signal
    pub fn record_signal_expired(&mut self) {
        self.signals_expired += 1;
    }

    /// Record a consensus finding
    pub fn record_consensus(&mut self, finding: &ConsensusFinding) {
        self.consensus_findings += 1;

        // Update average time to consensus
        let n = self.consensus_findings as f64;
        self.avg_time_to_consensus_ms =
            ((n - 1.0) * self.avg_time_to_consensus_ms + finding.time_to_consensus_ms as f64) / n;
    }

    /// Record token usage
    pub fn record_tokens(&mut self, input: u32, output: u32) {
        self.total_input_tokens += input;
        self.total_output_tokens += output;
        self.total_api_calls += 1;
    }

    /// Record TOON savings estimate
    pub fn record_toon_savings(&mut self, json_size: usize, toon_size: usize) {
        if json_size > toon_size {
            // Estimate token savings (rough: 4 chars per token)
            let chars_saved = json_size - toon_size;
            self.toon_tokens_saved += (chars_saved / 4) as u32;
        }
    }

    /// Record a tick
    pub fn record_tick(&mut self) {
        self.total_ticks += 1;
    }

    /// Take a trust snapshot
    pub fn snapshot_trust(&mut self, tick: u32, agents: &[VulnAgent]) {
        let agent_trust: HashMap<String, f64> = agents
            .iter()
            .map(|a| (a.id.clone(), a.node.get_trust(&a.id)))
            .collect();

        self.trust_snapshots
            .push(TrustSnapshot { tick, agent_trust });
    }

    /// Aggregate metrics from agents
    pub fn aggregate_from_agents(&mut self, agents: &[VulnAgent]) {
        for agent in agents {
            self.total_input_tokens += agent.metrics.input_tokens;
            self.total_output_tokens += agent.metrics.output_tokens;
            self.total_api_calls += agent.metrics.api_calls;
        }
    }

    /// Total tokens used
    pub fn total_tokens(&self) -> u32 {
        self.total_input_tokens + self.total_output_tokens
    }

    /// TOON savings percentage
    pub fn toon_savings_percentage(&self) -> f64 {
        let total = self.total_tokens() as f64;
        if total > 0.0 {
            (self.toon_tokens_saved as f64 / (total + self.toon_tokens_saved as f64)) * 100.0
        } else {
            0.0
        }
    }

    /// Generate summary report
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("======================================================================\n");
        report.push_str("                    VULN SWARM SCAN REPORT\n");
        report.push_str("======================================================================\n");

        report.push_str(&format!(
            "Duration: {:.1}s ({} ticks)\n\n",
            self.total_duration.as_secs_f64(),
            self.total_ticks
        ));

        // Findings summary
        report.push_str(&format!(
            "Findings: {} raw -> {} consensus\n",
            self.raw_findings, self.consensus_findings
        ));

        for severity in [
            Severity::Critical,
            Severity::High,
            Severity::Medium,
            Severity::Low,
            Severity::Info,
        ] {
            if let Some(count) = self.findings_by_severity.get(&severity) {
                if *count > 0 {
                    report.push_str(&format!("  {}: {}\n", severity.name(), count));
                }
            }
        }

        // Token usage
        report.push_str("\nToken Usage:\n");
        report.push_str(&format!("  Input:  {} tokens\n", self.total_input_tokens));
        report.push_str(&format!("  Output: {} tokens\n", self.total_output_tokens));
        report.push_str(&format!("  Total:  {} tokens\n", self.total_tokens()));

        if self.toon_tokens_saved > 0 {
            report.push_str(&format!(
                "  TOON Savings: {:.1}%\n",
                self.toon_savings_percentage()
            ));
        }

        // Signal metrics
        report.push_str("\nSignal Metrics:\n");
        report.push_str(&format!("  Emitted: {}\n", self.signals_emitted));
        report.push_str(&format!("  Reinforced: {}\n", self.signals_reinforced));
        report.push_str(&format!("  Expired: {}\n", self.signals_expired));

        if self.consensus_findings > 0 {
            report.push_str(&format!(
                "  Avg time to consensus: {:.0}ms\n",
                self.avg_time_to_consensus_ms
            ));
        }

        report.push_str("======================================================================\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_aggregation() {
        let mut metrics = SwarmMetrics::new();

        metrics.record_file_scanned();
        metrics.record_file_scanned();
        metrics.record_file_skipped();

        assert_eq!(metrics.files_scanned, 2);
        assert_eq!(metrics.files_skipped, 1);
    }

    #[test]
    fn test_token_tracking() {
        let mut metrics = SwarmMetrics::new();

        metrics.record_tokens(100, 50);
        metrics.record_tokens(200, 100);

        assert_eq!(metrics.total_input_tokens, 300);
        assert_eq!(metrics.total_output_tokens, 150);
        assert_eq!(metrics.total_tokens(), 450);
    }

    #[test]
    fn test_toon_savings() {
        let mut metrics = SwarmMetrics::new();

        // Simulate 100 char JSON becoming 80 char TOON
        metrics.record_toon_savings(100, 80);

        assert!(metrics.toon_tokens_saved > 0);
    }
}
