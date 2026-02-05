use anyhow::Result;
use tracing::{info, warn};

use crate::config::SmeshConfig;
use crate::error::SmeshError;
use crate::store::RedisSignalStore;

/// The SMESH orchestrator — manages concurrent call lifecycle.
///
/// Responsibilities:
///   - Enforce concurrency limits (global + per-payer)
///   - Dispatch calls when capacity is available
///   - Handle retry logic with exponential backoff
///   - Route based on signal-driven priority
///   - Aggregate results
///   - Feed adaptive learning signals
pub struct Orchestrator {
    config: SmeshConfig,
    store: RedisSignalStore,
}

impl Orchestrator {
    pub fn new(config: SmeshConfig, store: RedisSignalStore) -> Self {
        Self { config, store }
    }

    /// Check if a new call can be dispatched (global + payer limits).
    pub async fn can_dispatch(&mut self, payer_id: &str) -> Result<bool> {
        let global_count = self.store.active_call_count().await?;
        if global_count >= self.config.concurrency.max_concurrent_calls {
            warn!(
                current = global_count,
                max = self.config.concurrency.max_concurrent_calls,
                "Global concurrency limit reached"
            );
            return Ok(false);
        }

        let payer_count = self.store.payer_call_count(payer_id).await?;
        if payer_count >= self.config.concurrency.max_per_payer {
            warn!(
                payer_id,
                current = payer_count,
                max = self.config.concurrency.max_per_payer,
                "Payer concurrency limit reached"
            );
            return Ok(false);
        }

        Ok(true)
    }

    /// Calculate retry delay with exponential backoff + optional jitter.
    pub fn retry_delay_secs(&self, attempt: u32) -> u64 {
        let base = self.config.retry.base_delay_secs;
        let max = self.config.retry.max_delay_secs;
        let delay = base * 2u64.saturating_pow(attempt.saturating_sub(1));
        let capped = delay.min(max);

        if self.config.retry.jitter {
            // Simple jitter: 50-100% of capped delay
            let jitter_factor = 0.5 + (rand_simple() * 0.5);
            (capped as f64 * jitter_factor) as u64
        } else {
            capped
        }
    }

    /// Main run loop — listens for signals and dispatches work.
    pub async fn run(&self) -> Result<()> {
        info!("Orchestrator run loop started");
        // TODO: Implement Redis pub/sub listener for incoming signals
        // TODO: Process signal queue, enforce limits, dispatch calls
        // For now, placeholder that keeps the process alive
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }
}

/// Simple deterministic pseudo-random for jitter (no external dep).
fn rand_simple() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}
