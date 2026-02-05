use chrono::{NaiveTime, Utc, Timelike};
use serde::Deserialize;

/// Scheduling logic — determines optimal call windows per payer.
///
/// Uses payer-specific best-call-window hints + adaptive learning
/// signals to decide when to dispatch calls.

#[derive(Debug, Clone, Deserialize)]
pub struct CallWindow {
    pub start: NaiveTime,
    pub end: NaiveTime,
    pub timezone: String,
}

impl CallWindow {
    /// Check if the current time falls within this call window.
    /// Note: simplified — uses UTC offset, production should use chrono-tz.
    pub fn is_open_utc(&self, utc_offset_hours: i32) -> bool {
        let now_utc = Utc::now();
        let local_hour = (now_utc.hour() as i32 + utc_offset_hours).rem_euclid(24) as u32;
        let local_time = NaiveTime::from_hms_opt(local_hour, now_utc.minute(), 0)
            .unwrap_or(NaiveTime::from_hms_opt(0, 0, 0).unwrap());

        if self.start <= self.end {
            local_time >= self.start && local_time <= self.end
        } else {
            // Wraps midnight
            local_time >= self.start || local_time <= self.end
        }
    }
}

/// Priority scoring for call dispatch ordering.
///
/// Higher score = dispatch sooner.
/// Factors: SLA deadline proximity, signal strength, retry count, payer queue depth.
pub fn priority_score(
    sla_remaining_secs: i64,
    signal_strength: f64,
    retry_count: u32,
    payer_queue_depth: usize,
) -> f64 {
    let urgency = if sla_remaining_secs <= 0 {
        10.0 // Overdue — highest urgency
    } else if sla_remaining_secs < 3600 {
        5.0 + (1.0 - sla_remaining_secs as f64 / 3600.0) * 5.0
    } else {
        1.0
    };

    let retry_boost = 1.0 + (retry_count as f64 * 0.5);
    let queue_penalty = 1.0 / (1.0 + payer_queue_depth as f64 * 0.1);

    urgency * signal_strength * retry_boost * queue_penalty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overdue_has_highest_priority() {
        let overdue = priority_score(-100, 1.0, 0, 0);
        let normal = priority_score(7200, 1.0, 0, 0);
        assert!(overdue > normal);
    }

    #[test]
    fn test_retries_boost_priority() {
        let no_retry = priority_score(3600, 1.0, 0, 0);
        let retried = priority_score(3600, 1.0, 2, 0);
        assert!(retried > no_retry);
    }

    #[test]
    fn test_deep_queue_reduces_priority() {
        let empty_queue = priority_score(3600, 1.0, 0, 0);
        let deep_queue = priority_score(3600, 1.0, 0, 10);
        assert!(empty_queue > deep_queue);
    }
}
