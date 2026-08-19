//! A deterministic synthetic telemetry corpus for the analysis mesh.
//!
//! **This is a fixture, not a capture.** No real service produced these
//! numbers. It exists so the mesh demo replays identically on any machine: the
//! interesting behaviour is the emergent correlation between analyst nodes, and
//! that is only legible if the input never moves.
//!
//! The corpus describes one incident, seeded so every node in the run derives
//! byte-identical numbers from the same seed without sharing state:
//!
//! > At T+0, `checkout-api` ships `v2.3.1`, which cuts its connection pool from
//! > 200 to 20. The pool pins, requests queue, and callers time out.
//!
//! No single metric family proves that. Saturation sees a pinned pool but not
//! why it matters. Errors see `payments-api` throwing 503s and would blame
//! payments. Latency sees three services slow at once. Only the union of the
//! concerns identifies `checkout-api` as the origin and `payments-api` as a
//! casualty — which is precisely what the mesh has to discover on its own.
//!
//! Two decoys are planted to prove the mesh discriminates rather than
//! agreeing with everything: an unrelated CPU spike on `notification-worker`
//! before the incident, and a brief latency blip on `session-store`. Each is
//! visible to exactly one concern, so neither should ever reach consensus.

use serde::{Deserialize, Serialize};

/// First minute in the corpus, relative to the incident.
pub const FIRST_MINUTE: i32 = -20;
/// Last minute in the corpus, relative to the incident.
pub const LAST_MINUTE: i32 = 19;
/// Total buckets in the corpus.
pub const BUCKET_COUNT: usize = (LAST_MINUTE - FIRST_MINUTE + 1) as usize;

/// A release or config change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deploy {
    /// Service that changed.
    pub service: String,
    /// Version rolled out.
    pub version: String,
    /// Human-readable summary of what changed.
    pub change: String,
}

/// One service's metrics for one minute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bucket {
    /// Minutes relative to the incident; negative is before.
    pub minute: i32,
    /// Service these metrics belong to.
    pub service: String,
    /// 99th percentile request latency, milliseconds.
    pub p99_ms: f64,
    /// Fraction of requests returning an error.
    pub error_rate: f64,
    /// Requests per second.
    pub rps: f64,
    /// CPU utilisation, percent.
    pub cpu_pct: f64,
    /// Connection pool utilisation, 0.0 to 1.0.
    pub pool_utilization: f64,
    /// Fraction of spans that are retries.
    pub retry_rate: f64,
    /// Mean queued spans awaiting a pool slot.
    pub span_queue_depth: f64,
    /// Deploys landing in this minute.
    pub deploys: Vec<Deploy>,
}

/// Baseline behaviour of one service before anything goes wrong.
struct Baseline {
    service: &'static str,
    p99_ms: f64,
    error_rate: f64,
    rps: f64,
    cpu_pct: f64,
    pool_utilization: f64,
    retry_rate: f64,
    span_queue_depth: f64,
}

const BASELINES: &[Baseline] = &[
    Baseline {
        service: "edge-gateway",
        p99_ms: 40.0,
        error_rate: 0.0010,
        rps: 1800.0,
        cpu_pct: 45.0,
        pool_utilization: 0.30,
        retry_rate: 0.008,
        span_queue_depth: 1.5,
    },
    Baseline {
        service: "checkout-api",
        p99_ms: 90.0,
        error_rate: 0.0020,
        rps: 640.0,
        cpu_pct: 55.0,
        pool_utilization: 0.35,
        retry_rate: 0.010,
        span_queue_depth: 2.0,
    },
    Baseline {
        service: "payments-api",
        p99_ms: 120.0,
        error_rate: 0.0030,
        rps: 410.0,
        cpu_pct: 50.0,
        pool_utilization: 0.40,
        retry_rate: 0.012,
        span_queue_depth: 2.5,
    },
    Baseline {
        service: "inventory-svc",
        p99_ms: 60.0,
        error_rate: 0.0010,
        rps: 520.0,
        cpu_pct: 40.0,
        pool_utilization: 0.25,
        retry_rate: 0.006,
        span_queue_depth: 1.0,
    },
    Baseline {
        service: "session-store",
        p99_ms: 15.0,
        error_rate: 0.0005,
        rps: 2400.0,
        cpu_pct: 30.0,
        pool_utilization: 0.20,
        retry_rate: 0.002,
        span_queue_depth: 0.5,
    },
    Baseline {
        service: "notification-worker",
        p99_ms: 200.0,
        error_rate: 0.0040,
        rps: 90.0,
        cpu_pct: 35.0,
        pool_utilization: 0.15,
        retry_rate: 0.015,
        span_queue_depth: 3.0,
    },
];

/// A small deterministic PRNG.
///
/// Written out rather than pulled from `rand` so the corpus does not change if
/// a dependency changes its generator: a fixture whose values drift under you
/// is worse than no fixture.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the zero state, which xorshift cannot leave.
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Jitter in `[-magnitude, magnitude]`.
    fn jitter(&mut self, magnitude: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (unit * 2.0 - 1.0) * magnitude
    }
}

/// Smooth 0→1 ramp over `duration` minutes starting at `start`.
fn ramp(minute: i32, start: i32, duration: i32) -> f64 {
    if minute < start {
        return 0.0;
    }
    let progress = (minute - start) as f64 / duration as f64;
    progress.clamp(0.0, 1.0)
}

/// Generate the full corpus for a seed.
///
/// The same seed always yields the same numbers, on any machine, in any
/// process — which is what lets five separate analyst processes reason about
/// the same telemetry without a shared database.
pub fn generate(seed: u64) -> Vec<Bucket> {
    let mut buckets = Vec::with_capacity(BUCKET_COUNT * BASELINES.len());

    for base in BASELINES {
        // Derive a per-service stream so adding a service does not perturb the
        // numbers of the ones before it.
        let mut rng = Rng::new(seed ^ fnv(base.service));

        for minute in FIRST_MINUTE..=LAST_MINUTE {
            let mut bucket = Bucket {
                minute,
                service: base.service.to_string(),
                p99_ms: base.p99_ms * (1.0 + rng.jitter(0.06)),
                error_rate: (base.error_rate * (1.0 + rng.jitter(0.15))).max(0.0),
                rps: base.rps * (1.0 + rng.jitter(0.08)),
                cpu_pct: base.cpu_pct * (1.0 + rng.jitter(0.05)),
                pool_utilization: base.pool_utilization * (1.0 + rng.jitter(0.08)),
                retry_rate: base.retry_rate * (1.0 + rng.jitter(0.12)),
                span_queue_depth: base.span_queue_depth * (1.0 + rng.jitter(0.20)),
                deploys: Vec::new(),
            };

            apply_incident(&mut bucket, base, minute);
            apply_decoys(&mut bucket, base, minute);
            apply_deploys(&mut bucket, base, minute);

            buckets.push(bucket);
        }
    }

    buckets.sort_by_key(|b| (b.minute, b.service.clone()));
    buckets
}

/// The pool exhaustion and everything downstream of it.
fn apply_incident(bucket: &mut Bucket, base: &Baseline, minute: i32) {
    // The pool pins over three minutes, then stays pinned.
    let severity = ramp(minute, 0, 3);
    if severity <= 0.0 {
        return;
    }

    match base.service {
        // The origin. Note CPU stays flat: this is not a load problem, which is
        // the detail that separates cause from symptom.
        "checkout-api" => {
            bucket.p99_ms += (280.0 - base.p99_ms) * severity;
            bucket.error_rate += (0.030 - base.error_rate) * severity;
            bucket.pool_utilization += (0.98 - base.pool_utilization) * severity;
            bucket.retry_rate += (0.220 - base.retry_rate) * severity;
            bucket.span_queue_depth += (40.0 - base.span_queue_depth) * severity;
        }
        // A casualty: it calls checkout-api, times out, and returns 503s. Its
        // own pool and CPU are fine, which is why blaming it would be wrong.
        "payments-api" => {
            bucket.p99_ms += (240.0 - base.p99_ms) * severity;
            bucket.error_rate += (0.090 - base.error_rate) * severity;
            bucket.retry_rate += (0.140 - base.retry_rate) * severity;
        }
        // Further downstream again, and correspondingly weaker.
        "edge-gateway" => {
            bucket.p99_ms += (76.0 - base.p99_ms) * severity;
            bucket.error_rate += (0.0025 - base.error_rate) * severity;
        }
        _ => {}
    }
}

/// Unrelated anomalies, planted so the mesh has something to reject.
fn apply_decoys(bucket: &mut Bucket, base: &Baseline, minute: i32) {
    // A CPU spike on a worker, well before the incident and causally unrelated.
    // Only the saturation concern can see it.
    if base.service == "notification-worker" && (-8..=-5).contains(&minute) {
        bucket.cpu_pct = 88.0 + bucket.cpu_pct * 0.02;
    }

    // A brief latency blip on the session store, also before the incident.
    // Only the latency concern can see it.
    if base.service == "session-store" && (-14..=-12).contains(&minute) {
        bucket.p99_ms = 48.0 + bucket.p99_ms * 0.05;
    }
}

/// Release events, including one benign deploy that must not be blamed.
fn apply_deploys(bucket: &mut Bucket, base: &Baseline, minute: i32) {
    if base.service == "checkout-api" && minute == 0 {
        bucket.deploys.push(Deploy {
            service: "checkout-api".to_string(),
            version: "v2.3.1".to_string(),
            change: "pool_max_conns 200 -> 20".to_string(),
        });
    }

    // A deploy with no consequences. A concern that flags every deploy would
    // flag this one too, so it is the control case.
    if base.service == "inventory-svc" && minute == -12 {
        bucket.deploys.push(Deploy {
            service: "inventory-svc".to_string(),
            version: "v1.9.0".to_string(),
            change: "log sampling 1.0 -> 0.5".to_string(),
        });
    }
}

/// Baseline value of one metric for a service, for ratio comparisons.
pub fn baseline_of(service: &str, metric: Metric) -> f64 {
    BASELINES
        .iter()
        .find(|b| b.service == service)
        .map(|b| match metric {
            Metric::P99 => b.p99_ms,
            Metric::ErrorRate => b.error_rate,
            Metric::CpuPct => b.cpu_pct,
            Metric::PoolUtilization => b.pool_utilization,
            Metric::RetryRate => b.retry_rate,
            Metric::SpanQueueDepth => b.span_queue_depth,
        })
        .unwrap_or(0.0)
}

/// The metric families a concern can read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// 99th percentile latency.
    P99,
    /// Error rate.
    ErrorRate,
    /// CPU utilisation.
    CpuPct,
    /// Connection pool utilisation.
    PoolUtilization,
    /// Retry fraction.
    RetryRate,
    /// Queued spans.
    SpanQueueDepth,
}

fn fnv(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in s.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_deterministic_across_calls() {
        let a = generate(42);
        let b = generate(42);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.service, y.service);
            assert_eq!(x.minute, y.minute);
            assert_eq!(x.p99_ms.to_bits(), y.p99_ms.to_bits());
            assert_eq!(x.pool_utilization.to_bits(), y.pool_utilization.to_bits());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let a = generate(1);
        let b = generate(2);
        assert!(a.iter().zip(b.iter()).any(|(x, y)| x.p99_ms != y.p99_ms));
    }

    #[test]
    fn checkout_pool_pins_after_the_deploy() {
        let corpus = generate(42);
        let before = corpus
            .iter()
            .find(|b| b.service == "checkout-api" && b.minute == -5)
            .unwrap();
        let after = corpus
            .iter()
            .find(|b| b.service == "checkout-api" && b.minute == 10)
            .unwrap();

        assert!(before.pool_utilization < 0.5);
        assert!(after.pool_utilization > 0.9);
        // CPU stays flat: the incident is not load.
        assert!((after.cpu_pct - before.cpu_pct).abs() < 10.0);
    }

    #[test]
    fn the_deploy_is_present_exactly_once() {
        let corpus = generate(42);
        let deploys: Vec<_> = corpus
            .iter()
            .flat_map(|b| b.deploys.iter())
            .filter(|d| d.service == "checkout-api")
            .collect();
        assert_eq!(deploys.len(), 1);
        assert_eq!(deploys[0].version, "v2.3.1");
    }

    #[test]
    fn decoys_are_visible_only_in_their_own_metric() {
        let corpus = generate(42);
        let spike = corpus
            .iter()
            .find(|b| b.service == "notification-worker" && b.minute == -6)
            .unwrap();
        assert!(spike.cpu_pct > 85.0);
        // Everything else about the worker stays ordinary.
        assert!(spike.error_rate < 0.01);
        assert!(spike.p99_ms < 260.0);
    }
}
