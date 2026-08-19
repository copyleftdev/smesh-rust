//! Analyst concerns: one metric family each, deliberately partial.
//!
//! Every concern reads a different slice of the same telemetry and can only
//! reason about what it sees. That partiality is the point. A concern is not a
//! weak detector to be improved; it is one witness, and the mesh's job is to
//! decide which claims survive being heard by several of them.
//!
//! Concerns never talk to each other directly and never share evidence. They
//! meet only as signals in the field.

use serde::{Deserialize, Serialize};

use super::corpus::{baseline_of, Bucket, Metric};

/// Buckets a detector averages over before it will speak.
const WINDOW: usize = 3;

/// Minutes after a deploy in which a shift still counts as related.
const DEPLOY_BLAST_RADIUS: i32 = 5;

/// The claim carried on the wire.
///
/// This is the *entire* payload of an emitted signal, and it is deliberately
/// tiny: a subject and what is being said about it, nothing else. Two analysts
/// that independently reach the same conclusion serialise byte-identical
/// payloads, so the protocol's content-addressed hash puts them on the same
/// signal and treats the second one as corroboration rather than noise.
///
/// Evidence deliberately stays out. It differs per concern and would make every
/// hash unique, destroying the correlation this whole design depends on — so
/// evidence goes to the journal, and only the assertion goes to the mesh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assertion {
    /// The service being described.
    pub subject: String,
    /// What is claimed about it.
    pub claim: String,
}

impl Assertion {
    /// Canonical bytes for hashing and transmission.
    ///
    /// Field order is fixed by the struct definition, so this is stable across
    /// processes and runs.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("assertion serialises")
    }
}

/// One measurement supporting a finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Which metric.
    pub metric: String,
    /// Normal value for this service.
    pub baseline: f64,
    /// What was actually observed, averaged over the window.
    pub observed: f64,
    /// Observed over baseline.
    pub ratio: f64,
    /// Minutes the window covers, relative to the incident.
    pub window: String,
}

/// A conclusion one concern reached on its own.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// What is being claimed.
    pub assertion: Assertion,
    /// How sure this concern is, from its own evidence alone.
    pub confidence: f64,
    /// Why it thinks so.
    pub evidence: Vec<Evidence>,
}

/// The five analyst concerns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Concern {
    /// Request latency percentiles.
    Latency,
    /// Error rates and status codes.
    Errors,
    /// Resource and pool saturation.
    Saturation,
    /// Span-level retry and queueing behaviour.
    Traces,
    /// Releases and config changes.
    Deploys,
}

impl Concern {
    /// Every concern, in the order the demo arranges them.
    pub fn all() -> Vec<Concern> {
        vec![
            Concern::Latency,
            Concern::Errors,
            Concern::Saturation,
            Concern::Traces,
            Concern::Deploys,
        ]
    }

    /// Parse a concern by name.
    pub fn parse(name: &str) -> Option<Concern> {
        match name.to_ascii_lowercase().as_str() {
            "latency" => Some(Concern::Latency),
            "errors" => Some(Concern::Errors),
            "saturation" => Some(Concern::Saturation),
            "traces" => Some(Concern::Traces),
            "deploys" => Some(Concern::Deploys),
            _ => None,
        }
    }

    /// Stable lowercase name, used as the node id on the mesh.
    pub fn name(&self) -> &'static str {
        match self {
            Concern::Latency => "latency",
            Concern::Errors => "errors",
            Concern::Saturation => "saturation",
            Concern::Traces => "traces",
            Concern::Deploys => "deploys",
        }
    }

    /// What this concern is allowed to look at.
    pub fn metrics(&self) -> &'static [&'static str] {
        match self {
            Concern::Latency => &["p99_ms"],
            Concern::Errors => &["error_rate"],
            Concern::Saturation => &["pool_utilization", "cpu_pct"],
            Concern::Traces => &["retry_rate", "span_queue_depth"],
            Concern::Deploys => &["deploys", "error_rate"],
        }
    }

    /// One line describing what this analyst does.
    pub fn description(&self) -> &'static str {
        match self {
            Concern::Latency => "watches p99 latency for sustained regressions",
            Concern::Errors => "watches error rates for elevated failure",
            Concern::Saturation => "watches pool and CPU saturation",
            Concern::Traces => "watches retry amplification and span queueing",
            Concern::Deploys => "correlates releases with what follows them",
        }
    }

    /// Run this concern's detector over everything observed up to `now_minute`.
    ///
    /// Returns every finding the concern currently stands behind. Detectors are
    /// pure: the same telemetry always yields the same findings, so a rerun of
    /// the corpus is reproducible even though the mesh around it is not.
    pub fn detect(&self, buckets: &[Bucket], now_minute: i32) -> Vec<Finding> {
        let visible: Vec<&Bucket> = buckets.iter().filter(|b| b.minute <= now_minute).collect();

        match self {
            Concern::Latency => threshold_findings(
                &visible,
                now_minute,
                Metric::P99,
                "p99_ms",
                |b| b.p99_ms,
                Comparison::RatioAtLeast(1.8),
            ),
            Concern::Errors => threshold_findings(
                &visible,
                now_minute,
                Metric::ErrorRate,
                "error_rate",
                |b| b.error_rate,
                Comparison::RatioAtLeast(3.0),
            ),
            Concern::Saturation => {
                let mut findings = threshold_findings(
                    &visible,
                    now_minute,
                    Metric::PoolUtilization,
                    "pool_utilization",
                    |b| b.pool_utilization,
                    Comparison::AbsoluteAtLeast(0.90),
                );
                findings.extend(threshold_findings(
                    &visible,
                    now_minute,
                    Metric::CpuPct,
                    "cpu_pct",
                    |b| b.cpu_pct,
                    Comparison::AbsoluteAtLeast(85.0),
                ));
                merge_by_subject(findings)
            }
            Concern::Traces => {
                let mut findings = threshold_findings(
                    &visible,
                    now_minute,
                    Metric::RetryRate,
                    "retry_rate",
                    |b| b.retry_rate,
                    Comparison::AbsoluteAtLeast(0.10),
                );
                findings.extend(threshold_findings(
                    &visible,
                    now_minute,
                    Metric::SpanQueueDepth,
                    "span_queue_depth",
                    |b| b.span_queue_depth,
                    Comparison::AbsoluteAtLeast(20.0),
                ));
                merge_by_subject(findings)
            }
            Concern::Deploys => deploy_findings(&visible, now_minute),
        }
    }
}

/// How a detector decides a window is abnormal.
enum Comparison {
    /// Observed is at least this multiple of the service's baseline.
    RatioAtLeast(f64),
    /// Observed is at least this absolute value.
    AbsoluteAtLeast(f64),
}

/// Average the last [`WINDOW`] buckets per service and flag what crosses.
fn threshold_findings(
    visible: &[&Bucket],
    now_minute: i32,
    metric: Metric,
    metric_name: &str,
    extract: fn(&Bucket) -> f64,
    comparison: Comparison,
) -> Vec<Finding> {
    let mut findings = Vec::new();

    let mut services: Vec<&str> = visible.iter().map(|b| b.service.as_str()).collect();
    services.sort_unstable();
    services.dedup();

    for service in services {
        let mut window: Vec<&&Bucket> = visible.iter().filter(|b| b.service == service).collect();
        window.sort_by_key(|b| b.minute);
        let window: Vec<&&Bucket> = window.into_iter().rev().take(WINDOW).collect();

        if window.len() < WINDOW {
            continue;
        }

        let observed = window.iter().map(|b| extract(b)).sum::<f64>() / window.len() as f64;
        let baseline = baseline_of(service, metric);
        let ratio = if baseline > 0.0 {
            observed / baseline
        } else {
            0.0
        };

        let (crossed, strength) = match comparison {
            Comparison::RatioAtLeast(limit) => (ratio >= limit, (ratio / limit).min(3.0)),
            Comparison::AbsoluteAtLeast(limit) => (observed >= limit, (observed / limit).min(3.0)),
        };

        if !crossed {
            continue;
        }

        let earliest = window.iter().map(|b| b.minute).min().unwrap_or(now_minute);

        findings.push(Finding {
            assertion: Assertion {
                subject: service.to_string(),
                claim: "degraded".to_string(),
            },
            // A single concern is never certain. Even a blatant reading caps
            // below the threshold that would let one witness carry a verdict.
            confidence: (0.35 + (strength - 1.0) * 0.15).clamp(0.35, 0.72),
            evidence: vec![Evidence {
                metric: metric_name.to_string(),
                baseline,
                observed,
                ratio,
                window: format!("T{earliest:+}..T{now_minute:+}"),
            }],
        });
    }

    findings
}

/// Flag a deployed service when a shift follows its release closely enough.
///
/// The control case matters as much as the positive one: a deploy with nothing
/// after it must not be blamed, or this concern would indict every release.
fn deploy_findings(visible: &[&Bucket], now_minute: i32) -> Vec<Finding> {
    let mut findings = Vec::new();

    for bucket in visible {
        for deploy in &bucket.deploys {
            let deployed_at = bucket.minute;
            let after: Vec<&&Bucket> = visible
                .iter()
                .filter(|b| {
                    b.service == deploy.service
                        && b.minute > deployed_at
                        && b.minute <= deployed_at + DEPLOY_BLAST_RADIUS
                })
                .collect();

            if after.len() < WINDOW {
                continue;
            }

            let observed = after.iter().map(|b| b.error_rate).sum::<f64>() / after.len() as f64;
            let baseline = baseline_of(&deploy.service, Metric::ErrorRate);
            let ratio = if baseline > 0.0 {
                observed / baseline
            } else {
                0.0
            };

            if ratio < 3.0 {
                continue;
            }

            findings.push(Finding {
                assertion: Assertion {
                    subject: deploy.service.clone(),
                    claim: "degraded".to_string(),
                },
                confidence: (0.35 + (ratio / 3.0 - 1.0) * 0.15).clamp(0.35, 0.72),
                evidence: vec![Evidence {
                    metric: format!("deploy {} ({})", deploy.version, deploy.change),
                    baseline,
                    observed,
                    ratio,
                    window: format!("T{deployed_at:+}..T{now_minute:+}"),
                }],
            });
        }
    }

    merge_by_subject(findings)
}

/// Collapse several readings about one subject into one finding.
fn merge_by_subject(findings: Vec<Finding>) -> Vec<Finding> {
    let mut merged: Vec<Finding> = Vec::new();

    for finding in findings {
        if let Some(existing) = merged.iter_mut().find(|f| f.assertion == finding.assertion) {
            existing.confidence = existing.confidence.max(finding.confidence);
            existing.evidence.extend(finding.evidence);
        } else {
            merged.push(finding);
        }
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::corpus::{generate, FIRST_MINUTE, LAST_MINUTE};

    /// Everything a concern claims at any point as the run plays out.
    ///
    /// This is what the mesh actually sees: an analyst asserts a finding while
    /// its evidence is live, so a transient anomaly is claimed at the time and
    /// simply stops being re-asserted afterwards.
    fn subjects_over_run(concern: Concern) -> Vec<String> {
        let corpus = generate(42);
        let mut subjects: Vec<String> = (FIRST_MINUTE..=LAST_MINUTE)
            .flat_map(|minute| concern.detect(&corpus, minute))
            .map(|f| f.assertion.subject)
            .collect();
        subjects.sort();
        subjects.dedup();
        subjects
    }

    fn subjects_at(concern: Concern, minute: i32) -> Vec<String> {
        let corpus = generate(42);
        let mut subjects: Vec<String> = concern
            .detect(&corpus, minute)
            .into_iter()
            .map(|f| f.assertion.subject)
            .collect();
        subjects.sort();
        subjects.dedup();
        subjects
    }

    #[test]
    fn latency_alone_is_noisy() {
        // Four subjects across the run, only one of which is the real cause.
        // This is why a single concern cannot be trusted to call an incident.
        assert_eq!(
            subjects_over_run(Concern::Latency),
            vec![
                "checkout-api",
                "edge-gateway",
                "payments-api",
                "session-store"
            ]
        );
    }

    #[test]
    fn errors_would_blame_the_victim() {
        // Errors sees payments-api failing loudly. On its own it points at the
        // service that is suffering, not the one that is causing it.
        assert_eq!(
            subjects_over_run(Concern::Errors),
            vec!["checkout-api", "payments-api"]
        );
    }

    #[test]
    fn saturation_sees_the_pool_and_one_red_herring() {
        assert_eq!(
            subjects_over_run(Concern::Saturation),
            vec!["checkout-api", "notification-worker"]
        );
    }

    #[test]
    fn traces_see_retry_amplification() {
        assert_eq!(
            subjects_over_run(Concern::Traces),
            vec!["checkout-api", "payments-api"]
        );
    }

    #[test]
    fn deploys_blames_the_bad_release_and_not_the_benign_one() {
        // inventory-svc also shipped, and must not be implicated.
        assert_eq!(subjects_over_run(Concern::Deploys), vec!["checkout-api"]);
    }

    #[test]
    fn decoys_are_claimed_while_live_and_dropped_afterwards() {
        // The session-store blip is real while it is happening...
        assert!(subjects_at(Concern::Latency, -12).contains(&"session-store".to_string()));
        // ...and no longer claimed once it passes. Nothing retracts it on the
        // mesh, so the only thing that removes it is decay.
        assert!(!subjects_at(Concern::Latency, 19).contains(&"session-store".to_string()));

        assert!(subjects_at(Concern::Saturation, -5).contains(&"notification-worker".to_string()));
        assert!(!subjects_at(Concern::Saturation, 19).contains(&"notification-worker".to_string()));
    }

    #[test]
    fn corroboration_tally_separates_cause_from_symptom_from_noise() {
        let corpus = generate(42);
        let mut tally: std::collections::BTreeMap<String, Vec<&str>> = Default::default();

        for concern in Concern::all() {
            let mut seen: Vec<String> = Vec::new();
            for minute in FIRST_MINUTE..=LAST_MINUTE {
                for finding in concern.detect(&corpus, minute) {
                    if !seen.contains(&finding.assertion.subject) {
                        seen.push(finding.assertion.subject.clone());
                        tally
                            .entry(finding.assertion.subject)
                            .or_default()
                            .push(concern.name());
                    }
                }
            }
        }

        // The whole demo rests on this shape: the cause is corroborated by
        // every concern, the casualty by some, the decoys by exactly one.
        assert_eq!(tally["checkout-api"].len(), 5, "root cause");
        assert_eq!(tally["payments-api"].len(), 3, "downstream casualty");
        assert_eq!(tally["edge-gateway"].len(), 1, "weak downstream echo");
        assert_eq!(tally["session-store"].len(), 1, "planted decoy");
        assert_eq!(tally["notification-worker"].len(), 1, "planted decoy");
    }

    #[test]
    fn identical_assertions_serialise_identically() {
        // Content addressing depends on this: two concerns reaching the same
        // conclusion must produce the same bytes, or they will never correlate.
        let a = Assertion {
            subject: "checkout-api".to_string(),
            claim: "degraded".to_string(),
        };
        let b = Assertion {
            subject: "checkout-api".to_string(),
            claim: "degraded".to_string(),
        };
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn nothing_is_claimed_before_the_evidence_exists() {
        let corpus = generate(42);
        // Twenty minutes before the deploy, the only thing anyone can see is
        // the pre-incident decoys, and never checkout-api.
        for concern in Concern::all() {
            for finding in concern.detect(&corpus, -15) {
                assert_ne!(finding.assertion.subject, "checkout-api");
            }
        }
    }
}
