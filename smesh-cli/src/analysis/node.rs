//! One analyst: a single concern, a single mesh node, a single process.
//!
//! The analyst walks the telemetry corpus in compressed real time, and whenever
//! its own detector stands behind a claim it asserts that claim onto the mesh.
//! It never learns another analyst's evidence — only whether anyone else is
//! asserting the same thing.

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::json;

use smesh_core::{DecayFunction, Network, Node, Signal, SignalType};
use smesh_runtime::{Journal, MeshConfig, RuntimeConfig, SmeshRuntime};

use super::concern::{Assertion, Concern, Finding};
use super::corpus::{self, Bucket, FIRST_MINUTE, LAST_MINUTE};

/// Signal lifetime, in seconds of field time.
///
/// Tuned against the compressed timeline so a transient claim visibly fades
/// within one run while a sustained one survives to the end. Decay is the only
/// thing that retracts a claim: nothing in this protocol sends a retraction.
const SIGNAL_TTL_SECS: f64 = 22.0;

/// Exponential decay constant for emitted claims.
const SIGNAL_DECAY_RATE: f64 = 0.10;

/// Hop budget. Large enough to cross this topology twice over.
const SIGNAL_RADIUS: u32 = 6;

/// How much these analysts trust each other.
///
/// They are a known fleet, so trust is high; relay probability is proportional
/// to it, and an untrusting fleet would simply not gossip.
const FLEET_TRUST: f64 = 0.95;

/// How often, in corpus minutes, a standing finding is re-asserted.
///
/// This is anti-entropy. Relaying is probabilistic, so a single assertion can
/// fail to cross the mesh; re-asserting carries the *accumulated* attester set
/// again and lets a node that missed the first round catch up.
const REASSERT_EVERY_MINUTES: i32 = 4;

/// Configuration for one analyst process.
#[derive(Debug, Clone)]
pub struct AnalystConfig {
    /// This analyst's concern.
    pub concern: Concern,
    /// Address to listen on.
    pub bind: SocketAddr,
    /// Peers to dial.
    pub peers: Vec<SocketAddr>,
    /// Where to write the journal, if recording.
    pub journal: Option<PathBuf>,
    /// Shared run epoch, so every node's timestamps land on one timeline.
    pub run_epoch_ms: i64,
    /// Corpus seed. Must match across the fleet.
    pub seed: u64,
    /// Wall-clock milliseconds per corpus minute.
    pub bucket_ms: u64,
    /// Distinct attesters required before a claim is treated as consensus.
    pub consensus_threshold: usize,
    /// Peers to wait for before starting the timeline.
    pub expect_peers: usize,
    /// Extra time to keep gossiping after the corpus runs out.
    pub settle_ms: u64,
    /// Print progress to stdout.
    pub verbose: bool,
}

/// Run one analyst to completion.
pub async fn run(config: AnalystConfig) -> Result<()> {
    let concern = config.concern;
    let node_id = concern.name().to_string();

    let journal = match &config.journal {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            Journal::create(path, &node_id, Some(node_id.clone()), config.run_epoch_ms)
                .with_context(|| format!("opening journal at {}", path.display()))?
        }
        None => Journal::disabled(),
    };

    // One node per process. It trusts its fellow analysts but has no idea what
    // any of them can see.
    let mut node = Node::named(&node_id);
    for other in Concern::all() {
        if other != concern {
            node.trust_scores
                .insert(other.name().to_string(), FLEET_TRUST);
        }
    }

    let mut network = Network::new();
    network.add_node(node);

    let runtime = Arc::new(
        SmeshRuntime::with_network(
            network,
            RuntimeConfig {
                tick_interval_ms: 100,
            },
        )
        .with_journal(Arc::clone(&journal)),
    );

    let mesh = runtime
        .join_mesh(
            MeshConfig {
                bind_addr: config.bind,
                bootstrap: config.peers.clone(),
                keepalive_interval_ms: 2_000,
                // The topology is the thing under study here, so discovery is
                // off: the mesh stays exactly the shape it was given.
                peer_discovery: false,
                // Written as the journal's first line, before the endpoint can
                // accept anything.
                node_metadata: json!({
                    "concern": concern.name(),
                    "description": concern.description(),
                    "metrics": concern.metrics(),
                    "seed": config.seed,
                    "bucket_ms": config.bucket_ms,
                    "consensus_threshold": config.consensus_threshold,
                    "signal_ttl_secs": SIGNAL_TTL_SECS,
                    "signal_decay_rate": SIGNAL_DECAY_RATE,
                    "signal_radius": SIGNAL_RADIUS,
                    "corpus_first_minute": FIRST_MINUTE,
                    "corpus_last_minute": LAST_MINUTE,
                }),
                ..Default::default()
            },
            &node_id,
        )
        .await?;

    if config.verbose {
        println!(
            "[{}] listening on {} — {}",
            node_id,
            mesh.listen_addr(),
            concern.description()
        );
    }

    // Decay and local diffusion run for as long as we are on the mesh.
    {
        let runtime = Arc::clone(&runtime);
        tokio::spawn(async move { runtime.run().await });
    }

    wait_for_peers(&runtime, &journal, config.expect_peers, &config).await;

    let corpus = corpus::generate(config.seed);
    let mut asserted: BTreeMap<String, Assertion> = BTreeMap::new();
    let mut announced_consensus: BTreeSet<String> = BTreeSet::new();
    let mut last_reassert = FIRST_MINUTE;

    for minute in FIRST_MINUTE..=LAST_MINUTE {
        tokio::time::sleep(Duration::from_millis(config.bucket_ms)).await;

        record_observations(&journal, concern, &corpus, minute);

        let findings = concern.detect(&corpus, minute);
        let reassert = minute - last_reassert >= REASSERT_EVERY_MINUTES;
        if reassert {
            last_reassert = minute;
        }

        for finding in findings {
            let hash_key =
                String::from_utf8_lossy(&finding.assertion.canonical_bytes()).to_string();
            let is_new = !asserted.contains_key(&hash_key);

            if is_new || reassert {
                asserted.insert(hash_key, finding.assertion.clone());
                assert_finding(&runtime, &journal, &node_id, &finding, minute, is_new).await;
            }
        }

        check_consensus(
            &runtime,
            &journal,
            config.consensus_threshold,
            &mut announced_consensus,
            minute,
            config.verbose,
        )
        .await;
    }

    // Keep gossiping after the corpus ends so in-flight claims converge.
    let settle_steps = (config.settle_ms / 250).max(1);
    for _ in 0..settle_steps {
        tokio::time::sleep(Duration::from_millis(250)).await;
        check_consensus(
            &runtime,
            &journal,
            config.consensus_threshold,
            &mut announced_consensus,
            LAST_MINUTE,
            config.verbose,
        )
        .await;
    }

    record_summary(&runtime, &journal, &node_id, config.consensus_threshold).await;

    runtime.shutdown().await;
    mesh.shutdown().await;

    Ok(())
}

/// Block until the expected peers show up, or give up and proceed alone.
async fn wait_for_peers(
    runtime: &SmeshRuntime,
    journal: &Journal,
    expect: usize,
    config: &AnalystConfig,
) {
    if expect == 0 {
        return;
    }

    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        let connected = runtime.peers().connected_count().await;
        if connected >= expect {
            journal.record("mesh_ready", json!({ "connected_peers": connected }));
            if config.verbose {
                println!(
                    "[{}] mesh ready, {connected} peer(s)",
                    config.concern.name()
                );
            }
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            journal.record(
                "mesh_degraded",
                json!({ "connected_peers": connected, "expected": expect }),
            );
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Log exactly what this concern can see this minute, and nothing else.
fn record_observations(journal: &Journal, concern: Concern, corpus: &[Bucket], minute: i32) {
    let readings: Vec<serde_json::Value> = corpus
        .iter()
        .filter(|b| b.minute == minute)
        .map(|b| {
            let mut reading = serde_json::Map::new();
            reading.insert("service".into(), json!(b.service));
            for metric in concern.metrics() {
                let value = match *metric {
                    "p99_ms" => json!(b.p99_ms),
                    "error_rate" => json!(b.error_rate),
                    "cpu_pct" => json!(b.cpu_pct),
                    "pool_utilization" => json!(b.pool_utilization),
                    "retry_rate" => json!(b.retry_rate),
                    "span_queue_depth" => json!(b.span_queue_depth),
                    "deploys" => json!(b.deploys),
                    _ => continue,
                };
                reading.insert((*metric).to_string(), value);
            }
            serde_json::Value::Object(reading)
        })
        .collect();

    journal.record(
        "observation",
        json!({ "minute": minute, "readings": readings }),
    );
}

/// Put a finding onto the mesh as a signal.
async fn assert_finding(
    runtime: &SmeshRuntime,
    journal: &Journal,
    node_id: &str,
    finding: &Finding,
    minute: i32,
    is_new: bool,
) {
    journal.record(
        "finding",
        json!({
            "minute": minute,
            "subject": finding.assertion.subject,
            "claim": finding.assertion.claim,
            "confidence": finding.confidence,
            "evidence": finding.evidence,
            "first_time": is_new,
        }),
    );

    // The payload is the assertion and nothing else, so two analysts that reach
    // the same conclusion produce identical bytes.
    //
    // `.correlatable()` is what makes independent analysts converge: it keeps
    // the origin out of the content hash, so the same conclusion reached from
    // different evidence lands on the same signal. The origin is still stamped
    // at emit time for attribution.
    let signal = Signal::builder(SignalType::Alert)
        .correlatable()
        .payload(finding.assertion.canonical_bytes())
        .intensity(1.0)
        .confidence(finding.confidence)
        .ttl(SIGNAL_TTL_SECS)
        .decay_rate(SIGNAL_DECAY_RATE)
        .decay_function(DecayFunction::Exponential)
        .radius(SIGNAL_RADIUS)
        .build();

    runtime.emit(signal, node_id).await;
}

/// Look for claims that enough distinct parties now attest to.
async fn check_consensus(
    runtime: &SmeshRuntime,
    journal: &Journal,
    threshold: usize,
    announced: &mut BTreeSet<String>,
    minute: i32,
    verbose: bool,
) {
    let network = runtime.network();
    let network = network.read().await;

    for signal in network.field.signals.values() {
        // Signatures, not the local name list: this number is the claim.
        let attesters = signal.verified_attesters();
        if attesters.len() < threshold || announced.contains(&signal.origin_hash) {
            continue;
        }

        announced.insert(signal.origin_hash.clone());

        let assertion: Option<Assertion> = serde_json::from_slice(&signal.payload).ok();
        let subject = assertion
            .as_ref()
            .map(|a| a.subject.clone())
            .unwrap_or_default();

        journal.record(
            "consensus_reached",
            json!({
                "hash": signal.origin_hash,
                "minute": minute,
                "subject": subject,
                "claim": assertion.as_ref().map(|a| a.claim.clone()),
                "attesters": attesters,
                "attester_count": attesters.len(),
                "threshold": threshold,
                "confidence": signal.confidence,
                "intensity": signal.current_intensity,
            }),
        );

        if verbose {
            println!(
                "  [consensus] {subject} — {} concerns concur: {}",
                attesters.len(),
                attesters.join(", ")
            );
        }
    }
}

/// Final state of this node's field, as it saw things.
async fn record_summary(
    runtime: &SmeshRuntime,
    journal: &Journal,
    node_id: &str,
    threshold: usize,
) {
    let stats = runtime.stats().await;
    let network = runtime.network();
    let network = network.read().await;

    let mut claims: Vec<serde_json::Value> = network
        .field
        .signals
        .values()
        .map(|signal| {
            // Signatures, not the local name list: this number is the claim.
            let attesters = signal.verified_attesters();
            let assertion: Option<Assertion> = serde_json::from_slice(&signal.payload).ok();
            json!({
                "hash": signal.origin_hash,
                "subject": assertion.as_ref().map(|a| a.subject.clone()),
                "claim": assertion.as_ref().map(|a| a.claim.clone()),
                "attesters": attesters,
                "attester_count": attesters.len(),
                "consensus": attesters.len() >= threshold,
                "confidence": signal.confidence,
                "intensity": signal.current_intensity,
                "hops": signal.hops,
            })
        })
        .collect();

    claims.sort_by(|a, b| {
        b["attester_count"]
            .as_u64()
            .cmp(&a["attester_count"].as_u64())
    });

    let node_stats = network.get_node(node_id).map(|n| {
        json!({
            "signals_emitted": n.stats.signals_emitted,
            "signals_sensed": n.stats.signals_sensed,
            "signals_relayed": n.stats.signals_relayed,
            "signals_reinforced": n.stats.signals_reinforced,
        })
    });

    journal.record(
        "node_stopped",
        json!({
            "ticks": stats.tick_count,
            "peers_known": stats.peer_count,
            "peers_connected": stats.connected_peers,
            "active_signals": stats.active_signals,
            "node_stats": node_stats,
            "claims": claims,
        }),
    );
}
