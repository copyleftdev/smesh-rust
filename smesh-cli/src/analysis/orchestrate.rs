//! Run the analysis mesh as real, separate processes.
//!
//! The orchestrator is a launcher, not a coordinator. It picks the ports and
//! the shared run epoch, starts one OS process per concern, and then gets out
//! of the way — it holds no shared state the analysts can reach, relays nothing
//! between them, and makes no decisions on their behalf. Everything the
//! analysts learn from each other crosses a real QUIC socket.
//!
//! When the run ends it merges the per-node journals into one ordered timeline
//! and writes a manifest describing the run.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{bail, Context, Result};
use serde_json::json;
use tokio::process::Command;

use smesh_runtime::JournalEvent;

use super::concern::Concern;
use super::corpus::{FIRST_MINUTE, LAST_MINUTE};

/// The mesh topology, as (from, to) pairs dialled once each.
///
/// A ring with one chord. Deliberately not a full mesh: with everyone directly
/// connected there is no relaying, no hop count and nothing to watch diffuse.
/// The chord keeps the diameter at two so gossip still converges promptly.
const TOPOLOGY: &[(&str, &str)] = &[
    ("latency", "errors"),
    ("latency", "saturation"),
    ("errors", "saturation"),
    ("saturation", "traces"),
    ("traces", "deploys"),
    ("deploys", "latency"),
];

/// Settings for one orchestrated run.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Directory for journals and the merged timeline.
    pub out_dir: PathBuf,
    /// First TCP/UDP port; concerns take consecutive ports from here.
    pub base_port: u16,
    /// Corpus seed, shared by every analyst.
    pub seed: u64,
    /// Wall-clock milliseconds per corpus minute.
    pub bucket_ms: u64,
    /// Distinct attesters required for consensus.
    pub consensus_threshold: usize,
    /// Extra gossip time after the corpus ends.
    pub settle_ms: u64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("runs/latest"),
            base_port: 9301,
            seed: 42,
            bucket_ms: 700,
            consensus_threshold: 4,
            settle_ms: 6_000,
        }
    }
}

/// Where each concern listens.
fn addresses(base_port: u16) -> BTreeMap<&'static str, SocketAddr> {
    Concern::all()
        .into_iter()
        .enumerate()
        .map(|(i, concern)| {
            let addr: SocketAddr = format!("127.0.0.1:{}", base_port + i as u16)
                .parse()
                .expect("valid loopback address");
            (concern.name(), addr)
        })
        .collect()
}

/// Which peers a given concern dials.
fn bootstrap_for(concern: &str, addrs: &BTreeMap<&'static str, SocketAddr>) -> Vec<SocketAddr> {
    TOPOLOGY
        .iter()
        .filter(|(from, _)| *from == concern)
        .filter_map(|(_, to)| addrs.get(to).copied())
        .collect()
}

/// How many peers a concern should end up connected to.
fn degree_of(concern: &str) -> usize {
    TOPOLOGY
        .iter()
        .filter(|(from, to)| *from == concern || *to == concern)
        .count()
}

/// Launch the fleet, wait for it, and merge the results.
pub async fn run(config: RunConfig) -> Result<PathBuf> {
    let exe = std::env::current_exe().context("locating the smesh binary")?;
    std::fs::create_dir_all(&config.out_dir)
        .with_context(|| format!("creating {}", config.out_dir.display()))?;

    let addrs = addresses(config.base_port);
    let run_epoch_ms = chrono::Utc::now().timestamp_millis();

    let corpus_minutes = (LAST_MINUTE - FIRST_MINUTE + 1) as u64;
    let expected_secs = (corpus_minutes * config.bucket_ms + config.settle_ms) / 1000;

    println!("╭─ SMESH multi-concern analysis mesh");
    println!(
        "│  {} analyst processes, one per concern",
        Concern::all().len()
    );
    println!("│  topology   ring + chord, {} links", TOPOLOGY.len());
    println!(
        "│  corpus     seed {} · {corpus_minutes} minutes",
        config.seed
    );
    println!(
        "│  consensus  {} distinct concerns",
        config.consensus_threshold
    );
    println!("│  journals   {}", config.out_dir.display());
    println!("│  runtime    ~{expected_secs}s");
    println!("╰─\n");

    let mut children = Vec::new();

    for concern in Concern::all() {
        let name = concern.name();
        let bind = addrs[name];
        let peers = bootstrap_for(name, &addrs);
        let journal = config.out_dir.join(format!("{name}.jsonl"));

        let mut command = Command::new(&exe);
        command
            .arg("analyze")
            .arg("--concern")
            .arg(name)
            .arg("--bind")
            .arg(bind.to_string())
            .arg("--journal")
            .arg(&journal)
            .arg("--run-epoch")
            .arg(run_epoch_ms.to_string())
            .arg("--seed")
            .arg(config.seed.to_string())
            .arg("--bucket-ms")
            .arg(config.bucket_ms.to_string())
            .arg("--consensus-threshold")
            .arg(config.consensus_threshold.to_string())
            .arg("--expect-peers")
            .arg(degree_of(name).to_string())
            .arg("--settle-ms")
            .arg(config.settle_ms.to_string());

        for peer in &peers {
            command.arg("--peer").arg(peer.to_string());
        }

        // Children inherit stdout so their progress is visible live.
        let child = command
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawning analyst {name}"))?;

        println!(
            "  spawned {name:<11} pid {:<8} {bind}  dials {}",
            child.id().unwrap_or(0),
            if peers.is_empty() {
                "-".to_string()
            } else {
                peers
                    .iter()
                    .map(|p| p.port().to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            }
        );

        children.push((name, child));
    }

    println!();

    let mut failures = Vec::new();
    for (name, mut child) in children {
        let status = child
            .wait()
            .await
            .with_context(|| format!("waiting for analyst {name}"))?;
        if !status.success() {
            failures.push(format!("{name} exited with {status}"));
        }
    }

    if !failures.is_empty() {
        bail!("analyst processes failed: {}", failures.join("; "));
    }

    let merged = merge_journals(&config, run_epoch_ms)?;
    Ok(merged)
}

/// Merge the per-node journals into one ordered timeline plus a manifest.
///
/// Ordering is by `t_ms` — which every node measured against the same epoch —
/// then by node and per-node sequence, so the result is deterministic and two
/// events in the same millisecond never swap places between merges.
fn merge_journals(config: &RunConfig, run_epoch_ms: i64) -> Result<PathBuf> {
    let mut events: Vec<JournalEvent> = Vec::new();
    let mut per_node: BTreeMap<String, usize> = BTreeMap::new();

    for concern in Concern::all() {
        let path = config.out_dir.join(format!("{}.jsonl", concern.name()));
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading journal {}", path.display()))?;

        let mut count = 0;
        for (lineno, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let event: JournalEvent = serde_json::from_str(line).with_context(|| {
                format!(
                    "parsing {}:{} as a journal event",
                    path.display(),
                    lineno + 1
                )
            })?;
            events.push(event);
            count += 1;
        }
        per_node.insert(concern.name().to_string(), count);
    }

    events.sort_by(|a, b| {
        a.t_ms
            .cmp(&b.t_ms)
            .then_with(|| a.node.cmp(&b.node))
            .then_with(|| a.seq.cmp(&b.seq))
    });

    let merged_path = config.out_dir.join("run.jsonl");
    let mut merged = String::with_capacity(events.len() * 200);
    for event in &events {
        merged.push_str(&serde_json::to_string(event)?);
        merged.push('\n');
    }
    std::fs::write(&merged_path, merged)
        .with_context(|| format!("writing {}", merged_path.display()))?;

    let manifest = json!({
        "run_epoch_ms": run_epoch_ms,
        "seed": config.seed,
        "bucket_ms": config.bucket_ms,
        "consensus_threshold": config.consensus_threshold,
        "settle_ms": config.settle_ms,
        "corpus": {
            "first_minute": FIRST_MINUTE,
            "last_minute": LAST_MINUTE,
        },
        "nodes": Concern::all()
            .into_iter()
            .map(|c| json!({
                "id": c.name(),
                "concern": c.name(),
                "description": c.description(),
                "metrics": c.metrics(),
                "addr": addresses(config.base_port)[c.name()].to_string(),
                "events": per_node.get(c.name()).copied().unwrap_or(0),
            }))
            .collect::<Vec<_>>(),
        "topology": TOPOLOGY
            .iter()
            .map(|(a, b)| json!({"from": a, "to": b}))
            .collect::<Vec<_>>(),
        "events": events.len(),
        "duration_ms": events.last().map(|e| e.t_ms).unwrap_or(0),
    });

    let manifest_path = config.out_dir.join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("writing {}", manifest_path.display()))?;

    print_summary(&events, config);

    // The visualization is only as honest as this file, so check it here
    // rather than discovering a gap when the replay looks wrong.
    let report = super::validate::validate(&events);
    super::validate::print_report(&report);

    println!("\n  merged timeline  {}", merged_path.display());
    println!("  manifest         {}", manifest_path.display());

    Ok(merged_path)
}

/// Report what the mesh concluded, from the journal alone.
fn print_summary(events: &[JournalEvent], config: &RunConfig) {
    // Who reached consensus on what, and when.
    let mut consensus: BTreeMap<String, (i64, Vec<String>, Vec<String>)> = BTreeMap::new();
    for event in events.iter().filter(|e| e.kind == "consensus_reached") {
        let subject = event.data["subject"].as_str().unwrap_or("?").to_string();
        let attesters: Vec<String> = event.data["attesters"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let entry = consensus
            .entry(subject)
            .or_insert((event.t_ms, attesters.clone(), Vec::new()));
        entry.0 = entry.0.min(event.t_ms);
        if !entry.2.contains(&event.node) {
            entry.2.push(event.node.clone());
        }
        if attesters.len() > entry.1.len() {
            entry.1 = attesters;
        }
    }

    // Everything that was ever claimed, and by how many concerns.
    let mut claimed: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for event in events.iter().filter(|e| e.kind == "finding") {
        if let Some(subject) = event.data["subject"].as_str() {
            let entry = claimed.entry(subject.to_string()).or_default();
            if !entry.contains(&event.node) {
                entry.push(event.node.clone());
            }
        }
    }

    println!("\n╭─ what the mesh concluded");
    for (subject, finders) in &claimed {
        let reached = consensus.get(subject);
        let verdict = match reached {
            Some((t_ms, attesters, nodes)) => format!(
                "CONSENSUS at {:.1}s · {} attesters · seen by {} node(s)",
                *t_ms as f64 / 1000.0,
                attesters.len(),
                nodes.len()
            ),
            None => format!(
                "no consensus ({}/{} concerns)",
                finders.len(),
                config.consensus_threshold
            ),
        };
        println!("│  {subject:<21} {verdict}");
        println!("│  {:<21} claimed by {}", "", finders.join(", "));
    }
    println!("╰─");

    let counts = tally_kinds(events);
    println!("\n╭─ journal");
    println!("│  {} events across {} nodes", events.len(), 5);
    for (kind, count) in counts {
        println!("│  {kind:<20} {count}");
    }
    println!("╰─");
}

fn tally_kinds(events: &[JournalEvent]) -> Vec<(String, usize)> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for event in events {
        *counts.entry(event.kind.clone()).or_default() += 1;
    }
    let mut counts: Vec<(String, usize)> = counts.into_iter().collect();
    counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_is_connected_and_not_a_full_mesh() {
        let names: Vec<&str> = Concern::all().into_iter().map(|c| c.name()).collect();

        // Every concern appears.
        for name in &names {
            assert!(degree_of(name) >= 2, "{name} is underconnected");
        }

        // A full mesh of 5 would be 10 links; this is deliberately sparser so
        // signals have to be relayed to cross it.
        assert!(TOPOLOGY.len() < 10);

        // Each link is dialled by exactly one side.
        let mut seen = std::collections::HashSet::new();
        for (a, b) in TOPOLOGY {
            assert!(seen.insert((a, b)), "duplicate link {a}->{b}");
            assert!(
                !seen.contains(&(b, a)),
                "link {a}-{b} dialled from both ends"
            );
        }
    }

    #[test]
    fn every_concern_gets_a_distinct_port() {
        let addrs = addresses(9301);
        let mut ports: Vec<u16> = addrs.values().map(|a| a.port()).collect();
        ports.sort_unstable();
        ports.dedup();
        assert_eq!(ports.len(), Concern::all().len());
    }
}
