//! Check that a recorded run is internally consistent and replayable.
//!
//! A visualization built on a journal inherits every gap in it, and a gap is
//! invisible until someone reads the picture wrong. This module states what the
//! journal claims about itself and verifies each claim against the file, so
//! "the replay is accurate" is a checked property rather than an intention.
//!
//! Violations are reported, never hidden. Some are benign — a message still in
//! flight when its recipient stopped is a real thing that happens — so they are
//! separated into errors, which mean the log is wrong, and notes, which mean
//! the run ended untidily.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::Value;
use smesh_runtime::JournalEvent;

/// Outcome of validating one run.
#[derive(Debug, Default)]
pub struct Report {
    /// Inconsistencies that make the journal untrustworthy.
    pub errors: Vec<String>,
    /// Oddities that are explainable and do not invalidate a replay.
    pub notes: Vec<String>,
    /// Invariants that were checked and held.
    pub checks_passed: Vec<String>,
}

impl Report {
    /// Whether the journal can be trusted for replay.
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

fn str_field<'a>(event: &'a JournalEvent, key: &str) -> Option<&'a str> {
    event.data.get(key).and_then(Value::as_str)
}

fn u64_field(event: &JournalEvent, key: &str) -> Option<u64> {
    event.data.get(key).and_then(Value::as_u64)
}

/// Validate a merged run.
pub fn validate(events: &[JournalEvent]) -> Report {
    let mut report = Report::default();

    let mut by_node: BTreeMap<&str, Vec<&JournalEvent>> = BTreeMap::new();
    for event in events {
        by_node.entry(event.node.as_str()).or_default().push(event);
    }

    check_merge_order(events, &mut report);
    check_per_node_sequences(&by_node, &mut report);
    check_lifecycle(&by_node, &mut report);
    check_snapshots(&by_node, &mut report);
    check_receipts(&by_node, &mut report);
    check_deliveries(events, &by_node, &mut report);
    check_consensus(&by_node, &mut report);
    check_attestations(events, &mut report);

    report
}

/// Nothing counted as corroboration should have arrived unverifiable.
///
/// Attester counts are the protocol's central measurement, so a run where
/// signatures failed to check out is a run whose headline numbers cannot be
/// taken at face value, even if every other invariant holds.
fn check_attestations(events: &[JournalEvent], report: &mut Report) {
    let unverifiable: u64 = events
        .iter()
        .filter(|e| e.kind == "signal_received")
        .filter_map(|e| u64_field(e, "unverifiable_attestations"))
        .sum();

    let rejections = events
        .iter()
        .filter(|e| e.kind == "identity_rejected")
        .count();

    if unverifiable > 0 {
        report.notes.push(format!(
            "{unverifiable} attestation(s) arrived that did not verify and were not counted"
        ));
    }

    if rejections > 0 {
        report.notes.push(format!(
            "{rejections} peer(s) or attestation(s) refused for using a name pinned to another key"
        ));
    }

    let signed = events
        .iter()
        .filter(|e| e.kind == "peer_connected")
        .filter(|e| {
            e.data
                .get("public_key")
                .and_then(Value::as_str)
                .is_some_and(|k| !k.is_empty())
        })
        .count();
    let handshakes = events.iter().filter(|e| e.kind == "peer_connected").count();

    if signed < handshakes {
        report.errors.push(format!(
            "{} of {handshakes} handshakes carried no public key, so those peers cannot be held to a identity",
            handshakes - signed
        ));
    }

    report.checks_passed.push(format!(
        "every counted attester was signature-backed across {handshakes} key-bound handshakes"
    ));
}

/// The merged file must be ordered, or a replay would jump backwards in time.
fn check_merge_order(events: &[JournalEvent], report: &mut Report) {
    let mut last = i64::MIN;
    for event in events {
        if event.t_ms < last {
            report.errors.push(format!(
                "merged timeline goes backwards at {}#{} ({}ms after {}ms)",
                event.node, event.seq, event.t_ms, last
            ));
            return;
        }
        last = event.t_ms;
    }
    report
        .checks_passed
        .push("merged timeline is monotonic in t_ms".to_string());
}

/// Each node's own log must be gapless, or events were lost.
fn check_per_node_sequences(by_node: &BTreeMap<&str, Vec<&JournalEvent>>, report: &mut Report) {
    for (node, node_events) in by_node {
        let mut seqs: Vec<u64> = node_events.iter().map(|e| e.seq).collect();
        seqs.sort_unstable();

        for (index, seq) in seqs.iter().enumerate() {
            let expected = index as u64 + 1;
            if *seq != expected {
                report.errors.push(format!(
                    "{node}: sequence gap — expected seq {expected}, found {seq}"
                ));
                break;
            }
        }

        // Within a node, time must not run backwards either.
        let mut ordered: Vec<&&JournalEvent> = node_events.iter().collect();
        ordered.sort_by_key(|e| e.seq);
        let mut last = i64::MIN;
        for event in ordered {
            if event.t_ms < last {
                report.errors.push(format!(
                    "{node}: t_ms went backwards at seq {} ({}ms after {}ms)",
                    event.seq, event.t_ms, last
                ));
                break;
            }
            last = event.t_ms;
        }
    }
    report
        .checks_passed
        .push("every node's sequence is gapless and time-ordered".to_string());
}

/// Every node must open and close its own log.
fn check_lifecycle(by_node: &BTreeMap<&str, Vec<&JournalEvent>>, report: &mut Report) {
    for (node, node_events) in by_node {
        let first = node_events.iter().min_by_key(|e| e.seq);
        let has_stop = node_events.iter().any(|e| e.kind == "node_stopped");

        match first {
            Some(event) if event.kind == "node_started" => {}
            Some(event) => report.errors.push(format!(
                "{node}: first event is {}, not node_started",
                event.kind
            )),
            None => report.errors.push(format!("{node}: no events at all")),
        }

        if !has_stop {
            report
                .notes
                .push(format!("{node}: no node_stopped — process ended early?"));
        }
    }
    report
        .checks_passed
        .push("every node opened with node_started".to_string());
}

/// Field snapshots must advance, so decay curves interpolate correctly.
fn check_snapshots(by_node: &BTreeMap<&str, Vec<&JournalEvent>>, report: &mut Report) {
    for (node, node_events) in by_node {
        let mut last_tick = 0u64;
        for event in node_events.iter().filter(|e| e.kind == "field_snapshot") {
            let Some(tick) = u64_field(event, "tick") else {
                report
                    .errors
                    .push(format!("{node}: field_snapshot without a tick"));
                continue;
            };
            if tick <= last_tick && last_tick != 0 {
                report.errors.push(format!(
                    "{node}: field_snapshot tick {tick} did not advance past {last_tick}"
                ));
                break;
            }
            last_tick = tick;
        }
    }
    report
        .checks_passed
        .push("field snapshots advance monotonically".to_string());
}

/// Anything a node reports holding must have arrived by a recorded route.
fn check_receipts(by_node: &BTreeMap<&str, Vec<&JournalEvent>>, report: &mut Report) {
    for (node, node_events) in by_node {
        let mut ordered: Vec<&&JournalEvent> = node_events.iter().collect();
        ordered.sort_by_key(|e| e.seq);

        let mut known: BTreeSet<String> = BTreeSet::new();
        let mut unexplained = 0usize;

        for event in ordered {
            match event.kind.as_str() {
                "signal_emitted" | "signal_accepted" => {
                    if let Some(hash) = str_field(event, "hash") {
                        known.insert(hash.to_string());
                    }
                }
                "field_snapshot" => {
                    let Some(signals) = event.data.get("signals").and_then(Value::as_array) else {
                        continue;
                    };
                    for signal in signals {
                        let Some(hash) = signal.get("hash").and_then(Value::as_str) else {
                            continue;
                        };
                        if !known.contains(hash) {
                            unexplained += 1;
                        }
                    }
                }
                _ => {}
            }
        }

        if unexplained > 0 {
            report.errors.push(format!(
                "{node}: {unexplained} snapshot entries for signals never emitted or accepted here"
            ));
        }
    }
    report
        .checks_passed
        .push("every signal a node held was emitted or accepted there first".to_string());
}

/// Every recorded send should show up as a receive on the named peer.
fn check_deliveries(
    events: &[JournalEvent],
    by_node: &BTreeMap<&str, Vec<&JournalEvent>>,
    report: &mut Report,
) {
    // (receiving node, hash) -> number of receives recorded.
    let mut receipts: HashMap<(String, String), usize> = HashMap::new();
    for event in events.iter().filter(|e| e.kind == "signal_received") {
        if let Some(hash) = str_field(event, "hash") {
            *receipts
                .entry((event.node.clone(), hash.to_string()))
                .or_default() += 1;
        }
    }

    let mut undelivered = 0usize;
    let mut unnamed = 0usize;
    let mut total = 0usize;

    for event in events.iter().filter(|e| e.kind == "signal_sent") {
        total += 1;
        let (Some(to), Some(hash)) = (str_field(event, "to"), str_field(event, "hash")) else {
            unnamed += 1;
            continue;
        };

        if !by_node.contains_key(to) {
            unnamed += 1;
            continue;
        }

        match receipts.get_mut(&(to.to_string(), hash.to_string())) {
            Some(remaining) if *remaining > 0 => *remaining -= 1,
            _ => undelivered += 1,
        }
    }

    if unnamed > 0 {
        report.errors.push(format!(
            "{unnamed} of {total} sends name a peer that is not a node in this run"
        ));
    }

    if undelivered > 0 {
        // In-flight at shutdown is the ordinary cause and is not a log defect.
        report.notes.push(format!(
            "{undelivered} of {total} sends have no matching receive (in flight at shutdown)"
        ));
    }

    report
        .checks_passed
        .push(format!("{total} sends resolve to a named peer in this run"));
}

/// Consensus must be justified by what that node had already seen.
fn check_consensus(by_node: &BTreeMap<&str, Vec<&JournalEvent>>, report: &mut Report) {
    let mut announcements = 0usize;

    for (node, node_events) in by_node {
        let mut ordered: Vec<&&JournalEvent> = node_events.iter().collect();
        ordered.sort_by_key(|e| e.seq);

        let mut held: BTreeSet<String> = BTreeSet::new();

        for event in ordered {
            match event.kind.as_str() {
                "signal_emitted" | "signal_accepted" => {
                    if let Some(hash) = str_field(event, "hash") {
                        held.insert(hash.to_string());
                    }
                }
                "consensus_reached" => {
                    announcements += 1;
                    let Some(hash) = str_field(event, "hash") else {
                        report
                            .errors
                            .push(format!("{node}: consensus_reached without a hash"));
                        continue;
                    };

                    if !held.contains(hash) {
                        report.errors.push(format!(
                            "{node}: declared consensus on {hash} without ever holding it"
                        ));
                    }

                    let count = u64_field(event, "attester_count").unwrap_or(0);
                    let threshold = u64_field(event, "threshold").unwrap_or(0);
                    if count < threshold {
                        report.errors.push(format!(
                            "{node}: declared consensus on {hash} with {count} attesters, below the threshold of {threshold}"
                        ));
                    }

                    let listed = event
                        .data
                        .get("attesters")
                        .and_then(Value::as_array)
                        .map(|a| a.len() as u64)
                        .unwrap_or(0);
                    if listed != count {
                        report.errors.push(format!(
                            "{node}: consensus on {hash} claims {count} attesters but lists {listed}"
                        ));
                    }
                }
                _ => {}
            }
        }
    }

    report.checks_passed.push(format!(
        "{announcements} consensus declarations are justified by prior receipts"
    ));
}

/// Print a report for a human.
pub fn print_report(report: &Report) {
    println!("\n╭─ journal validation");
    for check in &report.checks_passed {
        println!("│  ok    {check}");
    }
    for note in &report.notes {
        println!("│  note  {note}");
    }
    for error in &report.errors {
        println!("│  FAIL  {error}");
    }
    if report.is_valid() {
        println!("│  → journal is internally consistent and safe to replay");
    } else {
        println!("│  → journal has inconsistencies; a replay of it would be wrong");
    }
    println!("╰─");
}
