//! Structured event journal for replayable mesh runs.
//!
//! Every node writes newline-delimited JSON to its own file. One line is one
//! event, and the union of all nodes' files is a complete, ordered account of a
//! run: enough to redraw the topology, replay every signal's diffusion, and
//! reproduce every decay curve without inferring anything.
//!
//! Three properties make the merged log trustworthy:
//!
//! - **A shared clock origin.** Every process is told the same `run_epoch_ms`
//!   and stamps `t_ms` relative to it, so lines from different processes sort
//!   onto one timeline without assuming they started together.
//! - **Per-node sequence numbers.** `seq` is monotonic within a node, so events
//!   that land in the same millisecond still have a defined order.
//! - **Observations, not conclusions.** A node records what it did and why —
//!   including the score and die roll behind a probabilistic relay — so the
//!   replay shows the run that happened rather than a plausible one.
//!
//! The schema is deliberately open: `kind` names the event and `data` carries
//! its payload. Readers should ignore kinds they do not know.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// One line of the journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEvent {
    /// Per-node monotonic sequence number, starting at 1.
    pub seq: u64,
    /// Milliseconds since the run epoch shared by every node in the run.
    pub t_ms: i64,
    /// Absolute wall clock, for correlating against outside systems.
    pub wall: String,
    /// The node that recorded this event.
    pub node: String,
    /// This node's concern, if it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concern: Option<String>,
    /// Event name.
    pub kind: String,
    /// Event payload.
    pub data: Value,
}

enum Sink {
    File(Mutex<BufWriter<File>>),
    Disabled,
}

/// A journal writer shared by every component of one node.
pub struct Journal {
    sink: Sink,
    node: String,
    concern: Option<String>,
    run_epoch_ms: i64,
    seq: AtomicU64,
}

impl Journal {
    /// Open a journal file for one node.
    ///
    /// `run_epoch_ms` must be identical across every node in the run; the
    /// orchestrator picks it once and passes it to each child.
    pub fn create(
        path: impl AsRef<Path>,
        node: impl Into<String>,
        concern: Option<String>,
        run_epoch_ms: i64,
    ) -> std::io::Result<Arc<Self>> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        Ok(Arc::new(Self {
            sink: Sink::File(Mutex::new(BufWriter::new(file))),
            node: node.into(),
            concern,
            run_epoch_ms,
            seq: AtomicU64::new(0),
        }))
    }

    /// A journal that discards everything, for runs that are not being recorded.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            sink: Sink::Disabled,
            node: String::new(),
            concern: None,
            run_epoch_ms: 0,
            seq: AtomicU64::new(0),
        })
    }

    /// Whether this journal writes anywhere.
    pub fn is_enabled(&self) -> bool {
        matches!(self.sink, Sink::File(_))
    }

    /// The node this journal belongs to.
    pub fn node(&self) -> &str {
        &self.node
    }

    /// Record one event.
    ///
    /// Failures are swallowed: a run must not die because its recorder did.
    /// Each line is flushed as it is written so a killed process still leaves a
    /// complete log up to the moment it stopped.
    pub fn record(&self, kind: &str, data: Value) {
        let Sink::File(writer) = &self.sink else {
            return;
        };

        let now = chrono::Utc::now();
        let event = JournalEvent {
            seq: self.seq.fetch_add(1, Ordering::SeqCst) + 1,
            t_ms: now.timestamp_millis() - self.run_epoch_ms,
            wall: now.to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            node: self.node.clone(),
            concern: self.concern.clone(),
            kind: kind.to_string(),
            data,
        };

        let Ok(line) = serde_json::to_string(&event) else {
            return;
        };

        if let Ok(mut writer) = writer.lock() {
            let _ = writeln!(writer, "{line}");
            let _ = writer.flush();
        }
    }

    /// Record the run's opening line: who this node is and how it is configured.
    pub fn node_started(&self, listen_addr: &str, bootstrap: &[String], extra: Value) {
        self.record(
            "node_started",
            json!({
                "listen_addr": listen_addr,
                "bootstrap": bootstrap,
                "run_epoch_ms": self.run_epoch_ms,
                "config": extra,
            }),
        );
    }
}

/// Render a signal's payload for the journal.
///
/// Payloads in this protocol are usually small canonical JSON documents, so
/// the log carries them verbatim when they parse and as text otherwise. That
/// keeps a replay self-describing: a reader never needs the emitting program
/// to interpret what a signal was about.
pub fn payload_preview(payload: &[u8], max_bytes: usize) -> Value {
    let Ok(text) = std::str::from_utf8(payload) else {
        return json!({ "bytes": payload.len(), "encoding": "binary" });
    };

    if let Ok(parsed) = serde_json::from_str::<Value>(text) {
        return parsed;
    }

    if text.len() > max_bytes {
        json!(format!("{}…", &text[..max_bytes]))
    } else {
        json!(text)
    }
}

impl std::fmt::Debug for Journal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Journal")
            .field("node", &self.node)
            .field("concern", &self.concern)
            .field("enabled", &self.is_enabled())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_journal_records_nothing() {
        let journal = Journal::disabled();
        journal.record("anything", json!({"a": 1}));
        assert!(!journal.is_enabled());
    }

    #[test]
    fn events_are_sequenced_and_parseable() {
        let dir = std::env::temp_dir().join(format!("smesh-journal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("node.jsonl");

        let epoch = chrono::Utc::now().timestamp_millis();
        let journal =
            Journal::create(&path, "latency", Some("latency".to_string()), epoch).unwrap();

        journal.node_started("127.0.0.1:9001", &["127.0.0.1:9002".to_string()], json!({}));
        journal.record("finding", json!({"subject": "checkout-api"}));

        let text = std::fs::read_to_string(&path).unwrap();
        let events: Vec<JournalEvent> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].seq, 1);
        assert_eq!(events[1].seq, 2);
        assert_eq!(events[0].kind, "node_started");
        assert_eq!(events[1].node, "latency");
        assert_eq!(events[1].concern.as_deref(), Some("latency"));
        // t_ms is relative to the shared run epoch, so it starts near zero.
        assert!(events[0].t_ms >= 0 && events[0].t_ms < 60_000);

        std::fs::remove_dir_all(&dir).ok();
    }
}
