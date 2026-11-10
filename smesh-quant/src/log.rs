//! The run's journal: one file per agent, one shared clock, one merged
//! timeline, written to be replayed as a kinetic visualization.
//!
//! Lines are [`smesh_runtime::JournalEvent`]s (`seq`, `t_ms`, `wall`, `node`,
//! `concern`, `kind`, `data`). Each agent writes `agents/<id>.jsonl`; at the
//! end of the run [`RunLog::merge`] writes `timeline.jsonl`, every agent's
//! lines sorted by `t_ms` then node then `seq`.
//!
//! # Event vocabulary
//!
//! The visualization needs to redraw the run without inferring anything, so
//! graph structure and motion are recorded explicitly:
//!
//! | kind | meaning | key fields |
//! |---|---|---|
//! | `run.started` / `run.finished` | run boundaries | config, totals |
//! | `entity` | a thing that can be drawn | `id`, `type` (agent, venue, dataset, hypothesis, window, track), `label`, `group` |
//! | `flow` | something moved between two entities | `from`, `to`, `what`, `count`, `bytes` |
//! | `fetch` / `backoff` | one HTTP request (or cache read) | `venue`, `endpoint`, `source`, `status`, `ms`, `bytes` |
//! | `phase` | an agent starts or ends a stage | `name`, `state` |
//! | `sample` | one observation assembled for research | track-specific |
//! | `walk.window` | a walk-forward window opens | `track`, `index`, `train`, `test`, `n_train`, `n_test` |
//! | `model.fit` | parameters fitted on a training window | `params`, `train` metrics |
//! | `prediction` | one out-of-sample forecast, with its outcome | `p_market`, `q_model`, `outcome` |
//! | `paper.trade` | a simulated trade (never sent anywhere) | `side`, `price`, `fee`, `pnl` |
//! | `score` | a window's out-of-sample metrics | metrics |
//! | `signal.*` | hypothesis signals on the SMESH field | `id`, `intensity`, `confidence`, `by` |
//! | `finding` | the run's consensus | hypothesis, evidence |
//!
//! Readers should ignore kinds they do not know.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use smesh_runtime::{Journal, JournalEvent};

/// The journals for one research run.
pub struct RunLog {
    dir: PathBuf,
    epoch_ms: i64,
    journals: Mutex<HashMap<String, Arc<Journal>>>,
}

impl RunLog {
    /// Create `dir` (and `dir/agents`) and start the run clock now.
    ///
    /// Journals left in `dir/agents` by an earlier run are removed, so
    /// [`merge`](Self::merge) cannot mix their events into this run's
    /// timeline (a crypto-only rerun would otherwise inherit macro events).
    pub fn create(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        let agents = dir.join("agents");
        if agents.exists() {
            std::fs::remove_dir_all(&agents)?;
        }
        std::fs::create_dir_all(&agents)?;
        Ok(Self {
            dir,
            epoch_ms: chrono::Utc::now().timestamp_millis(),
            journals: Mutex::new(HashMap::new()),
        })
    }

    /// The run directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The journal for agent `id`, created on first use, and announced as an
    /// `entity` the first time.
    pub fn agent(&self, id: &str, concern: &str) -> Arc<Journal> {
        let mut journals = self.journals.lock().expect("journal map poisoned");
        if let Some(j) = journals.get(id) {
            return Arc::clone(j);
        }
        let path = self.dir.join("agents").join(format!("{id}.jsonl"));
        let journal = Journal::create(&path, id, Some(concern.to_string()), self.epoch_ms)
            .unwrap_or_else(|_| Journal::disabled());
        journal.record(
            "entity",
            json!({"id": id, "type": "agent", "label": id, "group": concern}),
        );
        journals.insert(id.to_string(), Arc::clone(&journal));
        journal
    }

    /// Merge every agent's journal into `timeline.jsonl`, ordered by time,
    /// then node, then sequence. Returns the number of events.
    pub fn merge(&self) -> std::io::Result<usize> {
        let mut events: Vec<JournalEvent> = Vec::new();
        for entry in std::fs::read_dir(self.dir.join("agents"))? {
            let text = std::fs::read_to_string(entry?.path())?;
            events.extend(
                text.lines()
                    .filter_map(|l| serde_json::from_str::<JournalEvent>(l).ok()),
            );
        }
        events.sort_by(|a, b| (a.t_ms, &a.node, a.seq).cmp(&(b.t_ms, &b.node, b.seq)));
        let mut out = String::new();
        for e in &events {
            out.push_str(&serde_json::to_string(e).unwrap_or_default());
            out.push('\n');
        }
        std::fs::write(self.dir.join("timeline.jsonl"), out)?;
        Ok(events.len())
    }

    /// Write a JSON artifact (metrics, report data) into the run directory.
    pub fn write_json(&self, name: &str, value: &Value) -> std::io::Result<()> {
        std::fs::write(
            self.dir.join(name),
            serde_json::to_string_pretty(value).unwrap_or_default(),
        )
    }
}

/// Record that `what` moved from entity `from` to entity `to`.
pub fn flow(journal: &Journal, from: &str, to: &str, what: &str, count: usize) {
    journal.record(
        "flow",
        json!({"from": from, "to": to, "what": what, "count": count}),
    );
}

/// Announce a drawable entity that is not an agent.
pub fn entity(journal: &Journal, id: &str, kind: &str, label: &str, group: &str) {
    journal.record(
        "entity",
        json!({"id": id, "type": kind, "label": label, "group": group}),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("smesh-quant-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn test_agents_write_their_own_files_and_merge_in_time_order() {
        let dir = tmp("merge");
        let log = RunLog::create(&dir).unwrap();
        let a = log.agent("scout.kalshi", "data");
        let b = log.agent("walker", "validation");
        a.record("phase", json!({"name": "fetch", "state": "start"}));
        b.record("phase", json!({"name": "walk", "state": "start"}));
        a.record("phase", json!({"name": "fetch", "state": "end"}));

        assert!(dir.join("agents/scout.kalshi.jsonl").exists());
        let n = log.merge().unwrap();
        // Two entity announcements plus three phases.
        assert_eq!(n, 5);
        let lines: Vec<JournalEvent> = std::fs::read_to_string(dir.join("timeline.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert!(lines.windows(2).all(|w| w[0].t_ms <= w[1].t_ms));
        assert_eq!(lines[0].kind, "entity");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_a_rerun_does_not_inherit_an_earlier_runs_agents() {
        let dir = tmp("rerun");
        let first = RunLog::create(&dir).unwrap();
        first
            .agent("analyst.macro", "research")
            .record("sample", json!({}));
        drop(first);
        let second = RunLog::create(&dir).unwrap();
        second.agent("analyst.crypto", "research");
        second.merge().unwrap();
        let timeline = std::fs::read_to_string(dir.join("timeline.jsonl")).unwrap();
        assert!(!timeline.contains("analyst.macro"), "stale journal merged");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_same_agent_reuses_one_journal() {
        let dir = tmp("reuse");
        let log = RunLog::create(&dir).unwrap();
        let a1 = log.agent("x", "g");
        let a2 = log.agent("x", "g");
        assert!(Arc::ptr_eq(&a1, &a2));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
