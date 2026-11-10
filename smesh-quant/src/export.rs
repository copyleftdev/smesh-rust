//! Turn a run into the compact replay a web page animates.
//!
//! Only derived values leave the run: counts, scores, probabilities, signal
//! strengths, surprises in threshold units and percentage returns. Raw
//! vendor price series are not exported, so the replay can be published.
//! The journal and metrics files are hashed into the replay so every number
//! on a page can be traced back to the run that produced it.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use smesh_runtime::JournalEvent;

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// How many time buckets the gather phase is summarised into.
pub const BUCKETS: usize = 48;

/// Build the replay from a run directory.
pub fn replay(run_dir: &Path) -> anyhow::Result<Value> {
    let timeline_bytes = std::fs::read(run_dir.join("timeline.jsonl"))?;
    let metrics_bytes = std::fs::read(run_dir.join("metrics.json"))?;
    let metrics: Value = serde_json::from_slice(&metrics_bytes)?;
    let events: Vec<JournalEvent> = String::from_utf8_lossy(&timeline_bytes)
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    anyhow::ensure!(!events.is_empty(), "empty timeline");

    let duration_ms = events.iter().map(|e| e.t_ms).max().unwrap_or(0).max(1);

    // Entities, in announcement order.
    let mut entities = Vec::new();
    for e in events.iter().filter(|e| e.kind == "entity") {
        if !entities.iter().any(|x: &Value| x["id"] == e.data["id"]) {
            entities.push(json!({
                "id": e.data["id"], "type": e.data["type"],
                "label": e.data["label"], "group": e.data["group"],
            }));
        }
    }

    // Fetches per venue per time bucket: the gather phase's traffic.
    let mut traffic: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    let mut fetch_totals: BTreeMap<String, BTreeMap<String, u32>> = BTreeMap::new();
    for e in events.iter().filter(|e| e.kind == "fetch") {
        let venue = e.data["venue"].as_str().unwrap_or("?").to_string();
        let bucket =
            ((e.t_ms as f64 / duration_ms as f64) * (BUCKETS as f64 - 1.0)).round() as usize;
        traffic
            .entry(venue.clone())
            .or_insert_with(|| vec![0; BUCKETS])[bucket.min(BUCKETS - 1)] += 1;
        let outcome = match (e.data["source"].as_str(), e.data["status"].as_u64()) {
            (Some("cache"), _) => "cache",
            (_, Some(200..=299)) => "ok",
            (_, Some(429)) => "rate_limited",
            _ => "error",
        };
        *fetch_totals
            .entry(venue)
            .or_default()
            .entry(outcome.into())
            .or_default() += 1;
    }

    // Each hypothesis's recorded path on its field.
    let mut signals: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for e in events.iter().filter(|e| e.kind.starts_with("signal.")) {
        let id = e.data["id"].as_str().unwrap_or("?").to_string();
        signals.entry(id).or_default().push(json!({
            "kind": e.kind.trim_start_matches("signal."),
            "window": e.data["window"], "intensity": e.data["intensity"],
            "confidence": e.data["confidence"], "by": e.data["by"],
        }));
    }

    // Window scores.
    let windows: Vec<Value> = events
        .iter()
        .filter(|e| e.kind == "score")
        .map(|e| {
            let d = &e.data;
            let mut w = json!({"track": d["track"], "window": d["window"]});
            for key in [
                "n",
                "logloss",
                "brier",
                "ece_market",
                "paper",
                "event",
                "series",
                "z",
                "abs_error",
            ] {
                if let Some(v) = d.get(key) {
                    w[key] = v.clone();
                }
            }
            w
        })
        .collect();

    // Out-of-sample crypto predictions, without prices or strikes.
    let predictions: Vec<Value> = events
        .iter()
        .filter(|e| e.kind == "prediction" && e.data["track"] == "crypto")
        .map(|e| {
            json!({
                "window": e.data["window"], "asset": e.data["asset"],
                "p_market": round4(&e.data["p_market"]), "q_model": round4(&e.data["q_model"]),
                "outcome": e.data["outcome"],
            })
        })
        .collect();

    let trades: Vec<Value> = events
        .iter()
        .filter(|e| e.kind == "paper.trade")
        .map(|e| json!({"window": e.data["window"], "side": e.data["side"], "pnl": round4(&e.data["pnl"]), "outcome": e.data["outcome"]}))
        .collect();

    // Macro releases: Kalshi's expectation and the surprise, plus reactions
    // as returns (no price levels).
    let releases: Vec<Value> = events
        .iter()
        .filter(|e| e.kind == "sample" && e.node == "analyst.macro")
        .map(|e| {
            let d = &e.data;
            let spacing = d["spacing"].as_f64().unwrap_or(1.0).max(1e-9);
            let z = (d["actual"].as_f64().unwrap_or(0.0) - d["implied"].as_f64().unwrap_or(0.0)) / spacing;
            json!({"series": d["series"], "event": d["event"], "release": d["release"],
                   "implied": d["implied"], "actual": d["actual"], "spacing": spacing, "z": (z * 100.0).round() / 100.0,
                   "spy": d["spy"], "tlt": d["tlt"],
                   "ladder": d["ladder"]})
        })
        .collect();

    let mut kinds: BTreeMap<String, u32> = BTreeMap::new();
    for e in &events {
        *kinds.entry(e.kind.clone()).or_default() += 1;
    }
    let first_wall = events.iter().find(|e| e.t_ms == 0).map(|e| e.wall.clone());

    let mut metrics_out = Map::new();
    for track in ["crypto", "macro"] {
        if let Some(m) = metrics.get(track) {
            let mut m = m.clone();
            if let Some(o) = m.as_object_mut() {
                o.remove("per_window");
            }
            metrics_out.insert(track.into(), m);
        }
    }

    Ok(json!({
        "format": "smesh-replay/1",
        "run": {
            "id": run_dir.file_name().map(|n| n.to_string_lossy().into_owned()),
            "started": first_wall,
            "duration_ms": duration_ms,
            "events": events.len(),
            "event_kinds": kinds,
            "sha256": {"timeline.jsonl": sha256_hex(&timeline_bytes), "metrics.json": sha256_hex(&metrics_bytes)},
            "read_only": true,
        },
        "entities": entities,
        "gather": {"buckets": BUCKETS, "fetches_by_venue": traffic, "fetch_totals": fetch_totals},
        "signals": signals,
        "windows": windows,
        "predictions": predictions,
        "paper_trades": trades,
        "releases": releases,
        "metrics": metrics_out,
        "findings": metrics["findings"],
    }))
}

fn round4(v: &Value) -> Value {
    v.as_f64()
        .map(|x| json!((x * 10_000.0).round() / 10_000.0))
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::{self, CryptoConfig};
    use crate::log::RunLog;

    #[test]
    fn test_replay_carries_no_price_levels_and_hashes_its_sources() {
        let dir = std::env::temp_dir().join(format!("smesh-quant-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let log = RunLog::create(&dir).unwrap();
        let j = log.agent("analyst.crypto", "research");
        j.record(
            "sample",
            json!({"spot": 76812.5, "strike": 76799.99, "p_market": 0.5}),
        );
        let mut board = crypto::board(&log);
        crypto::walk_forward(&log, &mut board, &[], &CryptoConfig::default());
        let findings = board.conclude();
        log.write_json(
            "metrics.json",
            &json!({"findings": serde_json::to_value(
            findings.into_iter().map(|f| ("crypto".to_string(), f)).collect::<Vec<_>>()).unwrap()}),
        )
        .unwrap();
        log.merge().unwrap();

        let r = replay(&dir).unwrap();
        let text = r.to_string();
        assert!(
            !text.contains("76812.5"),
            "a raw spot price leaked into the replay"
        );
        assert!(
            !text.contains("76799.99"),
            "a raw strike leaked into the replay"
        );
        assert_eq!(r["format"], "smesh-replay/1");
        assert_eq!(
            r["run"]["sha256"]["timeline.jsonl"].as_str().unwrap().len(),
            64
        );
        assert!(r["entities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["id"] == "analyst.crypto"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
