//! The journal is the visualization's only input, so its shape is a
//! contract. This runs the crypto walk-forward offline, on synthetic samples,
//! and checks that contract: a replay can draw every node, place every
//! signal on a known hypothesis, and animate every prediction without
//! inferring anything.

use std::collections::HashSet;

use chrono::{Duration, TimeZone, Utc};
use serde_json::Value;
use smesh_quant::crypto::{self, CryptoConfig, Sample};
use smesh_quant::log::RunLog;
use smesh_runtime::JournalEvent;

fn synthetic(n: usize) -> Vec<Sample> {
    let start = Utc.with_ymd_and_hms(2026, 10, 1, 0, 0, 0).unwrap();
    (0..n)
        .map(|i| {
            // Deterministic pseudo-random spread around the money.
            let u = ((i * 7919) % 1000) as f64 / 1000.0;
            let spot = 100.0 * (1.0 + 0.004 * (u - 0.5));
            let p = if spot > 100.0 { 0.62 } else { 0.38 };
            Sample {
                asset: if i % 2 == 0 { "BTC" } else { "ETH" },
                ticker: format!("SYN-{i}"),
                t: start + Duration::hours(i as i64 * 3),
                close: start + Duration::hours(i as i64 * 3) + Duration::minutes(30),
                strike: 100.0,
                spot,
                spot_kraken: Some(spot * 1.0001),
                sigma_1m: 0.001,
                bid: p - 0.02,
                ask: p + 0.02,
                p_market: p,
                outcome: ((i * 104_729) % 997) as f64 / 997.0 < p,
                lead_minutes: 30.0,
            }
        })
        .collect()
}

fn events(dir: &std::path::Path) -> Vec<JournalEvent> {
    std::fs::read_to_string(dir.join("timeline.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).expect("every line is a JournalEvent"))
        .collect()
}

#[test]
fn journal_is_a_self_describing_replay() {
    let dir = std::env::temp_dir().join(format!("smesh-quant-contract-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let log = RunLog::create(&dir).unwrap();
    let cfg = CryptoConfig {
        train_days: 5,
        test_days: 2,
        ..CryptoConfig::default()
    };
    let mut board = crypto::board(&log);
    let summary = crypto::walk_forward(&log, &mut board, &synthetic(240), &cfg);
    board.conclude();
    log.merge().unwrap();
    let evs = events(&dir);

    assert!(summary["windows"].as_u64().unwrap() >= 3, "{summary}");

    // Time never runs backwards in the merged timeline.
    assert!(evs.windows(2).all(|w| w[0].t_ms <= w[1].t_ms));

    // Every entity is announced before anything refers to it.
    let mut known: HashSet<String> = HashSet::new();
    for e in &evs {
        if e.kind == "entity" {
            known.insert(e.data["id"].as_str().unwrap().to_string());
            continue;
        }
        assert!(
            known.contains(&e.node),
            "node {} acts before it is announced",
            e.node
        );
        if e.kind.starts_with("signal.") {
            let id = e.data["id"].as_str().unwrap();
            assert!(known.contains(id), "signal on unannounced hypothesis {id}");
            for field in ["intensity", "confidence", "window", "by"] {
                assert!(e.data.get(field).is_some(), "{} lacks {field}", e.kind);
            }
        }
    }

    // Each window opens once, as an entity, before its predictions.
    let windows: Vec<&JournalEvent> = evs.iter().filter(|e| e.kind == "walk.window").collect();
    assert_eq!(windows.len() as u64, summary["windows"].as_u64().unwrap());
    for i in 1..=windows.len() {
        assert!(known.contains(&format!("window.crypto.{i}")));
    }

    // Every prediction carries what a replay needs to draw it.
    let predictions: Vec<&Value> = evs
        .iter()
        .filter(|e| e.kind == "prediction")
        .map(|e| &e.data)
        .collect();
    assert_eq!(
        predictions.len() as u64,
        summary["test_predictions"].as_u64().unwrap()
    );
    for p in predictions {
        for field in [
            "window", "ticker", "t", "p_market", "q_model", "p_blend", "outcome",
        ] {
            assert!(p.get(field).is_some(), "prediction lacks {field}: {p}");
        }
    }

    // The field ticks once per window per hypothesis, and ends in findings.
    let ticks = evs.iter().filter(|e| e.kind == "signal.tick").count();
    assert_eq!(ticks, windows.len() * 4);
    assert_eq!(evs.iter().filter(|e| e.kind == "finding").count(), 4);

    let _ = std::fs::remove_dir_all(&dir);
}
