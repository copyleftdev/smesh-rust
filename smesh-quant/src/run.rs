//! A research run: venues, agents, tracks, the hypothesis field, the report.

use std::path::PathBuf;

use serde_json::{json, Value};

use crate::board::Finding;
use crate::crypto::{self, CryptoConfig, ASSETS};
use crate::http::ReadOnlyClient;
use crate::log::{entity, RunLog};
use crate::macro_track::{self, MacroConfig};

/// Which research tracks to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Tracks {
    /// Crypto price-odds calibration only.
    Crypto,
    /// Macro release markets only.
    Macro,
    /// Both.
    Both,
}

/// Settings for a run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunConfig {
    /// Tracks to run.
    pub tracks: Tracks,
    /// Crypto track settings.
    pub crypto: CryptoConfig,
    /// Macro track settings.
    pub macro_: MacroConfig,
    /// Run directory.
    pub out: PathBuf,
    /// Response cache directory.
    pub cache: PathBuf,
}

/// Read `key` from a dotenv file, last assignment winning.
pub fn read_env_file(path: &std::path::Path, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    // The last non-empty assignment wins, as when the file is sourced.
    text.lines()
        .rev()
        .filter_map(|l| {
            let l = l.trim();
            let l = l.strip_prefix("export ").unwrap_or(l);
            let (k, v) = l.split_once('=')?;
            (k.trim() == key).then(|| v.trim().trim_matches('"').trim_matches('\'').to_string())
        })
        .find(|v| !v.is_empty())
}

/// The Massive key from `MASSIVE_API_KEY` or `~/.creds/massive.env`.
pub fn massive_key() -> Option<String> {
    std::env::var("MASSIVE_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
        .or_else(|| {
            let home = std::env::var_os("HOME")?;
            read_env_file(
                &PathBuf::from(home).join(".creds/massive.env"),
                "MASSIVE_API_KEY",
            )
        })
}

/// Run the research and write the journal, metrics and report into
/// `cfg.out`. Returns the metrics.
pub async fn run(cfg: RunConfig) -> anyhow::Result<Value> {
    let log = RunLog::create(&cfg.out)?;
    let conductor = log.agent("conductor", "mesh");
    conductor.record("run.started", json!({"config": cfg}));
    for (id, label) in [
        ("venue.kalshi", "Kalshi"),
        ("venue.massive", "Massive"),
        ("venue.kraken", "Kraken"),
    ] {
        entity(&conductor, id, "venue", label, "venues");
    }

    let key = massive_key();
    conductor.record("credentials", json!({"massive": key.is_some(), "kalshi": "public market data, unsigned", "kraken": "public endpoints, no key loaded"}));
    let client = ReadOnlyClient::new(&cfg.cache, key);

    let mut metrics = json!({});
    let mut findings: Vec<(String, Finding)> = Vec::new();

    if matches!(cfg.tracks, Tracks::Crypto | Tracks::Both) {
        entity(
            &conductor,
            "track.crypto",
            "track",
            "Crypto odds vs volatility",
            "tracks",
        );
        let mut board = crypto::board(&log);
        let mut samples = Vec::new();
        for asset in ASSETS {
            match crypto::gather(&log, &client, asset, &cfg.crypto).await {
                Ok(s) => samples.extend(s),
                Err(e) => conductor.record(
                    "track.error",
                    json!({"track": "crypto", "asset": asset.name, "error": e.to_string()}),
                ),
            }
        }
        samples.sort_by_key(|s| s.t);
        let summary = crypto::walk_forward(&log, &mut board, &samples, &cfg.crypto);
        findings.extend(
            board
                .conclude()
                .into_iter()
                .map(|f| ("crypto".to_string(), f)),
        );
        metrics["crypto"] = summary;
    }

    if matches!(cfg.tracks, Tracks::Macro | Tracks::Both) {
        entity(
            &conductor,
            "track.macro",
            "track",
            "Macro releases vs markets",
            "tracks",
        );
        let mut board = macro_track::board(&log);
        match macro_track::gather(&log, &client, &cfg.macro_).await {
            Ok(events) => {
                let summary = macro_track::walk_forward(&log, &mut board, &events, &cfg.macro_);
                metrics["macro"] = summary;
            }
            Err(e) => {
                conductor.record(
                    "track.error",
                    json!({"track": "macro", "error": e.to_string()}),
                );
            }
        }
        findings.extend(
            board
                .conclude()
                .into_iter()
                .map(|f| ("macro".to_string(), f)),
        );
    }

    metrics["findings"] = serde_json::to_value(&findings)?;
    log.write_json("metrics.json", &metrics)?;
    std::fs::write(
        cfg.out.join("report.md"),
        crate::report::render(&cfg, &metrics, &findings),
    )?;
    conductor.record("run.finished", json!({"findings": findings.len()}));
    let events = log.merge()?;
    metrics["timeline_events"] = json!(events);
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_env_file_takes_the_last_non_empty_assignment() {
        let path = std::env::temp_dir().join(format!("smesh-quant-env-{}", std::process::id()));
        std::fs::write(
            &path,
            "MASSIVE_API_KEY=old\nOTHER=x\nexport MASSIVE_API_KEY=\"new\"\nMASSIVE_API_KEY=\n",
        )
        .unwrap();
        assert_eq!(
            read_env_file(&path, "MASSIVE_API_KEY").as_deref(),
            Some("new")
        );
        assert_eq!(read_env_file(&path, "MISSING"), None);
        let _ = std::fs::remove_file(&path);
    }
}
