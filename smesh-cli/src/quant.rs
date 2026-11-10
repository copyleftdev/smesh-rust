//! `smesh quant`: the read-only quant research mesh.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Subcommand, ValueEnum};
use smesh_quant::crypto::CryptoConfig;
use smesh_quant::macro_track::MacroConfig;
use smesh_quant::{RunConfig, Tracks};

/// Which research tracks to run.
#[derive(Clone, Copy, ValueEnum)]
pub enum TrackArg {
    /// Kalshi hourly BTC/ETH odds vs a volatility model.
    Crypto,
    /// Kalshi CPI, payrolls and Fed ladders, and market reactions.
    Macro,
    /// Both tracks.
    Both,
}

/// `smesh quant` subcommands.
#[derive(Subcommand)]
pub enum QuantCommand {
    /// Export a run as the replay the website animates (derived values only)
    ExportSite {
        /// Run directory
        #[arg(long)]
        run: PathBuf,
        /// Site data directory; writes replay.json and replay.js
        #[arg(long, default_value = "site/data")]
        out: PathBuf,
    },
    /// Run the research mesh: fetch (read-only), walk forward, journal everything
    Run {
        /// Tracks to run
        #[arg(long, value_enum, default_value = "both")]
        track: TrackArg,
        /// Crypto: days of history
        #[arg(long, default_value = "21")]
        days: i64,
        /// Crypto: sample one hourly event every N hours
        #[arg(long, default_value = "3")]
        every_hours: u32,
        /// Crypto: strikes per event, nearest the spot at decision time
        #[arg(long, default_value = "3")]
        strikes: usize,
        /// Crypto: walk-forward training window, days
        #[arg(long, default_value = "7")]
        train_days: i64,
        /// Crypto: walk-forward test window and step, days
        #[arg(long, default_value = "2")]
        test_days: i64,
        /// Macro: most recent settled releases per series
        #[arg(long, default_value = "24")]
        macro_events: usize,
        /// Run directory (default runs/quant-<timestamp>)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Response cache directory
        #[arg(long, default_value = "runs/quant-cache")]
        cache: PathBuf,
    },
}

/// Run an `smesh quant` subcommand.
pub async fn run(command: QuantCommand) -> Result<()> {
    match command {
        QuantCommand::ExportSite { run, out } => {
            let replay = smesh_quant::export::replay(&run)?;
            std::fs::create_dir_all(&out)?;
            let json = serde_json::to_string(&replay)?;
            std::fs::write(out.join("replay.json"), &json)?;
            // A script form, so the page also works opened from disk.
            std::fs::write(
                out.join("replay.js"),
                format!("window.SMESH_REPLAY={json};\n"),
            )?;
            println!(
                "replay: {} events, {} hypotheses, {} windows -> {} ({} KB)",
                replay["run"]["events"],
                replay["signals"].as_object().map(|s| s.len()).unwrap_or(0),
                replay["windows"].as_array().map(|w| w.len()).unwrap_or(0),
                out.display(),
                json.len() / 1024
            );
            Ok(())
        }
        QuantCommand::Run {
            track,
            days,
            every_hours,
            strikes,
            train_days,
            test_days,
            macro_events,
            out,
            cache,
        } => {
            let out = out.unwrap_or_else(|| {
                PathBuf::from(format!(
                    "runs/quant-{}",
                    chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
                ))
            });
            let cfg = RunConfig {
                tracks: match track {
                    TrackArg::Crypto => Tracks::Crypto,
                    TrackArg::Macro => Tracks::Macro,
                    TrackArg::Both => Tracks::Both,
                },
                crypto: CryptoConfig {
                    days,
                    every_hours,
                    strikes,
                    train_days,
                    test_days,
                    ..CryptoConfig::default()
                },
                macro_: MacroConfig {
                    max_events_per_series: macro_events,
                    ..MacroConfig::default()
                },
                out: out.clone(),
                cache,
            };
            println!("SMESH quant research mesh (read-only)");
            println!("  run directory: {}", out.display());
            let metrics = smesh_quant::run(cfg).await?;
            println!("\n{}", std::fs::read_to_string(out.join("report.md"))?);
            println!(
                "timeline: {} events in {}",
                metrics["timeline_events"],
                out.join("timeline.jsonl").display()
            );
            Ok(())
        }
    }
}
