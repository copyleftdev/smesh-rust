//! # SMESH Quant
//!
//! A read-only quant research mesh over Kalshi, Massive and Kraken.
//!
//! Scouts fetch market data through a client that can only issue `GET`s to
//! allowlisted market-data paths ([`http`]). Analysts turn it into samples
//! and hypotheses; a walker tests every hypothesis walk-forward, one unseen
//! window at a time; a skeptic dampens the weakly supported. Hypotheses are
//! signals on a SMESH field ([`board`]): reinforced by confirming windows,
//! weakened by contradicting ones, decaying without evidence, so what
//! survives is the mesh's consensus.
//!
//! Every step is journaled per agent and merged into one timeline built for
//! replay as a kinetic visualization ([`log`]).
//!
//! Two tracks:
//! - [`crypto`]: Kalshi's hourly BTC/ETH above/below odds vs a realized
//!   volatility model from Massive minute bars, cross-checked on Kraken.
//! - [`macro_track`]: Kalshi's CPI, payrolls and Fed ladders vs the actual
//!   prints, and SPY/TLT reactions to the surprise.

pub mod agent;
pub mod board;
pub mod crypto;
pub mod export;
pub mod feeds;
pub mod http;
pub mod log;
pub mod macro_track;
pub mod report;
pub mod run;
pub mod stats;

pub use run::{run, RunConfig, Tracks};
