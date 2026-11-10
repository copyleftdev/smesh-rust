//! Track A: are Kalshi's hourly crypto above/below odds well priced?
//!
//! For each sampled hourly event, at a decision time `lead` minutes before
//! close, the scouts assemble one [`Sample`] per strike near the spot price
//! *as known then* (never the settled price, which would leak the outcome):
//!
//! - Kalshi's two-sided quote on "will the price be above K at close?",
//! - the spot price from Massive's minute bars, and independently from
//!   Kraken's trade tape,
//! - realized volatility from the minute bars over a lookback window.
//!
//! The model's probability is driftless lognormal:
//! `q = Φ(ln(S/K) / (c · σ₁ₘ · √lead))`, with the volatility scale `c` fitted
//! per walk-forward window. Each window also fits a market/model blend and a
//! fee-aware trading threshold on its training data, then scores the next,
//! unseen window.

use chrono::{DateTime, Duration, Utc};
use serde_json::json;

use crate::board::Board;
use crate::feeds::kalshi::{event_date, Kalshi, Market};
use crate::feeds::kraken;
use crate::feeds::massive::{aggs, bar_at, Bar};
use crate::http::ReadOnlyClient;
use crate::log::{flow, RunLog};
use crate::stats::{
    brier, calibration, clamp_p, kalshi_taker_fee, log_loss, mean, norm_cdf, std_dev,
};

/// One crypto asset and where to find it on each venue.
#[derive(Debug, Clone, Copy)]
pub struct Asset {
    /// Display name.
    pub name: &'static str,
    /// Kalshi above/below series.
    pub kalshi_series: &'static str,
    /// Massive ticker.
    pub massive: &'static str,
    /// Kraken pair.
    pub kraken: &'static str,
}

/// BTC and ETH.
pub const ASSETS: [Asset; 2] = [
    Asset {
        name: "BTC",
        kalshi_series: "KXBTCD",
        massive: "X:BTCUSD",
        kraken: "XBTUSD",
    },
    Asset {
        name: "ETH",
        kalshi_series: "KXETHD",
        massive: "X:ETHUSD",
        kraken: "ETHUSD",
    },
];

/// Settings for the crypto track.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CryptoConfig {
    /// Days of history, ending yesterday.
    pub days: i64,
    /// Sample one event every this many hours.
    pub every_hours: u32,
    /// Strikes per event, nearest the decision-time spot.
    pub strikes: usize,
    /// Decision time, minutes before close.
    pub lead_minutes: i64,
    /// Realized-volatility lookback, minutes.
    pub vol_lookback_minutes: i64,
    /// Walk-forward training window, days.
    pub train_days: i64,
    /// Walk-forward test window (and step), days.
    pub test_days: i64,
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            days: 21,
            every_hours: 3,
            strikes: 3,
            lead_minutes: 30,
            vol_lookback_minutes: 360,
            train_days: 7,
            test_days: 2,
        }
    }
}

/// One market observed at decision time, with its outcome.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Sample {
    /// Asset name.
    pub asset: &'static str,
    /// Market ticker.
    pub ticker: String,
    /// Decision time.
    pub t: DateTime<Utc>,
    /// Market close.
    pub close: DateTime<Utc>,
    /// Strike: yes if the price is above it at close.
    pub strike: f64,
    /// Spot at decision time from Massive.
    pub spot: f64,
    /// Spot at decision time from Kraken's tape, if found.
    pub spot_kraken: Option<f64>,
    /// Realized 1-minute volatility (log returns) over the lookback.
    pub sigma_1m: f64,
    /// Yes bid at decision time.
    pub bid: f64,
    /// Yes ask at decision time.
    pub ask: f64,
    /// Bid/ask mid.
    pub p_market: f64,
    /// Whether it settled yes.
    pub outcome: bool,
    /// Minutes from decision to close.
    pub lead_minutes: f64,
}

impl Sample {
    /// Model probability with volatility scale `c`.
    pub fn q(&self, c: f64) -> f64 {
        let sd = c * self.sigma_1m * self.lead_minutes.sqrt();
        if sd <= 0.0 {
            return if self.spot > self.strike { 1.0 } else { 0.0 };
        }
        norm_cdf((self.spot / self.strike).ln() / sd)
    }
}

/// The start time of the last 1-minute bar that has closed by `t_ms`.
///
/// Bars are labelled by start, so the bar starting exactly at `t_ms` is still
/// open at `t_ms`. Using it would let the model see up to 59 seconds past the
/// decision time, which Kalshi's quote (from candles ending at `t_ms`) does
/// not.
pub fn known_at(t_ms: i64) -> i64 {
    t_ms - 60_000
}

/// Realized standard deviation of 1-minute log returns over the bars in
/// `(t - lookback, t]`.
pub fn realized_sigma(bars: &[Bar], t_ms: i64, lookback_ms: i64) -> Option<f64> {
    let lo = bars.partition_point(|b| b.t_ms <= t_ms - lookback_ms);
    let hi = bars.partition_point(|b| b.t_ms <= t_ms);
    let window = &bars[lo..hi];
    if window.len() < 30 {
        return None;
    }
    let rets: Vec<f64> = window
        .windows(2)
        .filter(|w| w[0].c > 0.0 && w[1].c > 0.0)
        .map(|w| {
            // Normalise gaps to a per-minute return.
            let minutes = ((w[1].t_ms - w[0].t_ms) as f64 / 60_000.0).max(1.0);
            (w[1].c / w[0].c).ln() / minutes.sqrt()
        })
        .collect();
    Some(std_dev(&rets))
}

/// Gather samples for one asset. Journals every step under the scouts.
pub async fn gather(
    log: &RunLog,
    client: &ReadOnlyClient,
    asset: Asset,
    cfg: &CryptoConfig,
) -> anyhow::Result<Vec<Sample>> {
    let kalshi_j = log.agent("scout.kalshi", "data");
    let massive_j = log.agent("scout.massive", "data");
    let kraken_j = log.agent("scout.kraken", "data");
    let analyst = log.agent("analyst.crypto", "research");

    let end = Utc::now().date_naive() - Duration::days(1);
    let start = end - Duration::days(cfg.days);

    // Minute bars covering the period plus the first lookback.
    massive_j.record(
        "phase",
        json!({"name": "minute_bars", "state": "start", "asset": asset.name}),
    );
    let bars = aggs(
        client,
        &massive_j,
        asset.massive,
        1,
        "minute",
        start - Duration::days(1),
        end + Duration::days(1),
    )
    .await?;
    massive_j.record(
        "dataset",
        json!({"asset": asset.name, "ticker": asset.massive, "bars": bars.len(),
               "from": bars.first().map(|b| b.t_ms), "to": bars.last().map(|b| b.t_ms)}),
    );
    flow(
        &massive_j,
        "venue.massive",
        "scout.massive",
        "minute_bars",
        bars.len(),
    );
    flow(
        &massive_j,
        "scout.massive",
        "analyst.crypto",
        "minute_bars",
        bars.len(),
    );

    // Settled hourly events in range, sampled every N hours.
    let kalshi = Kalshi::connect(client, &kalshi_j).await?;
    let events = kalshi
        .settled_events(asset.kalshi_series, pages_for(cfg.days))
        .await?;
    let chosen: Vec<String> = events
        .into_iter()
        .filter(|e| {
            event_date(e).is_some_and(|d| d >= start && d <= end)
                && event_hour(e).is_some_and(|h| h % cfg.every_hours == 0)
        })
        .collect();
    kalshi_j.record(
        "events.selected",
        json!({"asset": asset.name, "series": asset.kalshi_series, "count": chosen.len(),
               "from": start, "to": end, "every_hours": cfg.every_hours}),
    );

    let mut samples = Vec::new();
    for event in &chosen {
        let markets = kalshi.event_markets(event).await?;
        let ladder: Vec<&Market> = markets
            .iter()
            .filter(|m| m.strike_type == "greater" && m.floor_strike.is_some() && m.yes().is_some())
            .collect();
        let Some(first) = ladder.first() else {
            kalshi_j.record(
                "event.skipped",
                json!({"event": event, "reason": "no settled greater-than ladder"}),
            );
            continue;
        };
        let close = first.close_time;
        let t = close - Duration::minutes(cfg.lead_minutes);
        let t_ms = t.timestamp_millis();

        // Massive labels bars by their start, so the bar starting at `t`
        // closes after `t`. The last bar fully known at decision time is the
        // one that closed by `t`: it starts a minute earlier.
        let known_ms = known_at(t_ms);
        let Some(spot_bar) = bar_at(&bars, known_ms, 5 * 60_000) else {
            kalshi_j.record(
                "event.skipped",
                json!({"event": event, "reason": "no Massive bar near decision time"}),
            );
            continue;
        };
        let spot = spot_bar.c;
        let Some(sigma) = realized_sigma(&bars, known_ms, cfg.vol_lookback_minutes * 60_000) else {
            kalshi_j.record(
                "event.skipped",
                json!({"event": event, "reason": "too few bars for volatility"}),
            );
            continue;
        };
        let spot_kraken = kraken::price_at(client, &kraken_j, asset.kraken, t.timestamp())
            .await
            .ok()
            .flatten()
            .filter(|(_, at)| (*at - t.timestamp() as f64).abs() < 600.0)
            .map(|(p, _)| p);
        if let Some(k) = spot_kraken {
            kraken_j.record(
                "spot",
                json!({"asset": asset.name, "t": t, "kraken": k, "massive": spot,
                       "basis_bps": (k / spot - 1.0) * 1e4}),
            );
        }

        // Strikes nearest the spot known at decision time.
        let mut near: Vec<&&Market> = ladder.iter().collect();
        near.sort_by(|a, b| {
            (a.floor_strike.unwrap() - spot)
                .abs()
                .total_cmp(&(b.floor_strike.unwrap() - spot).abs())
        });
        for market in near.into_iter().take(cfg.strikes) {
            let quotes = kalshi
                .candles(
                    asset.kalshi_series,
                    market,
                    t.timestamp() - 300,
                    t.timestamp(),
                    1,
                )
                .await?;
            let Some(q) = quotes.iter().rev().find(|q| q.mid().is_some()) else {
                kalshi_j.record("market.skipped", json!({"ticker": market.ticker, "reason": "no two-sided quote at decision time"}));
                continue;
            };
            let s = Sample {
                asset: asset.name,
                ticker: market.ticker.clone(),
                t,
                close,
                strike: market.floor_strike.unwrap(),
                spot,
                spot_kraken,
                sigma_1m: sigma,
                bid: q.yes_bid.unwrap(),
                ask: q.yes_ask.unwrap(),
                p_market: q.mid().unwrap(),
                outcome: market.yes().unwrap(),
                lead_minutes: (close - t).num_seconds() as f64 / 60.0,
            };
            analyst.record("sample", serde_json::to_value(&s).unwrap_or_default());
            samples.push(s);
        }
    }
    flow(
        &kalshi_j,
        "venue.kalshi",
        "scout.kalshi",
        "quotes",
        samples.len(),
    );
    flow(
        &kalshi_j,
        "scout.kalshi",
        "analyst.crypto",
        "samples",
        samples.len(),
    );
    samples.sort_by_key(|s| s.t);
    Ok(samples)
}

fn event_hour(event: &str) -> Option<u32> {
    event.split('-').nth(1)?.get(7..9)?.parse().ok()
}

fn pages_for(days: i64) -> usize {
    // 24 events a day, 200 per page, plus slack for the days skipped.
    ((days as usize + 3) * 24).div_ceil(200) + 2
}

/// Fitted parameters for one window.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Fit {
    /// Volatility scale.
    pub c: f64,
    /// Weight on the model in the blend (rest on the market).
    pub w: f64,
    /// Minimum edge after fees to trade.
    pub threshold: f64,
}

fn pairs(samples: &[&Sample], f: impl Fn(&Sample) -> f64) -> Vec<(f64, bool)> {
    samples.iter().map(|s| (f(s), s.outcome)).collect()
}

/// Paper trade one sample: buy yes at the ask if the model's edge over it
/// beats `threshold` after fees, or buy no at `1 - bid` symmetrically.
/// Returns `(side, price, fee, pnl)` per contract, or `None` for no trade.
pub fn paper_trade(s: &Sample, q: f64, threshold: f64) -> Option<(&'static str, f64, f64, f64)> {
    let yes_edge = q - s.ask - kalshi_taker_fee(s.ask);
    let no_price = 1.0 - s.bid;
    let no_edge = (1.0 - q) - no_price - kalshi_taker_fee(no_price);
    if yes_edge > threshold && yes_edge >= no_edge {
        let fee = kalshi_taker_fee(s.ask);
        let payoff = if s.outcome { 1.0 } else { 0.0 };
        Some(("yes", s.ask, fee, payoff - s.ask - fee))
    } else if no_edge > threshold {
        let fee = kalshi_taker_fee(no_price);
        let payoff = if s.outcome { 0.0 } else { 1.0 };
        Some(("no", no_price, fee, payoff - no_price - fee))
    } else {
        None
    }
}

/// Fit `c`, `w` and the threshold on training samples.
pub fn fit(train: &[&Sample]) -> Fit {
    let c = (10..=60)
        .map(|i| i as f64 * 0.05)
        .min_by(|a, b| {
            log_loss(&pairs(train, |s| s.q(*a))).total_cmp(&log_loss(&pairs(train, |s| s.q(*b))))
        })
        .unwrap_or(1.0);
    let w = (0..=20)
        .map(|i| i as f64 * 0.05)
        .min_by(|a, b| {
            let blend = |w: f64| pairs(train, |s| w * s.q(c) + (1.0 - w) * s.p_market);
            log_loss(&blend(*a)).total_cmp(&log_loss(&blend(*b)))
        })
        .unwrap_or(0.0);
    let threshold = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]
        .into_iter()
        .max_by(|a, b| {
            let pnl = |th: f64| -> f64 {
                train
                    .iter()
                    .filter_map(|s| paper_trade(s, s.q(c), th))
                    .map(|t| t.3)
                    .sum()
            };
            pnl(*a).total_cmp(&pnl(*b))
        })
        .unwrap_or(0.1);
    Fit { c, w, threshold }
}

/// Walk forward over the samples, journaling every window, prediction,
/// paper trade and score, and moving the hypotheses on `board`.
pub fn walk_forward(
    log: &RunLog,
    board: &mut Board,
    samples: &[Sample],
    cfg: &CryptoConfig,
) -> serde_json::Value {
    let walker = log.agent("walker", "validation");
    let analyst = log.agent("analyst.crypto", "research");
    let skeptic = log.agent("skeptic", "validation");

    // A window must move forward, or the walk never ends.
    if cfg.test_days < 1 || cfg.train_days < 1 {
        walker.record(
            "walk.invalid",
            json!({"track": "crypto", "train_days": cfg.train_days, "test_days": cfg.test_days,
                   "reason": "train and test windows must be at least one day"}),
        );
        return json!({"windows": 0, "samples": samples.len(), "test_predictions": 0, "per_window": [],
                      "error": "train_days and test_days must be at least 1"});
    }

    let Some(first) = samples.first().map(|s| s.t) else {
        walker.record("walk.empty", json!({"track": "crypto"}));
        return json!({"windows": 0, "samples": 0, "test_predictions": 0, "per_window": []});
    };
    let last = samples.last().map(|s| s.t).unwrap_or(first);
    let mut windows = Vec::new();
    let mut all_test: Vec<(f64, f64, f64, bool)> = Vec::new(); // (market, model, blend, outcome)
    let mut total_pnl = 0.0;
    let mut total_trades = 0usize;

    let mut train_start = first;
    let mut index = 0;
    loop {
        let train_end = train_start + Duration::days(cfg.train_days);
        let test_end = train_end + Duration::days(cfg.test_days);
        if train_end > last {
            break;
        }
        let train: Vec<&Sample> = samples
            .iter()
            .filter(|s| s.t >= train_start && s.t < train_end)
            .collect();
        let test: Vec<&Sample> = samples
            .iter()
            .filter(|s| s.t >= train_end && s.t < test_end)
            .collect();
        train_start += Duration::days(cfg.test_days);
        if train.len() < 10 || test.is_empty() {
            continue;
        }
        index += 1;
        walker.record(
            "walk.window",
            json!({"track": "crypto", "index": index,
                   "train": [train.first().unwrap().t, train.last().unwrap().t],
                   "test": [test.first().unwrap().t, test.last().unwrap().t],
                   "n_train": train.len(), "n_test": test.len()}),
        );
        crate::log::entity(
            &walker,
            &format!("window.crypto.{index}"),
            "window",
            &format!("crypto window {index}"),
            "crypto",
        );

        let f = fit(&train);
        let train_market = log_loss(&pairs(&train, |s| s.p_market));
        let train_model = log_loss(&pairs(&train, |s| s.q(f.c)));
        analyst.record(
            "model.fit",
            json!({"track": "crypto", "window": index, "params": f,
                   "train": {"logloss_market": train_market, "logloss_model": train_model}}),
        );

        // Hypotheses are proposed from the first window's training data and
        // then tested on every window that follows.
        if index == 1 {
            board.propose(
                &analyst,
                "crypto.market_beats_model",
                "Kalshi's hourly crypto odds forecast better than a realized-volatility model",
                json!({"train_market": train_market, "train_model": train_model}),
            );
            board.propose(
                &analyst,
                "crypto.blend_beats_market",
                "Blending the volatility model into Kalshi's odds improves on the odds alone",
                json!({"w": f.w}),
            );
            board.propose(
                &analyst,
                "crypto.edge_strategy_profits",
                "A fee-aware edge strategy on the model earns positive out-of-sample P&L",
                json!({"threshold": f.threshold}),
            );
            board.propose(&analyst, "crypto.odds_calibrated", "Kalshi's hourly crypto odds are calibrated (calibration error within what chance alone would produce)", json!({}));
        }

        // Out of sample.
        let mut window_pnl = 0.0;
        let mut window_trades = 0;
        let mut test_rows = Vec::new();
        for s in &test {
            let q = s.q(f.c);
            let blend = f.w * q + (1.0 - f.w) * s.p_market;
            walker.record(
                "prediction",
                json!({"track": "crypto", "window": index, "ticker": s.ticker, "asset": s.asset,
                       "t": s.t, "strike": s.strike, "spot": s.spot, "p_market": s.p_market,
                       "q_model": clamp_p(q), "p_blend": blend, "outcome": s.outcome}),
            );
            if let Some((side, price, fee, pnl)) = paper_trade(s, q, f.threshold) {
                window_pnl += pnl;
                window_trades += 1;
                walker.record(
                    "paper.trade",
                    json!({"track": "crypto", "window": index, "ticker": s.ticker, "side": side,
                           "price": price, "fee": fee, "pnl": pnl, "q_model": q, "outcome": s.outcome}),
                );
            }
            test_rows.push((s.p_market, q, blend, s.outcome));
        }
        total_pnl += window_pnl;
        total_trades += window_trades;

        let market: Vec<_> = test_rows.iter().map(|r| (r.0, r.3)).collect();
        let model: Vec<_> = test_rows.iter().map(|r| (r.1, r.3)).collect();
        let blend: Vec<_> = test_rows.iter().map(|r| (r.2, r.3)).collect();
        let ece = expected_calibration_error(&market);
        let score = json!({
            "track": "crypto", "window": index, "n": test_rows.len(),
            "logloss": {"market": log_loss(&market), "model": log_loss(&model), "blend": log_loss(&blend)},
            "brier": {"market": brier(&market), "model": brier(&model), "blend": brier(&blend)},
            "ece_market": ece, "paper": {"trades": window_trades, "pnl_per_contract_sum": window_pnl},
        });
        walker.record("score", score.clone());
        let n_test = test_rows.len();
        all_test.extend(test_rows);

        // Evidence.
        board.tick();
        let ev = |b: bool, id: &str, board: &mut Board, data: serde_json::Value| {
            if b {
                board.confirm(&walker, id, data);
            } else {
                board.contradict(&walker, id, data);
            }
        };
        ev(
            log_loss(&market) <= log_loss(&model),
            "crypto.market_beats_model",
            board,
            score["logloss"].clone(),
        );
        ev(
            log_loss(&blend) < log_loss(&market),
            "crypto.blend_beats_market",
            board,
            score["logloss"].clone(),
        );
        if window_trades > 0 {
            ev(
                window_pnl > 0.0,
                "crypto.edge_strategy_profits",
                board,
                json!({"pnl": window_pnl, "trades": window_trades}),
            );
        }
        let ece_bound = calibration_noise_bound(&market, 500, index as u64);
        ev(
            ece <= ece_bound,
            "crypto.odds_calibrated",
            board,
            json!({"ece": ece, "noise_bound_95": ece_bound}),
        );

        // Doubt.
        if n_test < 30 {
            for id in [
                "crypto.market_beats_model",
                "crypto.blend_beats_market",
                "crypto.odds_calibrated",
            ] {
                board.dampen(&skeptic, id, 0.9, &format!("only {n_test} test samples"));
            }
        }
        if window_trades < 10 {
            board.dampen(
                &skeptic,
                "crypto.edge_strategy_profits",
                0.85,
                &format!("only {window_trades} paper trades"),
            );
        }
        windows.push(score);
    }

    // Four hypotheses tested together: say so once.
    skeptic.record(
        "multiple_testing",
        json!({"track": "crypto", "hypotheses": 4, "note": "four claims tested on the same windows; treat single-window wins as weak"}),
    );

    let market: Vec<_> = all_test.iter().map(|r| (r.0, r.3)).collect();
    let model: Vec<_> = all_test.iter().map(|r| (r.1, r.3)).collect();
    let blend: Vec<_> = all_test.iter().map(|r| (r.2, r.3)).collect();
    let summary = json!({
        "samples": samples.len(), "windows": windows.len(), "test_predictions": all_test.len(),
        "logloss": {"market": log_loss(&market), "model": log_loss(&model), "blend": log_loss(&blend)},
        "brier": {"market": brier(&market), "model": brier(&model), "blend": brier(&blend)},
        "calibration_market": calibration(&market, 10),
        "calibration_model": calibration(&model, 10),
        "paper": {"trades": total_trades, "pnl_per_contract_sum": total_pnl,
                   "pnl_per_trade": if total_trades > 0 { total_pnl / total_trades as f64 } else { 0.0 }},
        "kraken_basis_bps_mean": mean(samples.iter().filter_map(|s| s.spot_kraken.map(|k| (k / s.spot - 1.0) * 1e4))),
        "per_window": windows,
    });
    walker.record(
        "walk.summary",
        json!({"track": "crypto", "summary": summary.clone()}),
    );
    summary
}

/// The 95th percentile of the expected calibration error a perfectly
/// calibrated forecaster would show on these same predictions, by chance
/// alone: outcomes are redrawn as Bernoulli(p) `sims` times.
///
/// ECE is biased upward on small samples: with ~90 predictions over 10
/// buckets, a perfect forecaster often scores above 5 points. Comparing the
/// observed ECE with this bound tests calibration fairly at any sample size.
/// The generator is a fixed-seed xorshift, so the bound is reproducible.
pub fn calibration_noise_bound(pairs: &[(f64, bool)], sims: usize, seed: u64) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut eces: Vec<f64> = (0..sims.max(1))
        .map(|_| {
            let redrawn: Vec<(f64, bool)> =
                pairs.iter().map(|(p, _)| (*p, uniform() < *p)).collect();
            expected_calibration_error(&redrawn)
        })
        .collect();
    eces.sort_by(f64::total_cmp);
    eces[((eces.len() as f64 * 0.95) as usize).min(eces.len() - 1)]
}

/// Expected calibration error over 10 buckets: the prediction-weighted mean
/// gap between predicted and observed frequency.
pub fn expected_calibration_error(pairs: &[(f64, bool)]) -> f64 {
    let n = pairs.len().max(1) as f64;
    calibration(pairs, 10)
        .iter()
        .map(|b| b.n as f64 / n * (b.predicted - b.observed).abs())
        .sum()
}

/// The journal for the crypto board.
pub fn board(log: &RunLog) -> Board {
    Board::new("crypto", log.agent("field.crypto", "mesh"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(spot: f64, strike: f64, bid: f64, ask: f64, outcome: bool) -> Sample {
        Sample {
            asset: "BTC",
            ticker: "T".into(),
            t: Utc::now(),
            close: Utc::now(),
            strike,
            spot,
            spot_kraken: None,
            sigma_1m: 0.001,
            bid,
            ask,
            p_market: (bid + ask) / 2.0,
            outcome,
            lead_minutes: 30.0,
        }
    }

    #[test]
    fn test_model_probability_is_half_at_the_money_and_monotone() {
        let s = sample(100.0, 100.0, 0.45, 0.55, true);
        assert!((s.q(1.0) - 0.5).abs() < 1e-6);
        let above = sample(101.0, 100.0, 0.45, 0.55, true);
        assert!(above.q(1.0) > 0.5);
        // More volatility pulls a probability towards a half.
        assert!(above.q(2.0) < above.q(1.0));
    }

    #[test]
    fn test_paper_trade_buys_the_cheap_side_after_fees() {
        let s = sample(100.0, 100.0, 0.30, 0.32, true);
        let (side, price, fee, pnl) = paper_trade(&s, 0.60, 0.0).unwrap();
        assert_eq!(side, "yes");
        assert_eq!(price, 0.32);
        assert!((pnl - (1.0 - 0.32 - fee)).abs() < 1e-12);
        // No edge, no trade.
        assert!(paper_trade(&s, 0.33, 0.02).is_none());
        // Model says no: buy no at 1 - bid.
        let (side, price, ..) = paper_trade(&s, 0.05, 0.0).unwrap();
        assert_eq!((side, price), ("no", 0.70));
    }

    #[test]
    fn test_realized_sigma_of_constant_drift_is_zero() {
        let bars: Vec<Bar> = (0..100)
            .map(|i| Bar {
                t_ms: i * 60_000,
                o: 1.0,
                h: 1.0,
                l: 1.0,
                c: (0.001 * i as f64).exp(),
                v: 1.0,
            })
            .collect();
        let s = realized_sigma(&bars, 99 * 60_000, 60 * 60_000).unwrap();
        assert!(s < 1e-9, "{s}");
        assert!(realized_sigma(&bars, 10 * 60_000, 5 * 60_000).is_none());
    }

    #[test]
    fn test_noise_bound_accepts_a_calibrated_forecaster_and_rejects_a_biased_one() {
        // 90 predictions, outcomes drawn honestly from p: should pass.
        let mut state = 12345u64;
        let mut u = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let honest: Vec<(f64, bool)> = (0..90)
            .map(|i| {
                let p = 0.05 + 0.9 * (i as f64 / 89.0);
                (p, u() < p)
            })
            .collect();
        let bound = calibration_noise_bound(&honest, 500, 1);
        assert!(expected_calibration_error(&honest) <= bound, "{bound}");
        // Small samples are noisy: the bound is well above the old fixed 5 points.
        assert!(bound > 0.05, "{bound}");
        // A forecaster 25 points overconfident on yes: should fail.
        let biased: Vec<(f64, bool)> = honest
            .iter()
            .map(|(p, y)| ((p + 0.25).min(0.99), *y))
            .collect();
        assert!(expected_calibration_error(&biased) > calibration_noise_bound(&biased, 500, 1));
    }

    #[test]
    fn test_decision_inputs_use_only_bars_closed_by_decision_time() {
        // Flat at 100 until the bar that starts at t, which jumps to 200.
        let t = 120 * 60_000i64;
        let bars: Vec<Bar> = (0..=120)
            .map(|i| {
                let c = if i == 120 {
                    200.0
                } else {
                    100.0 * (1.0 + 0.0001 * (i % 2) as f64)
                };
                Bar {
                    t_ms: i * 60_000,
                    o: c,
                    h: c,
                    l: c,
                    c,
                    v: 1.0,
                }
            })
            .collect();
        let spot = bar_at(&bars, known_at(t), 5 * 60_000).unwrap().c;
        assert!(spot < 101.0, "spot {spot} came from a bar still open at t");
        let sigma = realized_sigma(&bars, known_at(t), 60 * 60_000).unwrap();
        assert!(sigma < 0.01, "sigma {sigma} saw the jump after t");
    }

    #[test]
    fn test_walk_refuses_a_window_that_cannot_move() {
        let dir = std::env::temp_dir().join(format!("smesh-quant-badwalk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let log = RunLog::create(&dir).unwrap();
        let mut b = board(&log);
        let samples = vec![sample(100.0, 100.0, 0.4, 0.6, true); 50];
        for test_days in [0, -2] {
            let cfg = CryptoConfig {
                test_days,
                ..CryptoConfig::default()
            };
            let out = walk_forward(&log, &mut b, &samples, &cfg);
            assert_eq!(out["windows"], 0);
            assert!(out["error"].is_string());
        }
        // The empty path keeps the summary's shape.
        let out = walk_forward(&log, &mut b, &[], &CryptoConfig::default());
        assert_eq!(
            (out["windows"].as_u64(), out["test_predictions"].as_u64()),
            (Some(0), Some(0))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_event_hour_parses_the_ticker() {
        assert_eq!(event_hour("KXBTCD-26SEP1019"), Some(19));
        assert_eq!(event_hour("KXBTCD-26SEP10"), None);
    }

    #[test]
    fn test_fit_recovers_an_inflated_volatility() {
        // Outcomes generated so the true vol is 2x the measured sigma: the fit
        // should choose c well above 1.
        let mut samples = Vec::new();
        for i in 0..400 {
            let z = ((i * 7919) % 1000) as f64 / 1000.0; // pseudo-uniform
            let spot = 100.0 * (1.0 + 0.004 * (z - 0.5));
            let true_q = sample(spot, 100.0, 0.0, 1.0, true).q(2.0);
            let u = ((i * 104_729) % 997) as f64 / 997.0;
            samples.push(sample(spot, 100.0, 0.3, 0.7, u < true_q));
        }
        let refs: Vec<&Sample> = samples.iter().collect();
        let f = fit(&refs);
        assert!(f.c > 1.4, "{f:?}");
    }
}
