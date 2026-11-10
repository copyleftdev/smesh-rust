//! Track B: what do Kalshi's macro-release markets know, and do markets
//! react to the surprise against them?
//!
//! For each settled release event (CPI, payrolls, Fed decisions), the scouts
//! read Kalshi's "greater than k" ladder `lead` minutes before the release.
//! The ladder is a survival function `P(X > k)`, from which the analyst
//! derives an implied expectation. The settled `expiration_value` is the
//! actual print. Massive's minute bars give SPY's and TLT's moves from just
//! before the release to `reaction` minutes after.
//!
//! The walk-forward is expanding: each event in turn is the unseen test, and
//! everything before it is training. Kalshi is judged against fair baselines
//! fitted only on the past: the previous print (naive) and the past
//! frequency of each outcome (climatology).

use chrono::{DateTime, Duration, Utc};
use serde_json::json;

use crate::board::Board;
use crate::feeds::kalshi::{Kalshi, Market};
use crate::feeds::massive::{aggs, bar_at, Bar};
use crate::http::ReadOnlyClient;
use crate::log::{flow, RunLog};
use crate::stats::{brier, mean, ols};

/// One release series.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Series {
    /// Display name.
    pub name: &'static str,
    /// Kalshi series ticker.
    pub kalshi: &'static str,
    /// Whether the value only takes values on the rung grid (CPI in 0.1
    /// steps, the Fed rate in 0.25), so `P(k_i < X <= k_i+1)` is all at
    /// `k_i+1` rather than spread between the rungs.
    pub discrete: bool,
}

/// CPI, payrolls and the Fed.
pub const SERIES: [Series; 3] = [
    Series {
        name: "CPI",
        kalshi: "KXCPI",
        discrete: true,
    },
    Series {
        name: "Payrolls",
        kalshi: "KXPAYROLLS",
        discrete: false,
    },
    Series {
        name: "Fed",
        kalshi: "KXFED",
        discrete: true,
    },
];

/// Settings for the macro track.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MacroConfig {
    /// Most recent settled events per series.
    pub max_events_per_series: usize,
    /// Read the ladder this many minutes before the release.
    pub lead_minutes: i64,
    /// Measure the market reaction over this many minutes after release.
    pub reaction_minutes: i64,
    /// Events before the first test.
    pub min_train: usize,
}

impl Default for MacroConfig {
    fn default() -> Self {
        Self {
            max_events_per_series: 24,
            lead_minutes: 60,
            reaction_minutes: 30,
            min_train: 8,
        }
    }
}

/// One threshold of a ladder at decision time.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Rung {
    /// Threshold `k` in `P(X > k)`.
    pub k: f64,
    /// Kalshi's probability, made monotone across the ladder.
    pub p: f64,
    /// Whether `X > k` settled yes.
    pub outcome: bool,
}

/// One release.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Release {
    /// Series name.
    pub series: &'static str,
    /// Event ticker.
    pub event: String,
    /// Release time (market close + 1 minute).
    pub release: DateTime<Utc>,
    /// The ladder at decision time.
    pub ladder: Vec<Rung>,
    /// Typical gap between thresholds.
    pub spacing: f64,
    /// Expectation implied by the ladder.
    pub implied: f64,
    /// The actual print.
    pub actual: f64,
    /// SPY return over the reaction window, if bars exist.
    pub spy: Option<f64>,
    /// TLT return over the reaction window, if bars exist.
    pub tlt: Option<f64>,
}

impl Release {
    /// Surprise in units of threshold spacing, comparable across series.
    pub fn z(&self) -> f64 {
        (self.actual - self.implied) / self.spacing.max(1e-9)
    }
}

/// Make `P(X > k)` non-increasing in `k` (rungs sorted by `k`).
pub fn monotone(ladder: &mut [Rung]) {
    let mut cap = 1.0f64;
    for r in ladder.iter_mut() {
        r.p = r.p.min(cap).clamp(0.0, 1.0);
        cap = r.p;
    }
}

/// The expectation implied by a monotone survival ladder `P(X > k)`.
///
/// For a continuous value the mass between two rungs sits at their midpoint
/// and each tail half a spacing beyond the outermost rung. For a `discrete`
/// value on the rung grid, `P(k_i < X <= k_i+1)` is the probability that
/// `X = k_i+1`, so that mass sits on the upper rung, the lower tail on the
/// lowest rung, and the upper tail one spacing above the highest. Placing
/// discrete mass at midpoints biases the expectation down by half a step.
pub fn implied_mean(ladder: &[Rung], spacing: f64, discrete: bool) -> f64 {
    let n = ladder.len();
    if n == 0 {
        return f64::NAN;
    }
    let (below, between, above): (f64, fn(f64, f64) -> f64, f64) = if discrete {
        (ladder[0].k, |_, hi| hi, ladder[n - 1].k + spacing)
    } else {
        (
            ladder[0].k - spacing / 2.0,
            |lo, hi| (lo + hi) / 2.0,
            ladder[n - 1].k + spacing / 2.0,
        )
    };
    let mut e = (1.0 - ladder[0].p) * below;
    for w in ladder.windows(2) {
        e += (w[0].p - w[1].p) * between(w[0].k, w[1].k);
    }
    e + ladder[n - 1].p * above
}

/// The settled value as pinned by a ladder's outcomes, when the market did
/// not report one: above the highest rung that settled yes and at or below
/// the lowest that settled no. Exact for a discrete value (the lowest no);
/// the bracket's midpoint for a continuous one. `None` if the ladder does
/// not bracket it.
pub fn value_from_outcomes(markets: &[Market], discrete: bool) -> Option<f64> {
    let max_yes = markets
        .iter()
        .filter(|m| m.yes() == Some(true))
        .filter_map(|m| m.floor_strike)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_no = markets
        .iter()
        .filter(|m| m.yes() == Some(false))
        .filter_map(|m| m.floor_strike)
        .fold(f64::INFINITY, f64::min);
    if !max_yes.is_finite() || !min_no.is_finite() || min_no <= max_yes {
        return None;
    }
    Some(if discrete {
        min_no
    } else {
        (max_yes + min_no) / 2.0
    })
}

/// Releases happen on the hour or half hour; markets close shortly before
/// (12:29 for an 8:30 ET print, 17:55 for a 2 pm ET Fed decision). The
/// release is the first :00 or :30 after close.
pub fn release_time(close: DateTime<Utc>) -> DateTime<Utc> {
    let secs = close.timestamp();
    let next = (secs / 1800 + 1) * 1800;
    DateTime::from_timestamp(next, 0).unwrap_or(close)
}

fn median_gap(ks: &[f64]) -> f64 {
    let mut gaps: Vec<f64> = ks
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|g| *g > 0.0)
        .collect();
    gaps.sort_by(f64::total_cmp);
    gaps.get(gaps.len() / 2).copied().unwrap_or(1.0)
}

async fn reaction(
    client: &ReadOnlyClient,
    journal: &smesh_runtime::Journal,
    ticker: &str,
    release: DateTime<Utc>,
    minutes: i64,
) -> Option<f64> {
    let day = release.date_naive();
    let bars: Vec<Bar> = aggs(client, journal, ticker, 1, "minute", day, day)
        .await
        .ok()?;
    let before = bar_at(
        &bars,
        (release - Duration::minutes(1)).timestamp_millis(),
        10 * 60_000,
    )?;
    let after = bar_at(
        &bars,
        (release + Duration::minutes(minutes)).timestamp_millis(),
        10 * 60_000,
    )?;
    (before.c > 0.0).then(|| after.c / before.c - 1.0)
}

/// Gather releases for every series.
pub async fn gather(
    log: &RunLog,
    client: &ReadOnlyClient,
    cfg: &MacroConfig,
) -> anyhow::Result<Vec<Release>> {
    let kalshi_j = log.agent("scout.kalshi", "data");
    let massive_j = log.agent("scout.massive", "data");
    let analyst = log.agent("analyst.macro", "research");
    let kalshi = Kalshi::connect(client, &kalshi_j).await?;

    let mut out = Vec::new();
    for series in SERIES {
        let events = kalshi.settled_events(series.kalshi, 2).await?;
        kalshi_j.record(
            "events.selected",
            json!({"series": series.kalshi, "available": events.len(), "taking": cfg.max_events_per_series.min(events.len())}),
        );
        for event in events.iter().take(cfg.max_events_per_series) {
            let markets: Vec<Market> = kalshi
                .event_markets(event)
                .await?
                .into_iter()
                .filter(|m| {
                    m.strike_type == "greater" && m.floor_strike.is_some() && m.yes().is_some()
                })
                .collect();
            if markets.len() < 3 {
                kalshi_j.record(
                    "event.skipped",
                    json!({"event": event, "reason": "fewer than 3 settled greater-than rungs"}),
                );
                continue;
            }
            let close = markets.iter().map(|m| m.close_time).max().unwrap();
            let reported = markets.iter().find_map(|m| m.expiration_value);
            let actual = reported.or_else(|| value_from_outcomes(&markets, series.discrete));
            let Some(actual) = actual else {
                kalshi_j.record("event.skipped", json!({"event": event, "reason": "no settled value and outcomes do not bracket one"}));
                continue;
            };
            if reported.is_none() {
                kalshi_j.record(
                    "value.inferred",
                    json!({"event": event, "value": actual, "from": "rung outcomes"}),
                );
            }
            let t = close - Duration::minutes(cfg.lead_minutes);

            let mut ladder = Vec::new();
            for m in &markets {
                let quotes = kalshi
                    .candles(series.kalshi, m, t.timestamp() - 3600, t.timestamp(), 1)
                    .await?;
                let p = quotes
                    .iter()
                    .rev()
                    .find_map(|q| q.mid().or(q.last.filter(|_| q.volume > 0.0)));
                match p {
                    Some(p) => ladder.push(Rung {
                        k: m.floor_strike.unwrap(),
                        p,
                        outcome: m.yes().unwrap(),
                    }),
                    None => kalshi_j.record("market.skipped", json!({"ticker": m.ticker, "reason": "no quote or trade in the hour before decision"})),
                }
            }
            if ladder.len() < 3 {
                kalshi_j.record(
                    "event.skipped",
                    json!({"event": event, "reason": "fewer than 3 priced rungs"}),
                );
                continue;
            }
            ladder.sort_by(|a, b| a.k.total_cmp(&b.k));
            monotone(&mut ladder);
            let ks: Vec<f64> = ladder.iter().map(|r| r.k).collect();
            let spacing = median_gap(&ks);
            let implied = implied_mean(&ladder, spacing, series.discrete);
            let release = release_time(close);

            let spy = reaction(client, &massive_j, "SPY", release, cfg.reaction_minutes).await;
            let tlt = reaction(client, &massive_j, "TLT", release, cfg.reaction_minutes).await;

            let r = Release {
                series: series.name,
                event: event.clone(),
                release,
                ladder,
                spacing,
                implied,
                actual,
                spy,
                tlt,
            };
            analyst.record("sample", serde_json::to_value(&r).unwrap_or_default());
            out.push(r);
        }
        flow(
            &kalshi_j,
            "venue.kalshi",
            "scout.kalshi",
            &format!("{} ladders", series.name),
            out.len(),
        );
    }
    flow(
        &massive_j,
        "venue.massive",
        "scout.massive",
        "reaction bars",
        out.len() * 2,
    );
    flow(
        &kalshi_j,
        "scout.kalshi",
        "analyst.macro",
        "releases",
        out.len(),
    );
    out.sort_by_key(|r| r.release);
    Ok(out)
}

/// The journal for the macro board.
pub fn board(log: &RunLog) -> Board {
    Board::new("macro", log.agent("field.macro", "mesh"))
}

/// Past frequency of `X > k` among earlier releases of the same series, with
/// a half-count prior so it is never exactly 0 or 1.
fn climatology(past: &[&Release], k: f64) -> f64 {
    let n = past.len() as f64;
    let above = past.iter().filter(|r| r.actual > k).count() as f64;
    (above + 0.5) / (n + 1.0)
}

/// Walk forward over releases, one unseen release per window.
pub fn walk_forward(
    log: &RunLog,
    board: &mut Board,
    releases: &[Release],
    cfg: &MacroConfig,
) -> serde_json::Value {
    let walker = log.agent("walker", "validation");
    let analyst = log.agent("analyst.macro", "research");
    let skeptic = log.agent("skeptic", "validation");

    let mut errors_kalshi = Vec::new();
    let mut errors_naive = Vec::new();
    let mut errors_by_series = std::collections::BTreeMap::<&str, (Vec<f64>, Vec<f64>)>::new();
    let mut brier_pairs_market = Vec::new();
    let mut brier_pairs_clim = Vec::new();
    let mut hits = std::collections::BTreeMap::<&str, (u32, u32)>::new();
    let mut windows = Vec::new();

    for (i, test) in releases.iter().enumerate().skip(cfg.min_train) {
        let train = &releases[..i];
        let index = i - cfg.min_train + 1;
        walker.record(
            "walk.window",
            json!({"track": "macro", "index": index, "train": [train.first().map(|r| r.release), train.last().map(|r| r.release)],
                   "test": [test.release, test.release], "n_train": train.len(), "n_test": 1, "event": test.event}),
        );
        crate::log::entity(
            &walker,
            &format!("window.macro.{index}"),
            "window",
            &test.event,
            "macro",
        );

        if index == 1 {
            board.propose(
                &analyst,
                "macro.kalshi_beats_last_print",
                "Kalshi's implied expectation forecasts the release better than the previous print",
                json!({}),
            );
            board.propose(
                &analyst,
                "macro.kalshi_beats_climatology",
                "Kalshi's threshold odds beat the historical frequency of each outcome",
                json!({}),
            );
            board.propose(&analyst, "macro.surprise_moves_spy", "The surprise against Kalshi's expectation predicts SPY's direction in the 30 minutes after release", json!({}));
            board.propose(&analyst, "macro.surprise_moves_tlt", "The surprise against Kalshi's expectation predicts TLT's direction in the 30 minutes after release", json!({}));
        }
        board.tick();

        let same: Vec<&Release> = train.iter().filter(|r| r.series == test.series).collect();
        let mut window = json!({"track": "macro", "window": index, "event": test.event, "series": test.series,
                                "implied": test.implied, "actual": test.actual, "z": test.z()});

        // Kalshi vs the last print.
        if let Some(prev) = same.last() {
            let ek = (test.actual - test.implied).abs();
            let en = (test.actual - prev.actual).abs();
            errors_kalshi.push(ek / test.spacing);
            errors_naive.push(en / test.spacing);
            let e = errors_by_series.entry(test.series).or_default();
            e.0.push(ek);
            e.1.push(en);
            walker.record("prediction", json!({"track": "macro", "window": index, "event": test.event,
                "kind": "point", "kalshi": test.implied, "naive": prev.actual, "actual": test.actual}));
            window["abs_error"] = json!({"kalshi": ek, "naive": en});
            if ek <= en {
                board.confirm(
                    &walker,
                    "macro.kalshi_beats_last_print",
                    json!({"kalshi": ek, "naive": en}),
                );
            } else {
                board.contradict(
                    &walker,
                    "macro.kalshi_beats_last_print",
                    json!({"kalshi": ek, "naive": en}),
                );
            }
        }

        // Kalshi's ladder vs climatology.
        if same.len() >= 4 {
            let market: Vec<(f64, bool)> = test.ladder.iter().map(|r| (r.p, r.outcome)).collect();
            let clim: Vec<(f64, bool)> = test
                .ladder
                .iter()
                .map(|r| (climatology(&same, r.k), r.outcome))
                .collect();
            for r in &test.ladder {
                walker.record("prediction", json!({"track": "macro", "window": index, "event": test.event,
                    "kind": "threshold", "k": r.k, "p_market": r.p, "p_climatology": climatology(&same, r.k), "outcome": r.outcome}));
            }
            let (bm, bc) = (brier(&market), brier(&clim));
            brier_pairs_market.extend(market);
            brier_pairs_clim.extend(clim);
            window["brier"] = json!({"kalshi": bm, "climatology": bc});
            if bm <= bc {
                board.confirm(
                    &walker,
                    "macro.kalshi_beats_climatology",
                    json!({"kalshi": bm, "climatology": bc}),
                );
            } else {
                board.contradict(
                    &walker,
                    "macro.kalshi_beats_climatology",
                    json!({"kalshi": bm, "climatology": bc}),
                );
            }
        } else {
            board.dampen(
                &skeptic,
                "macro.kalshi_beats_climatology",
                0.97,
                &format!(
                    "only {} past {} releases for a climatology",
                    same.len(),
                    test.series
                ),
            );
        }

        // Reactions: fit on past releases with a measured reaction.
        for (name, id, pick) in [
            (
                "SPY",
                "macro.surprise_moves_spy",
                (|r: &Release| r.spy) as fn(&Release) -> Option<f64>,
            ),
            ("TLT", "macro.surprise_moves_tlt", |r: &Release| r.tlt),
        ] {
            let pts: Vec<(f64, f64)> = train
                .iter()
                .filter_map(|r| pick(r).map(|y| (r.z(), y)))
                .collect();
            let (Some(actual), Some((a, b))) = (pick(test), ols(&pts)) else {
                continue;
            };
            let predicted = a + b * test.z();
            let hit = predicted.signum() == actual.signum();
            let e = hits.entry(name).or_default();
            e.0 += hit as u32;
            e.1 += 1;
            analyst.record("model.fit", json!({"track": "macro", "window": index, "asset": name, "params": {"a": a, "b": b}, "n_train": pts.len()}));
            walker.record("prediction", json!({"track": "macro", "window": index, "event": test.event, "kind": "reaction",
                "asset": name, "z": test.z(), "predicted": predicted, "actual": actual, "hit": hit}));
            if hit {
                board.confirm(
                    &walker,
                    id,
                    json!({"asset": name, "predicted": predicted, "actual": actual}),
                );
            } else {
                board.contradict(
                    &walker,
                    id,
                    json!({"asset": name, "predicted": predicted, "actual": actual}),
                );
            }
            if pts.len() < 12 {
                board.dampen(
                    &skeptic,
                    id,
                    0.95,
                    &format!("slope fitted on {} releases", pts.len()),
                );
            }
        }
        walker.record("score", window.clone());
        windows.push(window);
    }

    skeptic.record("multiple_testing", json!({"track": "macro", "hypotheses": 4,
        "note": "each window is a single release: one window's verdict is weak evidence on its own"}));

    let hit_rates: serde_json::Map<String, serde_json::Value> = hits
        .iter()
        .map(|(k, (h, n))| {
            (
                k.to_string(),
                json!({"hits": h, "n": n, "rate": *h as f64 / (*n).max(1) as f64}),
            )
        })
        .collect();
    let by_series: serde_json::Map<String, serde_json::Value> = SERIES
        .iter()
        .map(|s| {
            (
                s.name.to_string(),
                json!(releases.iter().filter(|r| r.series == s.name).count()),
            )
        })
        .collect();
    let mae_by_series: serde_json::Map<String, serde_json::Value> = errors_by_series
        .iter()
        .map(|(k, (ek, en))| {
            (k.to_string(), json!({"kalshi": mean(ek.iter().copied()), "naive": mean(en.iter().copied()), "n": ek.len()}))
        })
        .collect();
    let summary = json!({
        "releases": releases.len(), "by_series": by_series, "windows": windows.len(),
        "mean_abs_error_by_series": mae_by_series,
        "mean_abs_error_in_spacings": {"kalshi": mean(errors_kalshi.iter().copied()), "naive": mean(errors_naive.iter().copied())},
        "brier": {"kalshi": brier(&brier_pairs_market), "climatology": brier(&brier_pairs_clim), "n": brier_pairs_market.len()},
        "direction_hit_rate": hit_rates,
        "per_window": windows,
    });
    walker.record(
        "walk.summary",
        json!({"track": "macro", "summary": summary.clone()}),
    );
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rung(k: f64, p: f64) -> Rung {
        Rung {
            k,
            p,
            outcome: false,
        }
    }

    #[test]
    fn test_monotone_enforces_a_survival_function() {
        let mut l = vec![
            rung(0.1, 0.9),
            rung(0.2, 0.95),
            rung(0.3, 0.4),
            rung(0.4, 0.5),
        ];
        monotone(&mut l);
        let ps: Vec<f64> = l.iter().map(|r| r.p).collect();
        assert_eq!(ps, vec![0.9, 0.9, 0.4, 0.4]);
    }

    #[test]
    fn test_implied_mean_of_a_symmetric_continuous_ladder_is_its_centre() {
        // P(X>0.1)=0.9, P(X>0.2)=0.5, P(X>0.3)=0.1: symmetric about 0.2.
        let l = vec![rung(0.1, 0.9), rung(0.2, 0.5), rung(0.3, 0.1)];
        assert!((implied_mean(&l, 0.1, false) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_discrete_ladder_puts_mass_on_the_grid() {
        // A CPI print certain to be 0.6: P(X>0.5)=1, P(X>0.6)=0.
        let l = vec![
            rung(0.4, 1.0),
            rung(0.5, 1.0),
            rung(0.6, 0.0),
            rung(0.7, 0.0),
        ];
        assert!((implied_mean(&l, 0.1, true) - 0.6).abs() < 1e-12);
        // The continuous rule would put it half a step low.
        assert!((implied_mean(&l, 0.1, false) - 0.55).abs() < 1e-12);
    }

    #[test]
    fn test_implied_mean_of_a_certain_ladder() {
        // Certainly above 0.3: all mass beyond the top rung.
        let l = vec![rung(0.1, 1.0), rung(0.2, 1.0), rung(0.3, 1.0)];
        assert!((implied_mean(&l, 0.1, false) - 0.35).abs() < 1e-12);
        assert!((implied_mean(&l, 0.1, true) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_release_time_is_the_next_half_hour() {
        let t = |s: &str| s.parse::<DateTime<Utc>>().unwrap();
        assert_eq!(
            release_time(t("2026-10-14T12:29:00Z")),
            t("2026-10-14T12:30:00Z")
        );
        assert_eq!(
            release_time(t("2026-10-28T17:55:00Z")),
            t("2026-10-28T18:00:00Z")
        );
        assert_eq!(
            release_time(t("2026-11-06T13:29:00Z")),
            t("2026-11-06T13:30:00Z")
        );
    }

    fn market(k: f64, result: &str) -> Market {
        Market {
            ticker: "T".into(),
            event_ticker: "E".into(),
            open_time: Utc::now(),
            close_time: Utc::now(),
            strike_type: "greater".into(),
            floor_strike: Some(k),
            cap_strike: None,
            result: result.into(),
            expiration_value: None,
            volume: 0.0,
        }
    }

    #[test]
    fn test_value_from_outcomes_brackets_the_print() {
        // Fed: above 3.75 (yes), not above 4.0 (no): the rate is 4.0.
        let ms = [
            market(3.5, "yes"),
            market(3.75, "yes"),
            market(4.0, "no"),
            market(4.25, "no"),
        ];
        assert_eq!(value_from_outcomes(&ms, true), Some(4.0));
        assert_eq!(value_from_outcomes(&ms, false), Some(3.875));
        // Not bracketed.
        assert_eq!(value_from_outcomes(&[market(1.0, "yes")], true), None);
    }

    #[test]
    fn test_climatology_never_reaches_certainty() {
        let r = Release {
            series: "CPI",
            event: "E".into(),
            release: Utc::now(),
            ladder: vec![],
            spacing: 0.1,
            implied: 0.2,
            actual: 0.3,
            spy: None,
            tlt: None,
        };
        let past = vec![&r, &r, &r];
        let p = climatology(&past, 0.1);
        assert!(p > 0.8 && p < 1.0, "{p}");
    }
}
