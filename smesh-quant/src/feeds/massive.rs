//! Massive (formerly Polygon.io) aggregates.

use chrono::{NaiveDate, Utc};
use serde_json::Value;
use smesh_runtime::Journal;

use crate::http::{FetchError, ReadOnlyClient, Venue};

/// One OHLCV bar.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Bar {
    /// Bar start, unix milliseconds.
    pub t_ms: i64,
    /// Open.
    pub o: f64,
    /// High.
    pub h: f64,
    /// Low.
    pub l: f64,
    /// Close.
    pub c: f64,
    /// Volume.
    pub v: f64,
}

fn parse_bars(v: &Value) -> Vec<Bar> {
    v.get("results")
        .and_then(Value::as_array)
        .map(|rs| {
            rs.iter()
                .filter_map(|r| {
                    Some(Bar {
                        t_ms: r.get("t")?.as_i64()?,
                        o: r.get("o")?.as_f64()?,
                        h: r.get("h")?.as_f64()?,
                        l: r.get("l")?.as_f64()?,
                        c: r.get("c")?.as_f64()?,
                        v: r.get("v").and_then(Value::as_f64).unwrap_or(0.0),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Bars of `mult` x `span` (`minute`, `hour`, `day`) for `ticker` over
/// `[from, to]`, following `next_url` pages. Ranges that end before today
/// are cached, since closed bars do not change.
pub async fn aggs(
    client: &ReadOnlyClient,
    journal: &Journal,
    ticker: &str,
    mult: u32,
    span: &str,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<Vec<Bar>, FetchError> {
    let cache = to < Utc::now().date_naive();
    let mut path = format!(
        "/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{from}/{to}?adjusted=true&sort=asc&limit=50000"
    );
    let mut bars = Vec::new();
    for _ in 0..200 {
        let (v, _) = client.get(Venue::Massive, &path, cache, journal).await?;
        bars.extend(parse_bars(&v));
        match v.get("next_url").and_then(Value::as_str) {
            Some(next) => {
                // next_url is absolute; keep only the path so the allowlist
                // and the key handling apply as for any other request.
                path = next
                    .strip_prefix(Venue::Massive.base())
                    .unwrap_or(next)
                    .to_string();
            }
            None => break,
        }
    }
    bars.sort_by_key(|b| b.t_ms);
    bars.dedup_by_key(|b| b.t_ms);
    Ok(bars)
}

/// The last bar starting at or before `t_ms`, if within `max_gap_ms`.
pub fn bar_at(bars: &[Bar], t_ms: i64, max_gap_ms: i64) -> Option<&Bar> {
    let i = bars.partition_point(|b| b.t_ms <= t_ms);
    let bar = bars.get(i.checked_sub(1)?)?;
    (t_ms - bar.t_ms <= max_gap_ms).then_some(bar)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(t: i64, c: f64) -> Bar {
        Bar {
            t_ms: t,
            o: c,
            h: c,
            l: c,
            c,
            v: 1.0,
        }
    }

    #[test]
    fn test_bar_at_finds_the_bar_in_force() {
        let bars = [bar(0, 1.0), bar(60_000, 2.0), bar(120_000, 3.0)];
        assert_eq!(bar_at(&bars, 90_000, 120_000).unwrap().c, 2.0);
        assert_eq!(bar_at(&bars, 120_000, 120_000).unwrap().c, 3.0);
        assert!(bar_at(&bars, -1, 120_000).is_none());
        // Too stale.
        assert!(bar_at(&bars, 600_000, 120_000).is_none());
    }

    #[test]
    fn test_parses_aggregates() {
        let v = serde_json::json!({"results": [{"v": 5.1, "o": 1.0, "c": 2.0, "h": 3.0, "l": 0.5, "t": 1000}]});
        let bars = parse_bars(&v);
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].c, 2.0);
    }
}
