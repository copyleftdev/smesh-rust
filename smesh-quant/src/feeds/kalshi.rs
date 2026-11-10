//! Kalshi market data: events, markets and candlesticks, live or archived.
//!
//! Kalshi splits data at a cutoff (`GET /historical/cutoff`): markets that
//! settled before it, and their candlesticks, are only served by the
//! `/historical/...` endpoints. This module routes each request to the right
//! side. All of it is public market data; nothing here is signed.

use chrono::{DateTime, NaiveDate, Utc};
use serde_json::Value;
use smesh_runtime::Journal;

use crate::http::{FetchError, ReadOnlyClient, Venue};

/// A settled or open binary market.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Market {
    /// Market ticker, e.g. `KXBTCD-26SEP1019-T76799.99`.
    pub ticker: String,
    /// Event it belongs to.
    pub event_ticker: String,
    /// When trading opened.
    pub open_time: DateTime<Utc>,
    /// When trading closed.
    pub close_time: DateTime<Utc>,
    /// `greater`, `less`, `between`, ...
    pub strike_type: String,
    /// Lower strike, if any.
    pub floor_strike: Option<f64>,
    /// Upper strike, if any.
    pub cap_strike: Option<f64>,
    /// `yes`, `no`, or empty while open.
    pub result: String,
    /// The value the market settled against (a price, a CPI print, ...).
    pub expiration_value: Option<f64>,
    /// Contracts traded.
    pub volume: f64,
}

impl Market {
    /// Whether it resolved yes; `None` if unresolved.
    pub fn yes(&self) -> Option<bool> {
        match self.result.as_str() {
            "yes" => Some(true),
            "no" => Some(false),
            _ => None,
        }
    }
}

/// One candlestick's closing quote.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Quote {
    /// Inclusive end of the candle period (unix seconds).
    pub end_ts: i64,
    /// Best yes bid at period close, dollars.
    pub yes_bid: Option<f64>,
    /// Best yes ask at period close, dollars.
    pub yes_ask: Option<f64>,
    /// Last trade price at period close, dollars.
    pub last: Option<f64>,
    /// Contracts traded in the period.
    pub volume: f64,
}

impl Quote {
    /// Mid of a real two-sided quote: bid above 0, ask below 1, not crossed.
    pub fn mid(&self) -> Option<f64> {
        match (self.yes_bid, self.yes_ask) {
            (Some(b), Some(a)) if b > 0.0 && a < 1.0 && a >= b => Some((a + b) / 2.0),
            _ => None,
        }
    }
}

/// A Kalshi value that may be a number or a string, under `key` or
/// `key_dollars` / `key_fp` (live and historical responses differ).
///
/// Strings are parsed after removing thousands separators and unit marks:
/// settled values arrive as `"0.6"`, `"29,000"` or `"4.00%"`.
pub fn num(v: &Value, key: &str) -> Option<f64> {
    [
        key.to_string(),
        format!("{key}_dollars"),
        format!("{key}_fp"),
    ]
    .iter()
    .find_map(|k| match v.get(k)? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s
            .trim()
            .chars()
            .filter(|c| !matches!(c, ',' | '%' | '$' | ' '))
            .collect::<String>()
            .parse()
            .ok(),
        _ => None,
    })
}

fn time(v: &Value, key: &str) -> Option<DateTime<Utc>> {
    v.get(key)?.as_str()?.parse().ok()
}

fn parse_market(v: &Value) -> Option<Market> {
    Some(Market {
        ticker: v.get("ticker")?.as_str()?.to_string(),
        event_ticker: v.get("event_ticker")?.as_str()?.to_string(),
        open_time: time(v, "open_time")?,
        close_time: time(v, "close_time")?,
        strike_type: v
            .get("strike_type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        floor_strike: num(v, "floor_strike"),
        cap_strike: num(v, "cap_strike"),
        result: v
            .get("result")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        expiration_value: num(v, "expiration_value"),
        volume: num(v, "volume").unwrap_or(0.0),
    })
}

fn parse_quote(c: &Value) -> Option<Quote> {
    let close = |side: &str| c.get(side).and_then(|s| num(s, "close"));
    Some(Quote {
        end_ts: c.get("end_period_ts")?.as_i64()?,
        yes_bid: close("yes_bid"),
        yes_ask: close("yes_ask"),
        last: close("price"),
        volume: num(c, "volume").unwrap_or(0.0),
    })
}

/// The date and (Eastern) hour encoded in an event ticker such as
/// `KXBTCD-26SEP1019`, or just the date for `KXCPI-26SEP`-style monthly ones
/// (first of the month).
pub fn event_date(event_ticker: &str) -> Option<NaiveDate> {
    let code = event_ticker.split('-').nth(1)?;
    let yy: i32 = code.get(0..2)?.parse().ok()?;
    let month = match code.get(2..5)? {
        "JAN" => 1,
        "FEB" => 2,
        "MAR" => 3,
        "APR" => 4,
        "MAY" => 5,
        "JUN" => 6,
        "JUL" => 7,
        "AUG" => 8,
        "SEP" => 9,
        "OCT" => 10,
        "NOV" => 11,
        "DEC" => 12,
        _ => return None,
    };
    let day: u32 = code.get(5..7).and_then(|d| d.parse().ok()).unwrap_or(1);
    NaiveDate::from_ymd_opt(2000 + yy, month, day)
}

/// The Kalshi market-data client for one agent.
pub struct Kalshi<'a> {
    client: &'a ReadOnlyClient,
    journal: &'a Journal,
    cutoff: DateTime<Utc>,
}

impl<'a> Kalshi<'a> {
    /// Fetch the archive cutoff and return a client that routes by it.
    pub async fn connect(
        client: &'a ReadOnlyClient,
        journal: &'a Journal,
    ) -> Result<Self, FetchError> {
        let (v, _) = client
            .get(Venue::Kalshi, "/historical/cutoff", false, journal)
            .await?;
        let cutoff = time(&v, "market_settled_ts").unwrap_or_else(Utc::now);
        journal.record(
            "kalshi.cutoff",
            serde_json::json!({"market_settled_ts": cutoff}),
        );
        Ok(Self {
            client,
            journal,
            cutoff,
        })
    }

    /// Markets that settled before this are archived.
    pub fn cutoff(&self) -> DateTime<Utc> {
        self.cutoff
    }

    /// Settled event tickers in `series`, newest first, up to `max_pages`
    /// pages of 200.
    pub async fn settled_events(
        &self,
        series: &str,
        max_pages: usize,
    ) -> Result<Vec<String>, FetchError> {
        let mut out = Vec::new();
        let mut cursor = String::new();
        for _ in 0..max_pages {
            let mut path = format!("/events?series_ticker={series}&status=settled&limit=200");
            if !cursor.is_empty() {
                path.push_str(&format!("&cursor={cursor}"));
            }
            let (v, _) = self
                .client
                .get(Venue::Kalshi, &path, false, self.journal)
                .await?;
            let page: Vec<String> = v
                .get("events")
                .and_then(Value::as_array)
                .map(|evs| {
                    evs.iter()
                        .filter_map(|e| e.get("event_ticker")?.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let empty = page.is_empty();
            out.extend(page);
            cursor = v
                .get("cursor")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if cursor.is_empty() || empty {
                break;
            }
        }
        Ok(out)
    }

    /// Every market in a settled event. Asks the live set first for events
    /// dated after the cutoff and the archive first otherwise, falling back
    /// to the other side if the first is empty.
    pub async fn event_markets(&self, event: &str) -> Result<Vec<Market>, FetchError> {
        let recent = event_date(event)
            .map(|d| d >= self.cutoff.date_naive())
            .unwrap_or(true);
        let live = format!("/markets?event_ticker={event}&limit=1000");
        let archived = format!("/historical/markets?event_ticker={event}&limit=1000");
        let order = if recent {
            [&live, &archived]
        } else {
            [&archived, &live]
        };
        for path in order {
            let (v, _) = self
                .client
                .get(Venue::Kalshi, path, true, self.journal)
                .await?;
            let markets: Vec<Market> = v
                .get("markets")
                .and_then(Value::as_array)
                .map(|ms| ms.iter().filter_map(parse_market).collect())
                .unwrap_or_default();
            if !markets.is_empty() {
                return Ok(markets);
            }
        }
        Ok(Vec::new())
    }

    /// Candlesticks for a market between two unix times, `period` minutes
    /// each (1, 60 or 1440).
    pub async fn candles(
        &self,
        series: &str,
        market: &Market,
        start_ts: i64,
        end_ts: i64,
        period: u32,
    ) -> Result<Vec<Quote>, FetchError> {
        let path = if market.close_time < self.cutoff {
            format!(
                "/historical/markets/{}/candlesticks?start_ts={start_ts}&end_ts={end_ts}&period_interval={period}",
                market.ticker
            )
        } else {
            format!(
                "/series/{series}/markets/{}/candlesticks?start_ts={start_ts}&end_ts={end_ts}&period_interval={period}",
                market.ticker
            )
        };
        let settled = market.yes().is_some();
        let (v, _) = self
            .client
            .get(Venue::Kalshi, &path, settled, self.journal)
            .await?;
        Ok(v.get("candlesticks")
            .and_then(Value::as_array)
            .map(|cs| cs.iter().filter_map(parse_quote).collect())
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parses_historical_and_live_candle_shapes() {
        let historical = json!({"end_period_ts": 10, "volume": "118705.17",
            "price": {"close": "0.9900"}, "yes_bid": {"close": "0.9800"}, "yes_ask": {"close": "1.0000"}});
        let live = json!({"end_period_ts": 20, "volume_fp": "1569.89",
            "price": {"close_dollars": "0.4100"}, "yes_bid": {"close_dollars": "0.4000"},
            "yes_ask": {"close_dollars": "0.4300"}});
        let h = parse_quote(&historical).unwrap();
        let l = parse_quote(&live).unwrap();
        assert_eq!(
            (h.yes_bid, h.yes_ask, h.last, h.volume),
            (Some(0.98), Some(1.0), Some(0.99), 118705.17)
        );
        assert_eq!(l.yes_bid, Some(0.40));
        assert!((l.mid().unwrap() - 0.415).abs() < 1e-12);
        // An ask of 1.00 is no real offer, so there is no two-sided mid.
        assert_eq!(h.mid(), None);
    }

    #[test]
    fn test_parses_a_settled_market() {
        let v = json!({"ticker": "KXCPI-26SEP-T0.3", "event_ticker": "KXCPI-26SEP",
            "open_time": "2026-09-11T14:00:00Z", "close_time": "2026-10-14T12:29:00Z",
            "strike_type": "greater", "floor_strike": 0.3, "cap_strike": null,
            "result": "yes", "expiration_value": "0.6", "volume_fp": "12.5"});
        let m = parse_market(&v).unwrap();
        assert_eq!(m.yes(), Some(true));
        assert_eq!(m.expiration_value, Some(0.6));
        assert_eq!(m.floor_strike, Some(0.3));
        assert_eq!(m.cap_strike, None);
        assert_eq!(m.volume, 12.5);
    }

    #[test]
    fn test_settled_values_parse_in_every_format_seen() {
        for (raw, want) in [
            ("0.6", 0.6),
            ("29,000", 29000.0),
            ("4.00%", 4.0),
            ("-23000", -23000.0),
            ("$1,234.5", 1234.5),
        ] {
            assert_eq!(
                num(&json!({"expiration_value": raw}), "expiration_value"),
                Some(want),
                "{raw}"
            );
        }
        assert_eq!(
            num(&json!({"expiration_value": "n/a"}), "expiration_value"),
            None
        );
    }

    #[test]
    fn test_event_dates() {
        assert_eq!(
            event_date("KXBTCD-26SEP1019"),
            NaiveDate::from_ymd_opt(2026, 9, 10)
        );
        assert_eq!(
            event_date("KXCPI-26SEP"),
            NaiveDate::from_ymd_opt(2026, 9, 1)
        );
        assert_eq!(event_date("garbage"), None);
    }
}
