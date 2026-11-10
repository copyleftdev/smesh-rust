//! Read-only HTTP for the three venues, with a disk cache and a journal entry
//! for every request.
//!
//! # Read-only by construction
//!
//! The client has one verb, `GET`, and every path is checked against an
//! allowlist of market-data prefixes before a request is built. Order,
//! portfolio, account and private endpoints are not on any list, so they
//! cannot be reached through this module however a caller builds the path.
//! Kraken is reached only through `/0/public/`, so its key is never needed and
//! never loaded.
//!
//! # Secrets
//!
//! Massive takes its key as an `apiKey` query parameter. The key is added at
//! send time and is never part of the cache key, a cached file, an error
//! message or a journal entry.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use smesh_runtime::Journal;
use tokio::sync::Mutex;

/// A data venue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Venue {
    /// Kalshi prediction markets (public market data).
    Kalshi,
    /// Kraken spot exchange (public endpoints only).
    Kraken,
    /// Massive (formerly Polygon.io) market data.
    Massive,
}

impl Venue {
    /// Name used in logs and cache paths.
    pub fn name(self) -> &'static str {
        match self {
            Venue::Kalshi => "kalshi",
            Venue::Kraken => "kraken",
            Venue::Massive => "massive",
        }
    }

    /// Base URL, without a trailing slash.
    pub fn base(self) -> &'static str {
        match self {
            Venue::Kalshi => "https://api.elections.kalshi.com/trade-api/v2",
            Venue::Kraken => "https://api.kraken.com",
            Venue::Massive => "https://api.massive.com",
        }
    }

    /// Path prefixes this client may request. Everything else is refused.
    pub fn allowed_prefixes(self) -> &'static [&'static str] {
        match self {
            Venue::Kalshi => &[
                "/series",
                "/markets",
                "/events",
                "/historical/cutoff",
                "/historical/markets",
                "/exchange/status",
            ],
            Venue::Kraken => &["/0/public/"],
            Venue::Massive => &["/v2/aggs/", "/v3/reference/", "/fed/v1/"],
        }
    }

    /// Minimum spacing between requests, kept under each venue's public limits.
    fn min_interval(self) -> Duration {
        match self {
            // Unsigned requests are limited harder than the documented
            // signed budgets; 250 ms still drew a 429 on one request in five.
            Venue::Kalshi => Duration::from_millis(400),
            Venue::Kraken => Duration::from_millis(1100),
            Venue::Massive => Duration::from_millis(60),
        }
    }
}

/// Why a request failed.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    /// The path is not a read-only market-data endpoint.
    #[error("refused: {venue} path {path:?} is not on the read-only allowlist")]
    NotAllowed {
        /// Venue name.
        venue: &'static str,
        /// The refused path.
        path: String,
    },
    /// The request did not complete.
    #[error("{venue} {path}: {message}")]
    Transport {
        /// Venue name.
        venue: &'static str,
        /// Path requested.
        path: String,
        /// What went wrong.
        message: String,
    },
    /// The venue answered with an error status after retries.
    #[error("{venue} {path}: HTTP {status}: {body}")]
    Status {
        /// Venue name.
        venue: &'static str,
        /// Path requested.
        path: String,
        /// HTTP status.
        status: u16,
        /// Response body, truncated.
        body: String,
    },
}

/// Check that `path` (with any query string) is allowed for `venue`.
///
/// Rejects parent-directory segments too, so an allowed prefix cannot be
/// walked out of (`/markets/../portfolio`).
pub fn check_allowed(venue: Venue, path: &str) -> Result<(), FetchError> {
    let bare = path.split('?').next().unwrap_or("");
    let ok = bare.starts_with('/')
        && !bare.split('/').any(|seg| seg == ".." || seg == ".")
        && venue
            .allowed_prefixes()
            .iter()
            .any(|prefix| bare.starts_with(prefix));
    if ok {
        Ok(())
    } else {
        Err(FetchError::NotAllowed {
            venue: venue.name(),
            path: bare.to_string(),
        })
    }
}

struct Pace {
    last: Option<Instant>,
}

/// A read-only, cached, journaled client for the three venues.
pub struct ReadOnlyClient {
    http: reqwest::Client,
    cache_dir: PathBuf,
    massive_key: Option<String>,
    pace: [Arc<Mutex<Pace>>; 3],
}

/// Where a response came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Fetched from the venue now.
    Network,
    /// Read from the disk cache.
    Cache,
}

impl ReadOnlyClient {
    /// A client caching under `cache_dir`, with the Massive key if one is set.
    pub fn new(cache_dir: impl Into<PathBuf>, massive_key: Option<String>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .user_agent("smesh-quant/0.1 (read-only research)")
            .build()
            .unwrap_or_default();
        let pace = || Arc::new(Mutex::new(Pace { last: None }));
        Self {
            http,
            cache_dir: cache_dir.into(),
            massive_key,
            pace: [pace(), pace(), pace()],
        }
    }

    fn pace_for(&self, venue: Venue) -> &Arc<Mutex<Pace>> {
        match venue {
            Venue::Kalshi => &self.pace[0],
            Venue::Kraken => &self.pace[1],
            Venue::Massive => &self.pace[2],
        }
    }

    fn cache_path(&self, venue: Venue, path: &str) -> PathBuf {
        let digest = Sha256::digest(format!("{}{}", venue.base(), path).as_bytes());
        let hex: String = digest.iter().take(16).map(|b| format!("{b:02x}")).collect();
        self.cache_dir
            .join(venue.name())
            .join(format!("{hex}.json"))
    }

    /// `GET` a path, journaling the request on `journal`.
    ///
    /// With `cache` set, a cached response is returned without a request, and
    /// a successful response is cached. Use it for data that cannot change
    /// (settled markets, closed bars); leave it off for live state.
    pub async fn get(
        &self,
        venue: Venue,
        path: &str,
        cache: bool,
        journal: &Journal,
    ) -> Result<(Value, Source), FetchError> {
        check_allowed(venue, path)?;
        let cache_file = self.cache_path(venue, path);
        let endpoint = path.split('?').next().unwrap_or(path).to_string();

        if cache {
            if let Ok(bytes) = std::fs::read(&cache_file) {
                if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
                    journal.record(
                        "fetch",
                        json!({
                            "venue": venue.name(), "endpoint": endpoint, "query": redact(path),
                            "source": "cache", "status": 200, "bytes": bytes.len(), "ms": 0,
                        }),
                    );
                    return Ok((value, Source::Cache));
                }
            }
        }

        let mut url = format!("{}{}", venue.base(), path);
        if venue == Venue::Massive {
            if let Some(key) = &self.massive_key {
                url.push(if path.contains('?') { '&' } else { '?' });
                url.push_str("apiKey=");
                url.push_str(key);
            }
        }

        let mut delay = Duration::from_millis(500);
        for attempt in 1..=7u32 {
            {
                let mut pace = self.pace_for(venue).lock().await;
                if let Some(last) = pace.last {
                    let wait = venue.min_interval().saturating_sub(last.elapsed());
                    if !wait.is_zero() {
                        tokio::time::sleep(wait).await;
                    }
                }
                pace.last = Some(Instant::now());
            }

            let started = Instant::now();
            let result = self.http.get(&url).send().await;
            let ms = started.elapsed().as_millis() as u64;
            let response = match result {
                Ok(r) => r,
                Err(e) => {
                    let message = redact(&e.to_string());
                    journal.record(
                        "fetch",
                        json!({"venue": venue.name(), "endpoint": endpoint, "query": redact(path),
                               "source": "network", "status": null, "ms": ms, "attempt": attempt,
                               "error": message}),
                    );
                    if attempt < 7 {
                        tokio::time::sleep(delay).await;
                        delay *= 2;
                        continue;
                    }
                    return Err(FetchError::Transport {
                        venue: venue.name(),
                        path: endpoint,
                        message,
                    });
                }
            };
            let status = response.status().as_u16();
            let body = response.bytes().await.unwrap_or_default();
            journal.record(
                "fetch",
                json!({"venue": venue.name(), "endpoint": endpoint, "query": redact(path),
                       "source": "network", "status": status, "bytes": body.len(), "ms": ms,
                       "attempt": attempt}),
            );

            if (status == 429 || status >= 500) && attempt < 7 {
                journal.record(
                    "backoff",
                    json!({"venue": venue.name(), "status": status, "wait_ms": delay.as_millis() as u64}),
                );
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(Duration::from_secs(30));
                continue;
            }
            if !(200..300).contains(&status) {
                return Err(FetchError::Status {
                    venue: venue.name(),
                    path: endpoint,
                    status,
                    body: redact(
                        &String::from_utf8_lossy(&body)
                            .chars()
                            .take(300)
                            .collect::<String>(),
                    ),
                });
            }
            let value: Value =
                serde_json::from_slice(&body).map_err(|e| FetchError::Transport {
                    venue: venue.name(),
                    path: endpoint.clone(),
                    message: format!("invalid JSON: {e}"),
                })?;
            if cache {
                write_cache(&cache_file, &body);
            }
            return Ok((value, Source::Network));
        }
        unreachable!("the retry loop returns")
    }
}

fn write_cache(path: &Path, body: &[u8]) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, body).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// Remove any `apiKey=...` value from text bound for a log or error.
pub fn redact(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("apiKey=") {
        out.push_str(&rest[..i + "apiKey=".len()]);
        out.push_str("REDACTED");
        let after = &rest[i + "apiKey=".len()..];
        let end = after.find(['&', ' ', '"', ')']).unwrap_or(after.len());
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_data_paths_are_allowed() {
        for (venue, path) in [
            (Venue::Kalshi, "/historical/markets?series_ticker=KXBTCD"),
            (
                Venue::Kalshi,
                "/series/KXBTCD/markets/X/candlesticks?period_interval=1",
            ),
            (Venue::Kalshi, "/events?series_ticker=KXCPI"),
            (Venue::Kraken, "/0/public/Trades?pair=XBTUSD"),
            (
                Venue::Massive,
                "/v2/aggs/ticker/SPY/range/1/minute/2026-01-01/2026-01-02",
            ),
            (Venue::Massive, "/fed/v1/inflation"),
        ] {
            assert!(check_allowed(venue, path).is_ok(), "{path}");
        }
    }

    #[test]
    fn test_trading_and_account_paths_are_refused() {
        for (venue, path) in [
            (Venue::Kalshi, "/portfolio/orders"),
            (Venue::Kalshi, "/portfolio/balance"),
            (Venue::Kalshi, "/orders"),
            (Venue::Kalshi, "/account/limits"),
            (Venue::Kalshi, "/communications/rfqs"),
            (Venue::Kalshi, "/markets/../portfolio/orders"),
            (Venue::Kraken, "/0/private/AddOrder"),
            (Venue::Kraken, "/0/private/Withdraw"),
            (Venue::Kraken, "/0/public/../private/Balance"),
            (Venue::Massive, "/v1/marketstatus/now"),
            (Venue::Kalshi, "historical/markets"),
        ] {
            assert!(
                check_allowed(venue, path).is_err(),
                "{path} should be refused"
            );
        }
    }

    #[test]
    fn test_redact_hides_the_massive_key() {
        assert_eq!(
            redact("https://api.massive.com/v2/x?limit=5&apiKey=SECRET123&b=1"),
            "https://api.massive.com/v2/x?limit=5&apiKey=REDACTED&b=1"
        );
        assert_eq!(
            redact("error for url (...?apiKey=SECRET)"),
            "error for url (...?apiKey=REDACTED)"
        );
        assert_eq!(redact("no key here"), "no key here");
    }

    #[test]
    fn test_cache_key_never_contains_the_secret() {
        let client = ReadOnlyClient::new("/tmp/x", Some("SECRET".into()));
        let p = client.cache_path(Venue::Massive, "/v2/aggs/ticker/SPY/range/1/day/a/b");
        assert!(!p.to_string_lossy().contains("SECRET"));
    }
}
