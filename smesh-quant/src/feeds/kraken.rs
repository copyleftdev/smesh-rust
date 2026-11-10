//! Kraken public trades, used as an independent spot price.
//!
//! Only `/0/public/` is reachable through the client, so no Kraken key is
//! loaded or needed.

use serde_json::Value;
use smesh_runtime::Journal;

use crate::http::{FetchError, ReadOnlyClient, Venue};

/// The first trade at or after `ts` (unix seconds): `(price, trade time)`.
///
/// Kraken's trade history pages forward from any `since` (nanoseconds), so
/// the first trade returned is the market at that moment.
pub async fn price_at(
    client: &ReadOnlyClient,
    journal: &Journal,
    pair: &str,
    ts: i64,
) -> Result<Option<(f64, f64)>, FetchError> {
    let path = format!(
        "/0/public/Trades?pair={pair}&since={}&count=20",
        ts as i128 * 1_000_000_000
    );
    let (v, _) = client.get(Venue::Kraken, &path, true, journal).await?;
    if let Some(errors) = v.get("error").and_then(Value::as_array) {
        if !errors.is_empty() {
            journal.record(
                "kraken.error",
                serde_json::json!({"pair": pair, "errors": errors}),
            );
            return Ok(None);
        }
    }
    Ok(first_trade(&v))
}

fn first_trade(v: &Value) -> Option<(f64, f64)> {
    let result = v.get("result")?.as_object()?;
    let (_, trades) = result.iter().find(|(k, _)| k.as_str() != "last")?;
    let t = trades.as_array()?.first()?;
    let price: f64 = t.get(0)?.as_str()?.parse().ok()?;
    let at = t.get(2)?.as_f64()?;
    Some((price, at))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reads_the_first_trade() {
        let v = serde_json::json!({"error": [], "result": {
            "XXBTZUSD": [["76812.5", "0.01", 1789081200.123, "b", "l", "", 1]],
            "last": "1789081200123000000"}});
        assert_eq!(first_trade(&v), Some((76812.5, 1789081200.123)));
    }
}
