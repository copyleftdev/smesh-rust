//! Tools written in Rust that ship with SMESH.
//!
//! These double as examples of the [`Tool`] trait. A config file names them
//! as `builtin.<name>` (for example `builtin.now`).

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{Effect, Tool, ToolError, ToolOutput};

/// The current date and time in UTC. Models do not know what day it is.
#[derive(Debug, Default, Clone, Copy)]
pub struct Now;

#[async_trait]
impl Tool for Now {
    fn name(&self) -> &str {
        "builtin__now"
    }

    fn description(&self) -> &str {
        "The current date and time in UTC, as RFC 3339. Use it whenever a task depends on today's date."
    }

    fn input_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }

    fn effect(&self) -> Effect {
        Effect::Read
    }

    async fn call(&self, _input: Value) -> Result<ToolOutput, ToolError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| ToolError::Failed(e.to_string()))?;
        Ok(ToolOutput::text(rfc3339(now.as_secs() as i64)))
    }
}

/// Every built-in tool, by the name a config file uses (`builtin.<name>`).
pub fn all() -> Vec<(&'static str, std::sync::Arc<dyn Tool>)> {
    vec![("now", std::sync::Arc::new(Now))]
}

/// Seconds since the epoch as an RFC 3339 UTC timestamp.
fn rfc3339(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        rem % 3_600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_matches_known_instants() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(1_794_330_000), "2026-11-10T17:00:00Z");
    }
}
