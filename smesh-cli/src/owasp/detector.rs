//! The SMESH detection mesh for a single OWASP Benchmark test case.
//!
//! Each specialist agent reads the Java source and reports whether it contains a
//! real, exploitable vulnerability and of which OWASP category. Every report is
//! emitted as a signal into a shared [`Field`]; signals for the same category
//! reinforce each other, so agreement between agents raises a category's
//! consensus. A category is "flagged" when enough agents agree (reinforcement)
//! or a single agent is highly confident.

use std::collections::HashMap;

use smesh_agent::LlmBackend;
use smesh_core::{Field, Signal, SignalType};

use super::corpus::OwaspCategory;

/// Reference block listing every category so agents emit a valid code.
const CATEGORY_REFERENCE: &str = "\
OWASP categories (use the exact code):
- sqli: SQL injection (untrusted input reaches a SQL query)
- cmdi: OS command injection (untrusted input reaches Runtime.exec / ProcessBuilder)
- ldapi: LDAP injection (untrusted input reaches an LDAP filter/search)
- xpathi: XPath injection (untrusted input reaches an XPath expression)
- pathtraver: path traversal (untrusted input reaches a file path)
- xss: reflected cross-site scripting (untrusted input written to an HTTP response unencoded)
- trustbound: trust boundary violation (untrusted input stored into an HTTP session attribute)
- securecookie: insecure cookie (a sensitive cookie created without the Secure flag)
- crypto: weak encryption (DES, 3DES, RC2/4, ECB mode, etc.)
- hash: weak hashing (MD5, SHA-1, etc.)
- weakrand: weak randomness (java.util.Random used for security-sensitive values)";

const OUTPUT_CONTRACT: &str = "\
Respond with ONLY a JSON object, no prose:
{\"vulnerable\": <true|false>, \"category\": \"<one code above, or none>\", \"confidence\": <0.0-1.0>}
Report `vulnerable:true` ONLY when untrusted input actually reaches a dangerous sink \
without effective sanitisation (a genuinely exploitable flaw). If the input is validated, \
constant, or the sink is safe, report `vulnerable:false` with category `none`.";

/// A detector persona. Every agent can report any category; the persona only
/// shapes emphasis, giving the mesh diverse viewpoints that can still converge.
#[derive(Debug, Clone)]
pub struct DetectorAgent {
    #[allow(dead_code)] // label for readability/debug
    pub id: String,
    pub system_prompt: String,
}

impl DetectorAgent {
    fn new(id: &str, lens: &str) -> Self {
        let system_prompt = format!(
            "You are {id}, a precise Java application-security auditor. {lens}\n\n{CATEGORY_REFERENCE}\n\n{OUTPUT_CONTRACT}"
        );
        Self {
            id: id.to_string(),
            system_prompt,
        }
    }
}

/// Build the panel of detector agents (1..=4). Each has a distinct lens but
/// judges the full category set, so multiple agents can agree on one category.
pub fn default_agents(n: usize) -> Vec<DetectorAgent> {
    let all = vec![
        DetectorAgent::new(
            "DataFlowHunter",
            "You specialise in taint flow: tracing untrusted request data (parameters, headers, \
             cookies) into injection and path sinks (SQL, OS commands, LDAP, XPath, file paths).",
        ),
        DetectorAgent::new(
            "OutputGuard",
            "You specialise in how data leaves the app: unencoded output to HTTP responses (XSS), \
             session/trust-boundary storage, and cookie security flags.",
        ),
        DetectorAgent::new(
            "CryptoAuditor",
            "You specialise in cryptographic misuse: weak ciphers and modes, weak hashes, and \
             insecure randomness used for security-sensitive values.",
        ),
        DetectorAgent::new(
            "GeneralistReviewer",
            "You are a balanced reviewer who weighs whether a reported flaw is genuinely \
             reachable and exploitable before flagging it.",
        ),
    ];
    all.into_iter().take(n.clamp(1, 4)).collect()
}

/// A per-category detection outcome for one file.
#[derive(Debug, Clone)]
#[allow(dead_code)] // agents_agreeing/max_confidence retained for reporting/debug
pub struct Detection {
    /// How many agents agreed on this category (from field reinforcement).
    pub agents_agreeing: u32,
    /// Highest confidence any agent reported for this category.
    pub max_confidence: f64,
    /// Whether the mesh flags this category (consensus or high confidence).
    pub consensus: bool,
}

/// The mesh verdict for one file: every category that at least one agent raised.
#[derive(Debug, Clone, Default)]
pub struct FileVerdict {
    pub detected: HashMap<OwaspCategory, Detection>,
    /// Number of agents that returned a usable answer.
    #[allow(dead_code)] // retained for diagnostics
    pub agents_responded: u32,
}

impl FileVerdict {
    /// Does the mesh flag this category with consensus?
    pub fn flags(&self, cat: OwaspCategory) -> bool {
        self.detected.get(&cat).map(|d| d.consensus).unwrap_or(false)
    }
}

/// Run the full agent panel over one file and fuse their reports through the
/// SMESH field into a consensus verdict.
pub async fn detect_file(
    content: &str,
    backend: &dyn LlmBackend,
    agents: &[DetectorAgent],
    consensus_threshold: u32,
    high_conf_threshold: f64,
) -> FileVerdict {
    let prompt = format!(
        "Analyse this Java servlet for a genuinely exploitable vulnerability.\n\n```java\n{}\n```",
        truncate(content, 8000)
    );

    // Shared field: same-category reports reinforce each other.
    let mut field = Field::new();
    // Track the strongest per-agent confidence per category (the field's own
    // confidence field is mutated by reinforcement, so we keep this alongside).
    let mut max_conf: HashMap<OwaspCategory, f64> = HashMap::new();
    let mut responded = 0u32;

    for agent in agents {
        let raw = match backend
            .generate_with_system(&prompt, &agent.system_prompt)
            .await
        {
            Ok(r) => r,
            Err(_) => continue, // a failed agent simply doesn't vote
        };
        responded += 1;

        if let Some((cat, conf)) = parse_verdict(&raw) {
            // Emit into the shared field. A fixed origin means identical
            // category payloads collapse onto one signal and reinforce.
            let signal = Signal::builder(SignalType::Alert)
                .payload(cat.code().as_bytes().to_vec())
                .confidence(conf)
                .intensity(conf.max(0.01))
                .origin("owasp-mesh")
                .build();
            field.emit_anonymous(signal);

            let e = max_conf.entry(cat).or_insert(0.0);
            if conf > *e {
                *e = conf;
            }
        }
    }

    // Read consensus back out of the field.
    let mut detected = HashMap::new();
    for signal in field.active_signals() {
        let code = match std::str::from_utf8(&signal.payload) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let cat = match OwaspCategory::from_code(code) {
            Some(c) => c,
            None => continue,
        };
        let agents_agreeing = signal.reinforcement_count + 1; // creator + reinforcers
        let mc = max_conf.get(&cat).copied().unwrap_or(0.0);
        let consensus = agents_agreeing >= consensus_threshold || mc >= high_conf_threshold;
        detected.insert(
            cat,
            Detection {
                agents_agreeing,
                max_confidence: mc,
                consensus,
            },
        );
    }

    FileVerdict {
        detected,
        agents_responded: responded,
    }
}

/// Extract (category, confidence) from an agent's reply. Tolerant of code
/// fences and stray prose around the JSON object.
fn parse_verdict(raw: &str) -> Option<(OwaspCategory, f64)> {
    let json = extract_json_object(raw)?;
    let value: serde_json::Value = serde_json::from_str(&json).ok()?;

    let vulnerable = value
        .get("vulnerable")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !vulnerable {
        return None;
    }
    let code = value.get("category").and_then(|v| v.as_str())?;
    if code.eq_ignore_ascii_case("none") {
        return None;
    }
    let cat = OwaspCategory::from_code(code)?;
    let conf = value
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.6)
        .clamp(0.0, 1.0);
    Some((cat, conf))
}

/// Grab the first balanced `{...}` object from a string.
fn extract_json_object(s: &str) -> Option<String> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        // Respect UTF-8 boundaries.
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}\n// ...[truncated]", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_verdict_json() {
        let (cat, conf) =
            parse_verdict(r#"{"vulnerable": true, "category": "sqli", "confidence": 0.9}"#).unwrap();
        assert_eq!(cat, OwaspCategory::Sqli);
        assert!((conf - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_parse_verdict_fenced_and_prose() {
        let raw = "Here is my analysis:\n```json\n{\"vulnerable\":true,\"category\":\"cmdi\",\"confidence\":0.7}\n```\nDone.";
        let (cat, _) = parse_verdict(raw).unwrap();
        assert_eq!(cat, OwaspCategory::CommandInjection);
    }

    #[test]
    fn test_parse_verdict_not_vulnerable() {
        assert!(parse_verdict(r#"{"vulnerable": false, "category": "none", "confidence": 0.2}"#)
            .is_none());
    }

    #[test]
    fn test_default_agents_count() {
        assert_eq!(default_agents(3).len(), 3);
        assert_eq!(default_agents(9).len(), 4); // clamped
        assert_eq!(default_agents(0).len(), 1); // clamped
    }
}
