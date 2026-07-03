//! OWASP Benchmark corpus: categories, ground-truth loading, and sampling.

use std::path::{Path, PathBuf};

/// The 11 vulnerability categories used by the OWASP Benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OwaspCategory {
    PathTraversal,
    Sqli,
    Xss,
    CommandInjection,
    Crypto,
    Hash,
    LdapInjection,
    SecureCookie,
    TrustBoundary,
    WeakRandom,
    XPathInjection,
}

impl OwaspCategory {
    /// Every category, in a stable order.
    pub fn all() -> [OwaspCategory; 11] {
        use OwaspCategory::*;
        [
            PathTraversal,
            Sqli,
            Xss,
            CommandInjection,
            Crypto,
            Hash,
            LdapInjection,
            SecureCookie,
            TrustBoundary,
            WeakRandom,
            XPathInjection,
        ]
    }

    /// The short code used in the benchmark's expectedresults CSV.
    pub fn code(&self) -> &'static str {
        match self {
            OwaspCategory::PathTraversal => "pathtraver",
            OwaspCategory::Sqli => "sqli",
            OwaspCategory::Xss => "xss",
            OwaspCategory::CommandInjection => "cmdi",
            OwaspCategory::Crypto => "crypto",
            OwaspCategory::Hash => "hash",
            OwaspCategory::LdapInjection => "ldapi",
            OwaspCategory::SecureCookie => "securecookie",
            OwaspCategory::TrustBoundary => "trustbound",
            OwaspCategory::WeakRandom => "weakrand",
            OwaspCategory::XPathInjection => "xpathi",
        }
    }

    /// Parse a category from its CSV/agent code (case-insensitive, tolerant).
    pub fn from_code(s: &str) -> Option<OwaspCategory> {
        let s = s.trim().to_ascii_lowercase();
        let c = match s.as_str() {
            "pathtraver" | "path" | "path_traversal" | "pathtraversal" => {
                OwaspCategory::PathTraversal
            }
            "sqli" | "sql" | "sql_injection" | "sqlinjection" => OwaspCategory::Sqli,
            "xss" | "cross_site_scripting" => OwaspCategory::Xss,
            "cmdi" | "cmd" | "command_injection" | "commandinjection" => {
                OwaspCategory::CommandInjection
            }
            "crypto" | "cryptography" | "weak_crypto" => OwaspCategory::Crypto,
            "hash" | "weak_hash" => OwaspCategory::Hash,
            "ldapi" | "ldap" | "ldap_injection" => OwaspCategory::LdapInjection,
            "securecookie" | "secure_cookie" | "cookie" | "insecure_cookie" => {
                OwaspCategory::SecureCookie
            }
            "trustbound" | "trust_boundary" | "trustboundary" => OwaspCategory::TrustBoundary,
            "weakrand" | "weak_random" | "weakrandom" | "weak_randomness" => {
                OwaspCategory::WeakRandom
            }
            "xpathi" | "xpath" | "xpath_injection" => OwaspCategory::XPathInjection,
            _ => return None,
        };
        Some(c)
    }

    /// The CWE number this category maps to.
    pub fn cwe(&self) -> u32 {
        match self {
            OwaspCategory::PathTraversal => 22,
            OwaspCategory::Sqli => 89,
            OwaspCategory::Xss => 79,
            OwaspCategory::CommandInjection => 78,
            OwaspCategory::Crypto => 327,
            OwaspCategory::Hash => 328,
            OwaspCategory::LdapInjection => 90,
            OwaspCategory::SecureCookie => 614,
            OwaspCategory::TrustBoundary => 501,
            OwaspCategory::WeakRandom => 330,
            OwaspCategory::XPathInjection => 643,
        }
    }

    /// Short human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            OwaspCategory::PathTraversal => "Path Traversal",
            OwaspCategory::Sqli => "SQL Injection",
            OwaspCategory::Xss => "Cross-Site Scripting",
            OwaspCategory::CommandInjection => "Command Injection",
            OwaspCategory::Crypto => "Weak Encryption",
            OwaspCategory::Hash => "Weak Hash",
            OwaspCategory::LdapInjection => "LDAP Injection",
            OwaspCategory::SecureCookie => "Insecure Cookie",
            OwaspCategory::TrustBoundary => "Trust Boundary",
            OwaspCategory::WeakRandom => "Weak Randomness",
            OwaspCategory::XPathInjection => "XPath Injection",
        }
    }
}

/// A single benchmark test case with its ground truth.
#[derive(Debug, Clone)]
#[allow(dead_code)] // `name`/`cwe` are retained as ground-truth metadata (JSON/debug)
pub struct TestCase {
    /// e.g. "BenchmarkTest00001"
    pub name: String,
    /// The category this test case targets.
    pub category: OwaspCategory,
    /// Whether this case contains a real, exploitable vulnerability.
    pub is_real: bool,
    /// CWE number from the ground truth.
    pub cwe: u32,
    /// Absolute path to the Java source file.
    pub file_path: PathBuf,
}

/// Errors from loading the corpus.
#[derive(Debug)]
pub enum CorpusError {
    Io(std::io::Error),
    MissingCsv(PathBuf),
    Empty,
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorpusError::Io(e) => write!(f, "io error: {e}"),
            CorpusError::MissingCsv(p) => write!(
                f,
                "expectedresults CSV not found under {} (is this a BenchmarkJava checkout?)",
                p.display()
            ),
            CorpusError::Empty => write!(f, "no test cases parsed from the corpus"),
        }
    }
}

impl std::error::Error for CorpusError {}

/// Locate the testcode directory and expectedresults CSV under a BenchmarkJava
/// checkout, then parse every ground-truth row into a [`TestCase`].
pub fn load(benchmark_root: &Path) -> Result<Vec<TestCase>, CorpusError> {
    // The CSV lives at the repo root; testcode under src/main/java/...
    let csv_path = find_expected_csv(benchmark_root).ok_or_else(|| {
        CorpusError::MissingCsv(benchmark_root.to_path_buf())
    })?;
    let testcode_dir = benchmark_root
        .join("src/main/java/org/owasp/benchmark/testcode");

    let contents = std::fs::read_to_string(&csv_path).map_err(CorpusError::Io)?;
    let mut cases = Vec::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 4 {
            continue;
        }
        let name = fields[0].trim().to_string();
        let category = match OwaspCategory::from_code(fields[1]) {
            Some(c) => c,
            None => continue,
        };
        let is_real = fields[2].trim().eq_ignore_ascii_case("true");
        let cwe = fields[3].trim().parse::<u32>().unwrap_or(category.cwe());
        let file_path = testcode_dir.join(format!("{name}.java"));

        cases.push(TestCase {
            name,
            category,
            is_real,
            cwe,
            file_path,
        });
    }

    if cases.is_empty() {
        return Err(CorpusError::Empty);
    }
    Ok(cases)
}

fn find_expected_csv(root: &Path) -> Option<PathBuf> {
    // Prefer a versioned name; fall back to any expectedresults*.csv at the root.
    let direct = root.join("expectedresults-1.2.csv");
    if direct.exists() {
        return Some(direct);
    }
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("expectedresults") && name.ends_with(".csv") {
                return Some(p);
            }
        }
    }
    None
}

/// Deterministically pick up to `per_category` cases from each category.
///
/// Cases are spread evenly across each category's rows (which interleave real
/// and non-real cases), so the sample keeps a realistic true/false mix without
/// any randomness — the same corpus always yields the same sample.
pub fn sample_balanced(cases: &[TestCase], per_category: usize) -> Vec<TestCase> {
    let mut out = Vec::new();
    for cat in OwaspCategory::all() {
        let in_cat: Vec<&TestCase> = cases.iter().filter(|c| c.category == cat).collect();
        if in_cat.is_empty() {
            continue;
        }
        let take = per_category.min(in_cat.len());
        if take == in_cat.len() {
            out.extend(in_cat.into_iter().cloned());
            continue;
        }
        // Even stride across the category's rows.
        let step = in_cat.len() as f64 / take as f64;
        for i in 0..take {
            let idx = (i as f64 * step) as usize;
            out.push(in_cat[idx.min(in_cat.len() - 1)].clone());
        }
    }
    out
}
