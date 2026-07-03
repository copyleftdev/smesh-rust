//! OWASP Benchmark scoring: confusion matrices and the Youden index.

use std::collections::BTreeMap;

use super::corpus::{OwaspCategory, TestCase};
use super::detector::FileVerdict;

/// A binary confusion matrix for one category (or the overall total).
#[derive(Debug, Clone, Copy, Default)]
pub struct Confusion {
    pub tp: u32,
    pub fp: u32,
    pub tn: u32,
    pub fn_: u32,
}

impl Confusion {
    /// True positive rate (recall / sensitivity).
    pub fn tpr(&self) -> f64 {
        let denom = self.tp + self.fn_;
        if denom == 0 {
            0.0
        } else {
            self.tp as f64 / denom as f64
        }
    }

    /// False positive rate (fall-out).
    pub fn fpr(&self) -> f64 {
        let denom = self.fp + self.tn;
        if denom == 0 {
            0.0
        } else {
            self.fp as f64 / denom as f64
        }
    }

    /// Youden index (TPR - FPR): the OWASP Benchmark's headline score.
    /// 1.0 is perfect, 0.0 is no better than random guessing, negative is worse.
    pub fn youden(&self) -> f64 {
        self.tpr() - self.fpr()
    }

    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 {
            0.0
        } else {
            self.tp as f64 / denom as f64
        }
    }

    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.tpr();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// The full scorecard: per-category confusion + the overall total.
#[derive(Debug, Clone)]
pub struct Scorecard {
    pub per_category: BTreeMap<OwaspCategory, Confusion>,
    pub overall: Confusion,
    pub cases_scored: usize,
    pub model: String,
    pub agents: usize,
    pub elapsed_secs: f64,
}

/// Score each test case against its mesh verdict.
///
/// Each OWASP Benchmark case targets exactly one category; a case is scored
/// only on whether the mesh flagged *that* category, matching the official
/// methodology (detections of other categories on a case are not counted).
pub fn score(
    results: &[(TestCase, FileVerdict)],
    model: &str,
    agents: usize,
    elapsed_secs: f64,
) -> Scorecard {
    let mut per_category: BTreeMap<OwaspCategory, Confusion> = BTreeMap::new();
    let mut overall = Confusion::default();

    for (case, verdict) in results {
        let c = per_category.entry(case.category).or_default();
        let flagged = verdict.flags(case.category);
        match (case.is_real, flagged) {
            (true, true) => {
                c.tp += 1;
                overall.tp += 1;
            }
            (true, false) => {
                c.fn_ += 1;
                overall.fn_ += 1;
            }
            (false, true) => {
                c.fp += 1;
                overall.fp += 1;
            }
            (false, false) => {
                c.tn += 1;
                overall.tn += 1;
            }
        }
    }

    Scorecard {
        per_category,
        overall,
        cases_scored: results.len(),
        model: model.to_string(),
        agents,
        elapsed_secs,
    }
}

/// Reconstruct a scorecard from a previously written results JSON, so the
/// report can be re-rendered without re-running the mesh.
pub fn from_json(json: &str) -> Option<Scorecard> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;

    let confusion = |obj: &serde_json::Value| Confusion {
        tp: obj.get("tp").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        fp: obj.get("fp").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        tn: obj.get("tn").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        fn_: obj.get("fn").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
    };

    let overall = confusion(v.get("overall")?);

    let mut per_category = BTreeMap::new();
    for cat_obj in v.get("categories")?.as_array()? {
        let code = cat_obj.get("code")?.as_str()?;
        if let Some(cat) = OwaspCategory::from_code(code) {
            per_category.insert(cat, confusion(cat_obj));
        }
    }

    Some(Scorecard {
        per_category,
        overall,
        cases_scored: v.get("cases_scored").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
        model: v
            .get("model")
            .and_then(|x| x.as_str())
            .unwrap_or("?")
            .to_string(),
        agents: v.get("agents_per_case").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
        elapsed_secs: v.get("elapsed_secs").and_then(|x| x.as_f64()).unwrap_or(0.0),
    })
}
