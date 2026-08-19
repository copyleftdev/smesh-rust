//! Review-session state: the staged run, the decision ledger, and the
//! transition into a signed revision. Decisions persist to disk after every
//! change — a browser crash must never cost the reviewer their work.

use crate::signer::ReviewerKey;
use crate::RatifyError;
use serde::Serialize;
use smesh_world::{
    CandidateEdge, Changeset, Lane, ReviewDecision, SignedChangeset, StagedRun, WorldError,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub struct Session {
    pub staged: StagedRun,
    pub decisions: BTreeMap<String, ReviewDecision>,
    pub signed: Option<SignedChangeset>,
    pub decisions_path: PathBuf,
    pub revision_path: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct Progress {
    pub total: usize,
    pub decided: usize,
    pub green: usize,
    pub amber: usize,
    pub red: usize,
}

impl Session {
    /// Open a session over a staged run, restoring any persisted decisions.
    pub fn open(
        staged: StagedRun,
        decisions_path: PathBuf,
        revision_path: PathBuf,
    ) -> Result<Self, RatifyError> {
        // Only a missing file means "nothing recorded yet". Treating every
        // error that way turned a permissions problem or a bad disk into a
        // clean slate, and the next ratification would overwrite signed state
        // that was still there and merely unreadable.
        let decisions = match std::fs::read(&decisions_path) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(RatifyError::Io(e)),
        };
        let signed = match std::fs::read(&revision_path) {
            Ok(bytes) => Some(serde_json::from_slice(&bytes)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(RatifyError::Io(e)),
        };
        Ok(Self {
            staged,
            decisions,
            signed,
            decisions_path,
            revision_path,
        })
    }

    pub fn candidate(&self, key: &str) -> Option<&CandidateEdge> {
        self.staged.candidates.iter().find(|c| c.key() == key)
    }

    pub fn lane_of(&self, candidate: &CandidateEdge) -> Lane {
        Lane::assign(candidate)
    }

    pub fn progress(&self) -> Progress {
        let mut p = Progress {
            total: self.staged.candidates.len(),
            decided: 0,
            green: 0,
            amber: 0,
            red: 0,
        };
        for c in &self.staged.candidates {
            match Lane::assign(c) {
                Lane::Green => p.green += 1,
                Lane::Amber => p.amber += 1,
                Lane::Red => p.red += 1,
            }
            if self.decisions.contains_key(&c.key()) {
                p.decided += 1;
            }
        }
        p
    }

    /// Record one decision and persist the ledger.
    pub fn decide(&mut self, key: String, decision: ReviewDecision) -> Result<(), RatifyError> {
        if self.signed.is_some() {
            return Err(RatifyError::AlreadySigned);
        }
        if self.candidate(&key).is_none() {
            return Err(RatifyError::UnknownCandidate(key));
        }
        self.decisions.insert(key, decision);
        self.persist_decisions()
    }

    /// Approve every green-lane candidate that has no decision yet — the
    /// batch gesture; ratification stays total, attention stays tiered.
    pub fn approve_green_lane(&mut self) -> Result<usize, RatifyError> {
        if self.signed.is_some() {
            return Err(RatifyError::AlreadySigned);
        }
        let mut approved = 0;
        let keys: Vec<String> = self
            .staged
            .candidates
            .iter()
            .filter(|c| Lane::assign(c) == Lane::Green)
            .map(CandidateEdge::key)
            .collect();
        for key in keys {
            self.decisions.entry(key).or_insert_with(|| {
                approved += 1;
                ReviewDecision::Approve
            });
        }
        self.persist_decisions()?;
        Ok(approved)
    }

    /// Total-coverage gate, then the kernel path: stage → ratify → sign.
    /// Writes the signed revision to disk and returns it.
    pub fn ratify_and_sign(&mut self, key: &ReviewerKey) -> Result<SignedChangeset, RatifyError> {
        if self.signed.is_some() {
            return Err(RatifyError::AlreadySigned);
        }
        let undecided = self
            .staged
            .candidates
            .iter()
            .filter(|c| !self.decisions.contains_key(&c.key()))
            .count();
        if undecided > 0 {
            return Err(RatifyError::IncompleteCoverage {
                undecided,
                total: self.staged.candidates.len(),
            });
        }

        let record = key.sign_ratification(&self.staged.base_rev, self.decisions.clone());
        let changeset =
            Changeset::stage(self.staged.base_rev.clone(), self.staged.candidates.clone())?;
        let outcome = changeset.ratify(record)?;
        let signed = outcome.changeset.sign(outcome.ratification)?;

        std::fs::write(&self.revision_path, serde_json::to_vec_pretty(&signed)?)?;
        self.signed = Some(signed.clone());
        Ok(signed)
    }

    fn persist_decisions(&self) -> Result<(), RatifyError> {
        std::fs::write(
            &self.decisions_path,
            serde_json::to_vec_pretty(&self.decisions)?,
        )?;
        Ok(())
    }
}

/// Fully-rejected changesets surface the kernel's refusal as a first-class
/// outcome the UI can explain.
pub fn is_empty_changeset_error(err: &RatifyError) -> bool {
    matches!(err, RatifyError::World(WorldError::EmptyChangeset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demo::demo_staged_run;
    use smesh_world::ReviewerId;

    fn temp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("smesh-ratify-{name}-{}", std::process::id()))
    }

    fn session(tag: &str) -> Session {
        let d = temp(&format!("{tag}-decisions.json"));
        let r = temp(&format!("{tag}-revision.json"));
        let _ = std::fs::remove_file(&d);
        let _ = std::fs::remove_file(&r);
        Session::open(demo_staged_run(), d, r).unwrap()
    }

    fn reviewer(tag: &str) -> ReviewerKey {
        ReviewerKey::load_or_generate(&temp(&format!("{tag}.key")), ReviewerId("dj".into()))
            .unwrap()
    }

    #[test]
    fn ratify_refuses_partial_coverage() {
        let mut s = session("partial");
        let key = reviewer("partial");
        let first = s.staged.candidates[0].key();
        s.decide(first, ReviewDecision::Approve).unwrap();
        match s.ratify_and_sign(&key) {
            Err(RatifyError::IncompleteCoverage { undecided, total }) => {
                assert!(undecided > 0 && undecided < total);
            }
            other => panic!("expected coverage error, got {other:?}"),
        }
    }

    #[test]
    fn full_coverage_signs_and_persists_a_revision() {
        let mut s = session("signs");
        let key = reviewer("signs");
        s.approve_green_lane().unwrap();
        let keys: Vec<String> = s.staged.candidates.iter().map(CandidateEdge::key).collect();
        for k in keys {
            if !s.decisions.contains_key(&k) {
                s.decide(k, ReviewDecision::Approve).unwrap();
            }
        }
        let signed = s.ratify_and_sign(&key).unwrap();
        assert_eq!(signed.base_rev, s.staged.base_rev);
        assert_eq!(signed.new_rev.len(), 64);
        assert!(s.revision_path.exists());
        assert!(matches!(
            s.decide("anything".into(), ReviewDecision::Approve),
            Err(RatifyError::AlreadySigned)
        ));
    }

    #[test]
    fn decisions_survive_a_session_restart() {
        let d = temp("restart-decisions.json");
        let r = temp("restart-revision.json");
        let _ = std::fs::remove_file(&d);
        let _ = std::fs::remove_file(&r);
        let mut s = Session::open(demo_staged_run(), d.clone(), r.clone()).unwrap();
        let first = s.staged.candidates[0].key();
        s.decide(first.clone(), ReviewDecision::Defer).unwrap();
        drop(s);
        let restored = Session::open(demo_staged_run(), d, r).unwrap();
        assert_eq!(restored.decisions.get(&first), Some(&ReviewDecision::Defer));
    }

    #[test]
    fn green_lane_batch_approval_skips_decided_and_non_green() {
        let mut s = session("batch");
        let approved = s.approve_green_lane().unwrap();
        let p = s.progress();
        assert_eq!(approved, p.green);
        assert_eq!(p.decided, p.green, "only green candidates were decided");
        assert!(p.red > 0, "demo data must include contested red-lane rows");
    }
}
