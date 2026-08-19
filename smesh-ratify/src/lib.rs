//! # SMESH Ratify
//!
//! The human's bench. Serves a staged world-model changeset for review,
//! collects total-coverage decisions, and turns human sign-off into a real
//! Ed25519 signature over the ratification record — which the kernel's
//! type-state machinery then converts into a signed revision.
//!
//! The dashboard never edits the graph; it edits *decisions*. The kernel's
//! `Changeset<Staged> → ratify → Ratified → sign` path remains the only
//! road to a new revision.

pub mod demo;
pub mod signer;
pub mod state;
pub mod web;

#[derive(Debug, thiserror::Error)]
pub enum RatifyError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    World(#[from] smesh_world::WorldError),
    #[error("unknown candidate key {0:?}")]
    UnknownCandidate(String),
    #[error("cannot ratify: {undecided} of {total} candidates still undecided")]
    IncompleteCoverage { undecided: usize, total: usize },
    #[error("changeset already ratified and signed")]
    AlreadySigned,
    #[error("signing key error: {0}")]
    Key(String),
}
