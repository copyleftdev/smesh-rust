//! # SMESH A2A
//!
//! Bridges SMESH and the [A2A (Agent2Agent) protocol](https://a2a-protocol.org)
//! v1.0, in both directions:
//!
//! - [`gateway`]: the mesh as an A2A server. Any A2A client sends it a task;
//!   the mesh's agents settle who takes it through signals; the winner's output
//!   comes back as the task's artifact.
//! - [`remote`]: external A2A agents as mesh participants. An agent's card
//!   decides how strongly it claims each task type; when it wins, the task is
//!   sent to it over A2A.
//!
//! Agent Cards can be signed with a node's Ed25519 key and verified against a
//! set of trusted keys ([`card`]).
//!
//! Built on the A2A project's Rust SDK (`a2a-lf`, `a2a-server-lf`,
//! `a2a-client-lf`).

pub mod card;
pub mod classify;
pub mod gateway;
pub mod remote;

pub use card::{key_id, mesh_card, sign_card, verify_card, CardSignatureError, MeshCardConfig};
pub use classify::{task_from_message, ClassifyError};
pub use gateway::{router, serve, CoordinatorFactory, MeshExecutor};
pub use remote::{
    affinities_from_card, A2aBackend, CardTrust, RemoteAgent, RemoteAgentOptions, RemoteError,
};
