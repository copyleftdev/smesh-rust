//! HTTP surface: one page, four JSON endpoints. The browser edits
//! decisions; only `POST /api/ratify` reaches the kernel's signing path.

use crate::signer::ReviewerKey;
use crate::state::Session;
use crate::RatifyError;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use smesh_world::{CandidateEdge, EdgeKind, Lane, ProvenanceClass, ReviewDecision};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct App {
    pub session: Arc<Mutex<Session>>,
    pub reviewer: Arc<ReviewerKey>,
}

pub fn router(app: App) -> Router {
    Router::new()
        .route("/", get(page))
        .route("/api/state", get(api_state))
        .route("/api/decision", post(api_decision))
        .route("/api/approve-green", post(api_approve_green))
        .route("/api/ratify", post(api_ratify))
        .with_state(app)
}

async fn page() -> Html<&'static str> {
    Html(include_str!("../assets/index.html"))
}

#[derive(Serialize)]
struct VerdictView {
    by: String,
    judgment: String,
    rationale: String,
}

#[derive(Serialize)]
struct CandidateView {
    key: String,
    subject: String,
    kind: String,
    object: String,
    lane: String,
    emitted_by: String,
    corroborations: u32,
    refutations: u32,
    citations: usize,
    verdicts: Vec<VerdictView>,
    evidence: Vec<smesh_world::EvidenceView>,
    decision: Option<serde_json::Value>,
}

fn candidate_view(session: &Session, c: &CandidateEdge) -> CandidateView {
    let key = c.key();
    CandidateView {
        lane: format!("{:?}", Lane::assign(c)),
        subject: c.subject.clone(),
        kind: format!("{:?}", c.kind),
        object: c.object.clone(),
        emitted_by: format!("{:?}", c.emitted_by),
        corroborations: c.corroborations(),
        refutations: c.refutations(),
        citations: match &c.provenance {
            ProvenanceClass::CorpusDerived { citations } => citations.len(),
            ProvenanceClass::HumanAttested { .. } => 0,
        },
        verdicts: c
            .verdicts
            .iter()
            .map(|v| VerdictView {
                by: format!("{:?}", v.by),
                judgment: format!("{:?}", v.judgment),
                rationale: v.rationale.clone(),
            })
            .collect(),
        evidence: session
            .staged
            .evidence
            .get(&key)
            .cloned()
            .unwrap_or_default(),
        decision: session
            .decisions
            .get(&key)
            .map(|d| serde_json::to_value(d).expect("decision serializes")),
        key,
    }
}

async fn api_state(State(app): State<App>) -> Json<serde_json::Value> {
    let session = app.session.lock().expect("session lock");
    let candidates: Vec<CandidateView> = session
        .staged
        .candidates
        .iter()
        .map(|c| candidate_view(&session, c))
        .collect();
    Json(json!({
        "base_rev": session.staged.base_rev,
        "reviewer": app.reviewer.reviewer.0,
        "verifying_key": hex(&app.reviewer.verifying_key().to_bytes()),
        "progress": session.progress(),
        "candidates": candidates,
        "rejected": session.staged.rejected,
        "contradictions_caught": session.staged.contradictions_caught,
        "scorecard": session.staged.scorecard,
        "signed": session.signed.as_ref().map(|s| json!({
            "base_rev": s.base_rev,
            "new_rev": s.new_rev,
            "edges": s.edges.len(),
            "reviewer": s.ratification.reviewer.0,
        })),
    }))
}

#[derive(Deserialize)]
struct DecisionBody {
    key: String,
    action: String,
    #[serde(default)]
    reason: String,
    subject: Option<String>,
    kind: Option<String>,
    object: Option<String>,
}

fn build_decision(
    session: &Session,
    body: &DecisionBody,
) -> Result<ReviewDecision, (StatusCode, String)> {
    match body.action.as_str() {
        "approve" => Ok(ReviewDecision::Approve),
        "defer" => Ok(ReviewDecision::Defer),
        "reject" => Ok(ReviewDecision::Reject {
            reason: if body.reason.trim().is_empty() {
                "rejected by reviewer".to_owned()
            } else {
                body.reason.clone()
            },
        }),
        "edit" => {
            let original = session
                .candidate(&body.key)
                .ok_or((StatusCode::NOT_FOUND, "unknown candidate".to_owned()))?;
            let kind: EdgeKind = match &body.kind {
                Some(k) => serde_json::from_value(json!(k))
                    .map_err(|_| (StatusCode::BAD_REQUEST, format!("unknown kind {k:?}")))?,
                None => original.kind,
            };
            let mut amended = original.clone();
            amended.subject = body
                .subject
                .clone()
                .unwrap_or_else(|| original.subject.clone());
            amended.kind = kind;
            amended.object = body
                .object
                .clone()
                .unwrap_or_else(|| original.object.clone());
            Ok(ReviewDecision::Edit { amended })
        }
        other => Err((StatusCode::BAD_REQUEST, format!("unknown action {other:?}"))),
    }
}

async fn api_decision(
    State(app): State<App>,
    Json(body): Json<DecisionBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut session = app.session.lock().expect("session lock");
    let decision = build_decision(&session, &body)?;
    session
        .decide(body.key.clone(), decision)
        .map_err(ratify_status)?;
    Ok(Json(json!({ "progress": session.progress() })))
}

async fn api_approve_green(
    State(app): State<App>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut session = app.session.lock().expect("session lock");
    let approved = session.approve_green_lane().map_err(ratify_status)?;
    Ok(Json(
        json!({ "approved": approved, "progress": session.progress() }),
    ))
}

async fn api_ratify(
    State(app): State<App>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut session = app.session.lock().expect("session lock");
    let signed = session
        .ratify_and_sign(&app.reviewer)
        .map_err(ratify_status)?;
    Ok(Json(json!({
        "base_rev": signed.base_rev,
        "new_rev": signed.new_rev,
        "edges": signed.edges.len(),
        "reviewer": signed.ratification.reviewer.0,
    })))
}

fn ratify_status(err: RatifyError) -> (StatusCode, String) {
    let code = match &err {
        RatifyError::UnknownCandidate(_) => StatusCode::NOT_FOUND,
        RatifyError::IncompleteCoverage { .. } | RatifyError::AlreadySigned => StatusCode::CONFLICT,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (code, err.to_string())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demo::demo_staged_run;
    use smesh_world::ReviewerId;

    fn app(tag: &str) -> App {
        let base =
            std::env::temp_dir().join(format!("smesh-ratify-web-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_file(base.with_extension("decisions.json"));
        let _ = std::fs::remove_file(base.with_extension("revision.json"));
        App {
            session: Arc::new(Mutex::new(
                Session::open(
                    demo_staged_run(),
                    base.with_extension("decisions.json"),
                    base.with_extension("revision.json"),
                )
                .unwrap(),
            )),
            reviewer: Arc::new(
                ReviewerKey::load_or_generate(&base.with_extension("key"), ReviewerId("dj".into()))
                    .unwrap(),
            ),
        }
    }

    #[tokio::test]
    async fn state_endpoint_reports_lanes_and_progress() {
        let state = api_state(State(app("state"))).await;
        let v = state.0;
        assert_eq!(v["base_rev"], "meridian-rev0");
        assert_eq!(v["progress"]["total"], 8);
        assert_eq!(v["candidates"].as_array().unwrap().len(), 8);
        assert!(v["signed"].is_null());
    }

    #[tokio::test]
    async fn edit_decision_amends_and_ratify_gates_on_coverage() {
        let app = app("edit");
        let key = {
            let s = app.session.lock().unwrap();
            s.staged.candidates[0].key()
        };
        let _ = api_decision(
            State(app.clone()),
            Json(DecisionBody {
                key: key.clone(),
                action: "edit".into(),
                reason: String::new(),
                subject: None,
                kind: None,
                object: Some("a formal demand for coverage".into()),
            }),
        )
        .await
        .unwrap();
        {
            let s = app.session.lock().unwrap();
            match s.decisions.get(&key).unwrap() {
                ReviewDecision::Edit { amended } => {
                    assert_eq!(amended.object, "a formal demand for coverage");
                }
                other => panic!("expected edit, got {other:?}"),
            }
        }
        let err = api_ratify(State(app)).await.unwrap_err();
        assert_eq!(err.0, StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn full_flow_signs_through_the_kernel() {
        let app = app("flow");
        let _ = api_approve_green(State(app.clone())).await.unwrap();
        let keys: Vec<String> = {
            let s = app.session.lock().unwrap();
            s.staged
                .candidates
                .iter()
                .map(CandidateEdge::key)
                .filter(|k| !s.decisions.contains_key(k))
                .collect()
        };
        for key in keys {
            let _ = api_decision(
                State(app.clone()),
                Json(DecisionBody {
                    key,
                    action: "approve".into(),
                    reason: String::new(),
                    subject: None,
                    kind: None,
                    object: None,
                }),
            )
            .await
            .unwrap();
        }
        let receipt = api_ratify(State(app.clone())).await.unwrap().0;
        assert_eq!(receipt["edges"], 8);
        assert_eq!(receipt["new_rev"].as_str().unwrap().len(), 64);
        let state = api_state(State(app)).await.0;
        assert_eq!(state["signed"]["new_rev"], receipt["new_rev"]);
    }
}
