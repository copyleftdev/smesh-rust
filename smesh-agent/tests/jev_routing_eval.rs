//! Live evaluation: does Jev route tasks from any domain to agents described
//! only in plain language, through real settlement?
//!
//! Needs a TypeSafe key (TYPESAFE_API_KEY or ~/.creds/typesafe.env) and the
//! network, so it is ignored by default. Run it with:
//!
//! ```text
//! cargo test -p smesh-agent --test jev_routing_eval -- --ignored --nocapture
//! ```
//!
//! Each case is one task; the expected agent is the one a person would pick,
//! or none when no agent fits. Agents are only described, never executed, so
//! no LLM is called. Cost is about 6 agents x 9 tasks Jev calls, a fraction of
//! a cent.

use std::sync::Arc;

use smesh_agent::{
    AgentConfig, AgentCoordinator, AgentRole, CoordinatorConfig, JevAffinity, LlmAgent,
    OpenRouterClient, OpenRouterConfig, TaskDefinition,
};

const AGENTS: &[(&str, &str)] = &[
    (
        "Translator",
        "a translator between English, Portuguese and Spanish who preserves tone and register",
    ),
    (
        "Contracts",
        "a reviewer of commercial contracts who flags risky clauses: liability, termination, \
         auto-renewal and IP assignment",
    ),
    (
        "TripPlanner",
        "a travel planner who builds itineraries, routes, bookings and budgets",
    ),
    (
        "DataAnalyst",
        "a data analyst who finds trends and anomalies in metrics and tabular data and explains them",
    ),
    (
        "IncidentResponder",
        "an incident responder who diagnoses production outages and security incidents from logs and alerts",
    ),
    (
        "Tutor",
        "a tutor who explains school-level maths and science step by step",
    ),
];

/// (task, acceptable winners; empty means nobody should take it)
const CASES: &[(&str, &[&str])] = &[
    (
        "Translate our refund policy into European Portuguese for the Lisbon store.",
        &["Translator"],
    ),
    (
        "Does this supplier agreement renew automatically if we miss the notice date?",
        &["Contracts"],
    ),
    (
        "Three days in Kyoto in April on a budget, mostly temples and food.",
        &["TripPlanner"],
    ),
    (
        "Weekly signups dropped 30% since Tuesday; which channel caused it?",
        &["DataAnalyst"],
    ),
    (
        "Checkout is returning 502s for about a third of requests since the last deploy.",
        &["IncidentResponder"],
    ),
    (
        "My daughter doesn't get why you flip the second fraction when dividing.",
        &["Tutor"],
    ),
    (
        "Is the indemnity clause in this NDA one-sided?",
        &["Contracts"],
    ),
    (
        "We saw 40 failed admin logins from one IP overnight. Is someone attacking us?",
        &["IncidentResponder"],
    ),
    ("Bake me a sourdough loaf.", &[]),
];

#[tokio::test]
#[ignore = "live: needs a TypeSafe key and the network"]
async fn jev_routes_open_domain_tasks_to_described_agents() {
    let jev = Arc::new(JevAffinity::from_env().expect("TYPESAFE_API_KEY not configured"));

    let mut correct = 0;
    for (task, expected) in CASES {
        let mut coord = AgentCoordinator::new(
            CoordinatorConfig {
                n_agents: 0,
                ..CoordinatorConfig::default()
            },
            Box::new(|| Arc::new(OpenRouterClient::new(OpenRouterConfig::default()))),
        )
        .with_affinity_source(jev.clone());
        for (name, capability) in AGENTS {
            coord.add_agent(LlmAgent::new(
                AgentConfig::new(*name, AgentRole::General).with_capability(*capability),
                Arc::new(OpenRouterClient::new(OpenRouterConfig::default())),
            ));
        }

        let id = coord
            .add_task_assessed(TaskDefinition::untyped(*task, 0.5))
            .await;
        coord.settle_traced();
        let winner = coord.winner_for(&id).map(|node| {
            coord
                .agent_roster()
                .into_iter()
                .find(|a| a.id == node)
                .map(|a| a.name)
                .unwrap_or(node)
        });

        let ok = match winner.as_deref() {
            None => expected.is_empty(),
            Some(w) => expected.contains(&w),
        };
        correct += ok as usize;
        println!(
            "{} {:<20} <- {task}",
            if ok { "ok  " } else { "MISS" },
            winner.as_deref().unwrap_or("(rejected)")
        );
    }

    println!("\n{correct}/{} routed as expected", CASES.len());
    // A floor, not a target: this guards against a regression in the question
    // or the settlement wiring, not against one hard case.
    assert!(
        correct + 1 >= CASES.len(),
        "{correct}/{} routed as expected",
        CASES.len()
    );
}
