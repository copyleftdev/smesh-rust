use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::*;
use crate::backend::{
    GenerateRequest, GenerateResponse, GenerateResponseV2, LlmProvider, StopReason,
};

/// A tool that echoes its input and counts its calls.
struct Echo {
    name: &'static str,
    effect: Effect,
    calls: Arc<Mutex<u32>>,
}

impl Echo {
    fn new(name: &'static str, effect: Effect) -> Arc<Self> {
        Arc::new(Self {
            name,
            effect,
            calls: Arc::new(Mutex::new(0)),
        })
    }
}

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "Echo the input back. Useful for tests."
    }
    fn input_schema(&self) -> Value {
        json!({ "type": "object" })
    }
    fn effect(&self) -> Effect {
        self.effect
    }
    async fn call(&self, input: Value) -> Result<ToolOutput, ToolError> {
        *self.calls.lock().unwrap() += 1;
        if input.get("fail").is_some() {
            return Err(ToolError::Failed("asked to fail".into()));
        }
        Ok(ToolOutput::text(input.to_string()))
    }
}

/// A model that replays scripted responses and records what it was sent.
struct Scripted {
    replies: Mutex<VecDeque<GenerateResponseV2>>,
    seen: Mutex<Vec<GenerateRequestV2>>,
    tools: bool,
}

impl Scripted {
    fn new(replies: Vec<Vec<ContentBlock>>) -> Self {
        Self {
            replies: Mutex::new(
                replies
                    .into_iter()
                    .map(|content| GenerateResponseV2 {
                        stop_reason: if content.iter().any(ContentBlock::is_tool_use) {
                            StopReason::ToolUse
                        } else {
                            StopReason::EndTurn
                        },
                        content,
                        model: "scripted".into(),
                        id: String::new(),
                        input_tokens: 0,
                        output_tokens: 0,
                        latency: std::time::Duration::ZERO,
                    })
                    .collect(),
            ),
            seen: Mutex::new(Vec::new()),
            tools: true,
        }
    }
}

#[async_trait]
impl LlmBackend for Scripted {
    fn provider(&self) -> LlmProvider {
        LlmProvider::OpenRouter
    }
    fn model(&self) -> &str {
        "scripted"
    }
    async fn is_available(&self) -> bool {
        true
    }
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec![])
    }
    async fn generate(&self, _: GenerateRequest) -> Result<GenerateResponse, LlmError> {
        Err(LlmError::UnsupportedFeature("scripted is V2 only".into()))
    }
    async fn generate_v2(
        &self,
        request: GenerateRequestV2,
    ) -> Result<GenerateResponseV2, LlmError> {
        self.seen.lock().unwrap().push(request);
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LlmError::GenerationFailed("script ran out".into()))
    }
    fn supports_tools(&self) -> bool {
        self.tools
    }
}

fn use_tool(id: &str, name: &str, input: Value) -> ContentBlock {
    ContentBlock::ToolUse {
        id: id.into(),
        name: name.into(),
        input,
    }
}

fn text(t: &str) -> ContentBlock {
    ContentBlock::Text { text: t.into() }
}

#[test]
fn effects_are_ordered_and_parse_from_config_words() {
    assert!(Effect::Read < Effect::Write && Effect::Write < Effect::Execute);
    assert_eq!("read_only".parse::<Effect>().unwrap(), Effect::Read);
    assert_eq!("Execute".parse::<Effect>().unwrap(), Effect::Execute);
    assert!("admin".parse::<Effect>().is_err());
}

#[test]
fn a_toolbelt_offers_only_what_its_clearance_allows() {
    let belt = Toolbelt::new(Effect::Read)
        .with_tool(Echo::new("look", Effect::Read))
        .unwrap()
        .with_tool(Echo::new("change", Effect::Write))
        .unwrap();
    let offered: Vec<String> = belt.definitions().into_iter().map(|d| d.name).collect();
    assert_eq!(offered, vec!["look"]);
    assert!(belt.describe().contains("look (Echo the input back.)"));
    assert!(!belt.describe().contains("change"));
    assert!(Toolbelt::new(Effect::Read).describe().is_empty());
}

#[test]
fn names_must_be_unique_and_acceptable_to_every_model_api() {
    let mut belt = Toolbelt::new(Effect::Read);
    belt.add(Echo::new("look", Effect::Read)).unwrap();
    assert!(matches!(
        belt.add(Echo::new("look", Effect::Read)),
        Err(ToolbeltError::Duplicate(_))
    ));
    assert!(matches!(
        belt.add(Echo::new("has.dot", Effect::Read)),
        Err(ToolbeltError::InvalidName(_))
    ));
}

#[tokio::test]
async fn calls_above_clearance_or_unknown_are_refused_and_journaled() {
    let change = Echo::new("change", Effect::Write);
    let belt = Toolbelt::new(Effect::Read)
        .for_agent("Reader")
        .with_tool(change.clone())
        .unwrap();
    let refused = belt.call("change", json!({})).await;
    assert_eq!(refused.outcome, CallOutcome::Refused);
    assert!(refused.output.contains("needs write clearance"));
    assert_eq!(
        *change.calls.lock().unwrap(),
        0,
        "a refused tool never runs"
    );

    let unknown = belt.call("rm_rf", json!({})).await;
    assert_eq!(unknown.outcome, CallOutcome::Refused);
    assert_eq!(unknown.effect, None);

    let journal = belt.journal();
    assert_eq!(journal.len(), 2);
    assert_eq!(journal[0].agent.as_deref(), Some("Reader"));
}

#[tokio::test]
async fn failures_and_timeouts_are_errors_not_panics() {
    struct Slow;
    #[async_trait]
    impl Tool for Slow {
        fn name(&self) -> &str {
            "slow"
        }
        fn description(&self) -> &str {
            "Sleeps."
        }
        fn input_schema(&self) -> Value {
            json!({ "type": "object" })
        }
        fn effect(&self) -> Effect {
            Effect::Read
        }
        async fn call(&self, _: Value) -> Result<ToolOutput, ToolError> {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            Ok(ToolOutput::text("late"))
        }
    }
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let belt = Toolbelt::new(Effect::Read)
        .with_tool(Echo::new("echo", Effect::Read))
        .unwrap()
        .with_tool(Arc::new(Slow))
        .unwrap()
        .with_timeout(std::time::Duration::from_millis(50))
        .with_observer(Arc::new(move |c: &ToolCall| {
            sink.lock().unwrap().push(c.tool.clone())
        }));
    let failed = belt.call("echo", json!({ "fail": true })).await;
    assert_eq!(failed.outcome, CallOutcome::Error);
    assert_eq!(failed.output, "asked to fail");
    let slow = belt.call("slow", json!({})).await;
    assert_eq!(slow.outcome, CallOutcome::Error);
    assert!(slow.output.contains("timed out"));
    assert_eq!(*seen.lock().unwrap(), vec!["echo", "slow"]);
}

#[tokio::test]
async fn the_loop_runs_tools_and_feeds_results_back() {
    let look = Echo::new("look", Effect::Read);
    let belt = Toolbelt::new(Effect::Read)
        .with_tool(look.clone())
        .unwrap()
        .with_tool(Echo::new("change", Effect::Write))
        .unwrap();
    let model = Scripted::new(vec![
        vec![
            text("Checking."),
            use_tool("1", "look", json!({ "q": 1 })),
            use_tool("2", "change", json!({})),
        ],
        vec![text("The answer is 1.")],
    ]);
    let run = run_with_tools(&model, "You are a tester.", "What is q?", &belt, 4)
        .await
        .unwrap();
    assert_eq!(run.text, "The answer is 1.");
    assert_eq!(run.turns, 2);
    assert_eq!(
        run.calls.iter().map(|c| c.outcome).collect::<Vec<_>>(),
        vec![CallOutcome::Ok, CallOutcome::Refused]
    );
    assert_eq!(*look.calls.lock().unwrap(), 1);

    let seen = model.seen.lock().unwrap();
    // Only the usable tool is offered, and the system prompt explains tools.
    assert_eq!(seen[0].tools.len(), 1);
    assert!(seen[0].system.as_deref().unwrap().contains(TOOL_GUIDANCE));
    // The second turn carries the assistant's calls and both results, in order.
    let last = seen[1].messages.last().unwrap();
    assert_eq!(last.role, MessageRole::User);
    match (&last.content[0], &last.content[1]) {
        (
            ContentBlock::ToolResult {
                tool_use_id: a,
                is_error: false,
                ..
            },
            ContentBlock::ToolResult {
                tool_use_id: b,
                is_error: true,
                ..
            },
        ) => assert_eq!((a.as_str(), b.as_str()), ("1", "2")),
        other => panic!("unexpected results {other:?}"),
    }
}

#[tokio::test]
async fn when_turns_run_out_the_model_answers_without_tools() {
    let belt = Toolbelt::new(Effect::Read)
        .with_tool(Echo::new("look", Effect::Read))
        .unwrap();
    let model = Scripted::new(vec![
        vec![use_tool("1", "look", json!({}))],
        vec![use_tool("2", "look", json!({}))],
        vec![text("Best guess.")],
    ]);
    let run = run_with_tools(&model, "s", "p", &belt, 2).await.unwrap();
    assert_eq!(run.text, "Best guess.");
    assert_eq!(run.calls.len(), 2);
    let seen = model.seen.lock().unwrap();
    assert!(matches!(seen[2].tool_choice, ToolChoice::None));
}

#[tokio::test]
async fn a_failure_after_tools_ran_still_reports_them() {
    let write = Echo::new("write", Effect::Write);
    let belt = Toolbelt::new(Effect::Write)
        .with_tool(write.clone())
        .unwrap();
    // The model calls a write tool, then the backend fails on the next turn.
    let model = Scripted::new(vec![vec![use_tool("1", "write", json!({ "x": 1 }))]]);
    let err = run_with_tools(&model, "s", "p", &belt, 4)
        .await
        .unwrap_err();
    assert_eq!(*write.calls.lock().unwrap(), 1, "the write happened");
    assert_eq!(err.calls.len(), 1, "and the error says so");
    assert_eq!(err.calls[0].tool, "write");
    assert!(err.source.to_string().contains("script ran out"));
}

#[tokio::test]
async fn an_agent_whose_backend_cannot_call_tools_neither_advertises_nor_uses_them() {
    use crate::agent::{AgentConfig, AgentRole, LlmAgent};
    let belt = Toolbelt::new(Effect::Write)
        .with_tool(Echo::new("write", Effect::Write))
        .unwrap();
    let mut plain = Scripted::new(vec![]);
    plain.tools = false;
    let agent = LlmAgent::new(
        AgentConfig::new("A", AgentRole::General).with_capability("an assistant"),
        Arc::new(plain),
    )
    .with_toolbelt(belt.clone());
    assert_eq!(agent.profile(), "an assistant", "no tools in its profile");
    assert_eq!(
        agent.privilege(),
        Some(Effect::Read),
        "nor in its privilege"
    );

    let capable = LlmAgent::new(
        AgentConfig::new("B", AgentRole::General).with_capability("an assistant"),
        Arc::new(Scripted::new(vec![])),
    )
    .with_toolbelt(belt);
    assert!(capable.profile().contains("write"));
    assert_eq!(capable.privilege(), Some(Effect::Write));
}

#[cfg(unix)]
#[tokio::test]
async fn stdio_servers_get_a_basic_environment_not_our_secrets() {
    // A variable only the parent has, standing in for an API key.
    std::env::set_var("SMESH_TEST_PARENT_SECRET", "sk-should-not-leak");
    let env = HashMap::from([("GIVEN".to_string(), "on purpose".to_string())]);
    let args = vec!["-c".to_string(), "env".to_string()];

    let isolated = mcp::stdio_command("sh", &args, &env, None, false)
        .output()
        .await
        .unwrap();
    let seen = String::from_utf8_lossy(&isolated.stdout);
    assert!(!seen.contains("sk-should-not-leak"), "{seen}");
    assert!(seen.contains("GIVEN=on purpose"));
    assert!(seen.lines().any(|l| l.starts_with("PATH=")), "PATH is kept");

    let inherited = mcp::stdio_command("sh", &args, &env, None, true)
        .output()
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&inherited.stdout).contains("sk-should-not-leak"));
}

#[test]
fn truncation_respects_character_boundaries() {
    assert_eq!(truncate("short", 10), "short");
    let cut = truncate("ééééé", 3);
    assert!(cut.starts_with('é') && cut.contains("more bytes"), "{cut}");
}

/* --------------------------------------------- a real MCP server, in process */

mod mcp_server {
    use rmcp::model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool, ToolAnnotations,
    };
    use rmcp::service::{RequestContext, RoleServer};
    use rmcp::{ErrorData, ServerHandler};
    use serde_json::json;

    /// Two tools: `read_note` (annotated read-only) and `delete_note` (not annotated).
    #[derive(Clone)]
    pub struct Notes;

    fn schema() -> std::sync::Arc<rmcp::model::JsonObject> {
        let v = json!({ "type": "object", "properties": { "id": { "type": "string" } } });
        std::sync::Arc::new(v.as_object().unwrap().clone())
    }

    impl ServerHandler for Notes {
        fn get_info(&self) -> ServerConfig {
            ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
        }

        async fn list_tools(
            &self,
            _: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            let mut annotations = ToolAnnotations::new();
            annotations.read_only_hint = Some(true);
            Ok(ListToolsResult {
                tools: vec![
                    Tool::new(
                        "read_note",
                        "Read a note by id. Returns its text.",
                        schema(),
                    )
                    .with_annotations(annotations),
                    Tool::new("delete_note", "Delete a note by id.", schema()),
                ],
                ..Default::default()
            })
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            let id = request
                .arguments
                .as_ref()
                .and_then(|a| a.get("id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let result = match (request.name.as_ref(), id.as_str()) {
                ("read_note", "n1") => {
                    CallToolResult::success(vec![ContentBlock::text("buy oat milk")])
                }
                ("read_note", other) => {
                    CallToolResult::error(vec![ContentBlock::text(format!("no note {other}"))])
                }
                _ => CallToolResult::success(vec![ContentBlock::text("deleted")]),
            };
            Ok(CallToolResponse::Complete(result))
        }
    }
}

#[tokio::test]
async fn mcp_tools_join_a_toolbelt_with_effects_from_their_annotations() {
    use rmcp::ServiceExt;
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        let running = mcp_server::Notes.serve(server_io).await.unwrap();
        let _ = running.waiting().await;
    });
    let server = mcp::McpServer::connect_with("notes", client_io)
        .await
        .unwrap();
    assert_eq!(server.tool_names(), vec!["read_note", "delete_note"]);

    let tools = server.tools(&HashMap::new());
    assert_eq!(tools[0].name(), "notes__read_note");
    assert_eq!(tools[0].effect(), Effect::Read, "readOnlyHint: true");
    assert_eq!(
        tools[1].effect(),
        Effect::Write,
        "unannotated may change things"
    );

    let mut belt = Toolbelt::new(Effect::Read);
    for t in tools {
        belt.add(t).unwrap();
    }
    assert_eq!(belt.definitions().len(), 1);
    assert_eq!(
        belt.definitions()[0].input_schema["properties"]["id"]["type"],
        "string"
    );

    let ok = belt.call("notes__read_note", json!({ "id": "n1" })).await;
    assert_eq!(
        (ok.outcome, ok.output.as_str()),
        (CallOutcome::Ok, "buy oat milk")
    );
    let missing = belt.call("notes__read_note", json!({ "id": "n9" })).await;
    assert_eq!(
        missing.outcome,
        CallOutcome::Error,
        "isError reaches the journal"
    );
    let delete = belt.call("notes__delete_note", json!({ "id": "n1" })).await;
    assert_eq!(delete.outcome, CallOutcome::Refused);
    let bad = belt.call("notes__read_note", json!("n1")).await;
    assert!(
        bad.output.contains("expected a JSON object"),
        "{}",
        bad.output
    );

    // An override can correct a tool's effect.
    let overrides = HashMap::from([("delete_note".to_string(), Effect::Execute)]);
    assert_eq!(server.tools(&overrides)[1].effect(), Effect::Execute);
}

#[test]
fn local_names_are_safe_for_model_apis() {
    assert_eq!(mcp::local_name("files", "read_file"), "files__read_file");
    assert_eq!(
        mcp::local_name("git hub", "issues/list"),
        "git_hub__issues_list"
    );
    let long = mcp::local_name("s", &"x".repeat(100));
    assert_eq!(long.len(), 64);
    assert!(is_valid_tool_name(&long));
}
