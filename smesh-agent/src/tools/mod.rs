//! Tools that agents in a mesh can call.
//!
//! A [`Tool`] is anything an agent can do beyond writing text: read a file,
//! query a database, open a pull request. Tools come from two places:
//!
//! - **MCP servers** ([`mcp`]), declared in a config file ([`config`]). Any
//!   server that speaks the Model Context Protocol works without code.
//! - **Rust**, by implementing [`Tool`] (see [`builtin`] for an example).
//!
//! Each agent carries its own [`Toolbelt`]: an allowlist of tools and a
//! *clearance*, the most an agent may do ([`Effect::Read`] <
//! [`Effect::Write`] < [`Effect::Execute`]). A tool above the agent's clearance
//! is neither offered to the model nor run if the model asks for it anyway.
//! Every call, refused ones included, is journaled as a [`ToolCall`].
//!
//! A toolbelt also shapes routing: [`Toolbelt::describe`] is added to what an
//! agent says it does, so when agents judge whether they suit a task, the one
//! holding the right tools claims it.

pub mod builtin;
pub mod config;
pub mod mcp;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tracing::{debug, info};

use crate::backend::{
    ContentBlock, GenerateRequestV2, LlmBackend, LlmError, Message, MessageRole, ToolChoice,
    ToolDefinition,
};

/// The most a tool can do to the world outside the agent.
///
/// Ordered: a clearance of `Write` allows `Read` and `Write` tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// Observes only: reads, lists, searches, fetches.
    Read,
    /// Changes state: writes files, creates issues, sends messages.
    Write,
    /// Runs arbitrary code or commands.
    Execute,
}

impl Effect {
    /// The effect's name, as written in config files.
    pub fn as_str(self) -> &'static str {
        match self {
            Effect::Read => "read",
            Effect::Write => "write",
            Effect::Execute => "execute",
        }
    }
}

impl std::fmt::Display for Effect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Effect {
    /// Accepts the same words as [`FromStr`](std::str::FromStr), so a config
    /// file and the command line agree.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

impl std::str::FromStr for Effect {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "read" | "read_only" | "readonly" => Ok(Effect::Read),
            "write" => Ok(Effect::Write),
            "execute" | "exec" => Ok(Effect::Execute),
            other => Err(format!(
                "unknown effect {other:?} (expected read, write or execute)"
            )),
        }
    }
}

/// Why a tool call failed.
#[derive(Debug, Error)]
pub enum ToolError {
    /// The input did not fit the tool's schema.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// The tool ran and failed.
    #[error("{0}")]
    Failed(String),
}

/// What a tool returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Text for the model to read.
    pub content: String,
    /// Whether the tool reports this as an error (the model sees it either way).
    pub is_error: bool,
}

impl ToolOutput {
    /// A successful result.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// A result the tool reports as an error.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Something an agent can do beyond writing text.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Name the model calls it by: 1–64 of `[A-Za-z0-9_-]`, unique in a toolbelt.
    fn name(&self) -> &str;

    /// What the tool does, for the model and for routing.
    fn description(&self) -> &str;

    /// JSON Schema of the input object.
    fn input_schema(&self) -> Value;

    /// The most this tool can do. Checked against the agent's clearance.
    fn effect(&self) -> Effect;

    /// Run the tool.
    async fn call(&self, input: Value) -> Result<ToolOutput, ToolError>;

    /// The definition sent to the model.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(self.name(), self.description(), self.input_schema())
    }
}

/// Whether `name` is a tool name every supported model API accepts.
pub fn is_valid_tool_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// How a call ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallOutcome {
    /// The tool ran and reported success.
    Ok,
    /// The tool ran and reported an error, failed, or timed out.
    Error,
    /// The toolbelt did not run it: unknown, or above the agent's clearance.
    Refused,
}

/// One journaled tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Agent that made the call, when known.
    pub agent: Option<String>,
    /// Tool name, as the model called it.
    pub tool: String,
    /// The tool's effect (`None` for an unknown tool).
    pub effect: Option<Effect>,
    /// Input the model supplied.
    pub input: Value,
    /// How it ended.
    pub outcome: CallOutcome,
    /// Output, truncated for the journal.
    pub output: String,
    /// Wall-clock start, milliseconds since the Unix epoch.
    pub started_ms: i64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Called with every tool call as it completes, to stream it somewhere.
pub type CallObserver = Arc<dyn Fn(&ToolCall) + Send + Sync>;

/// Longest tool output a model is shown.
const MAX_MODEL_OUTPUT: usize = 16_000;
/// Longest tool output kept in the journal.
const MAX_JOURNAL_OUTPUT: usize = 2_000;
/// Calls a toolbelt's journal keeps; older ones are dropped. A long-running
/// gateway clones one toolbelt into every task, sharing this journal.
const MAX_JOURNAL_ENTRIES: usize = 1_000;
/// Up to this many tools, a profile says what each does; beyond it, names only.
const DESCRIBE_DETAIL_UP_TO: usize = 5;

/// An agent's tools, the most it may do with them, and a journal of its calls.
#[derive(Clone)]
pub struct Toolbelt {
    tools: Vec<Arc<dyn Tool>>,
    clearance: Effect,
    timeout: Duration,
    agent: Option<String>,
    journal: Arc<Mutex<Vec<ToolCall>>>,
    observer: Option<CallObserver>,
}

impl std::fmt::Debug for Toolbelt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Toolbelt")
            .field(
                "tools",
                &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>(),
            )
            .field("clearance", &self.clearance)
            .finish()
    }
}

/// A tool could not be added to a toolbelt.
#[derive(Debug, Error)]
pub enum ToolbeltError {
    /// Two tools share a name.
    #[error("duplicate tool name {0:?}")]
    Duplicate(String),
    /// The name is not one every model API accepts.
    #[error("invalid tool name {0:?}: use 1-64 of A-Z a-z 0-9 _ -")]
    InvalidName(String),
}

impl Toolbelt {
    /// An empty toolbelt with this clearance.
    pub fn new(clearance: Effect) -> Self {
        Self {
            tools: Vec::new(),
            clearance,
            timeout: Duration::from_secs(60),
            agent: None,
            journal: Arc::new(Mutex::new(Vec::new())),
            observer: None,
        }
    }

    /// Add a tool.
    pub fn add(&mut self, tool: Arc<dyn Tool>) -> Result<(), ToolbeltError> {
        let name = tool.name();
        if !is_valid_tool_name(name) {
            return Err(ToolbeltError::InvalidName(name.to_string()));
        }
        if self.tools.iter().any(|t| t.name() == name) {
            return Err(ToolbeltError::Duplicate(name.to_string()));
        }
        self.tools.push(tool);
        Ok(())
    }

    /// Builder form of [`add`](Self::add).
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Result<Self, ToolbeltError> {
        self.add(tool)?;
        Ok(self)
    }

    /// Name the agent this toolbelt belongs to, for the journal.
    pub fn for_agent(mut self, name: impl Into<String>) -> Self {
        self.agent = Some(name.into());
        self
    }

    /// How long one call may run before it is abandoned (default 60 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Be told about every call as it completes.
    pub fn with_observer(mut self, observer: CallObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// The most this agent may do.
    pub fn clearance(&self) -> Effect {
        self.clearance
    }

    /// Every tool, usable or not.
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    /// The tools within the agent's clearance.
    pub fn usable(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools.iter().filter(|t| t.effect() <= self.clearance)
    }

    /// Whether any tool is within the agent's clearance.
    pub fn has_usable(&self) -> bool {
        self.usable().next().is_some()
    }

    /// Definitions of the usable tools, to send to the model.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.usable().map(|t| t.definition()).collect()
    }

    /// The usable tools in plain language, to add to what the agent says it
    /// does. Empty when there are none.
    pub fn describe(&self) -> String {
        // A few tools read best with what each does; past that the names
        // alone say enough and keep the profile (and Agent Card) short.
        let detailed = self.usable().count() <= DESCRIBE_DETAIL_UP_TO;
        let lines: Vec<String> = self
            .usable()
            .map(|t| {
                if detailed {
                    format!("{} ({})", t.name(), first_sentence(t.description()))
                } else {
                    t.name().to_string()
                }
            })
            .collect();
        if lines.is_empty() {
            String::new()
        } else {
            let sep = if detailed { "; " } else { ", " };
            format!("It can use these tools: {}.", lines.join(sep))
        }
    }

    /// Every call so far.
    pub fn journal(&self) -> Vec<ToolCall> {
        self.journal.lock().map(|j| j.clone()).unwrap_or_default()
    }

    /// Run a tool the model asked for, enforcing the allowlist, the agent's
    /// clearance and the timeout, and journal the call.
    pub async fn call(&self, name: &str, input: Value) -> ToolCall {
        let started_ms = chrono_now_ms();
        let start = Instant::now();
        let tool = self.tools.iter().find(|t| t.name() == name);
        let (outcome, output, effect) = match tool {
            None => (
                CallOutcome::Refused,
                format!("no tool named {name:?} is available to this agent"),
                None,
            ),
            Some(t) if t.effect() > self.clearance => (
                CallOutcome::Refused,
                format!(
                    "{name} needs {} clearance; this agent has {}",
                    t.effect(),
                    self.clearance
                ),
                Some(t.effect()),
            ),
            Some(t) => {
                let effect = Some(t.effect());
                match tokio::time::timeout(self.timeout, t.call(input.clone())).await {
                    Err(_) => (
                        CallOutcome::Error,
                        format!("timed out after {} s", self.timeout.as_secs()),
                        effect,
                    ),
                    Ok(Err(e)) => (CallOutcome::Error, e.to_string(), effect),
                    Ok(Ok(out)) if out.is_error => (CallOutcome::Error, out.content, effect),
                    Ok(Ok(out)) => (CallOutcome::Ok, out.content, effect),
                }
            }
        };
        let call = ToolCall {
            agent: self.agent.clone(),
            tool: name.to_string(),
            effect,
            input,
            outcome,
            output,
            started_ms,
            duration_ms: start.elapsed().as_millis() as u64,
        };
        match call.outcome {
            CallOutcome::Ok => debug!(
                agent = call.agent.as_deref().unwrap_or("-"),
                "tool {} ok in {} ms", call.tool, call.duration_ms
            ),
            CallOutcome::Error => debug!(
                agent = call.agent.as_deref().unwrap_or("-"),
                "tool {} failed: {}",
                call.tool,
                truncate(&call.output, 200)
            ),
            CallOutcome::Refused => info!(
                agent = call.agent.as_deref().unwrap_or("-"),
                "tool {} refused: {}", call.tool, call.output
            ),
        }
        let mut journaled = call.clone();
        journaled.output = truncate(&journaled.output, MAX_JOURNAL_OUTPUT);
        if let Some(observer) = &self.observer {
            observer(&journaled);
        }
        if let Ok(mut j) = self.journal.lock() {
            if j.len() >= MAX_JOURNAL_ENTRIES {
                j.remove(0);
            }
            j.push(journaled);
        }
        call
    }
}

/// The result of [`run_with_tools`].
#[derive(Debug, Clone)]
pub struct ToolRun {
    /// The model's final answer.
    pub text: String,
    /// Every tool call made, in order.
    pub calls: Vec<ToolCall>,
    /// Model turns taken.
    pub turns: usize,
}

/// Instructions added to an agent's system prompt when it has tools.
pub const TOOL_GUIDANCE: &str = "You have tools, and they reach what you cannot otherwise see: \
files, systems, today's date. Before you say you lack information, or write about something, check \
whether a tool can find it. If a first attempt finds nothing, try another way (list what is \
available, search more broadly) before giving up. Answer directly only when no tool would help. \
Never invent a tool's result; if a tool fails or is refused, say so rather than guessing.";

/// Answer `prompt`, letting the model call the toolbelt's usable tools for up
/// to `max_turns` model turns. Calls run one at a time, in the order asked.
///
/// When the turns run out, the model is asked once more, with tools off, to
/// answer from what it has.
pub async fn run_with_tools(
    backend: &dyn LlmBackend,
    system: &str,
    prompt: &str,
    belt: &Toolbelt,
    max_turns: usize,
) -> Result<ToolRun, LlmError> {
    let tools = belt.definitions();
    let system = format!("{system}\n\n{TOOL_GUIDANCE}");
    let mut messages = vec![Message::user(prompt)];
    let mut calls = Vec::new();
    let request = |messages: &Vec<Message>, choice: ToolChoice| {
        let mut r = GenerateRequestV2::new(messages.clone());
        r.system = Some(system.clone());
        r.tools = tools.clone();
        r.tool_choice = choice;
        r
    };

    for turn in 1..=max_turns.max(1) {
        let response = backend
            .generate_v2(request(&messages, ToolChoice::Auto))
            .await?;
        let uses: Vec<(String, String, Value)> = response
            .tool_uses()
            .into_iter()
            .map(|(id, name, input)| (id.to_string(), name.to_string(), input.clone()))
            .collect();
        messages.push(Message {
            role: MessageRole::Assistant,
            content: response.content.clone(),
        });
        if uses.is_empty() {
            return Ok(ToolRun {
                text: response.text(),
                calls,
                turns: turn,
            });
        }
        let mut results = Vec::new();
        for (id, name, input) in uses {
            debug!("turn {turn}: calling {name}");
            let call = belt.call(&name, input).await;
            results.push(ContentBlock::ToolResult {
                tool_use_id: id,
                content: truncate(&call.output, MAX_MODEL_OUTPUT),
                is_error: call.outcome != CallOutcome::Ok,
            });
            let mut journaled = call;
            journaled.output = truncate(&journaled.output, MAX_JOURNAL_OUTPUT);
            calls.push(journaled);
        }
        messages.push(Message {
            role: MessageRole::User,
            content: results,
        });
    }

    messages.push(Message::user(
        "You have used all your tool turns. Answer now from what you have.",
    ));
    let response = backend
        .generate_v2(request(&messages, ToolChoice::None))
        .await?;
    Ok(ToolRun {
        text: response.text(),
        calls,
        turns: max_turns.max(1) + 1,
    })
}

fn first_sentence(text: &str) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let end = line.find(". ").map(|i| i + 1).unwrap_or(line.len());
    truncate(line[..end].trim(), 160)
}

/// `s` cut to at most `max` bytes on a character boundary, marked when cut.
pub(crate) fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… [{} more bytes]", &s[..end], s.len() - end)
}

fn chrono_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;
