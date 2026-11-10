//! Tools from MCP (Model Context Protocol) servers.
//!
//! An [`McpServer`] is one connection, over stdio to a local process or over
//! streamable HTTP to a remote one. Each tool it lists becomes a [`Tool`]
//! named `<server>__<tool>`, so tools from different servers never collide.
//!
//! What a tool may do comes from its MCP annotations: `readOnlyHint: true`
//! is [`Effect::Read`]; anything else is [`Effect::Write`], because the
//! protocol's defaults assume a tool may change things. MCP cannot say a tool
//! runs arbitrary code, so [`Effect::Execute`] comes only from an override in
//! the config file, as can any other correction.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use rmcp::model::{CallToolRequestParams, ClientConfig, Implementation};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::{IntoTransport, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::ServiceExt;
use serde_json::Value;
use thiserror::Error;
use tracing::debug;

use super::{is_valid_tool_name, Effect, Tool, ToolError, ToolOutput};

/// How to reach an MCP server.
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// Spawn a local process and speak MCP over its stdin and stdout.
    Command {
        /// Program to run.
        command: String,
        /// Its arguments.
        args: Vec<String>,
        /// Extra environment variables.
        env: HashMap<String, String>,
        /// Working directory, if not the current one.
        cwd: Option<std::path::PathBuf>,
        /// Pass the server's stderr through instead of discarding it.
        show_stderr: bool,
    },
    /// Connect to a streamable-HTTP MCP endpoint.
    Http {
        /// Endpoint URL, e.g. `http://127.0.0.1:8000/mcp`.
        url: String,
        /// Bearer token to send, if the server needs one.
        bearer: Option<String>,
    },
}

/// Connecting to, or talking with, an MCP server failed.
#[derive(Debug, Error)]
pub enum McpError {
    /// The server process could not be started.
    #[error("could not start MCP server {server:?}: {source}")]
    Spawn {
        /// Server name.
        server: String,
        /// Why.
        source: std::io::Error,
    },
    /// The MCP handshake failed or timed out.
    #[error("MCP server {server:?} did not initialize: {reason}")]
    Initialize {
        /// Server name.
        server: String,
        /// Why.
        reason: String,
    },
    /// Listing the server's tools failed.
    #[error("MCP server {server:?} would not list its tools: {reason}")]
    ListTools {
        /// Server name.
        server: String,
        /// Why.
        reason: String,
    },
}

/// A live connection to one MCP server and the tools it offers.
pub struct McpServer {
    name: String,
    service: RunningService<RoleClient, ClientConfig>,
    tools: Vec<rmcp::model::Tool>,
}

impl std::fmt::Debug for McpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServer")
            .field("name", &self.name)
            .field("tools", &self.tools.len())
            .finish()
    }
}

/// How long a server may take to start and complete the handshake. `npx`
/// may download a package the first time, so this is generous.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(120);

fn client_info() -> ClientConfig {
    let mut info = ClientConfig::default();
    info.client_info = Implementation::new("smesh", env!("CARGO_PKG_VERSION"));
    info
}

impl McpServer {
    /// Connect to a server and list its tools.
    pub async fn connect(name: &str, transport: &McpTransport) -> Result<Arc<Self>, McpError> {
        match transport {
            McpTransport::Command {
                command,
                args,
                env,
                cwd,
                show_stderr,
            } => {
                let mut cmd = tokio::process::Command::new(command);
                cmd.args(args).envs(env);
                if let Some(dir) = cwd {
                    cmd.current_dir(dir);
                }
                // Servers log banners and warnings to stderr; keep them out of
                // the way unless asked for, since they are not MCP traffic.
                let stderr = if *show_stderr {
                    std::process::Stdio::inherit()
                } else {
                    std::process::Stdio::null()
                };
                let (child, _) = TokioChildProcess::builder(cmd)
                    .stderr(stderr)
                    .spawn()
                    .map_err(|source| McpError::Spawn {
                        server: name.to_string(),
                        source,
                    })?;
                Self::connect_with(name, child).await
            }
            McpTransport::Http { url, bearer } => {
                let mut config =
                    rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig::with_uri(
                        url.as_str(),
                    );
                if let Some(token) = bearer {
                    config = config.auth_header(format!("Bearer {token}"));
                }
                Self::connect_with(name, StreamableHttpClientTransport::from_config(config)).await
            }
        }
    }

    /// Connect over any MCP transport (an in-process pipe, say) and list its tools.
    pub async fn connect_with<T, E, A>(name: &str, transport: T) -> Result<Arc<Self>, McpError>
    where
        T: IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let service = tokio::time::timeout(CONNECT_TIMEOUT, client_info().serve(transport))
            .await
            .map_err(|_| McpError::Initialize {
                server: name.to_string(),
                reason: format!("timed out after {} s", CONNECT_TIMEOUT.as_secs()),
            })?
            .map_err(|e| McpError::Initialize {
                server: name.to_string(),
                reason: e.to_string(),
            })?;
        let tools = service
            .list_all_tools()
            .await
            .map_err(|e| McpError::ListTools {
                server: name.to_string(),
                reason: e.to_string(),
            })?;
        debug!("MCP server {name}: {} tools", tools.len());
        Ok(Arc::new(Self {
            name: name.to_string(),
            service,
            tools,
        }))
    }

    /// The server's name in the config.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The names of the tools the server lists, as it names them.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.to_string()).collect()
    }

    /// The server's tools as [`Tool`]s. `overrides` maps a tool's own name to
    /// the effect to use instead of the one its annotations imply.
    pub fn tools(self: &Arc<Self>, overrides: &HashMap<String, Effect>) -> Vec<Arc<dyn Tool>> {
        self.tools
            .iter()
            .map(|t| {
                let remote = t.name.to_string();
                let effect = overrides
                    .get(&remote)
                    .copied()
                    .unwrap_or_else(|| effect_from_annotations(t.annotations.as_ref()));
                let description = t
                    .description
                    .as_deref()
                    .map(str::to_string)
                    .or_else(|| t.title.clone())
                    .unwrap_or_else(|| remote.clone());
                Arc::new(McpTool {
                    server: Arc::clone(self),
                    name: local_name(&self.name, &remote),
                    remote,
                    description,
                    schema: Value::Object((*t.input_schema).clone()),
                    effect,
                }) as Arc<dyn Tool>
            })
            .collect()
    }
}

/// What a tool may do, from its MCP annotations.
pub fn effect_from_annotations(annotations: Option<&rmcp::model::ToolAnnotations>) -> Effect {
    match annotations.and_then(|a| a.read_only_hint) {
        Some(true) => Effect::Read,
        _ => Effect::Write,
    }
}

/// The name a model calls a server's tool by: `<server>__<tool>`, with any
/// character a model API rejects replaced, cut to 64 bytes.
pub fn local_name(server: &str, tool: &str) -> String {
    let clean = |s: &str| -> String {
        s.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    };
    let mut name = format!("{}__{}", clean(server), clean(tool));
    name.truncate(64);
    debug_assert!(is_valid_tool_name(&name) || name.is_empty());
    name
}

/// One tool on an MCP server.
struct McpTool {
    server: Arc<McpServer>,
    name: String,
    remote: String,
    description: String,
    schema: Value,
    effect: Effect,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        self.schema.clone()
    }

    fn effect(&self) -> Effect {
        self.effect
    }

    async fn call(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let arguments = match input {
            Value::Object(map) => map,
            Value::Null => serde_json::Map::new(),
            other => {
                return Err(ToolError::InvalidInput(format!(
                    "expected a JSON object of arguments, got {other}"
                )))
            }
        };
        let mut params = CallToolRequestParams::new(self.remote.clone());
        params.arguments = Some(arguments);
        let result = self
            .server
            .service
            .call_tool(params)
            .await
            .map_err(|e| ToolError::Failed(format!("{}: {e}", self.server.name)))?;
        let mut parts: Vec<String> = result
            .content
            .iter()
            .map(|block| match block.as_text() {
                Some(text) => text.text.clone(),
                None => serde_json::to_string(block).unwrap_or_default(),
            })
            .collect();
        if parts.is_empty() {
            if let Some(structured) = &result.structured_content {
                parts.push(structured.to_string());
            }
        }
        let content = parts.join("\n");
        Ok(if result.is_error.unwrap_or(false) {
            ToolOutput::error(content)
        } else {
            ToolOutput::text(content)
        })
    }
}
