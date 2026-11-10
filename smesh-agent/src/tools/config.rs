//! A config file that gives a mesh its tools.
//!
//! ```toml
//! # An MCP server run as a local process (stdio).
//! [servers.files]
//! command = "npx"
//! args = ["-y", "@modelcontextprotocol/server-filesystem", "./docs"]
//!
//! # An MCP server over streamable HTTP. A value starting with `$` is read
//! # from the environment, so secrets stay out of the file.
//! [servers.tracker]
//! url = "https://mcp.example.com/mcp"
//! bearer = "$TRACKER_TOKEN"
//!
//! # Correct what a tool may do when its annotations are missing or wrong.
//! [servers.tracker.effects]
//! search_issues = "read"
//!
//! [[agents]]
//! name = "Librarian"
//! capability = "a librarian who answers questions from the documents in ./docs"
//! tools = ["files.read_*", "files.list_directory", "builtin.now"]
//! # read (the default) | write | execute
//! clearance = "read"
//! # optional; the model must support tool calling
//! model = "openai/gpt-5-mini"
//! ```
//!
//! An agent's `tools` are patterns over `<server>.<tool>`, where `*` matches
//! anything: `files.*` is every tool on `files`. `builtin.<name>` names a
//! tool written in Rust ([`super::builtin`]). Tools nobody's patterns match are
//! never offered to any agent.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use thiserror::Error;

use super::mcp::{McpError, McpServer, McpTransport};
use super::{builtin, Effect, Tool, Toolbelt, ToolbeltError};

/// The whole file.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolsConfig {
    /// MCP servers, by name.
    #[serde(default)]
    pub servers: BTreeMap<String, ServerConfig>,
    /// Agents and the tools each may use.
    #[serde(default)]
    pub agents: Vec<AgentTools>,
}

/// One MCP server.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Program to run for a stdio server.
    pub command: Option<String>,
    /// Its arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment for the process (`$NAME` reads `NAME` from ours).
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Working directory for the process.
    pub cwd: Option<PathBuf>,
    /// Show the process's stderr (hidden by default; turn on to debug a server).
    #[serde(default)]
    pub show_stderr: bool,
    /// Endpoint of a streamable-HTTP server.
    pub url: Option<String>,
    /// Bearer token for an HTTP server (`$NAME` reads `NAME` from the environment).
    pub bearer: Option<String>,
    /// Effect overrides, by the server's own tool name.
    #[serde(default)]
    pub effects: HashMap<String, Effect>,
}

/// One agent's tools.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentTools {
    /// The agent's name.
    pub name: String,
    /// What it does, in plain language, phrased to follow "You are NAME, ...".
    pub capability: String,
    /// Patterns over `<server>.<tool>` naming the tools it may use.
    #[serde(default)]
    pub tools: Vec<String>,
    /// The most it may do with them.
    #[serde(default = "default_clearance")]
    pub clearance: Effect,
    /// OpenRouter model for this agent; it must support tool calling.
    #[serde(default)]
    pub model: Option<String>,
}

fn default_clearance() -> Effect {
    Effect::Read
}

/// The config file is wrong.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// It could not be read.
    #[error("could not read {path}: {source}")]
    Read {
        /// File.
        path: PathBuf,
        /// Why.
        source: std::io::Error,
    },
    /// It is not valid TOML of the expected shape.
    #[error("{0}")]
    Parse(String),
    /// It parsed, but says something impossible.
    #[error("{0}")]
    Invalid(String),
    /// A server could not be reached.
    #[error(transparent)]
    Mcp(#[from] McpError),
    /// A tool could not join a toolbelt.
    #[error(transparent)]
    Toolbelt(#[from] ToolbeltError),
}

/// The name reserved for tools written in Rust.
pub const BUILTIN: &str = "builtin";

impl ToolsConfig {
    /// Read and check a config file.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text)
    }

    /// Parse and check config text.
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(text).map_err(|e| ConfigError::Parse(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: String| Err(ConfigError::Invalid(m));
        for (name, server) in &self.servers {
            if name == BUILTIN {
                return bad(format!(
                    "server name {BUILTIN:?} is reserved for built-in tools"
                ));
            }
            if name.is_empty()
                || !name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return bad(format!(
                    "server name {name:?}: use letters, digits, '_' and '-'"
                ));
            }
            match (&server.command, &server.url) {
                (Some(_), Some(_)) => {
                    return bad(format!(
                        "server {name:?}: give `command` or `url`, not both"
                    ))
                }
                (None, None) => return bad(format!("server {name:?}: give `command` or `url`")),
                (None, Some(_)) if !server.args.is_empty() || !server.env.is_empty() => {
                    return bad(format!(
                        "server {name:?}: `args` and `env` apply only to a `command` server"
                    ))
                }
                (None, Some(_)) if server.show_stderr || server.cwd.is_some() => {
                    return bad(format!(
                        "server {name:?}: `cwd` and `show_stderr` apply only to a `command` server"
                    ))
                }
                (Some(_), None) if server.bearer.is_some() => {
                    return bad(format!(
                        "server {name:?}: `bearer` applies only to a `url` server"
                    ))
                }
                _ => {}
            }
        }
        let mut seen = std::collections::HashSet::new();
        for agent in &self.agents {
            if agent.name.trim().is_empty() || agent.capability.trim().is_empty() {
                return bad("every agent needs a `name` and a `capability`".to_string());
            }
            if !seen.insert(agent.name.as_str()) {
                return bad(format!("agent {:?} is declared twice", agent.name));
            }
            for pattern in &agent.tools {
                let server = pattern.split('.').next().unwrap_or_default();
                if !pattern.contains('.') {
                    return bad(format!(
                        "agent {:?}: tool pattern {pattern:?} must look like <server>.<tool>",
                        agent.name
                    ));
                }
                if server != BUILTIN && !self.servers.contains_key(server) {
                    return bad(format!(
                        "agent {:?}: tool pattern {pattern:?} names unknown server {server:?}",
                        agent.name
                    ));
                }
            }
        }
        Ok(())
    }

    /// The servers some agent's patterns refer to.
    fn used_servers(&self) -> Vec<&str> {
        self.servers
            .keys()
            .filter(|s| {
                self.agents.iter().any(|a| {
                    a.tools
                        .iter()
                        .any(|p| p.split('.').next() == Some(s.as_str()))
                })
            })
            .map(String::as_str)
            .collect()
    }

    /// Connect to every server an agent uses, concurrently, and gather the
    /// tools they offer alongside the built-in ones.
    pub async fn connect(&self) -> Result<Catalog, ConfigError> {
        let used = self.used_servers();
        let connections = futures::future::join_all(used.iter().map(|name| {
            let server = &self.servers[*name];
            async move {
                let transport = transport_for(name, server)?;
                Ok::<_, ConfigError>((*name, McpServer::connect(name, &transport).await?))
            }
        }))
        .await;
        let mut catalog = Catalog::default();
        for (name, tool) in builtin::all() {
            catalog.register(BUILTIN, name, tool);
        }
        for connection in connections {
            let (name, server) = connection?;
            let overrides = &self.servers[name].effects;
            for (remote, tool) in server.tool_names().into_iter().zip(server.tools(overrides)) {
                catalog.register(name, &remote, tool);
            }
            catalog.servers.push(server);
        }
        Ok(catalog)
    }
}

/// Resolve `$NAME` from the environment; any other value is literal.
fn resolve(value: &str) -> Result<String, ConfigError> {
    match value.strip_prefix('$') {
        Some(var) => std::env::var(var)
            .map_err(|_| ConfigError::Invalid(format!("environment variable {var} is not set"))),
        None => Ok(value.to_string()),
    }
}

fn transport_for(name: &str, server: &ServerConfig) -> Result<McpTransport, ConfigError> {
    if let Some(command) = &server.command {
        let env = server
            .env
            .iter()
            .map(|(k, v)| Ok((k.clone(), resolve(v)?)))
            .collect::<Result<_, ConfigError>>()?;
        Ok(McpTransport::Command {
            command: command.clone(),
            args: server.args.clone(),
            env,
            cwd: server.cwd.clone(),
            show_stderr: server.show_stderr,
        })
    } else if let Some(url) = &server.url {
        Ok(McpTransport::Http {
            url: url.clone(),
            bearer: server.bearer.as_deref().map(resolve).transpose()?,
        })
    } else {
        Err(ConfigError::Invalid(format!(
            "server {name:?}: give `command` or `url`"
        )))
    }
}

/// Every tool available to a mesh, by `<server>.<tool>`, and the live
/// connections behind them. Dropping it closes the connections.
#[derive(Default)]
pub struct Catalog {
    entries: Vec<(String, Arc<dyn Tool>)>,
    servers: Vec<Arc<McpServer>>,
}

impl std::fmt::Debug for Catalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Catalog")
            .field(
                "tools",
                &self.entries.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Catalog {
    /// Add a tool under `<server>.<tool>`; patterns in the config match this name.
    /// Use it to offer tools written in Rust.
    pub fn register(&mut self, server: &str, tool: &str, implementation: Arc<dyn Tool>) {
        self.entries
            .push((format!("{server}.{tool}"), implementation));
    }

    /// Every tool, as `(<server>.<tool>, tool)`.
    pub fn entries(&self) -> &[(String, Arc<dyn Tool>)] {
        &self.entries
    }

    /// Build an agent's toolbelt from its patterns. Also returns the patterns
    /// that matched nothing, which usually means a typo.
    pub fn toolbelt_for(&self, agent: &AgentTools) -> Result<(Toolbelt, Vec<String>), ConfigError> {
        let mut belt = Toolbelt::new(agent.clearance).for_agent(&agent.name);
        let mut unmatched = Vec::new();
        for pattern in &agent.tools {
            let mut any = false;
            for (qualified, tool) in &self.entries {
                if glob(pattern, qualified) {
                    any = true;
                    if !belt.tools().iter().any(|t| t.name() == tool.name()) {
                        belt.add(Arc::clone(tool))?;
                    }
                }
            }
            if !any {
                unmatched.push(pattern.clone());
            }
        }
        Ok((belt, unmatched))
    }
}

/// Whether `text` matches `pattern`, where `*` matches any run of characters.
pub fn glob(pattern: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pattern.chars().collect(), text.chars().collect());
    let (mut pi, mut ti, mut star, mut mark) = (0, 0, None::<usize>, 0);
    while ti < t.len() {
        if pi < p.len() && p[pi] != '*' && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"
        [servers.files]
        command = "npx"
        args = ["-y", "@modelcontextprotocol/server-filesystem", "./docs"]

        [servers.tracker]
        url = "https://mcp.example.com/mcp"
        bearer = "$SMESH_TEST_UNSET_TOKEN"

        [servers.tracker.effects]
        search_issues = "read"
        run_query = "execute"

        [[agents]]
        name = "Librarian"
        capability = "a librarian"
        tools = ["files.read_*", "builtin.now"]

        [[agents]]
        name = "Triage"
        capability = "a triager of bug reports"
        tools = ["tracker.*"]
        clearance = "write"
    "#;

    #[test]
    fn the_documented_example_parses() {
        let c = ToolsConfig::parse(EXAMPLE).unwrap();
        assert_eq!(c.servers.len(), 2);
        assert_eq!(c.servers["tracker"].effects["run_query"], Effect::Execute);
        assert_eq!(c.agents[0].clearance, Effect::Read, "read is the default");
        assert_eq!(c.agents[1].clearance, Effect::Write);
        assert_eq!(c.used_servers(), vec!["files", "tracker"]);
    }

    #[test]
    fn impossible_configs_are_refused_with_a_reason() {
        let cases = [
            ("[servers.x]\n", "give `command` or `url`"),
            (
                "[servers.x]\ncommand = \"a\"\nurl = \"http://b\"\n",
                "not both",
            ),
            ("[servers.builtin]\ncommand = \"a\"\n", "reserved"),
            ("[servers.\"a.b\"]\ncommand = \"a\"\n", "server name"),
            (
                "[servers.x]\ncommand = \"a\"\nbearer = \"t\"\n",
                "only to a `url` server",
            ),
            (
                "[[agents]]\nname = \"A\"\ncapability = \"c\"\ntools = [\"nope.*\"]\n",
                "unknown server",
            ),
            (
                "[[agents]]\nname = \"A\"\ncapability = \"c\"\ntools = [\"star\"]\n",
                "<server>.<tool>",
            ),
            (
                "[[agents]]\nname = \"A\"\ncapability = \"c\"\n[[agents]]\nname = \"A\"\ncapability = \"d\"\n",
                "declared twice",
            ),
            (
                "[[agents]]\nname = \"A\"\ncapability = \"c\"\nclearance = \"root\"\n",
                "unknown effect",
            ),
            ("[surprise]\n", "unknown field"),
        ];
        for (text, expected) in cases {
            let err = ToolsConfig::parse(text).unwrap_err().to_string();
            assert!(err.contains(expected), "{text:?}: {err}");
        }
    }

    #[test]
    fn secrets_come_from_the_environment() {
        let c = ToolsConfig::parse(EXAMPLE).unwrap();
        let err = transport_for("tracker", &c.servers["tracker"]).unwrap_err();
        assert!(err.to_string().contains("SMESH_TEST_UNSET_TOKEN"), "{err}");
        assert_eq!(resolve("literal").unwrap(), "literal");
    }

    #[test]
    fn glob_matches_star_anywhere() {
        assert!(glob("files.*", "files.read_file"));
        assert!(glob("files.read_*", "files.read_text_file"));
        assert!(glob("*.read_*", "github.read_issue"));
        assert!(glob("files.list_directory", "files.list_directory"));
        assert!(!glob("files.read_*", "files.write_file"));
        assert!(!glob("files.*", "filesystem.read"));
        assert!(!glob("files.list", "files.list_directory"));
    }

    #[tokio::test]
    async fn builtin_tools_need_no_server() {
        let c = ToolsConfig::parse(EXAMPLE).unwrap();
        let mut catalog = Catalog::default();
        for (name, tool) in builtin::all() {
            catalog.register(BUILTIN, name, tool);
        }
        let (belt, unmatched) = catalog.toolbelt_for(&c.agents[0]).unwrap();
        assert_eq!(unmatched, vec!["files.read_*".to_string()]);
        assert_eq!(belt.definitions().len(), 1);
        assert_eq!(belt.definitions()[0].name, "builtin__now");
    }
}
