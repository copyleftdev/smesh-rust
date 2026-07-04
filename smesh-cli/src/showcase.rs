//! Showcase server: serves the two-tab kinetic visualization and exposes the
//! *real* SMESH engine over a tiny JSON API.
//!
//! - `GET  /`             → the showcase page (embedded HTML)
//! - `POST /api/diffuse`  → run real signal diffusion, return per-tick frames
//! - `POST /api/consensus`→ run real emergent claim/back-off settlement
//! - `GET  /api/owasp`    → the latest OWASP scorecard JSON (if present)
//!
//! Both mesh endpoints run pure, LLM-free Rust so the animation stays real-time.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

use serde_json::json;
use smesh_agent::{AgentCoordinator, AgentRole, CoordinatorConfig, TaskDefinition};
use smesh_core::{Network, NetworkTopology, Signal, SignalType};

const SHOWCASE_HTML: &str = include_str!("assets/showcase.html");

/// Serve the showcase on the given port.
pub fn serve(port: u16) -> anyhow::Result<()> {
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)?;
    let url = format!("http://{addr}/");

    println!("╔═══════════════════════════════════════╗");
    println!("║        SMESH Showcase                 ║");
    println!("╚═══════════════════════════════════════╝");
    println!("\n🌐 Serving at {url}");
    println!("   Tab 1: OWASP Benchmark scorecard");
    println!("   Tab 2: Live Mesh (diffusion + consensus, real engine)\n");
    let _ = open_browser(&url);

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                std::thread::spawn(move || {
                    if let Err(e) = handle(s) {
                        eprintln!("request error: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

fn handle(mut stream: TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);

    // Request line.
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    // Headers (we only need Content-Length).
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }

    // Body.
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    let body_str = String::from_utf8_lossy(&body);

    // Route.
    let (status, ctype, payload): (&str, &str, String) = match (method.as_str(), path.as_str()) {
        ("POST", "/api/diffuse") => ("200 OK", "application/json", api_diffuse(&body_str)),
        ("POST", "/api/consensus") => ("200 OK", "application/json", api_consensus(&body_str)),
        ("POST", "/api/chaos") => ("200 OK", "application/json", api_chaos(&body_str)),
        ("GET", "/api/owasp") => ("200 OK", "application/json", api_owasp()),
        ("GET", "/") | ("GET", "/index.html") => {
            ("200 OK", "text/html; charset=utf-8", SHOWCASE_HTML.to_string())
        }
        _ => ("404 Not Found", "application/json", "{\"error\":\"not found\"}".into()),
    };

    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

// ── Diffusion endpoint ──────────────────────────────────────────────────────

fn api_diffuse(body: &str) -> String {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let text = req.get("text").and_then(|v| v.as_str()).unwrap_or("signal");
    let n_nodes = req
        .get("nodes")
        .and_then(|v| v.as_u64())
        .unwrap_or(48)
        .clamp(6, 200) as usize;
    let ticks = req
        .get("ticks")
        .and_then(|v| v.as_u64())
        .unwrap_or(44)
        .clamp(4, 200) as usize;
    let seeds = req
        .get("seeds")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .clamp(1, 5) as usize;
    let topology = parse_topology(req.get("topology").and_then(|v| v.as_str()).unwrap_or("small_world"));

    let mut net = Network::with_topology(n_nodes, topology);
    let node_ids: Vec<String> = net.nodes.keys().cloned().collect();

    // Seed the signal at one or more origin nodes (spread apart).
    let mut origins = Vec::new();
    for i in 0..seeds.min(node_ids.len()) {
        let idx = i * node_ids.len() / seeds;
        origins.push(node_ids[idx.min(node_ids.len() - 1)].clone());
    }
    for origin in &origins {
        let mut sig = Signal::builder(SignalType::Data)
            .payload(text.as_bytes().to_vec())
            .intensity(1.0)
            .confidence(1.0)
            .ttl(1000.0)
            .radius(50)
            .origin(origin)
            .build();
        sig.mark_reached(origin);
        net.field.emit_anonymous(sig);
    }

    // Undirected edge list from the hyphae.
    let mut seen = std::collections::HashSet::new();
    let mut edges = Vec::new();
    for (from, hs) in &net.hyphae {
        for h in hs {
            let key = if from < &h.to {
                (from.clone(), h.to.clone())
            } else {
                (h.to.clone(), from.clone())
            };
            if seen.insert(key.clone()) {
                edges.push([key.0, key.1]);
            }
        }
    }

    // Per-tick frames.
    let mut frames = vec![snapshot(&net)];
    for _ in 0..ticks {
        net.tick(0.25);
        frames.push(snapshot(&net));
        if net.field.signals.is_empty() {
            break;
        }
    }

    json!({
        "mode": "diffuse",
        "text": text,
        "nodes": node_ids,
        "edges": edges,
        "origins": origins,
        "frames": frames,
    })
    .to_string()
}

/// A snapshot of every active signal's diffusion state at one tick.
fn snapshot(net: &Network) -> serde_json::Value {
    let signals: Vec<serde_json::Value> = net
        .field
        .signals
        .values()
        .map(|s| {
            json!({
                "origin": s.origin_node_id,
                "reached": s.reached_nodes,
                "intensity": s.current_intensity,
                "hops": s.hops,
                "reinforcement": s.reinforcement_count,
            })
        })
        .collect();
    json!({ "signals": signals, "active": net.field.signals.len() })
}

// ── Consensus endpoint ──────────────────────────────────────────────────────

fn api_consensus(body: &str) -> String {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let text = req.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();

    let config = CoordinatorConfig {
        n_agents: 3,
        roles: vec![AgentRole::Coder, AgentRole::Reviewer, AgentRole::Analyst],
        ..Default::default()
    };
    // Backend is constructed but never called — settlement is LLM-free.
    let mut coord = AgentCoordinator::with_openrouter(config, "google/gemini-2.5-flash-lite");

    let (task_type, label) = infer_task_type(text);
    let description = if text.is_empty() {
        "Review and analyze the submitted work".to_string()
    } else {
        text.to_string()
    };
    let task_id = coord.add_task(TaskDefinition {
        task_type: task_type.to_string(),
        description,
        priority: 0.85,
    });

    let round_traces = coord.settle_traced();
    let roster = coord.agent_roster();
    let winners = coord.converged_winners();

    let agents: Vec<serde_json::Value> = roster
        .iter()
        .map(|a| json!({ "id": a.id, "name": a.name, "role": a.role }))
        .collect();

    let rounds: Vec<serde_json::Value> = round_traces
        .iter()
        .map(|r| {
            json!({
                "claims": r.claims.iter().map(|(node, task, aff)| json!({
                    "agent": node, "task": task, "affinity": aff
                })).collect::<Vec<_>>(),
                "backoffs": r.backoffs.iter().map(|(node, task)| json!({
                    "agent": node, "task": task
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    let winners_json: Vec<serde_json::Value> = winners
        .iter()
        .map(|(node, task)| json!({ "agent": node, "task": task }))
        .collect();

    json!({
        "mode": "consensus",
        "agents": agents,
        "tasks": [{ "id": task_id, "type": task_type, "label": label }],
        "rounds": rounds,
        "winners": winners_json,
    })
    .to_string()
}

/// Map free text to a SMESH task type by keyword.
fn infer_task_type(text: &str) -> (&'static str, &'static str) {
    let t = text.to_ascii_lowercase();
    let has = |kws: &[&str]| kws.iter().any(|k| t.contains(k));
    if has(&["review", "audit", "vulnerab", "security", "bug"]) {
        ("code_review", "Code Review")
    } else if has(&["test", "coverage", "assert"]) {
        ("testing", "Testing")
    } else if has(&["document", "docs", "readme", "explain", "comment"]) {
        ("documentation", "Documentation")
    } else if has(&["implement", "write code", "build", "refactor", "function", "fix"]) {
        ("code_write", "Code Writing")
    } else {
        ("analysis", "Analysis")
    }
}

// ── Chaos endpoint (resilience, real engine) ────────────────────────────────

fn api_chaos(body: &str) -> String {
    use crate::resilience::scenario::{scenario_trace, Attack, ScenarioConfig};

    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let attack = match req.get("attack").and_then(|v| v.as_str()).unwrap_or("node_failure") {
        "eclipse" => Attack::Eclipse,
        "partition" => Attack::Partition,
        _ => Attack::NodeFailure,
    };
    let intensity = req
        .get("intensity")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.2)
        .clamp(0.0, 0.9);
    let nodes = req
        .get("nodes")
        .and_then(|v| v.as_u64())
        .unwrap_or(52)
        .clamp(6, 200) as usize;
    let topology = parse_topology(req.get("topology").and_then(|v| v.as_str()).unwrap_or("small_world"));

    let cfg = ScenarioConfig {
        nodes,
        topology,
        ticks: 46,
        trials: 1,
    };
    scenario_trace(attack, intensity, &cfg).to_string()
}

// ── OWASP scorecard passthrough ─────────────────────────────────────────────

fn api_owasp() -> String {
    for path in ["owasp-results.json", ".owasp-benchmark/owasp-results.json"] {
        if let Ok(s) = std::fs::read_to_string(path) {
            return s;
        }
    }
    json!({ "error": "no scorecard yet; run `smesh owasp --json owasp-results.json` first" })
        .to_string()
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn parse_topology(s: &str) -> NetworkTopology {
    match s {
        "ring" => NetworkTopology::Ring,
        "mesh" | "full" => NetworkTopology::FullMesh,
        "scale_free" | "sf" => NetworkTopology::ScaleFree,
        "random" => NetworkTopology::Random,
        "grid" => NetworkTopology::Grid,
        _ => NetworkTopology::SmallWorld,
    }
}

fn open_browser(url: &str) -> Result<(), String> {
    let cmd = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    std::process::Command::new(cmd)
        .arg(url)
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}
