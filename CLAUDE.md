# SMESH - Claude Code Configuration

> Plant-inspired signal diffusion protocol for distributed LLM coordination

## Project Overview

SMESH is a **Rust workspace** implementing a decentralized coordination protocol inspired by mycorrhizal networks (the "Wood Wide Web"). Agents communicate via environmental signals that diffuse, decay, and get reinforced—achieving consensus without central orchestration.

**Key insight:** Signals decay naturally (stale tasks fade), and consensus emerges through reinforcement rather than explicit voting.

## Architecture

```
smesh-rust/
├── smesh-core/      # Signals, nodes, fields, networks (core primitives)
├── smesh-runtime/   # QUIC P2P networking via Quinn
├── smesh-agent/     # LLM backends (OpenRouter + Claude), Constitutional AI
└── smesh-cli/       # CLI tools, benchmarks, code review, threat analysis
```

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Signal** | Message with intensity that decays over time (`smesh-core/src/signal.rs`) |
| **Node** | Entity that emits, senses, reinforces signals (`smesh-core/src/node.rs`) |
| **Field** | Shared space where signals propagate (`smesh-core/src/field.rs`) |
| **Network** | Topology connecting nodes via hyphae (`smesh-core/src/network.rs`) |
| **Trust** | Bayesian trust model for nodes (`smesh-core/src/trust.rs`) |

## Build & Test Commands

```bash
# Build (release optimized with LTO)
cargo build --release

# Run all tests (104 tests)
cargo test --workspace

# Run specific crate tests
cargo test -p smesh-core
cargo test -p smesh-agent

# Check without building
cargo check --workspace

# Lint
cargo clippy --workspace -- -D warnings

# Format
cargo fmt --all
cargo fmt --all -- --check   # CI check

# Run benchmarks
cargo bench -p smesh-core
```

## CLI Commands

```bash
# System status + OpenRouter connection check
cargo run --bin smesh -- status

# Signal simulation (watch emergent behavior)
cargo run --bin smesh -- sim --nodes 50 --ticks 100

# LLM agent coordination demo
cargo run --bin smesh -- agents --demo

# Compare OpenRouter vs Claude backends
cargo run --bin smesh -- compare

# Multi-agent code review
cargo run --bin smesh -- review --path ./some-repo --model qwen2.5-coder:7b

# Threat intelligence analysis
cargo run --bin smesh -- threat --path ./payloads --limit 20
```

## Code Style

- **IMPORTANT:** Use Rust 2021 edition idioms
- Follow existing patterns in the codebase
- All public items require doc comments (`///`)
- Use `thiserror` for error types, `anyhow` for application errors
- Prefer `tracing` over `println!` for logging
- Async code uses `tokio` runtime with `async-trait`
- Serialization via `serde` with `derive` feature

## Testing Guidelines

- Unit tests go in the same file as the code (`#[cfg(test)]` module)
- Integration tests in `tests/` directory
- Use descriptive test names: `test_signal_decay_exponential`
- **Run single tests for speed:** `cargo test test_name`
- Tests should be deterministic (seed RNG if needed)

## Key Files

| Purpose | File |
|---------|------|
| Signal primitives | `smesh-core/src/signal.rs` |
| Network topology | `smesh-core/src/network.rs` |
| LLM backend trait | `smesh-agent/src/backend.rs` |
| Claude client | `smesh-agent/src/claude.rs` |
| OpenRouter client | `smesh-agent/src/openrouter.rs` |
| Constitutional AI | `smesh-agent/src/constitutional.rs` |
| CLI entry point | `smesh-cli/src/main.rs` |

## Performance Targets

| Operation | Target |
|-----------|--------|
| Signal creation | < 300 ns |
| Signal reinforcement | < 100 ns |
| 100-node network tick | < 20 μs |

**IMPORTANT:** Run `cargo bench` to verify performance after changes to core primitives.

## Dependencies

- **Async:** `tokio`, `futures`, `async-trait`
- **Crypto:** `sha2`, `rand`, `uuid`
- **Networking:** `quinn` (QUIC), `rustls`
- **Math:** `nalgebra`
- **HTTP:** `reqwest` (for the OpenRouter / Claude APIs)
- **CLI:** `clap` with derive

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Required for Claude backend |
| `OPENROUTER_API_KEY` | Required for OpenRouter backend |
| `OPENROUTER_BASE_URL` | OpenRouter API base (default: `https://openrouter.ai/api/v1`) |
| `OPENROUTER_MODEL` | Default model slug (default: `google/gemini-2.5-flash-lite`) |
| `OPENROUTER_ENV_FILE` | Optional path to a dotenv file for the above (falls back to `~/.creds/openrouter.env`) |
| `RUST_LOG` | Tracing filter (e.g., `smesh=debug`) |

## Workflow Notes

1. **Before committing:** Run `cargo fmt --all` and `cargo clippy --workspace`
2. **After core changes:** Run benchmarks to verify no regression
3. **For LLM features:** Test with both OpenRouter (Gemini) and Claude (API)
4. **Branch naming:** `feature/`, `fix/`, `refactor/`

## Common Patterns

### Creating a Signal
```rust
use smesh_core::{Signal, SignalType};

let signal = Signal::builder(SignalType::Coordination)
    .payload(b"task data".to_vec())
    .intensity(1.0)
    .ttl(60.0)
    .build();
```

### Using LLM Backend
```rust
use smesh_agent::{OpenRouterClient, LlmBackend, GenerateRequest};

// Credentials resolved from env or ~/.creds/openrouter.env
let client = OpenRouterClient::from_env().expect("OPENROUTER_API_KEY not set");
let response = client.generate(GenerateRequest::new("Hello")).await?;
println!("{}", response.content);
```

## License

Dual-licensed: MIT OR Apache-2.0
