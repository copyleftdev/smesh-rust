# SMESH Rust Implementation

**A high-performance, decentralized coordination protocol for multi-agent LLM systems.**

## What Is SMESH?

SMESH (Signal-MESHing) is a **distributed coordination layer** for orchestrating multiple LLM agents without a central controller. Instead of traditional request-response orchestration (like LangChain's chains or AutoGPT's loops), SMESH uses a **signal diffusion model** inspired by plant communication:

### The Problem We Solve

Traditional multi-agent LLM orchestration has issues:
- **Central bottleneck**: One orchestrator routes all messages
- **Tight coupling**: Agents depend on specific APIs and protocols
- **No emergent behavior**: Fixed workflows, no adaptive coordination
- **Single point of failure**: Orchestrator dies, system dies

### Our Approach

SMESH treats agent communication like **chemical signals in a forest**:

```
┌─────────────────────────────────────────────────────────────┐
│                     SIGNAL FIELD                            │
│                                                             │
│   Agent A emits:        Signals decay        Agent B senses │
│   "Need code review"    over time            matching tasks │
│        ↓                   ↓                      ↓         │
│      ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████          │
│      intensity=1.0      intensity=0.5      claims task      │
│                                                             │
│   Multiple agents        Reinforcement       Consensus      │
│   observe same signal    increases trust     emerges        │
└─────────────────────────────────────────────────────────────┘
```

- **Signals diffuse** through a shared field (no point-to-point messages)
- **Signals decay** over time (stale tasks fade away)
- **Signals get reinforced** when multiple agents agree (consensus emerges)
- **No central orchestrator** — agents self-organize

## Architecture

```
smesh-rust/
├── smesh-core/      # Core primitives (Signal, Node, Field, Network)  ✓
├── smesh-runtime/   # QUIC P2P networking (quinn-based)               ✓
├── smesh-agent/     # LLM backends (Ollama + Claude)                  ✓
└── smesh-cli/       # CLI tools, benchmarks, comparison               ✓
```

### Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Signal Processing** | ✓ Complete | Decay functions, reinforcement, propagation |
| **Network Topologies** | ✓ Complete | Ring, Mesh, SmallWorld, ScaleFree, Grid |
| **QUIC P2P Transport** | ✓ Complete | Encrypted peer-to-peer via quinn |
| **Ollama Backend** | ✓ Complete | Local LLM inference |
| **Claude Backend** | ✓ Complete | Anthropic API integration |
| **Signal DNA** | ✓ Complete | Attribution fingerprinting |
| **CLI Tooling** | ✓ Complete | Status, sim, agents, compare, bench |

## Performance Benchmarks

Measured on typical hardware (AMD Ryzen / Intel i7, 32GB RAM):

### Signal Processing (smesh-core)

| Operation | Time | Throughput |
|-----------|------|------------|
| Signal creation | **216 ns** | 4.6M signals/sec |
| Signal hash | **235 ps** | 4.2B hashes/sec |
| Effective intensity | **16 ns** | 62M ops/sec |
| Signal reinforcement | **48 ns** | 20M ops/sec |
| Field emit (1000 signals) | **414 μs** | 2.4M signals/sec |
| Field tick (10000 signals) | **1.2 ms** | 8.3M signals/sec |

### Network Operations

| Operation | 50 nodes | 100 nodes | 500 nodes |
|-----------|----------|-----------|-----------|
| Ring topology creation | 32 μs | 63 μs | ~300 μs |
| Small-world creation | 51 μs | 98 μs | ~500 μs |
| Full mesh creation | 336 μs | 1.3 ms | ~30 ms |
| Network tick | 13 μs | 14 μs | ~70 μs |
| Stats computation | 1.1 μs | 2.2 μs | 12 μs |

### LLM Backend Comparison

| Backend | Latency | Tokens/sec | Cost | Privacy |
|---------|---------|------------|------|---------|
| **Ollama** (local, qwen2.5-coder:7b) | 0.1-1s | N/A | Free | ✓ Local |
| **Claude** (API, claude-sonnet-4) | 1-4s | 25-45 | API charges | Cloud |

```bash
# Run LLM comparison yourself
ANTHROPIC_API_KEY=... cargo run --bin smesh -- compare
```

---

## Comparison to Other Stacks

### vs LangChain / LangGraph

| Aspect | LangChain | SMESH |
|--------|-----------|-------|
| **Architecture** | Central chain orchestrator | Decentralized signal field |
| **Language** | Python | Rust (with Python bindings planned) |
| **Coordination** | Explicit DAG/chain | Emergent via signal diffusion |
| **Latency overhead** | ~10-50ms per chain step | ~1μs per signal operation |
| **Memory** | High (Python objects) | Low (zero-copy where possible) |
| **Multi-agent** | Requires explicit routing | Self-organizing |
| **Fault tolerance** | Chain breaks if step fails | Signals decay, others continue |

### vs AutoGPT / CrewAI

| Aspect | AutoGPT/CrewAI | SMESH |
|--------|----------------|-------|
| **Control flow** | Loops with explicit roles | Signal sensing with skill affinity |
| **Scaling** | Linear with agent count | Sublinear (shared field) |
| **Consensus** | Voting/aggregation | Signal reinforcement (emergent) |
| **Customization** | Role prompts | Trust models + decay functions |

### vs Ray / Dask (Distributed Computing)

| Aspect | Ray/Dask | SMESH |
|--------|----------|-------|
| **Focus** | General distributed compute | Agent coordination specifically |
| **Communication** | Task queues, actors | Signal diffusion |
| **State** | Distributed objects | Shared signal field |
| **Use case** | Data processing, ML training | Multi-LLM orchestration |

---

## Pros and Cons

### ✓ Advantages

| Advantage | Why It Matters |
|-----------|----------------|
| **No central bottleneck** | Scales horizontally without coordinator limits |
| **Fault tolerant** | Agent failure = signals decay, others continue |
| **Emergent consensus** | No explicit voting; agreement via reinforcement |
| **Sub-microsecond signals** | Coordination overhead doesn't dominate LLM latency |
| **Memory efficient** | ~10x less than Python equivalents |
| **Single binary** | Deploy one executable, no dependency hell |
| **Multiple LLM backends** | Ollama (local) + Claude (API) unified interface |
| **QUIC networking** | Encrypted P2P without TLS certificate management |

### ✗ Disadvantages

| Disadvantage | Mitigation |
|--------------|------------|
| **Novel paradigm** | Learning curve; provide good docs and examples |
| **Less ecosystem** | Not as many integrations as LangChain (yet) |
| **Rust complexity** | Provide Python bindings for easier adoption |
| **Eventual consistency** | Not suitable for strict ordering requirements |
| **Debugging difficulty** | Emergent behavior harder to trace than explicit chains |

### When to Use SMESH

**Good fit:**
- Multi-agent LLM systems with 3+ agents
- Systems requiring fault tolerance
- Scenarios where consensus should emerge naturally
- High-throughput coordination (many signals/sec)
- Edge deployment (single binary, low memory)

**Not ideal for:**
- Simple single-agent chatbots
- Strict sequential workflows
- Systems requiring exactly-once delivery guarantees
- Teams unfamiliar with Rust (until Python bindings ready)

---

## Core Concepts

### Signal
The fundamental message type. Signals have:
- **Intensity**: Strength that decays over time (exponential, linear, sigmoid)
- **TTL**: Time to live before expiration
- **Confidence**: Sender's belief in the signal
- **Reinforcement**: Count of corroborating observations

### Node
An entity in the network that can:
- Emit signals into the field
- Sense nearby signals (based on sensitivity threshold)
- Reinforce signals with local evidence
- Maintain trust relationships with other nodes

### Field
The shared space where signals exist:
- Stores active signals (hash-deduplicated)
- Handles decay and expiration each tick
- Manages signal reinforcement

### Network
Topology connecting nodes via hyphae (directed edges):
- **Ring**: Simple circular topology
- **FullMesh**: Every node connected to every other
- **SmallWorld**: Watts-Strogatz (clustered with shortcuts)
- **ScaleFree**: Barabási-Albert (hub-and-spoke)
- **Grid**: 2D lattice
- **Random**: Erdős-Rényi

---

## Quick Start

```bash
# Build
cargo build --release

# Check status
cargo run --bin smesh -- status

# Run simulation
cargo run --bin smesh -- sim --nodes 50 --topology small_world --ticks 100

# Run LLM agent demo (requires Ollama)
ollama serve &
cargo run --bin smesh -- agents --demo

# Compare LLM backends
ANTHROPIC_API_KEY=... cargo run --bin smesh -- compare

# Run benchmarks
cargo bench -p smesh-core
```

## License

MIT OR Apache-2.0
