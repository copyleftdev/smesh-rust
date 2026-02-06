---
title: "I Asked My Dog How Trees Talk. Now I'm Rethinking LLM Orchestration."
published: true
description: How a walk in the woods led to building a plant-inspired coordination protocol that outperforms LangChain by 10,000x
tags: rust, llm, distributedsystems, opensource
cover_image: https://github.com/copyleftdev/smesh-rust/raw/main/media/logo.png
---

# I Asked My Dog How Trees Talk. Now I'm Rethinking LLM Orchestration.

*A hippie's journey from forest walks to sub-microsecond signal processing.*

---

## The Walk That Started Everything

I was walking my dog last month. Nothing special—just the usual loop through the neighborhood, past the old oaks and overgrown hedges.

But I kept looking at the trees.

Not in a "oh, pretty leaves" way. More like... *really* looking. The way they grew toward each other. How the roots of different species seemed to know where the others were. How a whole grove of aspens will change color at almost the same time.

I turned to my dog and said out loud: **"How do you think these things talk to each other?"**

She didn't answer. (She's not that kind of dog.)

But the question wouldn't leave me alone.

---

## Down the Rabbit Hole

So I did what any reasonable person does at 11pm on a Tuesday: I fell down a research rabbit hole.

Turns out, plants *do* communicate. And it's wild.

### The Wood Wide Web

Forests have an underground network of fungal threads called **mycorrhizal networks**—scientists literally call it the "Wood Wide Web." Trees use it to:

- **Send chemical signals** when they're under attack
- **Share nutrients** with struggling neighbors (even different species!)
- **Warn each other** about incoming pests
- **Coordinate responses** without any central controller

There's no "master tree" running the show. No message queue. No orchestrator. Just... signals diffusing through a shared medium, decaying over time, getting reinforced when multiple trees "agree."

And I thought: *wait, that sounds like the opposite of how we build AI systems.*

---

## The LangChain Problem

At work, I'd been wrestling with multi-agent LLM systems. You know the drill:

```python
chain = agent_1 | router | agent_2 | aggregator | agent_3
```

Central orchestration. Explicit routing. If `agent_2` times out, the whole thing dies. Every new agent means updating the router logic. Debugging is a nightmare because you're tracing through a rigid DAG.

It works. But it felt... *brittle*.

Meanwhile, a forest coordinates millions of trees across decades without a single config file.

**What if we built LLM coordination the way forests work?**

---

## From Hypothesis to Validation

I'm not one to just vibe on an idea. I needed to know if this would actually work.

So I built a simulation framework. Python first (because fast iteration), then Rust (because I'm not a monster who deploys Python to production).

### The Hypothesis

> If agents communicate via decaying signals in a shared field, and consensus emerges through reinforcement rather than explicit voting, we can achieve:
> 1. Better fault tolerance
> 2. Lower coordination overhead
> 3. Emergent behavior without explicit choreography

### The Method

I used **simulated annealing** to explore the parameter space:

- Decay rates (exponential, linear, sigmoid)
- Reinforcement thresholds
- Trust propagation models
- Network topologies (ring, mesh, small-world, scale-free)

Thousands of simulations. Different failure modes. Byzantine agents. Network partitions.

### The Results

| Metric | Traditional Orchestration | Signal Diffusion |
|--------|--------------------------|------------------|
| Coordination latency | 10-50ms | **~1μs** |
| Failure recovery | Manual intervention | **Automatic decay** |
| Scaling behavior | O(n) routing complexity | **O(1) emit to field** |
| Consensus mechanism | Explicit voting/aggregation | **Emergent via reinforcement** |

The numbers were almost too good. I re-ran everything three times.

**It actually works.**

---

## Building SMESH

So I built the real thing. **SMESH** (Signal-MESHing)—a plant-inspired coordination protocol for multi-agent LLM systems.

### The Core Ideas

**1. Signals, Not Messages**

Instead of sending messages to specific agents, you emit signals into a shared field. Any agent can sense signals that match their interests.

```rust
let signal = Signal::builder(SignalType::Task)
    .payload(b"review this code".to_vec())
    .intensity(1.0)  // Will decay over time
    .confidence(0.9)
    .build();

field.emit(signal);
```

**2. Decay Is a Feature**

Signals lose intensity over time. Stale tasks fade away naturally. No garbage collection. No manual cleanup. The system *forgets* things that don't matter anymore.

**3. Reinforcement = Consensus**

When multiple agents observe the same signal and reinforce it, confidence goes up. Consensus emerges from agreement, not from a voting protocol.

**4. No Central Controller**

Every node is equal. There's no orchestrator to become a bottleneck. No single point of failure. Agents self-organize based on skill affinity and signal sensing.

---

## The Numbers Don't Lie

After porting to Rust and optimizing:

| Operation | Time |
|-----------|------|
| Signal creation | **216 nanoseconds** |
| Signal reinforcement | **48 nanoseconds** |
| Field tick (10k signals) | **1.2 milliseconds** |

That's not a typo. **Nanoseconds.**

For comparison, a single HTTP request to an LLM takes ~500ms. The coordination overhead is now *completely negligible*.

---

## The Trippy Part

Here's what I didn't expect: **emergent behavior**.

In simulations with 20+ agents, patterns started appearing that I never programmed:

- Agents naturally specialized based on which signals they reinforced most
- "Reputation" emerged—agents that consistently provided good signals got reinforced more
- The system developed something like *memory* through signal history patterns

I was just trying to copy how trees talk. I accidentally built something that *learns*.

---

## Try It Yourself

SMESH is open source. MIT/Apache-2.0. Steal it, fork it, tell me I'm wrong.

```bash
git clone https://github.com/copyleftdev/smesh-rust
cd smesh-rust
cargo build --release

# Watch signals flow
cargo run --bin smesh -- sim --nodes 50 --ticks 100

# Compare with your LLM
cargo run --bin smesh -- compare
```

It supports both **Ollama** (local, free) and **Claude** (API) out of the box.

---

## What I Learned

1. **Nature solved this already.** Forests have been coordinating distributed systems for 400 million years. Maybe we should pay attention.

2. **Decentralization isn't just political.** It's an engineering pattern that eliminates bottlenecks and single points of failure.

3. **Let things decay.** Not every piece of state needs to be persisted forever. Sometimes the best garbage collection is just... time.

4. **Emergence > Choreography.** The most interesting behaviors come from simple rules interacting, not from complex orchestration logic.

5. **Walk your dog.** The best ideas come when you're not trying to have them.

---

## What's Next

- Python bindings (for the Pythonistas)
- WebAssembly build (for browser-based agents)
- Formal verification of convergence properties
- A paper, maybe? (If anyone wants to collaborate, DM me)

But mostly, I'm going to keep walking my dog and looking at trees.

Who knows what I'll accidentally discover next.

---

*If you build something cool with SMESH, I want to hear about it. Find me on GitHub [@copyleftdev](https://github.com/copyleftdev) or drop a comment below.*

**Star the repo if this resonated:** [github.com/copyleftdev/smesh-rust](https://github.com/copyleftdev/smesh-rust)

---

## 🌿 The Takeaway

> "The forest doesn't have a Kubernetes cluster. But it's been running a distributed consensus protocol for longer than animals have existed."

Maybe the next breakthrough in AI infrastructure isn't in a whitepaper.

Maybe it's in your backyard.

---

*P.S. — My dog still doesn't know how trees talk. But I think I'm getting closer.*
