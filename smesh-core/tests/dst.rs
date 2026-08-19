//! Deterministic simulation of gossip convergence.
//!
//! This drives the real protocol code — `Signal::merge_attestations`,
//! `Node::relay_decision_with`, real Ed25519 attestations — under a simulated
//! network and scheduler. Only the network and the coin flips are fake, and
//! both come from one seed, so a failure is reproduced by its seed rather than
//! described in a bug report. A simulation that reimplements the protocol
//! proves nothing about the protocol, so nothing here reimplements it.
//!
//! Two claims are under test, and the mesh layer rests on both:
//!
//! - **Termination.** Forwarding only when local knowledge grew has to stop.
//!   Attester sets are grow-only over a finite set of nodes, so state changes
//!   are bounded — but that is an argument, and arguments are what simulations
//!   are for.
//! - **Convergence.** With delivery, every node agrees on who attested,
//!   whatever the order, delay, or losses along the way.
//!
//! Cost is bounded deliberately. The default seed count keeps this inside a
//! normal `cargo test`; `SMESH_DST_SEEDS` raises it for a soak run, which
//! belongs under `verify/` where the resource ceiling applies.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use std::sync::OnceLock;

use smesh_core::{Attestation, Node, NodeIdentity, Signal, SignalType};

/// Keys are expensive to make and the properties under test do not depend on
/// which keys they are, only that signatures verify. Generating them once keeps
/// a seed sweep affordable.
fn identities(count: usize) -> &'static [NodeIdentity] {
    static POOL: OnceLock<Vec<NodeIdentity>> = OnceLock::new();
    let pool = POOL.get_or_init(|| {
        (0..16)
            .map(|i| NodeIdentity::generate_named(format!("n{i}")))
            .collect()
    });
    &pool[..count]
}

/// Seeds explored when nothing says otherwise. Deliberately small: this runs on
/// every `cargo test`, and a verification tool that makes the normal loop
/// painful is a verification tool people turn off. `SMESH_DST_SEEDS` raises it.
const DEFAULT_SEEDS: u64 = 12;

/// Anti-entropy rounds allowed before convergence is called a failure.
const MAX_ANTI_ENTROPY_ROUNDS: usize = 12;
/// Hard stop per simulation, so a livelock fails loudly instead of hanging.
const MAX_STEPS: usize = 20_000;

fn seed_count() -> u64 {
    std::env::var("SMESH_DST_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_SEEDS)
}

/// Seeded generator, written out so a seed means the same thing forever.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

/// One message in flight.
struct InFlight {
    to: usize,
    attestations: Vec<Attestation>,
    /// Simulated arrival time. Lower is delivered sooner.
    due: u64,
}

/// One run of the protocol under one seed.
struct Sim {
    rng: Rng,
    nodes: Vec<Node>,
    identities: &'static [NodeIdentity],
    /// Each node's own copy of the single claim under test.
    held: Vec<Signal>,
    links: Vec<Vec<usize>>,
    queue: VecDeque<InFlight>,
    clock: u64,
    /// Fraction of messages the network loses outright.
    loss: f64,
    forwarded: usize,
}

impl Sim {
    fn new(seed: u64, node_count: usize, loss: f64) -> Self {
        let mut rng = Rng::new(seed);

        let nodes: Vec<Node> = (0..node_count)
            .map(|i| {
                let mut node = Node::named(format!("n{i}"));
                // Trust everyone, so relaying is driven by the simulated draw
                // rather than by trust bottoming out.
                for j in 0..node_count {
                    node.trust_scores.insert(format!("n{j}"), 0.9);
                }
                node
            })
            .collect();

        // Ring plus a chord: connected, but not everyone adjacent, so messages
        // genuinely have to be relayed to cross it.
        let mut links = vec![Vec::new(); node_count];
        for i in 0..node_count {
            let next = (i + 1) % node_count;
            links[i].push(next);
            links[next].push(i);
        }
        if node_count > 3 {
            let far = node_count / 2;
            links[0].push(far);
            links[far].push(0);
        }

        let held = (0..node_count)
            .map(|_| {
                Signal::builder(SignalType::Alert)
                    .correlatable()
                    .payload(b"the claim".to_vec())
                    .intensity(1.0)
                    .confidence(0.9)
                    .radius(8)
                    .build()
            })
            .collect();

        let loss = loss * rng.unit();

        Self {
            rng,
            nodes,
            identities: identities(node_count),
            held,
            links,
            queue: VecDeque::new(),
            clock: 0,
            loss,
            forwarded: 0,
        }
    }

    /// A node asserts the claim and tells its neighbours.
    fn assert_claim(&mut self, node: usize) {
        let attestation = self.identities[node].attest(&self.held[node].origin_hash);
        self.held[node].merge_attestations(&[attestation]);
        self.gossip_from(node);
    }

    fn gossip_from(&mut self, from: usize) {
        let attestations = self.held[from].attestations.clone();
        for to in self.links[from].clone() {
            if self.rng.unit() < self.loss {
                continue;
            }
            let due = self.clock + 1 + self.rng.next_u64() % 8;
            self.queue.push_back(InFlight {
                to,
                attestations: attestations.clone(),
                due,
            });
        }
    }

    /// Deliver one message. Returns false when nothing is left.
    fn step(&mut self) -> bool {
        if self.queue.is_empty() {
            return false;
        }

        let ready: Vec<usize> = self
            .queue
            .iter()
            .enumerate()
            .filter(|(_, m)| m.due <= self.clock)
            .map(|(i, _)| i)
            .collect();

        if ready.is_empty() {
            self.clock += 1;
            return true;
        }

        let index = ready[self.rng.below(ready.len())];
        let msg = self.queue.remove(index).expect("index came from the queue");

        // The real merge rule, under test.
        let grew = !self.held[msg.to]
            .merge_attestations(&msg.attestations)
            .is_empty();

        if grew {
            let roll = self.rng.unit();
            let signal = self.held[msg.to].clone();
            let remaining = signal.radius.saturating_sub(signal.hops);
            if self.nodes[msg.to].would_relay(&signal, remaining, roll) {
                self.forwarded += 1;
                self.gossip_from(msg.to);
            }
        }

        true
    }

    fn run(&mut self) -> usize {
        let mut steps = 0;
        while self.step() {
            steps += 1;
            assert!(
                steps < MAX_STEPS,
                "gossip did not terminate: forward-iff-changed should make state \
                 changes finite, but this schedule kept producing them"
            );
        }
        steps
    }

    /// Re-announce what every asserting node currently holds.
    ///
    /// The real nodes do this on a timer. Without it a declined relay is
    /// permanent, because nothing ever offers that information again.
    fn anti_entropy_round(&mut self, asserters: &[usize]) {
        for &node in asserters {
            self.gossip_from(node);
        }
        self.run();
    }

    /// Every node re-announces what it holds, not only the originators.
    fn full_anti_entropy_round(&mut self) {
        for node in 0..self.nodes.len() {
            if !self.held[node].attestations.is_empty() {
                self.gossip_from(node);
            }
        }
        self.run();
    }

    fn converged(&self) -> bool {
        let first = self.attesters(0);
        (1..self.nodes.len()).all(|n| self.attesters(n) == first)
    }

    fn attesters(&self, node: usize) -> BTreeSet<String> {
        self.held[node].verified_attesters().into_iter().collect()
    }
}

#[test]
fn gossip_terminates_under_every_schedule() {
    for seed in 0..seed_count() {
        let mut sim = Sim::new(seed, 6, 0.3);
        sim.assert_claim(0);
        sim.assert_claim(3);
        let steps = sim.run();
        assert!(steps < MAX_STEPS, "seed {seed} failed to settle");
    }
}

#[test]
fn a_single_gossip_round_does_not_guarantee_convergence() {
    // Worth stating outright, because it is easy to assume otherwise and the
    // simulation found it immediately. Relaying is a coin flip, so information
    // dies at any node that declines to forward — even with a perfect network.
    // Convergence is not a property of gossip here; it is a property of gossip
    // plus anti-entropy, and the next test is the one that holds.
    let mut ever_diverged = false;

    for seed in 0..seed_count() {
        let mut sim = Sim::new(seed, 6, 0.0);
        for node in [0usize, 2, 5] {
            sim.assert_claim(node);
        }
        sim.run();
        if !sim.converged() {
            ever_diverged = true;
            break;
        }
    }

    assert!(
        ever_diverged,
        "a declined relay should be able to strand information without \
         anti-entropy; if this no longer happens, relaying stopped being \
         probabilistic and the anti-entropy test below is no longer meaningful"
    );
}

#[test]
fn re_announcing_from_originators_alone_is_not_enough() {
    // This is why anti-entropy is every holder's job and not just the
    // originators'. Forward-iff-changed silences a node once it already knows
    // something, so a neighbour behind it never hears the claim again — and the
    // originator re-announcing does not help, because the silent node is in the
    // way. Locked in as a test so the fix cannot be "simplified" back.
    let mut originator_only_failed = false;

    for seed in 0..seed_count() {
        let asserters = [0usize, 2, 5];
        let mut sim = Sim::new(seed, 6, 0.0);
        for node in asserters {
            sim.assert_claim(node);
        }
        sim.run();

        for _ in 0..MAX_ANTI_ENTROPY_ROUNDS {
            if sim.converged() {
                break;
            }
            sim.anti_entropy_round(&asserters);
        }

        if !sim.converged() {
            originator_only_failed = true;
            break;
        }
    }

    assert!(
        originator_only_failed,
        "originator-only re-announcement converged on every seed tried. Either \
         the topology no longer has a node that can be stranded, or relaying \
         stopped being probabilistic — either way the anti-entropy design needs \
         revisiting rather than this test being deleted."
    );
}

#[test]
fn anti_entropy_converges_every_node() {
    // The real property. Re-announcing what you hold gives every gap another
    // chance, so agreement is reached despite declined relays, reordering and
    // delay. Order changes how long it takes, not what is agreed.
    let mut failures = Vec::new();

    for seed in 0..seed_count() {
        let asserters = [0usize, 2, 5];
        let mut sim = Sim::new(seed, 6, 0.0);
        for node in asserters {
            sim.assert_claim(node);
        }
        sim.run();

        // Re-announcing from the originators alone is not enough: a gap behind
        // a node that already knows is never offered the missing information
        // again, because forward-iff-changed means that node stays silent.
        // Every holder has to re-announce.
        let mut rounds = 0;
        while !sim.converged() && rounds < MAX_ANTI_ENTROPY_ROUNDS {
            sim.full_anti_entropy_round();
            rounds += 1;
        }
        let _ = &asserters;

        if !sim.converged() {
            failures.push((
                seed,
                sim.attesters(0),
                (1..6).map(|n| sim.attesters(n)).collect::<Vec<_>>(),
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "did not converge within {MAX_ANTI_ENTROPY_ROUNDS} anti-entropy rounds; first: {:?}",
        failures.first()
    );
}

#[test]
fn attester_sets_only_ever_grow() {
    // Convergence rests on the set being grow-only. If a merge can remove an
    // attester, ordering starts to matter and the argument collapses.
    for seed in 0..seed_count() {
        let mut sim = Sim::new(seed, 5, 0.2);
        let mut high_water: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();

        sim.assert_claim(1);
        sim.assert_claim(4);

        let mut steps = 0;
        while sim.step() {
            steps += 1;
            assert!(steps < MAX_STEPS);
            for node in 0..sim.nodes.len() {
                let now = sim.attesters(node);
                let seen = high_water.entry(node).or_default();
                assert!(
                    seen.is_subset(&now),
                    "seed {seed}: node {node} lost an attester it already had"
                );
                *seen = now;
            }
        }
    }
}

#[test]
fn loss_delays_agreement_without_corrupting_it() {
    // Under loss a node may know less. It must never know something false:
    // every attester reported has to be one that really signed.
    for seed in 0..seed_count() {
        let mut sim = Sim::new(seed, 6, 0.6);
        sim.assert_claim(0);
        sim.assert_claim(2);
        sim.run();

        let truthful: BTreeSet<String> = ["n0", "n2"].iter().map(|s| s.to_string()).collect();
        for node in 0..sim.nodes.len() {
            let seen = sim.attesters(node);
            assert!(
                seen.is_subset(&truthful),
                "seed {seed}: node {node} reported an attester nobody signed: {seen:?}"
            );
        }
    }
}
