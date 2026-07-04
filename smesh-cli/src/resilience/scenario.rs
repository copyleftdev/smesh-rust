//! One resilience experiment: build a mesh, inflict an attack, and measure
//! whether emergent consensus survives.
//!
//! All experiments are pure, LLM-free Rust over the real `smesh_core` engine,
//! so they run in milliseconds and can be swept and averaged.

use rand::seq::SliceRandom;
use serde_json::json;
use smesh_core::{MaliciousBehavior, Network, NetworkTopology, Signal, SignalType};

/// The adversarial condition applied to the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attack {
    /// A fraction of nodes crash (all their links go dark).
    NodeFailure,
    /// A fraction of nodes become eclipse attackers (black-hole all traffic).
    Eclipse,
    /// A fraction of nodes are Byzantine, reinforcing a competing false signal.
    Byzantine,
    /// The network splits in two, then heals (measures recovery, not a curve).
    Partition,
}

impl Attack {
    pub fn code(&self) -> &'static str {
        match self {
            Attack::NodeFailure => "node_failure",
            Attack::Eclipse => "eclipse",
            Attack::Byzantine => "byzantine",
            Attack::Partition => "partition",
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            Attack::NodeFailure => "Node Failure",
            Attack::Eclipse => "Eclipse Attack",
            Attack::Byzantine => "Byzantine Reinforce",
            Attack::Partition => "Partition + Heal",
        }
    }
    pub fn metric(&self) -> &'static str {
        match self {
            Attack::Byzantine => "consensus integrity",
            _ => "coverage fidelity",
        }
    }
}

/// Parameters shared across a resilience run.
#[derive(Debug, Clone)]
pub struct ScenarioConfig {
    pub nodes: usize,
    pub topology: NetworkTopology,
    pub ticks: usize,
    pub trials: usize,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            nodes: 60,
            topology: NetworkTopology::SmallWorld,
            ticks: 40,
            trials: 8,
        }
    }
}

const SENSE_THRESHOLD: f64 = 0.1;

/// Average the fidelity metric over `trials` for one attack at intensity `frac`.
pub fn measure(attack: Attack, frac: f64, cfg: &ScenarioConfig) -> f64 {
    let mut sum = 0.0;
    for _ in 0..cfg.trials {
        sum += one_trial(attack, frac, cfg);
    }
    sum / cfg.trials.max(1) as f64
}

/// Run a single trial and return its fidelity (0..1).
pub fn one_trial(attack: Attack, frac: f64, cfg: &ScenarioConfig) -> f64 {
    let mut net = Network::with_topology(cfg.nodes, cfg.topology);
    let ids: Vec<String> = net.nodes.keys().cloned().collect();
    let mut rng = rand::thread_rng();

    if attack == Attack::Byzantine {
        return byzantine_trial(&net, &ids, frac, &mut rng);
    }

    // Pick the victim set.
    let k = ((ids.len() as f64) * frac).round() as usize;
    let mut shuffled = ids.clone();
    shuffled.shuffle(&mut rng);
    let victims: Vec<String> = shuffled.into_iter().take(k).collect();

    match attack {
        Attack::NodeFailure => isolate_nodes(&mut net, &victims),
        Attack::Eclipse => {
            for v in &victims {
                if let Some(n) = net.get_node_mut(v) {
                    n.make_malicious(MaliciousBehavior::Eclipse);
                }
            }
        }
        _ => {}
    }

    // Honest, still-alive nodes.
    let honest: Vec<String> = ids
        .iter()
        .filter(|id| !victims.contains(id))
        .cloned()
        .collect();
    if honest.is_empty() {
        return 0.0;
    }

    coverage_fidelity(&mut net, &honest, cfg.ticks)
}

/// Inject a true signal at an honest origin, diffuse, and return the fraction of
/// honest nodes it reaches with sensible intensity.
fn coverage_fidelity(net: &mut Network, honest: &[String], ticks: usize) -> f64 {
    let origin = honest[0].clone();
    let mut sig = Signal::builder(SignalType::Data)
        .payload(b"true-consensus".to_vec())
        .intensity(1.0)
        .confidence(1.0)
        .ttl(1000.0)
        .radius(50)
        .origin(&origin)
        .build();
    sig.mark_reached(&origin);
    net.field.emit_anonymous(sig);

    for _ in 0..ticks {
        net.tick(0.25);
        if net.field.signals.is_empty() {
            break;
        }
    }

    let (reached, intensity) = net
        .field
        .signals
        .values()
        .next()
        .map(|s| (s.reached_nodes.clone(), s.current_intensity))
        .unwrap_or_default();

    if intensity < SENSE_THRESHOLD {
        return 0.0; // signal decayed below the sensing floor everywhere
    }
    let hit = honest.iter().filter(|h| reached.contains(h)).count();
    hit as f64 / honest.len() as f64
}

/// Byzantine trial: honest nodes reinforce the true signal, Byzantine nodes
/// reinforce a competing false signal. Integrity = the honest signal remains
/// the stronger consensus.
fn byzantine_trial(
    net: &Network,
    ids: &[String],
    frac: f64,
    rng: &mut rand::rngs::ThreadRng,
) -> f64 {
    let k = ((ids.len() as f64) * frac).round() as usize;
    let mut shuffled = ids.to_vec();
    shuffled.shuffle(rng);
    let byz: Vec<String> = shuffled.iter().take(k).cloned().collect();
    let honest: Vec<&String> = ids.iter().filter(|id| !byz.contains(id)).collect();
    if honest.is_empty() {
        return 0.0;
    }
    let _ = net; // topology already built; consensus here is reinforcement-based

    // True signal reinforced by every honest node.
    let mut truth = Signal::builder(SignalType::Data)
        .payload(b"true-value".to_vec())
        .intensity(1.0)
        .confidence(0.9)
        .build();
    for h in &honest {
        truth.reinforce(h);
    }

    // False signal reinforced by every Byzantine node.
    let mut lie = Signal::builder(SignalType::Data)
        .payload(b"false-value".to_vec())
        .intensity(1.0)
        .confidence(0.9)
        .build();
    for b in &byz {
        lie.reinforce(b);
    }

    // Consensus strength is the count of independent reinforcers (effective
    // intensity clamps to 1.0, so it can't separate the two). The honest signal
    // holds only while it out-numbers the Byzantine one — reinforcement is an
    // unweighted majority, so this breaks at ~50%.
    if truth.reinforcement_count > lie.reinforcement_count {
        1.0
    } else {
        0.0
    }
}

/// Isolate nodes by deactivating every hypha incident to them.
fn isolate_nodes(net: &mut Network, victims: &[String]) {
    let vset: std::collections::HashSet<&String> = victims.iter().collect();
    for (from, hyphae) in net.hyphae.iter_mut() {
        let from_dead = vset.contains(from);
        for h in hyphae.iter_mut() {
            if from_dead || vset.contains(&h.to) {
                h.active = false;
            }
        }
    }
}

// ── Partition recovery ──────────────────────────────────────────────────────

/// Measure how long the mesh takes to re-cover after a 50/50 partition heals.
/// Returns (ticks_to_recover, fidelity_during_partition, fidelity_after_heal).
pub fn partition_recovery(cfg: &ScenarioConfig) -> (usize, f64, f64) {
    let mut sum_recover = 0usize;
    let mut sum_during = 0.0;
    let mut sum_after = 0.0;
    for _ in 0..cfg.trials {
        let (r, d, a) = partition_trial(cfg);
        sum_recover += r;
        sum_during += d;
        sum_after += a;
    }
    let t = cfg.trials.max(1);
    (sum_recover / t, sum_during / t as f64, sum_after / t as f64)
}

fn partition_trial(cfg: &ScenarioConfig) -> (usize, f64, f64) {
    let mut net = Network::with_topology(cfg.nodes, cfg.topology);
    let ids: Vec<String> = net.nodes.keys().cloned().collect();
    let half = ids.len() / 2;
    let near: std::collections::HashSet<&String> = ids[..half].iter().collect();

    // Remember which hyphae we cut so we can heal exactly those.
    let mut cut: Vec<(String, usize)> = Vec::new();
    for (from, hyphae) in net.hyphae.iter_mut() {
        for (i, h) in hyphae.iter_mut().enumerate() {
            if near.contains(from) != near.contains(&h.to) {
                h.active = false;
                cut.push((from.clone(), i));
            }
        }
    }

    // Inject on the near side; measure far-side reach while partitioned.
    let origin = ids[0].clone();
    let mut sig = Signal::builder(SignalType::Data)
        .payload(b"true-consensus".to_vec())
        .intensity(1.0)
        .confidence(1.0)
        .ttl(1000.0)
        .radius(50)
        .origin(&origin)
        .build();
    sig.mark_reached(&origin);
    net.field.emit_anonymous(sig);
    for _ in 0..cfg.ticks {
        net.tick(0.25);
    }
    let far: Vec<String> = ids[half..].to_vec();
    let during = frac_reached(&net, &far);

    // Heal the partition.
    for (from, i) in &cut {
        if let Some(hyphae) = net.hyphae.get_mut(from) {
            if let Some(h) = hyphae.get_mut(*i) {
                h.active = true;
            }
        }
    }
    // Count ticks until the far side is (nearly) covered again.
    let mut recover = cfg.ticks;
    for t in 1..=cfg.ticks {
        net.tick(0.25);
        if frac_reached(&net, &far) >= 0.9 {
            recover = t;
            break;
        }
    }
    let after = frac_reached(&net, &far);
    (recover, during, after)
}

fn frac_reached(net: &Network, group: &[String]) -> f64 {
    if group.is_empty() {
        return 0.0;
    }
    let reached = net
        .field
        .signals
        .values()
        .next()
        .map(|s| s.reached_nodes.clone())
        .unwrap_or_default();
    let hit = group.iter().filter(|g| reached.contains(g)).count();
    hit as f64 / group.len() as f64
}

// ── Single scenario trace (for the interactive Chaos tab) ───────────────────

/// Run one scenario and return the graph, victim set, and per-tick diffusion
/// frames, so the browser can animate the attack in real time.
pub fn scenario_trace(attack: Attack, frac: f64, cfg: &ScenarioConfig) -> serde_json::Value {
    let mut net = Network::with_topology(cfg.nodes, cfg.topology);
    let ids: Vec<String> = net.nodes.keys().cloned().collect();
    let mut rng = rand::thread_rng();

    let k = ((ids.len() as f64) * frac).round() as usize;
    let mut shuffled = ids.clone();
    shuffled.shuffle(&mut rng);
    let victims: Vec<String> = shuffled.into_iter().take(k).collect();

    let mut partition_near: Vec<String> = Vec::new();
    match attack {
        Attack::NodeFailure => isolate_nodes(&mut net, &victims),
        Attack::Eclipse => {
            for v in &victims {
                if let Some(n) = net.get_node_mut(v) {
                    n.make_malicious(MaliciousBehavior::Eclipse);
                }
            }
        }
        Attack::Partition => {
            let half = ids.len() / 2;
            partition_near = ids[..half].to_vec();
            let near: std::collections::HashSet<&String> = ids[..half].iter().collect();
            for (from, hyphae) in net.hyphae.iter_mut() {
                for h in hyphae.iter_mut() {
                    if near.contains(from) != near.contains(&h.to) {
                        h.active = false;
                    }
                }
            }
        }
        Attack::Byzantine => {}
    }

    // Edge list (only currently-active hyphae, so the split/failed links vanish).
    let mut seen = std::collections::HashSet::new();
    let mut edges = Vec::new();
    for (from, hs) in &net.hyphae {
        for h in hs {
            if !h.active {
                continue;
            }
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

    // Diffuse a true signal from an honest node.
    let origin = ids
        .iter()
        .find(|id| !victims.contains(id))
        .cloned()
        .unwrap_or_else(|| ids[0].clone());
    let mut sig = Signal::builder(SignalType::Data)
        .payload(b"true-consensus".to_vec())
        .intensity(1.0)
        .confidence(1.0)
        .ttl(1000.0)
        .radius(50)
        .origin(&origin)
        .build();
    sig.mark_reached(&origin);
    net.field.emit_anonymous(sig);

    let mut frames = vec![frame_snapshot(&net)];
    for _ in 0..cfg.ticks {
        net.tick(0.25);
        frames.push(frame_snapshot(&net));
        if net.field.signals.is_empty() {
            break;
        }
    }

    let honest: Vec<String> = ids.iter().filter(|id| !victims.contains(id)).cloned().collect();
    let fidelity = frac_reached(&net, &honest);

    json!({
        "attack": attack.code(),
        "intensity": frac,
        "nodes": ids,
        "edges": edges,
        "origin": origin,
        "victims": victims,
        "partition_near": partition_near,
        "frames": frames,
        "fidelity": fidelity,
    })
}

fn frame_snapshot(net: &Network) -> serde_json::Value {
    let signals: Vec<serde_json::Value> = net
        .field
        .signals
        .values()
        .map(|s| {
            json!({
                "reached": s.reached_nodes,
                "intensity": s.current_intensity,
                "hops": s.hops,
            })
        })
        .collect();
    json!({ "signals": signals })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ScenarioConfig {
        ScenarioConfig {
            nodes: 40,
            topology: NetworkTopology::SmallWorld,
            ticks: 40,
            trials: 3,
        }
    }

    #[test]
    fn test_no_attack_keeps_full_coverage() {
        // With no failures the true signal should reach essentially every node.
        let f = measure(Attack::NodeFailure, 0.0, &cfg());
        assert!(f > 0.9, "no attack should keep near-full coverage, got {f}");
    }

    #[test]
    fn test_byzantine_is_majority_vote() {
        // Honest majority holds; honest minority loses (unweighted reinforcement).
        assert_eq!(measure(Attack::Byzantine, 0.1, &cfg()), 1.0);
        assert_eq!(measure(Attack::Byzantine, 0.7, &cfg()), 0.0);
    }

    #[test]
    fn test_partition_recovers_after_heal() {
        let (recover, during, after) = partition_recovery(&cfg());
        // The far side is cut off while split, then re-covered once healed.
        assert!(during < 0.2, "far side should be cut off, got {during}");
        assert!(after > during, "healing should restore coverage");
        assert!(recover >= 1, "recovery takes at least one tick");
    }
}
