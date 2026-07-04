//! Resilience / Byzantine-fault benchmark: sweep the mesh under attack and
//! chart how emergent consensus degrades — testing SMESH's fault-tolerance
//! claims directly.

pub mod report;
pub mod scenario;

use std::time::Instant;

use scenario::{Attack, ScenarioConfig};
use smesh_core::NetworkTopology;

/// Configuration for a resilience run.
#[derive(Debug, Clone)]
pub struct ResilienceConfig {
    pub nodes: usize,
    pub topology: NetworkTopology,
    pub ticks: usize,
    pub trials: usize,
    /// Highest attack intensity to sweep to (0.0–1.0).
    pub max_intensity: f64,
    /// Number of points along the sweep.
    pub steps: usize,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            nodes: 60,
            topology: NetworkTopology::SmallWorld,
            ticks: 40,
            trials: 8,
            max_intensity: 0.6,
            steps: 13,
        }
    }
}

/// One attack's degradation curve.
#[derive(Debug, Clone)]
pub struct Curve {
    pub attack: Attack,
    /// (intensity, fidelity) points.
    pub points: Vec<(f64, f64)>,
    /// First intensity at which fidelity falls below 0.5, if any.
    pub breaking_point: Option<f64>,
}

/// The full resilience scorecard.
#[derive(Debug, Clone)]
pub struct ResilienceReport {
    pub curves: Vec<Curve>,
    /// Partition recovery: (ticks_to_recover, fidelity_during, fidelity_after).
    pub partition: (usize, f64, f64),
    pub nodes: usize,
    pub topology: String,
    pub trials: usize,
    pub elapsed_secs: f64,
}

/// Run the sweep across attacks and intensities.
pub fn run_benchmark(cfg: ResilienceConfig) -> ResilienceReport {
    let scfg = ScenarioConfig {
        nodes: cfg.nodes,
        topology: cfg.topology,
        ticks: cfg.ticks,
        trials: cfg.trials,
    };
    let start = Instant::now();

    let intensities: Vec<f64> = (0..cfg.steps)
        .map(|i| cfg.max_intensity * i as f64 / (cfg.steps - 1).max(1) as f64)
        .collect();

    let attacks = [Attack::NodeFailure, Attack::Eclipse, Attack::Byzantine];
    let mut curves = Vec::new();
    for attack in attacks {
        println!("  sweeping {} …", attack.label());
        let points: Vec<(f64, f64)> = intensities
            .iter()
            .map(|&f| (f, scenario::measure(attack, f, &scfg)))
            .collect();
        let breaking_point = points
            .iter()
            .find(|(_, fid)| *fid < 0.5)
            .map(|(f, _)| *f);
        curves.push(Curve {
            attack,
            points,
            breaking_point,
        });
    }

    println!("  measuring partition recovery …");
    let partition = scenario::partition_recovery(&scfg);

    ResilienceReport {
        curves,
        partition,
        nodes: cfg.nodes,
        topology: format!("{:?}", cfg.topology),
        trials: cfg.trials,
        elapsed_secs: start.elapsed().as_secs_f64(),
    }
}
