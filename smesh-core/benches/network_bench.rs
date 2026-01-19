//! Benchmark for network operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use smesh_core::{Network, NetworkTopology, Signal, SignalType};

fn bench_network_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_creation");

    let topologies = [
        ("ring", NetworkTopology::Ring),
        ("full_mesh", NetworkTopology::FullMesh),
        ("small_world", NetworkTopology::SmallWorld),
        ("scale_free", NetworkTopology::ScaleFree),
    ];

    for (name, topology) in topologies {
        for n_nodes in [10, 50, 100] {
            group.bench_with_input(
                BenchmarkId::new(name, n_nodes),
                &(n_nodes, topology.clone()),
                |b, (n, topo)| b.iter(|| Network::with_topology(black_box(*n), topo.clone())),
            );
        }
    }

    group.finish();
}

fn bench_network_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_tick");

    for n_nodes in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("small_world", n_nodes),
            &n_nodes,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut network = Network::with_topology(n, NetworkTopology::SmallWorld);

                        // Add some signals
                        let node_ids: Vec<String> = network.nodes.keys().take(5).cloned().collect();
                        for (i, node_id) in node_ids.iter().enumerate() {
                            let signal = Signal::builder(SignalType::Data)
                                .payload(format!("signal {}", i).into_bytes())
                                .intensity(0.8)
                                .origin(node_id)
                                .build();
                            network.field.emit_anonymous(signal);
                        }

                        network
                    },
                    |mut network| {
                        network.tick(black_box(0.1));
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn bench_network_add_connection(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_add_connection");

    for n_nodes in [20, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("add_connection", n_nodes),
            &n_nodes,
            |b, &n| {
                b.iter_batched(
                    || Network::with_topology(n, NetworkTopology::Ring),
                    |mut network| {
                        // Add some random connections
                        let node_ids: Vec<String> = network.nodes.keys().cloned().collect();
                        if node_ids.len() >= 2 {
                            network.connect(&node_ids[0], &node_ids[node_ids.len() - 1]);
                        }
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn bench_network_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_stats");

    for n_nodes in [50, 100, 500] {
        group.bench_with_input(BenchmarkId::new("stats", n_nodes), &n_nodes, |b, &n| {
            let network = Network::with_topology(n, NetworkTopology::SmallWorld);

            b.iter(|| network.stats())
        });
    }

    group.finish();
}

fn bench_topology_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_comparison");
    group.sample_size(50);

    let topologies = [
        ("ring", NetworkTopology::Ring),
        ("small_world", NetworkTopology::SmallWorld),
        ("scale_free", NetworkTopology::ScaleFree),
    ];

    let n_nodes = 50;
    let n_ticks = 10;

    for (name, topology) in topologies {
        group.bench_with_input(
            BenchmarkId::new("full_simulation", name),
            &topology,
            |b, topo| {
                b.iter_batched(
                    || {
                        let mut network = Network::with_topology(n_nodes, topo.clone());

                        // Emit signals from multiple nodes
                        let node_ids: Vec<String> =
                            network.nodes.keys().take(10).cloned().collect();
                        for (i, node_id) in node_ids.iter().enumerate() {
                            let signal = Signal::builder(SignalType::Data)
                                .payload(format!("signal {}", i).into_bytes())
                                .intensity(0.8)
                                .ttl(50.0)
                                .origin(node_id)
                                .build();
                            network.field.emit_anonymous(signal);
                        }

                        network
                    },
                    |mut network| {
                        for _ in 0..n_ticks {
                            network.tick(black_box(0.1));
                        }
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_network_creation,
    bench_network_tick,
    bench_network_add_connection,
    bench_network_stats,
    bench_topology_comparison,
);

criterion_main!(benches);
