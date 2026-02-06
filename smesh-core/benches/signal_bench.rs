//! Benchmark for signal operations

use chrono::Utc;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use smesh_core::{Field, Node, Signal, SignalType};

fn bench_signal_creation(c: &mut Criterion) {
    c.bench_function("signal_creation", |b| {
        b.iter(|| {
            Signal::builder(SignalType::Data)
                .payload(black_box(b"test payload".to_vec()))
                .intensity(0.8)
                .ttl(60.0)
                .build()
        })
    });
}

fn bench_signal_effective_intensity(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_intensity");

    let signal = Signal::builder(SignalType::Data)
        .payload(b"test".to_vec())
        .intensity(1.0)
        .ttl(60.0)
        .build();

    group.bench_function("effective_intensity", |b| {
        let now = Utc::now();
        b.iter(|| signal.effective_intensity(black_box(now)))
    });

    // Also benchmark current_intensity access
    group.bench_function("current_intensity", |b| {
        b.iter(|| black_box(signal.current_intensity))
    });

    group.finish();
}

fn bench_signal_hash(c: &mut Criterion) {
    let signal = Signal::builder(SignalType::Data)
        .payload(b"test payload for hashing".to_vec())
        .intensity(0.8)
        .build();

    c.bench_function("signal_hash", |b| {
        b.iter(|| {
            let _ = black_box(&signal.origin_hash);
        })
    });
}

fn bench_field_emit(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_emit");

    for n_signals in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("emit", n_signals), &n_signals, |b, &n| {
            b.iter(|| {
                let mut field = Field::new();
                let mut node = Node::new();

                for i in 0..n {
                    let signal = Signal::builder(SignalType::Data)
                        .payload(format!("signal {}", i).into_bytes())
                        .intensity(0.8)
                        .build();
                    field.emit(signal, &mut node);
                }
            })
        });
    }

    group.finish();
}

fn bench_field_tick(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_tick");

    for n_signals in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("tick", n_signals), &n_signals, |b, &n| {
            b.iter_batched(
                || {
                    let mut field = Field::new();
                    let mut node = Node::new();

                    for i in 0..n {
                        let signal = Signal::builder(SignalType::Data)
                            .payload(format!("signal {}", i).into_bytes())
                            .intensity(0.8)
                            .ttl(100.0)
                            .build();
                        field.emit(signal, &mut node);
                    }
                    field
                },
                |mut field| {
                    field.tick(black_box(0.1));
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_field_sense(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_sense");

    for n_signals in [100, 1000] {
        group.bench_with_input(BenchmarkId::new("sense", n_signals), &n_signals, |b, &n| {
            b.iter_batched(
                || {
                    let mut field = Field::new();
                    let mut node = Node::new();

                    for i in 0..n {
                        let signal = Signal::builder(SignalType::Data)
                            .payload(format!("signal {}", i).into_bytes())
                            .intensity(0.8)
                            .build();
                        field.emit(signal, &mut node);
                    }
                    (field, node)
                },
                |(field, node)| {
                    let _ = field.sense(&node);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_signal_reinforcement(c: &mut Criterion) {
    c.bench_function("signal_reinforce", |b| {
        b.iter_batched(
            || {
                Signal::builder(SignalType::Data)
                    .payload(b"test".to_vec())
                    .intensity(0.8)
                    .build()
            },
            |mut signal| {
                signal.reinforce(black_box("node123"));
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    benches,
    bench_signal_creation,
    bench_signal_effective_intensity,
    bench_signal_hash,
    bench_field_emit,
    bench_field_tick,
    bench_field_sense,
    bench_signal_reinforcement,
);

criterion_main!(benches);
