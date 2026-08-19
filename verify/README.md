# Verification

Three layers, each answering a different question, none of them allowed to take
the machine.

| Layer | Question | Cost | When |
| --- | --- | --- | --- |
| `tla.sh` | Does the *design* have the property, on every schedule? | seconds | nightly, on demand |
| `cargo test` (dst) | Does the *implementation* have it, under sampled schedules? | ~20s | every test run |
| `mutants.sh` | Would the *tests notice* if it stopped having it? | ~20 min | nightly, on demand |

They overlap on purpose. The model proves a property for all schedules but only
for a three-node abstraction. The simulation runs the real merge and relay code
but samples schedules rather than exhausting them. Mutation testing checks the
thing neither can: whether the assertions are load-bearing or decorative.

## The bug all three were built around

Gossip forwards only when local knowledge grew, which is what makes it
terminate. It also means a node goes permanently silent about anything it
already knows — so a neighbour stranded behind it never hears the claim again.
Re-announcing from the nodes that originated a claim does not help, because the
silent node sits in between.

The simulation found it by failing to converge with no packet loss at all. The
model then produced the minimal counterexample: three nodes in a line, the
middle one relays, declines, and stutters forever. The fix is that every holder
re-announces, not just originators, and both the passing and failing variants
are checked so the reasoning cannot rot.

## Resource ceiling

`budget.sh` is sourced by every runner: a quarter of the cores capped at eight,
a memory limit, a wall-clock timeout, low priority, and a refusal to start at
all if the machine is already busy.

This is not caution for its own sake. `cargo-mutants` rebuilds and reruns the
suite per mutant and defaults to one job per core; TLC explores breadth-first
with a worker per core and grows its heap until the kernel intervenes. On a
large machine neither of those is fast — they are a stalled desktop.

```sh
./verify/tla.sh                  # model check, both variants
./verify/mutants.sh              # full mutation run
./verify/mutants.sh 0/4          # first quarter, for a shorter sitting
SMESH_DST_SEEDS=5000 cargo test -p smesh-core --test dst --release

VERIFY_JOBS=2 ./verify/mutants.sh   # quieter still
```
