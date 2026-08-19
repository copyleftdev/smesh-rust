#!/usr/bin/env bash
# Model check the gossip spec, bounded.
#
# TLC explores breadth-first, defaults to a worker per core, and grows its heap
# until the kernel intervenes. On a 64-core box with 250GB that is an excellent
# way to lose the machine to a three-node model. Everything here is pinned:
# workers, heap, wall clock, priority.
#
# Two runs, and both matter:
#   1. every holder re-announces  -> must hold
#   2. only originators do        -> must fail, with a counterexample
#
# The second is a regression test on the reasoning. If it ever starts passing,
# either the model stopped being able to strand a node or relaying stopped being
# refusable — and in both cases the first result no longer means what it says.
set -euo pipefail
cd "$(dirname "$0")/.."
source verify/budget.sh
cd verify/tla

JAR=tla2tools.jar
if [ ! -f "$JAR" ]; then
  echo "fetching TLC..."
  curl -sSL -o "$JAR" https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
fi

# Heap is capped well under the budget: this model is tiny, and a spec that
# needs more than this should be made smaller rather than given more memory.
TLC_HEAP="${TLC_HEAP:-2g}"

budget_banner
echo "TLC: ${VERIFY_JOBS} workers, ${TLC_HEAP} heap"
echo

run_tlc() {
  budgeted java -XX:+UseParallelGC -Xmx"$TLC_HEAP" -cp "$JAR" tlc2.TLC \
    -workers "$VERIFY_JOBS" -nowarning -config "$1" Gossip.tla 2>&1
}

echo "── every holder re-announces (must hold) ──"
if out=$(run_tlc Gossip.cfg) && grep -q "Model checking completed. No error has been found." <<<"$out"; then
  grep -E "^[0-9]+ states|^The depth" <<<"$out" | sed 's/^/  /'
  echo "  PASS: converges on every schedule"
else
  grep -E "^Error|^State [0-9]+|/\\\\ known" <<<"$out" | head -20 | sed 's/^/  /'
  echo "  FAIL: the fix does not hold" >&2
  exit 1
fi

echo
echo "── only originators re-announce (must fail) ──"
if out=$(run_tlc GossipOriginatorsOnly.cfg); then
  echo "  FAIL: this was expected to be violated, and was not." >&2
  echo "  The model can no longer strand a node, so the result above is hollow." >&2
  exit 1
else
  grep -E "^Error: Temporal" <<<"$out" | sed 's/^/  /'
  echo "  PASS: TLC found the counterexample, as it should"
  grep -E "^State [0-9]+:|Stuttering" <<<"$out" | tail -4 | sed 's/^/    /'
fi
