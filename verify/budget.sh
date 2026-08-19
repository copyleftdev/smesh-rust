#!/usr/bin/env bash
# Shared resource ceiling for every verification run.
#
# These tools are all happy to take the whole machine. cargo-mutants rebuilds
# and reruns the suite per mutant and defaults to one job per core; TLC explores
# a state space breadth-first and will grow its heap until the kernel intervenes.
# On a 64-core box that is not "fast", it is a stalled desktop and a thermal
# event. Nothing here is urgent enough to justify either.
#
# Policy: take a quarter of the cores, capped, at low priority, under a hard
# wall-clock limit and a hard memory limit. Interactive work always wins.

set -euo pipefail

# --- how much of the machine may a verification run take? -------------------
CORES_TOTAL="$(nproc)"
: "${VERIFY_JOBS:=$(( CORES_TOTAL / 4 ))}"
[ "$VERIFY_JOBS" -lt 1 ] && VERIFY_JOBS=1
[ "$VERIFY_JOBS" -gt 8 ] && VERIFY_JOBS=8          # a hard ceiling, not a ratio

RAM_TOTAL_MB="$(free -m | awk '/Mem:/{print $2}')"
: "${VERIFY_MEM_MB:=$(( RAM_TOTAL_MB / 4 ))}"
[ "$VERIFY_MEM_MB" -gt 16384 ] && VERIFY_MEM_MB=16384

: "${VERIFY_TIMEOUT:=1800}"                         # 30 minutes, then stop
: "${VERIFY_NICE:=15}"                              # yield to anything interactive

export VERIFY_JOBS VERIFY_MEM_MB VERIFY_TIMEOUT VERIFY_NICE

budget_banner() {
  echo "╭─ resource budget"
  echo "│  jobs      ${VERIFY_JOBS} of ${CORES_TOTAL} cores"
  echo "│  memory    ${VERIFY_MEM_MB} MB of ${RAM_TOTAL_MB} MB"
  echo "│  timeout   ${VERIFY_TIMEOUT}s"
  echo "│  priority  nice ${VERIFY_NICE}"
  echo "╰─  override with VERIFY_JOBS / VERIFY_MEM_MB / VERIFY_TIMEOUT"
  echo
}

# Refuse to pile onto a machine that is already busy.
guard_load() {
  local load cores_free
  load=$(awk '{print int($1)}' /proc/loadavg)
  cores_free=$(( CORES_TOTAL - load ))
  if [ "$cores_free" -lt "$VERIFY_JOBS" ]; then
    echo "load is already ${load}; only ${cores_free} cores idle." >&2
    echo "refusing to start a ${VERIFY_JOBS}-job run. Wait, or set VERIFY_JOBS lower." >&2
    exit 1
  fi
}

# Run a command inside the budget: capped memory, capped time, low priority.
#
# The virtual-memory ceiling turns a runaway into a clean allocation failure
# rather than an OOM kill that takes something else with it. It suits native
# processes; see `budgeted_jvm` for why it does not suit a JVM.
budgeted() {
  guard_load
  ( ulimit -v $(( VERIFY_MEM_MB * 1024 )) 2>/dev/null || true
    exec nice -n "$VERIFY_NICE" ionice -c3 timeout --signal=INT "$VERIFY_TIMEOUT" "$@" )
}

# As above, without the virtual-memory ceiling.
#
# A JVM reserves far more address space than it will ever commit -- compressed
# class space alone asks for a gigabyte before any heap -- so `ulimit -v` sized
# to the intended heap stops it starting at all. It fails as
# "Could not allocate compressed class space", which reads like a memory
# shortage and is really the guardrail. Found when this ran on a CI box with a
# smaller budget than the machine it was written on.
#
# The heap is capped by -Xmx instead, which is the JVM's own instrument for the
# job and is precise about what it limits.
budgeted_jvm() {
  guard_load
  nice -n "$VERIFY_NICE" ionice -c3 timeout --signal=INT "$VERIFY_TIMEOUT" "$@"
}
