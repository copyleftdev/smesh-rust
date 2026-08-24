#!/usr/bin/env bash
# Bounded mutation testing.
#
# The question: if the protocol were subtly wrong, would the suite notice?
# Mutants that survive are places where the tests assert less than they appear
# to — which is exactly how the dedup test in this repo came to assert nothing.
#
# Bounded because an unscoped run is hours of full-core load. Scope lives in
# .cargo/mutants.toml; the ceiling lives here.
set -euo pipefail
cd "$(dirname "$0")/.."
source verify/budget.sh

budget_banner

# --shard lets a long run be split across invocations instead of demanding one
# uninterrupted block: verify/mutants.sh 0/4 does the first quarter.
SHARD="${1:-}"
SHARD_ARG=()
[ -n "$SHARD" ] && SHARD_ARG=(--shard "$SHARD")

echo "scope: protocol crates only (see .cargo/mutants.toml)"
echo "listing mutants..."
cargo mutants --list 2>/dev/null | wc -l | xargs -I{} echo "  {} candidate mutants"
echo

# cargo-mutants creates only the last component of --output; if target/ is
# absent the run dies before the baseline.
mkdir -p target/mutants

budgeted cargo mutants \
  --jobs "$VERIFY_JOBS" \
  --no-shuffle \
  --output target/mutants \
  "${SHARD_ARG[@]}" \
  -- --profile test

echo
echo "survivors are listed in target/mutants/mutants.out/outcomes.json"
