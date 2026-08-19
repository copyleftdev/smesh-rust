#!/usr/bin/env bash
# Render every frame of the film in parallel. Frames are numbered by absolute
# frame index, so the two capture passes drop into one directory in order.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs frames

# Pass A: the authored scenes, split into chunks by film time (ms).
CHUNKS=(
  "0:40000" "40000:80000" "80000:118000" "118000:158000"
  "158000:196000" "196000:233540"
  "392700:430000" "430000:466140"
)
PIDS=()
for c in "${CHUNKS[@]}"; do
  from="${c%%:*}"; to="${c##*:}"
  node shoot.js --out=frames --from="$from" --to="$to" > "logs/film_${from}.log" 2>&1 &
  PIDS+=($!)
done

# Pass B: the demo, one process per shot.
for seg in s10_incident s11_mesh s12_claims s13_consensus s14_decoys s15_evidence; do
  node shootDemo.js --out=frames --only="$seg" > "logs/demo_${seg}.log" 2>&1 &
  PIDS+=($!)
done

# Bare `wait` reports only the last job, so a failed pass used to sail through
# and the encode would silently produce a film with missing frames.
failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || { echo "render job $pid failed; see logs/" >&2; failed=1; }
done
[ "$failed" -eq 0 ] || exit 1
echo "render complete: $(find frames -name '*.jpg' | wc -l) frames"
