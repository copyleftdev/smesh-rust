#!/usr/bin/env bash
# Render every frame of the film in parallel. Frames are numbered by absolute
# frame index, so the two capture passes drop into one directory in order.
set -euo pipefail
cd "$(dirname "$0")"

# Pass A: the authored scenes, split into chunks by film time (ms).
CHUNKS=(
  "0:40000" "40000:80000" "80000:118000" "118000:158000"
  "158000:196000" "196000:233540"
  "392700:430000" "430000:466140"
)
for c in "${CHUNKS[@]}"; do
  from="${c%%:*}"; to="${c##*:}"
  node shoot.js --out=frames --from="$from" --to="$to" > "logs/film_${from}.log" 2>&1 &
done

# Pass B: the demo, one process per shot.
for seg in s10_incident s11_mesh s12_claims s13_consensus s14_decoys s15_evidence; do
  node shootDemo.js --out=frames --only="$seg" > "logs/demo_${seg}.log" 2>&1 &
done

wait
echo "render complete: $(ls frames | wc -l) frames"
