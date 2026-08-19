#!/usr/bin/env bash
# Assemble the rendered frames and the mixed audio into the finished film.
set -euo pipefail
cd "$(dirname "$0")"

FPS=$(python3 -c "import json;print(json.load(open('timeline.json'))['fps'])")
EXPECTED=$(python3 -c "import json;tl=json.load(open('timeline.json'));print(int(tl['total_ms']/1000*tl['fps']))")
HAVE=$(ls frames | wc -l)
echo "frames: $HAVE (expected ~$EXPECTED)"

# Any gap would show as a stutter, so find them before encoding rather than after.
python3 - <<'PY'
import os, re
fs = sorted(int(re.sub(r'\D', '', f)) for f in os.listdir('frames') if f.endswith('.jpg'))
gaps = [(a, b) for a, b in zip(fs, fs[1:]) if b != a + 1]
print(f"frame range {fs[0]}..{fs[-1]}, count {len(fs)}")
if gaps:
    print(f"WARNING: {len(gaps)} gap(s); first few: {gaps[:5]}")
else:
    print("no gaps")
PY

# Frames are numbered by absolute index; glob keeps them in order.
ffmpeg -y -loglevel error -stats \
  -framerate "$FPS" -pattern_type glob -i 'frames/*.jpg' \
  -i narration.m4a \
  -map 0:v -map 1:a \
  -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p \
  -x264-params "keyint=60:min-keyint=30" \
  -c:a copy -movflags +faststart -shortest \
  smesh-film.mp4

echo
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=nw=1 smesh-film.mp4
