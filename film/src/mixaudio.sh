#!/usr/bin/env bash
# Place each narration segment at its exact offset on the timeline, then sit
# the ambient bed underneath. Offsets come from the same timeline the picture
# is cut to, so speech and image cannot drift apart.
set -euo pipefail
cd "$(dirname "$0")"

mapfile -t LINES < <(python3 -c "
import json
tl = json.load(open('timeline.json'))
for s in tl['segments']:
    print(s['file'], s['speech_start_ms'])
")

INPUTS=(); FILTERS=(); LABELS=""
i=0
for line in "${LINES[@]}"; do
  f="${line%% *}"; d="${line##* }"
  INPUTS+=(-i "$f")
  FILTERS+=("[$i:a]adelay=${d}|${d}[a$i]")
  LABELS="${LABELS}[a$i]"
  i=$((i+1))
done

DUR=$(python3 -c "import json;print(json.load(open('timeline.json'))['total_ms']/1000)")

# The narration segments never overlap, so amix here is placement rather than
# blending; normalize=0 keeps each segment at the level it was rendered.
# Output uncompressed first so the loudness pass has something clean to measure.
ffmpeg -y -loglevel error -stats "${INPUTS[@]}" -i score.wav \
  -filter_complex "$(IFS=';'; echo "${FILTERS[*]}");\
${LABELS}amix=inputs=${i}:normalize=0:dropout_transition=0[vo];\
[vo]alimiter=limit=0.95[voz];\
[${i}:a]volume=1.0[bed];\
[voz][bed]amix=inputs=2:normalize=0:dropout_transition=0[mix];\
[mix]atrim=0:${DUR},asetpts=N/SR/TB[out]" \
  -map "[out]" -c:a pcm_s16le -ar 48000 mix.wav

# Two-pass loudness normalisation. EBU R128's -23 LUFS is a broadcast target and
# plays far too quietly on a laptop; -16 LUFS is the sane figure for something
# shown in a meeting room or streamed.
echo "measuring loudness..."
MEASURED=$(ffmpeg -hide_banner -i mix.wav -af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json -f null - 2>&1 \
  | python3 -c "
import sys, json, re
text = sys.stdin.read()
blob = text[text.rindex('{'):text.rindex('}') + 1]
d = json.loads(blob)
print('%s|%s|%s|%s' % (d['input_i'], d['input_tp'], d['input_lra'], d['input_thresh']))
")
IFS='|' read -r MI MTP MLRA MTHRESH <<< "$MEASURED"
echo "  measured: I=${MI} TP=${MTP} LRA=${MLRA}"

ffmpeg -y -loglevel error -stats -i mix.wav \
  -af "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=${MI}:measured_TP=${MTP}:measured_LRA=${MLRA}:measured_thresh=${MTHRESH}:linear=true:print_format=summary" \
  -c:a aac -b:a 256k -ar 48000 narration.m4a

rm -f mix.wav

echo
echo -n "narration.m4a duration: "
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 narration.m4a
ffmpeg -hide_banner -i narration.m4a -af ebur128=framelog=quiet -f null - 2>&1 | grep -A3 "Integrated loudness"
