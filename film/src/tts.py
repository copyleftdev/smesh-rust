#!/usr/bin/env python3
"""Render the narration script to per-segment audio with ElevenLabs."""
import json, os, re, subprocess, sys, urllib.request

KEY = re.search(r'=(.*)', open(os.path.expanduser('~/.creds/eleven.env')).read()).group(1).strip().strip('"\'')
SPEC = json.load(open('script.json'))
VOICE = SPEC['voice']['id']
MODEL = 'eleven_multilingual_v2'

def synth(seg, prev_text, next_text, path):
    body = json.dumps({
        'text': seg['text'],
        'model_id': MODEL,
        # Context makes the model carry prosody across a cut instead of
        # restarting cold on every segment.
        'previous_text': prev_text or None,
        'next_text': next_text or None,
        'voice_settings': {
            'stability': 0.50,
            'similarity_boost': 0.80,
            'style': 0.15,
            'use_speaker_boost': True,
        },
    }).encode()

    req = urllib.request.Request(
        f'https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_44100_192',
        data=body,
        headers={'xi-api-key': KEY, 'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()

    # Write then rename. A failure partway through used to leave a short file
    # that the next run treated as finished, which silently shifted every later
    # segment's offset and drifted the picture against the voice.
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f'{path}.part'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, path)

def duration(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=nw=1:nk=1', path],
        capture_output=True, text=True).stdout.strip()
    return float(out)

segs = SPEC['segments']
timings = []
for i, seg in enumerate(segs):
    path = f"audio/{seg['id']}.mp3"
    if not os.path.exists(path):
        synth(seg,
              segs[i-1]['text'] if i else None,
              segs[i+1]['text'] if i + 1 < len(segs) else None,
              path)
        print(f"  synthesised {seg['id']}", flush=True)
    d = duration(path)
    timings.append({'id': seg['id'], 'scene': seg['scene'], 'speech_ms': round(d * 1000),
                    'lead_in_ms': seg['lead_in_ms'], 'tail_ms': seg['tail_ms'],
                    'file': path})

# Lay the segments end to end on one timeline.
t = 0
for x in timings:
    x['start_ms'] = t
    x['speech_start_ms'] = t + x['lead_in_ms']
    x['total_ms'] = x['lead_in_ms'] + x['speech_ms'] + x['tail_ms']
    t += x['total_ms']
    x['end_ms'] = t

json.dump({'fps': SPEC['fps'], 'width': SPEC['width'], 'height': SPEC['height'],
           'total_ms': t, 'segments': timings}, open('timeline.json', 'w'), indent=2)

print(f"\ntotal runtime: {t/1000:.1f}s  ({t/60000:.2f} min)")
for x in timings:
    print(f"  {x['id']:<18} {x['start_ms']/1000:7.2f}s  speech {x['speech_ms']/1000:6.2f}s  -> {x['end_ms']/1000:7.2f}s")
