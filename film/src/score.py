#!/usr/bin/env python3
"""A very quiet ambient bed for the film.

Deliberately minimal: a sustained low drone with a few partials that breathe
against each other, sitting far under the narration. It is there to keep eight
minutes of speech from sounding like a screen recording, not to be noticed.
Levels are conservative on purpose - if it is audible as *music*, it is wrong.
"""
import json, numpy as np, wave

SR = 44100
tl = json.load(open('timeline.json'))
DUR = tl['total_ms'] / 1000.0 + 0.5
n = int(DUR * SR)
t = np.arange(n) / SR

def partial(freq, amp, lfo_hz, lfo_depth, phase=0.0):
    """A sine with a slow amplitude drift, so the pad never sits still."""
    breathe = 1.0 - lfo_depth + lfo_depth * (0.5 + 0.5 * np.sin(2 * np.pi * lfo_hz * t + phase))
    return amp * breathe * np.sin(2 * np.pi * freq * t + phase)

# D minor: root, fifth, octave, tenth, plus a distant shimmer.
bed = (
    partial(73.416,  0.55, 0.031, 0.35, 0.0) +
    partial(110.00,  0.34, 0.023, 0.40, 1.1) +
    partial(146.832, 0.22, 0.017, 0.45, 2.3) +
    partial(174.614, 0.13, 0.013, 0.55, 0.7) +
    partial(220.00,  0.09, 0.011, 0.60, 3.0) +
    partial(440.00,  0.035, 0.007, 0.75, 1.9)
)

# Gentle swells where the picture asks for one.
def swell(start, end, gain, ramp=3.0):
    env = np.ones(n)
    s, e = int(start * SR), int(end * SR)
    r = int(ramp * SR)
    seg = np.ones(e - s) * gain
    seg[:r] = np.linspace(1.0, gain, r)
    seg[-r:] = np.linspace(gain, 1.0, r)
    env[s:e] = seg
    return env

by_id = {s['id']: s for s in tl['segments']}
def span(seg_id):
    s = by_id[seg_id]
    return s['start_ms'] / 1000.0, s['end_ms'] / 1000.0

env = np.ones(n)
for seg_id, gain in [('s03_reveal', 1.55), ('s13_consensus', 1.45), ('s17_close', 1.5)]:
    a, b = span(seg_id)
    env *= swell(a, min(b, DUR - 0.1), gain)

bed *= env

# Overall level: quiet enough to live under a voice without ducking.
bed /= np.max(np.abs(bed)) + 1e-9
bed *= 0.085

# Fade the ends so nothing clicks.
fi, fo = int(6 * SR), int(7 * SR)
bed[:fi] *= np.linspace(0, 1, fi)
bed[-fo:] *= np.linspace(1, 0, fo)

stereo = np.stack([bed, np.roll(bed, 240)], axis=1)  # a little width
pcm = np.clip(stereo, -1, 1)
pcm = (pcm * 32767).astype(np.int16)

with wave.open('score.wav', 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())

print(f"score.wav  {DUR:.1f}s")
