# SMESH: consensus, without a center

Agents don't vote. Evidence reinforces. Doubt decays.

SMESH is a Rust protocol for coordinating agents through signals that diffuse, decay and get reinforced. This page replays a read-only quant research run.

## The mechanism

- **Signals decay.** A hypothesis is a signal with an intensity. Left alone it fades, window by window.
- **Evidence reinforces.** Each walk-forward window tests the claim on unseen data. Agreement restores it; contradiction and a skeptic's doubt cut its confidence.
- **Jev routes.** Agents describe themselves in plain language and ask Jev whether they suit a task; settlement picks the winner.
- **A2A connects.** The mesh is one A2A v1.0 agent; other A2A agents can join it.

## The study

Scouts read Kalshi, Massive and Kraken through a client that can only issue GETs to market-data paths. Analysts posed eight hypotheses about Kalshi's prediction markets; a walker tested each on unseen windows; a skeptic doubted small samples and fee-sensitive results. The findings, metrics and source hashes are in [data/replay.json](data/replay.json).

## Ask the mesh

```
smesh a2a serve --jev --agents 0 --quant-run runs/quant-full \
  --agent "Translator=a translator between English and Portuguese"
smesh a2a send "Are Kalshi's hourly bitcoin odds well calibrated?"
```

Jev routes the question to the research agent, which asks Jev which recorded findings bear on it and returns them verbatim.

Source: https://github.com/copyleftdev/smesh-rust · Independent of TypeSafe AI, Kalshi, Massive and Kraken. Research, not trading advice.
