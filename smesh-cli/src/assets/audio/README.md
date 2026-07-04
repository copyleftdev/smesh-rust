# Guided-tour narration

`scenes.json` is the source of truth for the narrated tour (`smesh showcase` →
**Guided Tour**). Each scene's `text` is voiced by ElevenLabs (voice **Sarah**
`EXAVITQu4vr4xnSDxMaL`, model **eleven_v3**) into `<id>.mp3`, embedded into the
binary via `include_bytes!` in `showcase.rs` and served at `/audio/<id>.mp3`.
The `title`/`caption` fields drive the on-screen callout; `tab`/`action`/`claim`
drive the demo automatically as the voiceover plays.

## Regenerate after editing `scenes.json`

Needs `ELEVEN_LABS_API_KEY`. For each scene, POST its `text` to:

```
https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL?output_format=mp3_44100_128
  { "text": <scene.text>,
    "model_id": "eleven_v3",
    "voice_settings": { "stability": 0.5, "similarity_boost": 0.75 } }
```

Save the response body to `<scene.id>.mp3`, then rebuild (`cargo build -p smesh-cli`)
to re-embed the clips.
