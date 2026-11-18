# 🗑️ TTS Speak

Speaks text in a cloned voice with the model from **🗑️ TTS Loader**: Fish S2 Pro or Higgs v3. The node reads which engine the model is and runs that engine's code; the inputs are the same for both.

It works one of two ways:

| Mode | When | What it does |
|------|------|--------------|
| **Single line** | No JSON dialogue in the voice library | Speaks `text` as `speaker`, optionally as the next turn of a scene (`history`). |
| **Dialogue** | `voice_library_input` carries a JSON dialogue | Speaks every line in order and saves each one to disk the moment it finishes. |

```
  TTS Loader ──model──────────────────────┐
  TTS Voice ──voices──────────────────────┤
  TTS Scene Turn ──history────────────────┼──► TTS Speak ──audio / name──► Save Audio
  Audio Library Loader ──audio_selection──┘
```

## The two engines

| | Fish S2 Pro | Higgs v3 |
|---|---|---|
| Output | 44.1 kHz | 24 kHz |
| Delivery tags | free text in brackets: `[whisper]`, `[laughing]`, `[pause]`, `[sad]`, … | 43 fixed tags: `<\|emotion:anger\|>`, `<\|style:whispering\|>`, `<\|prosody:pause\|>`, `<\|sfx:laughter\|>Haha, ` |
| Scene context | hears **every** earlier line (the whole conversation) | hears only **its own speaker's** last lines (`context_lines` on the loader) |
| Speed options | `mode` and `capture_step` on the loader | `mode` and `capture_step` on the loader |

Fish brackets are read aloud by Higgs, and Higgs tags don't work in Fish; Higgs prints a console warning for tags it doesn't know.

## Voices and speakers

A speaker is a number: the position of a voice in the list Speak receives.

- **TTS Voice** nodes chain together; the first is speaker `0`, the next `1`, and so on.
- **Audio Library Loader** voices come after any chained voices. The ticked voices are numbered in list order, and the library shows each one's number (`#0`, `#1`, …).
- No voices at all: the model makes up a voice. Keep the `seed` fixed to get the same one again.

A reference should be 5–20 s of clean speech, and its transcript must match what is said exactly.

## Single line

- **text**: the line to speak, with the engine's tags.
- **speaker**: which voice says it.
- **history** (optional): earlier lines from **TTS Scene Turn** nodes. Fish hears them all (the oldest are dropped when the context fills up); Higgs hears the last `context_lines` of them spoken by the same speaker.

The `name` output is `%HMSf%`. Wire it into Save Audio's `filename_prefix` to get a timestamped file.

## Dialogue

Connect an **Audio Library Loader** that has a JSON dialogue wired in. Each key is the line's file name and each value gives its voice and text:

```json
{
  "line_01.flac": { "audio_ref": "Audio_01.wav", "dialogue": "Hello there!" },
  "line_02":      { "audio_ref": "Audio_02.wav", "dialogue": "General Kenobi!" },
  "line_03.wav":  { "audio_ref": "Audio_01.wav", "dialogue": "I've been waiting for you." }
}
```

- Speakers are numbered by first appearance in the JSON. `text`, `speaker`, `voices` and `history` on this node are ignored.
- Each line is saved to **save_folder** as soon as it is spoken. Blank saves to `output/tts_dialogue`.
- A name's own extension sets its format (`line_03.wav`). A name without one (`line_02`) uses **save_format**. WAV and FLAC are saved as 24-bit, and the spoken text goes in the file's comment tag.
- **skip_existing** (on): a line whose file already exists is loaded instead of spoken again. If a run fails or is cancelled part-way, run it again and it carries on from the first missing line; only the line in progress is lost. Delete a file to redo that line.
- **scene_context** (on): each line hears the lines before it, including ones loaded from disk (Fish: all of them; Higgs: its own speaker's). Turn it off to speak every line on its own.
- Line *i* uses `seed + i`, so each line gets its own seed and a rerun reproduces it.

The lines are already saved, so a Save Audio after Speak is optional in this mode. If you add one, wire `name` into its `filename_prefix` so its copies keep the line names.

## Sampling

| Input | Default | Effect |
|-------|---------|--------|
| **temperature** | 0.8 | Expressiveness. Higher values are more varied, lower values are more consistent. Suggested: Fish 0.8, Higgs 0.65. |
| **top_p** | 0.9 | Nucleus sampling: keeps the most likely tokens until their combined probability reaches p. |
| **top_k** | 30 | Keeps only the k most likely tokens at each step. Lower values are steadier, higher values allow rarer, more surprising delivery. |
| **max_new_tokens** | 0 | Cap on audio frames (Fish about 21.5 per second of speech, Higgs 25). `0` = the engine's default: Fish up to the context limit, Higgs 1500 (60 s). |
| **seed** | 42 | Same seed, settings and inputs give the same take. |

top_k caps how many candidates there are, top_p caps how much probability they cover, and temperature reshapes the probabilities of whatever survives. For speech, a low top_k mostly prevents the odd mispronunciation or glitch, at the cost of slightly flatter delivery. How strongly a delivery tag lands varies a lot from seed to seed: try a few.

## Progress and status

- The node's progress bar follows each line. Its length is estimated from the text, grows if the line runs longer, and fills when the line ends.
- The console shows a frame counter: `Fish S2 Pro` or `Higgs v3` for a single line, `… line 2/5` in a dialogue.
- The node shows **Speaking…** while it runs, then one status line per line:
  - `✓ line_01: 6.3 s of speech in 9.2 s · RTF 1.46 · graph`. RTF is seconds taken per second of speech (lower is faster). Lines that heard earlier lines add `· heard N earlier`.
  - `↺ line_02: file exists, loaded`: skipped by `skip_existing`.
  - Dialogue mode adds a summary first: `3 lines (2 spoken, 1 loaded) saved to …`.

## Outputs

All three outputs are lists with one entry per line. Nodes connected after Speak run once per line.

| Output | Contents |
|--------|----------|
| **audio** | The spoken line (44.1 kHz Fish, 24 kHz Higgs). |
| **info** | JSON per line: speech length, frames, timings, RTF, mode, how many earlier lines it heard; in dialogue mode also the line name and file path. |
| **name** | The line's file name (dialogue mode), or `%HMSf%` for a single line. |
