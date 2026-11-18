# 🗑️ TTS Loader

Loads a text-to-speech model for **🗑️ TTS Speak**: Fish S2 Pro or Higgs v3. Pick the engine, and only that engine's settings show.

The checkpoint lists come from ComfyUI's `models/audios` folders (and any `audios:` entry in `extra_model_paths.yaml`), up to two levels down. Each folder is sorted into its engine by what's inside it:

| Engine | A checkpoint folder holds |
|--------|---------------------------|
| **Fish S2 Pro** | `config.json`, the safetensors shards and `codec.pth` |
| **Higgs v3** | `config.json` with an `audio_encoder_config`, and `model.safetensors` |

**checkpoint_path** (optional) loads any folder instead of the list. The engine is read from the folder, so a Higgs folder loads as Higgs even with Fish selected (a console warning says so, and the engine's own defaults are used).

ComfyUI moves the model onto the GPU when a line is spoken and off again when something else needs the memory.

## Fish S2 Pro

| Setting | Default | Effect |
|---------|---------|--------|
| **max_seq_len** | 8192 | Context: voice references + earlier lines + this line. 8192 holds about 4 minutes of scene; the model's 32768 costs about 5.8 GB more. |
| **precision** | bfloat16 | The language model. |
| **codec_precision** | float32 | The codec is small; float32 avoids decode artefacts. |
| **mode** | graph | `graph`: faster decoding loop captured as an XPU graph, about 3–4× stock once warm (the first line of a run is slower while it captures). `nograph`: the same loop, run eagerly. `stock`: fish-speech's own loop. |
| **capture_step** | 4 | Graph mode: frames decoded normally before the graph is captured, so the XPU libraries warm up first. |

## Higgs v3

| Setting | Default | Effect |
|---------|---------|--------|
| **precision** | bfloat16 | The language model; the codec always runs in float32. |
| **mode** | graph | `graph`: fixed-size cache, end-of-speech checked every 8 frames, and one frame captured as an XPU graph: about 3.7× eager once warm (about 46 frames/s against 12.5 on the B70, where speech needs 25). `nograph`: the same loop, run eagerly. `eager`: transformers' own forward. |
| **capture_step** | 4 | Graph mode: frames decoded normally before the graph is captured. |
| **context_lines** | 1 | Scene context: how many of the speaker's own earlier lines go in the reference slot after the voice sample. Higgs never hears other speakers. `0` = plain cloning. |
| **codec_config** | blank | The Higgs v2 codec's `config.json`. Blank uses `higgs-audio-v2-3B/tokenizer/config.json` next to the checkpoint if there is one, else the copy shipped in `fg_higgs_audio/codec_config.json`. |
