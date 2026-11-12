# 🎲 Noisy Seeder

Outputs a seed and a matching `NOISE` generator for SamplerCustom, SamplerCustomAdvanced or any node with a `NOISE` input.

## Seed range

The seed goes from **0 to 4,294,967,295** (2³² − 1).

ComfyUI creates the starting noise on the CPU, and PyTorch only uses the lowest 32 bits of the seed there. A larger seed wraps around: subtract 4,294,967,296 until it fits. So 4,294,967,296 gives the same noise as 0, 4,294,967,297 the same as 1, and so on. Higher seeds add no new images, so this node stops at 4,294,967,295.

Neighbouring seeds are unrelated: seed 1000 and seed 1001 give completely different noise.

## Seed values

- **`-1`**: new random seed every run.
- **Anything else**: used exactly as entered.

To step through nearby seeds, click **♻️ Use Last Queued Seed**, then the **<** | **>** arrows on the seed widget.

The seed that was actually used is written into the image metadata, so a saved image always reloads with a real seed.

## Random seeds

Random seeds are today's month and day followed by six random digits, e.g. `1112849630` on November 12. That tops out at `1231999999`, well inside the range, and gives a million different seeds per day.

## Buttons

- **🎲 Randomize Each Time**: sets the seed to `-1`.
- **🎲 New Fixed Random**: picks one random seed and keeps it.
- **♻️ Use Last Queued Seed**: copies the seed from the last run into the widget, so you can keep a result you liked.
