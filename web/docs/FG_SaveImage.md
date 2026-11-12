# 🗑️ Save Image

Saves images to ComfyUI's `output` folder as PNG, JPG or WebP, with short clean filenames.

## Filenames

| Batch | Filename |
|-------|----------|
| 1 image | `1112133705.png` |
| Several images | `1112133705_000.png`, `1112133705_001.png`, … |

- **`%HMSf%`** in the prefix becomes the month, day, hour, minute and second the image was saved: `1112133705` is November 12, 13:37:05.
- Existing files are never overwritten. A single image gets `_01`, `_02`, … added (`portrait_01.png`); a batch gets it before the counter (`portrait_01_000.png`, `portrait_01_001.png`, …).
- Use `/` in the prefix to save into a subfolder: `portraits/%HMSf%` saves to `output/portraits/`. Missing folders are created. A prefix that points outside the output folder (`../` or an absolute path) saves there, with a warning in the console.

## Formats

| Format | Compression | Transparency | Metadata |
|--------|-------------|--------------|----------|
| **PNG** | Lossless | Kept | PNG text chunks |
| **JPG** | Lossy | Flattened onto white | EXIF, up to 64 KB |
| **WebP** | Lossless or lossy | Kept | EXIF |

## Options

- **embed_metadata**: stores the prompt and workflow in the image so it can be dragged back into ComfyUI. Ignored when ComfyUI runs with `--disable-metadata`.
- **quality_optimized** (JPG / WebP):
  - **WebP**: saves lossless; `quality` is ignored.
  - **JPG**: optimizes the file's compression tables and keeps full colour resolution (4:4:4, no chroma subsampling). Smaller files and sharper colour edges at the same quality.
- **quality** (JPG / WebP, 1–100): lossy compression quality. Used by JPG always and by WebP when `quality_optimized` is off.
- **webp_method** (WebP): compression effort, `0` fastest to `6` smallest file. `default` uses Pillow's default.

PNG ignores `quality_optimized`, `quality` and `webp_method`.

## JPG metadata limit

JPG stores metadata in a single EXIF block capped at 64 KB. Large workflows don't fit; the image is then saved **without** metadata and a warning is printed in the console. Use PNG or WebP when you need the workflow inside the file.
