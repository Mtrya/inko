---
title: Inko
emoji: 🖌️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.13.0
python_version: 3.12
app_file: app.py
pinned: false
models:
  - kaupane/Inko
suggested_hardware: zero-a10g
short_description: Generate artistic paintings from text prompts
---

# Inko

Paintings from a 0.5B diffusion transformer, with optional DeepSeek prompt enhancement.

Start with your own description or select a painting from the carousel. Selecting
an example restores its exact prompt, canvas, seed, steps, and CFG. Enhancement and
seed randomization turn off for that selection. **Use this prompt** copies the
displayed result's prompt and canvas into the input and turns off enhancement.

**Auto** lets enhancement choose from the Standard (~896²) and Large (~1280²)
canvas presets. Explicit selections are fixed. With enhancement off or unavailable,
Auto falls back to square and Standard. Custom dimensions support 480–1920 pixels
per axis, in multiples of 16.

The result viewer keeps the latest 20 successful generations in the current session.
Selecting a thumbnail restores its recipe. Downloaded PNGs include the actual prompt,
dimensions, seed, steps, CFG, sampler, and model identifier in the `inko` metadata field.
**Delete image** removes the selected generation and selects a neighboring image.
Deleting the last image clears the viewer and keeps your input for another generation.
History clears with a new session; download images you want to keep.

The carousel contains the 18 release-model images from
`notes/assets/gallery/hero*/originals/`, with prompts, dimensions, and seeds from
their manifests. They use 20 Euler steps and CFG 4.5. The PNGs in `demo/showcase/`
carry these settings; the WebP files are lightweight carousel previews.

## Run and deploy

Set `INKO_MODEL_ID=kaupane/Inko` for the published final RL EMA, or use a local
export directory. Set `DEEPSEEK_API_KEY` as a secret to enable enhancement. Run
`python app.py` from the source checkout. `demo/pe_prompt.py` contains the system
prompt; `demo/generation.py` handles canvases, enhancement, and image records.

The Space uses the same package layout: copy the root `app.py`, `demo/` (including
its assets), and inference `src/` modules. Copy this README and `demo/requirements.txt`
to the Space root. Keep Gradio and Spaces supplied by the hosting environment.
Gradio 6.13 supports the custom galleries and retains the Hub dependency range
required by the Transformers 4.x inference runtime.

The `/prepare_prompt` endpoint returns a prepared request as its first output.
Pass that JSON object to `/generate`, which returns the PNG with embedded metadata.
Preparation validates the full formatted prompt against Inko's tokenizer budget.
Incomplete, invalid, or over-budget enhancements fall back to the original request.

[Source](https://github.com/Mtrya/inko)
