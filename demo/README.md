---
title: Inko
emoji: 🖌️
colorFrom: indigo
colorTo: pink
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

English and Chinese text-to-image generation with the final RL EMA export.
Width and height each support 480–2560 pixels, in multiples of 16.
RoPE scaling and VAE tiling follow the requested dimensions automatically.

Set `INKO_MODEL_ID=kaupane/Inko` for the published final RL EMA, or use a local
export directory. Start `python -m demo.app` from the source checkout for local use. The loaded export supplies
the sampling defaults: 20 Euler steps and CFG 4.5 for the final RL EMA.
The gallery pictures are from the earlier pretrained model; selecting a prompt
uses the loaded model's defaults and the saved seed.

[Source](https://github.com/Mtrya/inko)
