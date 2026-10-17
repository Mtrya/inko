# Model evaluation

Standalone evaluation uses OneIG-Bench (English), original GenEval, DPG-Bench,
TIIF testmini with short/long descriptions, and Qwen-Image-Bench (English).
Generation saves lossless images. Official judges consume those saved images
in separate environments. Reports run on CPU without loading model weights.
Training loss, KID and interval reward readouts belong to `evaluation/monitoring`.
Painting comparison panels are assembled manually.

Use `configs/evaluation.toml` for inference and scorer settings. Machine-local
asset TOMLs locate weights, official source checkouts and isolated interpreters.
The external comparison uses DreamLite-Base, SANA-0.6B, PixArt-Σ, SD3.5 Medium
and MOSS-Image-SFT. External models retain their native recipes. Inko-Base uses the fixed 600k EMA
at CFG 5.0, selected from experience with that checkpoint. Select posttrain CFG
with a small sweep after training completes and before running its benchmarks;
keep the other Inko inference settings matched.

```bash
python -m scripts.eval.run prepare --config configs/evaluation.toml \
  --assets data/evaluation/scorers.toml --benchmark geneval \
  --output data/evaluation/geneval.jsonl
python -m scripts.eval.run generate --config configs/evaluation.toml \
  --assets data/evaluation/assets.toml --prompts data/evaluation/geneval.jsonl \
  --model inko_pretrain --output runs/evaluation/images
python -m scripts.eval.run score --config configs/evaluation.toml \
  --assets data/evaluation/scorers.toml --benchmark geneval \
  --images runs/evaluation/images --model inko_pretrain \
  --output runs/evaluation/scores/geneval/inko_pretrain
python -m scripts.eval.run report --config configs/evaluation.toml \
  --prompts data/evaluation/geneval.jsonl --images runs/evaluation/images \
  --scores runs/evaluation/scores --model inko_pretrain \
  --pending inko_posttrain --output runs/evaluation/report
```

A smaller catalog selects a slice of saved judgments without rescoring.
Repeat `--prompts` for each prepared benchmark catalog and `--model` for each
requested model in the report. Generation accepts a combined catalog too.
Repeat generation or scoring commands to resume completed work. Images are
keyed by model, benchmark, split, prompt ID, variant and draw. A changed inference
recipe requires a new image directory; a changed judge requires a new score
directory. Keep each score collection at `SCORES/benchmark/model/`.

`report.md` contains native scores and category tables; `report.json` also
contains exact coverage, missing keys, inference/judge settings, resource
measurements and raw judgment paths. An incomplete entry is distinct from an
explicitly pending model. Supply the full catalogs when interpreting a result
as a full benchmark result. There is no aggregate across the five benchmarks.

GenEval averages its six categories. DPG's overall propagates question
dependencies; its native category readout uses unfiltered answers from the last
draw. OneIG keeps five dimensions, painting styles and style-family means.
TIIF retains short/long, attribute groups, GNED and OCR recall. Lower GNED is
better; OneIG diversity measures variation rather than image quality. Qwen retains its
five dimensions and hierarchical total, which differs from averaging the five
displayed dimension means. Raw VLM reasoning and detector decisions remain
available for reviewing surprising scores.

When both Inko entries are scored, the report computes their paired difference
on common source prompts. Bootstrap intervals keep draws, description variants
and scorer dimensions together and recompute native aggregates. These intervals
describe prompt variation. Each entry's selected CFG is shown with the comparison.
To append the final EMA, generate and score only `inko_posttrain`, then rerun the
report with `--model inko_posttrain` instead of `--pending inko_posttrain`.
