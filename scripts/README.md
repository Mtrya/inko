# Project tools

Run Python commands from the repository root as `python -m scripts.<area>.<name>`.
Each command documents its arguments through `--help`. Run corpus processing
where the data is mounted. Platform setup and credentials are in the local
`INSPIRE.md`; training settings come from one complete run TOML.

## Posttraining preparation (`scripts/posttrain/`)

| Command | Purpose |
|---|---|
| `posttrain.prepare_assets` | Download or verify one scorer's source and weights, record checksum inventories, and write local service assets. |
| `posttrain.prepare_metric` | Produce the LPIPS diversity metric asset on a machine with download access. |
| `posttrain.export` | Materialize live or EMA student weights from a complete checkpoint, merge weight LoRA, and verify reload. |
| `posttrain.run_campaign` | Drive one training run with interval evaluations: resume committed segments, export weight views, benchmark suites and assemble blind panels. |
| `posttrain.readouts` | Join a completed blind ballot to benchmark scores; report comparable pairs and scorer/human disagreement. |

The single-stage DiffusionART trainer is `python -m src.posttrain.train`; the standalone evaluation
harness is `python -m src.evaluation.monitoring.run`. Commands, config schema
and current limits are in [the posttraining recipe](../notes/posttrain_recipe.md)
and [campaign guide](../notes/posttrain_campaign.md).
## Model release

`python -m scripts.export_model --checkpoint <weights> --output <directory>`
exports pretrained EMA or merged posttraining weights to the public pipeline
format and verifies an exact weight reload. Sampling defaults come from
`configs/inference.json`. The Gradio app and Space metadata are in `demo/`.

## Pretraining (`scripts/pretrain/`)

| Command | Purpose |
|---|---|
| `pretrain.launch` | Launch or resume one curriculum stage from a complete run config. |
| `pretrain/launch.sh` | Set up the Ascend environment and invoke `pretrain.launch`. |
| `pretrain.migrate_checkpoint` | Migrate model/tracking metadata and declared recipe changes on a verified checkpoint copy. |
| `pretrain.plan_buckets` | Fit measured calibration data and plan length boundaries and micro-batches against caption draws. |
| `pretrain.resize_bucket_plan` | Draft fixed-boundary batch/accumulation candidates for validation on the full workload. |
| `pretrain.pin_artifacts` | Record and verify input hashes before a training launch. |

Use `python -m src.pretrain.precompute --help` for VAE precomputation. Specify
the manifest, output, resolution and device for each job. Job wrappers must
check every worker's exit status before declaring success.

Launchers, planners, recipe migration and configured dataset transfer require
`--storage-root`. Prompts and bucket plans are tracked under `configs/` and
resolve from the repository. The runtime root is not stored in the TOML.

Ordinary resume requires the checkpoint-owned SwanLab project and run ID.
`--new-experiment` explicitly restores training state into a fresh experiment
in the configured project; it never modifies the source experiment. Migration
requires `--source-run-id` when the source lacks a tracking record. An explicit
`--source-assets RECORD.json` supplies hash-verified prompt and bucket bytes
when the source recipe paths are unavailable. For filtered row pools,
`--reset-sampler` rebuilds rank-local cycles at the saved curriculum position
and clears queued draws while preserving all other training state. At interior
stage endpoints, migration preserves sampler files because the next stage
constructs its own sampler. The shell launcher uses the package overlay at
`<storage-root>/runtime/python`. See the
[checkpoint migration procedure](../notes/pretrain_recipe.md#launch-checkpoints-and-resolution-transitions).

The planner takes `--config`, `--stage` and `--storage-root`; dataset weights
and caption policy come from that recipe. Calibration is a flat JSON list of
`latent_hw: [H, W]`, `txt_len`, `micro_batch`, `peak_mem_gb` and `ms_per_step`.
Memory is peak allocated GiB (bytes / 2**30); time is milliseconds for the
whole micro-batch DiT forward/backward. An OOM record contains the three shape
fields and `error: "oom"`, without measurements. Calibration inputs must
conform to this schema. Planner estimates require device validation.

## Evaluation (`scripts/eval/`)

| Command | Purpose |
|---|---|
| `eval.run prepare` | Materialize a released benchmark catalog with native metadata and draw seeds. |
| `eval.run generate` | Generate Inko or external-model images from a prepared catalog; resume missing outputs. |
| `eval.run score` | Score saved images with the benchmark's official judges and aggregation. |
| `eval.run report` | Assemble CPU comparison tables, coverage and paired Inko differences from saved judgments. |
| `eval.blind_panel panel` | Assemble blinded comparisons from directories of `<prompt_id>.png` images. |
| `eval.blind_panel tally` | Count wins and ties from a completed ballot. |
| `eval.cfg_steps_sweep` | Generate the prompt suite across a CFG x ODE-steps grid from one checkpoint, one arm directory per cell. |

```bash
python -m scripts.eval.run generate --config configs/evaluation.toml \
  --assets data/evaluation/assets.toml --prompts data/evaluation/prompts.jsonl \
  --model inko_pretrain --output runs/evaluation
```

The local asset TOML supplies `device_type`, `text_encoder`, `vae`, and a
`[checkpoints]` table mapping model asset names to EMA or merged export files.
Launch with the prepared evaluation interpreter; optional `python_paths` lists
package overlays for its workers. Each prompt JSONL row contains `benchmark`,
`split`, `prompt_id`, `variant`, `text`, `metadata` (including `aspect`), and an
explicit `seeds` array, one entry per draw. Metadata retains native benchmark fields.

External models use `[models.ASSET]` entries with a local pipeline `path` and
isolated `python` interpreter. Optional `[models.ASSET.env]` values locate caches.
The command launches that interpreter automatically. Model entries in
`configs/evaluation.toml` preserve native precision, dimensions and pipeline
arguments. Sample records include token counts and truncation; execution records
include component sizes, scheduler settings and peak device/host memory.

Outputs are grouped by model, benchmark and split, with a prompt catalog,
`samples.jsonl` and lossless images. Image paths resolve from the output root.
Each model saves its inference settings and execution measurements. Repeating
the command resumes missing images; changing image-affecting settings requires
a new output directory. Different models can generate concurrently into the
same output root.

Benchmark commands accept `--benchmark geneval|dpg|oneig|tiif|qwen_image`:

```bash
python -m scripts.eval.run prepare --config configs/evaluation.toml \
  --assets data/evaluation/scorers.toml --benchmark geneval \
  --output data/evaluation/geneval.jsonl
python -m scripts.eval.run score --config configs/evaluation.toml \
  --assets data/evaluation/scorers.toml --benchmark geneval \
  --images runs/evaluation --model inko_pretrain \
  --output runs/evaluation/scores/geneval/inko_pretrain
```

The scorer asset TOML has a `[benchmarks.NAME]` table for each selected benchmark.
Each table supplies an official checkout `source`, an isolated `python`
interpreter, and the assets below. Optional `env` entries locate model caches.

| Benchmark | Prepared catalog / draws | Additional assets |
|---|---|---|
| [GenEval](https://github.com/djghosh13/geneval) | Original 553 prompts / four | `detector` directory, `model_config`; OpenAI CLIP ViT-L/14 in its native cache. |
| [DPG](https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench) | Released 1,065 prompts / four | `model`: released mPLUG checkpoint directory. |
| [OneIG](https://github.com/OneIG-Bench/OneIG-Benchmark) | English 1,120 prompts / four | `qwen`, `csd`, `style_encoder`, `dreamsim`, `clip_processor`, `llm2clip_vision`, `llm2clip_text`. |
| [TIIF](https://github.com/A113N-W3I/TIIF-Bench) | Testmini 277 scenes, short and long / one per description | `base_url` for Qwen2.5-VL-72B; `ocr_python` and `ocr_options` for PaddleOCR. |
| [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) | Released 1,000 IDs, English / one | `data`: released prompt JSONL; `model`: Q-Judger checkpoint directory. |

Scoring constructs native layouts from saved image records. It writes
`scores.jsonl` with the prompt, draw records, raw judgments and elapsed time.
`summary.json` contains native aggregates and expected/evaluated prompt-dimension
counts; `execution.jsonl` records scorer load time and peak CUDA allocation.
Resume skips published prompt-dimension results. Missing images or scorer
failures stop scoring before a complete summary is published. Use a new score
directory when changing the judge or its settings; generation is independent.

## Figures (`scripts/figures/`)

| Command | Purpose |
|---|---|
| `figures.benchmark_figure` | Render benchmark overview bars and the posttraining gain radar from a benchmark `report.json`. |
| `figures.caption_lengths` | Render the data chapter's caption-length bands and cumulative profile from archived 256p metadata. |
| `figures.generate_pretrain_chapter_figures` | Render training stability and loss figures from `scripts/figures/data/`. |
| `figures.generate_pretrain_infra_figures` | Render batch-plan, memory and throughput figures from `scripts/figures/data/` and `configs/`. |

Run `node scripts/figures/export_pretrain_architecture.mjs` to export the
pretraining architecture diagram. Figure outputs default to `notes/assets/`.

Raw figure inputs live locally under `scripts/figures/data/` and are ignored by
Git. Supply these files before running the plotters in a fresh checkout.

### Data chapter figures

Reproduce the chart and pipeline diagram in [the data chapter](../notes/data.md):

```bash
python -m scripts.figures.caption_lengths
node scripts/figures/export_rollout_pipeline.mjs
```

The chart reads the 15 original NumPy sidecars in
`scripts/figures/data/caption_lengths/`. They contain
3,284,692 per-caption measurements from the archived 256p snapshot. The
`prompt_lengths` arrays measure retained Qwen3 prompt tokens after formatting,
prefix removal and truncation, not raw caption tokens. Each caption counts once;
training sampling weights are not applied. The historical “fewer than 100”
annotation comes from the chapter; these files contain only the enriched pool.
Use `--data-dir` to select another sidecar directory or `--output` to save elsewhere.

The diagram exporter reads [`figures/rollout_pipeline.html`](figures/rollout_pipeline.html).
It requires Node.js, Chromium and the `puppeteer-core` dependency installed with
Archify. `--browser`, `--archify-dir` and `--output` override their defaults.
After editing the diagram's JSON source, regenerate the HTML before exporting:

```bash
node ~/.agents/skills/archify/bin/archify.mjs render architecture \
  scripts/figures/rollout-pipeline.architecture.json \
  scripts/figures/rollout_pipeline.html
```

## Caption production (`scripts/caption/`)

Use `caption.select_rows` → `caption.generate` → `caption.freeze_captions` →
`data.apply_caption_enrichment`. Freezing consolidates retries, measures retained
length with the training tokenizer, applies the text-quality checks and emits
one measured record per image. `caption.review` packs image/caption bundles
for human review.

## Data acquisition and transfer (`scripts/data/`)

| Command | Purpose |
|---|---|
| `data.fetch_pexels` | Harvest resumable Pexels query batches with subject and shape filters. |
| `data.fetch_inat` | Harvest iNaturalist photos for count-capped `term \| cap` query files. |
| `data.fetch_museum` | Harvest Met and AIC public-domain artworks for count-capped `term \| cap` query files. |
| `data.fetch_commons` | Harvest Wikimedia Commons JPEG photos for count-capped `term \| cap` query files. |
| `data.fetch_gbif` | Harvest GBIF occurrence photos (taxon-resolved, iNat reexport excluded) for count-capped `term \| cap` query files. |
| `data.fetch_openverse` | Harvest Openverse CC-licensed photos for count-capped `term \| cap` query files. |

| Command | Purpose |
|---|---|
| `data.build_precompute_manifest` | Build D2–D4 training manifests and their held-out evaluation rows. |
| `data.apply_caption_enrichment` | Attach frozen captions and length metadata to an existing precomputed dataset. |
| `data.transfer_precomputed` | Upload a configured stage's datasets or download a repository snapshot through Hugging Face or ModelScope. |
| `data.publish_hf_d1` | Stage D1 metadata and optional image parquet, then optionally publish one release. |
| `data.publish_hf_pexels` | Publish photo references, attribution and captions. |

For D1, `publish_hf_d1 --out-dir <fresh-directory>` writes the metadata table;
`--images` adds bounded image-parquet shards. `--upload --repo-id OWNER/DATASET`
publishes the staged release to an existing repository. An image release
replaces its image-parquet shard set; a metadata-only release updates the table.

```bash
python -m scripts.data.transfer_precomputed upload \
  --provider hf --repo-id OWNER/DATASET --config configs/pretrain.toml --stage 896p --storage-root /external/inko
python -m scripts.data.transfer_precomputed download \
  --provider modelscope --repo-id OWNER/DATASET --storage-root /external/inko
```

Uploads include every training source and the evaluation set in the selected
stage. `--storage-root` selects the external data/model/output tree.
Downloads fetch the selected repository. Credentials come from `HF_TOKEN`, or
`MS_TOKEN` / `MODELSCOPE_TOKEN_PATH` for ModelScope.

Benchmark comparison reports and interpretation are in [model evaluation](../notes/model_evaluation.md).
