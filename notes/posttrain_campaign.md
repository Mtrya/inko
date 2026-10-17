# Posttrain campaign execution

The trainer runs one DiffusionART stage on a fixed base model. A configuration defines the experiment;
the [contract](posttrain_contract.md) defines the loss and the
[recipe](posttrain_recipe.md) records the selected settings and their evidence.
The runner never reduces image size, sample counts, update counts, model
capacity or precision after a failure.

## Training controls

A training config is TOML or JSON with one schema. Missing required fields or
unknown fields fail validation. Optimizer defaults and reward epsilon live in
`src/posttrain/config.py`; explicit overrides are allowed. Parsed
configs contain the effective values for resume. There is no config inheritance or environment
override. `configs/posttrain.toml` is the reference run.

The final configuration uses 64 prompts x 16 draws per round,
500 rounds and one 16-device learner node. A separate 16-device scoring node
hosts seven TP=2 Qwen replicas and one device each for HPS/VQA. The measured
estimate, including ten evaluations, is 5.17 days; allow 6.18 days for 20%
slower rounds. The recipe records the measurements and rejected runtime changes.

| Section | Meaning |
|---|---|
| `initialization` | Base-model asset ID, resolving to the exact weight file through the local `--assets` map. |
| `student` | `full` tuning, or weight LoRA with rank, alpha and target set (`attention` or `attention_mlp`). |
| `collection` | HF dataset, prompts per round, draws per prompt, aspect shapes, sampler step count, shift and text exit layer. |
| `art` | The ART reference coefficient and nothing else: guidance, two update windows, mask and time weights are the paper's constants. Required for every run. |
| `student_optimizer` | `adamw` or `muon_adamw` with constant learning rates and optional overrides for betas, epsilon, weight decay, momentum and clipping. |
| `execution` | Microbatch, timeouts, precision, activation checkpointing, RoCE and deterministic-kernel policy. |
| `training` | Seed, total collected rounds, checkpoint interval, recent-copy count and one `ema_decay`. The EMA stays on each learner device. |
| `evaluation` | Interval, suites, exported weight views, sampler, shapes, prompt selection, scoring channels and bootstrap controls. |
| `scoring`, `rewards` | HTTP timeout, retries and bounded pending submissions; named reward channels, weights and `none`/`round` standardization. All seven channels are higher-is-better. |

`configs/scoring_server.toml` owns model/source assets, device placement,
concurrency and inference limits. Prepare model files on a machine with Hub access:

```bash
python -m scripts.posttrain.prepare_assets --config configs/scoring_server.toml \
  --artifact-root /external/reward-models --output /external/scoring-assets.json
```

Each backend's asset entry can supply `python` and ordered `python_paths` for
its installed dependencies. The qualified Ascend deployment uses vLLM-Ascend
0.26.0/Torch 2.10 for Qwen and isolated Torch/torch_npu 2.9 environments for
HPS/VQA. HPS uses Transformers 4.45.2; VQA uses 5.16.1. VQA starts with
`python -S`, so its explicit paths must include all dependencies. The vLLM
image's eager Torch 2.10 path produced wrong tensor results for these auxiliary
models. Model preparation does not install these environments.

Launch the service separately:

```bash
python -m src.posttrain.scoring.service --config configs/scoring_server.toml \
  --assets /external/scoring-assets.json --output /external/scoring-service \
  --bind 0.0.0.0:8100
```

SIGTERM stops the service and its owned model processes. Restart with the same
command. Its asset map may contain a service `token`; the learner must use the
same token when authentication is enabled.

The learner asset map supplies `scoring.endpoint` and an optional `scoring.token`.
It also supplies `rollout` and `development` prepared JSONL paths. Preparation
uses the authoritative HF dataset plus row-keyed production aspect metadata:

```bash
python -m scripts.posttrain.prepare_prompts --config configs/posttrain.toml \
  --metadata configs/prompts/rollout.jsonl --output /external/rollout
```

The five rubric rewards encode win/tie/loss as `1/.5/0`, after merging both
presentation orders. They use `none` standardization. HPSv3 and VQAScore use
`round` standardization; each of the seven coefficients starts at 1/7.

A round is `prompts_per_round * draws_per_prompt` samples that must divide
evenly into two windows per visible learner. Shape-compatible microbatches may
have a shorter final batch. `training.rounds` and
`training.checkpoint_every` count collections, including skipped ART rounds.
Active rounds perform two optimizer updates.
ART collection is conditional-only because training uses the rollout's cached
predictions. The round must divide evenly into two windows across learners
(`prompts_per_round * draws_per_prompt % (2 * learners) == 0`). Validate before
launch:

```bash
python -m src.posttrain.train --config configs/posttrain.toml --validate-only
```

## Launch, stop and resume

```bash
python -m src.posttrain.train --config configs/posttrain.toml \
  --assets /external/posttrain-assets.json --output /external/run
python -m src.posttrain.train --config configs/posttrain.toml \
  --assets /external/posttrain-assets.json --output /external/run \
  --stop-after 100
python -m src.posttrain.train --config configs/posttrain.toml \
  --assets /external/posttrain-assets.json --output /external/run \
  --resume /external/run/checkpoints/round-000100 --stop-after 200
```

The coordinator connects to the running scoring service and spawns one resident
learner per visible device. The service stays up across training and evaluation. `--stop-after` ends the
run after the requested number of collected rounds without changing the
configured horizon, so a long horizon can be reached in segments. `--track` logs to the `inko`
SwanLab project; a recovered run keeps its experiment identity.

`--resume` requires an unchanged parsed training contract (config, prepared prompts, world size and scorer semantics) and restores the latest saved checkpoint: roles, optimizer state, clock,
prompt cursor, per-rank RNG and last reward. Work after that checkpoint is
discarded and recollected. A run with committed checkpoints refuses to start
without `--resume`; a run that stopped before its first publication restarts
from the declared initialization. Checkpoints are named
`round-<committed-collection-count>`. Checkpoint manifests retain the separate
optimizer step. Skipped collections can publish checkpoints at the same
optimizer step without overwriting earlier recovery points. `--stop-after`
and the horizon always publish a checkpoint.

`training.checkpoint_keep_last` keeps that many recent complete checkpoints
after each successful publication; `0` keeps all. The reference config keeps
two, saving every 20 rounds and at evaluation/final boundaries. Retention counts
collected rounds, including skipped rounds with unchanged optimizer step.
Copy selected recovery states outside the run's `checkpoints/`
directory before they age out. Interval exports and evaluation artifacts stay
available for final checkpoint selection.

## ART round mechanics

- Collection rolls out the live student and caches the visited latent and the
  velocity the rollout produced at every sampler step. The cache is fp32,
  host-resident and sharded per learner rank, and a rank's shard is released
  when the round's updates end.
- Guidance is a per-prompt reward-endpoint covariance over the round. Learners
  reduce reward sums and counts, reward extrema, then centered covariance
  products and absolute deviations before any direction exists. A prompt
  group with exactly constant rewards is inactive.
- Two optimizer updates follow each active round. Each learner shuffles its
  draws with a private round-keyed RNG and splits them into equal disjoint
  windows. Both windows train all of their trajectory sites against the same
  frozen targets.
- With no active guidance and `reference_coefficient = 0`, the collection is
  recorded as skipped. Collection IDs and seeds advance, while optimizer, LR
  and student EMA state stay fixed. An enabled reference penalty still trains.
- Update records add `window`, `art_items`, `art_kept` and `art_masked`, and
  tracking publishes inactive and trust-region-rejected fractions over all
  trajectory items in each update. Their remainder is the accepted fraction.

`execution.microbatch` counts trajectory sites per student forward. Update cost
scales with the trained sites: each learner runs two windows of
`local_draws/2 * steps` items. Collection and scoring still process one terminal
image per draw.

## Evaluation cadence

The training config's `evaluation` section controls interval evaluation.
`every` counts collected rounds, including skipped ART rounds; the final
round is always evaluated. `suites` defaults to development, monitor and rubric; `weights` defaults to EMA.
Rubric evaluation compares one student view with the entry model.

`steps`, `guidance`, `shift` and `shapes` describe evaluation generation,
independently of the training rollout. The base and each exported student use
these same settings. Precision, batch size, text exit layer and
timeouts come from training. Evaluation uses two in-flight batches per learner
and LPIPS pair batches of two. `channels` selects named auxiliary metrics; `diversity_asset` resolves the LPIPS model in the assets map.
Bootstrap repeats and confidence are configurable; `bootstrap_seed` also
controls blind-panel ordering.

`suite` points to `configs/prompts/drawbench.jsonl`. The reference run selects
the first 100 scenes with `prompt_limit` and expands four fixed seeds per scene
with `draws_per_prompt`. A limit of zero uses all rows; one draw preserves the
manifest's original seed. `monitor_suite` points to `monitor.jsonl`; its 48
entries are used unchanged for visual comparisons. Development evaluation
includes within-prompt LPIPS, so it needs at least two seeds per caption.

```bash
python -m scripts.posttrain.run_campaign --config configs/posttrain.toml --validate-only
python -m scripts.posttrain.run_campaign --config configs/posttrain.toml \
  --assets /external/posttrain-assets.json --output /external/campaign --track
```

The driver trains one segment per stop, then lets learners exit before exporting
and evaluating. The independently launched scoring service remains resident. It reuses complete exports
and reports on restart and retries interrupted evaluation before training
continues. Each checkpoint produces `exports/<view>`, a directory per suite,
a `<suite>-blind` panel and `evaluation.json`.

Standalone preparation and final assessments use local sampling configs under
`data/` with `python -m src.evaluation.monitoring.run --config <sampling config>`.
These declare `sampling`, `execution`, `evaluation`, `scoring` and `arms`.
Each arm names its checkpoint asset, weight view, steps, guidance and shift.
`evaluation.suite` supplies the prompt manifest for the chosen `--suite`
label; `--suite monitor` uses `monitor_suite` and skips automatic scoring.
`evaluation.diversity.suites` selects labels that compute LPIPS. Each caption
in those suites needs at least two seeds.

The `rubric` suite samples `rubric_prompt_limit` rows from the prepared development
split with fixed seeds. It compares the entry model and student in both orders,
keeps optional style references, and reports five category wins/ties/losses and
order-disagreement counts plus
raw auxiliary scores. Entry images are reused across checkpoints. The training
pairwise mean is always 0.5; this cross-model comparison measures change.
`--stop-after N` on the campaign driver evaluates the bounded endpoint too.

## Blind readouts

Complete a blind panel's ballot separately, then join human choices to the
scorer deltas:

```bash
python -m scripts.posttrain.readouts join --run /external/benchmark \
  --answer-key /external/blind/answer_key.json \
  --ballot /external/blind/ballot.json --output /external/disagreement.json
```

Use the panel filenames written by `scripts.eval.blind_panel`. The join requires
a complete ballot, preserves channel direction, and reports comparable pairs
with scorer-up/human-down counts instead of inventing missing judgments. A
ballot that chooses a third arm cannot order another arm against the reference.

## Cost and scope

Every arm declares its question, its decision and its budget before launch.
SwanLab uses four groups:

| Group | Readouts |
|---|---|
| `monitor` | One labeled grid per scene, with language/length variants as rows and declared arms as columns; separate images for each seed draw. Within-prompt LPIPS is labeled by suite and arm. |
| `evaluation` | Complete benchmark means and paired deltas with confidence bounds. Deltas are oriented so positive means improvement. Each suite's fixed baseline is logged once. Full blind panels and per-prompt statistics stay in the evaluation artifacts. |
| `train` | Objective loss, weighted reference contribution when enabled, pre-clipping gradient norm, and one applied LR per optimizer component. ART updates also report inactive and rejected item fractions. Collection readouts carry raw channel mean/std and active-guidance group fractions. |
| `runtime` | Collection, scoring-wait, guidance (ART), round and update durations; scored-image and skipped-collection counts; backend calls, inference seconds and failures; maximum learner process peak in GiB. |

Update metrics and evaluation images use optimizer step. Collection reward and
group diagnostics use a hidden collection-number axis and are sent even when
ART skips the collection. Runtime metrics use cumulative measured execution
seconds, with evaluation time accumulated separately. Manual restart downtime
is excluded. Scoring counts are snapshots of completed backend calls for the current
collection. Transport retries reuse the accepted sample identity.
Stage durations take the maximum over learners. Collection time includes
conditioning, decoding and image publication. Round time ends at collection
commit; cumulative execution time also includes startup and checkpoint writing.

`telemetry.json` preserves measured time and the baseline logging set between
training segments and evaluations. Tracked checkpoints copy the latest
snapshot for recovery into another output directory. The evaluation sampling
settings, requested channels and selected weight views are recorded in the
SwanLab configuration; image captions retain full prompts and seeds. Monitor
previews limit each image's longest side to 512 pixels; full generated images
remain in the benchmark artifacts.

Read the full reward curve and the internal metrics before declaring any
result: an endpoint score alone cannot select a variant, and HPS can rise on
visibly degraded outputs.
