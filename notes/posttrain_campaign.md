# Posttrain campaign execution

The trainer runs one DiffusionART stage on a fixed base model. A configuration defines the experiment;
the [contract](posttrain_contract.md) defines the loss and the
[recipe](posttrain_recipe.md) records the objective mechanisms and open values.
The runner never reduces image size, sample counts, update counts, model
capacity or precision after a failure.

## Training controls

A training config is TOML or JSON with one exact schema. Missing or unknown
fields fail validation; there is no config inheritance and no environment
override. `configs/posttrain.toml` is the reference run.

| Section | Meaning |
|---|---|
| `initialization` | Asset ID and weight view (`live` or `ema`) of the base model. The asset ID resolves through the local `--assets` map. |
| `student` | `full` tuning, or weight LoRA with rank, alpha and target set (`attention` or `attention_mlp`). |
| `collection` | Prompt suite, prompts per round, draws per prompt, resolution, sampler step count, shift and text exit layer. |
| `art` | The ART reference coefficient and nothing else: guidance, two update windows, mask and time weights are the paper's constants. Required for every run. |
| `student_optimizer` | `adamw` or `muon_adamw` with learning rates, betas, weight decay, clip limit and LR schedule. |
| `execution` | Learner devices, microbatch, timeouts, precision, activation checkpointing, RoCE and deterministic-kernel policy. |
| `training` | Seed, total collected rounds, checkpoint interval in rounds, and named EMA shadows with decay and storage device. |
| `scorers`, `rewards` | Declared score services with acquisition limits, and the reward channels with direction, weight and standardization. |

A round is `prompts_per_round * draws_per_prompt` samples that must divide
evenly across learner devices and microbatches. `training.rounds` and
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

The coordinator writes the run manifest, starts the declared scorer services,
and spawns one resident learner per configured device. `--stop-after` ends the
run after the requested number of collected rounds without changing the
configured horizon, so a long horizon can be reached in segments. `--track` logs to the `inko-posttrain`
SwanLab project; a recovered run keeps its experiment identity.

`--resume` requires an unchanged parsed training contract (config plus prompt
suite) and restores the last complete round: roles, optimizer state, clock,
prompt cursor, per-rank RNG and last reward. Work inside an uncommitted round is
discarded and recollected. A run with committed checkpoints refuses to start
without `--resume`; a run that stopped before its first publication restarts
from the declared initialization. Checkpoints are named
`round-<committed-collection-count>`. Checkpoint manifests retain the separate
optimizer step. Skipped collections can publish checkpoints at the same
optimizer step without overwriting earlier recovery points. `--stop-after`
and the horizon always publish a checkpoint.

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

Interval evaluation is declared by a separate config with one exact schema.
`sampling_config` names the benchmark's own arm declaration (schema:
`sampling`/`execution`/`evaluation`/`scorers` blocks plus an `arms` list, each
arm declaring `name`, `checkpoint`, `steps`, `guidance`, `shift`), and `weights` selects the
checkpoint views the driver exports and adds as arms:

```json
{
  "sampling_config": "/external/evaluation-sampling.json",
  "every": 50,
  "suites": ["development", "monitor"],
  "weights": ["live", "slow"],
  "include_final": true,
  "blind_seed": 7
}
```

`weights` names `live` or a `training.ema` shadow. `every` counts collected
rounds, including skipped ART rounds. The driver validates the interval
together with the training config:

```bash
python -m scripts.posttrain.run_campaign --config configs/posttrain.toml \
  --interval /external/interval.json --validate-only
python -m scripts.posttrain.run_campaign --config configs/posttrain.toml \
  --interval /external/interval.json --assets /external/posttrain-assets.json \
  --output /external/campaign
```

The driver trains one segment per declared stop, then lets every learner and
owned scorer process exit before it exports weight views and runs the benchmark
suites. Exports and reports are reconciled from already committed checkpoints
on restart, so an interrupted evaluation is retried rather than skipped; a
failed evaluation blocks further training. `complete.json` appears only after
the final required work succeeds.

Each evaluated checkpoint produces exports under `exports/<view>`, one directory
per benchmark suite, a blind panel under `<suite>-blind`, and an
`evaluation.json` receipt naming the exported views and suites. The benchmark
itself runs standalone with `python -m src.evaluation.benchmarks.run --config
<sampling config>`; `--suite monitor` is visual-only and skips automatic
reporting.

Suites and their prompt files come from `configs/prompts/`. Sampling configs
select each suite's manifest. The recurring development manifest has 100
DrawBench scenes with four seeds each; the single-seed 200-prompt manifest
provides breadth. `bringup` uses 16 scenes with four seeds each,
`monitor` the 48-entry visual panel, and `confirmation` the held-out final
suite. `scripts.posttrain.prepare_prompts` rebuilds them from the pinned
reference prompt source and keeps rollout, development and monitor texts
disjoint under Unicode, case and whitespace normalization. That check is exact
text overlap, not semantic paraphrase detection. `evaluation.diversity.suites`
lists the suites that compute LPIPS. Every exact caption in those suites must
have at least two seeds. Omit single-seed monitor, breadth and confirmation
reads from this list.

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
| `runtime` | Collection, scoring-wait, guidance (ART), round and update durations; scored-image and skipped-collection counts; scoring retries/failures; maximum learner process peak in GiB. |

Update metrics and evaluation images use optimizer step. Collection reward and
group diagnostics use a hidden collection-number axis and are sent even when
ART skips the collection. Runtime metrics use cumulative measured execution
seconds, with evaluation time accumulated separately. Manual restart downtime
is excluded. Scoring counts are snapshots for the current collection, measured
per request; a retry attempt is an attempt beyond the first.
Stage durations take the maximum over learners. Collection time includes
conditioning, decoding and image publication. Round time ends at collection
commit; cumulative execution time also includes startup and checkpoint writing.

`telemetry.json` preserves measured time and the baseline logging set between
training segments and evaluations. Tracked checkpoints copy the latest
snapshot for recovery into another output directory. The evaluation sampling
settings, scorer declarations and selected weight views are recorded in the
SwanLab configuration; image captions retain full prompts and seeds. Monitor
previews limit each image's longest side to 512 pixels; full generated images
remain in the benchmark artifacts.

Read the full reward curve and the internal metrics before declaring any
result: an endpoint score alone cannot select a variant, and HPS can rise on
visibly degraded outputs.
