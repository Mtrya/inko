# Posttrain campaign execution

The trainer runs one stage on a fixed base model, with DiffusionNFT or
DiffusionART selected by the run config. A configuration defines the experiment;
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
| `objective` | Top-level objective selector, `"nft"` or `"art"`. Each value requires exactly its own section and forbids the other. |
| `initialization` | Asset ID and weight view (`live` or `ema`) of the base model. The asset ID resolves through the local `--assets` map. |
| `student` | `full` tuning, or weight LoRA with rank, alpha and target set (`attention` or `attention_mlp`). |
| `collection` | Prompt suite, prompts per round, draws per prompt, resolution, sampler rule, step count, guidance, shift and text exit layer. |
| `nft` | NFT branch parameterization, advantage scaling and clip, `beta`, reduction epsilon, reference coefficient, inner updates per round, time-draw count, time distribution and bounds. Required under `objective = "nft"`. |
| `art` | The ART reference coefficient and nothing else: guidance, two update windows, mask and time weights are the paper's constants. Required under `objective = "art"`. |
| `old_policy` | NFT refresh strategy (`copy`, `constant`, `linear_ramp`), decay values, and the horizon counted in committed rounds. Required under `objective = "nft"` and rejected under `objective = "art"`. |
| `student_optimizer` | `adamw` or `muon_adamw` with learning rates, betas, weight decay, clip limit and LR schedule. |
| `execution` | Learner devices, microbatch, timeouts, precision, activation checkpointing, RoCE and deterministic-kernel policy. |
| `training` | Seed, total student updates, checkpoint interval, and named EMA shadows with decay and storage device. |
| `scorers`, `rewards` | Declared score services with acquisition limits, and the reward channels with direction, weight and standardization. |

A round is `prompts_per_round * draws_per_prompt` samples that must divide
evenly across learner devices and microbatches. `training.steps` and
`training.checkpoint_every` must contain whole rounds of the objective's
updates: `nft.updates_per_round` for NFT, two for ART. An ART run additionally
requires `collection.guidance = 1` (it trains against the rollout's own cached
prediction, so the rollout cannot use CFG) and a round that divides evenly into
the two windows across learners
(`prompts_per_round * draws_per_prompt % (2 * learners) == 0`). Validate before
launch:

```bash
python -m src.posttrain.train --config configs/posttrain.toml --validate-only
python -m src.posttrain.train --config configs/posttrain_art.toml --validate-only
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
  --resume /external/run/checkpoints/step-000100 --stop-after 200
```

The coordinator writes the run manifest, starts the declared scorer services,
and spawns one resident learner per configured device. `--stop-after` ends the
run at a committed boundary without changing the configured horizon, so a long
horizon can be reached in segments. `--track` logs to the `inko-posttrain`
SwanLab project; a recovered run keeps its experiment identity.

`--resume` requires a byte-identical training contract (config plus prompt
suite) and restores the last complete round: roles, optimizer state, clock,
prompt cursor, per-rank RNG and last reward. Work inside an uncommitted round is
discarded and recollected. A run with committed checkpoints refuses to start
without `--resume`; a run that stopped before its first publication restarts
from the declared initialization. Checkpoints are named
`step-<run-local-student-update>`; `--stop-after` and the horizon always publish
one, and a completed run's exports come from those directories.

## ART arm

`configs/posttrain_art.toml` mirrors `configs/posttrain.toml` with
`objective = "art"`, an `[art]` section, and no `[nft]` or `[old_policy]`. Its
round mechanics differ as follows:

- Collection rolls out the live student and caches the visited latent and the
  velocity the rollout produced at every sampler step. The cache is fp32,
  host-resident and sharded per learner rank, and a rank's shard is released
  when the round's updates end. The NFT arm never allocates it.
- Guidance is a per-prompt reward-endpoint covariance over the round. Learners
  reduce reward sums and counts, reward extrema, then centered covariance
  products and absolute deviations before any direction exists. A prompt
  group with exactly constant rewards is inactive.
- Two optimizer updates follow each active round. Each learner shuffles its
  draws with a private round-keyed RNG and splits them into equal disjoint
  windows. Both windows train all of their trajectory sites against the same
  frozen targets. No old-policy refresh happens, so `old_policy_version` stays 0
  and a checkpoint carries no old role.
- With no active guidance and `reference_coefficient = 0`, the collection is
  recorded as skipped. Collection IDs and seeds advance, while optimizer, LR
  and student EMA state stay fixed. An enabled reference penalty still trains.
- Update records add `window`, `art_items`, `art_kept` and `art_masked`, and
  tracking publishes `train/art_masked_fraction` next to the loss terms.

`execution.microbatch` counts items per student forward under both objectives,
so under ART it chunks the trajectory sites of one draw rather than a draw
group. Update cost scales with the trained sites: a round runs two windows of
`draws/2 * steps` forwards per learner, so the ART arm's student forward count
is `steps` times the NFT arm's at the same round workload, while rollout and
scoring costs are identical.

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

`weights` names `live` or a `training.ema` shadow. `every` is a student-update
interval and must fall on complete rounds. The driver validates the interval
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

Suites and their prompt files come from `configs/prompts/`:
`development` is the 200-prompt DrawBench suite, `bringup` its 16-prompt subset,
`monitor` the 48-entry visual panel, and `confirmation` the held-out final
suite. `scripts.posttrain.prepare_prompts` rebuilds them from the pinned
reference prompt source and keeps rollout, development and monitor texts
disjoint under Unicode, case and whitespace normalization. That check is exact
text overlap, not semantic paraphrase detection.

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
SwanLab update records carry the loss, its reference and objective terms,
gradient norm, learning rates, and reward statistics per committed update, plus
the ART masked fraction when that objective is active; the interval evaluation
carries the benchmark and diversity panels. Read the full reward curve and the
internal metrics before declaring any result: an endpoint score alone cannot
select a variant, and HPS can rise on visibly degraded outputs.
