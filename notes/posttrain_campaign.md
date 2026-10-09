# Posttrain campaign execution

The trainer can express the campaign's initial fitting, estimator and optimizer
comparisons, named student EMA views, cold-to-joint branches and periodic
evaluation. A configuration defines the experiment. The runner never reduces
image size, sample counts, update counts, model capacity or precision after a
failure. The [mathematical contract](posttrain_contract.md) defines the losses
and state transitions. The [recipe note](posttrain_recipe.md) distinguishes
implementation evidence from unresolved training choices.

## Training controls

Training configurations accept TOML or JSON with the same exact schema. Complete
examples live in `configs/posttrain/campaign_qualification/`. They are bounded
execution checks, not fitted recipe values. Their rollout source is the
512-prompt pool `configs/prompts/posttrain_rollout.jsonl`.

| Control | Meaning |
|---|---|
| `fake.initial_steps` | Total fake optimizer updates before the first cold student update. Zero uses the regular ratio from the start. |
| `fake.initial_checkpoint_every` | Publish recovery state at this initial-fitting interval. The requested fitting stop and the end of fitting also publish. |
| `fake.updates` | Fake updates before each subsequent cold student update and each joint DMD update. |
| `fake.target_refresh` | `round` retains collected targets; `update` regenerates targets for each fake update from the held student and same prompts. |
| `fake_probe.every`, `fake_probe.draws` | Optional fresh-sample regression diagnostic cadence and independent collection count. `every = 0` disables it. |
| `objective.estimator` | `teacher_residual`, `clean_difference` or `score_kl`. The contract defines their different scales. |
| `student_optimizer`, `fake_optimizer` | Independent AdamW or Muon/AdamW settings. Muon supports full matrices and adapter matrices. |
| `training.ema` | A list of `{name, decay, device}` student shadows. Use distinct names; `live` is reserved. Device is explicitly `learner` or `cpu`. |

For example, `initial_steps = 128` and `updates = 5` means 128 fake updates
before student update 1, then five before each later student update. Initial
fitting does not move the student or EMA, consume rollout prompts, acquire
rewards or advance a generator schedule. It has its own recoverable clock.
The fresh probe measures regression on new current-student samples. It cannot
certify the true student score or image quality.

Validate a complete training input before launch:

```bash
python -m src.posttrain.train \
  --config configs/posttrain/campaign_qualification/cold_full.json --validate-only
```

An explicit fitting boundary can be inspected and resumed in new processes:

```bash
python -m src.posttrain.train --config /external/training.json \
  --assets /external/assets.json --output /external/run --stop-after-fit 16
python -m src.posttrain.train --config /external/training.json \
  --assets /external/assets.json --output /external/run \
  --resume /external/run/checkpoints/fit-000016
```

Exact resume requires unchanged source, config, runtime and input assets. A
committed cold checkpoint can initialize a separate joint run through
`stages.initial_checkpoint` and its manifest SHA. Initialization imports live
student and active fake weights and resets optimizers, EMA, cursor and RNG.
It is different from resume. The fake lineage uses actual optimizer updates,
including initial fitting. Joint branches may choose a different fake ratio.

Declare the long joint horizon before launching a composition arm. Stop at a
complete collection round, evaluate it, and resume the same horizon if retained.
A completed joint checkpoint cannot initialize a longer run. Changing the
horizon changes the training contract and its schedules.

## Evaluation and readouts

The benchmark has four explicitly labeled suite slots: development, bringup,
monitor and confirmation. `prepare_prompts` reserves confirmation prompts after
excluding rollout, development and monitor texts under Unicode, case and
whitespace normalization. This checks exact normalized overlap, not semantic
paraphrases. Repeated-seed manifests preserve scene and caption identity.

Conditional diversity uses LPIPS-AlexNet on all unordered seed pairs for each
exact caption. It keeps the original image resolution and never pairs caption
variants. Caption means are averaged within a scene; scenes have equal weight.
Each caption needs at least two distinct seeds. A single-seed panel is rejected
when diversity is enabled.

Prepare metric weights on a machine with download access:

```bash
python -m scripts.posttrain.prepare_metric \
  --output /external/lpips-alex.pt --cache /external/torch-hub
```

Copy the complete metric asset to the execution machine. Set the benchmark's
`evaluation.diversity.asset` and `sha256`, and map that asset in
`assets.metrics`. Evaluation uses the pinned complete weights without downloads.
Pair batch size and CPU thread count are explicit settings. Raw reward reports
include standard deviation, range and quantiles alongside scene-level paired
comparisons. Reward scores remain descriptive measurements, not human judgments.

An interval input names the sampling config and its SHA, generator-step cadence,
suite names, weight views, final-evaluation choice and blind shuffle seed. See
`configs/posttrain/campaign_qualification/interval.json`. Every stop must be a
complete collection round. The cold boundary always receives an evaluation.

```bash
python -m scripts.posttrain.run_campaign --config /external/training.json \
  --interval /external/interval.json --validate-only
python -m scripts.posttrain.run_campaign --config /external/training.json \
  --interval /external/interval.json --assets /external/assets.json \
  --output /external/campaign
```

Run the same command after interruption. The driver checks committed checkpoints,
finishes their required evaluations and resumes training. Each segment releases
all training processes before generation and scoring. Exports carry their actual
view names and EMA decays. A failed evaluation blocks advancement. Existing blind
ballots are preserved on restart. `complete.json` is published only after the
required final work succeeds.

If startup stops before the first checkpoint is published, the matching run
manifest permits a restart from the pinned initialization. Uncommitted work has
no recoverable state. Once a checkpoint is published, training requires explicit
resume; the campaign driver supplies it. Changed contracts are rejected in both
cases.

Each checkpoint evaluation produces exports, per-suite benchmark outputs and a
blind panel. Complete its ballot separately, then join human choices to scores:

```bash
python -m scripts.posttrain.readouts join --run /external/benchmark \
  --answer-key /external/blind/answer_key.json \
  --ballot /external/blind/ballot.json --output /external/disagreement.json
```

Use the actual panel filenames written by `scripts.eval.blind_panel`. The join
requires complete ballots and preserves score direction. A choice of a third
arm does not order another arm against the reference. It reports comparable
pairs and scorer-up/human-down counts without inventing missing judgments.

## Execution qualification

The pinned plan in `configs/posttrain/campaign_qualification/plan.json` uses two
representative full-model cases. The cold case checks initial-fitting recovery,
Muon, two EMA exports and serialized interval evaluation. The full-fake joint
case checks fresh fitting targets, joint entry and the largest resident-role
memory footprint. It keeps all training updates and evaluates the cold boundary
and joint endpoint.

Both cases use `teacher_residual`. This plan does not exercise the other
estimators, an NFT-only comparison, cross-run cold checkpoint branching, or
student-LoRA with Muon. Those interfaces have separate mechanism or historical
execution evidence. They are not additional coverage of this pinned plan.

Memory qualification does not require launching every estimator or composition.
Under the same resolution, microbatch, precision and retained graph size, full
weights, an independent full fake model, old/reference models and two EMA shadows
bound the lighter role combinations. Larger fake-update ratios repeat backward
passes sequentially and do not retain their graphs. Separate mechanism checks
cover behavior that this memory bound cannot establish, including adapter Muon,
state restoration and readout admission. Include optimizer state and temporary
buffers when comparing memory: AdamW keeps two moments, while Muon keeps one
momentum buffer for its matrices. A new resource envelope requires a new check;
a new scalar loss coefficient alone does not.

```bash
python -m scripts.posttrain.qualify_campaign \
  --plan configs/posttrain/campaign_qualification/plan.json --validate-only
python -m scripts.posttrain.qualify_campaign \
  --plan configs/posttrain/campaign_qualification/plan.json \
  --assets /external/assets.json --output /external/qualification
```

The plan pins config bytes; edits require deliberate new pins and an output
directory with the new identity. The harness records each resolved cold source
and config before execution. Any failure stops the plan. Qualifying these paths
establishes execution under their recorded runtime and placement. It does not
establish convergence, a winning setting or safety at arbitrary scales and
horizons. New resource envelopes need their own bounded qualification with the
intended scientific workload preserved.
