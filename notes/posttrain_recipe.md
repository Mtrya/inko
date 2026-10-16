# Posttraining recipe

Selected training recipe: DiffusionART on the 600k pretrain EMA at 896p,
attention LoRA rank 64/alpha 64, Muon LR 1e-3, no reference anchor and
500 collected rounds (1000 optimizer updates). Export the slow EMA (0.999).
`configs/posttrain.toml` contains the full configuration. The experiment
campaign and final breadth/held-out confirmation are complete.

See the [mathematical contract](posttrain_contract.md),
[evaluation protocol](posttrain_evaluation.md) and
[campaign execution guide](posttrain_campaign.md).

## Fixed inputs

| Input | Value |
|---|---|
| Base | 600k pretrain checkpoint, EMA weights, decoded at 896p |
| Rollout | 20 Euler ODE steps, CFG 1.0, shift `shape`, 12 prompts x 8 draws per round |
| Rollout prompts | 512-prompt pickscore subset in `configs/prompts/posttrain_rollout.jsonl` |
| Reward | HPSv3 `mu` and VQAScore v3.1 `probability`, equal weights, round-standardized |
| Optimizer | Muon on LoRA A/B matrices; LR 1e-3, momentum 0.95, constant schedule, no weight decay, gradient-norm clip 1.0 |
| Reference anchor | Disabled, coefficient 0 |
| Student | Attention LoRA, rank 64, alpha 64 |
| Student EMA views | Fast 0.9 and slow 0.999, updated per optimizer step; interval reads use slow |

The pretrain CFG×steps sweep was flat between 20 and 50 steps at CFG 1,
which supports the 20-step rollout choice. Confirmed inference uses the same
20 Euler steps, CFG 1, shape shift and BF16 at 896p. Each sampler step uses
one conditional forward. LoRA trains only A/B matrices, so the optimizer's
AdamW component has no parameters in this scope.

## Algorithm choice and implemented mechanism

ART carries forward from the paired development comparison and repeated
visual reads. Its HPS/VQA gains exceed NFT across the useful learning regions;
this advantage also appears before ART's largest diversity contraction. For
example, ART round 300 has HPS 6.004, VQA .266 and LPIPS .752; NFT round 800
has .044, .209 and .754. Similar LPIPS does not guarantee the same forms of
diversity, so checkpoint selection also uses actual generations.

ART's later HPS approaches a plateau while LPIPS continues to decline and
some unrelated prompts become floral portraits. Select round 500 instead of
the 1000-round endpoint to limit that contraction.

The completed [ART run](https://swanlab.cn/@mtrya/inko/runs/85ut8u5j)
and [NFT control](https://swanlab.cn/@mtrya/inko/runs/2t30rxqx) retain
the score, diversity, optimizer and runtime trajectories. Their interval
artifacts retain blind monitor and development comparisons. Both completed
1000 collected rounds with no skipped rounds; ART performed 2000 optimizer
updates and NFT 1000. Both used slow EMA decay 0.999 per optimizer step;
ART therefore advances its EMA twice per active round. The algorithm comparison
includes this difference in EMA age.

The active trainer implements unit-RMS reward-endpoint covariance guidance,
two disjoint update windows per round and directional trust-region masking.
Targets are the cached rollout predictions plus each prompt group's guidance.
The reference transition noise level is 0.8; the KL threshold and epsilon
floors are 1e-5. An optional frozen-base velocity MSE composes with the ART
loss. These equations are defined in the [contract](posttrain_contract.md).

The 1e-4 reference anchor reduced training reward from about round 470 and
did not preserve diversity or prevent unrelated portrait substitutions. At
round 600 its development HPS was .496 lower and within-scene LPIPS .00965
lower than unanchored ART; both paired scene intervals excluded zero. VQA
did not separate clearly. Select no anchor for the remaining comparisons.

## Scope, rate and rank selection

LoRA at 1e-3 continues to gain quality and adherence through round 500 while
retaining more diversity than the higher LoRA rate. Full tuning learns faster,
but its later slow exports lose adherence and substitute similar portraits
for unrelated subjects. At full 1e-3, VQA falls from .2631 at 150 to .1453
at 500; LoRA 1e-3 reaches .3267 at 500. Its LPIPS is .7057 versus .6186
for LoRA 3e-3 and .4688 for full 1e-3. Select LoRA 1e-3 at round 500.

LoRA 3e-4 reaches HPS 8.016, VQA .2880 and LPIPS .7523 at 500.
It retains more diversity but trails 1e-3 by 2.279 HPS and .0387 VQA;
both paired confidence intervals exclude zero. Its live model also develops
the ornate-portrait bias. The lower rate delays learning and this bias without
improving the selected horizon's quality/adherence tradeoff.

Select [attention LoRA rank 64](https://swanlab.cn/@mtrya/inko/runs/sjatwpz3).
Alpha 64 keeps alpha/rank = 1, matching the rank-32 control. At round 500,
rank 64 has HPS 10.408, VQA .3249 and LPIPS .7077; rank 32 has
10.295/.3267/.7057. Paired differences are +.113 HPS [-.061, .292],
-.00175 VQA [-.02980, .02718] and +.00199 LPIPS [-.00333, .00707].
The full curves and fixed development images support comparable results, with mixed
local adherence changes and no broad new rank-64 slow-EMA subject collapse.
Both live models develop the ornate-portrait bias. Use the round-500 slow
export. The final assessment below uses that selected export.

Reward-hacking evidence is handled scorer-side or data-side first; the anchor
coefficient is the second lever.

Full tuning at Muon 3e-3 is rejected for broad mode collapse. Its slow export
loses prompt adherence from round 150 onward; at 350, 31 of the fixed 32
development samples are similar portraits. Full tuning at 3e-4 is closed
after selecting LoRA; its completed reads remain in the comparison evidence.

## Final assessment and selected artifact

Both final suites pair the selected slow EMA with the 600k base using one
fixed seed per prompt and the inference settings above. All 800 images and
1,600 scorer results completed successfully. Confidence intervals are paired
95% prompt-bootstrap intervals, conditional on those seeds.

| Suite | Base HPS / VQA | Selected HPS / VQA | HPS gain [95% CI] | VQA gain [95% CI] |
|---|---|---|---|---|
| [DrawBench breadth, 200 prompts](https://swanlab.cn/@mtrya/inko/runs/8x86zycf) | -4.869 / .1501 | 10.437 / .2993 | +15.306 [14.500, 16.056] | +.1493 [.0848, .2150] |
| [Held-out confirmation, 200 prompts](https://swanlab.cn/@mtrya/inko/runs/ohbpepsi) | .623 / .1429 | 11.099 / .1538 | +10.477 [9.942, 11.034] | +.0109 [-.0317, .0542] |

Preference-score improvement transfers to the holdout. A general adherence
gain remains unproven. Both scorers were optimized during training; direct
image review exposes failures hidden by their aggregate scores.

Reviewed 32 fixed random pairs and the eight largest VQA regressions per
suite, totaling 39 unique pairs in each. People and common objects render
more clearly. Exact text, counting and relations remain weak. The holdout
also shows genuine posttrain-induced regressions: crevices, magic relics and
factory creatures become ornate portraits where the matched base retains
the scene or object; a wooden cat becomes furry. Requested illustration styles
and actions often become static realistic portraits. These failures qualify
the selected recipe's quality/adherence tradeoff. Confirmation closes the
campaign without retuning.

The selected run is `e23-lora-r64-1e-3`, checkpoint
`training/checkpoints/round-000500`, view `slow` (decay .999 per update).
Its ready-to-load merged model is
`evaluation/round-000500/exports/slow/weights.pt`, alongside
`transformer_config.json`. Export merges LoRA into the base model; inference
does not require an adapter. The run's SwanLab link above identifies its
curves and configuration; platform paths and detailed reads are in the
internal experiment ledger.

## Configuration layout

| Path | Responsibility |
|---|---|
| `src/posttrain/` | ART trainer, sampling, reward transforms and score clients |
| `src/evaluation/benchmarks/` | Independently runnable benchmark execution and reporting |
| `configs/posttrain.toml` | Complete single-stage ART training config |
| `configs/prompts/` | Rollout, development, monitor and confirmation suites |
| `configs/scorers/` | Per-scorer source, weights and output declarations |

Training configs accept TOML or JSON with one exact schema; unknown or missing
fields fail. Machine-local paths, endpoints and credentials live in the
`--assets` JSON, never in the config. The
[campaign guide](posttrain_campaign.md) describes every config section, launch,
resume and evaluation cadence.

## Commands

Validate a training config without loading any model:

```bash
python -m src.posttrain.train --config configs/posttrain.toml --validate-only
```

Train, resume and export:

```bash
python -m src.posttrain.train --config configs/posttrain.toml \
  --assets /external/posttrain-assets.json --output /external/run
python -m src.posttrain.train --config configs/posttrain.toml \
  --assets /external/posttrain-assets.json --output /external/run \
  --resume /external/run/checkpoints/round-000100
python -m scripts.posttrain.export \
  --checkpoint /external/run/checkpoints/round-000500 \
  --weights slow --output /external/export
```

`--stop-after` ends a segment at a complete collection round without changing
the configured horizon; `--track` logs the run to the `inko` SwanLab
project. Export views are `live` or a named `training.ema` shadow; weight LoRA
is merged and the reloaded weights are verified exactly.

Score services are config-declared independent processes
(`configs/scorers/*.toml`), started by the coordinator when the assets name a
launch and otherwise reached as external endpoints. The client validates
service capabilities against the declared outputs before any rollout, requires
every channel of a round before an update, records explicit failures instead of
zero rewards, and keeps raw per-scorer outputs alongside the mixture.

## Execution envelope

The reference config uses 12 learner devices, microbatch 1, BF16 forward with
FP32 parameters/optimizer state/reductions, activation checkpointing off, and
explicit deterministic kernels plus intra-host RoCE. Learner DDP wraps only the
student; scorers never join collectives. Startup primes each wrapper's
reduction layout with a zero-loss backward and restores RNG before
initialization or resume.

The 896p, 20-step, 12-prompt by 8-draw envelope completed qualification
and all recipe comparisons. The selected rank-64 run completed 500 rounds/
1000 updates with zero skips, taking 13.3075 allocated hours
(212.92 NPU-hours) on sixteen Ascend devices. Peak learner allocation was
9.529 GiB. Final breadth and confirmation added 4.049 NPU-hours.

The full algorithm-comparison ART run used sixteen Ascend devices
(twelve learners and four scorer devices), took 26.74 allocated hours including
twenty interval evaluations, and used 427.81 NPU-hours. Its final collection
round took 75.57 seconds. Enabling the reference adds forwards; its incremental
cost is measured in the anchor comparison.

Mechanism tests in `tests/` cover the operator values, group statistics, role
isolation, scheduling, prediction admission and exact save/resume. They cannot
and do not establish generation quality or convergence.
