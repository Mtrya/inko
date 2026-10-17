# Posttraining recipe

Selected training recipe: DiffusionART on the 600k pretrain EMA at 896p,
attention LoRA rank 64/alpha 64, Muon LR 1e-3, no reference anchor and
500 collected rounds (1000 optimizer updates). Export the EMA (0.999).
`configs/posttrain.toml` contains the full configuration. The selection campaign
used two rewards and smaller collections; its breadth/held-out assessment is
complete. The coming run uses 64 prompts x 16 draws and seven reward channels
on the HF rollout pool. The execution envelope below uses the extended time budget.

See the [mathematical contract](posttrain_contract.md),
[evaluation protocol](posttrain_evaluation.md) and
[campaign execution guide](posttrain_campaign.md).

## Selection campaign inputs

| Input | Value |
|---|---|
| Base | 600k pretrain checkpoint, EMA weights, decoded at 896p |
| Rollout | 20 Euler ODE steps, CFG 1.0, shift `shape`, 12 prompts x 8 draws per round |
| Campaign rollout prompts | 512-prompt PickScore subset |
| Reward | HPSv3 `mu` and VQAScore v3.1 `probability`, equal weights, round-standardized |
| Optimizer | Muon on LoRA A/B matrices; LR 1e-3, momentum 0.95, constant schedule, no weight decay, gradient-norm clip 1.0 |
| Reference anchor | Disabled, coefficient 0 |
| Student | Attention LoRA, rank 64, alpha 64 |
| Student EMA | One shadow, decay 0.999 per optimizer update; interval reads use `ema` |

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
| `src/evaluation/monitoring/` | Training diagnostics and independently runnable interval reward/LPIPS comparisons |
| `src/evaluation/generation/` | Shared sampling and standalone image generation |
| `configs/posttrain.toml` | ART training and interval evaluation settings |
| `configs/prompts/` | Tracked DrawBench and monitor suites; local `rollout.jsonl` is ignored by Git |
| `configs/scoring_server.toml` | All reward-model sources, weights, placement and execution limits |

Training configs accept TOML or JSON with one schema; unknown or missing
required fields fail. Optimizer settings and reward epsilon have implementation
defaults, which are included in the parsed resume contract. Machine-local paths, endpoints and credentials live in the
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
  --weights ema --output /external/export
```

`--stop-after` ends a segment at a complete collection round without changing
the configured horizon; `--track` logs the run to the `inko` SwanLab
project. Export views are `live` or `ema`; `ema` is the default. Weight LoRA
is merged and the reloaded weights are verified exactly.

The completed campaign used named fast/slow shadows. Current checkpoints store
one EMA, controlled by `training.ema_decay`; the retained campaign's selected
merged export above remains directly loadable.

The separate scoring service owns Qwen pairwise rubric judging, HPSv3 and
VQAScore. Learners send encoded images as each microbatch completes. Training
waits for all seven channels before assembling rewards and updating the student.
Both-order decisions and reasons remain with the raw score records; parse or
inference failures abort the collection. The [campaign guide](posttrain_campaign.md)
contains the launch and data-preparation commands.

## Execution envelope

Use one 16-device learner node and one 16-device scoring node. The scorer runs
seven Qwen TP=2 replicas at concurrency 80 each, plus HPS and VQA on one device
each with batch size two. Keep medium thinking, sampled decoding and both
judging orders. A selective answer-masking runtime change removes full-batch
logit transfers; the prepared Ascend environment contains it.

Larger Qwen tensor parallelism and TP=1 replicas were slower. Speculative
decoding failed under real image load, and lower thinking effort saved no
tokens. Disabling thinking was faster but confused references with candidates
and invented objects. Learner batch, compilation and cast-cache comparisons
supplied no usable further speedup. Only then was collection size reduced.

| Prompts x draws | Measured round time | 500 rounds before overhead |
|---|---|---|
| 80 x 48 | 2,531.13 s | 14.65 days |
| 80 x 16 | 935.95 s | 5.42 days |
| 80 x 12 | 846.54 s mean over two rounds | 4.90 days |
| 80 x 10 | 679.26 s mean over three rounds | 3.93 days |
| **64 x 16 selected** | **869.45 s estimated from 80 x 10** | **5.03 days estimated** |

The 80 x 10 rounds took 639.95, 635.16 and 762.66 seconds; the last included
a long final judge response. Scale their mean by 1,024/800 for the selected
64 x 16 collection. Add seven seconds for each of 30 checkpoint publications,
ten complete 1,000-second evaluations, ten 180-second learner starts and six
minutes for scorer startup:
**124.2 hours / 5.17 days**, about **3,974 NPU-hours**. With round times 20%
longer, the estimate is **148.3 hours / 6.18 days**. Use a seven-day job limit.
The exact 64 x 16 workload was not timed; the heavier 80 x 16 round established
capacity, while three 80 x 10 rounds exercised the complete recovery/evaluation
path. Neither requires another capacity run for this intermediate workload.

The extended budget permits 16 draws per prompt instead of the five-day
80 x 10 fallback. The final 64 x 16 collection uses 1,024 images per round,
with more candidates for each prompt's reward covariance and fewer prompt groups.
It preserves 20 steps, resolution, seven channels, 1,000 updates and both judging
orders. Over 500 rounds it collects 512,000 images and trains 10.24 million
trajectory sites, versus 1.92 million images and 38.4 million sites at 80 x 48.
Final quality and sustained duration remain to be measured in the monitored run.

Learners use microbatch 4, BF16 forward with FP32 parameters/optimizer state/
reductions, activation checkpointing off, deterministic kernels and intra-host
RoCE. DDP wraps only the student. Startup primes each wrapper's reduction
layout with a zero-loss backward and restores RNG before initialization or
resume. The 80 x 10 qualification peaked at 25.44 GiB per learner; trajectories
occupied 23.55 GiB across all 16 learners, projecting to 30.1 GiB at 64 x 16.
Two retained recovery states and ten interval
exports/reports require about 60 GiB of storage. Checkpoints publish every 20
rounds and at evaluation/final boundaries.

All seven channels completed without failure in the selected rounds; all
prompt groups supplied active guidance. The supported driver completed the
entry/EMA benchmark path and resumed committed training through step six.
Reviewed images were nearly unchanged, with shared base-model weaknesses in
lettering, counting and geometry. This checks execution and gross update errors,
not long-term reward effectiveness.

The completed campaign's 896p, 20-step, 12-prompt by 8-draw envelope completed qualification
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
