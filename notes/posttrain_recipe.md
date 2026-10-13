# Posttraining recipe

Status: DiffusionART is selected for the remaining recipe comparisons after
complete 1000-round ART/NFT curves on the 600k pretrain EMA at 896p. The
anchor coefficient, learning rate, trainable scope and final checkpoint remain
open. `configs/posttrain.toml` is the unanchored ART comparison configuration.

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
| Optimizer | Muon for matrices plus AdamW for conditioning tensors |
| Student | LoRA r=32 on attention projections (full tuning is the alternative) |
| Student EMA views | Fast 0.9 and slow 0.999, updated per optimizer step; interval reads use slow |

The base model's ODE quality is flat between 20 and 50 steps at CFG 1, so 20
rollout steps cost nothing in rollout quality; the deployed step count is a
separate post-training decision. Each sampler step uses one conditional forward.

## Algorithm choice and implemented mechanism

ART carries forward from the paired development comparison and repeated
visual reads. Its HPS/VQA gains exceed NFT across the useful learning regions;
this advantage also appears before ART's largest diversity contraction. For
example, ART round 300 has HPS 6.004, VQA .266 and LPIPS .752; NFT round 800
has .044, .209 and .754. Similar LPIPS does not guarantee the same forms of
diversity, so checkpoint selection also uses actual generations.

ART's later HPS approaches a plateau while LPIPS continues to decline and
some unrelated prompts become floral portraits. Its final checkpoint is not
automatically the selected checkpoint. Shared loss of artistic texture alone
is not a harmful-collapse criterion. Final quality assessment remains reserved
for the settled recipe and its held-out confirmation.

The completed [ART run](https://swanlab.cn/@mtrya/inko-posttrain/runs/85ut8u5j)
and [NFT control](https://swanlab.cn/@mtrya/inko-posttrain/runs/2t30rxqx) retain
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

## Open values

| Value | Current setting | Decision it needs |
|---|---|---|
| Anchor coefficient | Off at 0.0; first on-state 1e-4 | Compare paired development curves, generations, adherence onset and diversity to determine whether it controls drift without capping gains. |
| Muon learning rate | `muon_lr = 1e-3` | Fit {3e-4, 1e-3, 3e-3} by learning trajectory and numerical stability. The LoRA arm has no trainable AdamW branch. |
| Trainable scope | LoRA r=32 | Compare full tuning at the settled configuration, starting its AdamW branch at 1e-4. LoRA shares the frozen reference base; the anchor still requires reference forwards. |
| Run length | 1000 rounds used for the algorithm comparison | Select the final horizon from the full quality/diversity trajectories; confirm the settled recipe once. |

Reward-hacking evidence is handled scorer-side or data-side first; the anchor
coefficient is the second lever.

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
  --checkpoint /external/run/checkpoints/round-000100 \
  --weights live --output /external/export
```

`--stop-after` ends a segment at a complete collection round without changing
the configured horizon; `--track` logs the run to the `inko-posttrain` SwanLab
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
and the full algorithm comparison. The ART run used sixteen Ascend devices
(twelve learners and four scorer devices), took 26.74 allocated hours including
twenty interval evaluations, and used 427.81 NPU-hours. Its final collection
round took 75.57 seconds. Enabling the reference adds forwards; its incremental
cost is measured in the anchor comparison.

Mechanism tests in `tests/` cover the operator values, group statistics, role
isolation, scheduling, prediction admission and exact save/resume. They cannot
and do not establish generation quality or convergence.
