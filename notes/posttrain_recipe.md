# Posttraining recipe

Status: the trainer implements one stage on the 600k pretrain EMA base at 896p,
with the objective selected by the run config — DiffusionNFT with a
soft-refreshed old policy, or DiffusionART with the cached rollout prediction as
its anchor. Both arms share the round workload below and compose with a frozen
reference anchor. No posttraining quality result exists yet; the recipe's open
values are listed below with the comparison each one needs.

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

The base model's ODE quality is flat between 20 and 50 steps at CFG 1, so 20
rollout steps cost nothing in rollout quality; the deployed step count is a
separate post-training decision. One 20-step conditional forward per sampler
step means the rollout does not pay for a second unconditional branch.

## Pinned by the reference

These are structural, not tunables, and follow the reference implementation of
the objective that names them:

- Per-prompt group-normalized advantage, clipped to `+/-5` and mapped linearly
  to the optimality probability `p in [0, 1]` (NFT).
- Implicit-policy guidance strength `beta = 1.0` (NFT).
- Per-branch, per-sample adaptive clean-target weighting with an epsilon floor
  of `1e-5` (NFT).
- One inner epoch per collected group (NFT).
- Old-policy soft refresh from 0 toward 0.5 over the run: a frozen old policy
  stops learning, and a hard copy collapses, per the reference ablation (NFT).
- Unit-RMS reward-endpoint covariance guidance, two update windows per round,
  and the trust-region mask on the reference transition variance with noise
  level `0.8` and KL threshold `1e-5` (ART; constants are not config values).

## Open values

| Value | Current setting | Decision it needs |
|---|---|---|
| Anchor coefficient | `reference_coefficient = 1e-4` | Anchor on/off at the reference value; if ambiguous, fit from {1e-3, 1e-4, 1e-5}. The anchor is the intended defense against reward-hacking drift, so reward curves and monitor generations decide it, not the endpoint score. |
| Muon learning rate | `muon_lr = 1e-3`, AdamW `lr = 1e-5` | Bounded sweep around the pretrain-comparable rate, read as reward-curve stability plus internal metrics. |
| Trainable scope | LoRA r=32 | One comparison against full tuning at the settled configuration. LoRA also makes the reference anchor free. |
| Run length | 1000 student updates | Long-horizon confirmation; checkpoint selection uses the development suite, the monitor panel and blind review. |
| Group size, time draws, EMA decays | 8 draws per prompt, 1 time draw, two named shadows | Not yet studied; each is a config value with a mechanism test, not a fitted recipe value. |

Reward-hacking evidence is handled scorer-side or data-side first; the anchor
coefficient is the second lever.

## Configuration layout

| Path | Responsibility |
|---|---|
| `src/posttrain/` | Trainer, NFT and ART objectives, sampling, reward transforms, score clients |
| `src/evaluation/benchmarks/` | Independently runnable benchmark execution and reporting |
| `configs/posttrain.toml` | Reference single-stage NFT training config |
| `configs/posttrain_art.toml` | Reference single-stage ART training config: same round workload, no old policy |
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
  --resume /external/run/checkpoints/step-000100
python -m scripts.posttrain.export \
  --checkpoint /external/run/checkpoints/step-000100 \
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

Execution mechanics — role separation, complete-round recovery, export,
interval evaluation and the scorer services — were qualified at 640p with 12
learners on the Ascend node. The 896p, 8-draw, 20-step rollout and the 600k EMA
base are a heavier envelope that has not been measured: allocate a bounded
qualification run before the long campaign, and treat memory and round-time
numbers from the earlier envelope as inapplicable. No configuration reduces
image size, sample count, update count, capacity or precision after a failure.

Mechanism tests in `tests/` cover the operator values, group statistics, role
isolation, scheduling, prediction admission and exact save/resume. They cannot
and do not establish generation quality or convergence.
