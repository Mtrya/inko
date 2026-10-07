# Posttraining recipe

Status: Plain distribution matching is implemented alongside the independent
sampling/scoring pipeline. Full-parameter and LoRA students, two fake-score
strategies, three explicit DMD gradients, learner DDP, optimization, recovery and
student export are available. No final training recipe is selected.
`configs/posttrain.toml` defines sampling/scoring; `configs/posttrain_dmd.toml`
defines plain DMD. RT-DM, dynamic cold-start schedules and NFT remain later work.

See the [mathematical contract](posttrain_contract.md),
[evaluation protocol](posttrain_evaluation.md), and
[project roadmap](redesign_plan.md).

## Starting baseline

The release target is an 8-step Inko student. The initial method follows
[DMDR](https://arxiv.org/abs/2511.13649v5) with:

1. Cold start: reward-tilted distribution matching (RT-DM), dynamic distribution
   guidance (DynaDG), and dynamic renoise sampling (DynaRS).
2. Joint training: [DiffusionNFT](https://arxiv.org/abs/2509.16117v2) with a
   persistent DMD term to anchor reward optimization to the teacher distribution.

This is an experimental starting point. Comparisons can change objectives,
components, optimization, sampling, rewards, data, or introduce custom methods.
The DMD/DMD2 component comparison is one possible question, not the scope of the
research. GAN supervision, alternative RL objectives, and targeted SFT remain
open possibilities requiring a concrete motivation and comparison.

## Implementation and config layout

| Path | Responsibility |
|---|---|
| `src/posttrain/` | Trainer, objectives, sampling, model states, reward/score clients, and reward transforms |
| `src/evaluation/benchmarks/` | Independently runnable benchmark execution and reporting |
| `configs/posttrain.toml` | Complete sampling/scoring baseline |
| `configs/posttrain_dmd.toml` | Complete plain-DMD training configuration |
| `configs/experiments/posttrain/` | Complete coverage configurations; later retained comparisons |
| `configs/prompts/` | Rollout and evaluation prompt suites with explicit split identity |

Sampling/scoring paths and initial prompt manifests are implemented. Validate
required fields and incompatible component combinations. Runs preserve the complete resolved config and
artifact identities. Machine-specific paths, services, and credentials belong
in local project configuration.

## Settings awaiting selection

| Area | Starting intent | Evidence or decision needed |
|---|---|---|
| Teacher | Preserved 480k EMA, SHA256 pinned in config, at 640p | Qualified starting control; final teacher transfer remains a later comparison |
| Student sampling | 8-step clean-prediction/renoise, shifted uniform grid, conditional CFG 1 | Generation quality and subsequent train/inference correspondence |
| Distribution matching | Online fake-score learning with backward simulation | Parameterization, normalization, update ratio, trainable parameter scope |
| Student tuning | Both full-parameter and LoRA training are in the initial implementation scope | Separate training/state/export qualification and later controlled quality/cost comparisons |
| Cold start | RT-DM, DynaDG, DynaRS | Score transform, adapter design, schedules, and transition timing |
| Joint objective | NFT with persistent DMD regularization | Loss reductions/balance, NFT old-policy update, and any additional regularization |
| Optimization | Configured separately for trainable roles | Optimizers, learning rates, accumulation, clipping, precision, and EMA |
| Rewards | HPSv3 and VQAScore v3.1 with Qwen3-VL-8B-Instruct; source/weight pins and separate raw outputs | Training mixture and normalization; broader scorer validity remains unestablished |
| Prompts | Prepared 512/32 rollout and 200/16 DrawBench manifests, plus the 48-entry visual panel | Semantic overlap review and later reward group size |
| Execution | Ascend with resident learner/rollout and HPSv3/VQAScore worker groups from the first version | 12 learner replicas fit the qualified DMD paths; export evaluation uses 12/2/2. Systematic throughput tuning remains separate |
| Scorer boundary | Independently runnable services behind a versioned shared client; configuration selects services and channels | Generic collection/report substitution implemented and mechanism-tested; future production VLM backend needs its own qualification |
| Final selection | Real generations, benchmarks, diversity, and cost | Comparison protocol, repeat budget, and acceptable regressions |

## Work layers and qualification

Stage 6 has four work layers, separate from the recipe's two training stages:

| Layer | Scope | Exit evidence |
|---|---|---|
| 1. Algorithm implementation and selection | Contracts, minimal prompts/scorer/benchmark, implementation, experiments, and comparisons | A selected algorithm with real generation/benchmark evidence and explicit limits |
| 2. Hardware efficiency | Profile and optimize the complete selected pipeline on target hardware | Measured end-to-end efficiency, memory, recovery, and checks that optimizations preserve algorithm semantics and observed quality |
| 3. Real-run preparation | Curate real-run prompts and evaluation suites; select/calibrate scorers and reward composition | Frozen input/scorer identities and a bounded pipeline qualification, including remeasurement if scorer speed changes |
| 4. Real run and monitoring | Launch, monitor, recover, evaluate, and select the final checkpoint | Completed run, final held-out evaluation, preserved artifacts, and a qualified release recipe |

Current work is layer 1, algorithm implementation. Use a preserved
pretrained checkpoint and a small working suite; algorithm development does not
wait for the hero-run prompt pool or a particular pretraining endpoint. Keep
initialization fixed within comparisons and assess transfer when moving to a
different teacher or resolution. The completed pretrained model remains the
intended source for the final run.

Initial algorithm experiments use 640p with the selected 480k EMA checkpoint
held fixed across variants. The initial scorer set is HPSv3 and VQAScore v3.1
with Qwen3-VL-8B-Instruct, without a custom VLM-rubric scorer.
Initial prompt-pool and benchmark choices are recorded in the
[evaluation note](posttrain_evaluation.md); their manifests are in `configs/prompts/`.

Inside layer 1, implement sampling/scoring, minimal distribution matching,
complete cold start, then joint NFT+DMD. Add each mechanism with a measurable
control before broader comparisons. Resident model roles, explicit learner/scorer
rank allocation, bounded in-round scoring concurrency, and correct recovery
belong in the first implementation. Qualify one viable allocation with measured
memory and round time; systematic tuning of the split, batching, transport, and
kernels belongs in layer 2. Routine model loading/offloading on every worker is
not the initial execution design. See the
[rank allocation contract](posttrain_contract.md#resident-roles-and-rank-allocation).
The [small evaluation design](posttrain_evaluation.md) defines the initial scope.

Layer 3 can require recalibrating batching/concurrency after scorer changes.
If new data or scorer evidence challenges the algorithm choice, reopen that
specific comparison explicitly rather than silently changing the selected
algorithm during preparation. Measure the final workload before setting the
real-run budget.

Checkpoint/resume must cover the roles and clocks in the
[contract](posttrain_contract.md). A successful mechanism check establishes
execution behavior; quality selection follows the
[evaluation protocol](posttrain_evaluation.md).

## Sampling/scoring commands

Validate the config and selected prompt suite without model dependencies:

```bash
python -m src.evaluation.benchmarks.run --config configs/posttrain.toml --validate-only
```

Prepare each scorer with its complete `configs/scorers/<name>.toml` using
`python -m scripts.posttrain.prepare_assets --help`. Preparation pins source and
model revisions, checks upstream file digests, and writes local service assets.
It does not install packages. `--assets` verifies existing files;
`--assets` plus `--artifact-root` builds an isolated view containing only pinned
model members. Supply native-runtime version files through `--runtime-files`.

Caller assets name `checkpoints` (a map from arm asset IDs to weight files),
`text_encoder`, `vae`, `device_type`, `learner.python_paths`, and `services`.
NPU learners also supply `learner.runtime_files`, a map of native-runtime
version-file names to paths; their content hashes enter the run identity.
Each service ID has `endpoints` and `launches`; an empty launch list connects to
an independently managed endpoint. Each owned launch supplies `config`, `assets`,
`device`, `bind`, and `python_paths`. Service assets contain `source`, `weights`,
`weights_manifest`, `device_type`, and `runtime_files`; HPS also requires
`backbone` and `backbone_manifest`. All locations are machine-local JSON data.
The initial placement is learners 0–11, HPS replicas 12–13 and VQA replicas 14–15.

A service can run without a coordinator:

```bash
python -m src.posttrain.scoring.service --config configs/scorers/hpsv3.toml \
  --assets /external/hps-assets.json --device 12 --bind 127.0.0.1:8100 \
  --cache /external/hps-cache
```

The service process needs its own dependency overlay on `PYTHONPATH`; owned
launches configure it automatically. Overlay `bin/` directories enter `PATH`.
These explicit Python paths must include the image's native compiler modules
as well as package overlays. In particular, Ascend inference needs CANN's `tbe`
paths; importing Torch and loading weights alone does not exercise that compiler.
The VQA source requires `ffmpeg` even for image scoring. Relevant package
versions and native-runtime version-file hashes are pinned in service semantics
and validated before loading; changing them changes the scorer fingerprint.
Other installed packages remain execution provenance.

HPSv3's source expects its Transformers 4.45-era runtime; VQAScore v3.1 targets
Transformers 5.x. Qualify separate role overlays without changing pretraining's
runtime. The VQA adapter overrides device placement and local artifact loading,
while retaining the pinned upstream scoring/preprocessing implementation.
The HPS adapter makes its FP32 reward-head input cast explicit across devices
and retains raw `mu` and `log_sigma` outputs. Only `mu` enters its preference
report. Both adapters have produced real 640p scores and passed service restart/cache reuse.
Upstream Qwen scoring currently iterates over images inside a request; request
batching alone does not establish batched model throughput.

Run the 16-prompt, three-arm bring-up benchmark:

Launch one coordinator per host; it spawns the configured resident processes.
This sampling-only command does not use `torchrun` or training collectives.

```bash
python -m src.evaluation.benchmarks.run --config configs/posttrain.toml \
  --assets /external/posttrain-assets.json --output /external/posttrain-bringup \
  --suite bringup
```

Use `--suite development` for all 200 DrawBench prompts and `--suite monitor`
for the 48-entry visual-only panel. Monitor mode starts only generation workers.
Reusing the same output directory resumes verified records under the same input
identity. A changed semantic config, source, scorer identity, or prompt suite requires
a new output directory. Endpoint relocation preserves score identity. Each arm writes PNGs named by prompt ID, suitable for
the existing `scripts/eval/blind_panel.py` workflow. Scores and reports retain
both metric scales separately; no reward mixture is applied by this harness.

For implementation smokes, `scripts.posttrain.qualify` consumes explicit
cases with an explicit `kind` (`sampling` or `dmd`) and a shared budget ledger. Its external watchdog
reserves one 16-NPU job, charges from provider allocation through confirmed
release, and stops near the cumulative 24-node-hour limit with shutdown margin.
It imposes no independent per-job cap. Provider/accounting uncertainty blocks
subsequent launches. The launcher is not a variant-selection sweep tool.

## Sampling/scoring qualification

The fixed 480k EMA control produced 48 bringup images, 144 visual-monitor images,
and 600 development images. HPSv3 and VQAScore each returned all 648 requested
results with no failures; HPS `log_sigma` remains a diagnostic channel. All three
arms use the same checkpoint, CFG 1 and their configured shifted grids. No
posttraining updates or variant-selection experiments were run.

DrawBench development means (200 prompts per arm, one fixed seed per prompt):

| Sampling arm | HPSv3 `mu` | VQAScore probability |
|---|---:|---:|
| Euler, 50 steps | -3.32768 | 0.153911 |
| Undistilled Euler, 8 steps | -7.06620 | 0.119670 |
| Undistilled clean-prediction renoise, 8 steps | -9.98205 | 0.021698 |

Blinded-position inspection covered all bringup and monitor images and a
24-prompt development subset selected without scores. The 50-step control gives
recognizable detailed scenes but frequently misses objects, counts, relations
and text. Euler8 often ghosts or washes out structure; untrained renoise8 often
produces low-information textures or silhouettes. Retain the latter as a weak
starting control, not a usable distilled sampler. These observations do not
select a training method or establish independent reward-model validity.

On one 16-NPU Ascend node, the resident 12 learner / 2 HPS / 2 VQA allocation
fit batch size 2. Maximum recorded allocated memory per process was 5.64 GiB
for generation, 16.17 GiB for HPS and 16.57 GiB for VQA. These are allocator
peaks, not whole-device HBM usage or training/optimizer-state requirements.
The 600-image development pipeline took 85.1 seconds after readiness and
228.1 seconds including resident loading; complete invocation took 248.4
seconds. Mean client acquisition per two-image batch was 0.538 seconds for
HPS and 0.365 seconds for VQA. This qualifies one working allocation; it is not
an optimized throughput claim.

Both real services were restarted and reused disk-cached scores through fresh
clients. Partial final score records were restored through external endpoints
without regenerating images. A completed development resume preserved all
1,801 report/record/score files and started no new sampling workers. Mechanism
coverage separately verifies config-only substitution by an unfamiliar service
and channel through the same collection, joining and reporting code.

Scope limits: actual images cover 640×640, 560×736 and 736×560; the two wider/taller
configured shapes remain unqualified. Visual review was by the implementation
agent, not a human preference study, and only 24 of the 200 development prompts
were inspected in that panel. The monitor is visual-only; bilingual automatic
scoring and diversity across seeds remain unqualified. Sampling/scoring qualification consumed 0.779 allocated node-hours, including
failed work, startup and retained allocation time; its two jobs were released.
Later training qualification uses the same cumulative 24-node-hour allowance.

## Plain-DMD commands and controls

Validate a complete training recipe without model loading:

```bash
python -m src.posttrain.train --config configs/posttrain_dmd.toml --validate-only
```

Training uses the same machine-local checkpoint/text/runtime assets as sampling.
It starts resident learners and no scorer processes; plain DMD does not acquire
rewards. Export evaluation calls the existing independent scoring services.

```bash
python -m src.posttrain.train --config configs/posttrain_dmd.toml \
  --assets /external/posttrain-assets.json --output /external/dmd-run
python -m src.posttrain.train --config configs/posttrain_dmd.toml \
  --assets /external/posttrain-assets.json --output /external/dmd-run \
  --resume /external/dmd-run/checkpoints/round-000002
python -m scripts.posttrain.export \
  --checkpoint /external/dmd-run/checkpoints/round-000004 \
  --weights ema --output /external/dmd-export
```

`--stop-after` permits an earlier committed boundary without changing the
configured schedule horizon. Resume requires the same source, algorithm,
runtime, prompts, artifacts and learner map. Export directories are immutable;
use their actual weight/architecture hashes in benchmark arms. Both `live` and
`ema` export views are supported, including LoRA merging.

The three complete coverage configs exercise different required mechanisms:

| Config | Student / fake | DMD / sampling / optimizer |
|---|---|---|
| `configs/posttrain_dmd.toml` | Full / teacher-base Q/V | Teacher-residual normalization; renoise; AdamW; selected-clean fake target |
| `configs/experiments/posttrain/dmd_lora_full.toml` | Attention+MLP LoRA / full | Clean difference; Euler; two fake updates; terminal target; LR warmup/cosine |
| `configs/experiments/posttrain/dmd_full_muon.toml` | Full / teacher-base Q/V | Score-KL time factor; renoise with CFG; Muon+AdamW |

These configurations change several axes to cover execution paths cheaply.
Their short runs cannot isolate treatment effects or rank variants. Student
evaluation EMA is enabled to qualify export and recovery, not as a selected
training choice. The shared Muon implementation lives in `src/optim/` and remains
used by pretraining.

Execution declares deterministic kernel policy and intra-host RoCE explicitly.
On the qualified Ascend topology, learner devices cross two interconnect planes;
HCCL required intra-host RoCE. Native fused-attention backward was nonrepeatable
under the default policy on identical BF16 inputs. The deterministic policy
removed that difference in a direct repeated-backward probe. DDP bucket
rebuilding also changed native reduction rounding; startup primes both wrappers
with a zero-loss backward, clears gradients and restores RNG before initialization
or resume. Full recovery evidence is recorded separately. Activation checkpointing has mechanism
coverage; it is disabled in the initial hardware cases.

## Plain-DMD qualification

Three 640×640 training cases used 24 samples per round and 12 learner replicas:
four full-student/Q/V-fake updates, four LoRA-student/full-fake updates, and two
full-student Muon/score-KL/CFG updates. Every trainable student/fake tensor changed;
frozen teacher and LoRA base tensors remained unchanged. These cases exercise
different mechanisms and cannot isolate or select variants.

Both four-update paths were repeated with a process exit at committed round 2
and a new-process resume to round 4. Role weights, optimizer state, EMA, clocks,
per-rank RNG/cursors, recorded inputs and updates matched the uninterrupted run
exactly under the declared deterministic execution policy and primed DDP layout.
This establishes repeatability on the exercised hardware/runtime, not across
different hardware, software, rank maps or arbitrary future kernels.

All six live/EMA exports reloaded exactly and were evaluated against matching
initial controls on 16 DrawBench prompts. The final evaluation produced 160
images and 320/320 required HPSv3/VQAScore results. Means below retain separate
metric scales; each cell is HPS `mu` / VQAScore probability.

| Coverage path | Matching initialization | Live export | EMA export |
|---|---:|---:|---:|
| Full / Q/V fake, renoise CFG 1 | -10.23899 / 0.000257 | -10.05377 / 0.000007 | -10.23961 / 0.000162 |
| LoRA / full fake, Euler CFG 1 | -8.18376 / 0.007350 | -8.19105 / 0.013606 | -8.13965 / 0.011561 |
| Full Muon / Q/V fake, renoise CFG 1.5 | -9.40974 / 0.087945 | -9.97678 / 0.004905 | -9.40572 / 0.014797 |

Thumbnail inspection covered all 160 final images with shuffled arm positions.
The initial controls already have poor prompt following. The full/Q/V live export
shows repetitive texture collapse across prompts; its EMA preserves the weak
initial silhouettes. HPS rises despite this visible regression. LoRA exports
remain visually close to the weak short-Euler control; the Muon case also fails
to establish usable renoise quality. These configurations are unqualified as
release recipes. Operator and execution evidence does not establish convergence,
scorer validity or successful distillation. Broader development suites, multiple
seeds and controlled comparisons remain separately budgeted selection work.

Maximum allocated memory per learner was 19.79 GiB for full/Q/V, 19.97 GiB for
LoRA/full fake and 20.94 GiB for full Muon/Q/V. After startup, observed round
times were approximately 1.9, 2.5–2.8 and 3.2 seconds respectively, excluding
checkpoint I/O. Different objectives, fake-update ratios and guidance prevent
interpreting these as a tuning-mode speed comparison. Export evaluation took
226.7 seconds including model/service loading and cleanup. These measurements
qualify one resident placement, not optimized throughput or whole-device HBM.

The numerical training/recovery/export runs used the recorded source before a
checkpoint-admission-only guard was added. That guard rejects unpublished staging
directories even if a manifest exists; its regression reproduced the old failure
and passed after correction. The final guard also verified all actual committed
full/LoRA/Muon checkpoint files. Numerical source and final source identities are
preserved separately; no resume under mismatched source is claimed.

All implementation jobs are released. The shared allowance totals 1.987
of 24 allocated node-hours, leaving 22.013; failures, preparation and
retained allocations are included. No RT-DM, dynamic controls, NFT or selection
experiments were run in this phase.

## Selected recipe and decision evidence

Pending. For each selected setting, record its rationale, comparison conditions,
observed effect, cost, and limitations here. Publish enough evidence to understand
the choice without access to an internal experiment ledger. Link the complete
config for each selected training recipe rather than duplicating its numerical
settings by hand.

## Final qualification and release artifacts

Pending. Record the selected checkpoint identity, exact config and prompt
versions, 8-step inference settings, final evaluation results and sample panels,
domain/diversity limitations, measured training/inference cost, and recovery
qualification. The final recipe is selected only after these results exist.
