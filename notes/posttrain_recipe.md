# Posttraining recipe

Status: Stage 6.1.2 sampling/scoring qualification is complete, including
independent review and real 640p bringup, monitor and development suites. No final training recipe is selected and
the optimizer/trainer is not implemented yet. The canonical config is
`configs/posttrain.toml`; its current scope is the sampling/scoring baseline.

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
| `configs/posttrain.toml` | Complete implemented baseline; currently sampling/scoring, extended with each training increment |
| `configs/experiments/posttrain/` | Complete configs for retained experimental comparisons |
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
| Execution | Ascend with resident learner/rollout and HPSv3/VQAScore worker groups from the first version | 12/2/2 fits sampling and development scoring; training-state memory, distributed updates and systematic throughput tuning remain unqualified |
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

Current work is layer 1, sampling and scoring. Use a preserved
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
sampling cases and a separate shared budget ledger. Its external watchdog
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
scoring and diversity across seeds remain unqualified. The shared implementation
allowance has consumed 0.779 of 24 allocated node-hours, including failed work,
startup and retained allocation time; both jobs are confirmed released.

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
