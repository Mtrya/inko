# Posttraining recipe

Status: the full 6.1.1 specification is closed after independent review. A partial
6.1.2 sampling/scoring harness exists; independent scorer services and
target-runtime qualification remain pending. No final training recipe is selected and
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
| Teacher | Preserved 480k EMA, SHA256 pinned in config, at 640p | Real generation/scoring comparison |
| Student sampling | 8-step clean-prediction/renoise, shifted uniform grid, conditional CFG 1 | Generation quality and subsequent train/inference correspondence |
| Distribution matching | Online fake-score learning with backward simulation | Parameterization, normalization, update ratio, trainable parameter scope |
| Student tuning | Both full-parameter and LoRA training are in the initial implementation scope | Separate training/state/export qualification and later controlled quality/cost comparisons |
| Cold start | RT-DM, DynaDG, DynaRS | Score transform, adapter design, schedules, and transition timing |
| Joint objective | NFT with persistent DMD regularization | Loss reductions/balance, NFT old-policy update, and any additional regularization |
| Optimization | Configured separately for trainable roles | Optimizers, learning rates, accumulation, clipping, precision, and EMA |
| Rewards | HPSv3 and VQAScore v3.1 with Qwen3-VL-8B-Instruct; source/weight pins and separate raw outputs | Runtime qualification, training mixture and normalization |
| Prompts | Prepared 512/32 rollout and 200/16 DrawBench manifests, plus the 48-entry visual panel | Semantic overlap review and later reward group size |
| Execution | Ascend with resident learner/rollout and HPSv3/VQAScore worker groups from the first version | Role-map memory/throughput, bounded score queues, distributed updates, and recovery; 12/2/2 is an unqualified candidate |
| Scorer boundary | Independently runnable services behind a versioned shared client; configuration selects services and channels | Replace current backend-specific dispatch and qualify config-only substitution |
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

Prepare scorer sources and weights in a network-enabled environment with
`python -m scripts.posttrain.prepare_assets --help`. It pins revisions and writes
an asset map; it does not modify installed dependencies. The machine-local asset
JSON names `checkpoint` (weight file), `text_encoder`, `vae`, `device_type`, and
`hpsv3`/`vqascore` objects containing `source`, `weights`, and ordered
`python_paths`. HPSv3 also requires `backbone`; an optional `learner.python_paths`
selects its dependency overlay. Keep this file under ignored internal storage.
Existing `bin/` directories inside those overlays are also prepended to the
worker's `PATH`; VQAScore requires a working `ffmpeg` executable even for image
scoring. Overlay contents, including that executable, enter the run identity.

HPSv3's source expects its Transformers 4.45-era runtime; VQAScore v3.1 targets
Transformers 5.x. Qualify separate role overlays without changing pretraining's
runtime. The VQA adapter overrides device placement and local artifact loading,
while retaining the pinned upstream scoring/preprocessing implementation.
The HPS adapter makes its FP32 reward-head input cast explicit across devices
and retains raw `mu` and `log_sigma` outputs. Only `mu` enters its preference
report. These adapters still need real scorer inference qualification.
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
identity. A changed config, source, dependency identity, or prompt suite requires
a new output directory. Each arm writes PNGs named by prompt ID, suitable for
the existing `scripts/eval/blind_panel.py` workflow. Scores and reports retain
both metric scales separately; no reward mixture is applied by this harness.

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
