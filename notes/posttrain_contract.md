# Posttraining mathematical and execution contract

Status: the full Stage 6.1 specification gate (6.1.1) is closed after independent
review. This public note describes the initial sampling/evaluation contract and
the required training boundaries. The existing 6.1.2 harness is partial: scorer
decoupling and target-hardware qualification remain pending. Training objectives
are specified for implementation in 6.1.3 through 6.1.5; they are not implemented
or qualified capabilities.
Numerical choices belong in the runnable config; this contract defines their
meaning. See the [recipe](posttrain_recipe.md) and
[evaluation protocol](posttrain_evaluation.md).

## Scope and ownership

Core training logic, rollout sampling, model-state management, and reward/score
clients belong in `src/posttrain/`. Benchmark execution and reporting belong in
`src/evaluation/benchmarks/`. Rollout and evaluation prompt suites belong in
`configs/prompts/`. The initial sampling/scoring harness is present; the
posttraining optimizer/trainer is not implemented yet.

The starting recipe has two training stages: RT-DM with DynaDG and DynaRS,
then DiffusionNFT with persistent distribution-matching regularization.
Alternative objectives, components, optimization methods, and custom techniques
remain experimental choices. Each implemented variant needs an explicit equation
and state-flow description; method names alone do not define behavior.

The initial implementation family includes both full-parameter and LoRA students,
plain DMD and independent RT-DM/DynaDG/DynaRS controls, followed by DMD-only,
NFT-only and joint controls. Implementation qualification and algorithm selection
are separate gates; short smoke runs cannot establish a winning recipe.

Scoring must be independently runnable. Training and evaluation use a shared
versioned client/protocol; scorer services own model loading, preprocessing,
batching, inference and rubric/response parsing. Config selects services and
numeric channels. A protocol-compatible VLM scorer can replace an initial
scorer through caller configuration; an unfamiliar upstream API needs an isolated
adapter. Scorer model/rubric identity enters score provenance and cache identity.
Config-only substitution through collection/evaluation and later reward/NFT
paths is required qualification. The current backend-specific worker dispatch
does not yet satisfy this boundary.

## Flow convention

Let `x` be a clean, normalized VAE latent, `epsilon` standard Gaussian noise,
`c` text conditioning, and `t` the physical flow time. Inko uses:

```text
t = 0: noise                 t = 1: data
x_t = (1 - t) epsilon + t x
v_target = x - epsilon
x_hat = x_t + (1 - t) v_theta(x_t, t, c)
epsilon_hat = x_t - t v_theta(x_t, t, c)
```

The interpolation and velocity target are implemented in
[flow.paths](../src/flow/paths.py). For a reference using noise time `tau = 1 - t`
and velocity `epsilon - x`, both the time direction and velocity sign change.
Translate its noising and clean-prediction equations together.

The existing resolution shift maps a base grid value `u` to physical time
`t = u / (s - (s - 1) u)`, where `s` is the shift. A sampling density and a
shifted grid are separate choices; record both and avoid applying a shift twice.
The existing [ODE sampler](../src/flow/solvers.py) accumulates times and state at
least in FP32. The proposed few-step update below is a separate sampling rule.

## Initial sampling design

Status: selected for the first algorithm-development comparison, with quality
qualification pending. Initial development uses 640p and the preserved 480k EMA.
Its SHA256 identity is pinned in [the config](../configs/posttrain.toml).
Availability of a newer checkpoint does not automatically advance the teacher.

Keep the Inko velocity parameterization for both teacher and student. Use a
preserved pretrained checkpoint, with its architecture, text encoder/template,
token limit, and VAE statistics recorded explicitly. For VAE latent `z`, model
space is `x = (z - mean) / std`; decoding inverts this transform. Text encoder
and VAE weights stay frozen. The selected teacher weight identity remains fixed
through a comparison; a new pretrained snapshot starts a new comparison.

The initial comparison has three arms:

| Arm | Sampling rule | Purpose |
|---|---|---|
| Teacher reference | Existing Euler ODE, initially 50 steps and CFG 1 | Reproduce the established monitoring baseline |
| Undistilled short ODE | Same pretrained weights, Euler ODE with 8 steps | Measure the consequence of shortening the existing solver |
| Initial student | Same initial weights, 8 clean-prediction/renoise steps | Establish the starting point of the proposed distillation sampler |

These step counts and guidance values are starting experiment inputs, not
quality claims or test expectations. Changing teacher guidance is a separate
comparison. The student's starting inference uses its conditional prediction
directly; any teacher CFG used later for score guidance is specified separately.

For `K` student steps, use the explicit increasing grid
`0 = t_0 < ... < t_K = 1`. The initial candidate applies the established
resolution shift once to `u_k = k / K`. At each step:

```text
y_k = x_k + (1 - t_k) v_theta(x_k, t_k, c)
x_(k+1) = t_(k+1) y_k + (1 - t_(k+1)) epsilon_(k+1)
```

Start with `x_0 = epsilon_0`. Each intermediate `epsilon_(k+1)` is independent
Gaussian noise; the last output is `y_(K-1)` and needs no extra noise draw. This
translates the clean-prediction/renoise sampler in the
[DMDR reference](https://github.com/vvvvvjdy/dmdr/blob/2beac25182637a7505f5bc9ff1d4481f09945b26/train_cc/sit/samplers.py).
Euler instead uses `x_(k+1) = x_k + (t_(k+1) - t_k) v_theta(...)`.
The transition rule, grid, and all noise streams are part of sampler identity.

For the proposed DMD backward simulation, select a grid index, generate its
prefix without gradients, detach the state at that index, and differentiate
only that step's clean prediction. Training inputs then follow the deployed
student sampler. The DMD renoise time is a separate draw from the selected
generator grid index; NFT forward-noising time is another distinct input.
Their distributions and update schedules remain to be specified.

RT-DM must associate a complete rollout score with the corresponding trajectory
and selected generator update. The proposed [detached-state procedure](#detached-rollout-state)
below makes this association explicit; the reward transform and full objective
derivation remain open. NFT learns from terminal latents and needs no gradient
through their rollout.

## Initial data and sampler interfaces

Use normalized clean latents as the training sample representation; decode them
for image scorers. Retain the original terminal latent for NFT rather than
decoding and re-encoding it. Benchmarks consume generated images and sample
metadata. Rollout records and benchmark records share sample identity, not a
training loop dependency.

| Record | Required content for the first increment |
|---|---|
| Prompt | Stable ID, text, scene/family ID, split, language, and domain; evaluation adds explicit seeds and aspect ratio |
| Sampling specification | Rule, actual time grid, step count, guidance, spatial shape, precision, conditioning/VAE identities, and weight identity |
| Sample | Prompt ID, draw index, initial and transition noise identities, collection-policy version, terminal latent and decoded-image identities |
| Raw score | Sample identity, scorer/model/rubric/preprocessing identity, raw values, request identity, and explicit success/failure |
| Evaluation result | Sample manifest, per-sample scores/failures, paired comparisons and panels, timings, and scope of the conclusion |

For the initial deterministic replay contract, use per-sample CPU noise streams
and retain their seed and generation specification. Changing rank, batch order,
or bucket order must not change the assigned random inputs. This promises the
same inputs, not bit-exact floating-point outputs across hardware. Generate
noise and maintain integration/clean-prediction arithmetic in FP32; cast at
model and VAE boundaries according to the recorded precision policy.

The sampling harness takes explicit latent shapes from its config; it must not
need a training dataset merely to discover image dimensions. Both teacher and
student see the same shape, conditioning, and initial noise in a paired case.
Map each development-panel aspect to one recorded shape. Preserve sample IDs
through bucketing so result matching does not depend on return order.

The initial scorer interface returns separate HPSv3 and VQAScore raw records.
Use VQAScore v3.1 (announced on 2026-06-04) with Qwen3-VL-8B-Instruct, explicitly
selecting `qwen3-vl-8b`. Pin HPSv3 weights/config and the VQAScore weight revision,
template, probability calculation, and immutable source artifact. A custom
VLM-rubric scorer is excluded from this initial setup. Establish persistent
scorer workers with the round/update boundary described below. Scoring requests
can execute concurrently without changing which policy produced the round.

Both scorers are evaluated on the same image/prompt identity. Reward transforms
and any mixture weights are explicit objective inputs, not score-client behavior.
Do not assume identical raw scales. Treat a sample as complete only when all
scores required by that objective are present. A failed request is retried on
the same image under an explicit bounded policy; after exhaustion, fail the
training collection round before updates rather than silently deleting cases.
Evaluation retains per-scorer failures and coverage without a success-only win
claim. Language/length behavior is qualified before interpreting bilingual or
long-caption score comparisons.

## Resident roles and rank allocation

The first implementation separates learner/rollout workers from frozen scorer
workers. Keep scorer weights resident for the workload lifetime; routine rounds
must not repeatedly load/offload each scorer on every learner device. This
placement and scheduling contract belongs to algorithm bring-up because it
determines memory ownership, communication, and update ordering. Systematic
hardware tuning remains a later work layer.

For an initial 16-device allocation, define three configurable, disjoint groups:

| Group | Responsibility | Candidate device count |
|---|---|---|
| Learner and rollout | Student and fake-score updates, teacher/old-policy views, text conditioning, generation, and VAE decode | 12 |
| HPSv3 scoring | Persistent frozen HPSv3 replicas and batched inference | 2 |
| VQAScore scoring | Persistent frozen Qwen3-VL-8B-Instruct replicas and batched inference | 2 |

The 12/2/2 split is a profiling candidate, not a qualified or optimal allocation.
Counts do not imply a physical rank ordering: map groups using the actual device
topology. Start from complete scorer replicas where measured weight/activation
memory permits useful batches; two scorer devices mean two independent replicas,
not an assumed tensor-parallel implementation. The learner group's memory also
needs qualification with its actual trainable/frozen roles and accumulation.

Training collectives include only the participating learner ranks; scorer
workers do not join learner DDP updates. Make global worker identity, role-local
rank, and learner world size explicit. Prompt groups and loss normalization use
the actual learner/sample population rather than assuming all 16 workers train.
Checkpoint the logical model roles and data/update state independently of the
physical role map; changes to learner count require an explicit repartition and
recovery procedure rather than an implicit same-layout resume.

Use this initial execution sequence:

1. Freeze the collection-policy version for the round. Learners generate and
   decode batches, retaining terminal latents and any replay inputs required by
   the objective. Transfer image payloads and sample/prompt identities to the
   scorer groups; model weights stay with their owners.
2. Dispatch each image to both scorer groups. Each group batches its own queue;
   HPSv3 and VQAScore can process concurrently while learners generate further
   batches for the same round. Bound outstanding work so host/device buffers
   cannot grow without limit. Resolve results by sample/scorer identity, not
   arrival order.
3. Join the required scores for complete prompt groups, then form reward
   transforms using the declared population. Start generator updates only when
   the round's required scores are complete. Keep the fake-score update order
   explicit rather than opportunistically changing it when one scorer is slow.
4. Apply the configured updates and refresh the collection policy at its declared
   boundary. The first design does not collect future rounds from stale policy
   versions while updating the current round.

Before selecting the initial split, measure steady-state generation/decode,
each scorer's service rate and memory at the actual image/text shapes, transfer
time, queue wait, learner update time, and end-to-end round time. Keep initial
load/warmup cost separate. Compare allocations such as 12/2/2 and 8/4/4 only
after locating the bottleneck; asymmetric scorer counts are valid. Match global
sample/group budgets across allocation comparisons. Recalibrate after a scorer
or resolution change, and validate score/sample identity and recovery under the
chosen layout.

## Detached rollout state

The inspected references establish different parts of the execution pattern:

- [DMD2 backward simulation](https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/sd_unified_model.py#L127)
  generates a prefix without gradients, prepares a noisy input at a selected
  step, and differentiates one generator prediction. It has no external reward
  wait. Its preparation re-noises the preceding clean prediction with fresh
  noise before the differentiated call.
- [DiffusionNFT collection](https://github.com/NVlabs/DiffusionNFT/blob/cbb14f84b8312b620390dfcbe2fab69c1383b104/scripts/train_nft_sd3.py#L640)
  samples under the frozen old adapter without gradients, submits reward futures,
  and retains terminal latents and conditioning. After joining rewards, training
  draws fresh forward noise and computes old/current predictions; it does not
  backpropagate through the collected trajectory.
- The [DMDR demo](https://github.com/vvvvvjdy/dmdr/blob/2beac25182637a7505f5bc9ff1d4481f09945b26/train_cc/sit/train_sitxl_refl_deepspeed.py#L413)
  caches clean predictions from a detached rollout, selects one, and re-noises
  it for a new generator call. Its reward branch differentiates through the VAE
  and DINO classifier (ReFL); it does not implement the v5 RT-DM reward wait.
  [DMDR v5, section 2.4](https://arxiv.org/html/2511.13649v5#S2.SS4) describes
  detached reward weighting using the completed few-step rollout, but does not
  specify a replay-buffer layout.

Proposed Inko RT-DM procedure: combine a detached full rollout with a single
recomputed generator call. This is an explicit adaptation, not an assertion
that the local DMDR demo implements this procedure.

1. Select the generator training index `k` independently of the resulting reward.
   Run the configured full trajectory under `no_grad` with fixed student weights.
   Save the actual noisy input `x_k` immediately before its generator call,
   together with `k`, physical time `t_k`, and conditioning identity.
2. Retain the terminal normalized latent `x_terminal`; decode and score that
   terminal output. Both scorer requests reference the same sample and round.
3. Once required scores are joined, recompute
   `y_k = x_k + (1 - t_k) v_theta(x_k, t_k, c)` with gradients, treating saved
   `x_k` as constant. Keep student weights and forward settings identical to
   collection; disable stochastic model behavior or retain its replay RNG state.
4. Draw the separate DMD renoise time/noise, compute detached teacher/fake-score
   targets, and weight the per-sample DMD surrogate by the detached terminal
   reward transform before reducing across samples.
5. For this initial RT-DM procedure, accumulate all microbatch gradients for the
   round before one generator optimizer step. Then retire those records and
   collect a new round. Reusing them for further generator steps would no longer
   recompute the prediction that produced the scored trajectory and needs its
   own explicit algorithm contract. Fake-score updates follow their separately
   specified schedule.

The learner therefore retains two detached latent tensors per RT-DM sample:
the selected noisy input and terminal latent, plus conditioning (or an immutable
cache reference), step/time, sample/round identity, policy version, and score
records. Weights are retained once per policy role, not copied per sample. Keep
the scored image or reproducible decode inputs available for retries. Generation
noise identities remain provenance; saving the actual selected input avoids
replaying the prefix or storing every intermediate latent. DMD training noise
can be drawn after scoring and is not required in the collection record.

NFT needs the terminal latent, conditioning, rewards/group identity, and frozen
old-policy role; its loss does not need the selected RT-DM input. Its configured
training epochs and old-policy refresh are a separate contract. Exact RT-DM
reward transforms, selected-step distribution, DMD reductions, and fake-score
ordering still belong to the objective derivation before training implementation.

## Sampling/scoring implementation contract

The [config](../configs/posttrain.toml) declares initial weight identity, text
exit layer, precision, explicit image shapes, sampler arms, role device lists,
queue capacity, retry/deadline limits, suite paths, uncertainty settings, and
scorer source/weight revisions. Unknown fields and incompatible settings fail
validation. Numerical settings are experiment inputs, not test expectations.

The first harness uses a coordinator and persistent single-host processes with
disjoint device ownership. The coordinator has no accelerator model. Generation
workers share a bounded task queue; each scorer group has its own queue. A batch
occupies an in-flight slot until both scorers respond. Images are immutable PNGs
on shared storage, referenced by path and content hash. Filesystem/queue costs
need measurement before choosing a faster transport.

Sampling performs FP32 state/time arithmetic with BF16 model calls. Shapes are
configured as image height/width and converted by the VAE's /8 factor. Each
shape's shift is `max(1, sqrt((latent_height * latent_width / 4) / 256))`, applied
once to the base grid. Initial noise uses the prompt seed; transition noise
uses separate SHA256-derived seeds. The student harness retains one uniformly
assigned step for interface qualification; the cold-start training step
distribution will be derived with its objective.

The generation record includes prompt/sample/policy identity, grid, shape,
noise specification, terminal and selected-step tensors, and image/latent
hashes. Frozen text conditioning can be rebuilt from the recorded prompt,
encoder/tokenizer artifact identity, and text contract. The score request binds
sample ID, image hash, and exact prompt. Its request ID also includes scorer
source, weight, preprocessing/config, and dependency identity. Results carry
separate raw values, success/failure, attempt count, and error details.

Initially each scoring request has two attempts on identical image bytes.
Exhaustion produces explicit failure records. Startup failures, dead workers,
timeouts, duplicate results, or identity mismatches abort execution. Evaluation
can report score failures; the training join rejects incomplete rounds.

Each run writes an immutable input/code/artifact manifest and atomic sample and
score records under a single-writer lock. Restart verifies those identities and
content hashes, reuses completed records, and resubmits work with no result.
Exhausted score failures remain recorded; a restart does not silently grant a
new retry budget. Temporary/unrecorded generation is recollected. Training
recovery will initially restart incomplete collection from the preceding
committed round boundary; optimizer-state recovery belongs to 6.1.3.

This inference harness performs no learner DDP updates. Its memory measurements
cover the resident generator, text encoder, VAE, and scorers. Trainable/frozen
DMD and NFT role memory must be measured when those increments are implemented.

## Model roles and gradients

| Role | Contract to preserve |
|---|---|
| Student | Trainable generator; identify its parameterization and the sampling steps through which gradients flow. |
| Base teacher | Preserved pretrained weights used as the distribution anchor. Record checkpoint and live/EMA selection. |
| DynaDG teacher view | An adapted score-estimator view during cold start. Specify adapter ownership, training objective, sharing, and annealing back to the base teacher. |
| Fake-score estimator | Learns the current student distribution from detached generated samples with its own optimization schedule. Its parameters receive no generator-loss gradients. |
| NFT old policy | Supplies collection and comparison predictions; frozen during its update interval, with an explicitly scheduled refresh. It is distinct from the base teacher. |
| Evaluation EMA | Optional student weight average for evaluation; distinguish it from the NFT old policy even if an implementation shares storage. |
| Reward scorer | Returns scores without a gradient path in the starting RT-DM/NFT recipe. A differentiable-reward experiment must specify a different contract. |

Storage sharing between roles must preserve their separate update and gradient
semantics. Teacher/fake-score predictions used to construct the DMD generator
update are stop-gradient targets; fake-score learning uses a separate loss.

## Training-objective contract coverage

The detailed implementation specification resolves these requirements. This
table records the required mathematical coverage; publish the corresponding
equations and evidence here as each training increment is implemented and qualified.

| Component | Required derivation and choices |
|---|---|
| Distribution matching | Translate score difference into Inko velocity/clean-prediction coordinates; specify sign, time factors, surrogate gradient, normalization, and endpoint treatment. |
| Fake-score learning | Specify generated sample source, detached target, renoise distribution, loss weighting, and update ratio. |
| Backward simulation | Define inference update, selected training step, detached prefix, differentiated step, noise reuse, and correspondence to deployment sampling. |
| RT-DM | Define how a completed rollout's detached reward weight attaches to a per-sample DMD update; state the transform, temperature, normalization, and reduction. |
| DynaDG | Define adapter parameterization, optimization, real/fake views, and scale schedule. |
| DynaRS | Define renoise sampling and annealing in Inko time, including bounds; high noise is near `t = 0`. |
| DiffusionNFT | Specify positive/negative implicit predictions, old-policy detachment, grouped score-to-optimality mapping, branch normalization, and refresh schedule. |
| Joint objective | Starting form: `L_DMD + lambda_NFT * L_NFT`. Fix each term's reduction before interpreting the coefficient; record gradient contributions as diagnostics. |

The exact objectives, reductions across samples/ranks/accumulation, numerical
guards, and stage-transition behavior are open. Distinguish an intentional
algorithm change from a numerical implementation detail. Record any departure
from the selected reference and the reason for it.

## Reward and rollout records

Separate raw score acquisition from objective-specific reward transforms.
A sample record needs prompt/group identity, latent or image identity, seed,
collection-policy version, sampler/grid/guidance, and resolution. A score record
needs scorer version, preprocessing/rubric identity, raw outputs, and success or
failure status. Cache identity must cover both the sample and scorer inputs.

Missing or failed scores are not zero rewards. Specify retry, incomplete-group,
and exclusion behavior before training; preserve per-scorer outputs so a changed
mixture does not require silently reinterpreting old records. Provider addresses
and credentials remain machine-local.

## Update clocks and checkpoint state

Name each clock: generator optimizer updates, fake-score optimizer updates,
rollout rounds, and samples consumed. Every schedule declares its clock.
Specify update ordering, accumulation, skipped-update behavior, stage transitions,
and when the collection policy and evaluation EMA change.

Checkpoint coverage must include trainable models/adapters, optimizers,
schedulers, model-role versions, all clocks, stage, RNG, prompt-sampler state,
reward-normalization state, and any retained rollout/reward work. Record the
complete config, code revision, and input artifact hashes. Define whether an
interrupted collection round is restored or explicitly discarded and recollected.
Recovery and a deliberate experimental branch are separate operations.

## Verification and references

Mechanism tests should cover coordinate conversion, gradient isolation,
reductions, clocks, group handling, and save/resume from independent contracts.
Generation quality is established by real samples and benchmarks under the
[evaluation protocol](posttrain_evaluation.md).

- [DMDR v5](https://arxiv.org/abs/2511.13649v5): two-stage starting framework,
  including RT-DM and dynamic cold start.
- [DiffusionNFT v2](https://arxiv.org/abs/2509.16117v2): forward-process reward
  optimization and the old-policy formulation.
- [DMD2 v2](https://arxiv.org/abs/2405.14867v2): distribution-matching training
  components, including backward simulation and two-timescale updates.

Pin reference-code revisions when deriving implementation details. These sources
provide starting methods, not evidence that the combined Inko recipe works.
