# Posttraining mathematical and execution contract

Status: independent sampling/scoring services and cold DMD/RT-DM training are
implemented. The DMD contract below covers full/LoRA students, fake-score roles,
learner-only DDP, optimizer clocks, complete-round recovery and export. Independent
RT-DM/DynaDG/DynaRS and NFT/joint controls are implemented. Full/LoRA NFT,
stage transitions and whole-round recovery have bounded target-run evidence;
algorithm quality and release-recipe choices remain open.
Numerical choices belong in the runnable config; this contract defines their
meaning. See the [recipe](posttrain_recipe.md) and
[evaluation protocol](posttrain_evaluation.md).

## Scope and ownership

Core training logic, rollout sampling, model-state management, and reward/score
clients belong in `src/posttrain/`. Benchmark execution and reporting belong in
`src/evaluation/benchmarks/`. Rollout and evaluation prompt suites belong in
`configs/prompts/`. Independent sampling/scoring and the cold/joint trainer are implemented.

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
paths is required qualification. The independent services and generic client satisfy collection/evaluation
substitution. An unfamiliar service/channel also has HTTP mechanism coverage
through cold reward transformation and update admission. Grouped NFT
reward normalization and update admission also have generic HTTP mechanism
coverage; this does not qualify a real VLM backend.

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

Status: implemented and exercised with real generations and scores. The
undistilled controls have weak generation quality; no training method is selected. Initial development uses 640p and the preserved 480k EMA.
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

For the implemented DMD backward simulation, select a grid index, generate its
prefix without gradients, detach the state at that index, and differentiate
only that step's clean prediction. Training inputs then follow the deployed
student sampler. The DMD renoise time is a separate draw from the selected
generator grid index; NFT forward-noising time is another distinct input.
Plain DMD uses an explicit selected-index PMF, bounded uniform score times and
a configured fake-update ratio. NFT uses a frozen old policy for terminal
collection and fresh independent forward time/noise for each generator update.

Implemented RT-DM associates a complete rollout score with the corresponding
trajectory and selected generator update through the
[detached-state procedure](#detached-rollout-state). Global reward transforms
and selected-call replay have mechanism and real cold-path qualification.
NFT learns from detached old-policy terminal latents and needs no gradient
through their rollout. Joint DMD recollects current-student prefixes for each
inner update; old-policy terminals are not DMD selected-state replay data.

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

Implemented Inko RT-DM procedure: combine a detached full rollout with a single
recomputed generator call. This is an explicit adaptation, not an assertion
that the local DMDR demo implements this procedure.

1. Select the generator training index `k` independently of the resulting reward.
   Run the configured full trajectory under `no_grad` with fixed student weights.
   Save the actual noisy input `x_k` immediately before its generator call,
   together with `k`, physical time `t_k`, its actual clean prediction, and
   conditioning identity.
2. Retain the terminal normalized latent `x_terminal`; decode and score that
   terminal output. Both scorer requests reference the same sample and round.
3. Once required scores are joined, recompute
   `y_k = x_k + (1 - t_k) v_theta(x_k, t_k, c)` with gradients, treating saved
   `x_k` as constant. Keep student weights and forward settings identical to
   collection; disable stochastic model behavior or retain its replay RNG state.
   The recomputed prediction must equal the retained selected-call prediction
   exactly under the declared execution policy. A mismatch aborts the round.
4. Draw the separate DMD renoise time/noise, compute detached teacher/fake-score
   targets, and weight the per-sample DMD surrogate by the detached terminal
   reward transform before reducing across samples.
5. For this initial RT-DM procedure, accumulate all microbatch gradients for the
   round before one generator optimizer step. Then retire those records and
   collect a new round. Reusing them for further generator steps would no longer
   recompute the prediction that produced the scored trajectory and needs its
   own explicit algorithm contract. Fake-score updates follow their separately
   specified schedule.

The learner therefore retains three detached latent tensors per RT-DM sample:
the selected noisy input, selected clean prediction and terminal latent, plus conditioning (or an immutable
cache reference), step/time, sample/round identity, policy version, and score
records. Weights are retained once per policy role, not copied per sample. Keep
the scored image or reproducible decode inputs available for retries. Generation
noise identities remain provenance; saving the actual selected input avoids
replaying the prefix or storing every intermediate latent. DMD training noise
can be drawn after scoring and is not required in the collection record.

NFT needs the terminal latent, conditioning, rewards/group identity, and frozen
old-policy role; its loss does not need the selected RT-DM input. Its configured
training epochs and old-policy refresh are a separate contract. Cold reward weighting, selected-step
distribution, global reductions and fake-score ordering are implemented below.

## Sampling/scoring implementation contract

The [caller config](../configs/posttrain.toml) uses strict frozen records for
sampling, explicit per-arm checkpoint/architecture digests and live/EMA choice,
execution limits, suite paths, declared score channels and acquisition limits.
[Service configs](../configs/scorers/) separately own source/model pins,
preprocessing, numeric runtime, output meanings and model batching. Asset paths
and endpoint/device placement live in separate local JSON. Unknown fields fail
validation; no config inheritance or environment hyperparameters are supported.

The CPU coordinator has no scorer model or backend imports. Independently
runnable services expose `/v1/health`, `/v1/capabilities` and `/v1/score` through
HTTP JSON protocol version 1. The same client handles owned loopback replicas
and externally managed endpoints. Scorers never join learner collectives.
Generation and acquisition overlap within a bounded number of unfinished
batches, held until all configured services respond. `learner` is reserved as
the generation role ID; scorer IDs and output channels otherwise come from config.

The handshake checks semantic fingerprint, input schema, output direction/range,
transport, batch size and stable replay. Requests accept verified shared-file
PNGs or encoded PNG bytes, exact prompts, structured inputs and an explicit seed
policy for stochastic services. A semantic cache key excludes sample IDs and
transport locations; wire IDs hash `(sample_id, cache_key)`. Cached results are
re-enveloped for the current sample. Diagnostic JSON never becomes a reward.

Sampling performs FP32 state/time arithmetic with BF16 model calls. Shapes are
configured as image height/width and converted by the VAE's /8 factor. Each
shape's shift is `max(1, sqrt((latent_height * latent_width / 4) / 256))`, applied
once when the arm selects shape shift; unshifted grids are explicit. CFG uses
true empty-prompt conditioning through the same encoder contract. Initial noise uses the prompt seed; transition noise
uses separate SHA256-derived seeds. The student harness retains one uniformly
assigned step for interface qualification; plain-DMD training selects from its
explicit configured PMF.

The generation record includes prompt/sample/policy identity, grid, shape,
noise specification, terminal and selected-step tensors, and image/latent
hashes. Frozen text conditioning can be rebuilt from the recorded prompt,
encoder/tokenizer artifact identity, and text contract. Score semantics include pinned source/model, preprocessing, adapter, numeric
runtime and structured-input identities. Results contain validated named raw
channels, typed failures and diagnostic metadata. Request deadlines cover DNS,
TLS and response bodies through a bounded, model-free transport process.
Stalled inference retires the owned service instead of leaving a permanently
blocked model worker advertised as healthy.

Initially each scoring request has two attempts on identical image bytes.
Exhaustion produces explicit failure records. Startup failures, dead workers,
timeouts, duplicate results, or identity mismatches abort execution. Evaluation
can report score failures; the training join rejects incomplete rounds.

Each run writes an immutable input/code/artifact manifest and atomic sample and
score records under a single-writer lock. Restart verifies those identities and
content hashes, reuses completed records, and resubmits unresolved work within its persisted retry allowance.
Each attempt is charged durably before transport; successful rows survive a
partial batch failure, and retries preserve request/seed identity.
Exhausted score failures remain recorded; a restart does not silently grant a
new retry budget. Temporary/unrecorded generation is recollected. Plain-DMD training restarts incomplete collection from the preceding committed
round boundary and restores optimizer state, per-rank RNG and prompt cursor.

This inference harness performs no learner DDP updates. Its memory measurements
cover the resident generator, text encoder, VAE, and scorers. Training memory is
reported separately in the recipe. Full/LoRA NFT and joint qualification retained
twelve learners, two HPSv3 replicas and two VQAScore replicas, with a maximum
learner allocation of 30.75 GiB including interrupted recovery. This placement
is qualified only for CFG 1, square 640p, Q/V fake adapters, AdamW and activation
checkpointing disabled; these are allocator measurements, not whole-device HBM.

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

Plain-DMD and reward-weighted RT-DM objectives, reductions and guards are
implemented and specified below, alongside NFT/joint objectives and stage
transitions. Distinguish an intentional algorithm change from a
numerical implementation detail. Record any departure from the selected
reference and the reason for it.

## Implemented plain DMD

For Inko time `t` (noise at zero, data at one), a velocity prediction gives
`clean(x,t) = x + (1-t)v(x,t)`. Let `y` be the selected student clean prediction,
`z = t·stopgrad(y) + (1-t)ε`, and `r`, `f` be the detached real/fake clean
predictions at `(z,t)`. The configured generator direction is one of:

- `clean_difference`: `g = f-r`.
- `teacher_residual`: `g = (f-r)/(mean_elements(abs(y-r))+epsilon)`, per sample.
- `score_kl`: `g = t²/(1-t)² · (f-r)`, the explicit flow-coordinate score factor.

Backpropagate `0.5·mean_elements((y-stopgrad(y-g))²)`. Its derivative is `g`
divided by the latent element count. Minimize the global mean across samples.
Teacher and fake forwards have no generator gradient. Score times are uniform
within explicit open-endpoint bounds; nonfinite directions fail without repair.

Collection holds the student fixed, samples a selected index from the explicit
PMF, and retains its input/time plus conditioning and terminal latent. Euler or
clean-prediction renoise prefixes are detached. Recompute only that selected
student prediction with gradients after the fake updates. Fake learning uses
either the detached selected clean prediction or terminal latent `y_target`,
fresh independent time/noise, and `mean_elements((v_fake-(y_target-ε))²)`.
Normally each DMD update performs `q = fake.updates` fake steps, then one student
step, then all configured student EMA updates. When `fake.initial_steps = N > 0`,
the cold entry instead performs exactly N fake steps before the first student
step. It does not add q more steps before that first student step. Later cold
steps and every joint DMD step use q. Initial fitting holds the student, EMA,
generator clock and prompt cursor fixed. Its own counter advances with the
actual fake optimizer counter and is recoverable through `fit-NNNNNN`
checkpoints. It does not acquire rewards.

`fake.target_refresh = "round"` retains the collected targets while fitting;
time and noise are fresh for each fake update. `"update"` explicitly generates
fresh targets from the same held student and prompt group for every fake update.
It uses an independent keyed stream and does not advance the prompt cursor.

`q` is a fitting/exposure control, not a correctness constant. Matching teacher
initialization does not establish an accurate score for the few-step student's
output distribution. The [texture-collapse investigation](posttrain_recipe.md#fake-score-fitting-and-texture-collapse-disposition)
demonstrates a causal initial fitting deficit and its correction through more
fake updates. The initial fitting count is explicit; no adaptive stopping rule is implemented;
neither a low fake loss nor a small generator surrogate certifies score accuracy
or generation quality.
The optional `fake_probe` regenerates samples from the current student and
reports fake regression MSE and its score-time-weighted form. It changes no
optimizer, EMA, cursor or training RNG state. It measures regression risk on
fresh samples, not error against an oracle student score.
All random streams identify round, sample and role; noise is generated from
keyed CPU seeds so packing does not change draws.

Full student tuning trains all parameters. Weight LoRA freezes the base and
adds `B(A(input))·alpha/rank` to resolved fused QKV/output projections, optionally
including MLP projections. A uses Kaiming initialization and B is zero. The
teacher-Q/V fake strategy instead adapts projected Q and V at full hidden width
before head splitting, normalization and RoPE; K is unchanged. Its frozen real
view sets the adapter scale to zero. The independent-full-fake strategy owns a
separate trainable teacher copy. DynaDG is a separate explicit switch requiring
the teacher-Q/V strategy; fake capacity alone does not enable it.

Resident learner replicas alone join DDP. With L learners and N global samples,
each backward uses `L/N` times its local sample sum; DDP averaging produces the
global mean without another accumulation divisor. Round divisibility ensures
matching collective schedules. FP32 master weights, losses and reductions are
separate from configured BF16 forward autocast. Each role has its own optimizer,
clip limit and successful-update LR clock. All of that role's loss, parameters,
gradients and optimizer state must be finite before any optimizer steps.

Complete-round checkpoints retain role weights/adapters, optimizer state and
clocks, EMA, cursor/permutation, per-rank RNG, source/config/runtime/artifact
identities and sample counts. Attempt journals retain discarded work. Files are
fsynced before publishing a manifest and atomically renaming the checkpoint;
incomplete directories are never recovery points. Resume rejects changed
contracts and verifies exact restoration before collecting again. Deterministic
kernel and HCCL policies are explicit in configuration. Student/fake DDP wrappers
receive a zero-loss setup backward before initialization or restore so their
first real forward uses the steady reduction layout. Setup gradients are cleared
and RNG restored without optimizer, cursor or clock updates. Numerical
repeatability remains a measured property separate from exact state restoration.

`training.ema` declares named student shadows with independent decays and an
explicit storage device (`learner` or `cpu`). Each successful student optimizer
step advances every shadow. Initial fake fitting never advances them. Stage
transition resets all shadows to the live joint-entry student. No fake-score
shadow exists. Save/resume preserves every named shadow and rejects a changed
inventory. Checkpoint schema 2 records the initial-fitting and actual fake-step
counters; earlier checkpoint schemas require their original source.

Exports merge only student weight LoRA, optionally substitute a named student EMA,
and verify an exact reload of the materialized FP32 weights. Fake Q/V adapters
never enter an inference export. The export manifest pins initialization,
architecture, tuning, resolved parameter names, selected view/decay and source
checkpoint identity. Export directories publish atomically.

Interval evaluation runs between committed training segments. All learner and
owned scorer processes exit before export and benchmark workers start. The next
segment resumes the unchanged training contract. Evaluation errors block further
training until the same checkpoint evaluation succeeds. This avoids concurrent
training/evaluation model allocations without changing the scientific workload.

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

## Cold reward and dynamic-control contract

`objective.mode` independently selects `dmd` or `rtdm`; `dyna_dg.enabled` and
`dyna_rs.enabled` are independent of that choice. All eight toggle combinations
are supported with teacher-Q/V fake score. Full independent fake is incompatible
with DynaDG and remains usable when that control is disabled. NFT is rejected.

RT-DM decodes the detached terminal normalized latent with the pinned frozen VAE.
Each learner publishes immutable PNG/latent identities and retains conditioning
and selected replay state. The CPU coordinator acquires scores through the shared
client and admits the complete global round before any optimizer update. Failed
or missing required channels abort; they never become zero rewards or dropped
samples. Raw channels and service identities remain separate from rewards.

Orient each configured channel by its declared `higher`/`lower` direction. With
round standardization, use population mean/std over all N samples and
`(oriented-mean)/(std+epsilon_reward)`; constant channels become zero. Mix with
explicit nonnegative weights summing to one. Statistics, transformed values,
combined rewards and weights are retained. Canonical sample-ID ordering makes
normalization independent of rank packing and score completion order.

For positive temperature, `mean_one_exp` uses
`w = N*softmax((reward-max(reward))/temperature)` in FP32. `paper_max_exp` uses
`exp(reward/max(reward))`; it requires raw, nonnegative combined rewards with a
positive global maximum and is incompatible with standardized channels. Its
weights are not mean-normalized. The student objective is the global mean of
detached weights times per-sample DMD loss; fake regression stays unweighted.
Weight concentration and effective sample size are observations, not controls.

DynaDG's real adapter scale is
`a_start*0.5*(1+cos(pi*min(g/H_dg,1)))`, where g counts committed cold student
updates. Disabled DynaDG uses real scale zero; fake scale stays one. Scoped views
restore after exceptions and reject overlapping views or cross-thread forwards.
The base teacher stays frozen.

DynaRS draws a sample-local host Beta noise coordinate with
`alpha=1+(alpha_start-1)*c(g)` and `beta=1+(beta_start-1)*c(g)`, using its own
cosine horizon, then maps `t=t_min+(t_max-t_min)*(1-u)`. The host draw is cast to
FP32 before latent arithmetic. Disabled DynaRS uses uniform physical time on
the same bounds. Fake and DMD streams remain separate; all q fake updates use
the same schedule point. Neither control changes the sampler grid/index PMF.

Checkpoint state includes generator/fake clocks, dynamic schedule definitions
and current point, last global reward statistics and scored/used sample counts.
An interrupted round is discarded and recollected from the last committed
boundary; attempted work and retry accounting remain recorded.

## NFT, joint objectives and complete-round transitions

NFT collects detached terminal latents with a frozen old policy. Exact prompt ID
is the group identity; all required scores join globally before advantages are
computed or any optimizer steps. For combined reward R, use population statistics:

```text
A_i = (R_i - mean_group(R)) / (std_group(R) + epsilon_reward)
# The round-scaled control substitutes std_round(R) in the denominator.
p_i = 0.5 + clip(A_i, -C, C) / (2*C)
```

Constant groups remain with p=0.5. Their gradient cancels at student=old, but
anchors the student after it moves; equal rewards do not imply an always-zero
update. Config-selected scorer channels pass through the shared client. Group
normalization and NFT admission contain no scorer backend names or model imports.

For old terminal x, independent noise e and physical time t, evaluate raw
conditional old/new velocities even when terminal collection uses CFG:

```text
z = t*x + (1-t)*e
v_plus  = (1-beta)*stopgrad(v_old) + beta*v_new
v_minus = (1+beta)*stopgrad(v_old) - beta*v_new
x_plus/minus = z + (1-t)*v_plus/minus
h_plus/minus = stopgrad(max(mean_elements(abs(x_plus/minus-x)), epsilon_nft))
l_plus/minus = mean_elements((x_plus/minus-x)^2) / h_plus/minus
l_nft = C/beta * (p*l_plus + (1-p)*l_minus)
```

C and beta are distinct positive parameters. The two detached normalizers are
separate. Inko reduces in FP32, checked against an independent CPU double oracle;
the inspected reference calculates its adaptive normalizers in double precision.
The velocity-MSE control instead uses `mean_elements((v_plus/minus-(x-e))^2)`
without adaptive denominators. Forward times are bounded continuous uniform or
draws from the deployed collection grid excluding its terminal endpoint t=1.
Every inner update/time draw uses independent keyed time and noise. Average time
draws within sample before the global sample mean.

The generator objective is
`lambda_dmd*mean(l_dmd) + lambda_nft*mean(l_nft) + lambda_ref*mean(l_ref)`.
The optional frozen stage-entry reference uses raw conditional prediction MSE,
`l_ref=mean_elements((v_new-stopgrad(v_reference))^2)`, on the same NFT inputs.
It is a prediction penalty, not an exact distribution KL. Its coefficient is zero
in the integration candidate. A reference requires an active NFT term. NFT-only
allocates no fake/DMD teacher; DMD-only allocates no old/reference policy.

Old stays fixed across all inner updates of one collection round. Joint DMD
recollects detached current-student prefixes/targets at each inner update, performs
its unweighted q fake updates and verifies selected-call replay before the student
step. Its real score is the original frozen teacher, with bounded uniform score
times. RT-DM, DynaDG and DynaRS apply only in the cold stage. Term loss and weighted
latent-prediction gradient norms are diagnostics; these norms stop at forward
outputs and do not traverse parameter/DDP hooks.

At joint entry, hard-copy old/reference from the live boundary student, retain
enabled fake weights, reset student/fake optimizers and schedules, and reset the
separate evaluation EMA. Full tuning has actual frozen model copies. LoRA has
independent adapter/module state over shared immutable base tensors. After a whole
round, refresh old with `old <- d(r)*old + (1-d(r))*student`:

- `copy`: d=0.
- `constant`: the configured decay.
- `linear_ramp`: `decay_start + (decay_end-decay_start)*min(r/horizon,1)`.

Here r counts previously committed joint rounds, starting at zero. It advances
only after every inner update and the old refresh succeed. Evaluation EMA advances
on each successful generator step; its state and clock are independent of old.

Complete configs declare cold/joint generator-step counts. NFT joint length and
checkpoint interval must contain whole inner-update rounds; explicit early stops
inside a round fail before launch. Checkpoints are named `step-<run-local-generator-step>`
and retain stage, local/global generator and fake clocks, committed round and old
versions, scored/used counts, next action, every active role/optimizer, old/reference
and evaluation EMA, last global reward, source/config/runtime identities and rank
RNG/cursor state. Cold boundary checkpoints identify the pending transition.
No intermediate NFT inner-update checkpoint is published. A failed inner update
is discarded; exact resume restores the last whole boundary and recollects.

Training RNG v3 keys include run seed, stage, global generator step, collection
round, sample ID, role and update/time-draw index. Old collection additionally
identifies its frozen policy version. An explicit SHA-pinned cold initialization
imports matching live student and enabled fake weights while resetting declared
optimization/cursor/RNG state. It preserves lineage offsets for global generator
and fake clocks. This is separate from exact resume, which requires unchanged
source/config/runtime/artifact identities.

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
