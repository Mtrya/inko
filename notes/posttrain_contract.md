# Posttraining mathematical and execution contract

Status: draft. The flow convention below is established by pretraining; the
posttraining objectives and execution contracts still need implementation-level
derivations. An open item is not an implemented or qualified capability.
Numerical choices belong in the runnable config; this contract defines their
meaning. See the [recipe](posttrain_recipe.md) and
[evaluation protocol](posttrain_evaluation.md).

## Scope and ownership

Core training logic, rollout sampling, model-state management, and reward/score
clients belong in `src/posttrain/`. Benchmark execution and reporting belong in
`src/evaluation/benchmarks/`. Rollout and evaluation prompt suites belong in
`configs/prompts/`. These posttraining components are planned, not yet present.

The starting recipe has two training stages: RT-DM with DynaDG and DynaRS,
then DiffusionNFT with persistent distribution-matching regularization.
Alternative objectives, components, optimization methods, and custom techniques
remain experimental choices. Each implemented variant needs an explicit equation
and state-flow description; method names alone do not define behavior.

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
least in FP32. The few-step student's update rule remains to be specified.

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

## Objectives to specify before implementation

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
