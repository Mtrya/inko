# Posttraining mathematical contract

The trainer optimizes the 600k base model with [DiffusionNFT](https://arxiv.org/abs/2509.16117v2)
alone: one stage, one objective, no distillation. `configs/posttrain.toml` is
the reference configuration. Numerical choices belong in the runnable config;
this note defines their meaning. See the [recipe](posttrain_recipe.md) and
[evaluation protocol](posttrain_evaluation.md).

## Flow convention

For a clean normalized VAE latent `x`, standard Gaussian `epsilon`, text
conditioning `c` and physical time `t`, Inko uses noise at `t = 0` and data at
`t = 1`:

```text
x_t   = t x + (1 - t) epsilon
v     = x - epsilon
clean(x_t, t, v) = x_t + (1 - t) v
```

The interpolation and velocity target live in [flow.paths](../src/flow/paths.py).
A reference using noise time `tau = 1 - t` and velocity `epsilon - x` changes
both the time direction and the velocity sign; translate its noising and
clean-prediction equations together. Gradients reach the model only through the
differentiated `clean_adaptive` branch below.

## Round structure

One collection round produces `prompts_per_round * draws_per_prompt` samples.
The frozen old policy collects every sample of a round at the same committed
step, at the configured step count and guidance. Each round is followed by
`nft.updates_per_round` student updates, then one old-policy refresh, then the
round commits. A stop boundary that splits a round is rejected before launch.

Noise is drawn from per-sample CPU generators keyed by run seed, generator
step, round, sample ID and stream, so rank packing and microbatch order do not
change any sample's inputs. Collection uses the old-policy version in its key;
training inputs additionally use the inner update and time-draw index.

## Reward to probability

Scorers return raw channels; `combine_round` orients each declared channel by
its direction, optionally standardizes it over the round, and mixes channels
with nonnegative weights that sum to one. Missing, failed or nonfinite required
scores abort before any optimizer update; they never become zero rewards.

Exact prompt ID defines a group. Over the round's samples `i` with combined
reward `R_i`, group mean `mu_g`, group population standard deviation `s_g` (or
the round population standard deviation under `advantage_scale = "round"`) and
`epsilon_reward`:

```text
A_i = (R_i - mu_g) / (s_g + epsilon_reward)
p_i = 0.5 + clip(A_i, -C, C) / (2 C)   in [0, 1]
```

`C = nft.advantage_clip`. A constant group keeps `p = 0.5`, which cancels at
`new = old` but anchors the student after it moves. The whole set of `p` values
for a round is admitted together; a partial round cannot start an update.

## NFT objective

Let `x` be a collected terminal latent, `epsilon` independent forward noise,
`t` a sampled physical time, `v_old` the frozen old-policy velocity and `v_new`
the differentiated student velocity at `(t x + (1 - t) epsilon, t, c)`:

```text
v_plus  = (1 - beta) stopgrad(v_old) + beta v_new
v_minus = (1 + beta) stopgrad(v_old) - beta v_new
x_plus/minus = clean(z, t, v_plus/minus)
h_plus/minus = stopgrad(max(mean_elements(abs(x_plus/minus - x)), epsilon_nft))
l_plus/minus = mean_elements((x_plus/minus - x)^2) / h_plus/minus
l_nft = C / beta * (p l_plus + (1 - p) l_minus)
```

The two detached normalizers are separate; `mean_elements` averages all latent
elements per sample. `C` and `beta` are distinct positive parameters, and the
`1/beta` and `C` scales match the fixed optimality weight. Inko evaluates this
in FP32 against an independent CPU double oracle. The `velocity_mse`
parameterization replaces both branch losses with
`mean_elements((v_plus/minus - (x - epsilon))^2)` and uses no adaptive
denominator.

Forward times are bounded uniform draws on `(nft.time_min, nft.time_max)`, or
draws from the deployed collection grid excluding its terminal endpoint
`t = 1`. Each time draw has independent keyed time and noise; branch losses
average over the draws within a sample, and the sample mean is reduced
globally across learners.

## Reference anchor

The frozen reference is the entry student: the base weights under LoRA, a full
copy under full tuning. On the same `(z, t, c)` inputs as the NFT branches:

```text
l_ref = mean_elements((v_new - stopgrad(v_reference))^2)
L = mean(l_nft) + reference_coefficient * mean(l_ref)
```

`l_ref` is a velocity prediction penalty, not a distribution divergence. Its
coefficient is zero only for the anchor-off comparison, in which case no
reference role is allocated. The reference never receives gradients and never
changes during a run.

## Old-policy refresh

After every committed round, with `r` the number of previously committed
rounds, `old <- d(r) old + (1 - d(r)) student`:

- `copy`: `d = 0` (the old policy is the student at every round boundary).
- `constant`: the configured decay.
- `linear_ramp`: `decay_start + (decay_end - decay_start) * min(r / horizon, 1)`.

The reference recipe ramps `d` from 0 to 0.5 over the run, matching the
reference implementation's ablation that a frozen old policy stops learning.
The old policy moves only at a committed boundary; a resumed run restores the
exact decay it recorded.

## Clocks and recovery

`step` counts successful student optimizer updates, `committed_rounds` counts
complete rounds, and `old_policy_version` counts committed refreshes; one
refresh happens per committed round, so the version equals the committed-round
count. Named student EMA shadows advance on each successful student step and
are independent of the old policy. The LR schedule runs on the student's
successful-step clock.

Complete-round checkpoints retain every role (student, old, optional reference,
EMA shadows), the student optimizer and its clock, `last_reward`, the prompt
cursor and per-rank RNG. Files are fsynced before a manifest-bearing directory
is atomically renamed; incomplete directories are never recovery points.
Resume requires an unchanged training contract and verifies exact restoration
before collecting again; work inside an uncommitted round is discarded and
recollected.

## Verification and references

`tests/` covers operator values and gradients against independent oracles,
group statistics, clock and scheduling mechanics, role isolation, probability
admission, and exact save/resume. Generation quality is established only by
real samples and benchmarks under the [evaluation protocol](posttrain_evaluation.md).

[DiffusionNFT](https://arxiv.org/abs/2509.16117v2) defines the forward-process
reward optimization, the implicit positive/negative branches and the
old-policy formulation implemented here.
