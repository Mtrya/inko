# Posttraining mathematical contract

The trainer optimizes the 600k base model in one stage: one objective, no
distillation. The objective is selected by the top-level `objective` key —
`"nft"` for [DiffusionNFT](https://arxiv.org/abs/2509.16117v2)
(`configs/posttrain.toml`, the reference configuration) or `"art"` for
[DiffusionART](https://arxiv.org/abs/2610.12572) (`configs/posttrain_art.toml`).
Each objective declares exactly its own section, `[nft]` or `[art]`. Numerical
choices belong in the runnable config; this note defines their meaning. See the
[recipe](posttrain_recipe.md) and
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
clean-prediction equations together, as the ART
[coordinate mapping](#reference-coordinate-mapping) does. Gradients reach the
model only through the differentiated `clean_adaptive` branch below.

## Round structure

One collection round produces `prompts_per_round * draws_per_prompt` samples.
The rollout policy collects every sample of a round at the same committed step,
at the configured step count and guidance: the old policy under NFT, the student
itself under ART. Each round is followed by the objective's student updates —
`nft.updates_per_round` under NFT, two disjoint windows under ART — then, under
NFT only, one old-policy refresh; then the round commits. A stop boundary that
splits a round is rejected before launch.

Collection noise is drawn from per-sample CPU generators keyed by run seed,
round and sample ID, using one shared collection stream. Optimizer steps and
policy versions do not affect it, so comparison arms share initial noise even
when update counts differ or ART skips a collection. Rank packing and
microbatch order do not change any sample's inputs. Training inputs use
separate streams keyed additionally by optimizer step and inner draw index.

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

## ART objective

[DiffusionART](https://arxiv.org/abs/2610.12572) replaces the NFT branches with
a frozen-target regression over the trajectory the rollout policy just visited:
the cached rollout prediction at each visited state is the anchor. The ART arm
therefore allocates no old policy and refreshes none.

For one prompt group of `K = draws_per_prompt` draws with terminal latents `x_k`
and combined rewards `R_k`, and `r_k = R_k - mean(R)`:

```text
cov       = sum_k r_k x_k / (K - 1)        unbiased over the complete group
MAD       = mean_k abs(r_k)
active    = MAD > 0
direction = cov / MAD / rms(cov / MAD)     unit RMS; zero when not active
magnitude = 1 when active, else 0
```

The statistics are complete-group: learners reduce reward sums and counts,
then global reward extrema, then centered reward-endpoint products and absolute
deviations. Exactly constant rewards have zero centered reward even when FP32
summation rounds their mean away from the reward. A group is one
collection slot of the round — `draw_index // draws_per_prompt` — never prompt
text, so a repeated prompt still yields independent groups. No rank ever
estimates a group from its own microbatches, and a group without reward spread
has `magnitude = 0`, so its items train on nothing.

For a cached rollout velocity `v_old` at a visited state `z`, physical time `t`,
collection step `dt = grid[k+1] - grid[k] > 0`, trainable student velocity
`v_new` and `d = v_new - v_old`:

```text
G           = magnitude * direction
target      = stopgrad(v_old + G)
residual    = target - v_new
movement    = stopgrad(v_new - v_old)
alignment   = mean_elements(movement * stopgrad(residual))
l_art       = (1 - t) mean_elements(residual^2)
std         = 0.8                                  at t = 0
              0.8 sqrt((1 - t) dt / t)              otherwise
coefficient = dt                                   at t = 0
              dt (1 + 0.5 * 0.8^2)                 otherwise
proxy       = coefficient^2 mean_elements(movement^2) / (2 std^2)
keep        = active and not (proxy >= 1e-5 and alignment > 0)
```

`l_art` enters the window only for kept items. A masked or inactive item
contributes zero but still counts in the window's item denominator, matching the
reference's full-window normalization with `restore_zero_signal = false`. The
`0.8` noise level, the `1e-5` KL threshold and the `1e-5` epsilon floors are the
paper's constants, not campaign knobs.

A trained ART round uses two student updates; the ART arm fixes that
count instead of reading `nft.updates_per_round`. Each learner shuffles its
draws with a private round-keyed RNG before splitting them into equal disjoint
halves. The order is independent of microbatch packing, and every trajectory
site of a half trains once in its window. Both windows regress targets frozen
at collection time, so the second window's residual `G - d` restores motion the first window
produced. `reference_coefficient` composes exactly as under NFT: a velocity MSE
between the student and the frozen reference at the same items.

When every prompt's guidance is inactive and `reference_coefficient = 0`, the
round commits as skipped without optimizer, LR or student EMA updates. The
next collection uses fresh sample IDs and seeds. An enabled reference penalty
remains an independent training objective, so it still receives both updates.
An active round retains both windows even if one window's items are all masked.

### Reference coordinate mapping

The ART reference runs in decreasing noise time `sigma` with velocity
`epsilon - x`, so its ODE coefficient is negative and its velocity points
opposite to Inko's. Set `t = 1 - sigma` and `v_inko = -v_ref`; the guidance
covariance is computed on terminal latents, which both conventions share.

| Reference (`sigma`, `epsilon - x`) | Inko (`t = 1 - sigma`, `x - epsilon`) |
|---|---|
| time `sigma` | `1 - t` |
| ODE coefficient `dt_ref < 0` | `dt_ours = grid[k+1] - grid[k] > 0` |
| loss weight `sigma` | `1 - t` |
| frozen target `v_old_ref - G` | `v_old + G`, carrying the same `G` |
| `residual = target - v_new` | unchanged (both factors change sign) |
| `alignment = <movement, residual>` | unchanged (both factors change sign) |
| `std = 0.8 sqrt(sigma / (1 - sigma) * (-dt_ref))` | `0.8 sqrt((1 - t) / t * dt_ours)` |
| initial branch at `sigma = 1` | initial branch at `t = 0` |
| `coefficient = dt_ref (1 + 0.5 * 0.8^2)` | `dt_ours (1 + 0.5 * 0.8^2)`, sign removed by the square |
| active requires `sigma > 0` | active requires `t < 1` |
| unbiased covariance over `K - 1` | unchanged |
| full-window item denominator | unchanged |

Every mapped quantity enters the proxy squared or through an inner product that
is invariant under the double sign flip, so the translated formulas are the
reference's, not an approximation of them.

## Reference anchor

The frozen reference is the entry student: the base weights under LoRA, a full
copy under full tuning. On the same `(z, t, c)` inputs as the trained items of
either objective:

```text
l_ref = mean_elements((v_new - stopgrad(v_reference))^2)
L = mean(l_nft or l_art) + reference_coefficient * mean(l_ref)
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

The current Inko recipe ramps `d` from 0 toward 0.5 over 1000 NFT rounds.
The NFT paper's default ramp is `min(0.001 * i, 0.5)` and its OCR schedules
retain more of the old policy; these are task-specific recipe choices.
The old policy moves only at a committed boundary; a resumed run restores the
exact decay it recorded.

## Clocks and recovery

`step` counts successful student optimizer updates, `committed_rounds` counts
complete scored collections including skipped ART rounds, and `skipped_rounds`
counts collections that produced no update. Thus
`step = (committed_rounds - skipped_rounds) * updates_per_round` at a committed
boundary. `old_policy_version` counts committed refreshes; under NFT
one refresh happens per committed round, so the version equals the
committed-round count, and the ART arm keeps it at zero. Named student EMA
shadows advance on each successful student step and are independent of the old
policy. The LR schedule runs on the student's successful-step clock.
The training horizon, checkpoint cadence and interval evaluation count
committed collections, including skips. A cosine LR schedule uses the nominal
maximum update count (`rounds * updates_per_round`); skipped rounds do not
advance that schedule or the student EMA.

Complete-round checkpoints retain every role in use (student, the NFT old
policy, optional reference, EMA shadows), the student optimizer and its clock,
`last_reward`, both round counts, the prompt cursor and per-rank RNG. Files are
fsynced before a manifest-bearing directory is atomically renamed; incomplete directories are
never recovery points. Resume requires an unchanged training contract and
verifies exact restoration before collecting again; work inside an uncommitted
round is discarded and recollected.

## Verification and references

`tests/` covers operator values and gradients against independent oracles,
group statistics both locally and through real learner collectives, clock and
scheduling mechanics, role isolation, probability admission, the cache-to-item
trajectory round trip, and exact save/resume. Generation quality is established
only by real samples and benchmarks under the
[evaluation protocol](posttrain_evaluation.md).

[DiffusionNFT](https://arxiv.org/abs/2509.16117v2) defines the forward-process
reward optimization, the implicit positive/negative branches and the
old-policy formulation implemented here.
[DiffusionART](https://arxiv.org/abs/2610.12572) defines the reward-endpoint
covariance guidance, the frozen-target regression and the directional
corrective mask.
