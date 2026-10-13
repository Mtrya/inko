# Posttraining mathematical contract

The trainer optimizes the 600k base model with
[DiffusionART](https://arxiv.org/abs/2610.12572) in one stage.
`configs/posttrain.toml` declares the complete training configuration. Numerical
choices belong in that config; this note defines their meaning. See the
[recipe](posttrain_recipe.md) and [evaluation protocol](posttrain_evaluation.md).

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
[coordinate mapping](#reference-coordinate-mapping) does. Gradients reach the model through the student velocity prediction.

## Round structure

One collection round produces `prompts_per_round * draws_per_prompt` samples.
The live student collects every sample at the same committed optimizer step,
using the configured sampler with guidance 1. Two disjoint training windows
follow an active collection, then the round commits. A stop boundary that
splits a round is rejected before launch.

Collection noise is drawn from per-sample CPU generators keyed by run seed,
round and sample ID. Optimizer steps do not affect it, so comparison arms share
initial noise even when one arm skips a collection. Rank packing and
microbatch order do not change any sample's inputs. The window permutation
uses a separate keyed stream.

## Combined reward

Scorers return raw channels; `combine_round` orients each declared channel by
its direction, optionally standardizes it over the round, and mixes channels
with nonnegative weights that sum to one. Missing, failed or nonfinite required
scores abort before any optimizer update; they never become zero rewards.
The complete combined-reward map is admitted before guidance is constructed.

## ART objective

ART regresses frozen targets over the trajectory the rollout policy just
visited: the cached rollout prediction at each visited state is the anchor.

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

A trained ART round uses exactly two student updates. Each learner shuffles its
draws with a private round-keyed RNG before splitting them into equal disjoint
halves. The order is independent of microbatch packing, and every trajectory
site of a half trains once in its window. Both windows regress targets frozen
at collection time, so the second window's residual `G - d` restores motion the first window
produced. The optional reference anchor adds a velocity MSE between the
student and the frozen reference at the same items.

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
the ART windows:

```text
l_ref = mean_elements((v_new - stopgrad(v_reference))^2)
L = mean(l_art) + reference_coefficient * mean(l_ref)
```

`l_ref` is a velocity prediction penalty, not a distribution divergence. Its
coefficient is zero only for the anchor-off comparison, in which case no
reference role is allocated. The reference never receives gradients and never
changes during a run.

## Clocks and recovery

`step` counts successful student optimizer updates, `committed_rounds` counts
complete scored collections including skipped ART rounds, and `skipped_rounds`
counts collections that produced no update. Thus
`step = (committed_rounds - skipped_rounds) * updates_per_round` at a committed
boundary. Named student EMA shadows advance on each successful student step.
The LR schedule runs on the student's successful-step clock.
The training horizon, checkpoint cadence and interval evaluation count
committed collections, including skips. A cosine LR schedule uses the nominal
maximum update count (`rounds * updates_per_round`); skipped rounds do not
advance that schedule or the student EMA.

Complete-round checkpoints retain every role in use (student, optional
reference, EMA shadows), the student optimizer and its clock,
`last_reward`, both round counts, the prompt cursor and per-rank RNG. Files are
fsynced before a manifest-bearing directory is atomically renamed; incomplete directories are
never recovery points. Resume requires an unchanged training contract and
verifies exact restoration before collecting again; work inside an uncommitted
round is discarded and recollected.

## Verification and references

`tests/` covers operator values and gradients against independent oracles,
group statistics both locally and through real learner collectives, clock and
scheduling mechanics, role isolation, reward admission, the cache-to-item
trajectory round trip, and exact save/resume. Generation quality is established
only by real samples and benchmarks under the
[evaluation protocol](posttrain_evaluation.md).

[DiffusionART](https://arxiv.org/abs/2610.12572) defines the reward-endpoint
covariance guidance, the frozen-target regression and the directional
corrective mask.
