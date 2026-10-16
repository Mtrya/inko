# Posttraining evaluation protocol

The seven-channel run evaluates the student EMA against the fixed 600k pretrain
EMA every 50 collected rounds and at the final round. DrawBench, monitor,
LPIPS and held-out rubric judging have completed real bounded qualification.
The earlier two-reward selection campaign and its separate confirmation results
are recorded in the [recipe](posttrain_recipe.md). The new rollout pool's
confirmation split remains unused.

## Evaluation ownership

Benchmark loaders, execution, scoring, and reports belong in
`src/evaluation/benchmarks/`. They may reuse score clients from `src/posttrain/`
without importing a trainer or inheriting its reward mixture and normalization.
Evaluation must run independently against a selected checkpoint and saved config.

Existing [prompt panels](../src/evaluation/prompt_grid.py),
[KID evaluation](../src/evaluation/kid_eval.py), and
[blind comparison tooling](../scripts/eval/blind_panel.py) are starting assets.
Their current preprocessing and sampling scope need qualification for each new
benchmark. Tests verify evaluator mechanics, not model quality.

## Scorers and interpretation

The scoring service supplies five Qwen3.8-27B pairwise rubric channels,
[HPSv3](https://github.com/MizzenAI/HPSv3) preference scores and
[VQAScore](https://github.com/linzhiqiu/t2v_metrics) image/text probabilities
with Qwen3-VL-8B-Instruct. Evaluation reports raw channels separately, without
the training mixture or round standardization. Backend code owns each scoring
procedure; `configs/scoring_server.toml` selects models and execution settings.

Both auxiliary scorers are optimized during training. Their held-out values
remain in-scorer signals, so compare them with actual images and diversity.
Call the report a DrawBench prompt evaluation with HPSv3/VQAScore; these values
are not DrawBench human-evaluation results. KID and independent evaluators are
separate extensions, not outputs of this run.

Keep the original prompt language for scoring. Judge reasons can overclaim
fine text, counts and anatomy even when a broad preference is supported.
Inspect both candidates and the reference when a preference is surprising.
Near-identical images can receive different auxiliary scores or noisy prose;
these differences alone do not establish an update-induced regression.

## Suites and splits

`configs/prompts/drawbench.jsonl` holds the canonical 200-prompt DrawBench suite;
`monitor.jsonl` holds the fixed visual panel. Evaluation settings select scenes
and expand fixed noise seeds at runtime. The full-run rollout and held-out pools are prepared from `kaupane/inko-rollout-25k`;
local production metadata supplies its missing aspects. Deployment assets name
the prepared training and development JSONL files. Results retain evaluated rows.

| Split | Use |
|---|---|
| Rollout training | Online sample collection and reward optimization. |
| Development evaluation | Repeated variant selection and diagnosis; disjoint from rollout training. |
| Final evaluation | Held out from recipe selection and used once for the final assessment. |
| Monitoring panel | Fixed visual comparisons during development; repeated use makes it a development signal. |

Keep translations, paraphrases, and variants of the same underlying scene in the
same split. Fix evaluation seeds explicitly and distinguish prompt diversity from
multiple noise seeds for one prompt. The confirmation suite runs once, after the
final configuration is chosen.

## Interval readouts

| Suite | Samples and report |
|---|---|
| DrawBench development | 100 fixed scenes x four seeds per arm; separate HPS/VQA means, paired scene-bootstrap differences and all-pairs LPIPS-AlexNet within each scene (600 pairs per arm). |
| Monitor | All 48 fixed entries per arm, with their original seeds and aspects; visual panels only. The 12 scenes each have Chinese/English and short/long variants. |
| Development rubric | 32 fixed rows from the prepared development split, one matching seed per arm; five category wins/ties/losses, mean win credit and order-disagreement counts, plus raw auxiliary scores. |

All suites compare the entry model with the student EMA using the same
20-step, CFG-1, shape-shift generation settings at the configured aspect sizes.
Entry images are cached across intervals. Scores are computed for each interval;
rubric judgments compare the entry image with the current student image.
Use scene groups when estimating uncertainty; translations and length variants
are not independent cases. Monitor and rubric have one seed per row and do not
compute within-prompt LPIPS.

## Comparison protocol

Before each experiment, declare its question, primary selection criterion,
regressions to watch, and budget. Share initialization and prompt/seed
assignments where the comparison permits. Record code/config/checkpoint
identities, live or EMA weights, sampler rule, step grid, guidance, resolution,
text conditioning, VAE/preprocessing, and evaluator version.

Keep those conditions fixed within a comparison unless one is the experimental
factor. If an evaluator changes, rerun the control. State whether a comparison
matches optimizer updates, samples, or accelerator-hours; equal update counts
can conceal different rollout volumes, scorer demand and waiting. Compare
quality versus measured cost as well as endpoint quality when the methods have
different workloads.

Report sample counts, missing scores, exclusions, seed variability, and an
appropriate uncertainty estimate. Initially use a paired scene-cluster bootstrap
with repeat count, confidence, and seed in config. With one seed per prompt,
intervals describe prompt variation conditional on those seeds, not training-run
or generation-seed variation. An ambiguous result remains inconclusive. An
increase in a training reward alone cannot select a variant.

## Result artifacts

Each evaluation should retain a manifest, per-sample scores and failures,
aggregate/slice metrics, generated images, comparison panels, and timings.
The manifest identifies exact inputs and evaluation settings. Large artifacts
live in external run storage; machine-specific locations remain in internal logs.
Keep concise findings and reproducible conditions with any published decision.

## Seven-channel run monitoring

The new service supplies five pairwise rubric channels plus HPSv3 and VQAScore.
Interval evaluation keeps DrawBench and the monitor panel. It also selects a
fixed development subset, generates the entry model and student with matching
prompts, aspects and noise, and judges each pair in both orders. The optional
reference remains third and affects only style. Reports include each category's
wins, ties, losses, mean win credit and order disagreements; saved scores include
both-order reasons.
Entry images are cached across intervals. The reserved confirmation split is
not used during qualification or interval monitoring.

Training pairs compare draws within one policy, so each category's mean is
0.5 by construction. Track cross-model development preferences and actual images
to assess change. The base is small and weak; entry-model comparisons distinguish
its existing limitations from a new update defect. Bounded qualification found
no broad visual regression after six updates; it does not establish the final
500-round quality or the long-term effect of this reward mixture.
