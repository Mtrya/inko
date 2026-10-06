# Posttraining evaluation protocol

Status: draft. This protocol defines the evidence needed to compare posttraining
variants. Benchmark implementations, suites, budgets, and acceptance margins are
still to be selected. No posttraining quality result is established by this
scaffold. See the [contract](posttrain_contract.md) and
[recipe](posttrain_recipe.md).

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

## Suites and splits

Keep prompt suites under `configs/prompts/` with stable prompt IDs, source and
revision, split, language, domain, and intended aspect/resolution metadata.
Record prompt bytes or their content hashes with every result.

| Split | Use |
|---|---|
| Rollout training | Online sample collection and reward optimization. |
| Development evaluation | Repeated variant selection and diagnosis; disjoint from rollout training. |
| Final evaluation | Held out from recipe selection and used for the final assessment. |
| Monitoring panel | Fixed visual comparisons during development; repeated use makes it a development signal. |

Keep translations, paraphrases, and variants of the same underlying scene in the
same split. Fix evaluation seeds explicitly and distinguish prompt diversity from
multiple noise seeds for one prompt. Candidate suite names, sources, and counts
remain open; keep large external benchmark assets outside the repository and
record their immutable identity.

## Evidence to report

| Dimension | Required evidence | Selection still open |
|---|---|---|
| Prompt following | Composition, objects, attributes, relations, and Chinese/English breakdowns | Benchmark suite and scorer |
| Visual quality | Blinded sample review and an independent quality/preference signal | Review rubric and scorer |
| Domain coverage | Chinese painting, Western painting, people, and world-content slices; anatomy and texture observations | Slice membership and counts |
| Distribution and diversity | KID with a specified reference population, plus within-prompt and across-prompt diversity evidence | Reference corpus and diversity measure |
| Generalization | Training reward, held-out reward from the same scorer, and independent evaluation reported separately | Independent scorer and final holdout |
| Cost | Wall time, accelerator-hours, memory, rollout volume, scorer demand, and reward waiting | Pilot measurement budget |

An unoptimized scorer is a useful additional signal, not proof of independence:
record whether evaluators share model families or training sources with rewards.
Do not collapse all evidence into an unexamined composite score. Retain slice
results and regression examples alongside aggregate metrics.

## Comparison protocol

Before each experiment, declare its question, primary selection criterion,
regressions to watch, and budget. Share initialization and prompt/seed assignments
where the comparison permits. Record code/config/checkpoint identities, live or
EMA weights, sampler rule, step grid, guidance, resolution, text conditioning,
VAE/preprocessing, and evaluator version.

Keep those conditions fixed within a comparison unless one is the experimental
factor. If an evaluator changes, rerun the control. State whether a comparison
matches optimizer updates, samples, or accelerator-hours; equal update counts
can conceal different fake-score or reward costs. Compare quality versus measured
cost as well as endpoint quality when the methods have different workloads.

Report sample counts, missing scores, exclusions, seed variability, and an
appropriate uncertainty estimate. The uncertainty method and repeat budget are
open. An ambiguous result remains inconclusive. Increasing a training reward
alone cannot select a variant.

## Result artifacts

Each evaluation should retain a manifest, per-sample scores and failures,
aggregate/slice metrics, generated images, comparison panels, and timings.
The manifest identifies exact inputs and evaluation settings. Large artifacts
live in external run storage; machine-specific locations remain in internal logs.
Keep concise findings and reproducible conditions with any published decision.

## First qualification

The first deliverable is a reproducible comparison of the selected pretrained
teacher and its initial 8-step student, with real images, scores, and cost records.
It establishes a baseline, not evidence that distillation has succeeded.
Before using it to rank trained variants, select the development suite, scorer,
sampling rule, and reporting format, and exercise failed-score accounting.

Stage promotion and final selection require sample review and benchmark evidence.
Numerical acceptance margins and the cold-start-to-joint transition criterion
must be chosen from the pilot and recorded before the corresponding comparison.
