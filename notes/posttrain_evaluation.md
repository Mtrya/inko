# Posttraining evaluation protocol

Status: initial design closed; sampling/scoring implementation is in progress.
The 480k EMA, 640p shapes, prompt manifests, and HPSv3/VQAScore v3.1 source/weight
identities are pinned. Target-runtime preparation and real-image qualification
remain pending; no quality results are established. Full-run preparation is a later
work layer. See the [contract](posttrain_contract.md) and
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

## Small starting suite for algorithm development

Start with the smallest real-image comparison that can expose a broken update
or an observable difference between variants. This does not require preparing
the full-run data distribution, a reward ensemble, or a new reference-image
corpus. The initial inputs are:

| Input | Choice | Preparation remaining |
|---|---|---|
| Initialization | Preserved 480k EMA, at 640p; digest in config | Real-image baseline qualification |
| Rollout prompts | Fixed 512-prompt subset of DiffusionNFT's `pickscore/train.txt`; 32 of these for bring-up | Manifests prepared; semantic overlap review can refine later comparisons |
| Scored development benchmark | 200 unique DrawBench prompts, one fixed seed each; fixed 16-prompt subset for bring-up | Manifests and seeds prepared; obtain real samples/scores |
| Visual monitoring | All 48 entries of [monitor.jsonl](../configs/prompts/monitor.jsonl), with existing seeds and aspects | Explicit shapes in config; generate and review panels |
| Reward/score clients | HPSv3 and VQAScore v3.1 with Qwen3-VL-8B-Instruct; no custom VLM-rubric scorer | Pins and adapters implemented; dependencies/weights and Ascend qualification pending |
| Initial report | Separate raw results, paired differences, scene-clustered bootstrap, failures, and cost telemetry | Report mechanism implemented; blinded review requires generated images |
| Diversity check | Multiple fixed seeds on a small declared subset when comparing trained variants | Select the subset and draw count before the comparison |

The inspected [DiffusionNFT prompt source](https://github.com/NVlabs/DiffusionNFT/tree/cbb14f84b8312b620390dfcbe2fab69c1383b104/dataset)
contains 25,432 unique `pickscore/train.txt` prompts. Its DrawBench test file has
1,000 rows but only 200 distinct prompts, each repeated five times. Use unique
prompts and our own explicit draw seeds rather than inheriting that repetition.
Only text/metadata are needed; no preference-pair images need downloading.
The 512-prompt rollout pool is the starting scope for investigating
learning behavior; its adequacy for selecting an algorithm still needs evidence.

Call the score report a DrawBench prompt evaluation with HPSv3/VQAScore and
record its sampling protocol. These metric values are not interchangeable with
DrawBench human-evaluation results or another model's different scorer setup.
The 16-prompt bring-up subset is a functionality check, not the basis for a
general algorithm-quality claim. Expand seeds/cases when a comparison is close.

The monitor suite has 12 scenes, each with Chinese/English and short/long
variants. Its 48 entries are not 48 independent semantic cases. Keep all scene
variants out of rollout training. Report per-prompt outcomes and scene-grouped
summaries; if estimating uncertainty, resample scenes rather than pretending the
language/length variants are independent observations.

## Initial scorer contract

[HPSv3](https://github.com/MizzenAI/HPSv3) supplies the preference signal;
[VQAScore](https://github.com/linzhiqiu/t2v_metrics) supplies an image/text
alignment signal. Keep both raw outputs. Their numeric scales are different;
do not silently average them. A training mixture needs explicit transforms and
weights, while evaluation reports the two metrics separately.

Use the upstream VQAScore v3.1 version announced on 2026-06-04 with the
Qwen3-VL-8B-Instruct backbone, explicitly selecting `qwen3-vl-8b`. The
[upstream announcement and model list](https://github.com/linzhiqiu/t2v_metrics)
describe this implementation. Pin the exact source revision and weight revision
when preparing the runtime rather than relying on a moving branch or an implicit
model default. Immutable revisions are recorded in `configs/posttrain.toml`.
The v3.1 source commit is `74c720890c6ee3916e2f4facf9c671c8fe47ddf4`;
its commit timestamp is June 5 UTC, while its announcement says June 4.

Pin the question/answer template and probability calculation alongside those
artifacts. HPSv3 similarly needs exact weights/config and output-field semantics.
This initial setup uses the scorers' defined scoring procedures;
it does not add an open-ended VLM judge or custom grading rubric.

Use the short English development prompts for the first automated comparison.
Keep the bilingual/long-caption panel as visual evidence until the selected
scorers' language and text-length behavior has been checked. Record truncation
and preprocessing; do not silently replace a Chinese generation prompt with
English for scoring. Any cross-language scoring protocol needs its own identity.

The first benchmark report contains raw held-out scores, paired score changes,
blinded wins/ties/regressions, images, score failures, and basic cost telemetry.
If both scorers are optimized, both held-out metrics remain in-scorer signals;
blinded review supplies an additional view. An independent composition benchmark
can be added during layer 1 when needed to resolve a selection. Do not claim
independent reward-model validation until an independent evaluator has been used.

KID/reference-corpus preparation and broad benchmark coverage are not entry
requirements for the first increment. The broader evidence table below describes
later comparisons and final qualification. Expand development evaluation only
when a question or an ambiguous result needs it; algorithm selection must carry
the limits of the evidence actually collected. A tiny panel can detect failures
and compare visible cases, but cannot establish broad superiority.

Use one preserved checkpoint for a comparison. Algorithm development can start
before a particular pretraining endpoint exists; replace the reference only via
an explicit comparison boundary. Use 640p and record the exact aspect-dependent
shapes. Higher-resolution validation becomes necessary when the algorithm
choice depends on it, not as a prerequisite for preparing the first suite.

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
multiple noise seeds for one prompt. Final suite names, sources, and counts
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
appropriate uncertainty estimate. Initially use a paired scene-cluster bootstrap
with repeat count, confidence, and seed in config. With one seed per prompt,
intervals describe prompt variation conditional on those seeds, not training-run
or generation-seed variation. An ambiguous result remains inconclusive. Increasing a training reward
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
The development suite, scorer identities, sampling rules, and reporting format
are selected; failed-score accounting has local mechanism coverage. Before
ranking trained variants, qualify these paths with actual images and scorer
inference on the target runtime.

Algorithm comparisons require sample review and benchmark evidence appropriate
to their scope. Choose numerical acceptance margins and the cold-start-to-joint
transition criterion from development experiments and record them before the
corresponding comparison. Full-run suites, scorer calibration, and final holdout
selection are prepared in work layer 3; results from the real run are assessed
in work layer 4.
