# Posttraining recipe

Status: provisional design. No runnable posttraining config or trainer exists
yet, and no final recipe has been selected. The intended canonical config is
`configs/posttrain.toml`. This note will explain the selected settings and their
evidence; the config will contain every numerical training choice.

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
| `configs/posttrain.toml` | Complete current baseline across both training stages |
| `configs/experiments/posttrain/` | Complete configs for retained experimental comparisons |
| `configs/prompts/` | Rollout and evaluation prompt suites with explicit split identity |

These posttraining paths describe the intended layout; creating this scaffold
does not create their implementations. Validate required fields and incompatible
component combinations. Runs must preserve the complete resolved config and
artifact identities. Machine-specific paths, services, and credentials belong
in local project configuration.

## Settings awaiting selection

| Area | Starting intent | Evidence or decision needed |
|---|---|---|
| Teacher | Preserved pretrained Inko checkpoint | Live versus EMA, sampling/guidance comparison, capability assessment |
| Student sampling | 8-step release target | Update rule, timestep grid, guidance, and train/inference correspondence |
| Distribution matching | Online fake-score learning with backward simulation | Parameterization, normalization, update ratio, trainable parameter scope |
| Cold start | RT-DM, DynaDG, DynaRS | Score transform, adapter design, schedules, and transition timing |
| Joint objective | NFT with persistent DMD regularization | Loss reductions/balance, NFT old-policy update, and any additional regularization |
| Optimization | Configured separately for trainable roles | Optimizers, learning rates, accumulation, clipping, precision, and EMA |
| Rewards | Separate raw scores from objective transforms | Qualified scorer(s), rubric, mixture, normalization, and failure handling |
| Prompts | Explicit training and held-out pools | Provenance, domain/language mix, counts, and group size |
| Execution | Ascend training; qualify scorer placement | Memory, throughput, reward waiting, distributed updates, and recovery |
| Final selection | Real generations, benchmarks, diversity, and cost | Comparison protocol, repeat budget, and acceptable regressions |

## Qualification sequence

Use a preserved intermediate pretrained checkpoint for infrastructure bring-up.
The planned 640p pilot uses the 570k endpoint; final 896p decisions use the
completed 600k teacher. Pilot findings establish feasibility and candidate
settings, with transfer to the final resolution checked explicitly.

Implement sampling/measurement, minimal distribution matching, complete cold
start, then joint NFT+DMD. Add each mechanism with a measurable control before
broad experiments. Development increments are distinct from the two training
stages. Measure the actual target workload before setting main-run budgets.

Checkpoint/resume must cover the roles and clocks in the
[contract](posttrain_contract.md). A successful mechanism check establishes
execution behavior; quality selection follows the
[evaluation protocol](posttrain_evaluation.md).

## Selected recipe and decision evidence

Pending. For each selected setting, record its rationale, comparison conditions,
observed effect, cost, and limitations here. Publish enough evidence to understand
the choice without access to an internal experiment ledger. Link the complete
config once it exists rather than duplicating its numerical settings by hand.

## Final qualification and release artifacts

Pending. Record the selected checkpoint identity, exact config and prompt
versions, 8-step inference settings, final evaluation results and sample panels,
domain/diversity limitations, measured training/inference cost, and recovery
qualification. The final recipe is selected only after these results exist.
