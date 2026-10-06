# Inko redesign plan

Inko is a bilingual text-to-image flow-matching
DiT, with a 532M-parameter model and an 8-step student as the release target.
**Pretraining and post-training target Ascend.** The 640p continuation runs
in SwanLab project `inko` from an independently migrated 480k checkpoint.
The 640p/896p source mixtures pass the eligible-row audit. Filtered row pools
use explicitly rebuilt sampler cycles; all 16 ranks verify exact restoration
of model, EMA, optimizer, scheduler and training RNG state. Longer-run
telemetry and generations establish training behavior under the mixture.

## Document map

- [Pretraining recipe](pretrain_recipe.md): architecture, complete training configuration,
  sampling and checkpoint contracts, and resolution transitions.
- [Pretraining infrastructure](infra_pretrain.md): measured execution gains and
  qualification limits.
- [Dataset plan](dataset_plan.md): domains, caption contract and eligible pools.
- [Posttraining contract](posttrain_contract.md): mathematical conventions,
  objectives, model roles, update clocks, and recovery requirements.
- [Posttraining evaluation](posttrain_evaluation.md): benchmark protocol,
  prompt splits, comparison evidence, and cost accounting.
- [Posttraining recipe](posttrain_recipe.md): provisional two-stage baseline,
  open selections, and the eventual selected recipe with supporting evidence.

Platform accounts, images, storage roots, secrets and job submission details
live in the machine-local `INSPIRE.md`. Completed experiment records with
useful additional evidence remain locally in the gitignored `notes/archive/`.
Working posttraining experiments and Stage 6 sub-stage progress live in
gitignored `notes/internal/posttrain_experiments.md` and
`notes/internal/posttrain_progress.md`. Selected findings and their rationale
are incorporated into the tracked notes so published decisions are self-contained.
The maintained notes contain the evidence needed to act on the current design.

## Design decisions

| Dimension | Choice |
|---|---|
| Data permissions | Research-only sources are permitted, including WikiArt, ArtBench-10 and FFHQ. Preserve per-sample provenance/license metadata and separate restricted mix entries so a clean variant can be assembled. Permission to use a source is distinct from its presence in the current corpus. |
| Anatomy coverage | Photos and paintings; balance face and full-body coverage, with roughly equal representation as the curation goal. |
| Corpus | Fixed eligible pools per resolution; source weights are explicit in the complete run config. See [dataset plan](dataset_plan.md). |
| Model | `inko`: h1152, 16 heads, 1 double-stream + 24 single-stream blocks, FFN width 3072, 532,496,992 parameters. |
| Text | Frozen Qwen3-0.6B, encoded online with a true early exit after layer 20. |
| Resolution | Variable-aspect 256p → 640p → 896p; 75:20:5 of optimizer updates, ending at 450k / 570k / 600k. |
| Positions | Centered image-grid RoPE; text pinned to a fixed diagonal. Progressive training supplies resolution transfer. |
| Platform account | Account-selected Inspire project, configured in `INSPIRE.md`. |
| Training hardware | Ascend for both pretraining and post-training; qualify each workload on its actual allocation. |
| Captioning | API-based VLM captioning with cached responses and recorded model/prompt provenance. |
| Modulation | Independent image/text and attention/MLP modulation in the double-stream block; shared attention/MLP modulation within each single-stream block. |
| Optimizer | Chunked Muon with original scaling, LR 0.02 and decay 0.0015; auxiliary AdamW LR 1e-4, ordinary decay 0.01 and conditioning-matrix decay 0.4. Full settings in [pretraining recipe](pretrain_recipe.md). |
| Monitoring | Fixed bilingual panels, loss probes and final KID; report observed strengths and limitations. Capability forecasts are not launch gates. |
| Release execution | Reproducible sampling settings; deployment device and performance targets are selected during publication work. |

The model uses rectified flow, logit-normal timestep sampling with resolution
time shift, AdaLN-zero initialization, QK-RMSNorm, gated SiLU FFNs, 2×2 latent
patches, fused pooled-text/time conditioning and branch RMSNorm before gates.
The Qwen-Image VAE supplies 16-channel, factor-8 latents. Caption dropout is 0.1.
Architecture and execution mechanisms live in code; numerical tunables live in
one strict run config covering the entire resolution curriculum.

## Stage status and decision evidence

| Stage | State | Result / next checkpoint |
|---|---|---|
| 0: platform setup | Complete | Repeatable launch/recovery tools and machine-local platform instructions. |
| 1: data curation | Prepared | Fixed eligible pools for each resolution, precomputed latents and online caption encoding. |
| 2: model experiments | Complete | Width/depth, modulation, text exit, positional scheme and optimizer selected. |
| 3: execution efficiency | Measured | CUDA operator and throughput comparisons; hardware-specific validation on Ascend. |
| 3.5: captions and buckets | Complete | Multi-caption row sampling, length curriculum and variable-aspect planning. |
| 4: Ascend qualification | Reference workloads and 640p restart verified | 16×910B2C plans, native transitions, recovery and monitoring; the 640p continuation passes all-rank restore and initial updates. The configured 896p pool needs its own qualification. |
| 5: pretraining | 640p continuation | Migrated 480k input, audited source weights, explicit sampler rebuild and exact all-rank restoration in `inko`; monitor through the 570k boundary. |
| 6: post-training | Contract and evaluation scaffolding | Sampling/measurement, DMD, RT-DM/DynaDG/DynaRS cold start, then NFT+DMD and comparative experiments; 640p pilot before final 896p qualification. |
| 7: publication | Pending completed training | Final inference pipeline, model release and demo. |

The following measurements explain retained choices; their experimental
recipes and hardware define their scope:

- Matched model experiments support h1152 and single-stream-heavy depth.
  Shared single-stream modulation yielded loss 0.9437 versus 0.9441, KID
  0.0190 versus 0.0195 and 8% lower peak memory in the matched comparison.
- Qwen k20 tied k28 on loss (0.92160 versus 0.92134), improved KID
  (0.00892 versus 0.00922), and received the stronger facial-structure review.
- The tested RoPE variants tied down to 320p transfer; both failed at
  zero-shot scale factors of at least 1.875. This supports progressive staging.
- An RMS-matched Muon comparison measures loss 0.91127 and KID 0.00699 at
  16k, versus AdamW 0.92134 and 0.00922. It does not qualify the selected
  original-scaling optimizer; that recipe has separate execution evidence.
- CUDA single-GPU steady throughput improved 54.99 → 72.98
  samples/s (1.327×); warm-inclusive throughput improved 55.11 → 67.94
  (1.233×). Target-topology performance requires its own measurement.
- Ascend qualification measures early-256p throughput
  656.29 → 878.91 samples/s (33.92%) with matched experiments and recovery
  checks. See [conditions and limits](infra_pretrain.md).

## Stage 5 — Pretraining

Execute [the complete pretraining recipe](pretrain_recipe.md) on Ascend. The continuous
600k schedule carries optimizer, EMA, caption curriculum and LR progress across
450k and 570k resolution boundaries. Record actual sample exposure, elapsed
time and NPU-hours throughout training.

All three plans target 16×910B2C, with accumulation **1/3/4**. The 640p/896p
plans use memory-sized micro-batches 5–12 / 3–6. Reference mixtures measure
173.47 / 83.90 samples/s; all 280 candidate aspect/length/batch tail cases pass.
These measurements do not establish throughput or training quality for the
configured eligible pools. Measurement scope and transition/recovery evidence are in [the infrastructure record](infra_pretrain.md).
Apply recipe and metadata amendments through the explicit checkpoint migration
tool before resume, preserving the original complete source. The selected
480k restart is mid-stage: its sampler cycles are rebuilt from the filtered
pools with empty queues at curriculum position 0.8. Continue normal
loss/gradient/functional monitoring through subsequent resolution transitions.

Review the fixed bilingual panel, live/EMA losses and internal stability
telemetry at the recipe's cadence. Follow the evidence-based review and
recovery rules in [the training recipe](pretrain_recipe.md). A final capability
benchmark at 600k (form pending) determines the Stage-6 SFT go/no-go.

Exit: a complete pretraining checkpoint, verified sampling and resolution transitions,
measured compute/exposure, and a record of observed abilities and limitations.

## Stage 6 — Ascend post-training

The release target is an **8-step student**. The provisional
[recipe](posttrain_recipe.md) follows two training stages:

1. Cold start with RT-DM, DynaDG, and DynaRS.
2. Joint DiffusionNFT and DMD, using distribution matching to regularize reward
   optimization against the teacher distribution.

This baseline starts the experiments. Objectives, components, optimization,
sampling, rewards, prompt/data mixtures, and custom techniques remain open
comparison axes. The exact losses and state transitions will be specified in the
[contract](posttrain_contract.md); benchmark and generation evidence determine
selection under the [evaluation protocol](posttrain_evaluation.md).

Develop through sampling and measurement, minimal distribution matching,
complete cold start, and joint training before broader comparisons. These are
implementation increments inside Stage 6, distinct from the two training stages.
Core logic, including reward/score clients, belongs in `src/posttrain/`;
benchmarks belong in `src/evaluation/benchmarks/`; rollout prompts belong in
`configs/prompts/`. The planned trainer uses complete explicit configs.

Qualify real rollout/decode, score acquisition, training updates, distributed
execution, and recovery on Ascend. Measure memory, compute, synchronization,
and reward waiting before choosing the topology and experiment budgets. Scorer
placement and any asynchronous execution follow measured workload needs.

Use preserved intermediate weights for infrastructure bring-up, the 570k
endpoint for the 640p pilot, and the completed 600k teacher for final 896p
decisions. The pilot informs cold-start transition timing and candidate settings;
transfer to the final resolution requires its own evidence.

Exit: a selected 8-step checkpoint with reproducible config and inference
settings, held-out benchmarks and image review, domain/diversity evidence,
measured cost, recovery qualification, and a recipe explaining the decisions
and limitations. The internal ledger and progress log track ongoing work;
selected evidence is preserved in the public notes.

## Stage 7 — Publication

Rewrite the standalone generation pipeline and demo **after all model training
is complete**, using finalized checkpoint metadata, text conditioning and
sampling behavior.

The project and model name is **Inko**, with architecture identifier `inko`.
Published model links, inference metadata and repository coordinates must
consistently identify the release. Artifact provenance retains exact source
experiment identities.

Publish the selected 8-step student to Hugging Face with weights, inference
configuration, model card, provenance/license constraints, evaluation and
limitations. Deploy and smoke-test a Hugging Face Space. Write the README with
installation, minimal inference/training examples and a navigable document map;
verify its quickstart in a clean environment.

Publish self-contained explanations of Flow Matching and DMD2/DMDR/NFT,
including equations, training/sampling procedures, implementation mapping and
limitations. Explain methods directly; private run nicknames or unavailable
discussions cannot carry the explanation. Estimate publication and hosting
costs before deployment.

## Current risks

- Persistent internal scale growth: follow the live/EMA, fixed-panel and
  checkpoint evidence in [the training recipe](pretrain_recipe.md).
- Resolution transitions: qualify each plan and preserve complete endpoints.
- Caption/judge provider changes: cache responses and record exact request
  identity; measure limits again on the intended workload.
- Reward hacking and diversity loss: keep held-out signals and domain review
  throughout joint training.
- Source restrictions: preserve provenance and separate restricted entries;
  release permissions depend on the selected data and artifacts.
