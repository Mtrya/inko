# Post-training

## Release overview

Post-training starts from the 600k pretrain EMA and uses DiffusionART to learn from rewards on the model's own generations. Inko's release weights come from two runs: a 300-round main run for general post-training and a 60-round specialized top-up. Both used attention LoRA rank 64 with alpha 64, Muon LR $10^{-3}$, 20-step Euler rollouts at CFG 1.0, and EMA decay 0.999.

| Run | Base initialization | Adapter initialization | Rounds / updates | Artifact or destination |
|---|---|---|---:|---|
| Main run | 600k pretrain EMA | Fresh rank-64 LoRA | 300 / 600 | Round-300 EMA |
| Specialized run | 600k base with the main round-300 EMA adapter merged in | Fresh rank-64 LoRA | 60 / 120 | Round-60 EMA, exported as the release |

Production collection draws 64 prompts with 16 images per prompt at 896p. Its seven reward channels combine five pairwise rubric judgments with HPSv3 and VQAScore.

![Six matched base and release pairs: green apples, a violinist, a dancer, an old man with a cane, two cats, and an ink-wash beauty.](assets/posttrain_base_final_comparisons.png)

Post-training improves counting and human structure in these matched pairs, but several outputs also drift toward photographic or Western watercolor aesthetics. The pairs were selected as case studies.

## Objectives and algorithm selection

Post-training was part of the plan before the 600k checkpoint was complete. Three weaknesses gave it a concrete target: counting, position and layout adherence; anatomical collapses; and composition across style and content, such as drawing a structurally coherent muscle car in ink wash. The third target is especially important for a painting model because conventional benchmarks rarely measure it directly. The specialized run later added a restorative target: recovering short-prompt ability weakened by long-caption prioritization during pretraining.

An earlier few-step distillation program investigated DMD at 480k, 640p and eight sampling steps. The experiments produced severe texture collapse and drift, and no tested endpoint justified a release recipe. A small model already generates quickly with 20 steps instead of 50, while further distillation threatened the diversity that Inko was designed to preserve. Few-step distillation was therefore closed, leaving full-step quality improvement as the post-training objective.

RL offered a different source of training signal from another supervised stage. Targeted SFT would require high-quality images that directly repair the base model's failures. Generating those images costs accelerator time or API fees, and the synthetic-data work in the [data chapter](data.md) had already shown that generation does not remove the need for careful filtering. The final pretraining data was also strong, so repeating the same supervised recipe promised less headroom than learning from comparisons between the model's own better and worse outputs.

The candidate set covered two algorithm families. Flow-GRPO and DanceGRPO represented policy-gradient RL, which was screened out on collection cost and implementation complexity. [DiffusionNFT](https://arxiv.org/abs/2509.16117) had already become a practical forward-process alternative: it was substantially faster and placed no special requirements on the generator. [DiffusionART](https://arxiv.org/abs/2610.12572) appeared while the implementation was underway. Its HunyuanVideo comparison reported 12.02× fewer GPU hours than DiffusionNFT, while its image experiments avoided NFT's per-task old-policy EMA tuning. That was enough to include ART as a challenger.

NFT constructs positive and negative objectives from forward-noised endpoints. ART instead aggregates each prompt group's rewards and terminal latents into one guidance direction, then applies that direction to states visited by the reverse rollout. Cached rollout velocities provide frozen targets, and two disjoint updates provide restoration without an old-policy EMA in the loss. The experiment compared how those differences behaved on Inko.

## Recipe selection

The recipe-selection experiments used the 600k EMA, 12 prompts with eight draws per round at 896p, and a 512-prompt PickScore rollout subset. HPSv3 and VQAScore had equal weights and round standardization. A separate development read used 100 DrawBench scenes with four seeds at CFG 1, with within-scene LPIPS measuring variation among the four draws. Higher LPIPS indicates more variation within a scene. These experiments established the recipe; their checkpoints did not initialize production. All development scores in this section belong to that two-reward setup.

### ART versus NFT

ART and NFT both used rank-32 attention LoRA with alpha 32 and Muon LR $10^{-3}$. Both completed 1,000 active collection rounds, with no round skipped for constant group rewards. ART made two updates per round and NFT one, giving 2,000 versus 1,000 optimizer updates and different EMA ages. Matching collection rounds matches rollout volume and nominal scorer demand; recipe selection used the development read and actual images rather than raw step counts.

ART produced stronger development HPSv3 and VQAScore gains throughout the useful learning region. The advantage was already visible before its largest diversity contraction. At similar within-scene LPIPS, ART round 300 had development HPSv3 6.004 and VQAScore 0.266 with LPIPS 0.752; NFT round 800 had development HPSv3 0.044 and VQAScore 0.209 with LPIPS 0.754. Round 500 improved both development scores further for ART. NFT's collection HPSv3 eventually collapsed to roughly −10. ART's later rounds brought diminishing score gains alongside continued diversity loss, so round 500 became the selection horizon for tuning scope, rate and rank.

![ART sustains higher collection HPSv3 than NFT, whose reward collapses late in the run. Lines are 20-round medians and shaded bands are interquartile ranges; these are training-side collection rewards, distinct from the development scores in the text.](assets/posttrain_art_nft_collection_hpsv3.png)

### Anchor, tuning scope, rate and rank

With ART selected, the next experiment tested whether an explicit pull toward the frozen base could improve the reward–diversity tradeoff. A fresh-base run changed only the frozen-reference velocity penalty from zero to $10^{-4}$. At round 600, anchored-minus-control differences were development HPSv3 $-0.4964$ with a paired 95% interval of $[-0.6501,-0.3430]$, VQAScore $-0.0103$ with an interval of $[-0.0362,0.0155]$, and within-scene LPIPS $-0.00965$ with an interval of $[-0.01631,-0.00266]$. Collection HPSv3 also dropped around round 470 and stayed near 6–8, versus 12–13 without the anchor, while portrait substitutions continued. The anchor was disabled for the remaining experiments and for production.

![The frozen-reference anchor lowers both collection HPSv3 and within-scene LPIPS on development prompts over most of the comparison. Reward lines show 20-round medians with interquartile bands; LPIPS points are evaluated every 50 rounds.](assets/posttrain_anchor_comparison.png)

The scope-and-rate grid compared full tuning with rank-32 attention LoRA at Muon rates $3\times10^{-4}$, $10^{-3}$ and $3\times10^{-3}$. Selection weighed adherence, within-scene diversity and actual generations at the 500-round horizon, rather than maximizing HPSv3 alone. The two higher-rate full-tuning arms learned quickly but later lost prompt adherence and subject diversity. LoRA at $10^{-3}$ gave the best measured balance. Raising the LoRA rate to $3\times10^{-3}$ added 1.131 HPSv3 but lost 0.0871 LPIPS and slightly reduced VQAScore; lowering it to $3\times10^{-4}$ learned too slowly for that horizon.

| Variant | Two-reward development read | Decision |
|---|---|---|
| Full, $3\times10^{-4}$ | Round 150: HPS 2.303, VQA 0.2184, LPIPS 0.7834 | Stopped at round 160 after LoRA selection |
| Full, $10^{-3}$ | Round 500: HPS 11.193, VQA 0.1453, LPIPS 0.4688 | Reject for adherence and diversity loss |
| Full, $3\times10^{-3}$ | At round 350, 31 of 32 fixed development samples were similar portraits | Stop for broad collapse |
| LoRA, $3\times10^{-4}$ | Round 500: HPS 8.016, VQA 0.2880, LPIPS 0.7523 | Slower learning; the live model also developed portrait bias |
| LoRA, $10^{-3}$ | Round 500: HPS 10.295, VQA 0.3267, LPIPS 0.7057 | Select |
| LoRA, $3\times10^{-3}$ | Round 500: HPS 11.426, VQA 0.3223, LPIPS 0.6186 | More diversity loss without an adherence gain |

Rank 32 and rank 64 were then compared at $10^{-3}$ with alpha equal to rank. Their round-500 measurements were comparable: rank 64 gained 0.113 development HPSv3, lost 0.00175 VQAScore and gained 0.00199 LPIPS relative to rank 32, with every paired 95% interval spanning zero. Rank 64 supplied extra capacity for the planned seven-channel run, and its additional adapter cost was small beside generation and scoring. That capacity argument, rather than a measured quality win, decided the rank. The selected export's 200-prompt held-out confirmation retained a clear HPSv3 gain of 10.477 with a 95% interval of $[9.942,11.034]$; its VQAScore gain was 0.0109 with an interval of $[-0.0317,0.0542]$.

### Portrait collapse under pointwise rewards

These selection experiments optimized only pointwise scorers. Across several arms, unrelated prompts gradually produced similar ornate portraits, often while collection HPSv3 approached 15. The full-tuning $3\times10^{-3}$ arm gave the clearest visual record: pine landscapes and baker portraits collapse into variants of the same hat-wearing figure. Reducing the tuning rate or using LoRA delayed the failure without eliminating it. The selected $10^{-3}$ LoRA arm later produced floral portraits on unrelated prompts, and the lower-rate live model showed the same bias.

![At full-tuning Muon $3\times10^{-3}$, round 350, the Chinese short-prompt pine and baker scenes both become hat-wearing portraits. The 600k base generations on the left remain distinct.](assets/posttrain_pointwise_collapse.png)

The working interpretation is overfitting to HPSv3 and VQAScore, with HPSv3 as the suspected primary driver and diversity loss amplifying the attractor. The experiments did not isolate the two scorers' individual contributions. This failure motivated production's custom pool and seven-channel reward, with five pairwise rubric channels primary and the pointwise scorers auxiliary.

Production started fresh from the 600k EMA with the selected ART recipe, the new pool and the new reward. The scope, rate and rank grid was not repeated under the seven-channel reward.

## Production architecture and reward design

### One round, from prompts to two updates

Production uses one 16-device learner node and one 16-device scoring node. The learner node holds the student, frozen text encoder, VAE and host-resident trajectory caches. The scoring node keeps seven Qwen3.8-27B serving replicas in TP=2 form on devices 0–13 for the five rubric channels, HPSv3 on device 14 and VQAScore on device 15. This separation makes model residency straightforward and lets rollout generation overlap the slow reasoning-based judgments.

The learner takes 64 prompts from the prepared training cursor and collects 16 draws for each. Collection uses 20 deterministic Euler steps, CFG 1.0, shape-dependent timestep shift and text exit layer 20. The three 896p collection shapes are 896×896, 768×1024 and 1024×768. Using CFG 1 for interval monitoring followed the pretraining image-grid convention and matched rollout collection; the final CFG 4.5 sweep belongs to [release evaluation](evaluation.md).

![One DiffusionART round. Asynchronously submitted images feed rubric and pointwise scoring in parallel; the complete coordinator reward file gates 16-draw ART guidance, two update windows and EMA/checkpoint state.](assets/posttrain_round.png)

For every draw, the learner caches 20 prediction-site latents and their predicted velocities in FP32 host memory; the terminal latent stays on device. After each local microbatch, VAE decoding and PNG encoding run synchronously, then a bounded HTTP pool submits the image and its IDs asynchronously. The prompt and rubric were already sent when the collection opened. Image-seeded rows also supplied their reference image for style-and-medium comparisons.

The scoring service forms eight disjoint pairs within each prompt group and judges every pair in both presentation orders. These pairs cover all 16 draws once per rubric category, while the two orders expose presentation-order disagreement. HPSv3 and VQAScore score the same terminal images pointwise. The coordinator waits for all seven channels on all 1,024 images, assembles the weighted rewards and writes a shared reward file. Every learner waits for that file before guidance, so scoring overlaps generation but never crosses the update barrier.

The combined reward for draw $k$ is $R_k=\sum_c w_c\tilde r_{c,k}$, where $\tilde r_{c,k}$ is the transformed value of channel $c$. For each complete 16-draw prompt group, ART computes a reward-endpoint covariance and normalizes it to unit RMS:

$$\overline R=\frac1K\sum_{k=1}^K R_k,\qquad r_k=R_k-\overline R,$$
$$\operatorname{cov}=\frac{1}{K-1}\sum_{k=1}^K r_kx_k,\qquad \operatorname{MAD}=\frac1K\sum_{k=1}^K|r_k|,$$
$$G=\frac{\operatorname{cov}/\operatorname{MAD}}{\operatorname{RMS}(\operatorname{cov}/\operatorname{MAD})}.$$

Here $x_k$ is the terminal latent and $K=16$. The group is completed across learner ranks before guidance is computed. Exactly constant combined rewards make the group inactive.

At every cached trajectory state, the frozen target is $v_{\text{target}}=\operatorname{stopgrad}(v_{\text{old}}+G)$. Inko uses $t=0$ for noise and $t=1$ for data, so the per-item loss weights velocity MSE by $1-t$. A directional trust-region mask uses a KL-style proxy for movement away from the cached rollout prediction. It rejects an item only above that threshold and when the displacement still points outward; corrections back toward the cached prediction remain active. The optional frozen-reference velocity penalty was set to zero.

Within each learner, the round splits whole trajectories into two disjoint windows. With 16 ranks, each rank holds 64 trajectories, one draw from each prompt, and privately shuffles them into two sets of 32. Each window therefore contains 512 trajectories globally and all 20 states from those trajectories, followed by one Muon update; the per-prompt counts vary between windows. Both windows use targets frozen at collection time. If $d=v_{\text{new}}-v_{\text{old}}$, the second window's residual is exactly $G-d$, which supplies restoration as the student moves away from the rollout prediction. All trainable tensors are LoRA A/B matrices optimized by Muon.

ART removes the old-policy EMA from its training objective. Inko updates a separate EMA of the trainable LoRA parameters after each optimizer update for interval evaluation and export. Checkpoint publication every 20 rounds retains the student, EMA, optimizer, prompt cursor, clocks and per-rank RNG state. Evaluation is outside the training loop: the trainer stops at an evaluation boundary, exports a committed EMA checkpoint, evaluates it and resumes. Resume requires an unchanged configuration, prompt pool, world size and scorer contract. The fixed visual monitor suite compares the entry model with the student EMA on 48 entries from 12 scenes, covering Chinese and English, short and long prompts at matched seeds and aspect ratios.

<details><summary>ART coordinate mapping and trust-region details</summary>

The ART reference uses decreasing noise time $\sigma$ and velocity $\epsilon-x$. Inko sets $t=1-\sigma$ and $v=x-\epsilon$, so the reference's negative integration step becomes a positive $dt$ and its target $v_{\text{old,ref}}-G$ becomes $v_{\text{old}}+G$ in Inko's coordinates. Products and squared terms in the restoration and trust-region calculations are invariant under the paired sign changes.

For a cached state $z$, physical time $t$, rollout step $dt$, current displacement $d=v_{\text{new}}-v_{\text{old}}$ and residual $\operatorname{target}-v_{\text{new}}$:

```text
alignment   = mean_elements(stopgrad(d) * stopgrad(residual))
l_art       = (1 - t) mean_elements(residual^2)
std         = 0.8                                  at t = 0
              0.8 sqrt((1 - t) dt / t)              otherwise
coefficient = dt                                   at t = 0
              dt (1 + 0.5 * 0.8^2)                 otherwise
proxy       = coefficient^2 mean_elements(stopgrad(d)^2) / (2 std^2)
keep        = active and not (proxy >= 1e-5 and alignment > 0)
```

The 0.8 reference noise level and $10^{-5}$ threshold are ART paper constants. Masked and inactive items contribute zero loss while remaining in the window's item denominator. Collection IDs and seeds advance through an entirely inactive round, but optimizer and EMA state remain fixed.

</details>

### Channel transformations and weights

Merging the two presentation-order decisions gives each draw a scalar rubric reward on the raw 0, 0.5 and 1 scale. HPSv3 and VQAScore are standardized over the complete round before the weighted sum: $\tilde r=(r-\mu_{\text{round}})/(\sigma_{\text{round}}+\epsilon)$. Round standardization was inherited from the earlier reward-combination pipeline and was not separately ablated. The deliberate production choice was to leave the rubric channels unstandardized.

Raw rubric rewards make the mixture responsive to the amount of decisive pairwise evidence. When the judge finds meaningful differences, those channels carry substantial within-group variation. As ties accumulate, rubric variation and influence naturally fall, so the standardized pointwise scorers account for a larger relative share of the remaining guidance. Standardizing every rubric channel would instead amplify the few remaining decisive pairs, which could make the updates less stable precisely when judgments become difficult.

<details><summary>From rubric standard deviation to decisive pairs and direction energy</summary>

Each prompt's 16 draws form eight disjoint pairs. For one rubric category, a decisive pair receives $(0,1)$ after both presentation orders are merged; a tie or order disagreement receives $(0.5,0.5)$. If $\rho$ is the decisive-pair fraction, the image-level channel has mean $1/2$ and population variance $\sigma^2=\rho/4$, giving

$$\rho=4\sigma^2.$$

The logged quantity is the round-level population standard deviation across all draws, so this conversion applies directly. For example, $\sigma=0.20$ means that 16% of pairs were decisive. The ratio describes judged outcomes: a tie can mean genuinely similar candidates or a distinction that the judge missed.

![Decisive-pair ratios decline in all five rubric channels during the main run. Lines are 10-round medians of $4\sigma^2$, converted from the logged population standard deviations.](assets/posttrain_rubric_decisive_pairs.png)

Standard deviation alone does not measure a pointwise channel's contribution to ART guidance. For prompt group $g$, center each transformed channel within that group and form its weighted covariance with the terminal latents:

$$\Delta_{g,c}=\frac{w_c}{K-1}\sum_{k=1}^{K}(\tilde r_{c,k}-\overline{\tilde r}_{c,g})x_{g,k},\qquad \Delta_g=\sum_c \Delta_{g,c}.$$

The vector $\Delta_{g,c}$ is channel $c$'s contribution before the common MAD and unit-RMS normalization. Comparing its norm across channels shows which rewards supply within-group directions; the squared norm is the corresponding direction energy. Summed per-channel energies are an attribution diagnostic rather than a decomposition of $\|\Delta_g\|_2^2$, because cross-channel inner products determine whether directions reinforce or cancel.

</details>

The main mixture guided general post-training. The specialized mixture was assigned with counting, people, spatial relations and short-prompt recovery as its targets.

| Channel | Main weight | Specialized weight | Transformation |
|---|---:|---:|---|
| Content alignment | 0.16 | 0.20 | Raw pairwise outcome |
| Form and structure | 0.16 | 0.20 | Raw pairwise outcome |
| Aesthetic composition | 0.16 | 0.06 | Raw pairwise outcome |
| Style and medium | 0.16 | 0.06 | Raw pairwise outcome |
| Artifacts | 0.16 | 0.20 | Raw pairwise outcome |
| HPSv3 | 0.05 | 0.06 | Round standardization |
| VQAScore | 0.15 | 0.22 | Round standardization |

The coefficients were hand-tuned after inspecting these measurements rather than fitted to an energy target. Early measurements put the rubric family's share of summed per-channel direction energy at about 68%. HPSv3 had a within-group standard deviation of only about 0.38, with roughly 85% of its variance occurring between prompts. Those observations informed the intended hierarchy: rubrics primary, HPSv3 and VQAScore auxiliary. The main coefficients were fixed before launch, and the specialized coefficients were fixed before the 60-round top-up; neither run changed weights mid-run.

HPSv3 stayed deliberately small because it moved quickly in selection experiments and was associated with artifacts and reward hacking. The style channel often produced few decisive pairs, limiting its usable style-direction information. The specialized weights moved more mass toward content, form and artifacts for the top-up targets, reduced aesthetics and style, and gave VQAScore a stronger auxiliary role. These coefficients also reflected the main run's weak style signal and the slight texture fading in its pine-style monitor entry.

### Scoring-time engineering

Reasoning-based scoring was the dominant production cost, so runtime work began with the serving path. Greedy decoding could repeat indefinitely on generated lettering. Sampled decoding with medium reasoning ended all three failing requests. On that targeted failure cohort, medium reasoning used 6,826 completion tokens versus 15,903 with the default xhigh setting, a 57% reduction. Disabling reasoning was faster but confused references with candidates and invented objects. A constrained five-line final response made verdict parsing strict without constraining the reasoning itself.

Seven TP=2 replicas provided more useful parallelism than the measured alternatives. Fourteen TP=1 replicas had limited KV capacity and much lower decode throughput. Three TP=4 replicas used 12 devices and sustained about 3,134 generated tokens per second in total, or 261 per device. The selected TP=2 deployment used 14 devices and sustained about 4,458 tokens per second, or 318 per device. Speculative decoding was rejected after premature termination and missing verdicts under real image load. Available-slot dispatch also replaced a scheduler that could send new work to a busy replica while another was idle.

The Ascend runner initially copied the full logits matrix to CPU for structured-output masking on every decoding step, including unconstrained reasoning rows. A selective path transfers only rows with an active grammar mask. It produces identical masks and reduced a separate 1,536-image qualification workload from 1,168.63 to 1,065.93 seconds: an 8.8% reduction in elapsed time and a 9.6% throughput increase. Mean judge call time fell 27.5%. The alternative NPU masking kernel was exact but slower for this workload.

Learner-batch increases, compilation and frozen-weight cast caching supplied no useful additional speedup. Without a larger serving win, the original 80-prompt × 48-draw collection could not fit the runtime budget. The final 64-prompt × 16-draw shape reduced the number of prompt groups while protecting 16 draws for pairwise guidance and 512 whole trajectories per update across 16 ranks. ART reports stability with groups smaller than 16; retaining 16 was a conservative choice for the new reward system.

The completed main run averaged about 784 seconds per round: roughly 107 seconds collecting, 445 seconds waiting for scoring, and 233 seconds across the two updates. The implementation did not achieve full pipeline overlap, and large bubbles remained. These measured components show why scorer optimization mattered, and why increasing LoRA rank added relatively little cost.

## Production training and release

### The 300-round main run

The main run started fresh from the 600k EMA with the production seven-channel reward. It collected 307,200 images and trained 6.144 million trajectory sites over 300 rounds and 600 updates. Neither production run skipped a collection for constant combined rewards. Qualitative review of the fixed 48-entry monitor suite found no severe portrait attractor in the main run.

Most collection-reward gains arrived within the first 100 rounds. Mean HPSv3 reached 10.420 in rounds 51–100 and stayed within 10.423–10.554 in the later 50-round blocks; VQAScore stayed within 0.824–0.830 after reaching 0.806. The rubric standard deviations declined over the same run as fewer pairs supplied decisive variation.

![Collection HPSv3 and VQAScore largely flatten within the first 100 rounds. Lines are 10-round medians and shaded bands are interquartile ranges.](assets/posttrain_main_collection_rewards.png)

Production development evaluation used 100 DrawBench scenes with four seeds at CFG 1, rather than prompts from the rollout pool. HPSv3 and VQAScore started much lower than their collection counterparts because they measured this different prompt distribution. These values also form a separate trajectory from the selection experiments: production used the seven-channel reward with small auxiliary HPSv3 and VQAScore weights, rather than the equal-weight two-reward mixture.

Both development point estimates were still rising at round 300, while within-scene LPIPS continued to decline. At round 300, gains over the 600k base were 7.81 HPSv3 with a paired 95% scene-bootstrap interval of $[7.11,8.52]$, and 0.110 VQAScore with an interval of $[0.071,0.153]$. Both scorers were optimized during training, so these curves remained diagnostics alongside actual images and LPIPS.

![Production development HPSv3 and VQAScore keep rising through round 300 while within-scene LPIPS declines. Points are 50-round evaluations on 100 DrawBench scenes with four seeds at CFG 1; dashed lines mark the 600k base.](assets/posttrain_main_development.png)

The stopping decision came from the collection curves and qualitative monitor reads. In `style_pines`, the student showed a very slight loss of the faded texture; the other entries in the 48-entry panel did not establish a broader texture regression. With collection rewards flattening, decisive rubric pairs falling, diversity declining and development scores still rising, the planned 500-round run stopped at 300 rather than retune the channels and begin another cycle.

### The 60-round specialized top-up

The specialized run targeted counting, people and spatial relations, and also attempted to recover short-prompt ability. Long-caption prioritization during pretraining had gradually weakened that ability. Its pool added targeted scenes and synthetic short prompts, while the reward weights favored content, form and artifacts. The prompt pool and channel weights changed together, so the resulting readouts measure their combined effect. The data split is documented with the rest of the [post-training pool](data.md).

A fixed 60-round budget was chosen before launch, without a more specific numerical justification. The scorer channel configuration is part of the resume contract and could not change mid-run, so the top-up required a new run. The main round-300 EMA adapter was merged into the 600k base, and a fresh rank-64 adapter was initialized on the resulting weights. This was an infrastructure consequence; a direct continuation with changed prompts and weights might be better.

The same 100-scene DrawBench interval protocol evaluated the top-up every 20 rounds.

| Checkpoint | HPSv3 | VQAScore | Within-scene LPIPS |
|---|---:|---:|---:|
| Main round 300 (top-up entry) | 3.061 | 0.259 | 0.7525 |
| Specialized round 20 | 3.154 | 0.263 | 0.7527 |
| Specialized round 40 | 3.378 | 0.272 | 0.7531 |
| Specialized round 60 | 3.630 | 0.269 | 0.7517 |

Relative to the main-run endpoint, specialized round 60 gained 0.570 HPSv3 with a paired 95% scene-bootstrap interval of $[0.443,0.722]$. VQAScore dipped from its round-40 read but remained 0.0105 above the top-up entry; its interval of $[-0.0053,0.0252]$ included zero. LPIPS ended 0.0008 below the main-run endpoint.

### Release and remaining headroom

The release is the specialized round-60 EMA, after 600 updates to the main adapter and 120 updates to a fresh specialized adapter. Each run's EMA starts at a zero adapter delta. After $n$ updates, its initial state retains coefficient $0.999^n$: about 0.549 after the 600 main updates and 0.887 after the 120 specialized updates.

Across the two adapters, the release received only 720 updates, and the main development curves were still rising at handoff. In that sense, the model is under-post-trained; more rounds remained technically feasible. The limiting factor was the reward system: even pairwise reasoning with reference images had limited sensitivity to artistic texture and style. Continuing could have optimized the measurable channels further while moving away from Inko's actual target. The conservative stop and EMA were chosen to limit diversity and texture loss while accepting unused score headroom, leaving room for a reward system with stronger artistic sensitivity.
