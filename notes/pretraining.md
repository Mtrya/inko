# Pretraining

## Pretraining at a glance

Inko's trainable diffusion backbone has **532,496,992 parameters**. Pretraining ran for 600,000 optimizer updates on 16 Ascend 910B2C cards, using [rectified flow](https://arxiv.org/abs/2209.03003) with a frozen Qwen3-0.6B text encoder and a frozen REPA-E image VAE. One global learning-rate schedule carried the model through three resolutions.

| Resolution | Optimizer updates | Active duration |
|---|---:|---:|
| 256p | 450,000 | 160h 12m |
| 640p | 120,000 | 119h 11m |
| 896p | 30,000 | 44h 03m |
| **Total** | **600,000** | **323h 26m** |

The durations sum the active training segments, including evaluation and checkpoint work, and exclude paused time: about 5,175 Ascend card-hours in total. The full-run curves mark four interruptions: new or refreshed training data at 250k and 400k, a source-weight correction at 480k, and the conditioning-decay intervention at 56k. The [data timeline](data.md#the-corpus-and-its-training-timeline) gives the dataset changes; the stability section reconstructs 56k.

## Architecture under the 0.6B budget

The architecture was designed to a **0.6B trainable-parameter budget**.

| Component | Final setting |
|---|---|
| Backbone | Inko, hidden size 1152, 16 attention heads, 1 double-stream block followed by 24 single-stream blocks |
| Feed-forward | SwiGLU, width 3072, an 8/3 ratio to the hidden size |
| Text path | Frozen Qwen3-0.6B, online early exit at layer 20; 1024-wide features projected to 1152 |
| Image latents | Frozen [`REPA-E/e2e-qwenimage-vae`](https://huggingface.co/REPA-E/e2e-qwenimage-vae), 16 channels, spatial factor 8, 2×2 latent patches |
| Conditioning | Masked-mean text pooling fused with sinusoidal timestep features; factor-1000 timestep input; branch RMSNorm before residual gates |
| Normalization | Affine-free pre-branch LayerNorm; QK-RMSNorm and branch-output RMSNorm with learned gains; AdaLN-zero gates |
| Modulation | Independent image/text and attention/MLP modulation in the double-stream block; shared attention/MLP modulation in each single-stream block |
| Positions | Centered image RoPE and a fixed text diagonal |
| Optimizer | Chunked Muon for the main hidden matrices, auxiliary AdamW for embeddings, conditioning and output heads, norms and biases |

![Inko architecture. A frozen Qwen3-0.6B encoder supplies token features and masked-mean pooled text. The token features join patchified image latents in one double-stream block and 24 single-stream blocks; pooled text and sinusoidal timestep features modulate every block. The transformer predicts latent velocity.](assets/pretrain_architecture.png)

Branch RMSNorm and the factor-1000 timestep input are explained through the evidence in [Training stability](#training-stability).

Single-stream processing was attractive because double-stream blocks carry roughly twice the parameters at comparable compute. [Z-Image](https://arxiv.org/abs/2511.22699)'s comparison of stream architectures favored all-single-stream processing for image quality, with weaker text rendering as the tradeoff. Since text rendering was not an Inko target, that tradeoff had little weight. [FLUX.1](https://github.com/black-forest-labs/flux) supplied the double-to-single ordering. A bounded matched-compute comparison of double-stream, single-stream and hybrid models found nearly identical loss, making parameter efficiency the deciding factor. The initial choice was full single-stream.

The final 1+24 assembly came later. Twenty-four single-stream blocks gave roughly 485M parameters; I wanted the model slightly larger, and the one added block was double-stream, bringing the total to 532M.

The width–depth balance extrapolated a directional result from only two roughly 100M candidates: wider and shallower won on loss and Kernel Inception distance (KID), so I used 1152 with 24 single-stream blocks. A grid search to fit an optimal ratio would have been better, but was not performed. Head dimension 72 with 16 heads had a slight advantage over 64 with 18 in a small comparison. The 8/3 FFN ratio was a SwiGLU convention; the later change from 3075 to 3072 was for Ascend alignment.

[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) followed the LLM-based conditioning direction established by [Qwen-Image](https://arxiv.org/abs/2508.02324). The smallest Qwen3 model suited bilingual and long prompts, while freezing it avoided additional optimizer groups, learning rates and training memory. Early exit addressed a different hypothesis: final-layer features are prepared for next-token prediction and need not be best for visual conditioning. A sweep over layers {8, 16, 20, 24, 28} selected layer 20. Layer 28 had slightly lower loss, while layer 20 gave better KID and facial-structure review; layers 8 and 16 were too early. Intermediate-feature aggregation was not explored.

The VAE is the REPA-E fine-tune of the Qwen-Image VAE, selected because the [REPA-E paper](https://arxiv.org/abs/2504.10483) reported better DiT learning even without end-to-end VAE tuning during DiT training. Inko keeps it frozen. The 2×2 latent patch size was inherited from the [original DiT](https://arxiv.org/abs/2212.09748).

Text conditions the blocks through joint attention and through a global modulation path. The latter mean-pools non-padding text features, projects them, and combines them with timestep features. Single-stream blocks share one shift, scale and gate between attention and MLP to save parameters. A matched comparison favored sharing on loss, KID and peak memory. The double-stream block was added after those experiments and kept independent modulation.

The positional layout follows Qwen-Image. Centered image coordinates were intended to reduce the largest absolute coordinates during extrapolation, while text tokens continue along a fixed diagonal beyond the image grid. An early comparison against a corner-origin image-diagonal variant tied at 320p transfer, and both failed at zero-shot scale factors of at least 1.875. Centering was not a measured winner; progressive training supplied the transfer mechanism.

[Muon](https://kellerjordan.github.io/posts/muon/) showed faster convergence than AdamW in a matched comparison. Fused QKV, modulation and gated-FFN matrices are split into functional chunks before orthogonalization, following [CMuon](https://arxiv.org/abs/2608.02502). Auxiliary parameters use AdamW by convention. The initial [Moonlight](https://arxiv.org/abs/2502.16982) scaling convention and its correction appear in the stability account.

<details><summary>Matched architecture comparisons</summary>

| Decision | Comparison | Result |
|---|---|---|
| Width and depth | Two roughly 100M candidates under matched compute | Wider and shallower won on loss and KID, establishing the direction later extrapolated to 1152 × 24 single-stream blocks |
| Head shape | 72 × 16 versus 64 × 18 | A slight advantage for 72 × 16 |
| Single-stream modulation | Shared attention/MLP modulation versus separate modulation | Loss 0.9437 versus 0.9441; KID 0.0190 versus 0.0195; 8% lower peak memory |
| Qwen exit | Layer 20 versus layer 28, within the five-layer sweep | Loss 0.92160 versus 0.92134; KID 0.00892 versus 0.00922; stronger facial-structure review for layer 20 |
| Optimizer | RMS-matched Muon versus AdamW at 16,000 updates | Loss 0.91127 versus 0.92134; KID 0.00699 versus 0.00922 |

The Muon comparison establishes the convergence advantage of the tested RMS-matched recipe. It does not by itself qualify the final original-scaling implementation.

</details>

<details><summary>Final optimization settings</summary>

| Setting | Value |
|---|---|
| Muon | LR 0.02, weight decay 0.0015, momentum 0.95, original per-chunk scaling `sqrt(max(1, rows / cols))` |
| Auxiliary AdamW | Start LR 1e-5, peak LR 1e-4, final LR 5e-6, betas 0.9/0.95, epsilon 1e-8, ordinary decay 0.01 |
| Conditioning decay | 0.4 on `txt_pooled_proj.weight`, `c_mlp.0.weight` and `c_mlp.2.weight` |
| Schedule | 20k warmup followed by cosine decay over one 600k global horizon |
| Gradient clipping | Global norm 1.0 |
| Exponential moving average (EMA) | Bias-corrected, decay 0.9999, updated every optimizer step |

</details>

## Resolution and caption-length curricula

### Resolution

The 75:20:5 allocation across 256p, 640p and 896p was fixed before the staged run. Existing DiT recipes commonly spend most updates at low resolution and very few at the largest; Inko followed that prior because 896p training remained expensive.

Timesteps start from a logit-normal distribution with mean 0 and standard deviation 1. The model uses $t=0$ for noise and $t=1$ for data, with the [SD3](https://arxiv.org/abs/2403.03206)-style resolution shift. For latent spatial dimensions $H\times W$, the image-patch token count is $N=(H/2)(W/2)$ because patches are 2×2. The shift is $s=\sqrt{N/256}$ when $N>256$, and $s=1$ otherwise:

$$t' = \frac{t}{s-(s-1)t}.$$

A square 256p image becomes a 32×32 latent grid and then 16×16 patches, giving the reference $N=256$ and a neutral shift. The corresponding patch grids are 40×40 at 640p and 56×56 at 896p, so their shifts are 2.5 and 3.5. Larger token counts therefore move sampled times toward noise under this convention.

Zero-shot positional scaling failed at factors of at least 1.875, short of the 2.5× jump to 640p, so progressive training supplied the transfer mechanism instead. In the actual 256p→640p transition, adaptation took about 20k updates with EMA decay 0.9999.

### Caption length

For retained caption lengths $n_i$ within a selected image row, the length-based sampling probability is proportional to $n_i^\beta$, with $\beta$ moving linearly from −1 to +1 over the global run. Early training therefore favors shorter texts, while later training favors detailed conditioning. This ordering follows the intuition that short prompts are easier to learn first.

For a row containing captions shorter than 256 tokens, 20% of the within-row probability mass is distributed uniformly over those short captions; the remaining 80% follows the power law, including any short captions within that part. Rows without short captions use the power law alone. The reserve maintains short-prompt exposure as $\beta$ rises.

Long captions were still a minority after the enrichment described in the [data chapter](data.md#the-caption-strategy), so each conditioned example also receives a loss weight based on retained caption length $n$:

$$w(n)=\max\left(1,\log_2\frac{n}{128}\right).$$

Captions up to 256 tokens receive weight 1; weights are 2 at 512, 3 at 1024 and 4 at the 2,048-token cap. The update uses a weighted mean, $\sum_i w_i\ell_i/\sum_i w_i$, changing where gradient mass goes without rescaling the whole update. Comparisons of curves and reference thresholds produced no clear winner, so the exact weighting remains a heuristic choice.

Separately from the length curriculum, caption dropout removes text for 10% of examples, supplying unconditional examples for classifier-free guidance.

The long-caption emphasis had a cost. Despite the 20% reserve, generation review near the end of pretraining showed weaker responses to short prompts. Recovering that ability (at least partially) became one of the objectives of the specialized post-training run.

## Packing and infrastructure optimization

### From a long-caption OOM to dynamic micro-batching

The initial infrastructure problem was a long-caption out-of-memory failure. Reducing every micro-batch to fit the worst caption would have wasted the rest of the workload, while a fixed micro-batch per aspect ratio would still ignore text length. The response was dynamic micro-batching by aspect and caption length, called packing here. Its goals were stable memory, limited padding waste and high measured throughput. Precompute stored image latents, caption candidates and length metadata, while tokenization and the frozen Qwen forward ran online, so the planner could account for each selected caption's actual length.

Each of the five aspect buckets receives 20 caption-length buckets with a 2,048-token cap. The planner chooses their upper bounds by minimizing expected padded compute. For image-token count $N$ and caption length $n$, $C(n)$ estimates the per-sample compute cost:

$$C(n)=\alpha(N+n)+\gamma(N+n)^2.$$

The linear term represents projections and MLPs, while the quadratic term represents attention. The coefficients are architecture-dependent, and only their ratio affects the optimum. A row padded to a bucket's upper bound $h$ is charged $C(h)$, so its padding waste is $C(h)-C(n)$. For a bucket with lower bound $a$ and upper bound $h$, expected waste is $\sum_{n\in(a,h]}p(n)(C(h)-C(n))$, where $p(n)$ is the weighted caption-length distribution for the stage and curriculum window. Dynamic programming finds the boundaries that minimize total waste.

Micro-batch sizes come from measured forward/backward calibration on the target cards. With padded sequence length $L=N+h$ and micro-batch size $B$, the fitted models are

$$M(B,L)=m_0+B\,m_1L,\qquad T(B,L)=t_0+B(t_1L+t_2L^2).$$

$M$ estimates peak allocated memory and $T$ estimates forward/backward time. A quadratic memory term is added when the linear fit has $R^2<0.95$; fitted memory residuals were at most 0.60 GiB. For 640p and 896p, the planner targeted 52 GiB of allocated memory per rank on 60.96 GiB devices. Calibration included the FP32 model, EMA, frozen BF16 text encoder, initialized optimizer states, gradients and a model-sized distributed-training buffer reserve. Within that resident state, the planner selects the largest permitted $B$ whose predicted peak fits. Each fixed later-resolution plan averages caption probabilities over its curriculum window: progress 0.75–0.95 at 640p and 0.95–1.0 at 896p.

Captions are padded only to their local bound, so the per-rank micro-batch shrinks as that bound grows. Gradient accumulation then increases samples per update without exceeding the micro-batch memory budget: accumulation 1 at 256p, 3 at 640p and 4 at 896p. The selected micro-batch ranges are 8–72, 5–12 and 3–6 per rank.

For a telemetry window with mean per-rank micro-batch $\bar B$ and stage accumulation $A$, the estimated global batch per optimizer update is

$$\widehat{B}_{\mathrm{global}}=16A\bar B.$$

The estimate follows realized bucket occupancy as the caption curriculum and source pools change.

![Bucket plans and realized global batch. Top: five aspect curves per resolution, each with 20 caption-length buckets; the y-axes differ by resolution. Bottom: estimated global samples per optimizer update from 25-update telemetry windows, with a 500-update median. Faint vertical dashed lines mark 56k, 250k, 400k and 480k; thicker dashed lines mark the resolution transitions at 450k and 570k.](assets/pretrain_batch_plan.png)

Median estimates across the staged run were **898**, **514** and **358** global samples per update at 256p, 640p and 896p, respectively. The downward trend within each stage reflects the movement toward longer captions as well as the changing data pools.

A forced-shape check exercised all 280 distinct aspect, length and batch combinations across the later-resolution candidates. Peak allocations were 51.57 and 52.03 GiB. The production trace below adds the time dimension: stage maxima were 49.68, 51.59 and 52.05 GiB. The small excursions above 52 GiB are consistent with a fitted planning target whose calibration residuals reached 0.60 GiB, while actual use remained below device capacity.

![Peak allocated memory on the logging rank over the 600k schedule. Lines are 500-update medians and ribbons preserve the full minimum–maximum range within each bin. Stage maxima are 49.68, 51.59 and 52.05 GiB. The dotted horizontal line marks the 52 GiB later-stage budget; the dashed line marks 60.96 GiB device capacity. Faint vertical dashed lines mark 56k, 250k, 400k and 480k; thicker dashed lines mark the resolution transitions at 450k and 570k.](assets/pretrain_memory.png)

### From CUDA exploration to Ascend execution

Packing controlled the workload's shape; execution work made that workload faster. The first pass on an RTX 4090 screened autotuning and recurring overheads. Autotuning reduced warm update time by **1.73%**, **4.41%** and **3.50%** at 256p, 640p and 896p. Together with the other selected changes, the CUDA path increased steady single-GPU throughput from **54.99 to 72.98 samples/s**, a 32.7% gain.

The screen rejected as well as selected. BF16 communication reduced the measured communication interval from roughly 141 to 80 ms, but added 2.33 GB of memory. More importantly, simulated Muon updates differed by 6.52% on two ranks and 8.58% on eight ranks despite gradient differences of only 0.23% and 0.317%. Orthogonalization amplified the reduction error, so the hook was rejected.

The CUDA screen informed the Ascend candidates, which then received fresh matched measurements on 16×910B2C cards. Long-caption memory depended first on the attention implementation. Generic float-mask scaled dot-product attention retained a quadratic attention-sized tensor with approximately 2.81 GiB of extra saved memory in the measured case. Native Ascend attention reduced that overhead to 0.06 GiB with a boolean drop mask.

The early-256p comparisons used the selected variable-aspect and caption workload over updates 100–450, excluding startup, profiling and periodic evaluation or checkpoint work. Native fused RMSNorm retained FP32 arithmetic and learned gains. The memory freed by the fused attention and norm kernels then made room for larger micro-batches, and aligning the FFN width to 3072 gave Ascend kernels more regular matrix shapes. Native SwiGLU replaced separate chunk, activation and multiplication operations with one call. The figure labels each incremental gain.

![CUDA and Ascend infrastructure measurements. Top left: warm baseline and autotuned seconds per update on one RTX 4090, with time reductions of 1.73%, 4.41% and 3.50%. Top right: cumulative early-256p throughput on 16 Ascend 910B2C cards, rising from 656.29 to 878.91 samples per second with incremental gains labeled. Bottom: EMA of per-update global throughput for the configured training runs, with a 500-update median and stage medians of approximately 787, 161 and 78 samples per second. Faint vertical dashed lines mark 56k, 250k, 400k and 480k; thicker dashed lines mark the resolution transitions at 450k and 570k.](assets/pretrain_throughput.png)

The cumulative Ascend changes gave **33.92% more throughput** on that early-256p comparison. The final SwiGLU comparison reduced peak allocation from 52.591 to 49.678 GiB. In the full curriculum, throughput followed resolution, caption length and pool composition, with stage medians of approximately 787, 161 and 78 samples/s.

### Choosing the later-resolution plans

At 640p and 896p, memory-sized batches competed with time-aligned batches. Time alignment reduces sizes in faster buckets to keep predicted micro-batch times under a shared target, since distributed updates wait for the slowest rank. Controlled probes used the same mature 180k model, fresh optimizers and fixed caption progress for both candidates.

| Stage | Time-aligned samples/s | Memory-sized samples/s | Memory-sized samples/update | Memory-sized seconds/update |
|---|---:|---:|---:|---:|
| 640p, accumulation 3 | 169.78 | **173.47** | 531.66 | 3.0648 |
| 896p, accumulation 4 | 83.30 | **83.90** | 375.43 | 4.4747 |

Time alignment reduced throughput by 2.13% at 640p and 0.72% at 896p, so the memory-sized plans remained selected. These short probes isolated the execution paths on a fixed reference pool; the full-run panel above supplies the production throughput record.

A three-update trace of the selected early-256p path attributed 84.56% of wall time to compute, 7.22% to exposed communication and 8.22% to free gaps. Further substantial gains would therefore need cheaper compute, not just another round of overhead removal.

## Training stability

### Reproducing and isolating loss spikes

Before the staged schedule, an Ascend run using the earlier recipe developed loss spikes. The investigation first examined platform numerics, including the Newton–Schulz iteration inside Muon. A separate H200 reproduction of the recipe then showed a corresponding loss and gradient event. With both platforms reaching their largest local gradient at update 12,705, the attention shifted from Ascend-specific numerics to the architecture and recipe.

![Raw training loss and pre-clip gradient norm around the step-12,705 event in the original Ascend run and its H200 reproduction. Both gradient series peak at update 12,705. The gradient panel uses a logarithmic y-axis.](assets/pretrain_spike_12705.png)

The investigation then used two additional captured H200 states for controlled replay. They came from separate spike events, not from update 12,705. At step 12,829, replaying the same batch and state in FP32 produced gradient norms 41.709637 on H200 and 41.709521 on RTX 4090. This cross-hardware agreement pointed to model-side dynamics rather than a card-specific or BF16-only failure.

At the captured step-12,172 event, an exact reconstruction of the preceding update reproduced the stored state bit-for-bit. A matrix-by-matrix inspection of that update singled out the final conditioning matrix, `c_mlp.2.weight`, as the anomaly. Its update was decomposed into a displacement of the mean hidden feature, which acts as a shared conditioning offset, and the remaining variation across features. Replaying the same following batch with selected parts of the preceding update isolated their effects:

| Intervention on the preceding update at step 12,172 | Gradient norm on the following batch |
|---|---:|
| All parameter updates | 45.0366 |
| `c_mlp.2.weight` update alone | 55.8181 |
| That matrix update without its shared mean-feature displacement | 0.247787 |
| Bias update alone | 0.2921 |

The matrix-induced shared displacement therefore supplied a causal path to this captured spike. Removing it brought the gradient close to the bias-only arm, while the remaining feature variation did not reproduce the event.

### Changes that followed the investigation

Branch-output RMSNorm was introduced in response to the spikes. It normalizes each attention or MLP branch before the residual addition, limiting raw branch amplitude rather than allowing a conditioning offset to pass directly into the residual stream. Together with the existing pre-branch LayerNorm, it forms a sandwich around each branch:

$$x \leftarrow x + \mathrm{gate}\odot\mathrm{RMSNorm}(\mathrm{branch}(x)).$$

A fixed diagnostic panel repeatedly measured internal responses on the same short-caption rows at three shifted timesteps. Using that panel and a matched 2,000-update comparison with effective batching fixed gave:

| Measurement | Control | Branch RMSNorm |
|---|---:|---:|
| Evaluation loss | 0.92154 | 0.91259 |
| Block-24 residual-stream RMS | 12,001 | 73.58 |

Muon had originally been selected after a 16,000-update comparison under the earlier RMS-matched recipe. It reached loss 0.91127 and KID 0.00699, compared with 0.92134 and 0.00922 for AdamW. The spike investigation then exposed a scaling mismatch in the implementation. The previous RMS-matched multiplier from [Moonlight](https://arxiv.org/abs/2502.16982) was `0.2 * sqrt(max(rows, cols))`; the corrected original convention is `sqrt(max(1, rows / cols))`. The configured LR remained 0.02, while the correction reduced the effective update magnitude. For square 1152 × 1152 matrices the old multiplier was 6.79 times the new one. For the wide 1152 × 3072 down-projection, the ratio was 11.09×.

The restart also multiplied $t\in[0,1]$ by 1000 before the sinusoidal timestep embedding. This addressed a separate hypothesis: adjacent times in $[0,1]$ might leave too few embedding channels responsive. The captured spikes did not establish that mechanism, but the change was reasonable and had no observed training cost. Later conditioning telemetry separately showed a healthier fraction of SiLU preactivations below −5.

The subsequent 600k schedule started from scratch with branch-output RMSNorm, the factor-1000 timestep input and the corrected Muon scaling.

### Scale growth at 46k and the 56k intervention

The restarted run showed a different problem around 46k: conditioning, gate and residual RMS were growing approximately exponentially. Even though loss still looked normal, this was concerning. Because conditioning modulates every block, a change in that path could propagate to the other two measurements, the first step was therefore to locate the change within the conditioning path.

A matrix-versus-bias swap provided that separation. Moving the 46k conditioning matrices into the 44k state while keeping the 44k biases reproduced 99.87% of the observed conditioning RMS growth; changing only the biases reproduced 0.12%. The matrices were carrying the change, but not through a simple increase in magnitude: their RMS remained nearly flat while their directional responses changed. Weight norms alone therefore could not describe the developing problem.

The next question was whether constraining the conditioning pathway could reverse the downstream growth. Four matched continuations from a preserved 48k checkpoint compared the existing recipe, conditioning weight decay 0.2, decay 0.5 and normalization of the conditioning output to a fixed RMS. Weight decay constrained the matrices implicated by the swap; output normalization constrained the vector they produced. All four arms used the same random draws and sample identities on 16 Ascend ranks for 1,000 updates. The table reports log-linear slopes over the last 500 updates, in percent per 1,000 updates. For gates, residuals and pre-branch activations, the maximum is taken across the fixed panel's measured block traces before fitting the slope.

| Arm | Conditioning | Gate maximum | Residual maximum | Pre-branch maximum |
|---|---:|---:|---:|---:|
| Control | +6.73 | +7.45 | +7.04 | +12.20 |
| Decay 0.2 | +3.20 | +4.42 | +4.68 | +7.20 |
| Output normalization | 0 | −3.56 | −4.40 | −6.73 |
| Decay 0.5 | −2.55 | −0.49 | −0.35 | +1.17 |

Decay 0.2 slowed the growth but left all four late slopes positive. Decay 0.5 reversed conditioning, gate and residual growth, although some pre-normalization traces still grew. Output normalization also reversed the downstream slopes, but fixing conditioner RMS introduced transients of roughly 12% in gates and 33% in pre-branch activations. An intermediate weight decay of 0.4 was finally selected to control the growth without suppressing the conditioning response too aggressively.

The stronger decay applied only to `txt_pooled_proj.weight`, `c_mlp.0.weight` and `c_mlp.2.weight`; other AdamW parameters kept ordinary decay. A 500-update continuation from 56k on 16 Ascend ranks then measured the selected setting. Conditioning, gate and residual RMS fell 1.385%, 1.301% and 2.314%, with no clipped update and a maximum gradient of 0.393383. The three quantities that had grown together around 46k were now decreasing together, and production resumed from 56k with decay 0.4.

![Conditioning, gate and residual RMS around the 56k intervention. The curve uses original telemetry through 56k and the corrected continuation afterward, omitting overlapping old-recipe points. The dashed line marks the introduction of conditioning weight decay 0.4.](assets/pretrain_scale_mitigation_56k.png)

### The full-run record

The 600k schedule completed with **zero skipped optimizer updates**. Across the exported record, 994 updates (0.166%) exceeded the global-norm clipping threshold of 1.0.

| Stage | Paired log points | Updates above clip threshold | Maximum pre-clip gradient | Maximum logged loss after update 1,000 |
|---|---:|---:|---:|---:|
| 256p | 450,000 | 591 (0.131%) | 7.873 | 0.978 |
| 640p | 118,996 | 194 (0.163%) | 4.093 | 0.886 |
| 896p | 29,877 | 209 (0.700%) | 6.588 | 0.717 |
| **Full schedule** | **598,873** | **994 (0.166%)** | **7.873** | **0.978** |

The loss, gradient, memory and throughput exports share 598,873 update indices. All 1,127 missing indices lie between updates 543,173 and 599,281. Timestamps on either side of each gap account for the missing updates at the surrounding training cadence, pointing to a SwanLab export problem rather than additional interruptions.

![Training loss and pre-clip gradient norm over the 600k schedule. Lines are 500-update medians and ribbons preserve the full minimum–maximum range within each bin. Faint vertical dashed lines mark 56k, 250k, 400k and 480k; thicker dashed lines mark the resolution transitions at 450k and 570k.](assets/pretrain_full_run_stability.png)

At step 599,576, the 896p gradient reached its maximum of 6.588 while the loss at that update was 0.587. The run completed the remaining 424 updates.

## Checkpointing and recovery

Complete checkpoints were saved every 2,000 updates, retaining the latest two plus stage endpoints. Each checkpoint contained model and EMA weights, optimizer and scheduler states, rank-local random states and sampler state.

A same-stage resume restored these states together. Sampler recovery included each rank's row cycles, cursors and queued batches, so row and caption sampling continued from the saved position. Resolution transitions instead constructed a fresh sampler for the new stage. A changed recipe or model configuration required explicit migration rather than an ordinary resume.

The **600k EMA checkpoint** became the base model for [post-training](posttraining.md). Its generations still missed counts, positions and layouts, sometimes collapsed human anatomy, and often broke subjects when an artistic style was combined with an unfamiliar content category. Those observed weaknesses set the next stage's objectives.
