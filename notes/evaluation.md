# Evaluation

## Benchmark results

Inko was evaluated alongside seven external models on GenEval, DPG-Bench, OneIG-Bench, TIIF testmini and Qwen-Image-Bench. Inko base, the 600k pretrain EMA, is included as a reference. Every entry generated the full catalogs, totaling 12,506 images per model and 112,554 images across the nine entries. External models retained their native inference recipes. The two Inko entries share their sampling settings except for CFG: 4.5 for Inko and 5.0 for Inko base. The protocol below gives the complete settings.

![Full-catalog results on GenEval, DPG, OneIG alignment, TIIF long and Qwen-Image. The seven references are ranked within each panel. Inko base and Inko are separated on the right, and a dashed line marks Inko's score across the references.](assets/benchmark_overview.png)

| Model | GenEval ↑ | DPG ↑ | OneIG align ↑ | OneIG div ↑ | OneIG reas ↑ | OneIG style ↑ | OneIG text ↑ | TIIF long ↑ | TIIF short ↑ | Qwen ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DreamLite-Base | 0.715 | 85.28 | 0.823 | 0.156 | 0.211 | 0.385 | 0.066 | 0.642 | 0.640 | 40.41 |
| SANA-0.6B * | 0.675 | 84.27 | 0.800 | 0.220 | 0.210 | 0.362 | 0.043 | 0.606 | 0.632 | 42.03 |
| PixArt-Σ | 0.556 | 80.47 | 0.764 | 0.239 | 0.217 | 0.375 | 0.002 | 0.579 | 0.585 | 41.40 |
| SD 1.5 * | 0.440 | 63.09 | 0.583 | 0.425 | 0.212 | 0.390 | 0.010 | 0.374 | 0.435 | 29.62 |
| SDXS * | 0.453 | 69.14 | 0.647 | 0.140 | 0.183 | 0.270 | 0.001 | 0.390 | 0.428 | 29.21 |
| SD3.5 Medium | 0.677 | 84.07 | 0.800 | 0.241 | 0.279 | 0.360 | 0.433 | 0.667 | 0.669 | 43.09 |
| MOSS-Image-SFT | 0.760 | 87.15 | 0.879 | 0.225 | 0.299 | 0.390 | 0.903 | 0.799 | 0.802 | 54.56 |
| Inko base | 0.494 | 82.58 | 0.789 | 0.298 | 0.172 | 0.359 | 0.003 | 0.612 | 0.604 | 38.94 |
| Inko | 0.577 | 83.24 | 0.810 | 0.245 | 0.182 | 0.351 | 0.002 | 0.630 | 0.631 | 42.02 |

\* Every model has 1,000 Qwen images and a judgment row for each image. Three dimension scores average 999 prompts because one expected dimension is missing: SANA Quality (ID 44), SD 1.5 Creative Generation (ID 417) and SDXS Creative Generation (ID 920). Native aggregation retains every available dimension.

Against PixArt-Σ, Inko scores higher on GenEval, DPG, OneIG alignment and diversity, both TIIF description lengths and Qwen-Image-Bench; PixArt leads on OneIG reasoning and style. Inko is close to SANA-0.6B on DPG, both TIIF lengths and Qwen, while SANA has a clear GenEval advantage. DreamLite-Base leads most compact-model readouts.

Inko is also close to SD3.5 Medium on DPG (83.24 versus 84.07), OneIG alignment (0.810 versus 0.800) and Qwen (42.02 versus 43.09). SD3.5 is ahead of Inko on GenEval, both TIIF lengths, reasoning and text rendering; MOSS-Image-SFT leads every metric in the five-panel chart. Backbone budgets and inference settings appear with the protocol below.

## Inside the benchmark scores

Native category and channel scores locate these comparative results. The matched panels show the first saved draw for selected prompts, using the same model order as the headline table.

### GenEval

[GenEval](https://arxiv.org/abs/2310.11513) measures object-focused prompt alignment with detector-based checks for single objects, two objects, counting, colors, position and color attribution. The overall score averages these six category scores, with four generated draws per prompt.

Inko's clearest strengths are position and color attribution. Only DreamLite and MOSS score higher on position. Inko ranks fourth on color attribution at 0.542, behind MOSS at 0.583, SD3.5 at 0.573 and DreamLite at 0.547. Its two-object score of 0.677 edges PixArt at 0.634 but remains below DreamLite, SANA, SD3.5 and MOSS. The weaker cells are substantial: counting is 0.350, below every external entry except SD 1.5 at 0.347, and the single-object score of 0.866 is below every external entry.

| Category | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single object | 0.984 | 0.991 | 0.994 | 0.988 | 0.981 | 0.997 | 0.978 | 0.853 | 0.866 |
| Two objects | 0.894 | 0.806 | 0.634 | 0.399 | 0.455 | 0.874 | 0.881 | 0.545 | 0.677 |
| Counting | 0.634 | 0.691 | 0.478 | 0.347 | 0.388 | 0.600 | 0.722 | 0.278 | 0.350 |
| Colors | 0.886 | 0.886 | 0.827 | 0.779 | 0.771 | 0.801 | 0.838 | 0.681 | 0.707 |
| Position | 0.345 | 0.228 | 0.155 | 0.045 | 0.048 | 0.217 | 0.560 | 0.207 | 0.323 |
| Color attribution | 0.547 | 0.448 | 0.247 | 0.085 | 0.075 | 0.573 | 0.583 | 0.398 | 0.542 |

![Nine models on the GenEval black-car and green-parking-meter case. Inko base is in the bottom center and Inko at bottom right.](assets/evaluation_geneval_car_meter.png)

Inko separates the black car from a green parking meter and retains the meter's display, but adds a second bright-green bollard-like object in the foreground. MOSS produces a cleaner arrangement with one clearly recognizable meter. SANA and SDXS also preserve the requested color binding; SD 1.5 instead makes a second car green, and SD3.5 replaces the meter with a sign.

<details><summary>Exact prompt for the black-car and green-parking-meter case</summary>

> a photo of a black car and a green parking meter

</details>

### DPG-Bench

[DPG-Bench](https://arxiv.org/abs/2403.05135), the Dense Prompt Graph Benchmark introduced with ELLA, evaluates dense prompt following by representing prompts as graphs of entities, attributes and relations and asking judge questions over them. Its overall score accounts for dependencies among those questions. The category tables below use the native unfiltered answers from the fourth of four draws, providing a separate category view.

Inko scores 83.24 overall, close to SD3.5 at 84.07 and above PixArt at 80.47. Its entity, attribute and relation scores are all close to SD3.5. Level-2 size leads all nine entries at 76.45, and color is highest among the external entries at 94.38. Inko's count score is 74.00, trailing DreamLite, SANA, SD3.5 and MOSS; its text score is 76.00, trailing DreamLite, SANA, PixArt, SD3.5 and MOSS.

| Level 1 | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Entity | 91.16 | 89.89 | 87.37 | 71.82 | 77.37 | 89.85 | 92.61 | 89.40 | 90.45 |
| Attribute | 89.44 | 88.31 | 86.29 | 71.89 | 76.73 | 88.39 | 90.24 | 89.88 | 90.08 |
| Other | 81.60 | 79.20 | 71.60 | 54.80 | 56.40 | 81.60 | 85.60 | 76.40 | 74.40 |
| Relation | 94.02 | 93.17 | 91.25 | 79.87 | 84.69 | 93.60 | 94.52 | 93.37 | 93.56 |
| Global | 86.02 | 86.63 | 86.32 | 82.67 | 80.55 | 83.89 | 85.11 | 83.89 | 86.02 |

<details><summary>Level-2 category scores</summary>

| Level 2 | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Entity: whole | 93.38 | 91.61 | 89.26 | 72.71 | 79.50 | 91.94 | 94.22 | 90.76 | 91.77 |
| Entity: part | 85.94 | 84.96 | 80.86 | 73.44 | 72.46 | 83.59 | 86.91 | 86.91 | 86.72 |
| Entity: state | 82.89 | 83.95 | 81.41 | 66.53 | 69.38 | 82.79 | 87.65 | 83.95 | 85.85 |
| Attribute: color | 92.77 | 91.62 | 89.71 | 69.49 | 76.42 | 92.77 | 93.23 | 94.38 | 94.38 |
| Attribute: shape | 79.04 | 81.66 | 79.91 | 73.80 | 73.36 | 78.60 | 85.59 | 80.35 | 81.22 |
| Attribute: texture | 90.08 | 88.99 | 86.51 | 74.29 | 78.95 | 88.14 | 90.32 | 90.08 | 90.14 |
| Attribute: size | 73.97 | 70.66 | 70.25 | 60.33 | 65.70 | 69.01 | 75.21 | 74.38 | 76.45 |
| Attribute: other | 87.71 | 86.16 | 84.27 | 75.42 | 77.19 | 86.82 | 88.70 | 86.16 | 86.38 |
| Other: count | 80.50 | 78.00 | 70.00 | 53.00 | 54.50 | 77.50 | 83.00 | 78.00 | 74.00 |
| Other: text | 86.00 | 84.00 | 78.00 | 62.00 | 64.00 | 98.00 | 96.00 | 70.00 | 76.00 |
| Relation: spatial | 94.41 | 93.47 | 91.66 | 80.36 | 85.05 | 93.76 | 94.86 | 93.63 | 93.80 |
| Relation: non-spatial | 88.05 | 88.68 | 84.91 | 72.33 | 79.25 | 91.19 | 89.31 | 89.31 | 89.94 |
| Global | 86.02 | 86.63 | 86.32 | 82.67 | 80.55 | 83.89 | 85.11 | 83.89 | 86.02 |

</details>

![Nine models on the dense DPG prompt describing an orange tent beside a lake and a gold bracelet on the forest floor.](assets/evaluation_dpg_tent_bracelet.png)

Inko keeps the large orange tent, zipped entrance, grassy knoll, lake and gold bracelet in one scene. The bracelet remains a separate, visible object on the foreground leaves rather than disappearing into the dense setting. MOSS and SD3.5 also preserve that small-object relationship; DreamLite makes the bracelet much less prominent, while PixArt omits visible jewelry.

![Nine models on the DPG prompt describing a skier in winter clothing descending a large sand dune.](assets/evaluation_dpg_sand_skier.png)

Inko retains the desert, insulated jacket, bright-orange snow pants and two skis. SD 1.5, by contrast, turns the ground into a pale snow-like surface. Inko's upright stance and splayed skis convey less downhill motion than SANA, PixArt and MOSS, whose skiers lean into the slope, and the requested carved trails are not clear.

<details><summary>Exact prompts for the two DPG cases</summary>

**Tent, lake and bracelet**

> In a tranquil forest clearing by the water's edge, a large orange tent with its entrance zipped shut is pitched on a grassy knoll overlooking a serene lake. A few steps away from the tent, on the forest floor scattered with autumn leaves, a circular gold bracelet catches the sunlight, creating a subtle sparkle amid the natural surroundings. The nearby trees cast gentle shadows over the area, enhancing the peaceful ambiance of the outdoor scene.

**Sand-dune skiing**

> A man dressed in a thick, insulated jacket and bright orange snow pants is expertly skiing down a large sand dune. The dune's golden sands contrast with the clear blue sky above. Despite the unusual setting, the man's skis carve out smooth trails, leaving a unique pattern behind him as he descends amidst the vast desert surroundings.

</details>

### OneIG-Bench

[OneIG-Bench](https://arxiv.org/abs/2506.07977) evaluates image generation across alignment, reasoning, style, text and diversity. These channels use different released prompt subsets: alignment covers anime, portrait and general-object prompts, while reasoning, text and style have their own subsets. Diversity measures pairwise DreamSim variation among four draws of a prompt, so a higher score means more output variation.

Inko's alignment score of 0.810 exceeds SANA (0.800), SD3.5 (0.800) and PixArt (0.764), while DreamLite (0.823) and MOSS (0.879) remain ahead. Portraits are particularly close between Inko and DreamLite, 0.830 to 0.831. General objects are less strong at 0.739: Inko edges SD3.5 and PixArt but trails SANA, DreamLite and MOSS.

| Alignment category | Prompts | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Anime stylization | 245 | 0.855 | 0.836 | 0.799 | 0.582 | 0.701 | 0.840 | 0.895 | 0.833 | 0.850 |
| Portrait | 244 | 0.831 | 0.797 | 0.767 | 0.613 | 0.621 | 0.812 | 0.887 | 0.811 | 0.830 |
| General object | 206 | 0.777 | 0.760 | 0.718 | 0.550 | 0.615 | 0.736 | 0.850 | 0.709 | 0.739 |

Reasoning is a clearer limitation. Inko scores 0.182, below every external entry; SDXS is next at 0.183, and the compact DreamLite, SANA and PixArt entries are all near 0.21. Inko trails DreamLite, SANA and PixArt in every knowledge category, and SD3.5 and MOSS score higher throughout as well.

| Reasoning category | Prompts | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Geography | 26 | 0.222 | 0.209 | 0.227 | 0.216 | 0.196 | 0.283 | 0.303 | 0.185 | 0.190 |
| Computer science | 48 | 0.188 | 0.194 | 0.197 | 0.203 | 0.159 | 0.283 | 0.309 | 0.149 | 0.160 |
| Common sense | 12 | 0.280 | 0.280 | 0.287 | 0.251 | 0.241 | 0.307 | 0.321 | 0.225 | 0.243 |
| Biology | 40 | 0.210 | 0.222 | 0.230 | 0.209 | 0.180 | 0.272 | 0.286 | 0.167 | 0.179 |
| Mathematics | 16 | 0.219 | 0.209 | 0.209 | 0.223 | 0.180 | 0.288 | 0.303 | 0.187 | 0.196 |
| Physics | 58 | 0.217 | 0.206 | 0.217 | 0.209 | 0.193 | 0.271 | 0.294 | 0.180 | 0.192 |
| Chemistry | 25 | 0.193 | 0.197 | 0.199 | 0.206 | 0.168 | 0.273 | 0.290 | 0.153 | 0.158 |

SD 1.5 leads the traditional-style family at 0.438. Inko scores 0.341, close to SANA at 0.342 and above SD3.5 at 0.321 and SDXS at 0.247, but behind DreamLite at 0.364, PixArt at 0.356 and MOSS at 0.367. Its media-family score of 0.396 is the highest in the table. Each family mean weights its styles equally; the fine-art comparisons below examine eleven individual cases.

| Style family | Styles | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Traditional | 13 | 0.364 | 0.342 | 0.356 | 0.438 | 0.247 | 0.321 | 0.367 | 0.359 | 0.341 |
| Media | 8 | 0.357 | 0.294 | 0.266 | 0.335 | 0.271 | 0.331 | 0.374 | 0.371 | 0.396 |
| Anime | 8 | 0.457 | 0.464 | 0.515 | 0.366 | 0.307 | 0.464 | 0.452 | 0.347 | 0.326 |

Only SD 1.5 has higher diversity among the external entries, 0.425 to Inko's 0.245, and its alignment is much lower at 0.583. Inko also edges PixArt (0.239) and SD3.5 (0.241) on diversity while scoring higher on alignment. The advantage is not uniform across content types: knowledge reasoning and text prompts produce substantial variation, whereas portrait diversity trails SANA, PixArt, SD3.5 and MOSS.

| Diversity category | Prompts | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Anime stylization | 245 | 0.087 | 0.141 | 0.144 | 0.353 | 0.091 | 0.162 | 0.166 | 0.205 | 0.143 |
| Portrait | 244 | 0.098 | 0.172 | 0.191 | 0.386 | 0.118 | 0.212 | 0.213 | 0.235 | 0.163 |
| General object | 206 | 0.119 | 0.181 | 0.205 | 0.405 | 0.137 | 0.225 | 0.220 | 0.266 | 0.210 |
| Knowledge reasoning | 225 | 0.279 | 0.339 | 0.357 | 0.535 | 0.184 | 0.351 | 0.334 | 0.437 | 0.400 |
| Text rendering | 200 | 0.207 | 0.284 | 0.319 | 0.459 | 0.179 | 0.268 | 0.196 | 0.366 | 0.331 |

For text, ED is mean Levenshtein distance, CR is the exact-match rate and WAC is total matched characters divided by total ground-truth characters. Inko produces no exact matches across the 200 prompts and has a WAC of 0.0021. PixArt and SDXS also have no exact matches, while DreamLite reaches 0.030, SANA 0.015, SD3.5 0.110 and MOSS 0.309. Inko's ED of 113.976 is worse than every external entry, making text generation a consistent limitation across both character-level measurements.

| Text measurement | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED ↓ | 101.305 | 104.854 | 110.343 | 112.147 | 111.192 | 89.657 | 52.580 | 115.775 | 113.976 |
| CR ↑ | 0.030 | 0.015 | 0.000 | 0.005 | 0.000 | 0.110 | 0.309 | 0.000 | 0.000 |
| WAC ↑ | 0.0372 | 0.0282 | 0.0022 | 0.0054 | 0.0007 | 0.2899 | 0.7320 | 0.0032 | 0.0021 |

<details><summary>Prompt-length breakdowns</summary>

| Channel | Length | Prompts | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Alignment | Short | 165 | 0.845 | 0.818 | 0.780 | 0.636 | 0.687 | 0.804 | 0.892 | 0.775 | 0.804 |
| Alignment | Middle | 323 | 0.823 | 0.800 | 0.769 | 0.581 | 0.650 | 0.807 | 0.879 | 0.805 | 0.816 |
| Alignment | Long | 207 | 0.807 | 0.785 | 0.742 | 0.544 | 0.611 | 0.785 | 0.868 | 0.774 | 0.804 |
| Diversity | Short | 436 | 0.203 | 0.269 | 0.286 | 0.466 | 0.155 | 0.286 | 0.279 | 0.369 | 0.320 |
| Diversity | Middle | 422 | 0.122 | 0.183 | 0.205 | 0.396 | 0.124 | 0.207 | 0.195 | 0.254 | 0.200 |
| Diversity | Long | 262 | 0.131 | 0.199 | 0.217 | 0.405 | 0.141 | 0.222 | 0.183 | 0.252 | 0.193 |
| Reasoning | Short | 216 | 0.209 | 0.209 | 0.217 | 0.210 | 0.182 | 0.277 | 0.298 | 0.171 | 0.181 |
| Reasoning | Middle | 4 | 0.261 | 0.229 | 0.242 | 0.226 | 0.207 | 0.325 | 0.335 | 0.215 | 0.219 |
| Reasoning | Long | 5 | 0.245 | 0.242 | 0.225 | 0.262 | 0.182 | 0.325 | 0.308 | 0.152 | 0.160 |
| Style | Short | 15 | 0.407 | 0.439 | 0.502 | 0.502 | 0.311 | 0.414 | 0.500 | 0.426 | 0.444 |
| Style | Middle | 61 | 0.376 | 0.341 | 0.353 | 0.415 | 0.254 | 0.352 | 0.379 | 0.370 | 0.364 |
| Style | Long | 38 | 0.390 | 0.365 | 0.361 | 0.306 | 0.278 | 0.351 | 0.366 | 0.316 | 0.294 |

The text channel has separate length breakdowns for all three measurements.

| Text measurement | Length | Prompts | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ED ↓ | Short | 55 | 24.427 | 29.723 | 34.377 | 36.277 | 30.809 | 18.264 | 23.982 | 37.741 | 34.850 |
| ED ↓ | Middle | 95 | 52.395 | 56.795 | 60.516 | 61.521 | 60.229 | 42.461 | 40.874 | 67.692 | 65.671 |
| ED ↓ | Long | 50 | 278.800 | 278.810 | 288.575 | 291.795 | 296.445 | 257.865 | 106.280 | 292.970 | 292.795 |
| CR ↑ | Short | 55 | 0.086 | 0.050 | 0.000 | 0.000 | 0.000 | 0.241 | 0.391 | 0.000 | 0.000 |
| CR ↑ | Middle | 95 | 0.013 | 0.003 | 0.000 | 0.011 | 0.000 | 0.092 | 0.403 | 0.000 | 0.000 |
| CR ↑ | Long | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.040 | 0.000 | 0.000 |
| WAC ↑ | Short | 55 | 0.2018 | 0.1510 | 0.0000 | 0.0102 | 0.0026 | 0.5324 | 0.5794 | 0.0010 | 0.0015 |
| WAC ↑ | Middle | 95 | 0.1039 | 0.0618 | 0.0060 | 0.0164 | 0.0022 | 0.5278 | 0.8234 | 0.0019 | 0.0019 |
| WAC ↑ | Long | 50 | 0.0212 | 0.0173 | 0.0022 | 0.0056 | 0.0001 | 0.3224 | 0.8286 | 0.0037 | 0.0037 |

</details>

### TIIF testmini

[TIIF-Bench](https://arxiv.org/abs/2506.02161) evaluates fine-grained instruction following with attribute-specific yes/no questions. The testmini split used here contains 277 scenes, each with short and long descriptions, and averages nine attribute groups separately for each description length. Text scenes additionally use Global Normalized Edit Distance (GNED) and OCR recall.

Inko's short score of 0.631 is nearly identical to SANA at 0.632, with DreamLite at 0.640 and SD3.5 at 0.669 slightly higher. Its long score of 0.630 is above SANA at 0.606 and PixArt at 0.579 but below DreamLite at 0.642 and SD3.5 at 0.667. MOSS leads both lengths at about 0.80.

Short basic attributes are a relative strength at 0.785, second to MOSS among the nine entries. Long basic reasoning reaches 0.794 and long numeracy 0.792, with the latter leading the table. Long negation also leads at 0.833. These strengths coexist with very low text-generation scores of 0.014 for short descriptions and 0.032 for long ones.

<details><summary>Group scores for short and long descriptions</summary>

**Short descriptions**

| Group | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Basic_Attribute | 0.740 | 0.745 | 0.730 | 0.435 | 0.550 | 0.750 | 0.840 | 0.725 | 0.785 |
| Basic_Relation | 0.800 | 0.759 | 0.752 | 0.564 | 0.646 | 0.761 | 0.845 | 0.727 | 0.737 |
| Basic_Reasoning | 0.783 | 0.650 | 0.625 | 0.495 | 0.476 | 0.701 | 0.782 | 0.732 | 0.736 |
| Advanced_Attribute+Relation | 0.680 | 0.691 | 0.592 | 0.409 | 0.493 | 0.637 | 0.767 | 0.656 | 0.680 |
| Advanced_Attribute+Reasoning | 0.606 | 0.587 | 0.588 | 0.424 | 0.324 | 0.590 | 0.755 | 0.571 | 0.677 |
| Advanced_Relation+Reasoning | 0.648 | 0.617 | 0.562 | 0.433 | 0.440 | 0.610 | 0.634 | 0.642 | 0.656 |
| Advanced_Text Generation | 0.109 | 0.276 | 0.072 | 0.045 | 0.045 | 0.570 | 0.869 | 0.027 | 0.014 |
| Advanced_Style Control | 0.733 | 0.700 | 0.700 | 0.800 | 0.533 | 0.700 | 0.800 | 0.667 | 0.700 |
| Real_Complex | 0.664 | 0.664 | 0.646 | 0.310 | 0.347 | 0.705 | 0.925 | 0.687 | 0.698 |

**Long descriptions**

| Group | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Basic_Attribute | 0.760 | 0.680 | 0.685 | 0.335 | 0.415 | 0.730 | 0.810 | 0.685 | 0.725 |
| Basic_Relation | 0.796 | 0.737 | 0.755 | 0.573 | 0.588 | 0.745 | 0.857 | 0.718 | 0.735 |
| Basic_Reasoning | 0.784 | 0.732 | 0.583 | 0.491 | 0.557 | 0.730 | 0.803 | 0.762 | 0.794 |
| Advanced_Attribute+Relation | 0.702 | 0.639 | 0.561 | 0.367 | 0.456 | 0.659 | 0.691 | 0.654 | 0.691 |
| Advanced_Attribute+Reasoning | 0.648 | 0.579 | 0.615 | 0.274 | 0.344 | 0.630 | 0.699 | 0.664 | 0.628 |
| Advanced_Relation+Reasoning | 0.668 | 0.574 | 0.532 | 0.425 | 0.340 | 0.628 | 0.694 | 0.610 | 0.647 |
| Advanced_Text Generation | 0.136 | 0.176 | 0.072 | 0.018 | 0.005 | 0.407 | 0.919 | 0.032 | 0.032 |
| Advanced_Style Control | 0.633 | 0.667 | 0.733 | 0.500 | 0.400 | 0.767 | 0.800 | 0.733 | 0.733 |
| Real_Complex | 0.653 | 0.672 | 0.675 | 0.381 | 0.407 | 0.709 | 0.918 | 0.649 | 0.687 |

</details>

The OCR channels cover 30 text scenes per description length. Inko has OCR recall of 0.000 for short descriptions and 0.004 for long descriptions, compared with 0.501 and 0.266 for SD3.5 and 0.868 and 0.895 for MOSS. Its GNED scores are close to PixArt's and worse than DreamLite's and SANA's, consistent with the near-zero OneIG character matches.

| Measurement | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short GNED ↓ | 0.819 | 0.725 | 0.841 | 0.924 | 0.968 | 0.481 | 0.229 | 0.873 | 0.874 |
| Long GNED ↓ | 0.814 | 0.756 | 0.866 | 0.966 | 0.998 | 0.659 | 0.150 | 0.836 | 0.863 |
| Short OCR recall ↑ | 0.068 | 0.078 | 0.011 | 0.017 | 0.000 | 0.501 | 0.868 | 0.000 | 0.000 |
| Long OCR recall ↑ | 0.067 | 0.055 | 0.000 | 0.011 | 0.000 | 0.266 | 0.895 | 0.004 | 0.004 |

<details><summary>All 39 attribute scores for short and long descriptions</summary>

**Short descriptions**

| Attribute | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2d_spatial_relation | 0.792 | 0.833 | 0.708 | 0.583 | 0.542 | 0.708 | 0.875 | 0.667 | 0.667 |
| 3d_spatial_relation | 0.750 | 0.625 | 0.750 | 0.417 | 0.708 | 0.708 | 0.792 | 0.667 | 0.708 |
| action+2d | 0.889 | 0.833 | 0.806 | 0.667 | 0.667 | 0.833 | 0.917 | 0.806 | 0.778 |
| action+3d | 0.769 | 0.744 | 0.744 | 0.590 | 0.667 | 0.795 | 0.795 | 0.769 | 0.795 |
| action+color | 0.731 | 0.731 | 0.654 | 0.577 | 0.615 | 0.538 | 0.692 | 0.462 | 0.500 |
| action+texture | 0.862 | 0.655 | 0.690 | 0.552 | 0.621 | 0.862 | 0.897 | 0.690 | 0.793 |
| color+2d | 0.750 | 0.850 | 0.600 | 0.600 | 0.550 | 0.900 | 1.000 | 0.950 | 1.000 |
| color+3d | 0.700 | 0.700 | 0.600 | 0.350 | 0.450 | 0.550 | 0.750 | 0.700 | 0.650 |
| color+texture | 0.800 | 0.840 | 0.760 | 0.400 | 0.600 | 0.880 | 0.920 | 0.840 | 1.000 |
| comparison+2d | 0.500 | 0.688 | 0.562 | 0.438 | 0.438 | 0.625 | 0.562 | 0.562 | 0.562 |
| comparison+3d | 0.625 | 0.562 | 0.500 | 0.375 | 0.375 | 0.438 | 0.625 | 0.500 | 0.562 |
| comparison+color | 0.667 | 0.667 | 0.667 | 0.333 | 0.095 | 0.667 | 0.619 | 0.571 | 0.762 |
| comparison+texture | 0.300 | 0.350 | 0.350 | 0.200 | 0.200 | 0.350 | 0.750 | 0.250 | 0.350 |
| comparison | 0.875 | 0.750 | 0.750 | 0.458 | 0.458 | 0.792 | 0.750 | 0.667 | 0.708 |
| differentiation+2d | 0.625 | 0.625 | 0.688 | 0.250 | 0.312 | 0.625 | 0.688 | 0.875 | 1.000 |
| differentiation+3d | 0.588 | 0.647 | 0.235 | 0.353 | 0.294 | 0.588 | 0.765 | 0.588 | 0.529 |
| differentiation+color | 0.810 | 0.762 | 0.762 | 0.524 | 0.476 | 0.524 | 0.857 | 0.667 | 0.952 |
| differentiation+texture | 0.750 | 0.600 | 0.450 | 0.500 | 0.150 | 0.700 | 0.900 | 0.700 | 0.750 |
| differentiation | 0.840 | 0.560 | 0.520 | 0.440 | 0.320 | 0.720 | 0.920 | 0.760 | 0.800 |
| negation+2d | 0.750 | 0.750 | 0.812 | 0.625 | 0.750 | 0.750 | 0.688 | 0.750 | 0.812 |
| negation+3d | 0.706 | 0.588 | 0.588 | 0.647 | 0.706 | 0.647 | 0.647 | 0.765 | 0.706 |
| negation+color | 0.611 | 0.611 | 0.667 | 0.667 | 0.556 | 0.722 | 0.778 | 0.667 | 0.667 |
| negation+texture | 0.556 | 0.556 | 0.778 | 0.556 | 0.500 | 0.611 | 0.722 | 0.722 | 0.722 |
| negation | 0.708 | 0.625 | 0.667 | 0.750 | 0.750 | 0.625 | 0.667 | 0.792 | 0.708 |
| numeracy+2d | 0.704 | 0.481 | 0.519 | 0.370 | 0.333 | 0.519 | 0.444 | 0.593 | 0.481 |
| numeracy+3d | 0.688 | 0.594 | 0.594 | 0.406 | 0.312 | 0.688 | 0.656 | 0.500 | 0.594 |
| numeracy+color | 0.700 | 0.650 | 0.600 | 0.200 | 0.225 | 0.600 | 0.775 | 0.425 | 0.600 |
| numeracy+texture | 0.455 | 0.500 | 0.432 | 0.409 | 0.386 | 0.545 | 0.636 | 0.568 | 0.614 |
| numeracy | 0.708 | 0.667 | 0.562 | 0.333 | 0.375 | 0.667 | 0.792 | 0.708 | 0.729 |
| real_world | 0.664 | 0.664 | 0.646 | 0.310 | 0.347 | 0.705 | 0.925 | 0.687 | 0.698 |
| shape+2d | 0.350 | 0.550 | 0.300 | 0.150 | 0.450 | 0.300 | 0.500 | 0.500 | 0.500 |
| shape+3d | 0.500 | 0.600 | 0.500 | 0.300 | 0.350 | 0.650 | 0.600 | 0.500 | 0.500 |
| shape+color | 0.600 | 0.620 | 0.640 | 0.420 | 0.440 | 0.700 | 0.760 | 0.580 | 0.600 |
| shape+texture | 0.600 | 0.640 | 0.600 | 0.360 | 0.440 | 0.580 | 0.720 | 0.640 | 0.660 |
| style | 0.733 | 0.700 | 0.700 | 0.800 | 0.533 | 0.700 | 0.800 | 0.667 | 0.700 |
| text | 0.109 | 0.276 | 0.072 | 0.045 | 0.045 | 0.570 | 0.869 | 0.027 | 0.014 |
| texture+2d | 0.714 | 0.667 | 0.667 | 0.190 | 0.238 | 0.571 | 0.810 | 0.667 | 0.667 |
| texture+3d | 0.833 | 0.778 | 0.722 | 0.556 | 0.667 | 0.722 | 0.889 | 0.778 | 0.833 |
| texture+color | 0.960 | 0.880 | 0.920 | 0.560 | 0.720 | 0.840 | 0.960 | 0.840 | 0.880 |

**Long descriptions**

| Attribute | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2d_spatial_relation | 0.833 | 0.750 | 0.792 | 0.500 | 0.542 | 0.750 | 0.917 | 0.667 | 0.708 |
| 3d_spatial_relation | 0.667 | 0.625 | 0.708 | 0.667 | 0.583 | 0.708 | 0.750 | 0.708 | 0.708 |
| action+2d | 0.889 | 0.778 | 0.778 | 0.611 | 0.639 | 0.778 | 0.889 | 0.778 | 0.778 |
| action+3d | 0.795 | 0.795 | 0.744 | 0.513 | 0.590 | 0.744 | 0.872 | 0.718 | 0.744 |
| action+color | 0.731 | 0.731 | 0.500 | 0.500 | 0.577 | 0.692 | 0.500 | 0.692 | 0.577 |
| action+texture | 0.793 | 0.793 | 0.655 | 0.517 | 0.586 | 0.724 | 0.690 | 0.724 | 0.759 |
| color+2d | 0.800 | 0.800 | 0.600 | 0.300 | 0.500 | 0.850 | 0.850 | 0.850 | 0.900 |
| color+3d | 0.800 | 0.700 | 0.600 | 0.250 | 0.500 | 0.500 | 0.750 | 0.850 | 0.850 |
| color+texture | 0.880 | 0.800 | 0.720 | 0.400 | 0.520 | 0.880 | 0.920 | 0.920 | 0.840 |
| comparison+2d | 0.625 | 0.688 | 0.500 | 0.562 | 0.250 | 0.562 | 0.688 | 0.562 | 0.562 |
| comparison+3d | 0.812 | 0.562 | 0.562 | 0.438 | 0.125 | 0.562 | 0.750 | 0.562 | 0.562 |
| comparison+color | 0.667 | 0.619 | 0.619 | 0.333 | 0.286 | 0.667 | 0.762 | 0.667 | 0.714 |
| comparison+texture | 0.550 | 0.350 | 0.400 | 0.150 | 0.200 | 0.500 | 0.500 | 0.500 | 0.400 |
| comparison | 0.875 | 0.875 | 0.667 | 0.667 | 0.750 | 0.708 | 0.750 | 0.792 | 0.750 |
| differentiation+2d | 0.750 | 0.375 | 0.438 | 0.188 | 0.188 | 0.625 | 0.875 | 0.500 | 0.688 |
| differentiation+3d | 0.706 | 0.529 | 0.353 | 0.353 | 0.294 | 0.706 | 0.647 | 0.765 | 0.765 |
| differentiation+color | 0.810 | 0.857 | 0.667 | 0.381 | 0.429 | 0.905 | 0.857 | 0.667 | 0.857 |
| differentiation+texture | 0.600 | 0.650 | 0.450 | 0.050 | 0.300 | 0.450 | 0.750 | 0.650 | 0.500 |
| differentiation | 0.760 | 0.760 | 0.560 | 0.400 | 0.520 | 0.880 | 0.920 | 0.800 | 0.800 |
| negation+2d | 0.688 | 0.688 | 0.750 | 0.625 | 0.562 | 0.688 | 0.688 | 0.750 | 0.812 |
| negation+3d | 0.647 | 0.588 | 0.765 | 0.647 | 0.706 | 0.706 | 0.765 | 0.647 | 0.647 |
| negation+color | 0.667 | 0.722 | 0.778 | 0.389 | 0.444 | 0.611 | 0.667 | 0.722 | 0.667 |
| negation+texture | 0.556 | 0.500 | 0.722 | 0.500 | 0.500 | 0.833 | 0.556 | 0.778 | 0.556 |
| negation | 0.792 | 0.667 | 0.625 | 0.708 | 0.708 | 0.667 | 0.792 | 0.792 | 0.833 |
| numeracy+2d | 0.556 | 0.444 | 0.481 | 0.185 | 0.185 | 0.519 | 0.519 | 0.593 | 0.667 |
| numeracy+3d | 0.562 | 0.719 | 0.406 | 0.406 | 0.406 | 0.656 | 0.625 | 0.500 | 0.469 |
| numeracy+color | 0.725 | 0.550 | 0.625 | 0.250 | 0.300 | 0.550 | 0.800 | 0.650 | 0.625 |
| numeracy+texture | 0.614 | 0.386 | 0.659 | 0.136 | 0.295 | 0.523 | 0.705 | 0.682 | 0.705 |
| numeracy | 0.708 | 0.625 | 0.479 | 0.188 | 0.250 | 0.667 | 0.750 | 0.667 | 0.792 |
| real_world | 0.653 | 0.672 | 0.675 | 0.381 | 0.407 | 0.709 | 0.918 | 0.649 | 0.687 |
| shape+2d | 0.450 | 0.450 | 0.250 | 0.300 | 0.200 | 0.450 | 0.500 | 0.400 | 0.400 |
| shape+3d | 0.600 | 0.500 | 0.350 | 0.300 | 0.400 | 0.650 | 0.500 | 0.300 | 0.400 |
| shape+color | 0.620 | 0.640 | 0.600 | 0.320 | 0.300 | 0.620 | 0.780 | 0.600 | 0.700 |
| shape+texture | 0.700 | 0.600 | 0.580 | 0.340 | 0.360 | 0.580 | 0.660 | 0.540 | 0.560 |
| style | 0.633 | 0.667 | 0.733 | 0.500 | 0.400 | 0.767 | 0.800 | 0.733 | 0.733 |
| text | 0.136 | 0.176 | 0.072 | 0.018 | 0.005 | 0.407 | 0.919 | 0.032 | 0.032 |
| texture+2d | 0.667 | 0.524 | 0.810 | 0.381 | 0.381 | 0.571 | 0.905 | 0.524 | 0.810 |
| texture+3d | 0.778 | 0.611 | 0.722 | 0.389 | 0.500 | 0.833 | 0.833 | 0.889 | 0.833 |
| texture+color | 0.840 | 0.680 | 0.840 | 0.280 | 0.480 | 0.840 | 0.880 | 0.680 | 0.800 |

</details>

The short cone-and-cushion prompt tests whether two shapes, their spatial separation and the bookstand's function remain intact.

![Nine models on the TIIF short prompt “A blue conical bookstand stands beside the spherical cushion, creating a vibrant contrast in the room.”](assets/evaluation_tiif_cone_cushion.png)

Inko merges the requested objects into one sculpture, with several tapered supports rising from a ring of lobed cushions. Neither a single conical bookstand nor a separate spherical cushion survives. MOSS makes the bookstand's function explicit by placing books on a conical pedestal; DreamLite and SANA preserve the cone-and-cushion pairing but omit the books.

![Nine models on the TIIF long prompt requesting a large modern metallic record player beside a smaller older one.](assets/evaluation_tiif_record_players.png)

The record-player case tests differentiation by size, age and material. Inko renders reflective metal and warm ambient light in detail, but substitutes one large device and a loose vinyl record for the requested pair. DreamLite and MOSS retain two separate devices with a clear size contrast; MOSS also gives the smaller unit a visibly older design.

![Nine models on the TIIF long prompt requesting a busy city intersection with a complete lack of blue cars.](assets/evaluation_tiif_negation.png)

The negation case is a direct failure: Inko fills the intersection with blue and cyan cars. Most photographic references make the same error, preserving the busy street and the named objects while losing the absence constraint. SD 1.5 instead produces a stylized road map, which avoids literal blue cars but not the requested city-intersection scene.

<details><summary>Chinese negation examples</summary>

The five benchmark evaluations above use English prompt variants, so their negation results never exercise Chinese prompting. Inko can nevertheless understand Chinese absence instructions, including the imperative and declarative forms shown here.

![Two Inko still-life generations following Chinese instructions to omit bananas.](assets/evaluation_chinese_negation_apples.png)

For the left image, the exact prompt is `静物油画：木桌上摆着一盘红苹果，不要出现香蕉。`, meaning “A still-life oil painting: a plate of red apples sits on a wooden table; do not allow bananas to appear.” The imperative `不要出现` asks that bananas not appear. Inko keeps the plate of red apples and adds no bananas.

The right image uses the exact prompt `静物油画：木桌上摆着一盘红苹果，没有香蕉。`, meaning “A still-life oil painting: a plate of red apples sits on a wooden table; there are no bananas.” Here `没有` states the absence directly. Again, Inko preserves the apples without painting bananas. Across both phrasings, it responds to the absence relation rather than treating `香蕉` (“bananas”) as an instruction to include the named fruit.

</details>

<details><summary>Exact prompts for the three TIIF cases</summary>

**Blue cone and spherical cushion**

> A blue conical bookstand stands beside the spherical cushion, creating a vibrant contrast in the room.

**Two metallic record players**

> In a cozy corner bathed in soft, ambient light, a stunningly beautiful metallic vinyl record player, sleek and modern in its design, sits regally beside a smaller, more modestly sized and older metallic vinyl record player, whose timeworn charm hints nostalgically at its storied past, both standing together as silent witnesses to the evolution of sound and the passing tides of aural history.

**No blue cars**

> In the vivid tableau of urban life captured at the intersection of bustling streets that form the intricate arteries of the city, there exists a notable absence, a striking void, marked by the complete lack of any blue cars which, though invisible in this particular moment, often weave through the metropolitan tapestry with their cerulean hues, leaving only the impression of their absence amid the vibrant cacophony of the cityscape.

</details>

### Qwen-Image-Bench

[Qwen-Image-Bench](https://arxiv.org/abs/2605.28091) is a creator-centric benchmark with 1,000 stratified prompts. Its five dimensions, Aesthetics, Quality, Alignment, Real-world Fidelity and Creative Generation, decompose into 23 second-level capabilities and 56 third-level rubrics. The native total uses this hierarchy rather than a simple mean of the top-level dimensions. The evaluation here uses the English prompt variants.

Inko scores 42.02, effectively tied with SANA at 42.03 and ahead of DreamLite at 40.41 and PixArt at 41.40. SD3.5 reaches 43.09 and MOSS 54.56. Inko's Alignment score of 41.85 leads the compact entries and also edges SD3.5 at 40.97; its level-2 Attributes and Layout scores are second only to MOSS. Quality is 47.38, above every compact reference but below SD3.5 and MOSS.

Inko and SANA reach similar totals through different profiles. Inko trails SANA on Aesthetics (42.55 to 45.41) and Creative Generation (31.81 to 32.90), while leading on Alignment and Quality. Anatomical Portraiture is 6.8, second to MOSS at 12.6, and Text Rendering is 14.5 against MOSS at 48.7. Logical Resolution is 17.0, behind SANA at 18.5, PixArt at 24.4, SD3.5 at 20.3 and MOSS at 32.9; World Knowledge is 16.4, behind SD3.5 at 20.5 and MOSS at 34.0.

| Dimension | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aesthetics | 43.34 | 45.41 | 45.55 | 29.35 | 32.40 | 45.93 | 57.62 | 39.46 | 42.55 |
| Quality | 44.52 | 45.89 | 46.17 | 36.88 | 34.27 | 48.45 | 53.38 | 43.48 | 47.38 |
| Alignment | 41.05 | 40.76 | 38.95 | 26.13 | 25.40 | 40.97 | 54.89 | 38.45 | 41.85 |
| Real-world Fidelity | 38.64 | 39.99 | 40.53 | 38.71 | 36.39 | 42.65 | 48.94 | 41.69 | 41.12 |
| Creative Generation | 28.15 | 32.90 | 30.42 | 15.39 | 14.41 | 32.61 | 55.49 | 28.09 | 31.81 |
| Native total | 40.41 | 42.03 | 41.40 | 29.62 | 29.21 | 43.09 | 54.56 | 38.94 | 42.02 |

The bar-chart case combines structured layout, exact quantities and legible text. Inko produces a dense multi-bar graphic with many more than four bars, unreadable labels and a multicolored palette. MOSS is the only entry with the correct title, four bars and values of 12, 18, 9 and 21, although its category labels are not fully legible. SD3.5 draws six bars; 12 and 18 are legible, but the remaining values are corrupted.

![Nine models on the Qwen-Image-Bench prompt requesting a minimalist Q1-Q4 sales chart with values 12, 18, 9 and 21.](assets/evaluation_qwen_bar_chart.png)

<details><summary>Exact prompt for the bar-chart case</summary>

> Generate a minimalist bar chart poster titled "Q1–Q4 Sales (USD)", with the four bars containing values of 12, 18, 9, and 21 (numbers labeled at the top of the bars). The color scheme should be restrained with ample white space. The numbers should be clear and not misaligned, and the axes and scales should be simple, making the overall design suitable for business presentations.

</details>

<details><summary>All level-2 scores</summary>

| Level 1 | Level 2 | DreamLite | SANA | PixArt-Σ | SD 1.5 | SDXS | SD3.5 | MOSS | Inko base | Inko |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aesthetics | Composition | 40.38 | 41.74 | 41.38 | 24.74 | 32.56 | 43.52 | 58.74 | 36.10 | 42.24 |
| Aesthetics | Color Harmony | 61.24 | 63.62 | 64.58 | 51.05 | 54.82 | 63.78 | 69.01 | 58.64 | 60.12 |
| Aesthetics | Lighting | 55.06 | 58.55 | 58.78 | 35.46 | 40.80 | 56.44 | 67.92 | 49.87 | 53.14 |
| Aesthetics | Anatomical Portraiture | 4.97 | 5.00 | 5.30 | 1.96 | 1.28 | 5.72 | 12.57 | 5.97 | 6.81 |
| Aesthetics | Emotional Expression | 47.21 | 45.86 | 47.39 | 23.94 | 25.64 | 46.72 | 58.21 | 36.99 | 41.64 |
| Aesthetics | Style Control | 24.81 | 29.39 | 27.74 | 14.20 | 10.72 | 31.99 | 55.29 | 24.73 | 27.53 |
| Quality | Realism | 30.00 | 33.66 | 33.74 | 25.88 | 18.19 | 38.68 | 45.70 | 32.71 | 36.21 |
| Quality | Detail | 43.40 | 43.84 | 44.61 | 33.94 | 29.29 | 46.63 | 53.62 | 40.46 | 45.85 |
| Quality | Resolution | 59.84 | 59.62 | 59.88 | 50.31 | 55.20 | 59.60 | 60.58 | 56.54 | 59.48 |
| Alignment | Attributes | 47.65 | 48.28 | 45.05 | 31.15 | 29.69 | 48.55 | 61.30 | 47.30 | 50.28 |
| Alignment | Actions | 34.32 | 30.27 | 29.00 | 18.32 | 17.87 | 31.28 | 42.51 | 30.12 | 32.01 |
| Alignment | Relations | 25.42 | 26.43 | 25.34 | 12.71 | 10.55 | 26.78 | 46.74 | 24.49 | 27.95 |
| Alignment | Scene | 56.64 | 56.62 | 54.70 | 40.73 | 43.41 | 55.12 | 64.28 | 51.31 | 55.39 |
| Alignment | Layout | 44.95 | 43.06 | 41.00 | 29.04 | 29.22 | 43.06 | 56.72 | 41.92 | 45.05 |
| Real-world Fidelity | Fairness | 58.96 | 59.04 | 59.48 | 58.07 | 58.62 | 59.68 | 60.24 | 58.77 | 58.73 |
| Real-world Fidelity | Safety & Compliance | 59.75 | 59.49 | 59.87 | 60.00 | 59.62 | 60.00 | 59.87 | 59.87 | 59.75 |
| Real-world Fidelity | World Knowledge | 10.31 | 14.19 | 14.82 | 12.25 | 6.70 | 20.53 | 33.98 | 18.33 | 16.37 |
| Creative Generation | Imagination | 44.00 | 52.83 | 44.59 | 19.83 | 22.27 | 47.38 | 60.96 | 38.73 | 45.38 |
| Creative Generation | Feature Matching | 24.91 | 31.42 | 29.93 | 9.10 | 11.09 | 31.01 | 44.01 | 26.10 | 28.81 |
| Creative Generation | Logical Resolution | 15.98 | 18.51 | 24.41 | 4.46 | 9.62 | 20.31 | 32.87 | 17.08 | 17.02 |
| Creative Generation | Text Rendering | 13.85 | 11.36 | 7.74 | 6.64 | 4.05 | 14.21 | 48.72 | 13.47 | 14.47 |
| Creative Generation | Design Applications | 34.95 | 38.82 | 38.15 | 18.79 | 17.61 | 42.05 | 64.43 | 32.42 | 38.50 |
| Creative Generation | Visual Storytelling | 41.49 | 49.71 | 44.47 | 24.83 | 23.48 | 45.57 | 67.13 | 42.20 | 45.93 |

</details>

## Fine-art comparisons beyond benchmarks

The custom suite examines Chinese traditional painting, Western fine-art movements and composition across style and subject. These cases belong to Inko's target domain, so the comparisons consider both prompt adherence and medium-specific structure: how ink washes establish depth, whether fine brushwork survives, and what happens when a modern object enters a traditional visual language.

For each prompt, the panels show the seed-42 generation from every entry, produced with its established native inference recipe. Inko base and Inko occupy the bottom row. Five prompts are Chinese and six are English, so language understanding is part of the comparisons. The complete prompt and aspect ratio appear with each case.

### Chinese traditional painting

The four prompts require different kinds of control: atmospheric depth, countable subjects and fine brushwork, mineral-pigment gradation, and a densely constructed architectural landscape.

#### Ink-wash mountain landscape (16:9)

<details><summary>Prompt and English translation</summary>

> 水墨山水画，群山层叠，云雾穿行于山谷之间，一条溪流从远山蜿蜒流向近景，岸边孤松横斜，一叶小舟泊在水面。以浓淡墨色表现山石远近，焦墨勾勒松枝，留白构成云雾与水面，纸本水墨，兼工带写。

**English translation**

> An ink-wash landscape painting, with mountain ranges layered one behind another and clouds and mist threading through the valleys. A stream winds from the distant mountains into the foreground; a lone pine leans sideways on the bank, and a small boat is moored on the water. Use light and dark ink to convey the mountains' distance and nearness, dry, dark ink to outline the pine branches, and reserved white space to form the clouds, mist and water. Ink on paper, combining meticulous and freehand brushwork.

</details>

![Nine models on the ink-wash mountain landscape prompt.](assets/fine_art_extended/panels/ink_landscape.jpg)

A continuous stream carries the eye from the pale mountains into the foreground, where a small boat sits on an unpainted expanse of water. Large reserves read as mist and water. Dense pine-needle clusters and dry, dark trunks stand against broad gray mountain washes, giving the requested mixture of controlled and free brushwork. Pale ochre washes cover several rock faces, while blue-green accents remain subordinate to the darker ink structure.

The solitary pine becomes a foreground group, however, and several similarly dark trunks make the near bank busier than requested. MOSS achieves comparable mountain and tree detail with a more continuous wooded atmosphere, while DreamLite and SANA use broader reserves and simpler geography. SD3.5 also retains layered ink mountains but gives the stream and boat no clear foreground role. PixArt replaces the stream-and-boat composition with red foliage and animals. SD 1.5 produces a portrait-and-text collage, and SDXS a modern figure beside water, so neither reconstructs the requested painting.

#### Gongbi plum blossom and two birds (1:1)

<details><summary>Prompt and English translation</summary>

> 工笔花鸟画，一枝白梅斜伸入画，两只山雀停在枝头，羽毛丝理细密，花瓣以白粉分染，枝干用中锋细线勾勒。绢本设色，构图疏朗，细节精致，色调清雅。

**English translation**

> A gongbi bird-and-flower painting: a branch of white plum blossoms extends diagonally into the picture, with two small tits perched on the branch. The feather filaments are fine and dense, the petals are graded with white pigment, and the branches are outlined with fine lines drawn with a centered brush tip. Color on silk, a spacious composition, exquisite details, and an elegant, restrained palette.

</details>

![Nine models on the gongbi plum-blossom and two-bird prompt.](assets/fine_art_extended/panels/gongbi_bird_flower.jpg)

Two perched birds sit side by side on a diagonal white-plum branch, both facing left within a large silk-colored reserve. The composition remains sparse: branches enter from several edges, but the central pair is undisturbed by extra birds or decorative motifs. Fine parallel lines are visible in the wing and tail feathers, and the white blossoms retain their outlines against the pale ground.

The gongbi discipline is less consistent in the broad shading of the birds' breasts and the repeated blossom shapes, which are smoother and simpler than the requested dense feather work. MOSS also satisfies the two-bird constraint and gives the plumage finer hatching. DreamLite draws precise contours but three birds; SANA likewise has three, with one in flight. SD 1.5 and SD3.5 retain plant imagery without birds, SDXS turns the branch into decoration on a plate, and PixArt substitutes a woman for the bird-and-flower study.

#### Gongbi peony in rich color (1:1)

<details><summary>Prompt and English translation</summary>

> 工笔重彩牡丹，绢本设色，宋代院体。一枝牡丹从右下斜出，花瓣层层分染，石青石绿点叶，赭石勾枝，背景旧绢微黄，画面留白，左下角朱红印章。

**English translation**

> A gongbi peony painting in rich color, color on silk, in the Song-dynasty court-academy style. A peony branch slants out from the lower right; the petals are graded in layer upon layer of color, the leaves are dotted with azurite blue and malachite green, and the branches are outlined in brown ochre. The background is old silk yellowed slightly with age; the picture retains blank space, with a vermilion seal in the lower-left corner.

</details>

![Nine models on the gongbi peony in rich color prompt.](assets/fine_art_extended/panels/gongbi_peony.jpg)

Inko follows the compositional instructions closely: one peony branch enters from the lower right, the blossom dominates a large reserve of yellowed silk, and a vermilion seal sits in the lower-left without added calligraphy. Ochre stems and outlined leaves provide a linear scaffold, while the blue petals carry dense, layered gradation from pale edges to a dark heart.

The pigment distribution departs from the prompt in an interesting way. Azurite blue concentrates in the flower rather than joining malachite green in the leaves, which are mostly green and ochre. The petals also rely on soft, broad shading, where MOSS retains finer contour lines and more explicitly blue-green foliage. DreamLite and SANA produce recognizable peonies but add unrequested inscriptions; Inko keeps only the requested seal.

#### Blue-and-green landscape with pavilions (16:9)

<details><summary>Prompt and English translation</summary>

> 青绿山水，绢本设色，仿唐宋金碧山水。层叠山峰以石青石绿敷色，山腰楼阁朱柱金瓦，云雾留白，近景松石，画面填满，无装裱边框，旧绢底色温润。

**English translation**

> A blue-and-green landscape, color on silk, imitating Tang- and Song-dynasty blue-green-and-gold landscape painting. Layered peaks are colored with azurite blue and malachite green; pavilions on the mountainsides have vermilion pillars and golden tiles; clouds and mist are formed by reserved blank space; pine trees and rocks stand in the foreground. The picture is fully filled, with no mounting border, and the old-silk ground has a warm, mellow tone.

</details>

![Nine models on the blue-and-green landscape with pavilions prompt.](assets/fine_art_extended/panels/bluegreen_landscape.jpg)

Inko establishes the blue-green-and-gold family through mineral-colored peaks, a warm old-silk ground and pale reserved clouds. The central pavilion has vermilion columns and gold roof outlines, while gnarled pines and rocks fill the foreground. These elements occupy one continuous landscape rather than a collection of isolated Chinese motifs.

The smaller constraints are uneven. Gold picks out the pavilion's roof ridges but not its tiles, and small dark mounting tabs intrude along the upper edge despite the request for no mounting border. DreamLite makes the same border error more explicitly with a full scroll mount. MOSS is stronger on the central medium: finer mountain contours, layered architecture and intricate pine work produce a denser blue-green-and-gold painting on a similarly warm ground.

### Western fine-art movements

The six movements make different demands: directed light and material, dense botanical description, a transient action, expressive color, dreamlike atmosphere and ornamental line.

#### Baroque still life with silver and fruit (4:3)

<details><summary>Prompt</summary>

> A Baroque still life oil painting. A silver tazza with pomegranates, grapes and a cut lemon on a dark wooden table, a tall glass, strong chiaroscuro light from the left, deep shadows, crisp highlights on silver and fruit skin, visible canvas texture.

</details>

![Nine models on the Baroque still-life prompt.](assets/fine_art_extended/panels/baroque_still_life.jpg)

Inko builds the scene around material contrast. Broad highlights travel across the central silver tazza, pomegranate seeds and grape skins carry smaller reflections, and the tall glass remains dark but translucent. Left-hand illumination and a deep background tie these textures into a convincing Baroque still life.

The object relations are less exact. The tazza stands empty behind fruit arranged on the table, and the lemon is whole rather than cut. DreamLite, SD3.5 and MOSS include a cut lemon; MOSS also lets fruit overlap the dish, making the still life feel assembled rather than spaced out. Inko's separated objects are readable, but its symmetrical central stem and evenly spaced fruit give the composition a more ceremonial than spontaneous character.

#### Pre-Raphaelite walled garden (4:3)

<details><summary>Prompt</summary>

> A Pre-Raphaelite painting of a walled medieval garden at twilight, a woman in a flowing green gown seated beneath an apple tree, long auburn hair, detailed roses, foxgloves and tapestry-like foliage, hands hidden in her lap, luminous color.

</details>

![Nine models on the Pre-Raphaelite walled-garden prompt.](assets/fine_art_extended/panels/pre_raphaelite_garden.jpg)

Inko makes the garden do more than supply a backdrop. Apples canopy the seated woman, pointed arches establish the medieval wall, and roses and foxgloves remain recognizable among the denser ground cover. Repeated strokes in the green drapery echo the tapestry-like foliage, while the auburn hair provides a controlled warm accent against the cool twilight.

The dense botanical setting is closer to the Pre-Raphaelite brief than SD 1.5's flatter ivy wall and SDXS's simplified garden. MOSS achieves comparable density with a darker, less luminous palette. Inko nevertheless leaves both hands visibly clasped over the lap rather than hidden within it, and its smooth, idealized face is less intensely observed than the fabrics and plants around it.

#### Impressionist ballet dancer (3:4)

<details><summary>Prompt</summary>

> An Impressionist oil painting of a ballet dancer tying her shoe beside a tall studio window, morning light falling across the wooden floor. Loose, visible brushstrokes, a muted palette with flashes of pale blue and rose, soft edges, and an informal behind-the-scenes composition.

</details>

![Nine models on the Impressionist ballet-dancer prompt.](assets/fine_art_extended/panels/impressionist_dancer.jpg)

Morning light falls across the wooden floor beside a tall studio window, catching the blue bodice and rose tutu. Long strokes in the skirt and floor contrast with the smoother window light, and the bent figure gives an informal backstage view rather than a performance pose. The face, arms and tights are smoothly modeled. MOSS uses softer, more broken handling across the figure, while DreamLite uses broader, more angular strokes in the tutu and floor.

A loose ribbon hangs from Inko's clasped hands, but the hands stop near the knee, well above the extended pointe shoe. The image suggests preparation without actually showing the shoe being tied. DreamLite raises the shoe onto a small bench and puts the fingers at its ribbon, and MOSS bends the hands directly toward the foot, making the action legible. SANA instead turns shoes into an oversized central motif and introduces casual sneakers; PixArt and SDXS show dancing or posing rather than tying.

#### Post-Impressionist wheat field (4:3)

<details><summary>Prompt</summary>

> A Post-Impressionist oil painting of a wheat field under a turbulent evening sky, with a winding path leading toward a small village. Thick impasto, directional rhythmic brushwork, intense ultramarine and ochre, expressive color used to shape the landscape rather than merely describe it.

</details>

![Nine models on the Post-Impressionist wheat-field prompt.](assets/fine_art_extended/panels/post_impressionist_wheat.jpg)

Ultramarine and ochre structure the landscape. A blue-green path pulls the eye through the wheat toward a compact village, while directional strokes follow the stalks, spiral clouds and outlined hills. Thick ridges of simulated impasto remain visible across both sky and field, giving the surface a rhythmic structure that suits the prompt.

The image also relies on highly recognizable Van Gogh motifs: spiral clouds, cypress-like trees and heavily outlined hills. The result is closer to a direct quotation of that vocabulary than DreamLite, SANA or PixArt, which retain directional wheat and blue-ochre contrast with less literal cloud imagery. MOSS chooses a much quieter sky and darker field. SD3.5 also retains thick impasto, but organizes its sky around a bright central glow rather than repeated spirals.

#### Symbolist moonlit pool (16:9)

<details><summary>Prompt</summary>

> A Symbolist painting of a veiled figure standing at the edge of a moonlit pool, surrounded by irises and floating points of light, deep violet, teal and silver, dreamlike atmosphere, soft edges, face turned away.

</details>

![Nine models on the Symbolist moonlit-pool prompt.](assets/fine_art_extended/panels/symbolist_garden.jpg)

Inko turns the moon into a luminous backdrop for the veiled figure, with irises framing the pool and floating lights connecting sky and water. Violet, teal and silver remain distinct across the large shapes, and the soft veil, flower edges and reflections sustain the dreamlike atmosphere without dissolving the composition.

The interpretation leans toward contemporary fantasy illustration: the moon forms an almost perfect halo, and the symmetrical flowers make the scene deliberately iconic. The woman also stands on a small island in profile, whereas the prompt places her at the pool's edge with her face turned away. SANA and MOSS use rear views to satisfy that constraint more directly and avoid both the visible profile and the explicit halo.

#### Art Nouveau portrait (3:4)

<details><summary>Prompt</summary>

> An Art Nouveau portrait in profile of a young woman, framed by whiplash curves of irises and ginkgo leaves, an ornamental halo, muted gold, olive and dusty rose, flat decorative patterns, lithograph-like texture, no visible hands.

</details>

![Nine models on the Art Nouveau portrait prompt.](assets/fine_art_extended/panels/art_nouveau_portrait.jpg)

Inko treats Art Nouveau as an organizing system rather than a color filter. Concentric arcs form the ornamental halo, plant shapes repeat around the border, and the whiplash line across the cheek connects the portrait to its frame. Muted gold, olive and dusty rose, fine hair lines and flat background ornament support the lithographic character, and no hands appear.

The botanical brief is less faithfully resolved. The border reads as oak-like foliage and the hair carries round pink flowers, not the specified irises and ginkgo leaves; both eyes are visible, so the head is not a strict profile. SANA and MOSS introduce recognizable irises, while MOSS integrates plants and figure most densely. DreamLite has the opposite problem: its profile is clear, but a hand enters at the bottom and the halo becomes a floating ring detached from the ornament.

### Cross style–subject composition

The mechanical subject tests whether each model can retain a vehicle's defining geometry while changing the painting medium.

#### American muscle car in ink wash (1:1)

<details><summary>Prompt and English translation</summary>

> 水墨画风格，一辆美式肌肉车，车身低矮宽大，长引擎盖、短车尾，侧身线条硬朗。以浓淡墨色渲染车身，焦墨勾勒轮廓与轮毂，淡墨晕染阴影与地面，留白处透出宣纸质感。构图取三分之四前侧视角，车头略微朝左，背景以极简远山与薄雾衬托，营造刚柔对比的静谧氛围。整体色调仅黑、白、灰，笔触有干湿浓淡变化，兼工带写，气势沉稳。

**English translation**

> Ink-wash painting style: an American muscle car with a low, wide body, a long hood, a short rear deck and hard, angular side lines. Use light and dark ink to render the body, dry, dark ink to outline the silhouette and wheel hubs, and pale ink to wash in the shadows and ground; leave white space that reveals the texture of xuan paper. Compose a three-quarter front-side view, with the nose pointing slightly left. Use minimal distant mountains and thin mist as the background, creating a quiet atmosphere of contrast between strength and softness. The overall palette is only black, white and gray, with dry and wet, dark and light variations in the brushwork, combining meticulous and freehand techniques and a calm, powerful presence.

</details>

![Nine models on the American muscle car in ink-wash prompt.](assets/fine_art_extended/panels/ink_muscle_car.jpg)

The muscle car retains its low, wide body, long hood, short rear and three-quarter front view with the nose turned slightly left. Ink density does the structural work: a dark grille and wheels anchor the car, diluted washes describe the door panels, and dry strokes spread across the ground. Pale mountain washes and large areas of reserved paper keep the vehicle dominant without isolating it from the landscape.

MOSS also preserves the muscle-car stance and monochrome medium with heavier contrast. DreamLite and SANA retain vintage cars in ink landscapes, but SANA chooses a near-side profile and DreamLite a more upright coupe silhouette. Inko's polished, chrome-like edging and evenly rendered wheels remain closer to automotive illustration than the freer mountains behind it, so the fusion is not completely uniform. PixArt replaces the car with a woman, SD 1.5 and SDXS with food, and SD3.5 with a yellow truck.

## Protocol and reproduction

Each entry generated the same released catalogs. GenEval has 553 prompts with four draws, DPG has 1,065 prompts with four draws, OneIG English has 1,120 prompts with four draws, TIIF testmini has 277 scenes with both short and long descriptions, and Qwen-Image-Bench has 1,000 IDs. Generation saves lossless PNGs. Official judges consume those saved images in isolated environments, and the report recomputes native aggregates on CPU. There is no aggregate score across the five benchmarks.

Inko's trainable backbone has 532,496,992 parameters, with a frozen Qwen3-0.6B text encoder and frozen REPA-E image VAE. The compact reference backbones are DreamLite-Base at 0.390B, SANA-0.6B at 0.592B and PixArt-Σ at 0.611B. SD3.5 Medium has a 2.243B backbone and MOSS-Image-SFT 5.082B. SD 1.5 and SDXS use CLIP conditioning, which truncates long prompts at 77 tokens.

Both Inko entries use 20 Euler steps, shape-dependent timestep shift and text exit layer 20. The Inko base CFG of 5.0 was selected from experience with the pretrained checkpoint; the Inko CFG of 4.5 was selected by a small sweep after post-training and before its benchmarks. External models use their released pipelines and native recipes at 1024-native shapes, apart from SD 1.5 and SDXS at 512-native shapes. Adapted external generations matched the native pipelines pixel-identically on the qualification cohort. SD 1.5 runs with its safety checker disabled.

<details><summary>Environment qualification</summary>

OneIG's LLM2CLIP text encoder requires `attn_implementation="flash_attention_2"` with the current Transformers version.

</details>
