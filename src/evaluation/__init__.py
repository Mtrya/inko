"""Evaluation module for Inko.

Live evaluation is driven by the training loop and diagnostics:
- eval_loss: teacher-forced eval loss probes (length-binned)
- prompt_grid: periodic sample grids on the fixed monitoring panel
- kid_eval: KID against precomputed reference statistics
"""

# Import the required submodule directly. Prompt manifests and benchmark
# orchestration must not load metric models or plotting libraries on import.
