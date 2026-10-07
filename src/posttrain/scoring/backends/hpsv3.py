"""Pinned HPSv3 with an explicit FP32 reward-head boundary."""

import importlib
import sys
import tempfile
from pathlib import Path


def float32_head_inputs(module, inputs):
    """HPSv3 specifies FP32 head weights after its BF16 transformer."""
    return (inputs[0].float(), *inputs[1:])


class HPSv3:
    def __init__(self, spec, assets, device):
        import yaml

        sys.path.insert(0, assets["source"])
        module = importlib.import_module("hpsv3.inference")
        config_path = Path(assets["source"]) / "hpsv3/config/HPSv3_7B.yaml"
        config = yaml.safe_load(config_path.read_text())
        config.update(
            model_name_or_path=assets["backbone"],
            disable_flash_attn2=True,
            deepspeed=None,
        )
        # The upstream loader consumes a YAML file and adjusts output_dir from it.
        with tempfile.TemporaryDirectory(prefix="inko-hps-") as directory:
            path = Path(directory) / "inference.yaml"
            path.write_text(yaml.safe_dump(config))
            self.scorer = module.HPSv3RewardInferencer(
                config_path=str(path),
                checkpoint_path=assets["weights"],
                device=str(device),
                differentiable=False,
            )
        self.scorer.model.requires_grad_(False)
        # Upstream uses CUDA autocast(dtype=float32), which disables autocast
        # rather than converting BF16 hidden states. Make the head boundary
        # explicit on every device without changing weights or preprocessing.
        self.scorer.model.rm_head.register_forward_pre_hook(float32_head_inputs)

    def score(self, requests):
        values = self.scorer.reward(
            prompts=[r.prompt for r in requests],
            image_paths=[r.image_path for r in requests],
        )
        if values.shape != (len(requests), 2):
            raise ValueError(f"unexpected HPSv3 output shape {values.shape}")
        return [
            {"mu": float(row[0]), "log_sigma": float(row[1])}
            for row in values.detach().cpu()
        ]
