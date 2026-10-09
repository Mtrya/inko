"""Prepare a complete LPIPS-AlexNet asset for download-free evaluation."""

import argparse
import inspect
from pathlib import Path

from src.posttrain.records import atomic_json, file_hash
from src.posttrain.schema import require


def prepare(output, cache):
    import torch
    import torchmetrics
    import torchvision
    from torchmetrics.functional.image.lpips import _LPIPS

    from src.posttrain.checkpoint import save_tensor_file

    require(not output.exists(), "refusing to overwrite a prepared metric asset")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache))
    model = _LPIPS(net="alex", pretrained=True, pnet_rand=False, resize=None).eval()
    save_tensor_file(output, model.state_dict())
    record = {
        "metric": "LPIPS-AlexNet",
        "sha256": file_hash(output),
        "torchmetrics": torchmetrics.__version__,
        "torchvision": torchvision.__version__,
        "implementation_sha256": file_hash(inspect.getfile(_LPIPS)),
        "backbone": "torchvision AlexNet_Weights.DEFAULT",
        "linear_heads": "torchmetrics bundled LPIPS alex.pth",
        "reference": "https://github.com/richzhang/PerceptualSimilarity",
    }
    atomic_json(output.with_suffix(".json"), record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output, args.cache))
