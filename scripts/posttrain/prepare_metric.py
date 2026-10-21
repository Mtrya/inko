"""Prepare a complete LPIPS-AlexNet asset for download-free evaluation."""

import argparse
from pathlib import Path

from src.posttrain.schema import require


def prepare(output, cache):
    import torch
    from torchmetrics.functional.image.lpips import _LPIPS

    from src.posttrain.checkpoint import save_tensor_file

    require(not output.exists(), "refusing to overwrite a prepared metric asset")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache))
    model = _LPIPS(net="alex", pretrained=True, pnet_rand=False, resize=None).eval()
    save_tensor_file(output, model.state_dict())
    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output, args.cache))
