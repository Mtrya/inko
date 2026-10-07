"""Content identity for the effective generation runtime after overlay resolution."""

import importlib.metadata
import json

from .records import file_hash

GENERATION_PACKAGES = (
    "torch",
    "torch-npu",
    "torchvision",
    "transformers",
    "tokenizers",
    "diffusers",
    "numpy",
    "pillow",
    "safetensors",
    "accelerate",
    "huggingface-hub",
)


def generation_runtime():
    result = {}
    for name in GENERATION_PACKAGES:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "torch-npu":
                continue
            raise
        files = {}
        for entry in distribution.files or ():
            path = distribution.locate_file(entry)
            if path.is_file() and path.suffix != ".pyc":
                files[str(entry)] = file_hash(path)
        if not files:
            raise ValueError(f"generation runtime has no content inventory: {name}")
        result[name] = {"version": distribution.version, "files": files}
    return result


if __name__ == "__main__":
    print(json.dumps(generation_runtime(), sort_keys=True))
