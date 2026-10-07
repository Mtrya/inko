"""Service-owned loading, source verification and device telemetry."""

import importlib.metadata
import os

from ...artifacts import package_versions, verify_model, verify_source
from ...records import file_hash
from ...schema import require


def load_backend(config, assets, index):
    import torch

    from ...models import peak_memory, set_device, synchronize

    semantics = config.semantics
    require(
        set(assets["runtime_files"]) == set(semantics.runtime_files),
        "native runtime inventory differs",
    )
    for name, expected in semantics.runtime_files.items():
        require(
            file_hash(assets["runtime_files"][name]) == expected,
            f"unqualified native runtime: {name}",
        )
    for name, version in semantics.packages.items():
        require(
            importlib.metadata.version(name) == version,
            f"unqualified scorer runtime version: {name}",
        )
    require(
        assets["device_type"] != "npu" or "torch-npu" in semantics.packages,
        "NPU backend version must enter scorer identity",
    )
    provenance = {
        "native_runtime": semantics.runtime_files,
        "source": verify_source(assets["source"], semantics.source),
        "weights": verify_model(
            assets["weights"], assets["weights_manifest"], semantics.weights
        ),
        "packages": package_versions(),
    }
    device = set_device(assets["device_type"], index, config.limits.cpu_threads)
    os.environ["ACCELERATE_TORCH_DEVICE"] = str(device)
    if semantics.backend == "hpsv3":
        from .hpsv3 import HPSv3

        provenance["backbone"] = verify_model(
            assets["backbone"], assets["backbone_manifest"], config.options.backbone
        )
        model = HPSv3(config.options, assets, device)
    else:
        from .vqascore import VQAScore

        model = VQAScore(config.options, assets, device)

    class Backend:
        def score(self, requests):
            if device.type != "cpu":
                getattr(torch, device.type).set_device(device)
            with torch.inference_mode():
                result = model.score(requests)
            synchronize(device)
            return result

    synchronize(device)

    def telemetry():
        return {"device": str(device), "peak_memory_bytes": peak_memory(device)}

    return Backend(), telemetry, provenance
