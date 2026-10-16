"""Service-owned loading and device telemetry."""

import os

from ...artifacts import package_versions, verify_model


def load_backend(config, assets, index):
    import torch

    from ...models import peak_memory, set_device, synchronize

    semantics = config.semantics
    provenance = {
        "source": str(assets["source"]),
        "weights": verify_model(assets["weights"], assets["weights_manifest"]),
        "packages": package_versions(),
    }
    device = set_device(assets["device_type"], index, config.limits.cpu_threads)
    os.environ["ACCELERATE_TORCH_DEVICE"] = str(device)
    if semantics.backend == "hpsv3":
        from .hpsv3 import HPSv3

        provenance["backbone"] = verify_model(
            assets["backbone"], assets["backbone_manifest"]
        )
        model = HPSv3(assets, device)
    else:
        from .vqascore import VQAScore

        model = VQAScore(assets, device)

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
