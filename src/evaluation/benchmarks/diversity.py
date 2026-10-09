"""All-pairs LPIPS across independent seeds of the same exact caption."""

import inspect
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from src.posttrain.records import file_hash
from src.posttrain.schema import require


def prompt_key(prompt):
    return (
        prompt["scene_id"],
        prompt["text"],
        prompt.get("lang", ""),
        prompt.get("variant", ""),
        prompt["aspect"],
    )


def metric_identity(config, assets):
    import torchmetrics
    from torchmetrics.functional.image.lpips import _LPIPS

    require(config.asset in assets["metrics"], "metric asset is not provisioned")
    return {
        "asset": config.asset,
        "torchmetrics_version": torchmetrics.__version__,
        "implementation_sha256": file_hash(inspect.getfile(_LPIPS)),
    }


def grouped_pairs(records):
    groups = defaultdict(list)
    seen = set()
    for record in records:
        require(record["sample_id"] not in seen, "duplicate diversity sample")
        seen.add(record["sample_id"])
        groups[(record["arm"], prompt_key(record["prompt"]))].append(record)
    for (arm, key), rows in sorted(groups.items()):
        seeds = [r["prompt"]["seed"] for r in rows]
        require(
            len(set(seeds)) == len(seeds),
            "diversity needs distinct seeds per exact caption",
        )
        require(
            len(rows) >= 2,
            "diversity requires at least two seeds for every exact caption",
        )
        yield arm, key, rows, list(combinations(rows, 2))


def validate_panel(config, prompts):
    if not config.evaluation.diversity.enabled:
        return
    require(
        all(min(config.sampling.shapes[p["aspect"]]) >= 32 for p in prompts),
        "LPIPS-AlexNet requires image dimensions of at least 32",
    )
    for arm in config.arms:
        records = [
            {"sample_id": p["id"], "arm": arm.name, "prompt": p} for p in prompts
        ]
        list(grouped_pairs(records))


def summarize(groups):
    """Each scene has equal weight; caption variants are never image pairs."""
    by_arm = defaultdict(lambda: defaultdict(list))
    for group in groups:
        by_arm[group["arm"]][group["scene_id"]].append(group["mean_lpips"])
    return {
        arm: {
            "mean_by_scene": statistics.mean(
                statistics.mean(v) for v in scenes.values()
            ),
            "scenes": len(scenes),
            "exact_captions": sum(g["arm"] == arm for g in groups),
            "pairs": sum(g["pairs"] for g in groups if g["arm"] == arm),
        }
        for arm, scenes in sorted(by_arm.items())
    }


def measure_diversity(records, config, assets):
    import numpy as np
    import torch
    from PIL import Image
    from torchmetrics.functional.image.lpips import _LPIPS

    metric = metric_identity(config, assets)
    path = Path(assets["metrics"][config.asset])
    torch.set_num_threads(config.cpu_threads)
    # The prepared complete state replaces both backbone and linear heads. There
    # is no hub request and no random feature path in metric execution.
    model = _LPIPS(pretrained=False, pnet_rand=True, net="alex", resize=None)
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True
    )
    model.eval().requires_grad_(False)

    def image_tensor(record):
        require(
            file_hash(record["image_path"]) == record["image_sha256"],
            "diversity image bytes changed",
        )
        with Image.open(record["image_path"]) as picture:
            array = np.array(picture.convert("RGB"), dtype=np.float32, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1

    groups = []
    with torch.no_grad():
        for arm, key, rows, pairs in grouped_pairs(records):
            values = []
            for start in range(0, len(pairs), config.pair_batch_size):
                batch = pairs[start : start + config.pair_batch_size]
                left = torch.stack([image_tensor(a) for a, _ in batch])
                right = torch.stack([image_tensor(b) for _, b in batch])
                distances = model(left, right).flatten()
                require(
                    len(distances) == len(batch)
                    and bool(torch.isfinite(distances).all()),
                    "incomplete or nonfinite LPIPS distances",
                )
                values.extend(float(v) for v in distances)
            scene, text, lang, variant, aspect = key
            groups.append(
                {
                    "arm": arm,
                    "scene_id": scene,
                    "text": text,
                    "lang": lang,
                    "variant": variant,
                    "aspect": aspect,
                    "seeds": [r["prompt"]["seed"] for r in rows],
                    "pairs": len(pairs),
                    "mean_lpips": statistics.mean(values),
                }
            )
    require(bool(groups), "empty diversity panel")
    return {
        "metric": "LPIPS-AlexNet",
        **metric,
        "image_transform": "RGB; original resolution; pixel/127.5-1; no resize",
        "aggregation": "All seed pairs within an exact caption; caption mean within scene; equal scene mean.",
        "groups": groups,
        "arms": summarize(groups),
    }
