"""Rebuild the small posttraining suites from a pinned DiffusionNFT checkout."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from src.evaluation.benchmarks.prompts import normalized_text
from src.posttrain.records import atomic_json, file_hash


def prepare(
    source: Path,
    destination: Path,
    monitor: Path,
    rollout_count: int,
    rollout_bringup: int,
    eval_bringup: int,
    confirmation_count: int,
    evaluation_seeds: int,
):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "cbb14f84b8312b620390dfcbe2fab69c1383b104":
        raise ValueError(
            "prompt source differs from the inspected DiffusionNFT revision"
        )
    paths = {
        "rollout": source / "dataset/pickscore/train.txt",
        "development": source / "dataset/drawbench/test.txt",
    }
    monitor_records = [
        json.loads(s) for s in monitor.read_text().splitlines() if s.strip()
    ]
    excluded = {normalized_text(p["text"]) for p in monitor_records}

    def unique(path):
        result = {}
        for text in path.read_text().splitlines():
            if text.strip():
                result.setdefault(normalized_text(text), text.strip())
        return result

    dev = unique(paths["development"])
    if excluded & dev.keys():
        raise ValueError("monitor and development texts overlap")
    excluded.update(dev)
    train = {k: v for k, v in unique(paths["rollout"]).items() if k not in excluded}

    def ordered(pool, namespace):
        return sorted(
            pool,
            key=lambda text: hashlib.sha256(f"{namespace}:{text}".encode()).hexdigest(),
        )

    def records(pool, keys, split, prefix, source_key):
        result = []
        for key in keys:
            digest = hashlib.sha256(key.encode()).hexdigest()
            result.append(
                {
                    "id": f"{prefix}_{digest[:16]}",
                    "scene_id": f"{prefix}_{digest[:16]}",
                    "text": pool[key],
                    "split": split,
                    "lang": "en",
                    "domain": "general",
                    "aspect": "1:1",
                    "seed": int(digest[:15], 16),
                    "source_revision": revision,
                    "source_path": str(paths[source_key].relative_to(source)),
                }
            )
        return result

    if not 0 < rollout_bringup <= rollout_count <= len(
        train
    ) or not 0 < eval_bringup <= len(dev):
        raise ValueError("invalid subset sizes")
    rollout = records(
        train,
        ordered(train, "rollout")[:rollout_count],
        "rollout",
        "pickscore",
        "rollout",
    )
    development = records(
        dev, ordered(dev, "development"), "development", "drawbench", "development"
    )
    used = {normalized_text(r["text"]) for r in rollout}
    remaining = {k: v for k, v in train.items() if k not in used}
    if not 0 < confirmation_count <= len(remaining) or evaluation_seeds < 2:
        raise ValueError(
            "confirmation count must fit the disjoint pool; evaluation needs two seeds"
        )
    confirmation = records(
        remaining,
        ordered(remaining, "confirmation")[:confirmation_count],
        "confirmation",
        "confirmation",
        "rollout",
    )

    def repeated(rows):
        return [
            {
                **r,
                "id": f"{r['id']}_seed{draw}",
                "seed": int(
                    hashlib.sha256(
                        f"eval-noise:{r['scene_id']}:{draw}".encode()
                    ).hexdigest()[:15],
                    16,
                ),
            }
            for r in rows
            for draw in range(evaluation_seeds)
        ]

    validate_pools(
        {
            "rollout": rollout,
            "development": development,
            "monitor": monitor_records,
            "confirmation": confirmation,
        }
    )
    suites = {
        "posttrain_rollout": rollout,
        "posttrain_rollout_bringup": rollout[:rollout_bringup],
        "posttrain_drawbench": development,
        "posttrain_drawbench_bringup": development[:eval_bringup],
        "posttrain_confirmation": confirmation,
        "posttrain_drawbench_diversity": repeated(development),
        "posttrain_drawbench_bringup_diversity": repeated(development[:eval_bringup]),
        "posttrain_confirmation_diversity": repeated(confirmation),
        "posttrain_monitor_diversity": repeated(monitor_records),
    }
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, rows in suites.items():
        path = destination / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        outputs[path.name] = {"count": len(rows), "sha256": file_hash(path)}
    atomic_json(
        destination / "posttrain_sources.json",
        {
            "repository": "https://github.com/NVlabs/DiffusionNFT",
            "revision": revision,
            "sources": {
                str(p.relative_to(source)): file_hash(p) for p in paths.values()
            },
            "monitor_sha256": file_hash(monitor),
            "outputs": outputs,
            "selection": "NFKC/casefold/whitespace deduplication; exclude development and monitor exact-normalized texts; SHA256(namespace:text) ordering",
            "overlap_scope": "Exact-normalized exclusion only; semantic paraphrase overlap is not certified.",
            "reservation": "Confirmation texts are excluded from rollout, development and monitor; do not select recipes on this pool.",
            "evaluation_seeds": evaluation_seeds,
        },
    )


def validate_pools(pools):
    inventories = {
        name: {normalized_text(r["text"]) for r in rows} for name, rows in pools.items()
    }
    for index, (name, texts) in enumerate(inventories.items()):
        for other in list(inventories)[index + 1 :]:
            if texts & inventories[other]:
                raise ValueError(f"prompt pools overlap: {name}, {other}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("configs/prompts"))
    parser.add_argument(
        "--monitor", type=Path, default=Path("configs/prompts/monitor.jsonl")
    )
    parser.add_argument("--rollout-count", type=int, default=512)
    parser.add_argument("--rollout-bringup", type=int, default=32)
    parser.add_argument("--eval-bringup", type=int, default=16)
    parser.add_argument("--confirmation-count", type=int, default=200)
    parser.add_argument("--evaluation-seeds", type=int, default=2)
    args = parser.parse_args()
    prepare(
        args.source,
        args.destination,
        args.monitor,
        args.rollout_count,
        args.rollout_bringup,
        args.eval_bringup,
        args.confirmation_count,
        args.evaluation_seeds,
    )
