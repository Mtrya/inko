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

    def records(pool, keys, split, prefix):
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
                    "source_path": str(
                        paths[
                            "rollout" if split == "rollout" else "development"
                        ].relative_to(source)
                    ),
                }
            )
        return result

    if not 0 < rollout_bringup <= rollout_count <= len(
        train
    ) or not 0 < eval_bringup <= len(dev):
        raise ValueError("invalid subset sizes")
    rollout = records(
        train, ordered(train, "rollout")[:rollout_count], "rollout", "pickscore"
    )
    development = records(dev, ordered(dev, "development"), "development", "drawbench")
    suites = {
        "posttrain_rollout": rollout,
        "posttrain_rollout_bringup": rollout[:rollout_bringup],
        "posttrain_drawbench": development,
        "posttrain_drawbench_bringup": development[:eval_bringup],
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
        },
    )


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
    args = parser.parse_args()
    prepare(
        args.source,
        args.destination,
        args.monitor,
        args.rollout_count,
        args.rollout_bringup,
        args.eval_bringup,
    )
