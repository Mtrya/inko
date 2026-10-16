"""Materialize Hub rollout rows and scene-disjoint training/development splits."""

import argparse
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.evaluation.benchmarks.prompts import normalized_text
from src.posttrain.config import load_training_config
from src.posttrain.prompts import load_rollout_prompts
from src.posttrain.schema import name, require
from src.posttrain.scoring.contracts import rubric_criteria


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def scene_groups(rows, metadata):
    """Keep reused content seeds and duplicate prompt text in the same split."""
    parent = {row["id"]: row["id"] for row in rows}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for row in rows:
        meta = metadata[row["id"]]
        source = (
            meta["img2"] if meta["seed_kind"] == "image" else meta["caption_source"]
        )
        keys = [
            ("seed", source["dataset"], source["shard"], source["row"]),
            ("text", normalized_text(row["text"])),
        ]
        for key in keys:
            if key in seen:
                a, b = sorted((root(row["id"]), root(seen[key])))
                parent[b] = a
            else:
                seen[key] = row["id"]
    groups = defaultdict(list)
    for row in rows:
        row["scene_id"] = root(row["id"])
        groups[row["scene_id"]].append(row)
    return list(groups.values())


def split_rows(rows, metadata, *, heldout, seed, excluded_texts):
    groups = scene_groups(rows, metadata)
    rejected = [
        g
        for g in groups
        if any(normalized_text(r["text"]) in excluded_texts for r in g)
    ]
    rejected_ids = {g[0]["scene_id"] for g in rejected}
    strata = defaultdict(list)
    for group in groups:
        if group[0]["scene_id"] in rejected_ids:
            continue
        meta = metadata[group[0]["id"]]
        source = (
            meta["img2"] if meta["seed_kind"] == "image" else meta["caption_source"]
        )
        stratum = (
            meta["family"],
            meta["language"],
            meta["tier"],
            source["dataset"].split("@")[0],
        )
        strata[stratum].append(group)
    size = sum(len(g) for bucket in strata.values() for g in bucket)
    require(size > 2 * heldout, "held-out sets leave no rollout data")
    rng = random.Random(seed)
    splits = {key: [] for key in ("train", "development", "confirmation")}
    for _, bucket in sorted(strata.items()):
        bucket.sort(key=lambda group: group[0]["scene_id"])
        rng.shuffle(bucket)
        target = round(sum(len(g) for g in bucket) * heldout / size)
        counts = {"development": 0, "confirmation": 0}
        for group in bucket:
            destination = min(counts, key=counts.get)
            if counts[destination] >= target:
                destination = "train"
            else:
                counts[destination] += len(group)
            splits[destination].extend(group)
    return splits, [r["id"] for group in rejected for r in group]


def prepare(source, metadata_path, output, shapes, *, heldout, seed, benchmarks):
    import pyarrow.parquet as pq

    metadata_rows = read_jsonl(metadata_path)
    metadata = {row["row_id"]: row for row in metadata_rows}
    require(len(metadata) == len(metadata_rows), "duplicate production row ID")
    output = Path(output)
    references = output / "references"
    references.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(source).rglob("*.parquet"))
    require(bool(paths), "no Hub parquet shards found")
    rows, seen = [], set()
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for original in batch.to_pylist():
                key = original["row_id"]
                name(key)
                require(key not in seen, "duplicate Hub row ID")
                seen.add(key)
                meta = metadata[key]
                require(
                    meta["prompt"] == original["prompt"],
                    f"production prompt differs: {key}",
                )
                require(
                    meta["aspect"] in shapes, f"no shape for {key}: {meta['aspect']}"
                )
                ref_images = original["ref_images"] or []
                require(
                    len(ref_images) == (1 if original["seed_kind"] == "image" else 0),
                    f"unexpected reference count: {key}",
                )
                reference = None
                if ref_images:
                    data = ref_images[0]["bytes"]
                    require(bool(data), f"reference is not embedded: {key}")
                    with Image.open(io.BytesIO(data)) as image:
                        require(
                            image.format in {"PNG", "JPEG"},
                            "unsupported reference image",
                        )
                        suffix = ".png" if image.format == "PNG" else ".jpg"
                        image.verify()
                    reference = f"references/{key}{suffix}"
                    (output / reference).write_bytes(data)
                rows.append(
                    {
                        "id": key,
                        "text": original["prompt"],
                        "aspect": meta["aspect"],
                        "rubric": rubric_criteria(original["rubric"]),
                        "reference": reference,
                        "family": meta["family"],
                        "language": meta["language"],
                        "tier": meta["tier"],
                    }
                )
    excluded_texts = {
        normalized_text(row.get("text", row.get("prompt", "")))
        for path in benchmarks
        for row in read_jsonl(path)
    }
    splits, excluded = split_rows(
        rows, metadata, heldout=heldout, seed=seed, excluded_texts=excluded_texts
    )
    for split, selected in splits.items():
        path = output / f"{split}.jsonl"
        with path.open("w") as stream:
            for row in sorted(selected, key=lambda r: r["id"]):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        load_rollout_prompts(path, shapes)
    return {
        "rows": len(rows),
        "references": sum(r["reference"] is not None for r in rows),
        "excluded_overlap": excluded,
        "splits": {
            key: {
                "rows": len(value),
                "scenes": len({r["scene_id"] for r in value}),
                "aspects": dict(Counter(r["aspect"] for r in value)),
                "strata": dict(
                    Counter(f"{r['family']}/{r['language']}/{r['tier']}" for r in value)
                ),
            }
            for key, value in splits.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="production rows used for aspect and split metadata",
    )
    parser.add_argument(
        "--source", type=Path, help="already downloaded Hub parquet directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--heldout",
        type=int,
        default=500,
        help="approximate rows per held-out split; scenes stay intact",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[
            Path("configs/prompts/drawbench.jsonl"),
            Path("configs/prompts/monitor.jsonl"),
        ],
    )
    args = parser.parse_args()
    config = load_training_config(args.config)
    source = args.source
    if source is None:
        from huggingface_hub import snapshot_download

        source = snapshot_download(
            config.collection.dataset, repo_type="dataset", allow_patterns="*.parquet"
        )
    print(
        json.dumps(
            prepare(
                source,
                args.metadata,
                args.output,
                config.collection.shapes,
                heldout=args.heldout,
                seed=args.seed,
                benchmarks=args.exclude,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
