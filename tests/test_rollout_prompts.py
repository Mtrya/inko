"""Dataset conversion keeps the Hub payload and joins only missing metadata."""

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from scripts.posttrain.prepare_prompts import prepare, split_rows
from src.posttrain.prompts import load_rollout_prompts


def rubric():
    return {
        category: [
            {"text": "An observable criterion", "weight": "high", "text_note": None}
        ]
        for category in (
            "content_alignment",
            "form_structure",
            "aesthetics",
            "style_medium",
            "artifacts",
        )
    }


@pytest.mark.parametrize("hub_split", ["train", "specialized"])
def test_prepare_reads_hub_images_and_preserves_joined_aspects(tmp_path, hub_split):
    source = tmp_path / "source"
    source.mkdir()
    encoded = io.BytesIO()
    Image.new("RGB", (19, 11), "red").save(encoded, format="JPEG")
    rows, metadata = [], []
    for index in range(8):
        key, text = f"p{index}", f"An individual scene {index}"
        rows.append(
            {
                "row_id": key,
                "prompt": text,
                "rubric": rubric(),
                "seed_kind": "image",
                "ref_images": [{"bytes": encoded.getvalue(), "path": "elsewhere.jpg"}],
            }
        )
        metadata.append(
            {
                "row_id": key,
                "prompt": text,
                "aspect": "portrait",
                "family": "painting",
                "language": "en",
                "tier": "short",
                "seed_kind": "image",
                "img2": {"dataset": "scenes", "shard": 0, "row": index},
            }
        )
    pq.write_table(
        pa.Table.from_pylist(rows), source / f"{hub_split}-00000-of-00001.parquet"
    )
    other_split = "specialized" if hub_split == "train" else "train"
    pq.write_table(
        pa.Table.from_pylist([{"row_id": "other-pool"}]),
        source / f"{other_split}-00000-of-00001.parquet",
    )
    meta = tmp_path / "metadata.jsonl"
    meta.write_text("\n".join(json.dumps(row) for row in metadata))
    output = tmp_path / "prepared"
    result = prepare(
        source,
        meta,
        output,
        {"portrait": (32, 16)},
        heldout=1,
        seed=23,
        benchmarks=[],
        split=hub_split,
    )
    assert result["rows"] == 8 and result["references"] == 8
    prepared = [
        row
        for split in ("train", "development", "confirmation")
        for row in load_rollout_prompts(
            output / f"{split}.jsonl", {"portrait": (32, 16)}
        )
    ]
    assert len(prepared) == 8 and len({r["id"] for r in prepared}) == 8
    assert all(r["aspect"] == "portrait" for r in prepared)
    assert (output / "references/p0.jpg").read_bytes() == encoded.getvalue()
    assert all(
        "text_note" not in item
        for r in prepared
        for items in r["rubric"].values()
        for item in items
    )
    metadata[0]["prompt"] = "A different prompt"
    meta.write_text("\n".join(json.dumps(row) for row in metadata))
    with pytest.raises(ValueError, match="production prompt differs"):
        prepare(
            source,
            meta,
            output,
            {"portrait": (32, 16)},
            heldout=1,
            seed=23,
            benchmarks=[],
            split=hub_split,
        )


def test_scene_variants_do_not_cross_splits_and_overlap_excludes_the_scene():
    rows = [{"id": f"p{i}", "text": f"scene {i}"} for i in range(12)]
    metadata = {
        r["id"]: {
            "seed_kind": "caption",
            "caption_source": {"dataset": "scenes", "shard": 0, "row": i // 2},
            "family": "photo",
            "language": "en",
            "tier": "short",
        }
        for i, r in enumerate(rows)
    }
    splits, rejected = split_rows(
        rows, metadata, heldout=2, seed=31, excluded_texts={"scene 0"}
    )
    assert set(rejected) == {"p0", "p1"}
    by_id = {r["id"]: split for split, values in splits.items() for r in values}
    assert set(by_id) == {f"p{i}" for i in range(2, 12)}
    for index in range(2, 12, 2):
        assert by_id[f"p{index}"] == by_id[f"p{index + 1}"]
