"""Saved-image joins and native layouts shared by benchmark adapters."""

import importlib.util
import json
from pathlib import Path

from PIL import Image

from src.evaluation.records import prompt_key, sample_key


def prompt(benchmark, split, identifier, text, metadata, seeds, variant="default"):
    return {
        "benchmark": benchmark,
        "split": split,
        "prompt_id": str(identifier),
        "variant": variant,
        "text": text,
        "metadata": {"aspect": "1:1", **metadata},
        "seeds": seeds,
    }


def join_images(catalog, samples, model, root):
    indexed = {sample_key(row): row for row in samples}
    if len(indexed) != len(samples):
        raise ValueError("duplicate image records")
    groups = []
    for row in catalog:
        images = []
        for draw, seed in enumerate(row["seeds"]):
            key = (model, *prompt_key(row), draw)
            if key not in indexed:
                raise ValueError(f"missing image record: {key}")
            sample = indexed[key]
            if sample["seed"] != seed:
                raise ValueError(f"image seed differs from catalog: {key}")
            path = Path(root) / sample["image_path"]
            with Image.open(path) as image:
                if image.size != (sample["width"], sample["height"]):
                    raise ValueError(f"image dimensions differ from record: {key}")
                image.verify()
            images.append({**sample, "image_path": str(path.resolve())})
        groups.append({"prompt": row, "samples": images})
    return groups


def link_image(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == Path(source).resolve():
            return
        destination.unlink()
    destination.symlink_to(Path(source).resolve())


def grid(samples, destination):
    """Four draws in upstream's row-major 2x2 order, with unchanged pixels."""
    if len(samples) != 4:
        raise ValueError("this benchmark requires four draws per prompt")
    images = [Image.open(row["image_path"]).convert("RGB") for row in samples]
    width, height = images[0].size
    if any(im.size != (width, height) for im in images):
        raise ValueError("grid draws must have equal dimensions")
    canvas = Image.new("RGB", (2 * width, 2 * height))
    for i, image in enumerate(images):
        canvas.paste(image, ((i % 2) * width, (i // 2) * height))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return width, height


def load_source(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None
