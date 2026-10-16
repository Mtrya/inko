"""Named scores, rubric inputs and lossless image transport."""

import base64
import io
from pathlib import Path

from PIL import Image

from ..schema import require

CATEGORIES = (
    "content_alignment",
    "form_structure",
    "aesthetics",
    "style_medium",
    "artifacts",
)


def rubric_criteria(value):
    """Keep judge inputs from the dataset's richer production records."""
    require(
        isinstance(value, dict) and set(value) == set(CATEGORIES),
        "rubric requires all five categories",
    )
    result = {}
    for category in CATEGORIES:
        items = value[category]
        require(isinstance(items, list) and bool(items), "empty rubric category")
        result[category] = []
        for item in items:
            require(
                isinstance(item, dict)
                and isinstance(item.get("text"), str)
                and bool(item["text"].strip())
                and item.get("weight") in {"high", "mid", "low"},
                "rubric criterion requires text and high/mid/low weight",
            )
            result[category].append({"text": item["text"], "weight": item["weight"]})
    return result


CHANNELS = (
    *(f"rubric.{key}" for key in CATEGORIES),
    "hpsv3.mu",
    "vqascore.probability",
)


def image_payload(data):
    with Image.open(io.BytesIO(data)) as image:
        require(image.format in {"PNG", "JPEG"}, "scoring accepts PNG or JPEG images")
        mime = "image/png" if image.format == "PNG" else "image/jpeg"
        image.verify()
    return {"mime_type": mime, "data": base64.b64encode(data).decode()}


def image_bytes(payload):
    require(
        payload["mime_type"] in {"image/png", "image/jpeg"},
        "unsupported image MIME type",
    )
    data = base64.b64decode(payload["data"], validate=True)
    with Image.open(io.BytesIO(data)) as image:
        require(
            Image.MIME[image.format] == payload["mime_type"],
            "image MIME type differs from its bytes",
        )
        image.verify()
    return data


def data_url(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()
