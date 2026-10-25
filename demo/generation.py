"""Canvas selection, prompt enhancement, and portable generation records."""

import json
import math
import os
import secrets
import uuid
from pathlib import Path

import requests
from PIL import Image, PngImagePlugin

from demo.pe_prompt import PE_SYSTEM_PROMPT
from src.utils.prompt_contract import (
    DROP_IDX,
    MAX_SEQUENCE_LENGTH,
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)

CANVASES = {
    "Standard": {
        "16:9": (1184, 672),
        "4:3": (1024, 768),
        "1:1": (896, 896),
        "3:4": (768, 1024),
        "9:16": (672, 1184),
    },
    "Large": {
        "16:9": (1712, 960),
        "4:3": (1472, 1104),
        "1:1": (1280, 1280),
        "3:4": (1104, 1472),
        "9:16": (960, 1712),
    },
}
ENHANCEMENT_FALLBACK = "Enhancement unavailable; used your original prompt."
HISTORY_LIMIT = 20


def whole_number(value, minimum, maximum, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value != int(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be a whole number from {minimum} to {maximum}.")
    return int(value)


def validate_dimensions(width, height):
    width = whole_number(width, 480, 1920, "Width")
    height = whole_number(height, 480, 1920, "Height")
    if width % 16 or height % 16:
        raise ValueError("Width and height must be multiples of 16.")
    return width, height


def canvas_choices(aspect, resolution, width, height):
    """List only canvases allowed by the user's explicit selections."""
    if aspect == "Custom":
        w, h = validate_dimensions(width, height)
        return [{"width": w, "height": h, "resolution": "Custom"}]
    if aspect not in ("Auto", *CANVASES["Standard"]):
        raise ValueError("Choose an available aspect ratio.")
    if resolution not in ("Auto", *CANVASES):
        raise ValueError("Choose an available resolution.")
    return [
        {"width": w, "height": h, "resolution": size, "aspect_ratio": ratio}
        for size, ratios in CANVASES.items()
        for ratio, (w, h) in ratios.items()
        if resolution in ("Auto", size) and aspect in ("Auto", ratio)
    ]


def fallback_canvas(aspect, resolution, width, height):
    canvas_choices(aspect, resolution, width, height)
    if aspect == "Custom":
        return validate_dimensions(width, height)
    return CANVASES["Standard" if resolution == "Auto" else resolution][
        "1:1" if aspect == "Auto" else aspect
    ]


def canvas_selection(width, height):
    for resolution, ratios in CANVASES.items():
        for aspect, dimensions in ratios.items():
            if dimensions == (width, height):
                return aspect, resolution
    return "Custom", "Standard"


def validate_prompt(prompt, tokenizer):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Describe the image you want to generate.")
    prompt = prompt.strip()
    formatted = PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
    tokens = tokenizer(formatted, truncation=False)["input_ids"]
    if len(tokens) > MAX_SEQUENCE_LENGTH + DROP_IDX:
        raise ValueError(
            f"The prompt exceeds Inko's {MAX_SEQUENCE_LENGTH}-token limit. "
            "Shorten it before generating."
        )
    return prompt


def sampling_settings(seed, randomize_seed, steps, guidance):
    steps = whole_number(steps, 10, 50, "Steps")
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(guidance)
        or not 0 <= guidance <= 10
    ):
        raise ValueError("CFG must be from 0 to 10.")
    seed = (
        secrets.randbelow(2**31)
        if randomize_seed
        else whole_number(seed, 0, 2**31 - 1, "Seed")
    )
    return {"seed": seed, "steps": steps, "guidance": float(guidance)}


def enhance_prompt(prompt, enhance, aspect, resolution, width, height, tokenizer):
    """Return a complete prompt/canvas pair, or the original request on PE failure."""
    prompt = validate_prompt(prompt, tokenizer)
    choices = canvas_choices(aspect, resolution, width, height)
    w, h = fallback_canvas(aspect, resolution, width, height)
    original = {"prompt": prompt, "width": w, "height": h, "notice": ""}
    if not enhance:
        return original
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return original | {"notice": ENHANCEMENT_FALLBACK}
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-flash",
                "messages": [
                    {"role": "system", "content": PE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "description": prompt,
                                "aspect_ratio": aspect,
                                "resolution": resolution,
                                "canvas_choices": choices,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
                "max_tokens": 4096,
                "stream": False,
            },
            timeout=(10, 90),
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("Incomplete enhancement")
        result = json.loads(choice["message"]["content"])
        if not isinstance(result, dict) or set(result) != {"prompt", "width", "height"}:
            raise ValueError("Invalid enhancement fields")
        if type(result["width"]) is not int or type(result["height"]) is not int:
            raise ValueError("Canvas dimensions must be integers")
        if (result["width"], result["height"]) not in {
            (item["width"], item["height"]) for item in choices
        }:
            raise ValueError("Enhancement changed an explicit canvas choice")
        result["prompt"] = validate_prompt(result["prompt"], tokenizer)
        return result | {"notice": ""}
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return original | {"notice": ENHANCEMENT_FALLBACK}


def save_image(image, record, directory):
    """Store the exact image and its recipe together, including Unicode prompts."""
    if image.size != (record["width"], record["height"]):
        raise ValueError("The generated image does not match the requested canvas.")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("inko", json.dumps(record, ensure_ascii=False))
    path = Path(directory) / f"inko-{uuid.uuid4().hex}.png"
    image.save(path, pnginfo=metadata)
    return str(path)


def read_record(path, directory):
    """Read a generated result from the app's output directory."""
    path = Path(path).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError("Choose a result generated in this session.")
    with Image.open(path) as image:
        record = json.loads(image.info["inko"])
    return record | {"image": str(path), "kind": "Generated"}


def delete_results(records, directory):
    root = Path(directory).resolve()
    for record in records:
        path = Path(record["image"]).resolve()
        if path.is_relative_to(root):
            path.unlink(missing_ok=True)


def append_history(history, record, directory, limit=HISTORY_LIMIT):
    updated = [*history, record]
    delete_results(updated[:-limit], directory)
    return updated[-limit:]
