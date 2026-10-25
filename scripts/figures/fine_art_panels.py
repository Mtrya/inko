"""Rebuild nine-model fine-art comparison panels.

Usage:
    uv run python -m scripts.figures.fine_art_panels \
      --root notes/assets/fine_art_extended
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.figures.benchmark_figure import MODELS, ORDER

PROMPTS = (
    "ink_landscape",
    "gongbi_bird_flower",
    "gongbi_peony",
    "bluegreen_landscape",
    "baroque_still_life",
    "pre_raphaelite_garden",
    "impressionist_dancer",
    "post_impressionist_wheat",
    "symbolist_garden",
    "art_nouveau_portrait",
    "ink_muscle_car",
)
CELL = 560
LABEL = 28


def render(images, prompt, output):
    sheet = Image.new("RGB", (3 * CELL, 3 * (CELL + LABEL)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    for index, model in enumerate(ORDER):
        cell_x = (index % 3) * CELL
        row_y = (index // 3) * (CELL + LABEL)
        with Image.open(images / prompt / f"{model}-s42.jpg") as source:
            image = ImageOps.contain(source.convert("RGB"), (CELL, CELL))
        x = cell_x + (CELL - image.width) // 2
        y = row_y + (CELL - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text(
            (cell_x + 8, row_y + CELL + 2), MODELS[model], fill="black", font=font
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("notes/assets/fine_art_extended")
    )
    args = parser.parse_args()
    outputs = []
    for prompt in PROMPTS:
        output = args.root / "panels" / f"{prompt}.jpg"
        render(args.root / "images", prompt, output)
        outputs.append(output)
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
