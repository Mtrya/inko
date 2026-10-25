"""Assemble a gallery image from mixed-aspect images.

Two layouts: `columns` is Pinterest-style masonry (uniform column width,
ragged bottom); `rows` is justified rows (flush left and right edges, ragged
final row). Images are center-cropped to a supported aspect range before
layout; the output is one image.

Usage:
    python scripts/eval/gallery.py --input notes/assets/gallery/hero/originals \
      --output notes/assets/gallery/hero/gallery.jpg --columns 3 --width 2000
    python scripts/eval/gallery.py --list order.txt --layout rows --width 2400
"""

import argparse
import random
from pathlib import Path

from PIL import Image

SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def collect(args):
    if args.list:
        files = [Path(line.strip()) for line in Path(args.list).read_text().splitlines() if line.strip()]
    else:
        files = sorted(p for p in Path(args.input).iterdir() if p.suffix.lower() in SUFFIXES)
    if args.order == "shuffle":
        random.Random(args.seed).shuffle(files)
    return files


def prepare(path, crop_range, long_edge=None):
    image = Image.open(path).convert("RGB")
    aspect = image.width / image.height
    low, high = crop_range
    if aspect < low:
        height = round(image.width / low)
        top = (image.height - height) // 2
        image = image.crop((0, top, image.width, top + height))
    elif aspect > high:
        width = round(image.height * high)
        left = (image.width - width) // 2
        image = image.crop((left, 0, left + width, image.height))
    if long_edge and max(image.size) > long_edge:
        scale = long_edge / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


def balance_plan(images, columns, width, gap):
    from itertools import product

    column_width = (width - gap * (columns - 1)) // columns
    heights = [round(image.height * column_width / image.width) for image in images]
    best, best_plan = None, None
    for assignment in product(range(columns), repeat=len(images)):
        totals = [0] * columns
        counts = [0] * columns
        for column, height in zip(assignment, heights):
            totals[column] += height
            counts[column] += 1
        score = max(totals[c] + gap * max(counts[c] - 1, 0) for c in range(columns))
        if best is None or score < best:
            best, best_plan = score, assignment
    return [c + 1 for c in best_plan]


def layout_columns(images, columns, width, gap, plan=None):
    column_width = (width - gap * (columns - 1)) // columns
    heights = [0] * columns
    slots = []
    for index, image in enumerate(images):
        scale = column_width / image.width
        size = (column_width, round(image.height * scale))
        column = plan[index] - 1 if plan else heights.index(min(heights))
        slots.append((image, column, heights[column], size))
        heights[column] += size[1] + gap
    height = max(heights) - gap
    canvas = Image.new("RGB", (width, height), "white")
    for image, column, top, size in slots:
        x = column * (column_width + gap)
        canvas.paste(image.resize(size, Image.LANCZOS), (x, top))
    return canvas


def layout_rows(images, width, gap, target_height):
    rows, current, aspect_sum = [], [], 0.0
    for image in images:
        current.append(image)
        aspect_sum += image.width / image.height
        if (width - gap * (len(current) - 1)) / aspect_sum <= target_height:
            rows.append((current, (width - gap * (len(current) - 1)) / aspect_sum, False))
            current, aspect_sum = [], 0.0
    if current:
        height = (width - gap * (len(current) - 1)) / aspect_sum
        rows.append((current, min(height, target_height), height > target_height))
    total = sum(round(height) for _, height, _ in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, total), "white")
    y = 0
    for row, height, capped in rows:
        height = round(height)
        widths = [image.width * height / image.height for image in row]
        available = width - gap * (len(row) - 1)
        if capped:
            widths = [round(w) for w in widths]
        else:
            scale = available / sum(widths)
            widths = [round(w * scale) for w in widths]
            widths[-1] += available - sum(widths)
        x = 0
        for image, w in zip(row, widths):
            canvas.paste(image.resize((w, height), Image.LANCZOS), (x, y))
            x += w + gap
        y += height + gap
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--list", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=["columns", "rows"], default="columns")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--crop-aspect", default="0.5,2.0")
    parser.add_argument("--long-edge", type=int, default=1600)
    parser.add_argument("--order", choices=["name", "shuffle"], default="name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan", help="comma-separated 1-based column per input image")
    parser.add_argument("--balance", action="store_true",
                        help="search the column assignment that minimizes the tallest column")
    parser.add_argument("--row-height", type=float, default=0.22,
                        help="rows layout target row height as a fraction of width")
    args = parser.parse_args()

    if not args.input and not args.list:
        parser.error("pass --input or --list")
    low, high = (float(x) for x in args.crop_aspect.split(","))
    files = collect(args)
    images = [prepare(path, (low, high), args.long_edge) for path in files]
    plan = [int(x) for x in args.plan.split(",")] if args.plan else None
    if args.balance:
        plan = balance_plan(images, args.columns, args.width, args.gap)
    if args.layout == "columns":
        canvas = layout_columns(images, args.columns, args.width, args.gap, plan)
    else:
        canvas = layout_rows(images, args.width, args.gap, args.row_height * args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() in {".jpg", ".jpeg"}:
        canvas.save(args.output, "JPEG", quality=92)
    else:
        canvas.save(args.output)
    print(f"{args.output} {canvas.width}x{canvas.height} {len(images)} images")


if __name__ == "__main__":
    main()
