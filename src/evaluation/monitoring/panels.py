"""Labeled scene comparisons from the benchmark's existing generated images."""

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.posttrain.schema import require

VARIANTS = (("zh", "short"), ("en", "short"), ("zh", "long"), ("en", "long"))


def scene_grids(run, arms, step):
    run = Path(run)
    scenes = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for path in sorted((run / "records").glob("*.json")):
        row = json.loads(path.read_text())
        prompt = row["prompt"]
        group = scenes[prompt["scene_id"]][(prompt["lang"], prompt["variant"])][
            row["arm"]
        ]
        require(prompt["seed"] not in group, "duplicate monitor caption seed")
        group[prompt["seed"]] = row
    require(bool(scenes), "monitor has no generated images")
    directory = run / "monitor"
    directory.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=18)
    names = [arm.name for arm in arms]
    grids = {}
    for scene, variants in sorted(scenes.items()):
        require(
            set(variants) == set(VARIANTS),
            "monitor scene requires four language/length variants",
        )
        counts = set()
        for variant in VARIANTS:
            per_arm = variants[variant]
            require(set(per_arm) == set(names), "monitor scene is missing an arm")
            seeds = set(per_arm[names[0]])
            counts.add(len(seeds))
            require(
                all(set(per_arm[name]) == seeds for name in names),
                "monitor arm seeds differ",
            )
        require(len(counts) == 1, "monitor variants have different seed counts")
        grids[scene] = []
        for draw in range(counts.pop()):
            images, captions = [], [f"Step {step}; scene {scene}; draw {draw + 1}"]
            for variant in VARIANTS:
                per_arm = variants[variant]
                seed = sorted(per_arm[names[0]])[draw]
                prompt = per_arm[names[0]][seed]["prompt"]
                captions.append(
                    f"{variant[0]} / {variant[1]}; seed={seed}: {prompt['text']}"
                )
                for name in names:
                    row = per_arm[name][seed]
                    require(
                        row["prompt"] == prompt, "monitor arms have different prompts"
                    )
                    with Image.open(row["image_path"]) as source:
                        image = source.convert("RGB")
                        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        images.append(image)
            width = max(
                max(image.width for image in images),
                int(max(font.getlength(name) for name in names)) + 16,
            )
            height = max(image.height for image in images)
            margin, header, gap = 120, 42, 8
            canvas = Image.new(
                "RGB",
                (margin + len(names) * (width + gap), header + 4 * (height + gap)),
                "white",
            )
            labels = ImageDraw.Draw(canvas)
            for col, name in enumerate(names):
                labels.text(
                    (margin + col * (width + gap), 10), name, fill="black", font=font
                )
            for row, variant in enumerate(VARIANTS):
                top = header + row * (height + gap)
                labels.text(
                    (4, top + 10), f"{variant[0]} {variant[1]}", fill="black", font=font
                )
                for col in range(len(names)):
                    image = images[row * len(names) + col]
                    canvas.paste(
                        image,
                        (
                            margin + col * (width + gap) + (width - image.width) // 2,
                            top,
                        ),
                    )
            path = directory / f"{scene}-draw-{draw:03d}.jpg"
            canvas.save(path, quality=92)
            grids[scene].append((path, "\n\n".join(captions)))
    return grids
