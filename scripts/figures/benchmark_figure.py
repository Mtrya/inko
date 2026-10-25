"""Render nine-model benchmark bars and selected matched case panels.

Usage:
    python -m scripts.figures.benchmark_figure \
      --report /path/to/report.json --samples notes/assets/benchmarks \
      --output-dir notes/assets
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps

MODELS = {
    "dreamlite_base": "DreamLite-Base",
    "sana_06b": "SANA-0.6B",
    "pixart_sigma": "PixArt-Σ",
    "sd15": "SD 1.5",
    "sdxs": "SDXS",
    "sd35_medium": "SD3.5 Medium",
    "moss_image_sft": "MOSS-Image-SFT",
    "inko_pretrain": "Inko base",
    "inko_posttrain": "Inko",
}
REFERENCES = tuple(MODELS)[:7]
INKO = tuple(MODELS)[7:]
ORDER = REFERENCES + INKO
COLORS = {
    "reference": "#5B84B1",
    "base": "#E59866",
    "release": "#C0392B",
}
PANELS = [
    ("geneval", "GenEval", lambda s: s["overall"], 3),
    ("dpg", "DPG", lambda s: s["overall"], 2),
    ("oneig", "OneIG alignment", lambda s: s["alignment"]["score"], 3),
    ("tiif", "TIIF long", lambda s: s["long"]["overall"], 3),
    ("qwen_image", "Qwen-Image", lambda s: s["total"], 2),
]
CASES = [
    (
        "geneval",
        "08-540-default",
        "GenEval 540: a black car and a green parking meter",
        "evaluation_geneval_car_meter.png",
    ),
    (
        "dpg",
        "01-205-default",
        "DPG 205: tent, lake, and gold bracelet",
        "evaluation_dpg_tent_bracelet.png",
    ),
    (
        "dpg",
        "09-whoops9-default",
        "DPG whoops9: skiing down a sand dune",
        "evaluation_dpg_sand_skier.png",
    ),
    (
        "tiif",
        "06-shape_color_0-short",
        "TIIF shape+color 0: blue cone and spherical cushion",
        "evaluation_tiif_cone_cushion.png",
    ),
    (
        "tiif",
        "03-differentiation_texture_1-long",
        "TIIF differentiation+texture 1: two metallic record players",
        "evaluation_tiif_record_players.png",
    ),
    (
        "tiif",
        "04-negation_4-long",
        "TIIF negation 4: no blue cars",
        "evaluation_tiif_negation.png",
    ),
    (
        "qwen_image",
        "09-999-en",
        "Qwen-Image 999: Q1-Q4 sales bar chart",
        "evaluation_qwen_bar_chart.png",
    ),
]


def headline(report, benchmark, model):
    entry = report["benchmarks"][benchmark]["models"].get(model)
    if not entry or entry["status"] != "complete":
        return None
    return next(f for key, _, f, _ in PANELS if key == benchmark)(entry["scores"])


def collect(report):
    return {
        benchmark: {
            model: value
            for model in MODELS
            if (value := headline(report, benchmark, model)) is not None
        }
        for benchmark, _, _, _ in PANELS
    }


def render_overview(table, output):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(24, 4.8), constrained_layout=True)
    for ax, (benchmark, label, _, precision) in zip(axes, PANELS, strict=True):
        values = table[benchmark]
        references = sorted(
            (model for model in REFERENCES if model in values),
            key=lambda model: (-values[model], model),
        )
        order = references + [model for model in INKO if model in values]
        positions = list(range(len(references)))
        positions += [len(references) + 1.0, len(references) + 2.0][
            : len(order) - len(references)
        ]
        colors = [COLORS["reference"]] * len(references) + [
            COLORS["base"],
            COLORS["release"],
        ][: len(order) - len(references)]
        bars = ax.bar(
            positions,
            [values[model] for model in order],
            color=colors,
            width=0.72,
            zorder=3,
        )
        for model, bar in zip(order, bars, strict=True):
            ax.annotate(
                f"{values[model]:.{precision}f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        if "inko_posttrain" in values:
            ax.hlines(
                values["inko_posttrain"],
                -0.55,
                positions[-1],
                color=COLORS["release"],
                linestyle=(0, (4, 3)),
                linewidth=1.1,
                zorder=2,
            )
        ax.set_title(label, fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [MODELS[model] for model in order], rotation=35, ha="right", fontsize=8
        )
        ax.set_xlim(-0.65, positions[-1] + 0.65)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.16, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Full-catalog benchmark results", fontsize=15)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_case(samples, benchmark, sample, title, output):
    size, label_height, title_height = 340, 34, 56
    sheet = Image.new(
        "RGB", (3 * size, title_height + 3 * (size + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 23)
    label_font = ImageFont.truetype("DejaVuSans.ttf", 21)
    draw.text((12, 13), title, fill="black", font=title_font)
    for index, model in enumerate(ORDER):
        x = (index % 3) * size
        y = title_height + (index // 3) * (size + label_height)
        path = samples / benchmark / "images" / sample / f"{model}.jpg"
        with Image.open(path) as source:
            image = ImageOps.contain(source.convert("RGB"), (size, size))
        sheet.paste(
            image, (x + (size - image.width) // 2, y + (size - image.height) // 2)
        )
        draw.text((x + 8, y + size + 5), MODELS[model], fill="black", font=label_font)
        draw.rectangle(
            (x, y, x + size - 1, y + size + label_height - 1), outline="#CCCCCC"
        )
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("notes/assets"))
    args = parser.parse_args()

    with args.report.open() as stream:
        report = json.load(stream)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / "benchmark_overview.png"]
    render_overview(collect(report), outputs[0])
    if args.samples:
        for benchmark, sample, title, filename in CASES:
            output = args.output_dir / filename
            render_case(args.samples, benchmark, sample, title, output)
            outputs.append(output)
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
