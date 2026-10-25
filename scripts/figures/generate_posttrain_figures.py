from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theme
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "posttrain"
ASSETS = ROOT / "notes" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
COLORS = {
    "blue": theme.VERMILION,
    "orange": theme.CYCLE[1],
    "green": theme.CYCLE[2],
    "pink": theme.CYCLE[4],
    "yellow": theme.CYCLE[3],
    "gray": theme.MUTED,
}

theme.apply()
UPDATES_PER_ROUND = {
    "selection-e0-art": 2,
    "selection-e0-nft": 1,
    "selection-e1-anchor-on": 2,
    "posttrain-600": 2,
}
RUNS = {}


def load_run(name):
    if name not in RUNS:
        frame = pd.read_csv(DATA / f"{name}-scalars.csv")
        RUNS[name] = {
            key: group[["step", "value"]].sort_values("step")
            for key, group in frame.groupby("key", sort=False)
        }
    return RUNS[name]


def series(name, key, evaluation=False):
    frame = load_run(name)[key]
    divisor = UPDATES_PER_ROUND[name] if evaluation else 1
    return pd.Series(frame.value.to_numpy(), index=frame.step.to_numpy() / divisor)


def binned(values, width):
    frame = pd.DataFrame({"round": values.index, "value": values.to_numpy()})
    frame["bin"] = (frame["round"] - 1) // width
    return frame.groupby("bin").agg(
        center=("round", "mean"),
        median=("value", "median"),
        lower=("value", lambda values: values.quantile(0.25)),
        upper=("value", lambda values: values.quantile(0.75)),
    )


def style(axis, legend=True):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid()
    axis.set_xlabel("Collection round")
    if legend:
        axis.legend(frameon=False)


def reward_band(axis, values, label, color, linestyle="-", width=20):
    stats = binned(values, width)
    axis.fill_between(
        stats.center,
        stats.lower,
        stats.upper,
        color=color,
        alpha=0.13,
        linewidth=0,
    )
    axis.plot(
        stats.center,
        stats["median"],
        color=color,
        linestyle=linestyle,
        linewidth=1.6,
        label=label,
    )


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=180, facecolor=theme.PAPER)
    plt.close(fig)


def art_nft_figure():
    fig, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    reward_band(
        axis,
        series("selection-e0-art", "train/reward/hpsv3.mu/mean"),
        "ART",
        COLORS["blue"],
    )
    reward_band(
        axis,
        series("selection-e0-nft", "train/reward/hpsv3.mu/mean"),
        "NFT",
        COLORS["orange"],
        "--",
    )
    axis.set_xlim(1, 1000)
    axis.set_ylabel("Collection HPSv3 mean")
    style(axis)
    save(fig, "posttrain_art_nft_collection_hpsv3.png")


def anchor_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for label, run, color, linestyle, marker in (
        ("No anchor", "selection-e0-art", COLORS["blue"], "-", "o"),
        ("Anchor 1e-4", "selection-e1-anchor-on", COLORS["orange"], "--", "s"),
    ):
        reward_band(
            axes[0],
            series(run, "train/reward/hpsv3.mu/mean"),
            label,
            color,
            linestyle,
        )
        student = series(
            run,
            "monitor/lpips/development/student_slow",
            evaluation=True,
        ).loc[:600]
        axes[1].plot(
            student.index,
            student.to_numpy(),
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            linewidth=1.5,
            label=label,
        )
    base = series(
        "selection-e0-art", "monitor/lpips/development/base", evaluation=True
    ).iloc[0]
    axes[1].axhline(
        base,
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.1,
        label="600k base",
    )
    axes[0].set_xlim(1, 650)
    axes[0].set_ylabel("Collection HPSv3 mean")
    axes[0].set_title("Collection reward")
    axes[1].set_xlim(0, 600)
    axes[1].set_ylabel("Within-scene LPIPS (higher = more diverse)")
    axes[1].set_title("Development diversity")
    for axis in axes:
        style(axis)
    save(fig, "posttrain_anchor_comparison.png")


def collection_figure():
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    for axis, key, label, color, linestyle in (
        (axes[0], "train/reward/hpsv3.mu/mean", "HPSv3", COLORS["blue"], "-"),
        (
            axes[1],
            "train/reward/vqascore.probability/mean",
            "VQAScore",
            COLORS["orange"],
            "--",
        ),
    ):
        reward_band(
            axis,
            series("posttrain-600", key),
            label,
            color,
            linestyle,
            width=10,
        )
        axis.set_ylabel(f"Collection {label} mean")
        axis.set_xlim(1, 300)
        style(axis)
    axes[0].set_xlabel("")
    save(fig, "posttrain_main_collection_rewards.png")


def decisive_figure():
    channels = (
        ("content_alignment", "Content alignment", COLORS["blue"], "-"),
        ("form_structure", "Form and structure", COLORS["orange"], "--"),
        ("aesthetics", "Aesthetic composition", COLORS["green"], "-."),
        ("style_medium", "Style and medium", COLORS["pink"], ":"),
        ("artifacts", "Artifacts", COLORS["yellow"], (0, (3, 1, 1, 1, 1, 1))),
    )
    fig, axis = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    for channel, label, color, linestyle in channels:
        mean = series("posttrain-600", f"train/reward/rubric.{channel}/mean")
        np.testing.assert_allclose(mean.to_numpy(), 0.5)
        std = series("posttrain-600", f"train/reward/rubric.{channel}/std")
        decisive = 4 * std**2
        if not decisive.between(0, 1).all():
            raise ValueError(f"Invalid decisive-pair ratio for {channel}")
        stats = binned(decisive, 10)
        axis.plot(
            stats.center,
            100 * stats["median"],
            color=color,
            linestyle=linestyle,
            linewidth=1.55,
            label=label,
        )
    axis.set_xlim(1, 300)
    axis.set_ylim(0, 75)
    axis.set_ylabel("Decisive pairs (%)")
    style(axis)
    axis.legend(frameon=False, ncol=2)
    save(fig, "posttrain_rubric_decisive_pairs.png")


def development_figure():
    specifications = (
        (
            "evaluation/development/student_ema/hpsv3.mu/mean",
            "evaluation/development/base/hpsv3.mu/mean",
            "HPSv3",
            3,
            COLORS["blue"],
        ),
        (
            "evaluation/development/student_ema/vqascore.probability/mean",
            "evaluation/development/base/vqascore.probability/mean",
            "VQAScore",
            3,
            COLORS["blue"],
        ),
        (
            "monitor/lpips/development/student_ema",
            "monitor/lpips/development/base",
            "Within-scene LPIPS",
            4,
            COLORS["orange"],
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5), sharex=True, constrained_layout=True)
    fig.suptitle(
        "100 DrawBench scenes × 4 seeds · CFG 1 · evaluation every 50 rounds",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="normal",
        color=COLORS["gray"],
    )
    for axis, (student_key, base_key, label, precision, color) in zip(axes, specifications, strict=True):
        student = series("posttrain-600", student_key, evaluation=True)
        base = series("posttrain-600", base_key, evaluation=True).iloc[0]
        is_lpips = label == "Within-scene LPIPS"
        axis.plot(
            student.index,
            student.to_numpy(),
            color=color,
            marker="o",
            markersize=5.5,
            markeredgecolor=theme.PANEL,
            markeredgewidth=0.6,
            linewidth=2.0,
        )
        axis.axhline(
            base,
            color=COLORS["gray"],
            linestyle=(0, (3, 2)),
            linewidth=1.2,
        )
        axis.set_title(label, loc="left", fontsize=13, fontweight="bold", pad=12)
        axis.set_xlim(0, 300)
        axis.set_xticks(np.arange(0, 301, 50))
        axis.margins(y=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
        axis.tick_params(labelsize=10, length=3, colors=theme.MUTED)
        final_round = student.index[-1]
        final_value = student.iloc[-1]
        axis.annotate(
            f"EMA {final_value:.{precision}f}",
            xy=(final_round, final_value),
            xytext=(0, -20 if is_lpips else 10),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        axis.annotate(
            "600k base",
            xy=(0, base),
            xytext=(5, 8 if is_lpips else -14),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=COLORS["gray"],
        )
        if is_lpips:
            axis.text(
                0,
                1.015,
                "higher = more diverse",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=9.5,
                color=COLORS["gray"],
            )
    fig.supxlabel("Collection round", fontsize=11)
    save(fig, "posttrain_main_development.png")


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def collapse_figure():
    source = DATA / "media" / "selection-e23-full-3e-3"
    crops = {
        "pines": (
            Image.open(source / "00700-style_pines.jpg").convert("RGB"),
            ((120, 42, 633, 426), (640, 42, 1152, 426)),
        ),
        "baker": (
            Image.open(source / "00700-figures_baker.jpg").convert("RGB"),
            ((120, 42, 504, 554), (512, 42, 896, 554)),
        ),
    }
    tile = 420
    left = 180
    header = 110
    gap = 24
    margin = 28
    width = left + 2 * tile + gap + 2 * margin
    height = header + 2 * tile + gap + 2 * margin
    canvas = Image.new("RGB", (width, height), theme.PAPER)
    draw = ImageDraw.Draw(canvas)
    headings = ("600k base", "Full tuning, Muon 3e-3\n(round 350)")
    for column, heading in enumerate(headings):
        x = left + column * (tile + gap)
        draw.multiline_text(
            (x + tile / 2, 24),
            heading,
            fill=theme.INK,
            font=font(24, bold=True),
            anchor="ma",
            align="center",
            spacing=4,
        )
    for row, (scene, label) in enumerate((("pines", "Pine scene"), ("baker", "Baker scene"))):
        y = header + row * (tile + gap)
        draw.text(
            (margin, y + tile / 2),
            label,
            fill=theme.INK,
            font=font(24, bold=True),
            anchor="lm",
        )
        image, boxes = crops[scene]
        for column, box in enumerate(boxes):
            x = left + column * (tile + gap)
            fitted = ImageOps.contain(image.crop(box), (tile, tile), Image.Resampling.LANCZOS)
            px = x + (tile - fitted.width) // 2
            py = y + (tile - fitted.height) // 2
            canvas.paste(fitted, (px, py))
            draw.rectangle((px, py, px + fitted.width - 1, py + fitted.height - 1), outline=theme.GRID, width=1)
    canvas.save(ASSETS / "posttrain_pointwise_collapse.png")


def comparison_figure():
    source = ASSETS / "comparisons"
    specifications = (
        (
            (
                "03-counting-oil-apples3-4x3",
                "COUNTING",
                "Three green apples",
                "Four apples become three, matching the prompt.",
            ),
            (
                "04-limbs-oil-violinist-3x4",
                "HUMAN STRUCTURE",
                "A violinist by the window",
                "One coherent musician; the right hand remains malformed.",
            ),
            (
                "10-short-cat-sofa-896",
                "TEXTURE DRIFT",
                "Two cats on a sofa",
                "Both images show two cats, but the painted fur and sofa become photographic.",
            ),
        ),
        (
            (
                "05-limbs-impress-dancer-1280",
                "ANATOMY + TEXTURE",
                "A spinning dancer",
                "Feet and legs become coherent, while the painted texture turns more photorealistic.",
            ),
            (
                "08-short-oldman-hand-896",
                "HANDS + POSTURE",
                "Old man with a cane",
                "The hands and stance become more coherent.",
            ),
            (
                "12-ink-beauty-style-896",
                "ANATOMY + STYLE",
                "Ink-wash beauty",
                "The figure is better formed, but Chinese ink wash shifts toward Western watercolor.",
            ),
        ),
    )
    width = 2400
    margin = 64
    top = 56
    column_gap = 48
    card_gap = 42
    card_width = (width - 2 * margin - column_gap) // 2
    image_gap = 20
    image_width = (card_width - image_gap) // 2
    header_height = 112
    note_line_height = 36
    footer_height = 92
    kicker_font = font(22, bold=True)
    title_font = font(34, bold=True)
    label_font = font(25)
    note_font = font(27)
    footer_font = font(24)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def wrap(text):
        lines = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if current and measure.textlength(candidate, font=note_font) > card_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    columns = []
    for specification in specifications:
        cards = []
        for slug, kicker, title, note in specification:
            base = Image.open(source / f"{slug}-base.png").convert("RGB")
            final = Image.open(source / f"{slug}-final.png").convert("RGB")
            if base.size != final.size:
                raise ValueError(f"Mismatched comparison dimensions for {slug}")
            image_height = round(image_width * base.height / base.width)
            note_lines = wrap(note)
            height = header_height + image_height + 14 + len(note_lines) * note_line_height + 28
            cards.append((base, final, kicker, title, note_lines, image_height, height))
        columns.append(cards)

    column_heights = [
        sum(card[-1] for card in cards) + card_gap * (len(cards) - 1)
        for cards in columns
    ]
    height = top + max(column_heights) + footer_height
    canvas = Image.new("RGB", (width, height), theme.PAPER)
    draw = ImageDraw.Draw(canvas)
    for column, cards in enumerate(columns):
        x = margin + column * (card_width + column_gap)
        y = top
        for index, (base, final, kicker, title, note_lines, image_height, card_height) in enumerate(cards):
            draw.text((x, y), kicker, fill=theme.MUTED, font=kicker_font)
            draw.text((x, y + 30), title, fill=theme.INK, font=title_font)
            draw.text((x, y + 74), "600k base", fill=theme.MUTED, font=label_font)
            draw.text((x + image_width + image_gap, y + 74), "Release", fill=theme.MUTED, font=label_font)
            image_y = y + header_height
            for image_x, image in ((x, base), (x + image_width + image_gap, final)):
                resized = image.resize((image_width, image_height), Image.Resampling.LANCZOS)
                canvas.paste(resized, (image_x, image_y))
                draw.rectangle(
                    (image_x, image_y, image_x + image_width - 1, image_y + image_height - 1),
                    outline=theme.GRID,
                    width=1,
                )
            note_y = image_y + image_height + 14
            for line_index, line in enumerate(note_lines):
                draw.text((x, note_y + line_index * note_line_height), line, fill=theme.INK, font=note_font)
            if index < len(cards) - 1:
                rule_y = y + card_height - 14
                draw.line((x, rule_y, x + card_width, rule_y), fill=theme.GRID, width=1)
            y += card_height + card_gap

    footer_y = height - 58
    draw.line((margin, footer_y - 20, width - margin, footer_y - 20), fill=theme.GRID, width=1)
    draw.text(
        (margin, footer_y),
        "Matched prompt, seed and shape · 20 Euler steps · Base CFG 5.0 · Release CFG 4.5",
        fill=theme.MUTED,
        font=footer_font,
    )
    canvas.save(ASSETS / "posttrain_base_final_comparisons.png", dpi=(180, 180))


def main():
    art_nft_figure()
    anchor_figure()
    collection_figure()
    decisive_figure()
    development_figure()
    collapse_figure()
    comparison_figure()


if __name__ == "__main__":
    main()
