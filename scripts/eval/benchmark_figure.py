"""Render the benchmark overview bars and the r060-vs-base gain radar.

Reads a cluster report.json (see `scripts.eval.run report`) and writes
notes/assets/benchmark_overview.png plus notes/assets/posttrain_gain_radar.png.

Usage:
    python scripts/eval/benchmark_figure.py \
      --report /path/to/report.json --output-dir notes/assets
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

MODELS = {
    "inko_pretrain": ("Inko-Base", "#E59866", "main"),
    "inko_posttrain": ("Inko-Posttrain", "#C0392B", "main"),
    "sana_06b": ("SANA-0.6B", "#5B84B1", "main"),
    "pixart_sigma": ("PixArt-Σ", "#5B84B1", "main"),
    "dreamlite_base": ("DreamLite-Base", "#5B84B1", "main"),
    "sd15": ("SD 1.5", "#5B84B1", "main"),
    "sdxs": ("SDXS", "#5B84B1", "main"),
    "sd35_medium": ("SD3.5 Medium", "#9E9E9E", "large"),
    "moss_image_sft": ("MOSS-Image-SFT", "#9E9E9E", "large"),
}

PANELS = [
    ("geneval", "GenEval", lambda s: s["overall"]),
    ("dpg", "DPG", lambda s: s["overall"]),
    ("oneig", "OneIG (alignment)", lambda s: s["alignment"]["score"]),
    ("tiif", "TIIF (long)", lambda s: s["long"]["overall"]),
    ("qwen_image", "Qwen-Image", lambda s: s["total"]),
]


def headline(report, benchmark, model):
    entry = report["benchmarks"][benchmark]["models"].get(model)
    if not entry or entry["status"] != "complete":
        return None
    return next(f for key, _, f in PANELS if key == benchmark)(entry["scores"])


def collect(report):
    table = {}
    for benchmark, label, _ in PANELS:
        table[benchmark] = {}
        for model in MODELS:
            value = headline(report, benchmark, model)
            if value is not None:
                table[benchmark][model] = value
    return table


def render_overview(table, output):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(22, 4.2))
    for ax, (benchmark, label, _) in zip(axes, PANELS):
        values = table[benchmark]
        main = sorted(
            (m for m in values if MODELS[m][2] == "main"),
            key=lambda m: -values[m],
        )
        large = sorted(
            (m for m in values if MODELS[m][2] == "large"),
            key=lambda m: -values[m],
        )
        order = main + large
        skipped = [MODELS[m][0] for m in MODELS if m not in values]
        bars = ax.bar(
            range(len(order)),
            [values[m] for m in order],
            color=[MODELS[m][1] for m in order],
            width=0.72,
        )
        for index, bar in enumerate(bars):
            ax.annotate(
                f"{values[order[index]]:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        anchor = values.get("inko_posttrain")
        if anchor is not None:
            ax.axhline(anchor, color="#C0392B", linestyle="--", linewidth=1, alpha=0.8)
        if large:
            boundary = len(main) - 0.5
            ax.axvline(boundary, color="#BBBBBB", linewidth=1)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([MODELS[m][0] for m in order], rotation=38, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.12)
        if skipped:
            ax.set_xlabel("pending: " + ", ".join(skipped), fontsize=7)
    fig.suptitle("Standalone benchmark comparison", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def render_radar(table, output):
    labels = [label for _, label, _ in PANELS]
    base = [table[b]["inko_pretrain"] for b, _, _ in PANELS]
    post = [table[b]["inko_posttrain"] for b, _, _ in PANELS]
    peaks = [max(a, b) for a, b in zip(base, post)]
    base = [v / p for v, p in zip(base, peaks)]
    post = [v / p for v, p in zip(post, peaks)]

    angles = [n / len(labels) * 2 * 3.141592653589793 for n in range(len(labels))]
    base += base[:1]
    post += post[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(4.5, 4.5))
    ax.plot(angles, post, color="#C0392B", linewidth=2, label="Inko-Posttrain")
    ax.fill(angles, post, color="#C0392B", alpha=0.15)
    ax.plot(angles, base, color="#7F7F7F", linewidth=2, label="Inko-Base")
    ax.fill(angles, base, color="#7F7F7F", alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("notes/assets"))
    args = parser.parse_args()

    with args.report.open() as stream:
        report = json.load(stream)
    table = collect(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview = args.output_dir / "benchmark_overview.png"
    radar = args.output_dir / "posttrain_gain_radar.png"
    render_overview(table, overview)
    render_radar(table, radar)
    print(json.dumps({"overview": str(overview), "radar": str(radar)}))


if __name__ == "__main__":
    main()
