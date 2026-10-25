"""Reproduce the 256p caption-length figure in notes/data.md.

The archived sidecars contain retained Qwen3 prompt lengths, including prompt
formatting, DROP_IDX removal and truncation. They are not raw caption lengths.
Each caption is counted once, without source or caption-sampling weights.

Run from the repository root:
    python scripts/figures/caption_lengths.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import theme
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "caption_lengths"
BANDS = [0, 64, 256, 512, 896, np.inf]
LABELS = ["<64", "64-255", "256-511", "512-895", "896+"]
COLORS = theme.CYCLE[:5]

theme.apply()


def load_lengths(directory):
    paths = sorted(directory.glob("*@256p.npz"))
    if not paths:
        raise FileNotFoundError(f"No 256p caption-length sidecars in {directory}")
    datasets = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as sidecar:
            datasets[path.stem.removesuffix("@256p")] = sidecar["prompt_lengths"]
    return datasets


def render(datasets, output):
    ordered = sorted(datasets, key=lambda name: -np.mean(datasets[name] >= 256))
    lengths = np.concatenate(list(datasets.values()))
    long_share = np.mean(lengths >= 256)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 10, "axes.labelsize": 8})
    fig, (bars, profile) = plt.subplots(
        2, 1, figsize=(9.2, 6.4), gridspec_kw={"height_ratios": [2.1, 1]}
    )
    fig.subplots_adjust(left=0.12, right=0.885, top=0.9, bottom=0.075, hspace=0.62)

    shares = np.array(
        [
            np.histogram(datasets[name], bins=BANDS)[0] / len(datasets[name])
            for name in ordered
        ]
    )
    left = np.zeros(len(ordered))
    for column, (label, color) in enumerate(zip(LABELS, COLORS)):
        bars.barh(
            ordered, shares[:, column], left=left, height=0.72, label=label, color=color
        )
        left += shares[:, column]
    bars.invert_yaxis()
    bars.set_xlim(0, 1)
    bars.set_xticks(np.linspace(0, 1, 5))
    bars.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bars.set_title(
        "Caption length bands per dataset (256p pool), sorted by long-caption share",
        loc="left",
        pad=26,
    )
    bars.legend(
        title="band (tokens)",
        ncol=5,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        fontsize=7,
        title_fontsize=7,
        handlelength=1.2,
        columnspacing=1.1,
    )
    bars.set_xlabel(
        "share of the dataset's captions per length band; right margin: share 256+ of caption count"
    )
    for row, name in enumerate(ordered):
        values = datasets[name]
        bars.text(
            1.015,
            row,
            f"{np.mean(values >= 256):.0%} of {len(values):,}",
            va="center",
            fontsize=7,
            color=theme.INK,
        )

    thresholds = np.arange(0, 1301, 8)
    sorted_lengths = np.sort(lengths)
    survival = 1 - np.searchsorted(sorted_lengths, thresholds, side="left") / len(
        lengths
    )
    profile.plot(thresholds, survival, color=COLORS[-1], linewidth=1.6)
    for threshold in BANDS[1:-1]:
        profile.axvline(threshold, color=theme.GRID, linestyle="--", linewidth=0.7)
    profile.set(
        xlim=(0, 1300),
        ylim=(0, 1),
        xlabel="caption length (Qwen3 tokens)",
        ylabel="captions at least\nthis long",
    )
    profile.set_yticks(np.linspace(0, 1, 5))
    profile.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    profile.set_title("Whole 256p pool, cumulative length profile", loc="left")
    profile.annotate(
        f"{long_share:.1%} of {len(lengths) / 1e6:.2f}M captions are 256+ tokens\n"
        "(first assembly: fewer than 100)",
        xy=(256, long_share),
        xytext=(380, 0.3),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": theme.INK, "lw": 0.8},
    )
    for axis in (bars, profile):
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(
        f"{len(datasets)} datasets, {len(lengths):,} captions, "
        f"{np.count_nonzero(lengths >= 256):,} at 256+ tokens -> {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "notes/assets/caption_lengths.png"
    )
    args = parser.parse_args()
    render(load_lengths(args.data_dir), args.output)


if __name__ == "__main__":
    main()
