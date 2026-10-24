import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
TELEMETRY = ROOT / "telemetry"
ASSETS = ROOT / "notes" / "assets"
COLORS = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "gray": "#666666"}
THOUSANDS = FuncFormatter(lambda value, _: f"{value / 1000:g}k")
STAGES = (("256p", 1, 450000, 1), ("640p", 450001, 570000, 3), ("896p", 570001, 600000, 4))


def read_series(name):
    frame = pd.read_csv(TELEMETRY / name)
    return pd.Series(frame.iloc[:, 1].to_numpy(), index=frame.iloc[:, 0].to_numpy(), name="value")


def stitched(old_name, new_name):
    old = read_series(old_name)
    new = read_series(new_name)
    result = pd.concat([old.loc[:480000], new.loc[480001:]]).sort_index()
    assert not result.index.duplicated().any()
    return result


def binned(series, width=500):
    frame = series.to_frame()
    frame["bin"] = (frame.index.to_numpy() - 1) // width
    grouped = frame.groupby("bin").value
    stats = grouped.agg(["median", "min", "max"])
    stats["center"] = grouped.apply(lambda values: values.index.to_numpy().mean())
    return stats


def style(axis, x_thousands=True):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.16, linewidth=0.6)
    if x_thousands:
        axis.xaxis.set_major_formatter(THOUSANDS)


def boundaries(axis):
    for event in (56000, 250000, 400000, 480000):
        axis.axvline(event, color="#777777", lw=0.65, ls="--", alpha=0.35, zorder=1)
    for boundary in (450000, 570000):
        axis.axvline(boundary, color="#555555", lw=1.25, ls="--", alpha=0.75, zorder=2)
    axis.set_xlim(1, 600000)


micro = stitched("staged-run-pretrain-samples-per-micro-batch.csv", "pretrain-600k-samples-per-micro-batch.csv")
accumulation = np.select([micro.index <= 450000, micro.index <= 570000], [1, 3], default=4)
effective = micro * 16 * accumulation
assert len(effective) == 23954

fig = plt.figure(figsize=(13, 9), constrained_layout=True)
grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
for column, (stage, _, _, accum) in enumerate(STAGES):
    axis = fig.add_subplot(grid[0, column])
    plan = json.loads((ROOT / "configs" / "bucket_plans" / f"{stage}-k20.json").read_text())
    assert len(plan) == 5 and all(len(rows) == 20 for rows in plan.values())
    for rows in plan.values():
        axis.plot([row["max_length"] for row in rows], [row["batch_size"] for row in rows], color=COLORS["blue"], alpha=0.55, lw=1.15, marker="o", ms=2.2)
    axis.set_xscale("log", base=2)
    axis.set_xticks([16, 64, 256, 1024, 2048], ["16", "64", "256", "1,024", "2,048"])
    axis.set_xlim(14, 2300)
    axis.set_ylim(0, {"256p": 76, "640p": 14, "896p": 7}[stage])
    axis.set_title(f"{stage} · five aspect curves")
    axis.set_xlabel("Caption upper bound (tokens)")
    style(axis, x_thousands=False)
    axis.text(0.03, 0.05, "20 length buckets per aspect", transform=axis.transAxes, fontsize=8, color=COLORS["gray"])
    if column == 0:
        axis.set_ylabel("Samples per micro-batch / rank")

axis = fig.add_subplot(grid[1, :])
stats = binned(effective)
axis.plot(effective.index, effective.to_numpy(), color="#777777", alpha=0.22, lw=0.55, label="25-update window estimate")
axis.plot(stats.center, stats["median"], color=COLORS["green"], lw=1.5, label="500-update median")
boundaries(axis)
axis.set_ylim(0, 1100)
axis.set_ylabel("Estimated global samples / update")
axis.set_xlabel("Optimizer update")
axis.set_title("Global batch estimate: 16 ranks × accumulation × measured micro-batch mean")
style(axis)
axis.legend(frameon=False, loc="lower left")
for stage, lo, hi, _ in STAGES:
    median = float(effective.loc[lo:hi].median())
    center = (lo + hi) / 2
    axis.text(center, 1040, stage, ha="center", va="top", fontsize=11)
    axis.text(center, median + 45, f"median ≈ {median:.0f}", ha="center", va="bottom", fontsize=9, color=COLORS["green"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
fig.savefig(ASSETS / "pretrain_batch_plan.png", dpi=180)
plt.close(fig)

memory = stitched("staged-run-pretrain-mem-peak-gb.csv", "pretrain-600k-mem-peak-gb.csv")
assert len(memory) == 598873
stats = binned(memory)
fig, axis = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
axis.fill_between(stats.center, stats["min"], stats["max"], color=COLORS["blue"], alpha=0.14, linewidth=0, label="500-update range")
axis.plot(stats.center, stats["median"], color=COLORS["blue"], lw=1.35, label="500-update median")
axis.hlines(52, 450000, 600000, color=COLORS["orange"], lw=1.0, ls=":", label="52 GiB later-stage budget")
axis.axhline(60.96, color="#B22222", lw=1.0, ls="--", label="60.96 GiB device capacity")
boundaries(axis)
axis.set_ylim(44, 62)
axis.set_ylabel("Peak allocated GiB · logging rank")
axis.set_xlabel("Optimizer update")
style(axis)
axis.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4)
for stage, lo, hi, _ in STAGES:
    center = (lo + hi) / 2
    maximum = float(memory.loc[lo:hi].max())
    label_x, align = (595000, "right") if stage == "896p" else (center, "center")
    axis.text(center, 44.5, stage, ha="center", va="bottom", fontsize=11)
    axis.text(label_x, 53.2, f"stage max {maximum:.2f}", ha=align, va="bottom", fontsize=9, color=COLORS["gray"])
fig.savefig(ASSETS / "pretrain_memory.png", dpi=180)
plt.close(fig)

fig = plt.figure(figsize=(13, 9), constrained_layout=True)
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
axis = fig.add_subplot(grid[0, 0])
resolutions = ["256p", "640p", "896p"]
baseline = [1.32176, 4.80127, 7.05191]
autotuned = [1.29891, 4.58938, 6.80480]
x = np.arange(3)
width = 0.35
axis.bar(x - width / 2, baseline, width, color="#999999", label="Baseline")
axis.bar(x + width / 2, autotuned, width, color=COLORS["green"], label="Autotuned")
for index, (before, after) in enumerate(zip(baseline, autotuned)):
    axis.text(index, before + 0.25, f"−{100 * (1 - after / before):.2f}% time", ha="center", va="bottom", fontsize=8, color=COLORS["green"])
    for offset, value in ((-width / 2, before), (width / 2, after)):
        axis.text(index + offset, value - 0.08, f"{value:.2f}", ha="center", va="top", fontsize=7, color="white")
axis.set_xticks(x, resolutions)
axis.set_ylim(0, 8.3)
axis.set_ylabel("Warm seconds / update · lower is better")
axis.set_title("RTX 4090 · autotuning")
style(axis, x_thousands=False)
axis.legend(frameon=False, loc="upper left")

axis = fig.add_subplot(grid[0, 1])
labels = ["Baseline", "Fused RMSNorm", "Larger batches", "FFN 3072", "Native SwiGLU"]
values = [656.29, 737.68, 765.34, 843.29, 878.91]
gains = [None, 12.40, 3.75, 10.19, 4.22]
y = np.arange(len(labels))
bars = axis.barh(y, values, color=["#999999", COLORS["blue"], COLORS["blue"], COLORS["blue"], COLORS["green"]])
axis.set_yticks(y, labels)
axis.invert_yaxis()
axis.set_xlim(0, 1030)
axis.set_xlabel("Global samples / second")
axis.set_title("16×Ascend 910B2C · cumulative early-256p path")
style(axis, x_thousands=False)
for index, (bar, value, gain) in enumerate(zip(bars, values, gains)):
    suffix = "" if gain is None else f"  (+{gain:.2f}%)"
    axis.text(value + 12, bar.get_y() + bar.get_height() / 2, f"{value:.2f}{suffix}", va="center", fontsize=8)
axis.xaxis.set_major_formatter(ScalarFormatter())

axis = fig.add_subplot(grid[1, :])
throughput = stitched("staged-run-pretrain-samples-per-sec.csv", "pretrain-600k-samples-per-sec.csv")
assert len(throughput) == 598873
stats = binned(throughput)
axis.plot(throughput.index[::20], throughput.to_numpy()[::20], color=COLORS["blue"], alpha=0.14, lw=0.45, label="Logged EMA, subsampled")
axis.plot(stats.center, stats["median"], color=COLORS["blue"], lw=1.45, label="500-update median")
boundaries(axis)
axis.set_ylim(0, 950)
axis.set_ylabel("Global samples / second")
axis.set_xlabel("Optimizer update")
axis.set_title("Configured training runs · EMA of per-update throughput")
style(axis)
axis.legend(frameon=False, loc="lower left")
for stage, lo, hi, _ in STAGES:
    center = (lo + hi) / 2
    median = float(throughput.loc[lo:hi].median())
    axis.text(center, 925, stage, ha="center", va="top", fontsize=11)
    axis.text(center, median + 45, f"median ≈ {median:.0f}", ha="center", va="bottom", fontsize=9, color=COLORS["blue"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
fig.savefig(ASSETS / "pretrain_throughput.png", dpi=180)
plt.close(fig)

summary = {
    "effective_batch": {stage: {"median": round(float(effective.loc[lo:hi].median()), 3), "min": round(float(effective.loc[lo:hi].min()), 3), "max": round(float(effective.loc[lo:hi].max()), 3)} for stage, lo, hi, _ in STAGES},
    "memory_gib": {stage: {"median": round(float(memory.loc[lo:hi].median()), 4), "max": round(float(memory.loc[lo:hi].max()), 4)} for stage, lo, hi, _ in STAGES},
    "throughput": {stage: {"median": round(float(throughput.loc[lo:hi].median()), 3)} for stage, lo, hi, _ in STAGES},
    "paired_points": len(memory),
    "micro_windows": len(effective),
}
print(json.dumps(summary, indent=2))
