import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import theme
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
TELEMETRY = Path(__file__).resolve().parent / "data"
ASSETS = ROOT / "notes" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
COLORS = {
    "blue": theme.VERMILION,
    "orange": theme.CYCLE[1],
    "green": theme.CYCLE[5],
    "gray": theme.MUTED,
}
THOUSANDS = FuncFormatter(lambda value, _: f"{value / 1000:g}k")

theme.apply()


def read_series(name):
    frame = pd.read_csv(TELEMETRY / name)
    return pd.Series(frame.iloc[:, 1].to_numpy(), index=frame.iloc[:, 0].to_numpy())


def stitched(old_name, new_name):
    old = read_series(old_name)
    new = read_series(new_name)
    return pd.concat([old.loc[:480000], new.loc[480001:]])


def binned(series, width=500):
    frame = series.rename("value").to_frame()
    frame["bin"] = (frame.index.to_numpy() - 1) // width
    grouped = frame.groupby("bin").value
    stats = grouped.agg(["median", "min", "max"])
    stats["center"] = grouped.apply(lambda values: values.index.to_numpy().mean())
    return stats


def style(axis, x_thousands=True):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid()
    axis.set_xlabel("Optimizer update")
    if x_thousands:
        axis.xaxis.set_major_formatter(THOUSANDS)
    axis.legend(frameon=False)


loss = stitched("staged-run-pretrain-train-loss.csv", "pretrain-600k-train-loss.csv")
grad = stitched("staged-run-pretrain-grad-norm.csv", "pretrain-600k-grad-norm.csv")
assert loss.index.equals(grad.index)
fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)
for axis, series, label in zip(axes, (loss, grad), ("Training loss", "Pre-clip gradient norm")):
    stats = binned(series)
    axis.fill_between(stats.center, stats["min"], stats["max"], color=COLORS["blue"], alpha=0.14, linewidth=0, label="500-update range")
    axis.plot(stats.center, stats["median"], color=COLORS["blue"], lw=1.35, label="500-update median")
    for event in (56000, 250000, 400000, 480000):
        axis.axvline(event, color=theme.FAINT, lw=0.65, ls="--", alpha=0.35, zorder=1)
    for boundary in (450000, 570000):
        axis.axvline(boundary, color=theme.MUTED, lw=1.25, ls="--", alpha=0.75, zorder=2)
    axis.set_ylabel(label)
    style(axis)
axes[0].set_ylim(0.45, 2.1)
axes[1].set_yscale("log")
axes[1].axhline(1.0, color=COLORS["orange"], lw=1.0, ls=":", label="Clipping threshold")
axes[1].legend(frameon=False, loc="upper right")
axes[1].set_xlim(1, 600000)
axes[0].text(225000, 0.49, "256p", ha="center", va="bottom", fontsize=11)
axes[0].text(510000, 0.49, "640p", ha="center", va="bottom", fontsize=11)
axes[0].text(585000, 0.49, "896p", ha="center", va="bottom", fontsize=11)
axes[0].set_xlabel("")
fig.savefig(ASSETS / "pretrain_full_run_stability.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
for platform, label, color in (("ascend", "Ascend", COLORS["blue"]), ("h200", "H200", COLORS["orange"])):
    for axis, metric in zip(axes, ("train-loss", "grad-norm")):
        series = read_series(f"{platform}-hero-256p-{metric}.csv")
        axis.plot(series.index, series.to_numpy(), lw=1.15, alpha=0.9, color=color, label=label)
axes[0].set_ylabel("Training loss")
axes[1].set_yscale("log")
axes[1].set_ylabel("Pre-clip gradient norm (log scale)")
axes[1].set_xlim(12650, 12780)
for axis in axes:
    axis.axvline(12705, color=theme.MUTED, lw=0.9, ls="--")
    style(axis, x_thousands=False)
axes[0].set_xlabel("")
fig.savefig(ASSETS / "pretrain_spike_12705.png", dpi=180)
plt.close(fig)

KEYS = {
    "condition": "stability/conditioning/condition_rms",
    "hidden": "stability/conditioning/hidden_rms",
    "gate": "stability/summary/gate_rms_max",
    "residual": "stability/summary/residual_rms_max",
}


def load_jsonl(name):
    rows = [json.loads(line) for line in (TELEMETRY / name).read_text().splitlines()]
    return pd.DataFrame({key: [row["metrics"][metric] for row in rows] for key, metric in KEYS.items()}, index=[row["step"] for row in rows])


continuation = load_jsonl("staged-run-56k-continuation-stability-subset.jsonl")
cloud = {
    "condition": read_series("staged-run-pretrain-condition-rms.csv"),
    "hidden": read_series("staged-run-pretrain-hidden-rms.csv"),
    "gate": read_series("staged-run-pretrain-gate-rms-max.csv"),
    "residual": read_series("staged-run-pretrain-residual-rms-max.csv"),
}
corrected = {name: pd.concat([cloud[name].loc[:56000], continuation[name].loc[56001:100000]]) for name in KEYS}
labels = {"condition": "Conditioning RMS", "hidden": "Conditioning hidden RMS", "gate": "Maximum gate RMS", "residual": "Maximum residual RMS"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, constrained_layout=True)
for axis, names in zip(axes, (("condition", "hidden"), ("gate", "residual"))):
    for name in names:
        series = corrected[name].loc[30000:100000]
        axis.plot(series.index, series.to_numpy(), lw=1.35, label=labels[name])
    axis.axvline(56000, color=theme.MUTED, lw=1.0, ls="--")
    axis.set_ylabel("RMS")
    style(axis)
axes[0].set_title("Conditioning path")
axes[1].set_title("Residual stream")
fig.savefig(ASSETS / "pretrain_scale_mitigation_56k.png", dpi=180)
plt.close(fig)
