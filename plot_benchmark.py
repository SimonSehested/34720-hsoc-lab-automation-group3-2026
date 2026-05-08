import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


RAW_PATH = "benchmark_raw.csv"
SUMMARY_PATH = "benchmark_summary.csv"


raw = pd.read_csv(RAW_PATH)
summary = pd.read_csv(SUMMARY_PATH)

if "run_order" not in raw.columns:
    raw = raw.copy()
    raw["run_order"] = np.arange(1, len(raw) + 1)

if "test_started_at" in raw.columns:
    raw["test_started_at"] = pd.to_datetime(raw["test_started_at"], errors="coerce")

raw["config"] = raw.apply(
    lambda row: f"C{int(row['coarse'])}F{int(row['fine'])}",
    axis=1,
)
summary["config"] = summary.apply(
    lambda row: f"C{int(row['coarse'])}F{int(row['fine'])}",
    axis=1,
)

trial_counts = (
    raw.groupby(["coarse", "fine"])
    .size()
    .rename("n_trials_from_raw")
    .reset_index()
)
summary = summary.merge(trial_counts, on=["coarse", "fine"], how="left")
if "n_trials" not in summary.columns:
    summary["n_trials"] = summary["n_trials_from_raw"]
summary["n_trials"] = summary["n_trials"].fillna(summary["n_trials_from_raw"]).astype(int)

coarse_vals = sorted(summary["coarse"].unique())
fine_vals = sorted(summary["fine"].unique())
summary = summary.sort_values(["coarse", "fine"]).reset_index(drop=True)


def to_grid(col):
    return summary.pivot(index="fine", columns="coarse", values=col).loc[
        fine_vals, coarse_vals
    ].values


def annotate_heatmap(ax, data, fmt, cmap_name):
    finite = data[np.isfinite(data)]
    mid = (np.nanmin(finite) + np.nanmax(finite)) / 2 if finite.size else 0
    for (row_idx, col_idx), value in np.ndenumerate(data):
        if not np.isfinite(value):
            continue
        color = "white" if value < mid and cmap_name.endswith("_r") else "black"
        if cmap_name in {"viridis", "magma", "plasma"} and value < mid:
            color = "white"
        ax.text(
            col_idx,
            row_idx,
            format(value, fmt),
            ha="center",
            va="center",
            fontsize=8,
            color=color,
        )


def heatmap(ax, data, title, cmap, fmt=".1f", unit=""):
    im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(coarse_vals)))
    ax.set_xticklabels(coarse_vals)
    ax.set_yticks(range(len(fine_vals)))
    ax.set_yticklabels(fine_vals)
    ax.set_xlabel("Coarse iterationer")
    ax.set_ylabel("Fine iterationer")
    ax.set_title(title, fontsize=11)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(unit)
    annotate_heatmap(ax, data, fmt, cmap)


power_mean = to_grid("power_mean")
power_std = to_grid("power_std")
time_mean = to_grid("elapsed_mean")
n_trials = to_grid("n_trials")

fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig.suptitle("fast_sweep benchmark: coarse vs. fine", fontsize=15, fontweight="bold")

heatmap(
    axes[0, 0],
    power_mean,
    "Mean final power (dBm)\nlower is better",
    "RdYlGn_r",
    fmt=".1f",
    unit="dBm",
)
heatmap(
    axes[0, 1],
    power_std,
    "Power spread (standard deviation)\nlower is more stable",
    "YlOrRd",
    fmt=".2f",
    unit="dBm std",
)
heatmap(
    axes[1, 0],
    time_mean,
    "Mean runtime\nlower is faster",
    "RdYlGn_r",
    fmt=".0f",
    unit="seconds",
)
heatmap(
    axes[1, 1],
    n_trials,
    "Measurements per setup",
    "Blues",
    fmt=".0f",
    unit="trials",
)
fig.savefig("benchmark_heatmaps2.png", dpi=300, bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax.set_title("Pareto overview: final power vs. runtime", fontsize=14, fontweight="bold")
ax.set_xlabel("Mean runtime (s)")
ax.set_ylabel("Mean final power (dBm, lower is better)")

scatter = ax.scatter(
    summary["elapsed_mean"],
    summary["power_mean"],
    c=summary["fine"],
    s=np.clip(summary["n_trials"] * 2.0, 60, 260),
    cmap="viridis",
    edgecolor="black",
    linewidth=0.6,
    alpha=0.9,
)
ax.errorbar(
    summary["elapsed_mean"],
    summary["power_mean"],
    xerr=summary["elapsed_std"],
    yerr=summary["power_std"],
    fmt="none",
    alpha=0.25,
    color="black",
    linewidth=0.8,
)
for _, row in summary.iterrows():
    ax.annotate(
        row["config"],
        (row["elapsed_mean"], row["power_mean"]),
        textcoords="offset points",
        xytext=(5, 4),
        fontsize=8,
    )
ax.grid(True, alpha=0.25)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Fine iterationer")
cbar.set_ticks(fine_vals)
best = summary.loc[summary["power_mean"].idxmin()]
ax.scatter(
    [best["elapsed_mean"]],
    [best["power_mean"]],
    s=260,
    facecolors="none",
    edgecolors="crimson",
    linewidth=2,
)
ax.text(
    best["elapsed_mean"],
    best["power_mean"],
    " best mean",
    color="crimson",
    fontsize=9,
    va="top",
)
fig.savefig("benchmark_pareto2.png", dpi=300, bbox_inches="tight")
plt.close(fig)


ordered_configs = summary.sort_values("power_mean")["config"].tolist()
data_by_config = [
    raw.loc[raw["config"] == config, "final_power_dbm"].dropna().to_numpy()
    for config in ordered_configs
]

fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
ax.set_title("Final power distribution per setup", fontsize=14, fontweight="bold")
ax.set_ylabel("Final power (dBm, lower is better)")
ax.set_xlabel("Configuration sorted by mean final power")

box = ax.boxplot(
    data_by_config,
    patch_artist=True,
    showfliers=False,
    medianprops={"color": "black", "linewidth": 1.4},
)
for patch, config in zip(box["boxes"], ordered_configs):
    coarse = int(config.split("F")[0].replace("C", ""))
    patch.set_facecolor(plt.cm.tab10((coarse - 1) % 10))
    patch.set_alpha(0.35)

jitter_rng = np.random.default_rng(7)
for idx, values in enumerate(data_by_config, start=1):
    if values.size == 0:
        continue
    sample = values
    if values.size > 160:
        sample = jitter_rng.choice(values, size=160, replace=False)
    x = jitter_rng.normal(idx, 0.055, size=sample.size)
    ax.scatter(x, sample, s=10, alpha=0.28, color="black", linewidths=0)

ax.set_xticks(range(1, len(ordered_configs) + 1))
ax.set_xticklabels(ordered_configs, rotation=45, ha="right", fontsize=8)
ax.grid(True, axis="y", alpha=0.25)
fig.savefig("benchmark_trials2.png", dpi=300, bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
ax.set_title("Measurement order check", fontsize=14, fontweight="bold")
ax.set_ylabel("Final power (dBm)")
ordered_raw = raw.sort_values("run_order")
timestamp_available = (
    "test_started_at" in ordered_raw.columns
    and ordered_raw["test_started_at"].notna().any()
)
if timestamp_available:
    x_values = ordered_raw["test_started_at"]
    ax.set_xlabel("Test start time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
else:
    x_values = ordered_raw["run_order"]
    ax.set_xlabel("Run order")
ax.scatter(
    x_values,
    ordered_raw["final_power_dbm"],
    s=14,
    color="0.35",
    alpha=0.32,
    linewidths=0,
)
rolling = ordered_raw["final_power_dbm"].rolling(
    window=min(75, max(5, len(raw) // 20)),
    min_periods=3,
    center=True,
).median()
ax.plot(
    x_values,
    rolling,
    color="crimson",
    linewidth=2,
    label="Rolling median",
)
if timestamp_available:
    fig.autofmt_xdate()
ax.grid(True, alpha=0.25)
ax.legend()
fig.savefig("benchmark_drift2.png", dpi=300, bbox_inches="tight")
plt.close(fig)


best = summary.loc[summary["power_mean"].idxmin()]
fastest = summary.loc[summary["elapsed_mean"].idxmin()]
print("\nBest mean final power:")
print(
    f"  C{int(best.coarse)}F{int(best.fine)}: "
    f"{best.power_mean:.2f} +/- {best.power_std:.2f} dBm, "
    f"{best.elapsed_mean:.1f} +/- {best.elapsed_std:.1f} s, "
    f"n={int(best.n_trials)}"
)
print("Fastest mean runtime:")
print(
    f"  C{int(fastest.coarse)}F{int(fastest.fine)}: "
    f"{fastest.elapsed_mean:.1f} +/- {fastest.elapsed_std:.1f} s, "
    f"{fastest.power_mean:.2f} +/- {fastest.power_std:.2f} dBm, "
    f"n={int(fastest.n_trials)}"
)
print("\nSaved:")
print("  benchmark_heatmaps2.png")
print("  benchmark_pareto2.png")
print("  benchmark_trials2.png")
print("  benchmark_drift2.png")
