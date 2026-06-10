import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import os

OUTPUT_DIR = "graphs_100"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_PATH = "benchmark_raw.csv"
SUMMARY_PATH = "benchmark_summary.csv"
SUMMARY_C1C3_PATH = "benchmark_summary_c1_c3.csv"

raw = pd.read_csv(RAW_PATH)
summary = pd.read_csv(SUMMARY_PATH)
summary_c1c3 = pd.read_csv(SUMMARY_C1C3_PATH)

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

summary = summary.merge(summary_c1c3[["config", "balanced_rank_score", "good_30_rate", "excellent_32_rate"]], on="config", how="left")

raw = raw[raw["coarse"].isin([1, 2, 3])].copy()
summary = summary[summary["coarse"].isin([1, 2, 3])].copy()

raw["move_total"] = (
    abs(raw["final_arm1"] - raw["start_arm1"])
    + abs(raw["final_arm2"] - raw["start_arm2"])
    + abs(raw["final_arm3"] - raw["start_arm3"])
)
raw["start_spread"] = raw[["start_arm1", "start_arm2", "start_arm3"]].std(axis=1)
raw["final_spread"] = raw[["final_arm1", "final_arm2", "final_arm3"]].std(axis=1)

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

coarse_vals = sorted(raw["coarse"].unique())
fine_vals = sorted(raw["fine"].unique())
summary = summary.sort_values(["coarse", "fine"]).reset_index(drop=True)

def to_grid(col):
    return summary.pivot(index="fine", columns="coarse", values=col).loc[
        fine_vals, coarse_vals
    ].values

def to_grid_median(raw_col):
    medians = raw.groupby(["coarse", "fine"])[raw_col].median().reset_index()
    return medians.pivot(index="fine", columns="coarse", values=raw_col).loc[fine_vals, coarse_vals].values

def save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)

print("Generating 100 graphs...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, c in enumerate(coarse_vals):
    subset = raw[raw["coarse"] == c]["final_power_dbm"]
    axes[idx].hist(subset, bins=30, color=plt.cm.tab10(idx), alpha=0.7, edgecolor="black")
    axes[idx].set_title(f"C{c} Power Distribution")
    axes[idx].set_xlabel("dBm")
    axes[idx].set_ylabel("Count")
save(fig, "001_hist_power_by_coarse.png")

fig, ax = plt.subplots(figsize=(10, 5))
raw.boxplot(column="final_power_dbm", by="coarse", ax=ax, patch_artist=True)
ax.set_title("Power by Coarse Setting")
ax.set_ylabel("dBm (lower is better)")
plt.suptitle("")
save(fig, "002_boxplot_power_by_coarse.png")

fig, ax = plt.subplots(figsize=(10, 5))
raw.boxplot(column="final_power_dbm", by="fine", ax=ax, patch_artist=True)
ax.set_title("Power by Fine Setting")
ax.set_ylabel("dBm (lower is better)")
plt.suptitle("")
save(fig, "003_boxplot_power_by_fine.png")

fig, ax = plt.subplots(figsize=(10, 5))
raw.boxplot(column="elapsed_seconds", by="coarse", ax=ax, patch_artist=True)
ax.set_title("Runtime by Coarse Setting")
ax.set_ylabel("Seconds")
plt.suptitle("")
save(fig, "004_boxplot_time_by_coarse.png")

fig, ax = plt.subplots(figsize=(10, 5))
raw.boxplot(column="elapsed_seconds", by="fine", ax=ax, patch_artist=True)
ax.set_title("Runtime by Fine Setting")
ax.set_ylabel("Seconds")
plt.suptitle("")
save(fig, "005_boxplot_time_by_fine.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(raw["move_total"], bins=40, color="steelblue", alpha=0.7, edgecolor="black")
ax.set_title("Distribution of Total Arm Movement")
ax.set_xlabel("Total movement (degrees)")
ax.set_ylabel("Count")
save(fig, "006_hist_total_movement.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(raw["start_spread"], bins=30, color="forestgreen", alpha=0.7, edgecolor="black")
ax.set_title("Distribution of Start Position Spread")
ax.set_xlabel("Std dev of start positions")
ax.set_ylabel("Count")
save(fig, "007_hist_start_spread.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(raw["final_spread"], bins=30, color="darkorange", alpha=0.7, edgecolor="black")
ax.set_title("Distribution of Final Position Spread")
ax.set_xlabel("Std dev of final positions")
ax.set_ylabel("Count")
save(fig, "008_hist_final_spread.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["coarse"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Coarse vs Power")
ax.set_xlabel("Coarse setting")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "009_scatter_coarse_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["fine"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Fine vs Power")
ax.set_xlabel("Fine setting")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "010_scatter_fine_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["elapsed_seconds"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Runtime vs Power")
ax.set_xlabel("Elapsed seconds")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "011_scatter_time_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["start_arm1"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Start Arm1 vs Power")
ax.set_xlabel("Start position arm1")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "012_scatter_start_arm1_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["start_arm2"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Start Arm2 vs Power")
ax.set_xlabel("Start position arm2")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "013_scatter_start_arm2_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["start_arm3"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Start Arm3 vs Power")
ax.set_xlabel("Start position arm3")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "014_scatter_start_arm3_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["final_arm1"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Final Arm1 vs Power")
ax.set_xlabel("Final position arm1")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "015_scatter_final_arm1_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["final_arm2"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Final Arm2 vs Power")
ax.set_xlabel("Final position arm2")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "016_scatter_final_arm2_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["final_arm3"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Final Arm3 vs Power")
ax.set_xlabel("Final position arm3")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "017_scatter_final_arm3_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["move_total"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Total Movement vs Power")
ax.set_xlabel("Total movement")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "018_scatter_move_total_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["run_order"], raw["final_power_dbm"], alpha=0.3, s=10)
ax.set_title("Run Order vs Power")
ax.set_xlabel("Run order")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "019_scatter_run_order_vs_power.png")

configs = sorted(raw["config"].unique())
fig, axes = plt.subplots(5, 1, figsize=(12, 15))
for idx, fine in enumerate(fine_vals):
    subset = raw[raw["fine"] == fine]
    for coarse in coarse_vals:
        data = subset[subset["coarse"] == coarse]["final_power_dbm"]
        bp = axes[idx].boxplot([data], positions=[coarse], widths=0.6, patch_artist=True)
        bp["boxes"][0].set_facecolor(plt.cm.tab10(coarse - 1))
        bp["boxes"][0].set_alpha(0.6)
    axes[idx].set_title(f"Fine={fine}")
    axes[idx].set_ylabel("dBm")
    axes[idx].set_xticks(coarse_vals)
save(fig, "020_boxplot_power_fine_faceted.png")

fig, axes = plt.subplots(5, 1, figsize=(12, 15))
for idx, coarse in enumerate(coarse_vals):
    subset = raw[raw["coarse"] == coarse]
    for fine in fine_vals:
        data = subset[subset["fine"] == fine]["final_power_dbm"]
        bp = axes[idx].boxplot([data], positions=[fine], widths=0.6, patch_artist=True)
        bp["boxes"][0].set_facecolor(plt.cm.viridis((fine - 1) / 4))
        bp["boxes"][0].set_alpha(0.6)
    axes[idx].set_title(f"Coarse={coarse}")
    axes[idx].set_ylabel("dBm")
    axes[idx].set_xticks(fine_vals)
save(fig, "021_boxplot_power_coarse_faceted.png")

ordered_raw = raw.sort_values("run_order")
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(range(len(ordered_raw)), ordered_raw["final_power_dbm"], alpha=0.2, s=8)
ax.set_title("Power Over Run Order")
ax.set_xlabel("Trial index (sorted by run order)")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "022_timeseries_power_run_order.png")

fig, ax = plt.subplots(figsize=(12, 5))
for c in coarse_vals:
    subset = ordered_raw[ordered_raw["coarse"] == c]
    ax.scatter(subset["run_order"], subset["final_power_dbm"], alpha=0.4, s=10, label=f"C{c}")
ax.set_title("Power Over Time by Coarse")
ax.set_xlabel("Run order")
ax.set_ylabel("dBm")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "023_timeseries_power_by_coarse.png")

fig, ax = plt.subplots(figsize=(12, 5))
for c in coarse_vals:
    subset = ordered_raw[ordered_raw["coarse"] == c]
    rolling = subset["final_power_dbm"].rolling(window=20, min_periods=5, center=True).median()
    ax.plot(subset["run_order"], rolling, label=f"C{c}", linewidth=1.5)
ax.set_title("Rolling Median Power (window=20) by Coarse")
ax.set_xlabel("Run order")
ax.set_ylabel("dBm")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "024_rolling_median_power_coarse.png")

fig, ax = plt.subplots(figsize=(12, 5))
cummin = ordered_raw["final_power_dbm"].cummin()
ax.plot(ordered_raw["run_order"], cummin, color="crimson", linewidth=1.5)
ax.set_title("Cumulative Minimum Power (Best so Far)")
ax.set_xlabel("Run order")
ax.set_ylabel("dBm (lower is better)")
ax.grid(True, alpha=0.3)
save(fig, "025_cumulative_minimum_power.png")

fig, ax = plt.subplots(figsize=(12, 5))
window = max(5, len(raw) // 20)
rolling = ordered_raw["final_power_dbm"].rolling(window=window, min_periods=3, center=True).mean()
ax.plot(ordered_raw["run_order"], rolling, color="purple", linewidth=2)
ax.set_title(f"Moving Average Power (window={window})")
ax.set_xlabel("Run order")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "026_moving_average_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    ax.scatter(subset["start_arm1"], subset["final_power_dbm"], alpha=0.3, s=10, label=f"C{c}")
ax.set_title("Start Arm1 Position vs Power (colored by Coarse)")
ax.set_xlabel("Start arm1 position")
ax.set_ylabel("dBm")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "027_scatter_start_arm1_colored.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    ax.scatter(subset["start_arm2"], subset["final_power_dbm"], alpha=0.3, s=10, label=f"C{c}")
ax.set_title("Start Arm2 Position vs Power (colored by Coarse)")
ax.set_xlabel("Start arm2 position")
ax.set_ylabel("dBm")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "028_scatter_start_arm2_colored.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    ax.scatter(subset["start_arm3"], subset["final_power_dbm"], alpha=0.3, s=10, label=f"C{c}")
ax.set_title("Start Arm3 Position vs Power (colored by Coarse)")
ax.set_xlabel("Start arm3 position")
ax.set_ylabel("dBm")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "029_scatter_start_arm3_colored.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    ax.scatter(subset["final_arm1"], subset["final_power_dbm"], alpha=0.3, s=10, label=f"C{c}")
ax.set_title("Final Arm1 Position vs Power (colored by Coarse)")
ax.set_xlabel("Final arm1 position")
ax.set_ylabel("dBm")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "030_scatter_final_arm1_colored.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    ax.scatter(subset["final_arm2"], subset["final_power_dbm"], alpha=0.3, s=10, label=f"C{c}")
ax.set_title("Final Arm2 Position vs Power (colored by Coarse)")
ax.set_xlabel("Final arm2 position")
ax.set_ylabel("dBm")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "031_scatter_final_arm2_colored.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    ax.scatter(subset["final_arm3"], subset["final_power_dbm"], alpha=0.3, s=10, label=f"C{c}")
ax.set_title("Final Arm3 Position vs Power (colored by Coarse)")
ax.set_xlabel("Final arm3 position")
ax.set_ylabel("dBm")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "032_scatter_final_arm3_colored.png")

fig, ax = plt.subplots(figsize=(10, 6))
move1 = abs(raw["final_arm1"] - raw["start_arm1"])
ax.scatter(move1, raw["final_power_dbm"], alpha=0.3, s=10, c=raw["coarse"], cmap="tab10")
ax.set_title("Movement Arm1 vs Power")
ax.set_xlabel("Movement arm1")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "033_scatter_move_arm1_vs_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
move2 = abs(raw["final_arm2"] - raw["start_arm2"])
ax.scatter(move2, raw["final_power_dbm"], alpha=0.3, s=10, c=raw["coarse"], cmap="tab10")
ax.set_title("Movement Arm2 vs Power")
ax.set_xlabel("Movement arm2")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "034_scatter_move_arm2_vs_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
move3 = abs(raw["final_arm3"] - raw["start_arm3"])
ax.scatter(move3, raw["final_power_dbm"], alpha=0.3, s=10, c=raw["coarse"], cmap="tab10")
ax.set_title("Movement Arm3 vs Power")
ax.set_xlabel("Movement arm3")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "035_scatter_move_arm3_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["start_arm1"], raw["start_arm2"], c=raw["final_power_dbm"], cmap="RdYlGn_r", alpha=0.5, s=15)
ax.set_title("Start Arm1 vs Arm2 (colored by Power)")
ax.set_xlabel("Start arm1")
ax.set_ylabel("Start arm2")
plt.colorbar(ax.collections[0], label="dBm")
ax.grid(True, alpha=0.3)
save(fig, "036_scatter_start_arm1_vs_arm2.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["start_arm1"], raw["start_arm3"], c=raw["final_power_dbm"], cmap="RdYlGn_r", alpha=0.5, s=15)
ax.set_title("Start Arm1 vs Arm3 (colored by Power)")
ax.set_xlabel("Start arm1")
ax.set_ylabel("Start arm3")
plt.colorbar(ax.collections[0], label="dBm")
ax.grid(True, alpha=0.3)
save(fig, "037_scatter_start_arm1_vs_arm3.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["final_arm1"], raw["final_arm2"], c=raw["final_power_dbm"], cmap="RdYlGn_r", alpha=0.5, s=15)
ax.set_title("Final Arm1 vs Arm2 (colored by Power)")
ax.set_xlabel("Final arm1")
ax.set_ylabel("Final arm2")
plt.colorbar(ax.collections[0], label="dBm")
ax.grid(True, alpha=0.3)
save(fig, "038_scatter_final_arm1_vs_arm2.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(raw["final_arm1"], raw["final_arm3"], c=raw["final_power_dbm"], cmap="RdYlGn_r", alpha=0.5, s=15)
ax.set_title("Final Arm1 vs Arm3 (colored by Power)")
ax.set_xlabel("Final arm1")
ax.set_ylabel("Final arm3")
plt.colorbar(ax.collections[0], label="dBm")
ax.grid(True, alpha=0.3)
save(fig, "039_scatter_final_arm1_vs_arm3.png")

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.hexbin(raw["elapsed_seconds"], raw["final_power_dbm"], gridsize=25, cmap="YlOrRd", mincnt=1)
ax.set_title("Hexbin: Runtime vs Power")
ax.set_xlabel("Elapsed seconds")
ax.set_ylabel("dBm")
plt.colorbar(im, label="Count")
save(fig, "040_hexbin_runtime_vs_power.png")

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.hexbin(raw["move_total"], raw["final_power_dbm"], gridsize=25, cmap="YlOrRd", mincnt=1)
ax.set_title("Hexbin: Movement vs Power")
ax.set_xlabel("Total movement")
ax.set_ylabel("dBm")
plt.colorbar(im, label="Count")
save(fig, "041_hexbin_movement_vs_power.png")

power_median_grid = to_grid_median("final_power_dbm")
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(power_median_grid, cmap="RdYlGn_r", aspect="auto", origin="lower")
ax.set_xticks(range(len(coarse_vals)))
ax.set_xticklabels([f"C{c}" for c in coarse_vals])
ax.set_yticks(range(len(fine_vals)))
ax.set_yticklabels([f"F{f}" for f in fine_vals])
ax.set_title("Median Power Heatmap (C vs F)")
ax.set_xlabel("Coarse")
ax.set_ylabel("Fine")
for (i, j), v in np.ndenumerate(power_median_grid):
    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color="black")
plt.colorbar(im, label="dBm")
save(fig, "042_heatmap_power_mean.png")

elapsed_median_grid = to_grid_median("elapsed_seconds")
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(elapsed_median_grid, cmap="YlOrRd", aspect="auto", origin="lower")
ax.set_xticks(range(len(coarse_vals)))
ax.set_xticklabels([f"C{c}" for c in coarse_vals])
ax.set_yticks(range(len(fine_vals)))
ax.set_yticklabels([f"F{f}" for f in fine_vals])
ax.set_title("Median Runtime Heatmap (C vs F)")
ax.set_xlabel("Coarse")
ax.set_ylabel("Fine")
for (i, j), v in np.ndenumerate(elapsed_median_grid):
    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="black")
plt.colorbar(im, label="Seconds")
save(fig, "043_heatmap_runtime_mean.png")

power_std_grid = to_grid("power_std")
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(power_std_grid, cmap="Blues", aspect="auto", origin="lower")
ax.set_xticks(range(len(coarse_vals)))
ax.set_xticklabels([f"C{c}" for c in coarse_vals])
ax.set_yticks(range(len(fine_vals)))
ax.set_yticklabels([f"F{f}" for f in fine_vals])
ax.set_title("Power Std Dev Heatmap (C vs F)")
ax.set_xlabel("Coarse")
ax.set_ylabel("Fine")
for (i, j), v in np.ndenumerate(power_std_grid):
    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color="black")
plt.colorbar(im, label="dBm std")
save(fig, "044_heatmap_power_std.png")

n_trials_grid = to_grid("n_trials")
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(n_trials_grid, cmap="Greens", aspect="auto", origin="lower")
ax.set_xticks(range(len(coarse_vals)))
ax.set_xticklabels([f"C{c}" for c in coarse_vals])
ax.set_yticks(range(len(fine_vals)))
ax.set_yticklabels([f"F{f}" for f in fine_vals])
ax.set_title("Trial Count Heatmap (C vs F)")
ax.set_xlabel("Coarse")
ax.set_ylabel("Fine")
for (i, j), v in np.ndenumerate(n_trials_grid):
    ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=9, color="black")
plt.colorbar(im, label="n trials")
save(fig, "045_heatmap_trial_count.png")

success_rate_grid = np.zeros_like(power_mean_grid)
for i, f in enumerate(fine_vals):
    for j, c in enumerate(coarse_vals):
        subset = raw[(raw["fine"] == f) & (raw["coarse"] == c)]
        success_rate_grid[i, j] = (subset["final_power_dbm"] < -30).mean() * 100

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(success_rate_grid, cmap="RdYlGn", aspect="auto", origin="lower")
ax.set_xticks(range(len(coarse_vals)))
ax.set_xticklabels([f"C{c}" for c in coarse_vals])
ax.set_yticks(range(len(fine_vals)))
ax.set_yticklabels([f"F{f}" for f in fine_vals])
ax.set_title("Success Rate Heatmap (<-30 dBm)")
ax.set_xlabel("Coarse")
ax.set_ylabel("Fine")
for (i, j), v in np.ndenumerate(success_rate_grid):
    ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="black")
plt.colorbar(im, label="%")
save(fig, "046_heatmap_success_rate.png")

fig, ax = plt.subplots(figsize=(10, 6))
sorted_summary = summary.sort_values("power_mean")
colors = [plt.cm.RdYlGn_r((sorted_summary["power_mean"].max() - r["power_mean"]) / (sorted_summary["power_mean"].max() - sorted_summary["power_mean"].min())) for _, r in sorted_summary.iterrows()]
bars = ax.barh(sorted_summary["config"], sorted_summary["power_mean"], color=colors, edgecolor="black", alpha=0.8)
ax.set_xlabel("Mean Power (dBm, lower is better)")
ax.set_title("Configurations Ranked by Mean Power")
ax.axvline(x=sorted_summary["power_mean"].median(), color="gray", linestyle="--", label="Median")
ax.legend()
save(fig, "047_bar_ranking_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(summary["elapsed_mean"], summary["power_mean"], c=summary["fine"], s=100, cmap="viridis", edgecolor="black")
for _, r in summary.iterrows():
    ax.annotate(r["config"], (r["elapsed_mean"], r["power_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Mean Runtime (s)")
ax.set_ylabel("Mean Power (dBm)")
ax.set_title("Pareto: Runtime vs Power")
ax.grid(True, alpha=0.3)
best = summary.loc[summary["power_mean"].idxmin()]
ax.scatter([best["elapsed_mean"]], [best["power_mean"]], s=200, facecolors="none", edgecolors="red", linewidth=2, label="Best power")
ax.legend()
save(fig, "048_pareto_runtime_vs_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(summary["power_mean"], summary["balanced_rank_score"], s=100, c=summary["coarse"], cmap="tab10", edgecolor="black")
for _, r in summary.iterrows():
    ax.annotate(r["config"], (r["power_mean"], r["balanced_rank_score"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Mean Power (dBm)")
ax.set_ylabel("Balanced Rank Score")
ax.set_title("Balanced Rank Score vs Power")
ax.grid(True, alpha=0.3)
save(fig, "049_scatter_power_vs_rank.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(summary["power_mean"], summary["good_30_rate"], s=100, c=summary["fine"], cmap="viridis", edgecolor="black")
for _, r in summary.iterrows():
    ax.annotate(r["config"], (r["power_mean"], r["good_30_rate"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Mean Power (dBm)")
ax.set_ylabel("Good 30 Rate (<-30 dBm)")
ax.set_title("Good 30 Rate vs Mean Power")
ax.grid(True, alpha=0.3)
save(fig, "050_scatter_power_vs_good30.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(summary["power_mean"], summary["excellent_32_rate"], s=100, c=summary["fine"], cmap="viridis", edgecolor="black")
for _, r in summary.iterrows():
    ax.annotate(r["config"], (r["power_mean"], r["excellent_32_rate"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Mean Power (dBm)")
ax.set_ylabel("Excellent 32 Rate (<-32 dBm)")
ax.set_title("Excellent 32 Rate vs Mean Power")
ax.grid(True, alpha=0.3)
save(fig, "051_scatter_power_vs_excellent32.png")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for idx, trial in enumerate([1, 2, 3]):
    subset = raw[raw["trial"] == trial]
    axes[idx].boxplot([subset[subset["config"] == c]["final_power_dbm"] for c in sorted(subset["config"].unique())],
                       labels=sorted(subset["config"].unique()), patch_artist=True)
    axes[idx].set_title(f"Trial {trial} Power Distribution by Config")
    axes[idx].set_ylabel("dBm")
    axes[idx].tick_params(axis='x', rotation=45)
save(fig, "052_boxplot_trial123_power.png")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for idx, trial in enumerate([4, 5, 6]):
    subset = raw[raw["trial"] == trial]
    if len(subset) == 0:
        continue
    axes[idx].boxplot([subset[subset["config"] == c]["final_power_dbm"] for c in sorted(subset["config"].unique())],
                       labels=sorted(subset["config"].unique()), patch_artist=True)
    axes[idx].set_title(f"Trial {trial} Power Distribution by Config")
    axes[idx].set_ylabel("dBm")
    axes[idx].tick_params(axis='x', rotation=45)
save(fig, "053_boxplot_trial456_power.png")

trial_means = raw.groupby("trial")["final_power_dbm"].mean()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(trial_means.index, trial_means.values, marker="o", linewidth=1.5, markersize=4)
ax.set_title("Mean Power per Trial Over Time")
ax.set_xlabel("Trial number")
ax.set_ylabel("Mean dBm")
ax.grid(True, alpha=0.3)
save(fig, "054_line_trial_mean_power.png")

fig, ax = plt.subplots(figsize=(12, 5))
trial_stds = raw.groupby("trial")["final_power_dbm"].std()
ax.plot(trial_stds.index, trial_stds.values, marker="s", linewidth=1.5, markersize=4, color="orange")
ax.set_title("Power Std Dev per Trial Over Time")
ax.set_xlabel("Trial number")
ax.set_ylabel("dBm std")
ax.grid(True, alpha=0.3)
save(fig, "055_line_trial_std_power.png")

pivot_trial_config = raw.pivot_table(index="trial", columns="config", values="final_power_dbm", aggfunc="mean")
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(pivot_trial_config.values, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(pivot_trial_config.columns)))
ax.set_xticklabels(pivot_trial_config.columns, rotation=45, ha="right")
ax.set_yticks(range(len(pivot_trial_config.index)))
ax.set_yticklabels(pivot_trial_config.index)
ax.set_title("Mean Power: Trial vs Config")
plt.colorbar(im, label="dBm")
save(fig, "056_heatmap_trial_vs_config.png")

pivot_trial_success = raw.pivot_table(index="trial", columns="config", values="final_power_dbm", aggfunc=lambda x: (x < -30).mean() * 100)
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(pivot_trial_success.values, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(pivot_trial_success.columns)))
ax.set_xticklabels(pivot_trial_success.columns, rotation=45, ha="right")
ax.set_yticks(range(len(pivot_trial_success.index)))
ax.set_yticklabels(pivot_trial_success.index)
ax.set_title("Success Rate (<-30 dBm): Trial vs Config")
plt.colorbar(im, label="%")
save(fig, "057_heatmap_trial_vs_config_success.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, c in enumerate(coarse_vals):
    subset = raw[raw["coarse"] == c]
    axes[idx].violinplot([subset[subset["fine"] == f]["final_power_dbm"] for f in fine_vals], positions=fine_vals, showmeans=True, showmedians=True)
    axes[idx].set_title(f"C{c} Power by Fine")
    axes[idx].set_xlabel("Fine")
    axes[idx].set_ylabel("dBm")
save(fig, "058_violin_power_by_coarse.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    power_values = np.sort(subset["final_power_dbm"].values)
    cdf = np.arange(1, len(power_values) + 1) / len(power_values)
    ax.plot(power_values, cdf, label=f"C{c}", linewidth=2)
ax.set_xlabel("Power (dBm)")
ax.set_ylabel("Cumulative Probability")
ax.set_title("CDF of Power by Coarse")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "059_cdf_power_by_coarse.png")

fig, ax = plt.subplots(figsize=(10, 6))
for f in fine_vals:
    subset = raw[raw["fine"] == f]
    power_values = np.sort(subset["final_power_dbm"].values)
    cdf = np.arange(1, len(power_values) + 1) / len(power_values)
    ax.plot(power_values, cdf, label=f"F{f}", linewidth=2)
ax.set_xlabel("Power (dBm)")
ax.set_ylabel("Cumulative Probability")
ax.set_title("CDF of Power by Fine")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "060_cdf_power_by_fine.png")

for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    fig, ax = plt.subplots(figsize=(10, 6))
    for f in fine_vals:
        data = subset[subset["fine"] == f]["final_power_dbm"]
        power_values = np.sort(data.values)
        cdf = np.arange(1, len(power_values) + 1) / len(power_values)
        ax.plot(power_values, cdf, label=f"F{f}", linewidth=2)
    ax.set_xlabel("Power (dBm)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"C{c} CDF of Power by Fine")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, f"061_cdf_power_C{c}.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    percentiles = np.percentile(subset["final_power_dbm"], [10, 25, 50, 75, 90])
    ax.boxplot([percentiles], positions=[c], widths=0.5, patch_artist=True)
ax.set_title("Power Percentiles by Coarse")
ax.set_xlabel("Coarse")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "062_percentile_whisker_coarse.png")

fig, ax = plt.subplots(figsize=(10, 6))
for f in fine_vals:
    subset = raw[raw["fine"] == f]
    percentiles = np.percentile(subset["final_power_dbm"], [10, 25, 50, 75, 90])
    ax.boxplot([percentiles], positions=[f], widths=0.5, patch_artist=True)
ax.set_title("Power Percentiles by Fine")
ax.set_xlabel("Fine")
ax.set_ylabel("dBm")
ax.grid(True, alpha=0.3)
save(fig, "063_percentile_whisker_fine.png")

numeric_cols = ["coarse", "fine", "elapsed_seconds", "final_power_dbm", "run_order", "move_total", "start_spread", "final_spread"]
corr = raw[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
ax.set_yticklabels(numeric_cols)
for (i, j), v in np.ndenumerate(corr.values):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, label="Correlation")
ax.set_title("Correlation Heatmap")
save(fig, "064_correlation_heatmap.png")

configs_sorted = summary.sort_values("power_mean")["config"].tolist()
config_means = summary.set_index("config").loc[configs_sorted, "power_mean"].values.reshape(-1, 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(config_means)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(scaled, cmap="RdYlGn_r", aspect="auto")
ax.set_yticks(range(len(configs_sorted)))
ax.set_yticklabels(configs_sorted, fontsize=7)
ax.set_title("Clustered Config Heatmap (by power)")
ax.set_xticks([])
save(fig, "065_clustered_config_heatmap.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(raw["elapsed_seconds"], raw["final_power_dbm"], c=raw["coarse"], cmap="tab10", alpha=0.4, s=20)
ax.set_xlabel("Elapsed seconds")
ax.set_ylabel("dBm")
ax.set_title("2D Density: Runtime vs Power (colored by Coarse)")
ax.grid(True, alpha=0.3)
save(fig, "066_density_runtime_vs_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(raw["move_total"], raw["final_power_dbm"], c=raw["coarse"], cmap="tab10", alpha=0.4, s=20)
ax.set_xlabel("Total movement")
ax.set_ylabel("dBm")
ax.set_title("2D Density: Movement vs Power (colored by Coarse)")
ax.grid(True, alpha=0.3)
save(fig, "067_density_movement_vs_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(raw["run_order"], raw["final_power_dbm"], c=raw["coarse"], cmap="tab10", alpha=0.4, s=20)
ax.set_xlabel("Run order")
ax.set_ylabel("dBm")
ax.set_title("2D Density: Run Order vs Power (colored by Coarse)")
ax.grid(True, alpha=0.3)
save(fig, "068_density_runorder_vs_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(raw["start_spread"], raw["final_power_dbm"], alpha=0.4, s=20, c=raw["coarse"], cmap="tab10")
ax.set_xlabel("Start spread (std)")
ax.set_ylabel("dBm")
ax.set_title("Start Spread vs Power")
ax.grid(True, alpha=0.3)
save(fig, "069_scatter_start_spread_vs_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(raw["final_spread"], raw["final_power_dbm"], alpha=0.4, s=20, c=raw["coarse"], cmap="tab10")
ax.set_xlabel("Final spread (std)")
ax.set_ylabel("dBm")
ax.set_title("Final Spread vs Power")
ax.grid(True, alpha=0.3)
save(fig, "070_scatter_final_spread_vs_power.png")

raw["efficiency"] = -raw["final_power_dbm"] / raw["elapsed_seconds"]
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(raw["efficiency"], raw["final_power_dbm"], alpha=0.3, s=10, c=raw["coarse"], cmap="tab10")
ax.set_xlabel("Efficiency (|dBm|/second)")
ax.set_ylabel("dBm")
ax.set_title("Efficiency vs Power")
ax.grid(True, alpha=0.3)
save(fig, "071_scatter_efficiency_vs_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(raw["start_spread"], raw["move_total"], c=raw["final_power_dbm"], cmap="RdYlGn_r", alpha=0.5, s=20)
ax.set_xlabel("Start spread")
ax.set_ylabel("Total movement")
ax.set_title("Start Spread vs Movement (colored by Power)")
plt.colorbar(ax.collections[0], label="dBm")
ax.grid(True, alpha=0.3)
save(fig, "072_scatter_start_spread_vs_movement.png")

fig, ax = plt.subplots(figsize=(10, 6))
threshold = -30
below = raw[raw["final_power_dbm"] < threshold]
above = raw[raw["final_power_dbm"] >= threshold]
ax.scatter(above["elapsed_seconds"], above["final_power_dbm"], alpha=0.3, s=15, label=">= -30 dBm", color="red")
ax.scatter(below["elapsed_seconds"], below["final_power_dbm"], alpha=0.3, s=15, label="< -30 dBm", color="green")
ax.axhline(y=threshold, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Elapsed seconds")
ax.set_ylabel("dBm")
ax.set_title("Outlier Highlighting: Runtime vs Power")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "073_outlier_runtime_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(raw["final_power_dbm"], bins=50, color="steelblue", alpha=0.7, edgecolor="black")
ax.axvline(x=-30, color="red", linestyle="--", linewidth=2, label="-30 dBm threshold")
ax.axvline(x=-32, color="orange", linestyle="--", linewidth=2, label="-32 dBm threshold")
ax.set_xlabel("dBm")
ax.set_ylabel("Count")
ax.set_title("Power Distribution with Thresholds")
ax.legend()
save(fig, "074_hist_power_with_thresholds.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    bins = np.linspace(subset["final_power_dbm"].min(), subset["final_power_dbm"].max(), 30)
    ax.hist(subset["final_power_dbm"], bins=bins, alpha=0.5, label=f"C{c}", edgecolor="black")
ax.set_xlabel("dBm")
ax.set_ylabel("Count")
ax.set_title("Power Distribution Comparison by Coarse")
ax.legend()
save(fig, "075_hist_power_overlay_coarse.png")

fig, ax = plt.subplots(figsize=(10, 6))
for f in fine_vals:
    subset = raw[raw["fine"] == f]
    bins = np.linspace(subset["final_power_dbm"].min(), subset["final_power_dbm"].max(), 30)
    ax.hist(subset["final_power_dbm"], bins=bins, alpha=0.5, label=f"F{f}", edgecolor="black")
ax.set_xlabel("dBm")
ax.set_ylabel("Count")
ax.set_title("Power Distribution Comparison by Fine")
ax.legend()
save(fig, "076_hist_power_overlay_fine.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, c in enumerate(coarse_vals):
    subset = raw[raw["coarse"] == c]
    axes[idx].scatter(subset["start_arm1"], subset["start_arm2"], c=subset["final_power_dbm"], cmap="RdYlGn_r", alpha=0.6, s=15)
    axes[idx].set_title(f"C{c}: Start Arm1 vs Arm2")
    axes[idx].set_xlabel("Start arm1")
    axes[idx].set_ylabel("Start arm2")
    plt.colorbar(axes[idx].collections[0], ax=axes[idx], label="dBm")
save(fig, "077_ternary_start_arm1_arm2.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, c in enumerate(coarse_vals):
    subset = raw[raw["coarse"] == c]
    axes[idx].scatter(subset["final_arm1"], subset["final_arm2"], c=subset["final_power_dbm"], cmap="RdYlGn_r", alpha=0.6, s=15)
    axes[idx].set_title(f"C{c}: Final Arm1 vs Arm2")
    axes[idx].set_xlabel("Final arm1")
    axes[idx].set_ylabel("Final arm2")
    plt.colorbar(axes[idx].collections[0], ax=axes[idx], label="dBm")
save(fig, "078_ternary_final_arm1_arm2.png")

fig, ax = plt.subplots(figsize=(10, 6))
configs_all = sorted(raw["config"].unique())
config_stats = raw.groupby("config")["final_power_dbm"].agg(["mean", "std", "min", "max"])
config_stats = config_stats.loc[configs_all]
x = range(len(configs_all))
ax.errorbar(x, config_stats["mean"], yerr=config_stats["std"], fmt="o", capsize=3, markersize=5)
ax.set_xticks(x)
ax.set_xticklabels(configs_all, rotation=45, ha="right")
ax.set_ylabel("dBm")
ax.set_title("Config Summary: Mean +/- Std")
ax.grid(True, alpha=0.3)
save(fig, "079_errorbar_config_summary.png")

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(raw["elapsed_seconds"], raw["final_power_dbm"], alpha=0.2, s=8, c="0.5")
ax.set_xlabel("Elapsed seconds")
ax.set_ylabel("dBm")
ax.set_title("All Trials: Runtime vs Power")
ax.grid(True, alpha=0.3)
save(fig, "080_all_trials_runtime_power.png")

fig, ax = plt.subplots(figsize=(12, 5))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    ax.scatter(subset["move_total"], subset["elapsed_seconds"], alpha=0.3, s=10, label=f"C{c}")
ax.set_xlabel("Total movement")
ax.set_ylabel("Elapsed seconds")
ax.set_title("Movement vs Runtime by Coarse")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "081_scatter_movement_vs_runtime.png")

fig, ax = plt.subplots(figsize=(10, 6))
configs_good = summary[summary["good_30_rate"] > 0.3]["config"].tolist()
configs_bad = summary[summary["good_30_rate"] <= 0.3]["config"].tolist()
for config in configs_good:
    subset = raw[raw["config"] == config]
    ax.scatter(subset["run_order"], subset["final_power_dbm"], alpha=0.2, s=8, color="green")
for config in configs_bad:
    subset = raw[raw["config"] == config]
    ax.scatter(subset["run_order"], subset["final_power_dbm"], alpha=0.2, s=8, color="red")
ax.set_xlabel("Run order")
ax.set_ylabel("dBm")
ax.set_title("Trials by Config Quality (Green: good_30_rate > 30%, Red: <= 30%)")
ax.grid(True, alpha=0.3)
save(fig, "082_trials_colored_by_quality.png")

fig, ax = plt.subplots(figsize=(10, 6))
best_configs = summary.nsmallest(5, "power_mean")["config"].tolist()
for config in best_configs:
    subset = raw[raw["config"] == config]
    ax.scatter(subset["elapsed_seconds"], subset["final_power_dbm"], alpha=0.5, s=20, label=config)
ax.set_xlabel("Elapsed seconds")
ax.set_ylabel("dBm")
ax.set_title("Top 5 Best Configs: Runtime vs Power")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "083_top5_configs_runtime_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
fast_configs = summary.nsmallest(5, "elapsed_mean")["config"].tolist()
for config in fast_configs:
    subset = raw[raw["config"] == config]
    ax.scatter(subset["elapsed_seconds"], subset["final_power_dbm"], alpha=0.5, s=20, label=config)
ax.set_xlabel("Elapsed seconds")
ax.set_ylabel("dBm")
ax.set_title("Top 5 Fastest Configs: Runtime vs Power")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "084_top5_fast_configs_runtime_power.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    ax.scatter(subset["start_spread"], subset["final_power_dbm"], alpha=0.4, s=15, label=f"C{c}")
ax.set_xlabel("Start spread (std)")
ax.set_ylabel("dBm")
ax.set_title("Start Spread vs Power by Coarse")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "085_start_spread_vs_power_coarse.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    ax.scatter(subset["final_spread"], subset["final_power_dbm"], alpha=0.4, s=15, label=f"C{c}")
ax.set_xlabel("Final spread (std)")
ax.set_ylabel("dBm")
ax.set_title("Final Spread vs Power by Coarse")
ax.legend()
ax.grid(True, alpha=0.3)
save(fig, "086_final_spread_vs_power_coarse.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, c in enumerate(coarse_vals):
    subset = raw[raw["coarse"] == c]
    axes[idx].scatter(subset["start_spread"], subset["move_total"], c=subset["final_power_dbm"], cmap="RdYlGn_r", alpha=0.5, s=15)
    axes[idx].set_title(f"C{c}: Start Spread vs Movement")
    axes[idx].set_xlabel("Start spread")
    axes[idx].set_ylabel("Total movement")
    plt.colorbar(axes[idx].collections[0], ax=axes[idx], label="dBm")
save(fig, "087_start_spread_vs_movement_coarse.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, c in enumerate(coarse_vals):
    subset = raw[raw["coarse"] == c]
    axes[idx].hist(subset["elapsed_seconds"], bins=30, alpha=0.7, color=plt.cm.tab10(idx), edgecolor="black")
    axes[idx].set_title(f"C{c}: Runtime Distribution")
    axes[idx].set_xlabel("Seconds")
    axes[idx].set_ylabel("Count")
save(fig, "088_hist_runtime_by_coarse.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, c in enumerate(coarse_vals):
    subset = raw[raw["coarse"] == c]
    data = [subset[subset["fine"] == f]["final_power_dbm"] for f in fine_vals]
    bp = axes[idx].boxplot(data, labels=[f"F{f}" for f in fine_vals], patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[idx].set_title(f"C{c}: Power by Fine")
    axes[idx].set_ylabel("dBm")
save(fig, "089_boxplot_power_by_fine_coarse.png")

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.hexbin(raw["run_order"], raw["final_power_dbm"], gridsize=30, cmap="YlOrRd", mincnt=1)
ax.set_title("Hexbin: Run Order vs Power")
ax.set_xlabel("Run order")
ax.set_ylabel("dBm")
plt.colorbar(im, label="Count")
save(fig, "090_hexbin_runorder_vs_power.png")

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.hexbin(raw["start_spread"], raw["final_spread"], gridsize=20, cmap="YlOrRd", mincnt=1)
ax.set_title("Hexbin: Start Spread vs Final Spread")
ax.set_xlabel("Start spread")
ax.set_ylabel("Final spread")
plt.colorbar(im, label="Count")
save(fig, "091_hexbin_spread_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(raw["start_arm1"], raw["final_arm1"], c=raw["final_power_dbm"], cmap="RdYlGn_r", alpha=0.4, s=15)
axes[0].set_title("Start vs Final Arm1")
axes[0].set_xlabel("Start arm1")
axes[0].set_ylabel("Final arm1")
axes[1].scatter(raw["start_arm2"], raw["final_arm2"], c=raw["final_power_dbm"], cmap="RdYlGn_r", alpha=0.4, s=15)
axes[1].set_title("Start vs Final Arm2")
axes[1].set_xlabel("Start arm2")
axes[1].set_ylabel("Final arm2")
for ax in axes:
    plt.colorbar(ax.collections[0], ax=ax, label="dBm")
save(fig, "092_start_vs_final_arm1_arm2.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(raw["start_arm3"], raw["final_arm3"], c=raw["final_power_dbm"], cmap="RdYlGn_r", alpha=0.4, s=15)
axes[0].set_title("Start vs Final Arm3")
axes[0].set_xlabel("Start arm3")
axes[0].set_ylabel("Final arm3")
move_sum = abs(raw["final_arm1"] - raw["start_arm1"]) + abs(raw["final_arm2"] - raw["start_arm2"]) + abs(raw["final_arm3"] - raw["start_arm3"])
axes[1].scatter(move_sum, raw["final_power_dbm"], c=raw["coarse"], cmap="tab10", alpha=0.4, s=15)
axes[1].set_title("Total Movement vs Power")
axes[1].set_xlabel("Sum of absolute movements")
axes[1].set_ylabel("dBm")
for ax in axes:
    ax.grid(True, alpha=0.3)
save(fig, "093_arm3_and_movement.png")

fig, ax = plt.subplots(figsize=(10, 6))
for c in coarse_vals:
    subset = raw[raw["coarse"] == c]
    density = subset["final_power_dbm"]
    ax.hist(density, bins=30, alpha=0.4, label=f"C{c}", density=True, edgecolor="black")
ax.set_xlabel("dBm")
ax.set_ylabel("Density")
ax.set_title("Power Density by Coarse (Normalized)")
ax.legend()
save(fig, "094_density_power_coarse.png")

fig, ax = plt.subplots(figsize=(10, 6))
good_trials = raw[raw["final_power_dbm"] < -30].groupby("trial").size()
bad_trials = raw[raw["final_power_dbm"] >= -30].groupby("trial").size()
x = range(1, max(len(good_trials), len(bad_trials)) + 1)
ax.bar(x, [good_trials.get(i, 0) for i in x], alpha=0.7, label="< -30 dBm", color="green")
ax.bar(x, [-bad_trials.get(i, 0) for i in x], alpha=0.7, label=">= -30 dBm", color="red")
ax.set_xlabel("Trial")
ax.set_ylabel("Count")
ax.set_title("Success/Failure Count per Trial")
ax.legend()
save(fig, "095_success_failure_trial.png")

fig, ax = plt.subplots(figsize=(10, 6))
raw["log_elapsed"] = np.log1p(raw["elapsed_seconds"])
ax.scatter(raw["log_elapsed"], raw["final_power_dbm"], c=raw["coarse"], cmap="tab10", alpha=0.4, s=15)
ax.set_xlabel("Log(Elapsed seconds + 1)")
ax.set_ylabel("dBm")
ax.set_title("Log Runtime vs Power")
ax.grid(True, alpha=0.3)
save(fig, "096_log_runtime_vs_power.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, c in enumerate(coarse_vals):
    subset = raw[raw["coarse"] == c]
    axes[idx].scatter(subset["move_total"], subset["elapsed_seconds"], c=subset["final_power_dbm"], cmap="RdYlGn_r", alpha=0.5, s=15)
    axes[idx].set_title(f"C{c}: Movement vs Runtime")
    axes[idx].set_xlabel("Total movement")
    axes[idx].set_ylabel("Seconds")
    plt.colorbar(axes[idx].collections[0], ax=axes[idx], label="dBm")
save(fig, "097_movement_vs_runtime_heatmaps.png")

fig, ax = plt.subplots(figsize=(12, 5))
power_by_trial_config = raw.pivot_table(index="trial", columns="coarse", values="final_power_dbm", aggfunc="mean")
power_by_trial_config.plot(ax=ax, marker="o", linewidth=1.5)
ax.set_title("Mean Power by Trial (by Coarse)")
ax.set_xlabel("Trial")
ax.set_ylabel("dBm")
ax.legend(title="Coarse")
ax.grid(True, alpha=0.3)
save(fig, "098_line_power_trial_coarse.png")

fig, ax = plt.subplots(figsize=(12, 5))
power_by_trial_config = raw.pivot_table(index="trial", columns="fine", values="final_power_dbm", aggfunc="mean")
power_by_trial_config.plot(ax=ax, marker="s", linewidth=1.5)
ax.set_title("Mean Power by Trial (by Fine)")
ax.set_xlabel("Trial")
ax.set_ylabel("dBm")
ax.legend(title="Fine")
ax.grid(True, alpha=0.3)
save(fig, "099_line_power_trial_fine.png")

fig, ax = plt.subplots(figsize=(10, 6))
summary_plot = summary.copy()
ax.scatter(summary_plot["elapsed_mean"], summary_plot["power_mean"], c=summary_plot["good_30_rate"], cmap="RdYlGn", s=150, edgecolor="black", alpha=0.8)
for _, r in summary_plot.iterrows():
    ax.annotate(r["config"], (r["elapsed_mean"], r["power_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Mean Runtime (s)")
ax.set_ylabel("Mean Power (dBm)")
ax.set_title("Config Summary with Good 30 Rate")
plt.colorbar(ax.collections[0], label="Good 30 Rate")
ax.grid(True, alpha=0.3)
save(fig, "100_summary_good30rate.png")

print(f"\nGenerated 100 graphs in '{OUTPUT_DIR}/' folder")
print("Graph list:")
for i in range(1, 101):
    name = f"{i:03d}_" + [
        "hist_power_by_coarse", "boxplot_power_by_coarse", "boxplot_power_by_fine",
        "boxplot_time_by_coarse", "boxplot_time_by_fine", "hist_total_movement",
        "hist_start_spread", "hist_final_spread", "scatter_coarse_vs_power",
        "scatter_fine_vs_power", "scatter_time_vs_power", "scatter_start_arm1_vs_power",
        "scatter_start_arm2_vs_power", "scatter_start_arm3_vs_power",
        "scatter_final_arm1_vs_power", "scatter_final_arm2_vs_power",
        "scatter_final_arm3_vs_power", "scatter_move_total_vs_power",
        "scatter_run_order_vs_power", "boxplot_power_fine_faceted",
        "boxplot_power_coarse_faceted", "timeseries_power_run_order",
        "timeseries_power_by_coarse", "rolling_median_power_coarse",
        "cumulative_minimum_power", "moving_average_power",
        "scatter_start_arm1_colored", "scatter_start_arm2_colored",
        "scatter_start_arm3_colored", "scatter_final_arm1_colored",
        "scatter_final_arm2_colored", "scatter_final_arm3_colored",
        "scatter_move_arm1_vs_power", "scatter_move_arm2_vs_power",
        "scatter_move_arm3_vs_power", "scatter_start_arm1_vs_arm2",
        "scatter_start_arm1_vs_arm3", "scatter_final_arm1_vs_arm2",
        "scatter_final_arm1_vs_arm3", "hexbin_runtime_vs_power",
        "hexbin_movement_vs_power", "heatmap_power_mean", "heatmap_runtime_mean",
        "heatmap_power_std", "heatmap_trial_count", "heatmap_success_rate",
        "bar_ranking_power", "pareto_runtime_vs_power", "scatter_power_vs_rank",
        "scatter_power_vs_good30", "scatter_power_vs_excellent32",
        "boxplot_trial123_power", "boxplot_trial456_power", "line_trial_mean_power",
        "line_trial_std_power", "heatmap_trial_vs_config",
        "heatmap_trial_vs_config_success", "violin_power_by_coarse",
        "cdf_power_by_coarse", "cdf_power_by_fine", "cdf_power_C1",
        "cdf_power_C2", "cdf_power_C3", "percentile_whisker_coarse",
        "percentile_whisker_fine", "correlation_heatmap", "clustered_config_heatmap",
        "density_runtime_vs_power", "density_movement_vs_power",
        "density_runorder_vs_power", "scatter_start_spread_vs_power",
        "scatter_final_spread_vs_power", "scatter_efficiency_vs_power",
        "scatter_start_spread_vs_movement", "outlier_runtime_power",
        "hist_power_with_thresholds", "hist_power_overlay_coarse",
        "hist_power_overlay_fine", "ternary_start_arm1_arm2",
        "ternary_final_arm1_arm2", "errorbar_config_summary",
        "all_trials_runtime_power", "scatter_movement_vs_runtime",
        "trials_colored_by_quality", "top5_configs_runtime_power",
        "top5_fast_configs_runtime_power", "start_spread_vs_power_coarse",
        "final_spread_vs_power_coarse", "start_spread_vs_movement_coarse",
        "hist_runtime_by_coarse", "boxplot_power_by_fine_coarse",
        "hexbin_runorder_vs_power", "hexbin_spread_comparison",
        "start_vs_final_arm1_arm2", "arm3_and_movement", "density_power_coarse",
        "success_failure_trial", "log_runtime_vs_power",
        "movement_vs_runtime_heatmaps", "line_power_trial_coarse",
        "line_power_trial_fine", "summary_good30rate"
    ][i-1] + ".png"
    print(f"  {name}")