import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_PATH = "data/benchmark_raw.csv"
OUTPUT_DIR = "output/boobs"
MAX_COARSE = 3
DPI = 200

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

raw = pd.read_csv(RAW_PATH)
raw = raw[raw["coarse"] <= MAX_COARSE].copy()

coarse_vals = sorted(raw["coarse"].unique())
fine_vals = sorted(raw["fine"].unique())

stats = raw.groupby(["coarse", "fine"]).agg(
    power_median=("final_power_dbm", "median"),
    power_q25=("final_power_dbm", lambda x: x.quantile(0.25)),
    power_q75=("final_power_dbm", lambda x: x.quantile(0.75)),
    runtime_median=("elapsed_seconds", "median"),
    runtime_q25=("elapsed_seconds", lambda x: x.quantile(0.25)),
    runtime_q75=("elapsed_seconds", lambda x: x.quantile(0.75)),
    n=("final_power_dbm", "size"),
).reset_index()


def to_grid(df, col):
    return df.pivot(index="fine", columns="coarse", values=col).loc[fine_vals, coarse_vals].values


power_med = to_grid(stats, "power_median")
power_q25 = to_grid(stats, "power_q25")
power_q75 = to_grid(stats, "power_q75")
runtime_med = to_grid(stats, "runtime_median")
runtime_q25 = to_grid(stats, "runtime_q25")
runtime_q75 = to_grid(stats, "runtime_q75")
n_grid = to_grid(stats, "n")

# --- Median Power Heatmap with IQR ---
fig, ax = plt.subplots(figsize=(8.5, 6.5))
im = ax.imshow(power_med, cmap="RdYlGn_r", aspect="auto", origin="lower")
ax.set_xticks(range(len(coarse_vals)))
ax.set_xticklabels([f"C{c}" for c in coarse_vals])
ax.set_yticks(range(len(fine_vals)))
ax.set_yticklabels([f"F{f}" for f in fine_vals])
ax.set_xlabel("Coarse")
ax.set_ylabel("Fine")
ax.set_title("Median Power Heatmap (C vs F)")

finite = power_med[np.isfinite(power_med)]
mid = (finite.min() + finite.max()) / 2

for (i, j), med_val in np.ndenumerate(power_med):
    q25 = power_q25[i, j]
    q75 = power_q75[i, j]
    n = int(n_grid[i, j])
    text_color = "white" if med_val > mid else "black"
    ax.text(j, i, f"{med_val:.1f}", ha="center", va="center",
            fontsize=12, fontweight="bold", color=text_color)
    ax.text(j, i - 0.28, f"[{q25:.1f}, {q75:.1f}]", ha="center", va="center",
            fontsize=7.5, color=text_color, alpha=0.85)
    ax.text(j, i + 0.30, f"n={n}", ha="center", va="center",
            fontsize=7, color=text_color, alpha=0.7)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("dBm")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/heatmap_power_median.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_DIR}/heatmap_power_median.png")

# --- Median Runtime Heatmap with IQR ---
fig, ax = plt.subplots(figsize=(8.5, 6.5))
im = ax.imshow(runtime_med, cmap="YlOrRd", aspect="auto", origin="lower")
ax.set_xticks(range(len(coarse_vals)))
ax.set_xticklabels([f"C{c}" for c in coarse_vals])
ax.set_yticks(range(len(fine_vals)))
ax.set_yticklabels([f"F{f}" for f in fine_vals])
ax.set_xlabel("Coarse")
ax.set_ylabel("Fine")
ax.set_title("Median Runtime Heatmap (C vs F)")

finite_rt = runtime_med[np.isfinite(runtime_med)]
mid_rt = (finite_rt.min() + finite_rt.max()) / 2

for (i, j), med_val in np.ndenumerate(runtime_med):
    q25 = runtime_q25[i, j]
    q75 = runtime_q75[i, j]
    n = int(n_grid[i, j])
    text_color = "white" if med_val > mid_rt else "black"
    ax.text(j, i, f"{med_val:.0f}", ha="center", va="center",
            fontsize=12, fontweight="bold", color=text_color)
    ax.text(j, i - 0.28, f"[{q25:.0f}, {q75:.0f}]", ha="center", va="center",
            fontsize=7.5, color=text_color, alpha=0.85)
    ax.text(j, i + 0.30, f"n={n}", ha="center", va="center",
            fontsize=7, color=text_color, alpha=0.7)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Seconds")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/heatmap_runtime_median.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_DIR}/heatmap_runtime_median.png")
