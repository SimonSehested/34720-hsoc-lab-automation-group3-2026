import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from pathlib import Path

plt.rcParams.update({"font.size": 12, "figure.dpi": 180})

OUT_POWER = Path("output/figures/monitor2/power")
OUT_DRIFT = Path("output/figures/monitor2/drift")
OUT_INTERVALS = Path("output/figures/monitor2/intervals")
for d in (OUT_POWER, OUT_DRIFT, OUT_INTERVALS):
    d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/monitor_log2.csv", parse_dates=["timestamp"])
opt = df[df["event"] == "optimization"].copy()
opt_indices = opt.index.tolist()

# ---- Compute per-segment drift stats ----
segments = []
for i in range(len(opt_indices) - 1):
    start_pos = df.index.get_loc(opt_indices[i])
    end_pos = df.index.get_loc(opt_indices[i + 1])
    seg = df.iloc[start_pos + 1:end_pos]
    drift_seg = seg[seg["event"] == "drift"]
    if len(drift_seg) < 2:
        continue
    t_min = (drift_seg["timestamp"] - drift_seg["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
    p = drift_seg["power_dbm"].values
    slope = np.polyfit(t_min, p, 1)[0]
    duration_min = t_min[-1]
    mid_time = drift_seg["timestamp"].iloc[len(drift_seg) // 2]
    segments.append({
        "start": drift_seg["timestamp"].iloc[0],
        "mid": mid_time,
        "hour": mid_time.hour + mid_time.minute / 60.0,
        "drift_rate": slope,
        "duration_min": duration_min,
        "start_power": p[0],
        "end_power": p[-1],
        "power_change": p[-1] - p[0],
    })
seg_df = pd.DataFrame(segments)
seg_df["hour_bin"] = seg_df["mid"].dt.hour

# Compute time between optimizations
opt_times = opt["timestamp"].values
intervals_min = np.diff(opt_times) / np.timedelta64(1, "m")
interval_starts = opt["timestamp"].iloc[:-1].values

# ================================================================
# GRAPH 1: Full power time series (context)
# ================================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["timestamp"], df["power_dbm"], linewidth=0.4, color="steelblue", alpha=0.7, label="Power")
ax.scatter(opt["timestamp"], opt["power_dbm"], color="red", s=18, zorder=5, label="Optimization event")
ax.set_ylabel("Power (dBm)")
ax.set_xlabel("Time")
ax.set_title("Coupled Power Over Time")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
ax.tick_params(axis="x", rotation=25)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_POWER / "plot_01_power_timeseries.png")
plt.close()

# ================================================================
# GRAPH 2: Drift rate over time (scatter)
# ================================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.scatter(seg_df["start"], seg_df["drift_rate"], color="crimson", s=20, alpha=0.6)
ax.axhline(seg_df["drift_rate"].median(), color="blue", linestyle="--",
           label=f"Median: {seg_df['drift_rate'].median():.3f} dBm/min")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Drift Rate (dBm/min)")
ax.set_xlabel("Time")
ax.set_title("Drift Rate Between Optimizations Over Time")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
ax.tick_params(axis="x", rotation=25)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DRIFT / "plot_02_drift_rate_over_time.png")
plt.close()

# ================================================================
# GRAPH 3: Drift rate histogram
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(seg_df["drift_rate"], bins=25, color="crimson", edgecolor="white", alpha=0.8)
ax.axvline(seg_df["drift_rate"].median(), color="blue", linestyle="--", linewidth=2,
           label=f"Median: {seg_df['drift_rate'].median():.3f} dBm/min")
ax.axvline(seg_df["drift_rate"].mean(), color="orange", linestyle="--", linewidth=2,
           label=f"Mean: {seg_df['drift_rate'].mean():.3f} dBm/min")
ax.set_xlabel("Drift Rate (dBm/min)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Drift Rate Between Optimizations")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DRIFT / "plot_03_drift_rate_histogram.png")
plt.close()

# ================================================================
# GRAPH 4: Drift rate by hour of day (box plot)
# ================================================================
fig, ax = plt.subplots(figsize=(12, 6))
hours = sorted(seg_df["hour_bin"].unique())
data_by_hour = [seg_df[seg_df["hour_bin"] == h]["drift_rate"].values for h in hours]
bp = ax.boxplot(data_by_hour, positions=hours, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor="lightsalmon", alpha=0.8),
                medianprops=dict(color="darkred", linewidth=2),
                flierprops=dict(markersize=3))
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Drift Rate (dBm/min)")
ax.set_title("Drift Rate by Hour of Day")
ax.set_xticks(hours)
ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DRIFT / "plot_04_drift_rate_by_hour_boxplot.png")
plt.close()

# ================================================================
# GRAPH 5: Median drift rate per hour (bar chart)
# ================================================================
fig, ax = plt.subplots(figsize=(12, 6))
hourly_median = seg_df.groupby("hour_bin")["drift_rate"].median()
hourly_count = seg_df.groupby("hour_bin")["drift_rate"].count()
bars = ax.bar(hourly_median.index, hourly_median.values, color="coral", alpha=0.8, edgecolor="white")
ax.axhline(seg_df["drift_rate"].median(), color="blue", linestyle="--",
           label=f"Overall median: {seg_df['drift_rate'].median():.3f} dBm/min")
for h, med, n in zip(hourly_median.index, hourly_median.values, hourly_count.values):
    ax.text(h, med + 0.005, f"n={n}", ha="center", va="bottom", fontsize=8, color="gray")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Median Drift Rate (dBm/min)")
ax.set_title("Median Drift Rate by Hour of Day")
ax.set_xticks(hourly_median.index)
ax.set_xticklabels([f"{int(h):02d}:00" for h in hourly_median.index], rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DRIFT / "plot_05_median_drift_rate_by_hour.png")
plt.close()

# ================================================================
# GRAPH 6: Drift rate as scatter vs hour (continuous)
# ================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(seg_df["hour"], seg_df["drift_rate"], color="crimson", s=25, alpha=0.5)
# LOESS-style: rolling median in hour bins
hour_fine = np.linspace(seg_df["hour"].min(), seg_df["hour"].max(), 50)
rolling_med = []
bw = 1.5
for h in hour_fine:
    mask = (seg_df["hour"] >= h - bw) & (seg_df["hour"] <= h + bw)
    if mask.sum() >= 3:
        rolling_med.append(seg_df.loc[mask, "drift_rate"].median())
    else:
        rolling_med.append(np.nan)
ax.plot(hour_fine, rolling_med, color="blue", linewidth=2, label="Rolling median (3h window)")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Drift Rate (dBm/min)")
ax.set_title("Drift Rate vs. Time of Day")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DRIFT / "plot_06_drift_rate_vs_hour_scatter.png")
plt.close()

# ================================================================
# GRAPH 7: Time between optimizations over time
# ================================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(intervals_min)), intervals_min, color="teal", alpha=0.7, width=1.0)
ax.axhline(np.median(intervals_min), color="red", linestyle="--",
           label=f"Median: {np.median(intervals_min):.0f} min")
ax.axhline(np.mean(intervals_min), color="orange", linestyle="--",
           label=f"Mean: {np.mean(intervals_min):.0f} min")
ax.set_xlabel("Optimization Cycle #")
ax.set_ylabel("Time Until Next Optimization (min)")
ax.set_title("Time Between Consecutive Optimizations")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_INTERVALS / "plot_07_time_between_optimizations.png")
plt.close()

# ================================================================
# GRAPH 8: Histogram of time between optimizations
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(intervals_min, bins=30, color="teal", edgecolor="white", alpha=0.8)
ax.axvline(np.median(intervals_min), color="red", linestyle="--", linewidth=2,
           label=f"Median: {np.median(intervals_min):.0f} min")
ax.axvline(np.mean(intervals_min), color="orange", linestyle="--", linewidth=2,
           label=f"Mean: {np.mean(intervals_min):.0f} min")
ax.set_xlabel("Time Between Optimizations (min)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Time Between Optimizations")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_INTERVALS / "plot_08_time_between_optimizations_hist.png")
plt.close()

# ================================================================
# GRAPH 9: Time between optimizations by hour of day
# ================================================================
fig, ax = plt.subplots(figsize=(12, 6))
interval_hours = pd.to_datetime(interval_starts).hour
int_df = pd.DataFrame({"hour": interval_hours, "interval_min": intervals_min})
hours_i = sorted(int_df["hour"].unique())
data_by_hour_i = [int_df[int_df["hour"] == h]["interval_min"].values for h in hours_i]
bp = ax.boxplot(data_by_hour_i, positions=hours_i, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor="lightblue", alpha=0.8),
                medianprops=dict(color="darkblue", linewidth=2),
                flierprops=dict(markersize=3))
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Time Until Next Optimization (min)")
ax.set_title("Time Between Optimizations by Hour of Day")
ax.set_xticks(hours_i)
ax.set_xticklabels([f"{h:02d}:00" for h in hours_i], rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_INTERVALS / "plot_09_time_between_opt_by_hour.png")
plt.close()

# ================================================================
# GRAPH 10: Overlaid drift segments (normalized)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, max(len(opt_indices) - 1, 1)))
for i in range(len(opt_indices) - 1):
    start_pos = df.index.get_loc(opt_indices[i])
    end_pos = df.index.get_loc(opt_indices[i + 1])
    seg = df.iloc[start_pos + 1:end_pos]
    drift_seg = seg[seg["event"] == "drift"]
    if len(drift_seg) >= 2:
        t = (drift_seg["timestamp"] - drift_seg["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        p = drift_seg["power_dbm"].values - drift_seg["power_dbm"].values[0]
        ax.plot(t, p, color=colors[i], alpha=0.35, linewidth=0.8)
ax.axhline(0, color="black", linewidth=0.5)
med_rate = seg_df["drift_rate"].median()
t_line = np.linspace(0, 60, 100)
ax.plot(t_line, med_rate * t_line, color="red", linewidth=2, linestyle="--",
        label=f"Median drift rate: {med_rate:.3f} dBm/min")
ax.set_xlabel("Minutes After Optimization")
ax.set_ylabel("Power Change From Start (dBm)")
ax.set_title("Overlaid Drift Segments (Normalized to 0 dBm at Start)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DRIFT / "plot_10_overlaid_drift_segments.png")
plt.close()

print("All plots saved (plot_01 through plot_10).")
