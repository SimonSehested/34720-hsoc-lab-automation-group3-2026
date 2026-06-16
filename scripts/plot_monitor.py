import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("monitor_log.csv", parse_dates=["timestamp"])
drift = df[df["event"] == "drift"].copy()
opt   = df[df["event"] == "optimization"].copy()

print(f"Rows: {len(df)}  |  Drift: {len(drift)}  |  Optimizations: {len(opt)}")
print(f"Period: {df['timestamp'].min()} til {df['timestamp'].max()}")

# ── 1. Power over time – all events ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(drift["timestamp"], drift["power_dbm"], lw=0.7, color="steelblue", label="Drift")
ax.scatter(opt["timestamp"], opt["power_dbm"], color="red", s=30, zorder=5, label="Optimization")
ax.set_title("1. Power over time (all events)")
ax.set_xlabel("Tid"); ax.set_ylabel("Power (dBm)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "01_power_over_time.png")
plt.close()

# ── 2. Drift power over time with rolling mean ────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(drift["timestamp"], drift["power_dbm"], lw=0.5, alpha=0.5, color="steelblue", label="Rådata")
roll = drift.set_index("timestamp")["power_dbm"].rolling("60min").mean()
ax.plot(roll, lw=1.5, color="navy", label="60-min rullende gennemsnit")
ax.set_title("2. Drift power over tid med rullende gennemsnit")
ax.set_xlabel("Tid"); ax.set_ylabel("Power (dBm)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "02_drift_power_rolling.png")
plt.close()

# ── 3. Optimization power over time ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(opt["timestamp"], opt["power_dbm"], "o-", color="crimson", ms=4)
ax.set_title("3. Optimization power over tid")
ax.set_xlabel("Tid"); ax.set_ylabel("Power (dBm)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "03_optimization_power.png")
plt.close()

# ── 4. Histogram: drift power ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(drift["power_dbm"], bins=60, color="steelblue", edgecolor="white", lw=0.3)
ax.axvline(drift["power_dbm"].mean(), color="red", lw=1.5, label=f"Middelværdi {drift['power_dbm'].mean():.1f} dBm")
ax.set_title("4. Fordeling af drift power")
ax.set_xlabel("Power (dBm)"); ax.set_ylabel("Antal målinger")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "04_drift_power_histogram.png")
plt.close()

# ── 5. Histogram: optimization power ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(opt["power_dbm"], bins=30, color="crimson", edgecolor="white", lw=0.3)
ax.axvline(opt["power_dbm"].mean(), color="navy", lw=1.5, label=f"Middelværdi {opt['power_dbm'].mean():.1f} dBm")
ax.set_title("5. Fordeling af optimization power")
ax.set_xlabel("Power (dBm)"); ax.set_ylabel("Antal")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "05_optimization_power_histogram.png")
plt.close()

# ── 6. Arm positions over time ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
colors = ["tab:blue", "tab:orange", "tab:green"]
for i, (arm, ax) in enumerate(zip(["arm1", "arm2", "arm3"], axes)):
    ax.plot(opt["timestamp"], opt[arm], "o-", ms=3, color=colors[i], label=arm)
    ax.set_ylabel(arm)
    ax.legend(loc="upper right")
axes[-1].set_xlabel("Tid")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
fig.suptitle("6. Arm-positioner ved optimering over tid")
fig.tight_layout()
fig.savefig(OUT / "06_arm_positions_over_time.png")
plt.close()

# ── 7. Histogram: arm positions ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for arm, ax, color in zip(["arm1", "arm2", "arm3"], axes, colors):
    ax.hist(opt[arm], bins=range(0, 165, 5), color=color, edgecolor="white", lw=0.3)
    ax.set_title(f"Fordeling {arm}")
    ax.set_xlabel("Trin (0-160)"); ax.set_ylabel("Antal")
fig.suptitle("7. Histogram over arm-positioner (alle optimeringer)")
fig.tight_layout()
fig.savefig(OUT / "07_arm_histograms.png")
plt.close()

# ── 8. Optimization duration histogram ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(opt["duration_s"].dropna(), bins=30, color="darkorange", edgecolor="white", lw=0.3)
ax.axvline(opt["duration_s"].mean(), color="red", lw=1.5, label=f"Middel {opt['duration_s'].mean():.1f} s")
ax.set_title("8. Fordeling af optimeringstid (duration)")
ax.set_xlabel("Varighed (s)"); ax.set_ylabel("Antal")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "08_duration_histogram.png")
plt.close()

# ── 9. Optimization duration over time ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(opt["timestamp"], opt["duration_s"], "s-", color="darkorange", ms=3)
ax.set_title("9. Optimeringstid over tid")
ax.set_xlabel("Tid"); ax.set_ylabel("Varighed (s)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "09_duration_over_time.png")
plt.close()

# ── 10. Time between optimizations ───────────────────────────────────────────
opt_sorted = opt.sort_values("timestamp").copy()
opt_sorted["gap_min"] = opt_sorted["timestamp"].diff().dt.total_seconds() / 60
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(opt_sorted["timestamp"].iloc[1:], opt_sorted["gap_min"].iloc[1:],
       width=0.005, color="purple", alpha=0.7)
ax.set_title("10. Tid mellem optimeringer")
ax.set_xlabel("Tid"); ax.set_ylabel("Gap (min)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "10_time_between_optimizations.png")
plt.close()

# ── 11. Power delta: before vs after optimization ─────────────────────────────
# For each optimization, find drift power 5 min before and 5 min after
deltas_before = []
deltas_after  = []
timestamps    = []
for _, row in opt_sorted.iterrows():
    t = row["timestamp"]
    window_before = drift[(drift["timestamp"] >= t - pd.Timedelta("5min")) &
                          (drift["timestamp"] <  t)]
    window_after  = drift[(drift["timestamp"] >  t) &
                          (drift["timestamp"] <= t + pd.Timedelta("5min"))]
    if not window_before.empty and not window_after.empty:
        deltas_before.append(window_before["power_dbm"].mean())
        deltas_after.append(window_after["power_dbm"].mean())
        timestamps.append(t)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(timestamps, deltas_before, "^--", color="steelblue", ms=4, label="Drift 5 min FØR")
ax.plot(timestamps, deltas_after,  "v-",  color="seagreen",  ms=4, label="Drift 5 min EFTER")
ax.set_title("11. Gennemsnitlig drift power 5 min før og efter optimering")
ax.set_xlabel("Tid"); ax.set_ylabel("Power (dBm)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "11_power_before_after_optimization.png")
plt.close()

# ── 12. Scatter: arm1 vs arm2, arm2 vs arm3, arm1 vs arm3 ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
pairs = [("arm1", "arm2"), ("arm2", "arm3"), ("arm1", "arm3")]
for (x, y), ax in zip(pairs, axes):
    sc = ax.scatter(opt[x], opt[y], c=opt["power_dbm"], cmap="RdYlGn_r",
                    s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="dBm")
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
fig.suptitle("12. Arm-positioner farvet efter power (grøn = bedre)")
fig.tight_layout()
fig.savefig(OUT / "12_arm_scatter.png")
plt.close()

# ── 13. Cumulative optimizations per day ─────────────────────────────────────
opt["date"] = opt["timestamp"].dt.date
per_day = opt.groupby("date").size()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([str(d) for d in per_day.index], per_day.values, color="teal")
ax.set_title("13. Antal optimeringer pr. dag")
ax.set_xlabel("Dato"); ax.set_ylabel("Antal")
fig.tight_layout()
fig.savefig(OUT / "13_optimizations_per_day.png")
plt.close()

# ── 14. Box plot: arm positions ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([opt["arm1"], opt["arm2"], opt["arm3"]],
           labels=["Arm 1", "Arm 2", "Arm 3"],
           patch_artist=True,
           boxprops=dict(facecolor="lightblue"))
ax.set_title("14. Boksplot over arm-positioner")
ax.set_ylabel("Trin (0-160)")
fig.tight_layout()
fig.savefig(OUT / "14_arm_boxplot.png")
plt.close()

# ── 15. Heatmap hour-of-day vs day-of-week for optimizations ─────────────────
opt["hour"] = opt["timestamp"].dt.hour
opt["dow"]  = opt["timestamp"].dt.day_name()
pivot = opt.pivot_table(index="hour", columns="dow", values="power_dbm",
                        aggfunc="mean")
dow_order = ["Friday","Saturday","Sunday","Monday","Tuesday","Wednesday","Thursday"]
pivot = pivot.reindex(columns=[c for c in dow_order if c in pivot.columns])
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", origin="lower")
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30)
ax.set_yticks(range(len(pivot.index)));  ax.set_yticklabels(pivot.index)
plt.colorbar(im, ax=ax, label="Gns. power (dBm)")
ax.set_title("15. Gns. optimization power pr. time og ugedag")
ax.set_xlabel("Dag"); ax.set_ylabel("Time")
fig.tight_layout()
fig.savefig(OUT / "15_heatmap_hour_day.png")
plt.close()

print(f"\nAlle grafer gemt i '{OUT}/'")
for p in sorted(OUT.glob("*.png")):
    print(f"  {p.name}")
