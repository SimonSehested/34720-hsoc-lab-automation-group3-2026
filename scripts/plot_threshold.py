import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

THRESHOLD = -20.0   # dBm

df    = pd.read_csv("monitor_log.csv", parse_dates=["timestamp"])
drift = df[df["event"] == "drift"].copy().sort_values("timestamp").reset_index(drop=True)
opt   = df[df["event"] == "optimization"].copy().sort_values("timestamp").reset_index(drop=True)

below = drift["power_dbm"] < THRESHOLD
pct_below = below.mean() * 100
print(f"Andel maalinger under {THRESHOLD} dBm: {pct_below:.1f} %")

# ── 1. Tidsserie med taerskel markeret ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(drift["timestamp"], drift["power_dbm"],
        lw=0.6, color="steelblue", label="Drift power")
ax.axhline(THRESHOLD, color="red", lw=1.5, linestyle="--",
           label=f"Taerskel {THRESHOLD} dBm")
ax.fill_between(drift["timestamp"], drift["power_dbm"], THRESHOLD,
                where=drift["power_dbm"] < THRESHOLD,
                alpha=0.25, color="tomato", label=f"Under {THRESHOLD} dBm")
ax.scatter(opt["timestamp"], opt["power_dbm"], color="black", s=20, zorder=5,
           label="Kalibrering")
ax.set_title(f"1. Power over tid med taerskel {THRESHOLD} dBm\n"
             f"(rod omraade = under taerskel, {pct_below:.1f}% af maalingerne)")
ax.set_xlabel("Tid"); ax.set_ylabel("Power (dBm)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax.legend(fontsize=8, ncol=2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "thr_01_timeseries_threshold.png")
plt.close()

# ── 2. Sammenhængende perioder under taerskel ────────────────────────────────
# Find runs af konsekutive maalinger under taersklen
drift["below"] = drift["power_dbm"] < THRESHOLD
drift["run_id"] = (drift["below"] != drift["below"].shift()).cumsum()
runs = drift[drift["below"]].groupby("run_id").agg(
    t_start=("timestamp", "first"),
    t_end=("timestamp",   "last"),
    n=("power_dbm",       "count"),
).reset_index(drop=True)
runs["duration_min"] = (runs["t_end"] - runs["t_start"]).dt.total_seconds() / 60

print(f"Sammenhængende perioder under {THRESHOLD} dBm: {len(runs)}")
print(f"Median laengde: {runs['duration_min'].median():.1f} min")
print(f"Max laengde:    {runs['duration_min'].max():.1f} min")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(runs["duration_min"], bins=30, color="tomato", edgecolor="white", lw=0.3)
axes[0].axvline(10, color="navy", lw=2, linestyle="--", label="10 min")
axes[0].axvline(runs["duration_min"].median(), color="orange", lw=1.5, linestyle=":",
                label=f"Median {runs['duration_min'].median():.0f} min")
axes[0].set_title(f"Fordeling af perioder under {THRESHOLD} dBm")
axes[0].set_xlabel("Varighed (min)"); axes[0].set_ylabel("Antal perioder")
axes[0].legend()

# Cumulativ fordeling
sorted_dur = np.sort(runs["duration_min"])
cdf = np.arange(1, len(sorted_dur)+1) / len(sorted_dur)
axes[1].plot(sorted_dur, cdf * 100, color="tomato", lw=2)
axes[1].axvline(10, color="navy", lw=2, linestyle="--", label="10 min")
p10 = np.interp(10, sorted_dur, cdf) * 100
axes[1].axhline(p10, color="navy", lw=1, linestyle=":")
axes[1].scatter([10], [p10], color="navy", s=60, zorder=5)
axes[1].annotate(f"{p10:.0f}% af perioderne\nvarer >= 10 min",
                 xy=(10, p10), xytext=(40, p10 - 20),
                 arrowprops=dict(arrowstyle="->", color="navy"), fontsize=9)
axes[1].set_title(f"CDF: andel af perioder kortere end T min")
axes[1].set_xlabel("Varighed (min)"); axes[1].set_ylabel("Kumulativ andel (%)")
axes[1].legend()

fig.suptitle(f"2. Laengde af sammenhængende perioder under {THRESHOLD} dBm")
fig.tight_layout()
fig.savefig(OUT / "thr_02_run_lengths.png")
plt.close()

# ── 3. Overlevelseskurve – P(holder sig under taerskel i >= T min) ────────────
window_sizes = np.arange(1, 120, 1)  # 1–119 min

# Brug sliding window: for hvert tidspunkt, er de naeste T min alle under taersklen?
# Effektivt: for hvert run, bidrager det til P(>=T) for alle T <= run_laengde
prob = np.zeros(len(window_sizes))
for i, T in enumerate(window_sizes):
    # Andel af runs der varer mindst T min
    prob[i] = (runs["duration_min"] >= T).mean() * 100

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(window_sizes, prob, lw=2.5, color="steelblue")
ax.axvline(10, color="red", lw=2, linestyle="--", label="10 min")
p10_surv = np.interp(10, window_sizes, prob)
ax.axhline(p10_surv, color="red", lw=1, linestyle=":")
ax.scatter([10], [p10_surv], color="red", s=80, zorder=5)
ax.annotate(f"  P(>= 10 min) = {p10_surv:.0f}%",
            xy=(10, p10_surv), xytext=(25, p10_surv + 5), fontsize=11,
            color="red", fontweight="bold")

# Ekstra markeringer
for t_mark in [30, 60, 90]:
    p = np.interp(t_mark, window_sizes, prob)
    ax.scatter([t_mark], [p], color="gray", s=40, zorder=4)
    ax.annotate(f"{t_mark} min\n{p:.0f}%", xy=(t_mark, p),
                xytext=(t_mark + 3, p + 3), fontsize=8, color="gray")

ax.set_title(f"3. Overlevelseskurve: P(forbliver under {THRESHOLD} dBm i mindst T minutter)\n"
             "(beregnet fra faktiske sammenhængende perioder under taersklen)")
ax.set_xlabel("T (minutter)"); ax.set_ylabel("Sandsynlighed (%)")
ax.set_ylim(0, 105)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "thr_03_survival_curve.png")
plt.close()

# ── 4. Sliding-window sandsynlighed ──────────────────────────────────────────
# For hvert 10-min vindue i tidsserien: er ALLE maelinger under taersklen?
results = []
window_td = pd.Timedelta("10min")
for i, row in drift.iterrows():
    t0 = row["timestamp"]
    window = drift[(drift["timestamp"] >= t0) & (drift["timestamp"] < t0 + window_td)]
    if len(window) < 5:
        continue
    all_below = (window["power_dbm"] < THRESHOLD).all()
    results.append({"timestamp": t0, "all_below": all_below})

res_df = pd.DataFrame(results)
overall_prob = res_df["all_below"].mean() * 100
print(f"Sandsynlighed (sliding 10-min vindue, alle maalinger under {THRESHOLD}): "
      f"{overall_prob:.1f} %")

# Rullende sandsynlighed per time
res_df = res_df.set_index("timestamp")
rolling_prob = res_df["all_below"].astype(float).rolling("3h").mean() * 100

fig, ax = plt.subplots(figsize=(13, 4))
ax.fill_between(rolling_prob.index, rolling_prob.values,
                alpha=0.4, color="steelblue")
ax.plot(rolling_prob.index, rolling_prob.values, lw=1.2, color="steelblue")
ax.axhline(overall_prob, color="red", lw=1.5, linestyle="--",
           label=f"Samlet gns: {overall_prob:.0f}%")
ax.set_title(f"4. Rullende sandsynlighed (3-timers vindue) for at et 10-min interval\n"
             f"holder sig helt under {THRESHOLD} dBm  (samlet: {overall_prob:.1f}%)")
ax.set_xlabel("Tid"); ax.set_ylabel("Sandsynlighed (%)")
ax.set_ylim(0, 105)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "thr_04_rolling_probability.png")
plt.close()

print("\nTaerskel-grafer gemt:")
for p in sorted(OUT.glob("thr_0*.png")):
    print(f"  {p.name}")
