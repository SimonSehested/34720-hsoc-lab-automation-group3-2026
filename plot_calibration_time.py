import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

df  = pd.read_csv("monitor_log.csv", parse_dates=["timestamp"])
opt = df[df["event"] == "optimization"].copy().sort_values("timestamp").reset_index(drop=True)
drift = df[df["event"] == "drift"].copy().reset_index(drop=True)

dur = opt["duration_s"].dropna()
print(f"Optimeringer med duration: {len(dur)}")
print(f"Gennemsnit:  {dur.mean():.1f} s")
print(f"Median:      {dur.median():.1f} s")
print(f"Std:         {dur.std():.1f} s")
print(f"Min / Max:   {dur.min():.1f} / {dur.max():.1f} s")

# Tid til naeste kalibrering (cyklus-laengde)
opt["next_t"] = opt["timestamp"].shift(-1)
opt["gap_min"] = (opt["next_t"] - opt["timestamp"]).dt.total_seconds() / 60

# ── 1. Varighed over tid ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(opt["timestamp"], opt["duration_s"], width=pd.Timedelta("6min"),
       color="steelblue", alpha=0.85, label="Varighed (s)")
ax.axhline(dur.mean(),   color="red",    lw=1.5, linestyle="--",
           label=f"Gennemsnit {dur.mean():.1f} s")
ax.axhline(dur.median(), color="orange", lw=1.5, linestyle=":",
           label=f"Median {dur.median():.1f} s")
ax.set_title("1. Kalibrerings­varighed over tid")
ax.set_xlabel("Tidspunkt"); ax.set_ylabel("Varighed (s)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "cal_01_duration_over_time.png")
plt.close()

# ── 2. Histogram ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.arange(dur.min() - 1, dur.max() + 3, 2)
ax.hist(dur, bins=bins, color="steelblue", edgecolor="white", lw=0.4)
ax.axvline(dur.mean(),   color="red",    lw=1.5, linestyle="--",
           label=f"Gennemsnit {dur.mean():.1f} s")
ax.axvline(dur.median(), color="orange", lw=1.5, linestyle=":",
           label=f"Median {dur.median():.1f} s")
ax.set_title("2. Fordeling af kalibreringsvarighed")
ax.set_xlabel("Varighed (s)"); ax.set_ylabel("Antal")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "cal_02_duration_histogram.png")
plt.close()

# ── 3. Kumulativ tid brugt paa kalibrering ────────────────────────────────────
opt_s = opt.dropna(subset=["duration_s"]).copy()
opt_s = opt_s.sort_values("timestamp")
opt_s["cumul_min"] = opt_s["duration_s"].cumsum() / 60

total_span_min = (opt_s["timestamp"].max() - opt_s["timestamp"].min()).total_seconds() / 60
total_cal_min  = opt_s["duration_s"].sum() / 60
pct = 100 * total_cal_min / total_span_min

fig, ax = plt.subplots(figsize=(13, 4))
ax.step(opt_s["timestamp"], opt_s["cumul_min"], where="post",
        color="darkorange", lw=2)
ax.fill_between(opt_s["timestamp"], opt_s["cumul_min"],
                step="post", alpha=0.2, color="darkorange")
ax.set_title(f"3. Kumulativ tid brugt paa kalibreringer  "
             f"(total {total_cal_min:.1f} min = {pct:.1f}% af logperioden)")
ax.set_xlabel("Tidspunkt"); ax.set_ylabel("Akkumuleret kalibreringstid (min)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "cal_03_cumulative_time.png")
plt.close()

# ── 4. Varighed vs opnaaet power ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(opt_s["duration_s"], opt_s["power_dbm"],
                c=mdates.date2num(opt_s["timestamp"]), cmap="viridis",
                s=40, alpha=0.8)
plt.colorbar(sc, ax=ax, label="Tidspunkt")
z = np.polyfit(opt_s["duration_s"], opt_s["power_dbm"], 1)
x_fit = np.linspace(opt_s["duration_s"].min(), opt_s["duration_s"].max(), 100)
ax.plot(x_fit, np.polyval(z, x_fit), "r--", lw=1.5,
        label=f"Fit: {z[0]:.3f}x + {z[1]:.1f}")
ax.set_title("4. Varighed vs. opnaaet power\n(kortere = bedre kalibrering?)")
ax.set_xlabel("Varighed (s)"); ax.set_ylabel("Power ved optimering (dBm)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "cal_04_duration_vs_power.png")
plt.close()

# ── 5. Varighed vs tid til naeste kalibrering ─────────────────────────────────
sub = opt_s.dropna(subset=["gap_min"])
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(sub["duration_s"], sub["gap_min"],
                c=mdates.date2num(sub["timestamp"]), cmap="plasma",
                s=40, alpha=0.8)
plt.colorbar(sc, ax=ax, label="Tidspunkt")
z2 = np.polyfit(sub["duration_s"], sub["gap_min"], 1)
x2 = np.linspace(sub["duration_s"].min(), sub["duration_s"].max(), 100)
ax.plot(x2, np.polyval(z2, x2), "r--", lw=1.5,
        label=f"Fit: {z2[0]:.2f}x + {z2[1]:.1f}")
ax.set_title("5. Laengere kalibrering = laengere til naeste?\n"
             "(varighed vs. minutter til naeste optimering)")
ax.set_xlabel("Varighed (s)"); ax.set_ylabel("Minutter til naeste kalibrering")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "cal_05_duration_vs_gap.png")
plt.close()

# ── 6. Kalibrerings­varighed pr. time paa dagen ───────────────────────────────
opt_s["hour"] = opt_s["timestamp"].dt.hour
grp = opt_s.groupby("hour")["duration_s"].agg(["mean", "std", "count"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(grp.index, grp["mean"], color="teal", alpha=0.85)
axes[0].errorbar(grp.index, grp["mean"], yerr=grp["std"],
                 fmt="none", color="black", capsize=3, lw=1)
axes[0].set_title("Gns. varighed pr. time (med std)")
axes[0].set_xlabel("Time paa dagen"); axes[0].set_ylabel("Varighed (s)")
axes[0].set_xticks(range(0, 24, 2))

axes[1].bar(grp.index, grp["count"], color="slateblue", alpha=0.85)
axes[1].set_title("Antal kalibreringer pr. time")
axes[1].set_xlabel("Time paa dagen"); axes[1].set_ylabel("Antal")
axes[1].set_xticks(range(0, 24, 2))

fig.suptitle("6. Kalibreringsadfaerd fordelt paa time af dagen")
fig.tight_layout()
fig.savefig(OUT / "cal_06_duration_by_hour.png")
plt.close()

# ── 7. Kort vs lang kalibrering – drift efterfoelgende ───────────────────────
# Del i korte (under median) og lange (over median)
median_dur = opt_s["duration_s"].median()
short_opts = opt_s[opt_s["duration_s"] <= median_dur]
long_opts  = opt_s[opt_s["duration_s"] >  median_dur]

def mean_subsequent_drift(opt_subset, drift_df, minutes=60, n_points=60):
    all_curves = []
    t_axis = np.linspace(0, minutes, n_points)
    for _, row in opt_subset.iterrows():
        t0 = row["timestamp"]
        seg = drift_df[(drift_df["timestamp"] > t0) &
                       (drift_df["timestamp"] <= t0 + pd.Timedelta(f"{minutes}min"))].copy()
        if len(seg) < 5:
            continue
        seg["min_since"] = (seg["timestamp"] - t0).dt.total_seconds() / 60
        interp = np.interp(t_axis, seg["min_since"].values, seg["power_dbm"].values,
                           left=np.nan, right=np.nan)
        all_curves.append(interp)
    if not all_curves:
        return t_axis, np.full(n_points, np.nan)
    return t_axis, np.nanmean(all_curves, axis=0)

t_ax, short_curve = mean_subsequent_drift(short_opts, drift)
_,    long_curve  = mean_subsequent_drift(long_opts,  drift)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_ax, short_curve, lw=2, color="steelblue",
        label=f"Kort kalibrering (<= {median_dur:.0f} s, n={len(short_opts)})")
ax.plot(t_ax, long_curve,  lw=2, color="crimson",
        label=f"Lang kalibrering (> {median_dur:.0f} s, n={len(long_opts)})")
ax.set_title("7. Efterfoelgende drift: kort vs. lang kalibrering\n"
             "(gns. driftforloeb de foelgende 60 min)")
ax.set_xlabel("Minutter efter kalibrering"); ax.set_ylabel("Power (dBm)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "cal_07_short_vs_long_drift.png")
plt.close()

print("\nKalibrerings­tids-grafer gemt:")
for p in sorted(OUT.glob("cal_0*.png")):
    print(f"  {p.name}")
