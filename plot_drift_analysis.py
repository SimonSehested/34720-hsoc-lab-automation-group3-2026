import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("monitor_log.csv", parse_dates=["timestamp"])
drift = df[df["event"] == "drift"].copy().reset_index(drop=True)
opt   = df[df["event"] == "optimization"].copy().sort_values("timestamp").reset_index(drop=True)

# ── Byg cykler: fra én optimering til den næste ──────────────────────────────
cycles = []
for i in range(len(opt) - 1):
    t_start = opt.loc[i,   "timestamp"]
    t_end   = opt.loc[i+1, "timestamp"]
    seg = drift[(drift["timestamp"] > t_start) & (drift["timestamp"] < t_end)].copy()
    if len(seg) < 3:
        continue
    seg["minutes_since_opt"] = (seg["timestamp"] - t_start).dt.total_seconds() / 60
    cycles.append({
        "seg":        seg,
        "t_start":    t_start,
        "t_end":      t_end,
        "duration_min": (t_end - t_start).total_seconds() / 60,
        "pwr_start":  seg["power_dbm"].iloc[0],
        "pwr_end":    seg["power_dbm"].iloc[-1],
        "pwr_min":    seg["power_dbm"].min(),
        "pwr_max":    seg["power_dbm"].max(),
        "drop_dbm":   seg["power_dbm"].iloc[0] - seg["power_dbm"].iloc[-1],
        "idx":        i,
    })

dur   = np.array([c["duration_min"] for c in cycles])
drop  = np.array([c["drop_dbm"]     for c in cycles])
pend  = np.array([c["pwr_end"]      for c in cycles])

print(f"Analyserede {len(cycles)} kalibreringscykler")
print(f"Gns. tid mellem kalibreringer: {dur.mean():.1f} min  (std {dur.std():.1f} min)")
print(f"Min/max: {dur.min():.1f} / {dur.max():.1f} min")

# ── A. Overlappede driftkurver ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
cmap = plt.cm.plasma
for c in cycles:
    color = cmap(c["idx"] / len(cycles))
    ax.plot(c["seg"]["minutes_since_opt"], c["seg"]["power_dbm"],
            lw=0.8, alpha=0.5, color=color)

# Mediankurve
all_times  = np.linspace(0, 90, 200)
all_interp = []
for c in cycles:
    if c["duration_min"] > 15:
        t = c["seg"]["minutes_since_opt"].values
        p = c["seg"]["power_dbm"].values
        interp = np.interp(all_times, t, p, left=np.nan, right=np.nan)
        all_interp.append(interp)
med = np.nanmedian(all_interp, axis=0)
ax.plot(all_times, med, "k-", lw=2.5, label="Median drift-kurve", zorder=5)

ax.set_title("A. Overlappede driftkurver – fra optimering til naeste kalibrering\n"
             "(farve = tid siden start, lys = nyest)")
ax.set_xlabel("Minutter siden optimering")
ax.set_ylabel("Power (dBm)")
ax.legend()
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Cyklus (tidlig -> sen)")
fig.tight_layout()
fig.savefig(OUT / "A_drift_overlay.png")
plt.close()

# ── B. Tid mellem kalibreringer over tid ─────────────────────────────────────
t_starts = [c["t_start"] for c in cycles]
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(t_starts, dur, width=pd.Timedelta("8min"), color="steelblue", alpha=0.8)
ax.axhline(dur.mean(), color="red", lw=1.5, linestyle="--",
           label=f"Gennemsnit {dur.mean():.0f} min")
ax.set_title("B. Tid mellem kalibreringer over tid")
ax.set_xlabel("Tidspunkt for optimering")
ax.set_ylabel("Minutter til naeste kalibrering")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "B_time_between_calibrations.png")
plt.close()

# ── C. Histogram: tid mellem kalibreringer ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(dur, bins=25, color="steelblue", edgecolor="white", lw=0.3)
ax.axvline(dur.mean(), color="red", lw=1.5, linestyle="--",
           label=f"Gennemsnit {dur.mean():.0f} min")
ax.axvline(np.median(dur), color="orange", lw=1.5, linestyle=":",
           label=f"Median {np.median(dur):.0f} min")
ax.set_title("C. Fordeling: minutter mellem kalibreringer")
ax.set_xlabel("Minutter"); ax.set_ylabel("Antal cykler")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "C_calibration_interval_histogram.png")
plt.close()

# ── D. Drifthastighed (dBm/min) per cyklus ───────────────────────────────────
# Beregn lineaer regression for drift-raten i hvert segment
rates = []
for c in cycles:
    t = c["seg"]["minutes_since_opt"].values
    p = c["seg"]["power_dbm"].values
    if len(t) < 5:
        rates.append(np.nan)
        continue
    coeffs = np.polyfit(t, p, 1)   # slope = dBm/min
    rates.append(coeffs[0])

rates = np.array(rates)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
# Over tid
axes[0].bar(t_starts, rates, width=pd.Timedelta("8min"),
            color=["seagreen" if r > 0 else "tomato" for r in rates], alpha=0.85)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].axhline(np.nanmean(rates), color="navy", lw=1.5, linestyle="--",
                label=f"Gennemsnit {np.nanmean(rates):.3f} dBm/min")
axes[0].set_title("Drifthastighed per cyklus over tid")
axes[0].set_xlabel("Tidspunkt"); axes[0].set_ylabel("Haldning (dBm/min)")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
axes[0].legend()
# Histogram
axes[1].hist(rates[~np.isnan(rates)], bins=20, color="mediumpurple", edgecolor="white", lw=0.3)
axes[1].axvline(np.nanmean(rates), color="red", lw=1.5, linestyle="--",
                label=f"Gennemsnit {np.nanmean(rates):.3f} dBm/min")
axes[1].set_title("Fordeling af drifthastighed")
axes[1].set_xlabel("dBm/min"); axes[1].set_ylabel("Antal cykler")
axes[1].legend()
fig.suptitle("D. Drifthastighed (positiv = power stiger, negativ = falder)")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "D_drift_rate.png")
plt.close()

# ── E. Power ved kalibreringens start vs slut ────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(t_starts, [c["pwr_start"] for c in cycles], "o-", ms=4, color="crimson",
        label="Power rett efter optimering")
ax.plot(t_starts, [c["pwr_end"]   for c in cycles], "s-", ms=4, color="steelblue",
        label="Power rett foer naeste kalibrering (trigger)")
ax.axhline(np.mean(pend), color="steelblue", lw=1, linestyle=":",
           label=f"Gns. trigger-level: {np.mean(pend):.1f} dBm")
ax.set_title("E. Power ved start og slut af hver kalibreringscyklus")
ax.set_xlabel("Tidspunkt")
ax.set_ylabel("Power (dBm)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "E_power_start_end_cycle.png")
plt.close()

# ── F. Samlet power-tab per cyklus ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(t_starts, drop, width=pd.Timedelta("8min"), color="darkorange", alpha=0.85)
axes[0].axhline(drop.mean(), color="red", lw=1.5, linestyle="--",
                label=f"Gennemsnit {drop.mean():.1f} dBm")
axes[0].set_title("Power-tab per cyklus over tid\n(slut - start, positiv = faerer power)")
axes[0].set_ylabel("dB tab"); axes[0].set_xlabel("Tidspunkt")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
axes[0].legend()

axes[1].hist(drop, bins=20, color="darkorange", edgecolor="white", lw=0.3)
axes[1].axvline(drop.mean(), color="red", lw=1.5, linestyle="--",
                label=f"Gennemsnit {drop.mean():.1f} dBm")
axes[1].set_xlabel("dB tab"); axes[1].set_ylabel("Antal cykler")
axes[1].set_title("Fordeling af power-tab per cyklus")
axes[1].legend()

fig.suptitle("F. Samlet power-tab i lobet af en kalibreringscyklus")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "F_power_drop_per_cycle.png")
plt.close()

# ── G. Scatter: cyklus-laengde vs power-tab ──────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(dur, drop, c=mdates.date2num(t_starts), cmap="viridis", s=40, alpha=0.8)
plt.colorbar(sc, ax=ax, label="Tidspunkt (dato-numerisk)")
ax.set_title("G. Cykluslaengde vs power-tab\n(farve = tidspunkt)")
ax.set_xlabel("Minutter til kalibrering")
ax.set_ylabel("Power-tab (dBm)")
# Bedst fit
mask = ~np.isnan(dur) & ~np.isnan(drop)
z = np.polyfit(dur[mask], drop[mask], 1)
x_fit = np.linspace(dur.min(), dur.max(), 100)
ax.plot(x_fit, np.polyval(z, x_fit), "r--", lw=1.5,
        label=f"Lineaer fit: {z[0]:.3f}x + {z[1]:.1f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "G_duration_vs_drop_scatter.png")
plt.close()

print("\nDrift-analyse grafer gemt:")
for p in sorted(OUT.glob("[A-G]_*.png")):
    print(f"  {p.name}")
