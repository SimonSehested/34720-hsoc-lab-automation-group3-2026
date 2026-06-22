import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

OUT = Path("report/figures")

THRESHOLD = -20.0

df = pd.read_csv("monitor_log2.csv", parse_dates=["timestamp"])
drift = df[df["event"] == "drift"].copy().sort_values("timestamp").reset_index(drop=True)

drift["below"] = drift["power_dbm"] < THRESHOLD
drift["run_id"] = (drift["below"] != drift["below"].shift()).cumsum()
runs = drift[drift["below"]].groupby("run_id").agg(
    t_start=("timestamp", "first"),
    t_end=("timestamp", "last"),
    n=("power_dbm", "count"),
).reset_index(drop=True)
runs["duration_min"] = (runs["t_end"] - runs["t_start"]).dt.total_seconds() / 60

print(f"Contiguous periods below {THRESHOLD} dBm: {len(runs)}")
print(f"Median length: {runs['duration_min'].median():.1f} min")
print(f"Max length:    {runs['duration_min'].max():.1f} min")

window_sizes = np.arange(1, 121, 1)
prob = np.array([(runs["duration_min"] >= T).mean() * 100 for T in window_sizes])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(window_sizes, prob, lw=2.5, color="steelblue")

ax.axvline(10, color="red", lw=2, linestyle="--", label="10 min")
p10 = np.interp(10, window_sizes, prob)
ax.axhline(p10, color="red", lw=1, linestyle=":")
ax.scatter([10], [p10], color="red", s=80, zorder=5)
ax.text(55, 90, f"$P(T > 10\\,\\mathrm{{min}}) = {p10:.0f}\\%$",
        fontsize=14, color="red", fontweight="bold", ha="center")

for t_mark in [30, 60, 90]:
    p = np.interp(t_mark, window_sizes, prob)
    ax.scatter([t_mark], [p], color="gray", s=40, zorder=4)
    ax.annotate(f"{t_mark} min\n{p:.0f}%", xy=(t_mark, p),
                xytext=(t_mark + 3, p + 3), fontsize=8, color="gray")

ax.set_title(f"Survival Curve (threshold {THRESHOLD} dBm)")
ax.set_xlabel("T (minutes)")
ax.set_ylabel("Probability (%)")
ax.set_ylim(0, 105)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "survival_curve2.png")
plt.close()
print(f"Saved {OUT / 'survival_curve2.png'}")
