"""
Generate report plots for the laser polarization benchmark.

Only coarse settings C1-C3 are included. C4 and C5 are excluded because the
newer runs showed that they should not be part of the final comparison.

Important: lower / more negative dBm is better.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


RAW_PATH = "benchmark_raw.csv"
SUMMARY_OUT = "benchmark_summary_c1_c3.csv"
MAX_COARSE = 3
DPI = 300


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def config_name(row):
    return f"C{int(row['coarse'])}F{int(row['fine'])}"


def annotate_cells(ax, data, fmt=".1f", color_rule="dark_high"):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return

    mid = (finite.min() + finite.max()) / 2
    for (row_idx, col_idx), value in np.ndenumerate(data):
        if not np.isfinite(value):
            continue

        if color_rule == "dark_high":
            text_color = "white" if value > mid else "black"
        elif color_rule == "dark_low":
            text_color = "white" if value < mid else "black"
        else:
            text_color = "black"

        ax.text(
            col_idx,
            row_idx,
            format(value, fmt),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=text_color,
        )


def plot_heatmap(ax, data, coarse_vals, fine_vals, title, cbar_label, cmap, fmt=".1f"):
    im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(coarse_vals)))
    ax.set_xticklabels(coarse_vals)
    ax.set_yticks(range(len(fine_vals)))
    ax.set_yticklabels(fine_vals)
    ax.set_xlabel("Coarse iterations")
    ax.set_ylabel("Fine iterations")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    annotate_cells(ax, data, fmt=fmt, color_rule="dark_high")
    return im


def save(fig, filename):
    fig.savefig(filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def main():
    raw = pd.read_csv(RAW_PATH)
    raw = raw[raw["coarse"] <= MAX_COARSE].copy()

    fallback_order = pd.Series(np.arange(1, len(raw) + 1), index=raw.index)
    if "run_order" not in raw.columns or raw["run_order"].isna().all():
        raw["run_order"] = fallback_order
    else:
        raw["run_order"] = raw["run_order"].fillna(fallback_order)

    raw["config"] = raw.apply(config_name, axis=1)
    raw["move_arm1"] = (raw["final_arm1"] - raw["start_arm1"]).abs()
    raw["move_arm2"] = (raw["final_arm2"] - raw["start_arm2"]).abs()
    raw["move_arm3"] = (raw["final_arm3"] - raw["start_arm3"]).abs()
    raw["move_total"] = raw[["move_arm1", "move_arm2", "move_arm3"]].sum(axis=1)
    raw["good_30"] = raw["final_power_dbm"] <= -30
    raw["excellent_32"] = raw["final_power_dbm"] <= -32
    raw["very_best_33"] = raw["final_power_dbm"] <= -33

    grouped = raw.groupby(["coarse", "fine"], as_index=False)
    summary = grouped.agg(
        n_trials=("final_power_dbm", "size"),
        elapsed_mean=("elapsed_seconds", "mean"),
        elapsed_std=("elapsed_seconds", "std"),
        elapsed_median=("elapsed_seconds", "median"),
        power_mean=("final_power_dbm", "mean"),
        power_std=("final_power_dbm", "std"),
        power_median=("final_power_dbm", "median"),
        power_min=("final_power_dbm", "min"),
        power_max=("final_power_dbm", "max"),
        power_q10=("final_power_dbm", lambda x: x.quantile(0.10)),
        power_q90=("final_power_dbm", lambda x: x.quantile(0.90)),
        good_30_rate=("good_30", "mean"),
        excellent_32_rate=("excellent_32", "mean"),
        very_best_33_rate=("very_best_33", "mean"),
        move_total_mean=("move_total", "mean"),
        move_total_median=("move_total", "median"),
    )
    summary["config"] = summary.apply(config_name, axis=1)

    # A compact score for one extra overview: negative power is good, runtime is bad.
    power_rank = summary["power_mean"].rank(method="min", ascending=True)
    time_rank = summary["elapsed_mean"].rank(method="min", ascending=True)
    stability_rank = summary["power_std"].rank(method="min", ascending=True)
    summary["balanced_rank_score"] = 0.55 * power_rank + 0.30 * time_rank + 0.15 * stability_rank

    summary = summary.sort_values(["coarse", "fine"]).reset_index(drop=True)
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"Saved: {SUMMARY_OUT}")

    coarse_vals = sorted(summary["coarse"].unique())
    fine_vals = sorted(summary["fine"].unique())

    def pivot(col):
        return (
            summary.pivot(index="fine", columns="coarse", values=col)
            .loc[fine_vals, coarse_vals]
            .values
        )

    # Figure 1: heatmaps for the main metrics.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle("Benchmark C1-C3 only: lower dBm is better", fontweight="bold")
    heatmaps = [
        ("power_mean", "Mean final power\nmore negative is better", "dBm", "RdYlGn_r", ".1f"),
        ("power_median", "Median final power\nmore negative is better", "dBm", "RdYlGn_r", ".1f"),
        ("elapsed_mean", "Mean runtime\nlower is faster", "seconds", "YlOrRd", ".0f"),
        ("power_std", "Power spread\nlower is more stable", "dBm std.", "YlOrRd", ".2f"),
    ]
    for ax, (col, title, label, cmap, fmt) in zip(axes.ravel(), heatmaps):
        plot_heatmap(ax, pivot(col), coarse_vals, fine_vals, title, label, cmap, fmt)

    best_idx = np.unravel_index(np.argmin(pivot("power_mean")), pivot("power_mean").shape)
    axes[0, 0].add_patch(
        plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="lime",
            linewidth=2.5,
            zorder=5,
        )
    )
    save(fig, "report_fig1_heatmaps_c1_c3.png")

    # Figure 2: power line plots.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Final power by iteration setting, C1-C3 only", fontweight="bold")
    coarse_colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(coarse_vals)))
    fine_colors = plt.cm.Oranges(np.linspace(0.40, 0.95, len(fine_vals)))

    ax = axes[0]
    for idx, coarse in enumerate(coarse_vals):
        sub = summary[summary["coarse"] == coarse].sort_values("fine")
        ax.plot(sub["fine"], sub["power_mean"], "o-", color=coarse_colors[idx], linewidth=2, label=f"C{coarse}")
        ax.fill_between(
            sub["fine"],
            sub["power_q10"],
            sub["power_q90"],
            color=coarse_colors[idx],
            alpha=0.12,
        )
    ax.set_xlabel("Fine iterations")
    ax.set_ylabel("Final power (dBm)")
    ax.set_title("Mean power vs. fine iterations\nband = 10th-90th percentile")
    ax.set_xticks(fine_vals)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Coarse")
    ax.annotate("more negative is better", xy=(0.98, 0.04), xycoords="axes fraction", ha="right", color="0.35")

    ax = axes[1]
    for idx, fine in enumerate(fine_vals):
        sub = summary[summary["fine"] == fine].sort_values("coarse")
        ax.plot(sub["coarse"], sub["power_mean"], "s-", color=fine_colors[idx], linewidth=2, label=f"F{fine}")
    ax.set_xlabel("Coarse iterations")
    ax.set_ylabel("Final power (dBm)")
    ax.set_title("Mean power vs. coarse iterations")
    ax.set_xticks(coarse_vals)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Fine")
    ax.annotate("more negative is better", xy=(0.98, 0.04), xycoords="axes fraction", ha="right", color="0.35")
    save(fig, "report_fig2_power_lines_c1_c3.png")

    # Figure 3: runtime line plots.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Runtime by iteration setting, C1-C3 only", fontweight="bold")

    ax = axes[0]
    for idx, coarse in enumerate(coarse_vals):
        sub = summary[summary["coarse"] == coarse].sort_values("fine")
        ax.plot(sub["fine"], sub["elapsed_mean"], "o-", color=coarse_colors[idx], linewidth=2, label=f"C{coarse}")
    ax.set_xlabel("Fine iterations")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Runtime vs. fine iterations")
    ax.set_xticks(fine_vals)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Coarse")

    ax = axes[1]
    for idx, fine in enumerate(fine_vals):
        sub = summary[summary["fine"] == fine].sort_values("coarse")
        ax.plot(sub["coarse"], sub["elapsed_mean"], "s-", color=fine_colors[idx], linewidth=2, label=f"F{fine}")
    ax.set_xlabel("Coarse iterations")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("Runtime vs. coarse iterations")
    ax.set_xticks(coarse_vals)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Fine")
    save(fig, "report_fig3_runtime_lines_c1_c3.png")

    # Figure 4: Pareto overview.
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    for _, row in summary.iterrows():
        ax.errorbar(
            row["elapsed_mean"],
            row["power_mean"],
            xerr=row["elapsed_std"] / np.sqrt(row["n_trials"]),
            yerr=row["power_std"] / np.sqrt(row["n_trials"]),
            fmt="none",
            color="0.55",
            alpha=0.55,
            capsize=3,
            linewidth=1,
        )

    scatter = ax.scatter(
        summary["elapsed_mean"],
        summary["power_mean"],
        c=summary["fine"],
        cmap="viridis",
        s=95,
        edgecolors="black",
        linewidth=0.7,
        zorder=4,
    )
    for _, row in summary.iterrows():
        ax.annotate(row["config"], (row["elapsed_mean"], row["power_mean"]), xytext=(6, 3), textcoords="offset points", fontsize=8)

    pareto_df = summary.sort_values("elapsed_mean")
    pareto_rows = []
    best_power = float("inf")
    for _, row in pareto_df.iterrows():
        if row["power_mean"] < best_power:
            pareto_rows.append(row)
            best_power = row["power_mean"]
    pareto = pd.DataFrame(pareto_rows)
    ax.plot(pareto["elapsed_mean"], pareto["power_mean"], "k--", alpha=0.45, label="Pareto front")

    best = summary.loc[summary["power_mean"].idxmin()]
    ax.scatter(best["elapsed_mean"], best["power_mean"], s=240, facecolors="none", edgecolors="#c00000", linewidth=2.2, zorder=6)
    ax.annotate(
        f"Best mean: {best['config']}\n{best['power_mean']:.2f} dBm, {best['elapsed_mean']:.0f} s",
        (best["elapsed_mean"], best["power_mean"]),
        xytext=(10, -24),
        textcoords="offset points",
        color="#c00000",
        fontweight="bold",
        fontsize=9,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Fine iterations")
    cbar.set_ticks(fine_vals)
    ax.set_xlabel("Mean runtime (s)")
    ax.set_ylabel("Mean final power (dBm, more negative is better)")
    ax.set_title("Pareto overview: final power vs. runtime")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, "report_fig4_pareto_c1_c3.png")

    # Figure 5: boxplots for all included configurations.
    ordered_configs = summary.sort_values("power_mean")["config"].tolist()
    data_by_config = [raw.loc[raw["config"] == cfg, "final_power_dbm"].dropna().to_numpy() for cfg in ordered_configs]

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    box = ax.boxplot(
        data_by_config,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.7},
        widths=0.58,
    )
    palette = plt.cm.tab10(np.linspace(0, 1, len(coarse_vals)))
    coarse_to_color = {coarse: palette[idx] for idx, coarse in enumerate(coarse_vals)}
    for patch, cfg in zip(box["boxes"], ordered_configs):
        coarse = int(cfg.split("F")[0].replace("C", ""))
        patch.set_facecolor(coarse_to_color[coarse])
        patch.set_alpha(0.48)

    rng = np.random.default_rng(7)
    for idx, values in enumerate(data_by_config, start=1):
        x = rng.normal(idx, 0.055, size=values.size)
        ax.scatter(x, values, s=8, alpha=0.23, color="black", linewidths=0)

    ax.set_xticks(range(1, len(ordered_configs) + 1))
    ax.set_xticklabels(ordered_configs, rotation=45, ha="right")
    ax.set_xlabel("Configuration sorted by mean final power")
    ax.set_ylabel("Final power (dBm)")
    ax.set_title("Final power distribution for all C1-C3 configurations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.annotate("more negative is better", xy=(0.99, 0.04), xycoords="axes fraction", ha="right", color="0.35")
    save(fig, "report_fig5_boxplots_all_c1_c3.png")

    # Figure 6: success rates for useful power thresholds.
    rate_cols = ["good_30_rate", "excellent_32_rate", "very_best_33_rate"]
    rate_labels = ["<= -30 dBm", "<= -32 dBm", "<= -33 dBm"]
    rate_summary = summary.sort_values("power_mean")
    x = np.arange(len(rate_summary))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for idx, (col, label) in enumerate(zip(rate_cols, rate_labels)):
        ax.bar(x + (idx - 1) * width, rate_summary[col] * 100, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(rate_summary["config"], rotation=45, ha="right")
    ax.set_ylabel("Share of trials (%)")
    ax.set_xlabel("Configuration sorted by mean final power")
    ax.set_title("How often each configuration reaches useful dBm thresholds")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Final power")
    save(fig, "report_fig6_success_rates_c1_c3.png")

    # Figure 7: ranked mean power with standard error.
    rank_summary = summary.sort_values("power_mean", ascending=False).reset_index(drop=True)
    values = rank_summary["power_mean"].to_numpy()
    errors = rank_summary["power_std"].to_numpy() / np.sqrt(rank_summary["n_trials"].to_numpy())
    labels = rank_summary["config"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    norm = Normalize(vmin=values.min(), vmax=values.max())
    colors = [cm.RdYlGn_r(norm(v)) for v in values]
    ax.barh(np.arange(len(labels)), values, xerr=errors, color=colors, edgecolor="white", error_kw={"capsize": 3})
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean final power (dBm, more negative is better)")
    ax.set_title("Ranking by mean final power, C1-C3 only")
    ax.grid(True, axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value - 0.12, idx, f"{value:.1f}", va="center", ha="right", fontsize=8)
    save(fig, "report_fig7_ranking_c1_c3.png")

    # Figure 8: balanced score ranking.
    score_summary = summary.sort_values("balanced_rank_score", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.barh(score_summary["config"], score_summary["balanced_rank_score"], color="#2e75b6", alpha=0.82)
    ax.set_xlabel("Balanced rank score (lower power weighted most)")
    ax.set_title("Balanced ranking: power, runtime and stability")
    ax.grid(True, axis="x", alpha=0.25)
    for idx, row in score_summary.iterrows():
        ax.text(row["balanced_rank_score"] + 0.08, idx, f"{row['balanced_rank_score']:.1f}", va="center", fontsize=8)
    save(fig, "report_fig8_balanced_ranking_c1_c3.png")

    # Figure 9: raw trials in measurement order.
    ordered_raw = raw.sort_values("run_order")
    rolling_window = min(75, max(5, len(ordered_raw) // 25))
    rolling = ordered_raw["final_power_dbm"].rolling(rolling_window, min_periods=3, center=True).median()
    config_codes = ordered_raw["coarse"] * 10 + ordered_raw["fine"]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    scatter = ax.scatter(
        ordered_raw["run_order"],
        ordered_raw["final_power_dbm"],
        c=config_codes,
        cmap="tab20",
        s=12,
        alpha=0.32,
        linewidths=0,
    )
    ax.plot(ordered_raw["run_order"], rolling, color="#c00000", linewidth=2, label="Rolling median")
    ax.set_xlabel("Run order within filtered data")
    ax.set_ylabel("Final power (dBm)")
    ax.set_title("Raw measurement order check, C1-C3 only")
    ax.grid(True, alpha=0.25)
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("C/F code")
    save(fig, "report_fig9_measurement_order_c1_c3.png")

    # Figure 10: motor movement vs. final power.
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    scatter = ax.scatter(
        raw["move_total"],
        raw["final_power_dbm"],
        c=raw["coarse"],
        cmap="plasma",
        s=16,
        alpha=0.35,
        linewidths=0,
    )
    ax.set_xlabel("Total absolute arm movement")
    ax.set_ylabel("Final power (dBm)")
    ax.set_title("Movement needed vs. final power")
    ax.grid(True, alpha=0.25)
    ax.annotate("more negative is better", xy=(0.99, 0.04), xycoords="axes fraction", ha="right", color="0.35")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Coarse setting")
    cbar.set_ticks(coarse_vals)
    save(fig, "report_fig10_movement_vs_power_c1_c3.png")

    # Figure 11: final arm positions for the best mean configuration.
    best_config = summary.loc[summary["power_mean"].idxmin(), "config"]
    best_raw = raw[raw["config"] == best_config]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharey=True)
    arm_pairs = [
        ("final_arm1", "final_arm2", "Arm 1", "Arm 2"),
        ("final_arm1", "final_arm3", "Arm 1", "Arm 3"),
        ("final_arm2", "final_arm3", "Arm 2", "Arm 3"),
    ]
    for ax, (x_col, y_col, x_label, y_label) in zip(axes, arm_pairs):
        sc = ax.scatter(best_raw[x_col], best_raw[y_col], c=best_raw["final_power_dbm"], cmap="RdYlGn_r", s=24, alpha=0.85)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{x_label} vs. {y_label}")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Final arm positions for best mean configuration: {best_config}", fontweight="bold")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.026, pad=0.02)
    cbar.set_label("Final power (dBm)")
    save(fig, "report_fig11_best_config_arm_positions_c1_c3.png")

    best = summary.loc[summary["power_mean"].idxmin()]
    fastest = summary.loc[summary["elapsed_mean"].idxmin()]
    stable = summary.loc[summary["power_std"].idxmin()]
    balanced = summary.loc[summary["balanced_rank_score"].idxmin()]

    print()
    print("Filtered benchmark result, C1-C3 only")
    print("-" * 48)
    print(
        f"Best mean power:      {best['config']}  {best['power_mean']:.2f} +/- {best['power_std']:.2f} dBm"
        f"  ({best['elapsed_mean']:.1f} s, n={int(best['n_trials'])})"
    )
    print(
        f"Fastest mean runtime: {fastest['config']}  {fastest['elapsed_mean']:.1f} +/- {fastest['elapsed_std']:.1f} s"
        f"  ({fastest['power_mean']:.2f} dBm)"
    )
    print(
        f"Most stable power:    {stable['config']}  std={stable['power_std']:.2f} dBm"
        f"  ({stable['power_mean']:.2f} dBm)"
    )
    print(
        f"Best balanced score:  {balanced['config']}  score={balanced['balanced_rank_score']:.2f}"
        f"  ({balanced['power_mean']:.2f} dBm, {balanced['elapsed_mean']:.1f} s)"
    )
    print()
    print("Top 5 by mean final power:")
    for idx, row in summary.sort_values("power_mean").head(5).reset_index(drop=True).iterrows():
        print(f"  {idx + 1}. {row['config']}: {row['power_mean']:.2f} dBm | {row['elapsed_mean']:.1f} s")


if __name__ == "__main__":
    main()
