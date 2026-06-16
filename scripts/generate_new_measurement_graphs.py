import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = "graphs_new_measurements"
OPT_DIR = os.path.join(OUTPUT_DIR, "opt_results")
DRIFT_DIR = os.path.join(OUTPUT_DIR, "drift")
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

for folder in [OPT_DIR, DRIFT_DIR, COMBINED_DIR, TABLE_DIR]:
    os.makedirs(folder, exist_ok=True)


plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


ORDER_COLORS = {"QHQ": "#1f77b4", "HQQ": "#d62728"}
POWER_THRESHOLDS = [(-30, "good <= -30 dBm"), (-32, "excellent <= -32 dBm")]


def parse_local_time(series):
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def save(fig, folder, name):
    fig.tight_layout()
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def add_thresholds(ax):
    for value, label in POWER_THRESHOLDS:
        ax.axhline(value, color="#444444", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(
            0.99,
            value,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444",
        )


def format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def scatter_power(ax, x, y, c, xlabel, title):
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=28, alpha=0.82, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Power (dBm)")
    ax.set_title(title)
    add_thresholds(ax)
    return sc


def heatmap_from_bins(df, x_col, y_col, value_col, folder, filename, title):
    bins = np.linspace(0, 160, 17)
    data = df.copy()
    data["x_bin"] = pd.cut(data[x_col], bins=bins, include_lowest=True)
    data["y_bin"] = pd.cut(data[y_col], bins=bins, include_lowest=True)
    pivot = data.pivot_table(
        index="y_bin",
        columns="x_bin",
        values=value_col,
        aggfunc="mean",
        observed=False,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in pivot.columns]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ylabels = [f"{int(interval.left)}-{int(interval.right)}" for interval in pivot.index]
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    fig.colorbar(im, ax=ax, label=f"Mean {value_col}")
    save(fig, folder, filename)


def plot_box_by_category(df, category, value, folder, filename, title, max_categories=None):
    grouped = [(str(k), g[value].dropna().values) for k, g in df.groupby(category, sort=True)]
    grouped = [(label, values) for label, values in grouped if len(values)]
    if max_categories is not None and len(grouped) > max_categories:
        grouped = grouped[:max_categories]

    fig, ax = plt.subplots(figsize=(max(10, len(grouped) * 0.34), 6))
    ax.boxplot([values for _, values in grouped], labels=[label for label, _ in grouped], showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(category)
    ax.set_ylabel(value)
    ax.tick_params(axis="x", rotation=70)
    if value == "power_dbm":
        add_thresholds(ax)
    save(fig, folder, filename)


def make_opt_graphs(opt):
    opt = opt.sort_values("timestamp_measured").reset_index(drop=True)
    opt["run_index"] = np.arange(1, len(opt) + 1)
    opt["elapsed_hours"] = (
        opt["timestamp_measured"] - opt["timestamp_measured"].min()
    ).dt.total_seconds() / 3600
    opt["measure_delay_s"] = (
        opt["timestamp_measured"] - opt["timestamp_optimized"]
    ).dt.total_seconds()
    opt["arm_mean"] = opt[["arm1", "arm2", "arm3"]].mean(axis=1)
    opt["arm_spread"] = opt[["arm1", "arm2", "arm3"]].std(axis=1)
    opt["arm_range"] = opt[["arm1", "arm2", "arm3"]].max(axis=1) - opt[
        ["arm1", "arm2", "arm3"]
    ].min(axis=1)
    opt["hour"] = opt["timestamp_measured"].dt.hour
    opt["date"] = opt["timestamp_measured"].dt.date.astype(str)

    opt.describe(include="all").to_csv(os.path.join(TABLE_DIR, "opt_results_describe.csv"))
    (
        opt.groupby("arm_order")["power_dbm"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .to_csv(os.path.join(TABLE_DIR, "opt_power_by_arm_order.csv"))
    )

    fig, ax = plt.subplots()
    for order, group in opt.groupby("arm_order"):
        ax.scatter(
            group["timestamp_measured"],
            group["power_dbm"],
            s=24,
            alpha=0.78,
            label=order,
            color=ORDER_COLORS.get(order),
        )
    ax.plot(opt["timestamp_measured"], opt["power_dbm"], color="#555555", alpha=0.28, linewidth=1)
    ax.set_title("Optimized power over time")
    ax.set_xlabel("Measured time")
    ax.set_ylabel("Power (dBm)")
    ax.legend(title="Arm order")
    add_thresholds(ax)
    format_time_axis(ax)
    save(fig, OPT_DIR, "001_power_timeline_by_arm_order.png")

    fig, ax = plt.subplots()
    ranked = opt.sort_values("power_dbm").reset_index(drop=True)
    ax.plot(np.arange(1, len(ranked) + 1), ranked["power_dbm"], color="#1f77b4")
    ax.set_title("Ranked optimized power, best to worst")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Power (dBm)")
    add_thresholds(ax)
    save(fig, OPT_DIR, "002_ranked_power_best_to_worst.png")

    fig, ax = plt.subplots()
    ax.hist(opt["power_dbm"], bins=38, color="#4c78a8", alpha=0.85)
    ax.set_title("Optimized power distribution")
    ax.set_xlabel("Power (dBm)")
    ax.set_ylabel("Count")
    for value, label in POWER_THRESHOLDS:
        ax.axvline(value, color="#333333", linestyle="--", linewidth=1, label=label)
    ax.legend()
    save(fig, OPT_DIR, "003_power_histogram.png")

    plot_box_by_category(
        opt,
        "arm_order",
        "power_dbm",
        OPT_DIR,
        "004_power_boxplot_by_arm_order.png",
        "Power by arm order",
    )

    fig, ax = plt.subplots()
    values = [g["power_dbm"].values for _, g in opt.groupby("arm_order", sort=True)]
    labels = [str(k) for k, _ in opt.groupby("arm_order", sort=True)]
    violin = ax.violinplot(values, showmeans=True, showmedians=True)
    for body in violin["bodies"]:
        body.set_alpha(0.6)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title("Power violin by arm order")
    ax.set_xlabel("Arm order")
    ax.set_ylabel("Power (dBm)")
    add_thresholds(ax)
    save(fig, OPT_DIR, "005_power_violin_by_arm_order.png")

    for i, arm in enumerate(["arm1", "arm2", "arm3"], start=6):
        fig, ax = plt.subplots()
        scatter_power(
            ax,
            opt[arm],
            opt["power_dbm"],
            opt["elapsed_hours"],
            arm,
            f"{arm} position vs optimized power",
        )
        fig.colorbar(ax.collections[0], ax=ax, label="Elapsed hours")
        save(fig, OPT_DIR, f"{i:03d}_{arm}_vs_power.png")

    arm_pairs = [("arm1", "arm2"), ("arm1", "arm3"), ("arm2", "arm3")]
    for plot_no, (x_col, y_col) in enumerate(arm_pairs, start=9):
        fig, ax = plt.subplots()
        sc = ax.scatter(
            opt[x_col],
            opt[y_col],
            c=opt["power_dbm"],
            cmap="viridis",
            s=30,
            alpha=0.82,
            edgecolors="none",
        )
        ax.set_title(f"{x_col} vs {y_col}, colored by power")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.colorbar(sc, ax=ax, label="Power (dBm)")
        save(fig, OPT_DIR, f"{plot_no:03d}_{x_col}_vs_{y_col}_colored_by_power.png")

    for plot_no, (x_col, y_col) in enumerate(arm_pairs, start=12):
        fig, ax = plt.subplots()
        hb = ax.hexbin(opt[x_col], opt[y_col], C=opt["power_dbm"], gridsize=18, cmap="viridis", reduce_C_function=np.mean)
        ax.set_title(f"Mean power map: {x_col} vs {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.colorbar(hb, ax=ax, label="Mean power (dBm)")
        save(fig, OPT_DIR, f"{plot_no:03d}_{x_col}_{y_col}_mean_power_hexbin.png")

    fig, ax = plt.subplots()
    opt.set_index("timestamp_measured")["power_dbm"].rolling("60min").median().plot(ax=ax, label="60 min median")
    opt.set_index("timestamp_measured")["power_dbm"].rolling("60min").mean().plot(ax=ax, label="60 min mean")
    ax.scatter(opt["timestamp_measured"], opt["power_dbm"], s=12, alpha=0.26, color="#444444", label="Measurements")
    ax.set_title("Rolling optimized power")
    ax.set_xlabel("Measured time")
    ax.set_ylabel("Power (dBm)")
    ax.legend()
    add_thresholds(ax)
    format_time_axis(ax)
    save(fig, OPT_DIR, "015_rolling_power_mean_median.png")

    fig, ax = plt.subplots()
    ax.plot(opt["timestamp_measured"], opt["power_dbm"].cummin(), label="Best so far", color="#2ca02c")
    ax.plot(opt["timestamp_measured"], opt["power_dbm"].cummax(), label="Worst so far", color="#d62728", alpha=0.75)
    ax.set_title("Cumulative best and worst optimized power")
    ax.set_xlabel("Measured time")
    ax.set_ylabel("Power (dBm)")
    ax.legend()
    add_thresholds(ax)
    format_time_axis(ax)
    save(fig, OPT_DIR, "016_cumulative_best_and_worst_power.png")

    fig, ax = plt.subplots()
    for arm in ["arm1", "arm2", "arm3"]:
        ax.plot(opt["timestamp_measured"], opt[arm], linewidth=1.2, label=arm)
    ax.set_title("Arm positions over time")
    ax.set_xlabel("Measured time")
    ax.set_ylabel("Arm position")
    ax.legend()
    format_time_axis(ax)
    save(fig, OPT_DIR, "017_arm_positions_over_time.png")

    fig, ax = plt.subplots()
    opt["arm_order"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_title("Optimization count by arm order")
    ax.set_xlabel("Arm order")
    ax.set_ylabel("Count")
    save(fig, OPT_DIR, "018_arm_order_counts.png")

    fig, ax = plt.subplots()
    hourly = opt.groupby("hour")["power_dbm"].mean()
    ax.plot(hourly.index, hourly.values, marker="o", color="#1f77b4")
    ax.set_title("Mean optimized power by hour of day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean power (dBm)")
    ax.set_xticks(range(0, 24, 2))
    add_thresholds(ax)
    save(fig, OPT_DIR, "019_mean_power_by_hour.png")

    plot_box_by_category(
        opt,
        "date",
        "power_dbm",
        OPT_DIR,
        "020_power_boxplot_by_date.png",
        "Optimized power by date",
    )

    for filename, title, data in [
        ("021_best_25_optimization_points.png", "Best 25 optimization points", opt.nsmallest(25, "power_dbm")),
        ("022_worst_25_optimization_points.png", "Worst 25 optimization points", opt.nlargest(25, "power_dbm")),
    ]:
        fig, ax = plt.subplots(figsize=(12, 7))
        labels = data.apply(lambda r: f"{r['run_index']} {r['arm_order']} ({r['arm1']},{r['arm2']},{r['arm3']})", axis=1)
        ax.barh(labels, data["power_dbm"], color="#4c78a8")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Power (dBm)")
        save(fig, OPT_DIR, filename)

    fig, ax = plt.subplots()
    ax.hist(opt["measure_delay_s"], bins=30, color="#72b7b2")
    ax.set_title("Delay between optimized and measured timestamps")
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Count")
    save(fig, OPT_DIR, "023_measurement_delay_histogram.png")

    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    numeric = opt[["arm1", "arm2", "arm3", "power_dbm"]]
    cols = ["arm1", "arm2", "arm3"]
    for row, y_col in enumerate(cols):
        for col, x_col in enumerate(cols):
            ax = axes[row, col]
            if row == col:
                ax.hist(opt[x_col], bins=24, color="#4c78a8", alpha=0.8)
            else:
                ax.scatter(opt[x_col], opt[y_col], c=opt["power_dbm"], cmap="viridis", s=10, alpha=0.75)
            if row == 2:
                ax.set_xlabel(x_col)
            if col == 0:
                ax.set_ylabel(y_col)
    fig.suptitle("Arm position pair matrix")
    save(fig, OPT_DIR, "024_arm_pair_matrix.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    corr = numeric.join(opt[["arm_mean", "arm_spread", "arm_range"]]).corr()
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Optimization correlation heatmap")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for row in range(len(corr.index)):
        for col in range(len(corr.columns)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Correlation")
    save(fig, OPT_DIR, "025_correlation_heatmap.png")

    for plot_no, metric in enumerate(["arm_spread", "arm_range", "arm_mean"], start=26):
        fig, ax = plt.subplots()
        scatter_power(ax, opt[metric], opt["power_dbm"], opt["elapsed_hours"], metric, f"{metric} vs optimized power")
        fig.colorbar(ax.collections[0], ax=ax, label="Elapsed hours")
        save(fig, OPT_DIR, f"{plot_no:03d}_{metric}_vs_power.png")

    for plot_no, (x_col, y_col) in enumerate(arm_pairs, start=29):
        heatmap_from_bins(
            opt,
            x_col,
            y_col,
            "power_dbm",
            OPT_DIR,
            f"{plot_no:03d}_{x_col}_{y_col}_binned_mean_power.png",
            f"Binned mean power: {x_col} vs {y_col}",
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, (order, group) in zip(axes, opt.groupby("arm_order", sort=True)):
        ax.scatter(group["timestamp_measured"], group["power_dbm"], s=22, alpha=0.78, color=ORDER_COLORS.get(order))
        ax.set_title(f"{order} optimized power timeline")
        ax.set_ylabel("Power (dBm)")
        add_thresholds(ax)
        format_time_axis(ax)
    axes[-1].set_xlabel("Measured time")
    save(fig, OPT_DIR, "032_power_timeline_faceted_by_arm_order.png")

    return opt


def make_drift_graphs(drift):
    drift = drift.sort_values("timestamp").reset_index(drop=True)
    drift["sample_index"] = np.arange(1, len(drift) + 1)
    drift["elapsed_hours"] = (
        drift["timestamp"] - drift["timestamp"].min()
    ).dt.total_seconds() / 3600
    drift["hour"] = drift["timestamp"].dt.hour
    drift["date"] = drift["timestamp"].dt.date.astype(str)
    drift["hour_block"] = drift["timestamp"].dt.floor("h")
    drift["delta_power"] = drift["power_dbm"].diff()
    drift["sample_interval_s"] = drift["timestamp"].diff().dt.total_seconds()
    drift["drift_from_start"] = drift["power_dbm"] - drift["power_dbm"].iloc[0]

    drift.describe(include="all").to_csv(os.path.join(TABLE_DIR, "drift_describe.csv"))
    (
        drift.groupby("hour_block")["power_dbm"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .to_csv(os.path.join(TABLE_DIR, "drift_hourly_summary.csv"))
    )

    fig, ax = plt.subplots()
    ax.plot(drift["timestamp"], drift["power_dbm"], linewidth=1, color="#1f77b4")
    ax.set_title("Drift power over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (dBm)")
    add_thresholds(ax)
    format_time_axis(ax)
    save(fig, DRIFT_DIR, "001_drift_power_timeline.png")

    fig, ax = plt.subplots()
    indexed = drift.set_index("timestamp")["power_dbm"]
    ax.plot(drift["timestamp"], drift["power_dbm"], linewidth=0.7, alpha=0.32, color="#444444", label="Raw")
    indexed.rolling("5min").mean().plot(ax=ax, label="5 min mean", color="#1f77b4")
    indexed.rolling("30min").mean().plot(ax=ax, label="30 min mean", color="#ff7f0e")
    indexed.rolling("2h").mean().plot(ax=ax, label="2 h mean", color="#2ca02c")
    ax.set_title("Drift rolling means")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (dBm)")
    ax.legend()
    add_thresholds(ax)
    format_time_axis(ax)
    save(fig, DRIFT_DIR, "002_drift_rolling_means.png")

    fig, ax = plt.subplots()
    indexed.rolling("5min").std().plot(ax=ax, label="5 min std", color="#1f77b4")
    indexed.rolling("30min").std().plot(ax=ax, label="30 min std", color="#ff7f0e")
    indexed.rolling("2h").std().plot(ax=ax, label="2 h std", color="#2ca02c")
    ax.set_title("Drift rolling standard deviation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power std (dB)")
    ax.legend()
    format_time_axis(ax)
    save(fig, DRIFT_DIR, "003_drift_rolling_std.png")

    fig, ax = plt.subplots()
    ax.hist(drift["power_dbm"], bins=48, color="#4c78a8", alpha=0.85)
    ax.set_title("Drift power distribution")
    ax.set_xlabel("Power (dBm)")
    ax.set_ylabel("Count")
    for value, label in POWER_THRESHOLDS:
        ax.axvline(value, color="#333333", linestyle="--", linewidth=1, label=label)
    ax.legend()
    save(fig, DRIFT_DIR, "004_drift_power_histogram.png")

    fig, ax = plt.subplots()
    values = np.sort(drift["power_dbm"].dropna().values)
    ax.plot(values, np.linspace(0, 1, len(values)), color="#1f77b4")
    ax.set_title("Drift power CDF")
    ax.set_xlabel("Power (dBm)")
    ax.set_ylabel("Fraction <= value")
    save(fig, DRIFT_DIR, "005_drift_power_cdf.png")

    fig, ax = plt.subplots()
    ax.plot(drift["timestamp"], drift["delta_power"], linewidth=0.8, color="#9467bd")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Power change between drift samples")
    ax.set_xlabel("Time")
    ax.set_ylabel("Delta power (dB)")
    format_time_axis(ax)
    save(fig, DRIFT_DIR, "006_drift_sample_to_sample_delta.png")

    fig, ax = plt.subplots()
    intervals = drift["sample_interval_s"].dropna()
    ax.hist(intervals[intervals <= intervals.quantile(0.99)], bins=40, color="#72b7b2")
    ax.set_title("Drift sampling interval distribution")
    ax.set_xlabel("Seconds between samples")
    ax.set_ylabel("Count")
    save(fig, DRIFT_DIR, "007_drift_sampling_interval_histogram.png")

    fig, ax = plt.subplots()
    by_hour = drift.groupby("hour")["power_dbm"].mean()
    ax.plot(by_hour.index, by_hour.values, marker="o", color="#1f77b4")
    ax.set_title("Mean drift power by hour of day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean power (dBm)")
    ax.set_xticks(range(0, 24, 2))
    add_thresholds(ax)
    save(fig, DRIFT_DIR, "008_drift_mean_power_by_hour.png")

    plot_box_by_category(
        drift,
        "hour",
        "power_dbm",
        DRIFT_DIR,
        "009_drift_power_boxplot_by_hour.png",
        "Drift power by hour of day",
    )

    plot_box_by_category(
        drift,
        "date",
        "power_dbm",
        DRIFT_DIR,
        "010_drift_power_boxplot_by_date.png",
        "Drift power by date",
    )

    fig, ax = plt.subplots()
    ax.plot(drift["timestamp"], drift["power_dbm"].cummin(), label="Cumulative min", color="#2ca02c")
    ax.plot(drift["timestamp"], drift["power_dbm"].cummax(), label="Cumulative max", color="#d62728", alpha=0.75)
    ax.set_title("Drift cumulative min and max")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (dBm)")
    ax.legend()
    add_thresholds(ax)
    format_time_axis(ax)
    save(fig, DRIFT_DIR, "011_drift_cumulative_min_max.png")

    fig, ax = plt.subplots()
    ax.plot(drift["timestamp"], drift["drift_from_start"], linewidth=1, color="#1f77b4")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Power drift relative to first sample")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power change from first sample (dB)")
    format_time_axis(ax)
    save(fig, DRIFT_DIR, "012_drift_relative_to_first_sample.png")

    fig, ax = plt.subplots()
    q10 = indexed.rolling("30min").quantile(0.10)
    q50 = indexed.rolling("30min").quantile(0.50)
    q90 = indexed.rolling("30min").quantile(0.90)
    ax.fill_between(q50.index, q10.values, q90.values, alpha=0.25, color="#4c78a8", label="10-90% band")
    ax.plot(q50.index, q50.values, color="#1f77b4", label="30 min median")
    ax.set_title("Drift rolling percentile band")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (dBm)")
    ax.legend()
    add_thresholds(ax)
    format_time_axis(ax)
    save(fig, DRIFT_DIR, "013_drift_rolling_percentile_band.png")

    fig, ax = plt.subplots()
    sc = ax.scatter(drift["elapsed_hours"], drift["power_dbm"], c=drift["sample_index"], cmap="viridis", s=13, alpha=0.72)
    ax.set_title("Drift elapsed time scatter")
    ax.set_xlabel("Elapsed hours")
    ax.set_ylabel("Power (dBm)")
    add_thresholds(ax)
    fig.colorbar(sc, ax=ax, label="Sample index")
    save(fig, DRIFT_DIR, "014_drift_elapsed_time_scatter.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    hour_block = drift.groupby("hour_block")["power_dbm"].mean()
    ax.bar(hour_block.index, hour_block.values, width=0.03, color="#4c78a8")
    ax.set_title("Hourly mean drift power")
    ax.set_xlabel("Hour block")
    ax.set_ylabel("Mean power (dBm)")
    add_thresholds(ax)
    format_time_axis(ax)
    save(fig, DRIFT_DIR, "015_drift_hourly_mean_bars.png")

    fig, ax = plt.subplots()
    centered = drift["power_dbm"] - drift["power_dbm"].mean()
    max_lag = min(240, len(centered) - 1)
    lags = np.arange(1, max_lag + 1)
    autocorr = [centered.autocorr(lag=int(lag)) for lag in lags]
    ax.plot(lags, autocorr, color="#1f77b4")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Drift autocorrelation")
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Autocorrelation")
    save(fig, DRIFT_DIR, "016_drift_autocorrelation.png")

    fig, ax = plt.subplots()
    ax.hexbin(drift["elapsed_hours"], drift["power_dbm"], gridsize=45, cmap="viridis", mincnt=1)
    ax.set_title("Drift density over elapsed time")
    ax.set_xlabel("Elapsed hours")
    ax.set_ylabel("Power (dBm)")
    add_thresholds(ax)
    save(fig, DRIFT_DIR, "017_drift_elapsed_time_density.png")

    fig, ax = plt.subplots()
    drift.groupby("hour_block")["power_dbm"].std().plot(ax=ax, marker="o", color="#9467bd")
    ax.set_title("Hourly drift volatility")
    ax.set_xlabel("Hour block")
    ax.set_ylabel("Power std (dB)")
    format_time_axis(ax)
    save(fig, DRIFT_DIR, "018_drift_hourly_std.png")

    return drift


def make_combined_graphs(opt, drift):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(drift["timestamp"], drift["power_dbm"], linewidth=0.7, alpha=0.45, label="Drift", color="#555555")
    for order, group in opt.groupby("arm_order"):
        ax.scatter(
            group["timestamp_measured"],
            group["power_dbm"],
            s=26,
            alpha=0.82,
            label=f"Opt {order}",
            color=ORDER_COLORS.get(order),
        )
    ax.set_title("Drift timeline with optimization points")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (dBm)")
    ax.legend()
    add_thresholds(ax)
    format_time_axis(ax)
    save(fig, COMBINED_DIR, "001_drift_with_optimization_points.png")

    nearest = pd.merge_asof(
        opt.sort_values("timestamp_measured"),
        drift.sort_values("timestamp"),
        left_on="timestamp_measured",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("10s"),
        suffixes=("_opt", "_drift"),
    )
    nearest["nearest_delta_db"] = nearest["power_dbm_opt"] - nearest["power_dbm_drift"]
    nearest.to_csv(os.path.join(TABLE_DIR, "opt_nearest_drift_samples.csv"), index=False)

    fig, ax = plt.subplots()
    valid = nearest.dropna(subset=["nearest_delta_db"])
    ax.hist(valid["nearest_delta_db"], bins=36, color="#4c78a8", alpha=0.85)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Optimization power minus nearest drift sample")
    ax.set_xlabel("Delta (dB)")
    ax.set_ylabel("Count")
    save(fig, COMBINED_DIR, "002_opt_minus_nearest_drift_histogram.png")

    fig, ax = plt.subplots()
    ax.scatter(valid["timestamp_measured"], valid["nearest_delta_db"], c=valid["power_dbm_opt"], cmap="viridis", s=24, alpha=0.8)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Nearest drift agreement over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Opt - drift (dB)")
    format_time_axis(ax)
    save(fig, COMBINED_DIR, "003_opt_minus_nearest_drift_over_time.png")

    window_rows = []
    drift_idx = drift.set_index("timestamp")
    for row in opt.itertuples(index=False):
        start = row.timestamp_measured - pd.Timedelta("60s")
        end = row.timestamp_measured + pd.Timedelta("60s")
        window = drift_idx.loc[start:end].copy()
        if window.empty:
            continue
        window["relative_s"] = (window.index - row.timestamp_measured).total_seconds()
        window["run_index"] = row.run_index
        window["arm_order"] = row.arm_order
        window_rows.append(window.reset_index())

    if window_rows:
        windows = pd.concat(window_rows, ignore_index=True)
        windows.to_csv(os.path.join(TABLE_DIR, "drift_windows_around_optimizations.csv"), index=False)

        fig, ax = plt.subplots()
        ax.scatter(windows["relative_s"], windows["power_dbm"], s=7, alpha=0.08, color="#555555", label="Window samples")
        mean_window = windows.groupby(
            pd.cut(windows["relative_s"], bins=np.arange(-60, 65, 5), include_lowest=True),
            observed=False,
        )["power_dbm"].mean()
        centers = [interval.mid for interval in mean_window.index]
        ax.plot(centers, mean_window.values, color="#1f77b4", linewidth=2, label="5 s binned mean")
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title("Drift samples around optimization timestamps")
        ax.set_xlabel("Seconds from optimization measurement")
        ax.set_ylabel("Drift power (dBm)")
        ax.legend()
        add_thresholds(ax)
        save(fig, COMBINED_DIR, "004_drift_windows_around_optimizations.png")

        fig, ax = plt.subplots()
        for order, group in windows.groupby("arm_order"):
            mean_window = group.groupby(
                pd.cut(group["relative_s"], bins=np.arange(-60, 65, 5), include_lowest=True),
                observed=False,
            )["power_dbm"].mean()
            centers = [interval.mid for interval in mean_window.index]
            ax.plot(centers, mean_window.values, linewidth=2, label=order, color=ORDER_COLORS.get(order))
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title("Average drift window by arm order")
        ax.set_xlabel("Seconds from optimization measurement")
        ax.set_ylabel("Drift power (dBm)")
        ax.legend()
        add_thresholds(ax)
        save(fig, COMBINED_DIR, "005_average_drift_window_by_arm_order.png")

    fig, ax = plt.subplots()
    ax.scatter(opt["elapsed_hours"], opt["power_dbm"], s=20, alpha=0.75, label="Optimization", color="#1f77b4")
    ax.scatter(drift["elapsed_hours"], drift["power_dbm"], s=9, alpha=0.2, label="Drift", color="#555555")
    ax.set_title("Optimization and drift power vs elapsed hours")
    ax.set_xlabel("Elapsed hours")
    ax.set_ylabel("Power (dBm)")
    ax.legend()
    add_thresholds(ax)
    save(fig, COMBINED_DIR, "006_opt_and_drift_elapsed_hours.png")

    fig, ax = plt.subplots()
    ax.hist(drift["power_dbm"], bins=48, alpha=0.55, label="Drift", color="#555555", density=True)
    ax.hist(opt["power_dbm"], bins=38, alpha=0.55, label="Optimization", color="#1f77b4", density=True)
    ax.set_title("Power distribution comparison")
    ax.set_xlabel("Power (dBm)")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, COMBINED_DIR, "007_power_distribution_comparison.png")

    fig, ax = plt.subplots()
    opt_hour = opt.groupby("hour")["power_dbm"].mean()
    drift_hour = drift.groupby("hour")["power_dbm"].mean()
    ax.plot(drift_hour.index, drift_hour.values, marker="o", label="Drift", color="#555555")
    ax.plot(opt_hour.index, opt_hour.values, marker="o", label="Optimization", color="#1f77b4")
    ax.set_title("Mean power by hour: drift vs optimization")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean power (dBm)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    add_thresholds(ax)
    save(fig, COMBINED_DIR, "008_mean_power_by_hour_comparison.png")

    fig, ax = plt.subplots()
    opt_rate = pd.Series(
        {
            "opt <= -30": (opt["power_dbm"] <= -30).mean(),
            "opt <= -32": (opt["power_dbm"] <= -32).mean(),
            "drift <= -30": (drift["power_dbm"] <= -30).mean(),
            "drift <= -32": (drift["power_dbm"] <= -32).mean(),
        }
    )
    opt_rate.plot(kind="bar", ax=ax, color=["#1f77b4", "#1f77b4", "#555555", "#555555"])
    ax.set_title("Fraction of measurements below thresholds")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    save(fig, COMBINED_DIR, "009_threshold_success_fraction.png")


def main():
    opt = pd.read_csv("opt_results.csv")
    drift = pd.read_csv("drift.csv")

    opt["timestamp_optimized"] = parse_local_time(opt["timestamp_optimized"])
    opt["timestamp_measured"] = parse_local_time(opt["timestamp_measured"])
    drift["timestamp"] = parse_local_time(drift["timestamp"])

    opt = opt.dropna(subset=["timestamp_optimized", "timestamp_measured", "power_dbm"]).copy()
    drift = drift.dropna(subset=["timestamp", "power_dbm"]).copy()

    opt = make_opt_graphs(opt)
    drift = make_drift_graphs(drift)
    make_combined_graphs(opt, drift)

    graph_count = sum(
        1
        for root, _, files in os.walk(OUTPUT_DIR)
        for file_name in files
        if file_name.lower().endswith(".png")
    )
    print(f"Generated {graph_count} PNG graphs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
