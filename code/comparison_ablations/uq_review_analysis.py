#!/usr/bin/env python3
"""Analyze the adaptive conformal results for the reviewer response.

Expected input files are the outputs of the property-specific conformal
scripts:

    jrmpnn_nbp_conformal_results.csv
    jrmpnn_mp_conformal_results.csv
    jrmpnn_tc_conformal_results.csv
    jrmpnn_pc_conformal_results.csv
    jrmpnn_vc_conformal_results.csv

The updated conformal scripts also save fixed-width split-conformal columns.
The outputs from this script address the reviewers' requests for:

* error versus embedding dissimilarity;
* Pearson and Spearman error/dissimilarity correlations;
* conditional coverage by dissimilarity bin; and
* adaptive versus fixed-width conformal intervals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROPERTY_ORDER = ("nbp", "mp", "tc", "pc", "vc")
PROPERTY_LABELS = {"nbp": "NBP", "mp": "MP", "tc": r"$T_c$", "pc": r"$P_c$", "vc": r"$V_c$"}
PROPERTY_UNITS = {"nbp": "K", "mp": "K", "tc": "K", "pc": "MPa", "vc": "source units"}
RESULT_FILES = {property_name: f"jrmpnn_{property_name}_conformal_results.csv" for property_name in PROPERTY_ORDER}


def correlation(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    if rank:
        x = pd.Series(x[mask]).rank(method="average").to_numpy()
        y = pd.Series(y[mask]).rank(method="average").to_numpy()
    else:
        x, y = x[mask], y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_results(input_dir: Path) -> dict[str, pd.DataFrame]:
    results = {}
    missing = []
    for property_name, filename in RESULT_FILES.items():
        path = input_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        frame = pd.read_csv(path)
        required = {"y_true", "y_pred", "knn_difficulty", "interval_width", "covered"}
        missing_columns = required.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{filename} is missing columns: {sorted(missing_columns)}")
        frame["absolute_error"] = (frame["y_true"] - frame["y_pred"]).abs()
        results[property_name] = frame
    if missing:
        raise FileNotFoundError(f"Missing conformal result files in {input_dir}: {', '.join(missing)}")
    return results


def summarize(results: dict[str, pd.DataFrame], nominal_coverage: float) -> pd.DataFrame:
    rows = []
    for property_name in PROPERTY_ORDER:
        frame = results[property_name]
        row = {
            "property": property_name,
            "n_test": len(frame),
            "nominal_coverage": nominal_coverage,
            "adaptive_coverage": frame["covered"].mean(),
            "adaptive_mean_width": frame["interval_width"].mean(),
            "adaptive_median_width": frame["interval_width"].median(),
            "mae": frame["absolute_error"].mean(),
            "pearson_error_difficulty": correlation(frame["knn_difficulty"].to_numpy(), frame["absolute_error"].to_numpy()),
            "spearman_error_difficulty": correlation(frame["knn_difficulty"].to_numpy(), frame["absolute_error"].to_numpy(), rank=True),
        }
        if "fixed_covered" in frame.columns:
            row["fixed_coverage"] = frame["fixed_covered"].mean()
            row["fixed_mean_width"] = frame["fixed_interval_width"].mean()
            row["fixed_median_width"] = frame["fixed_interval_width"].median()
            row["adaptive_to_fixed_width_ratio"] = row["adaptive_mean_width"] / row["fixed_mean_width"]
        else:
            row["fixed_coverage"] = np.nan
            row["fixed_mean_width"] = np.nan
            row["fixed_median_width"] = np.nan
            row["adaptive_to_fixed_width_ratio"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def difficulty_bins(results: dict[str, pd.DataFrame], n_bins: int = 4) -> pd.DataFrame:
    rows = []
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    for property_name in PROPERTY_ORDER:
        frame = results[property_name].copy()
        # Quantile bins keep the groups comparable in size across properties.
        frame["difficulty_bin"] = pd.qcut(frame["knn_difficulty"], q=n_bins, labels=labels, duplicates="drop")
        for bin_label, group in frame.groupby("difficulty_bin", observed=False):
            if len(group) == 0:
                continue
            row = {
                "property": property_name,
                "difficulty_bin": str(bin_label),
                "n": len(group),
                "mean_difficulty": group["knn_difficulty"].mean(),
                "mean_absolute_error": group["absolute_error"].mean(),
                "adaptive_coverage": group["covered"].mean(),
                "adaptive_mean_width": group["interval_width"].mean(),
            }
            if "fixed_covered" in group.columns:
                row["fixed_coverage"] = group["fixed_covered"].mean()
                row["fixed_mean_width"] = group["fixed_interval_width"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def plot_error_vs_difficulty(results: dict[str, pd.DataFrame], output_path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 13), squeeze=False)
    for index, property_name in enumerate(PROPERTY_ORDER):
        axis = axes.flat[index]
        frame = results[property_name]
        x = frame["knn_difficulty"].to_numpy()
        y = frame["absolute_error"].to_numpy()
        axis.scatter(x, y, color="#2166ac", alpha=0.70, s=40, edgecolor="none")
        if np.unique(x).size > 1:
            coefficients = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            axis.plot(x_line, np.poly1d(coefficients)(x_line), "#d95f02", linestyle="--", linewidth=1.7)
        pearson = correlation(x, y)
        spearman = correlation(x, y, rank=True)
        axis.set_title(f"{PROPERTY_LABELS[property_name]}  (r = {pearson:.2f}, ρ = {spearman:.2f})", loc="left", fontweight="bold")
        axis.set_xlabel("k-NN embedding dissimilarity")
        axis.set_ylabel(f"Absolute error ({PROPERTY_UNITS[property_name]})")
        axis.grid(axis="y", alpha=0.30)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes.flat[-1].set_visible(False)
    figure.suptitle("Prediction Error versus Embedding Dissimilarity", fontsize=17, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_conditional_coverage(bins: pd.DataFrame, nominal_coverage: float, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 13), squeeze=False)
    for index, property_name in enumerate(PROPERTY_ORDER):
        axis = axes.flat[index]
        subset = bins[bins["property"] == property_name]
        x = np.arange(len(subset))
        axis.plot(x, subset["adaptive_coverage"] * 100, marker="o", linewidth=2, color="#2166ac", label="Adaptive")
        if "fixed_coverage" in subset.columns:
            axis.plot(x, subset["fixed_coverage"] * 100, marker="s", linewidth=1.6, color="#8c8c8c", label="Fixed width")
        axis.axhline(nominal_coverage * 100, color="#d95f02", linestyle="--", linewidth=1.5, label=f"Nominal {nominal_coverage * 100:.0f}%")
        axis.set_xticks(x, subset["difficulty_bin"])
        axis.set_ylim(0, 105)
        axis.set_title(PROPERTY_LABELS[property_name], loc="left", fontweight="bold")
        axis.set_xlabel("Dissimilarity quartile")
        axis.set_ylabel("Empirical coverage (%)")
        axis.grid(axis="y", alpha=0.30)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes.flat[-1].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    figure.suptitle("Conditional Coverage by Embedding Dissimilarity", fontsize=17, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_adaptive_vs_fixed(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(summary))
    width = 0.36
    axes[0].bar(x - width / 2, summary["adaptive_coverage"] * 100, width, label="Adaptive", color="#2166ac")
    axes[0].bar(x + width / 2, summary["fixed_coverage"] * 100, width, label="Fixed width", color="#8c8c8c")
    axes[0].axhline(95, color="#d95f02", linestyle="--", linewidth=1.5, label="Nominal 95%")
    axes[0].set_xticks(x, [PROPERTY_LABELS[name] for name in summary["property"]])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_title("Coverage", fontweight="bold")
    axes[0].legend(frameon=False)

    axes[1].bar(x, summary["adaptive_to_fixed_width_ratio"], color="#1b9e77")
    axes[1].axhline(1.0, color="#d95f02", linestyle="--", linewidth=1.5)
    axes[1].set_xticks(x, [PROPERTY_LABELS[name] for name in summary["property"]])
    axes[1].set_ylabel("Adaptive / fixed mean width")
    axes[1].set_title("Interval-width comparison", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.30)

    figure.suptitle("Adaptive versus Fixed-Width Conformal Prediction", fontsize=17, y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_interval_width_vs_difficulty(results: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Show whether adaptive interval widths increase with test-set difficulty."""
    figure, axes = plt.subplots(3, 2, figsize=(12, 13), squeeze=False)
    for index, property_name in enumerate(PROPERTY_ORDER):
        axis = axes.flat[index]
        frame = results[property_name].sort_values("knn_difficulty")
        x = frame["knn_difficulty"].to_numpy()
        width = frame["interval_width"].to_numpy()
        axis.scatter(x, width, color="#2166ac", alpha=0.65, s=40, edgecolor="none", label="Test compounds")

        # Overlay means in four equal-count difficulty bins to make the trend
        # visible even when individual compounds overlap.
        binned = frame.assign(
            difficulty_bin=pd.qcut(frame["knn_difficulty"], q=4, labels=False, duplicates="drop")
        )
        means = binned.groupby("difficulty_bin", observed=True)[["knn_difficulty", "interval_width"]].mean()
        axis.plot(
            means["knn_difficulty"],
            means["interval_width"],
            color="#d95f02",
            marker="o",
            linewidth=2.0,
            label="Quartile mean",
        )
        axis.set_title(PROPERTY_LABELS[property_name], loc="left", fontweight="bold")
        axis.set_xlabel("k-NN embedding dissimilarity")
        axis.set_ylabel(f"Prediction-interval width ({PROPERTY_UNITS[property_name]})")
        axis.grid(axis="y", alpha=0.30)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes.flat[-1].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    figure.suptitle("Prediction-Interval Width versus Embedding Dissimilarity", fontsize=17, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_sorted_prediction_intervals(results: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Show the actual adaptive error bars after sorting compounds by difficulty."""
    figure, axes = plt.subplots(3, 2, figsize=(13, 13), squeeze=False)
    for index, property_name in enumerate(PROPERTY_ORDER):
        axis = axes.flat[index]
        frame = results[property_name].sort_values("knn_difficulty").reset_index(drop=True)
        positions = np.arange(len(frame))
        prediction = frame["y_pred"].to_numpy()
        lower_error = prediction - frame["lower"].to_numpy()
        upper_error = frame["upper"].to_numpy() - prediction
        difficulty = frame["knn_difficulty"].to_numpy()

        axis.errorbar(
            positions,
            prediction,
            yerr=np.vstack([lower_error, upper_error]),
            fmt="none",
            ecolor="#2166ac",
            alpha=0.32,
            elinewidth=1.0,
            capsize=1.5,
        )
        scatter = axis.scatter(
            positions,
            prediction,
            c=difficulty,
            cmap="viridis",
            s=32,
            edgecolor="none",
            zorder=3,
        )
        axis.set_title(PROPERTY_LABELS[property_name], loc="left", fontweight="bold")
        axis.set_xlabel("Test compounds sorted by k-NN difficulty")
        axis.set_ylabel(f"Prediction ({PROPERTY_UNITS[property_name]})")
        axis.grid(axis="y", alpha=0.30)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        figure.colorbar(scatter, ax=axis, pad=0.02, label="k-NN dissimilarity")
    axes.flat[-1].set_visible(False)
    figure.suptitle("Adaptive Prediction Intervals Sorted by Molecular Difficulty", fontsize=17, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent / "uq_runs" / "k_10"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_dir, help="Directory containing the five conformal result CSV files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: input-dir/analysis.")
    parser.add_argument("--nominal-coverage", type=float, default=0.95, help="Nominal conformal coverage. Default: 0.95.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir / "analysis").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(input_dir)
    summary = summarize(results, args.nominal_coverage)
    bins = difficulty_bins(results)
    summary.to_csv(output_dir / "uq_summary.csv", index=False)
    bins.to_csv(output_dir / "uq_difficulty_bins.csv", index=False)
    plot_error_vs_difficulty(results, output_dir / "uq_error_vs_difficulty.png")
    plot_interval_width_vs_difficulty(results, output_dir / "uq_interval_width_vs_difficulty.png")
    plot_sorted_prediction_intervals(results, output_dir / "uq_sorted_prediction_intervals.png")
    plot_conditional_coverage(bins, args.nominal_coverage, output_dir / "uq_conditional_coverage.png")
    if summary["fixed_coverage"].notna().all():
        plot_adaptive_vs_fixed(summary, output_dir / "uq_adaptive_vs_fixed.png")
    else:
        print("Fixed-width columns were not found; rerun the updated conformal scripts before making the fixed-width comparison.")

    print(f"Wrote analysis outputs to {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
