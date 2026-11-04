#!/usr/bin/env python3
"""Summarize the conformal k-neighbour sensitivity runs.

Expected layout:

    uq_runs/k_5/seed_42/*.csv
    uq_runs/k_5/seed_43/*.csv
    uq_runs/k_10/seed_42/*.csv
    uq_runs/k_20/seed_42/*.csv

If seed subdirectories are absent, the script also accepts one run directly in
each ``k_<value>`` directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uq_review_analysis import PROPERTY_LABELS, PROPERTY_ORDER, load_results, summarize


def run_directories(run_root: Path, k_value: int) -> list[Path]:
    k_dir = run_root / f"k_{k_value}"
    seed_dirs = sorted(path for path in k_dir.glob("seed_*") if path.is_dir())
    return seed_dirs or [k_dir]


def collect(run_root: Path, k_values: list[int], nominal_coverage: float) -> pd.DataFrame:
    frames = []
    for k_value in k_values:
        for run_dir in run_directories(run_root, k_value):
            results = load_results(run_dir)
            summary = summarize(results, nominal_coverage)
            summary["k"] = k_value
            summary["run"] = run_dir.name
            frames.append(summary)
    if not frames:
        raise FileNotFoundError(f"No k-sensitivity runs found below {run_root}")
    return pd.concat(frames, ignore_index=True)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "adaptive_coverage",
        "adaptive_mean_width",
        "adaptive_median_width",
        "mae",
        "pearson_error_difficulty",
        "spearman_error_difficulty",
        "fixed_coverage",
        "fixed_mean_width",
        "adaptive_to_fixed_width_ratio",
    ]
    grouped = raw.groupby(["k", "property"], as_index=False)[numeric].agg(["mean", "std"])
    grouped.columns = ["_".join(column).strip("_") if isinstance(column, tuple) else column for column in grouped.columns]
    return grouped.reset_index(drop=True)


def plot_sensitivity(summary: pd.DataFrame, output_path: Path, nominal_coverage: float) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for property_name in PROPERTY_ORDER:
        subset = summary[summary["property"] == property_name].sort_values("k")
        x = subset["k"].to_numpy()
        coverage = subset["adaptive_coverage_mean"].to_numpy() * 100
        coverage_std = subset["adaptive_coverage_std"].fillna(0).to_numpy() * 100
        width = subset["adaptive_mean_width_mean"].to_numpy()
        width_std = subset["adaptive_mean_width_std"].fillna(0).to_numpy()
        axes[0].errorbar(x, coverage, yerr=coverage_std, marker="o", capsize=3, linewidth=1.8, label=PROPERTY_LABELS[property_name])
        axes[1].errorbar(x, width, yerr=width_std, marker="o", capsize=3, linewidth=1.8, label=PROPERTY_LABELS[property_name])
    axes[0].axhline(nominal_coverage * 100, color="#d95f02", linestyle="--", linewidth=1.5, label=f"Nominal {nominal_coverage * 100:.0f}%")
    axes[0].set_xlabel("Number of nearest neighbours, k")
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Coverage", fontweight="bold")
    axes[1].set_xlabel("Number of nearest neighbours, k")
    axes[1].set_ylabel("Mean adaptive interval width")
    axes[1].set_title("Interval width", fontweight="bold")
    for axis in axes:
        axis.set_xticks(sorted(summary["k"].unique()))
        axis.grid(axis="y", alpha=0.3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, fontsize=10)
    figure.suptitle("Sensitivity to the k-NN Difficulty Parameter", fontsize=17, y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parent / "uq_runs")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--nominal-coverage", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_dir = (args.output_dir or run_root / "k_sensitivity_analysis").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = collect(run_root, args.k_values, args.nominal_coverage)
    summary = aggregate(raw)
    raw.to_csv(output_dir / "uq_k_sensitivity_raw.csv", index=False)
    summary.to_csv(output_dir / "uq_k_sensitivity_summary.csv", index=False)
    plot_sensitivity(summary, output_dir / "uq_k_sensitivity.png", args.nominal_coverage)
    print(f"Wrote k-sensitivity outputs to {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
