#!/usr/bin/env python3
"""Create parity plots with and without conformal prediction intervals.

The first figure uses the comparison-run prediction files. The second figure
uses the conformal result files and displays the lower/upper interval as a
vertical error bar around each prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_property_errors import (
    PROPERTY_LABELS,
    PROPERTY_ORDER,
    PROPERTY_UNITS,
    load_predictions,
)


VARIANT_LABELS = {
    "jrmpnn_full": "Full JR-MPNN",
    "mpnn_control": "MPNN",
    "joback_original": "Original Joback",
    "joback_refit": "Refit Joback",
    "absolute_conditioned": "Absolute-conditioned JR-MPNN",
    "residual_unconditioned": "Unconditioned residual model",
    "residual_joback_only": "Joback-only residual model",
    "residual_sum_pool": "Sum-pooling residual model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True, help="Root containing <property>/predictions directories.")
    parser.add_argument("--uq-dir", type=Path, required=True, help="Directory containing jrmpnn_<property>_conformal_results.csv files.")
    parser.add_argument("--variant", default=None, help="One model variant. Kept for backward compatibility.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Model variants for ordinary parity plots. Defaults to Full JR-MPNN, MPNN, and Original Joback.",
    )
    parser.add_argument("--fraction", default="1.00")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def plot_basic_parity(frames: dict[str, pd.DataFrame], variant: str, output_dir: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 13), squeeze=False)
    for index, property_name in enumerate(PROPERTY_ORDER):
        axis = axes.flat[index]
        frame = frames[property_name]
        x = frame["y_true"].to_numpy()
        y = frame["y_pred_mean"].to_numpy()
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        axis.scatter(x, y, color="#2166ac", alpha=0.70, s=28)
        axis.plot(limits, limits, "k--", linewidth=1.4)
        mae = np.abs(x - y).mean()
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{PROPERTY_LABELS[property_name]}  (MAE = {mae:.3g} {PROPERTY_UNITS[property_name]})", loc="left", fontweight="bold")
        axis.set_xlabel(f"Experimental {PROPERTY_LABELS[property_name]} ({PROPERTY_UNITS[property_name]})")
        axis.set_ylabel(f"Predicted {PROPERTY_LABELS[property_name]} ({PROPERTY_UNITS[property_name]})")
        axis.grid(alpha=0.30)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes.flat[-1].set_visible(False)
    model_label = VARIANT_LABELS.get(variant, variant)
    figure.suptitle(f"{model_label} Parity Plots", fontsize=17, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_dir / f"parity_{variant}_all_properties.png", dpi=600, bbox_inches="tight")
    figure.savefig(output_dir / f"parity_{variant}_all_properties.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_interval_parity(uq_dir: Path, output_dir: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 13), squeeze=False)
    for index, property_name in enumerate(PROPERTY_ORDER):
        axis = axes.flat[index]
        path = uq_dir / f"jrmpnn_{property_name}_conformal_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing UQ file: {path}")
        frame = pd.read_csv(path)
        required = {"y_true", "y_pred", "lower", "upper"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain {sorted(required)}")
        x = frame["y_true"].to_numpy()
        y = frame["y_pred"].to_numpy()
        lower = frame["lower"].to_numpy()
        upper = frame["upper"].to_numpy()
        limits = [min(x.min(), lower.min()), max(x.max(), upper.max())]
        yerr = np.vstack([y - lower, upper - y])
        axis.errorbar(x, y, yerr=yerr, fmt="o", markersize=3.5, color="#2166ac", ecolor="#7f7f7f", elinewidth=0.65, capsize=0, alpha=0.62)
        axis.plot(limits, limits, "k--", linewidth=1.4)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        coverage = frame["covered"].mean() * 100 if "covered" in frame.columns else float("nan")
        axis.set_title(f"{PROPERTY_LABELS[property_name]}  (coverage = {coverage:.1f}%)", loc="left", fontweight="bold")
        axis.set_xlabel(f"Experimental {PROPERTY_LABELS[property_name]} ({PROPERTY_UNITS[property_name]})")
        axis.set_ylabel(f"Predicted {PROPERTY_LABELS[property_name]} ({PROPERTY_UNITS[property_name]})")
        axis.grid(alpha=0.30)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes.flat[-1].set_visible(False)
    figure.suptitle("Full JR-MPNN Parity Plots with 95% Prediction Intervals", fontsize=17, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_dir / "parity_with_intervals_all_properties.png", dpi=600, bbox_inches="tight")
    figure.savefig(output_dir / "parity_with_intervals_all_properties.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    predictions_root = args.predictions_root.expanduser().resolve()
    uq_dir = args.uq_dir.expanduser().resolve()
    output_dir = (args.output_dir or predictions_root / "parity_plots").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variants or ([args.variant] if args.variant else ["jrmpnn_full", "mpnn_control", "joback_original"])
    for variant in variants:
        frames = {}
        for property_name in PROPERTY_ORDER:
            frames[property_name] = load_predictions(
                predictions_root / property_name / "predictions",
                property_name,
                variant,
                args.fraction,
                args.seeds,
            )
        plot_basic_parity(frames, variant, output_dir)
    plot_interval_parity(uq_dir, output_dir)
    print(f"Wrote parity plots to {output_dir}")


if __name__ == "__main__":
    main()
