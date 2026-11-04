#!/usr/bin/env python3
"""Analyze compound-level errors for all thermophysical properties.

Expected prediction layout from the comparison runner:

    results/<property>/predictions/<property>_jrmpnn_full_seed42_fraction1.00.csv

For each property, the script averages predictions across the requested seeds,
summarizes the error distribution, ranks outliers, and creates a parity/error
distribution figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


PROPERTY_ORDER = ("nbp", "mp", "tc", "pc", "vc")
PROPERTY_LABELS = {"nbp": "NBP", "mp": "MP", "tc": r"$T_c$", "pc": r"$P_c$", "vc": r"$V_c$"}
PROPERTY_UNITS = {"nbp": "K", "mp": "K", "tc": "K", "pc": "MPa", "vc": "source units"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True, help="Root containing <property>/predictions directories.")
    parser.add_argument("--variant", default="jrmpnn_full", help="Model variant. Default: jrmpnn_full.")
    parser.add_argument("--fraction", default="1.00", help="Training fraction in filenames. Default: 1.00.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--properties", nargs="+", choices=PROPERTY_ORDER, default=list(PROPERTY_ORDER))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def load_predictions(predictions_dir: Path, property_name: str, variant: str, fraction: str, seeds: list[int]) -> pd.DataFrame:
    combined = None
    missing = []
    for seed in seeds:
        filename = f"{property_name}_{variant}_seed{seed}_fraction{float(fraction):.2f}.csv"
        path = predictions_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        frame = pd.read_csv(path)
        required = {"SMILES", "y_true", "y_pred"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{filename} must contain {sorted(required)}")
        frame = frame[["SMILES", "y_true", "y_pred"]].rename(columns={"y_pred": f"y_pred_seed{seed}"})
        if combined is None:
            combined = frame
        else:
            combined = combined.merge(frame, on="SMILES", how="outer", suffixes=("", f"_seed{seed}"))
            true_columns = [column for column in combined.columns if column.startswith("y_true")]
            if len(true_columns) > 1:
                combined["y_true"] = combined[true_columns].bfill(axis=1).iloc[:, 0]
                combined = combined.drop(columns=[column for column in true_columns if column != "y_true"])
    if combined is None:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")
    if missing:
        print(f"Warning: missing {property_name} prediction files:")
        for filename in missing:
            print(f"  {filename}")
    prediction_columns = [column for column in combined.columns if column.startswith("y_pred_seed")]
    if not prediction_columns:
        raise ValueError(f"No predictions loaded for {property_name}.")
    combined["y_pred_mean"] = combined[prediction_columns].mean(axis=1)
    combined["y_pred_std"] = combined[prediction_columns].std(axis=1, ddof=0).fillna(0.0)
    combined["absolute_error"] = (combined["y_true"] - combined["y_pred_mean"]).abs()
    seed_errors = [(combined["y_true"] - combined[column]).abs() for column in prediction_columns]
    combined["mean_seed_absolute_error"] = pd.concat(seed_errors, axis=1).mean(axis=1)
    return combined


def summarize(property_name: str, frame: pd.DataFrame) -> dict:
    return {
        "property": property_name,
        "n_compounds": len(frame),
        "mae_mean_prediction": frame["absolute_error"].mean(),
        "median_absolute_error": frame["absolute_error"].median(),
        "p90_absolute_error": frame["absolute_error"].quantile(0.90),
        "p95_absolute_error": frame["absolute_error"].quantile(0.95),
        "maximum_absolute_error": frame["absolute_error"].max(),
        "rmse_mean_prediction": np.sqrt(mean_squared_error(frame["y_true"], frame["y_pred_mean"])),
        "r2_mean_prediction": r2_score(frame["y_true"], frame["y_pred_mean"]),
    }


def plot_property(property_name: str, frame: pd.DataFrame, output_dir: Path) -> None:
    label = PROPERTY_LABELS[property_name]
    unit = PROPERTY_UNITS[property_name]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(frame["y_true"], frame["y_pred_mean"], color="#2166ac", alpha=0.70, s=30)
    limits = [min(frame["y_true"].min(), frame["y_pred_mean"].min()), max(frame["y_true"].max(), frame["y_pred_mean"].max())]
    axes[0].plot(limits, limits, "k--", linewidth=1.4)
    axes[0].set_xlabel(f"Experimental {label} ({unit})")
    axes[0].set_ylabel(f"Predicted {label} ({unit})")
    axes[0].set_title(f"Full JR-MPNN {label} predictions", fontweight="bold")
    axes[1].hist(frame["absolute_error"], bins=25, color="#1b9e77", alpha=0.80, edgecolor="white")
    axes[1].axvline(frame["absolute_error"].median(), color="#d95f02", linestyle="--", linewidth=1.5, label="Median error")
    axes[1].set_xlabel(f"Absolute error ({unit})")
    axes[1].set_ylabel("Number of compounds")
    axes[1].set_title(f"{label} absolute-error distribution", fontweight="bold")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.30)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(f"Full JR-MPNN {label} Error Analysis", fontsize=17, y=1.02)
    figure.tight_layout()
    figure.savefig(output_dir / f"{property_name}_error_analysis.png", dpi=600, bbox_inches="tight")
    figure.savefig(output_dir / f"{property_name}_error_analysis.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    predictions_root = args.predictions_root.expanduser().resolve()
    output_dir = (args.output_dir or predictions_root / "all_property_error_analysis").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for property_name in args.properties:
        predictions_dir = predictions_root / property_name / "predictions"
        try:
            frame = load_predictions(predictions_dir, property_name, args.variant, args.fraction, args.seeds)
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipping {property_name}: {error}")
            continue
        summary_rows.append(summarize(property_name, frame))
        outliers = frame.sort_values("absolute_error", ascending=False).head(args.top_n).copy()
        outliers.insert(0, "rank", np.arange(1, len(outliers) + 1))
        output_columns = ["rank", "SMILES", "y_true", "y_pred_mean", "y_pred_std", "absolute_error", "mean_seed_absolute_error"]
        outliers[output_columns].to_csv(output_dir / f"{property_name}_outliers.csv", index=False)
        frame.to_csv(output_dir / f"{property_name}_compound_level_errors.csv", index=False)
        plot_property(property_name, frame, output_dir)

    if not summary_rows:
        raise FileNotFoundError("No property prediction files were found.")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "all_property_error_distribution_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
