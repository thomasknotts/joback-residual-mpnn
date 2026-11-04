#!/usr/bin/env python3
"""Plot data-efficiency learning curves from comparison/ablation summaries.

The script reads the per-property ``*_comparison_ablations_summary.csv`` files
created by ``summarize_results.py`` and produces publication-style small-multiple
figures. Each point is the mean over random seeds and each shaded band is ±1
standard deviation across those seeds.

Examples
--------
Plot all metrics and all model variants:

    python plot_learning_curves.py

Plot only MAE:

    python plot_learning_curves.py --metric mae --output-dir plots

Plot a less crowded main-text comparison:

    python plot_learning_curves.py --metric mae \
        --variants jrmpnn_full mpnn_control joback_original
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 17,
    }
)


PROPERTY_ORDER = ("nbp", "mp", "tc", "pc", "vc")
PROPERTY_LABELS = {
    "nbp": "NBP",
    "mp": "MP",
    "tc": r"$T_c$",
    "pc": r"$P_c$",
    "vc": r"$V_c$",
}
PROPERTY_UNITS = {
    "nbp": "K",
    "mp": "K",
    "tc": "K",
    "pc": "MPa",
    # The source summary files do not encode the Vc unit.
    "vc": "source units",
}

VARIANT_ORDER = (
    "joback_original",
    "joback_refit",
    "mpnn_control",
    "jrmpnn_full",
    "absolute_conditioned",
    "residual_unconditioned",
    "residual_joback_only",
    "residual_sum_pool",
)
VARIANT_LABELS = {
    "joback_original": "Original Joback",
    "joback_refit": "Refit Joback",
    "mpnn_control": "MPNN",
    # This is the manuscript's full residual-conditioned JR-MPNN.
    "jrmpnn_full": "Residual conditioned (Full JR-MPNN)",
    "absolute_conditioned": "Absolute conditioned",
    "residual_unconditioned": "Residual unconditioned",
    "residual_joback_only": "Residual Joback-only",
    "residual_sum_pool": "Residual sum pooling",
}

# Stable colors and markers across every metric and property figure.
VARIANT_COLORS = {
    "joback_original": "#4d4d4d",
    "joback_refit": "#8c8c8c",
    "mpnn_control": "#e17c05",
    "jrmpnn_full": "#2166ac",
    "absolute_conditioned": "#762a83",
    "residual_unconditioned": "#1b9e77",
    "residual_joback_only": "#d95f02",
    "residual_sum_pool": "#66a61e",
}
VARIANT_STYLES = {
    "joback_original": ("--", "o"),
    "joback_refit": (":", "s"),
    "mpnn_control": ("-", "^"),
    "jrmpnn_full": ("-", "o"),
    "absolute_conditioned": ("-", "D"),
    "residual_unconditioned": ("-", "v"),
    "residual_joback_only": ("-", "P"),
    "residual_sum_pool": ("-", "X"),
}

METRIC_CONFIG = {
    "mae": {
        "mean_column": "test_mae_mean",
        "std_column": "test_mae_std",
        "label": "MAE",
        "percent": False,
        "lower_is_better": True,
    },
    "mape": {
        "mean_column": "test_mape_mean",
        "std_column": "test_mape_std",
        "label": "MAPE",
        "percent": True,
        "lower_is_better": True,
    },
    "r2": {
        "mean_column": "test_r2_mean",
        "std_column": "test_r2_std",
        "label": r"$R^2$",
        "percent": False,
        "lower_is_better": False,
    },
}


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_summary_rows(input_dir: Path) -> Dict[str, List[dict]]:
    """Load one summary file per property, or a single combined summary file."""
    rows_by_property: Dict[str, List[dict]] = {}
    combined = input_dir / "comparison_ablations_summary.csv"
    if combined.exists():
        for row in read_csv_rows(combined):
            property_name = row.get("property", "").strip().lower()
            if property_name in PROPERTY_ORDER:
                rows_by_property.setdefault(property_name, []).append(row)

    for property_name in PROPERTY_ORDER:
        path = input_dir / f"{property_name}_comparison_ablations_summary.csv"
        if path.exists():
            rows_by_property[property_name] = read_csv_rows(path)

    missing = [name for name in PROPERTY_ORDER if name not in rows_by_property]
    if missing:
        raise FileNotFoundError(
            f"Missing summary CSV files for {', '.join(missing)} in {input_dir}."
        )
    return rows_by_property


def available_variants(rows_by_property: Mapping[str, Sequence[dict]]) -> List[str]:
    found = {
        row.get("variant", "").strip()
        for rows in rows_by_property.values()
        for row in rows
        if row.get("variant")
    }
    return [variant for variant in VARIANT_ORDER if variant in found]


def metric_value(row: Mapping[str, str], metric_name: str) -> tuple[float, float]:
    config = METRIC_CONFIG[metric_name]
    mean = float(row[config["mean_column"]])
    std = float(row[config["std_column"]])
    if config["percent"]:
        mean *= 100.0
        std *= 100.0
    return mean, std


def rows_for_variant(rows: Iterable[dict], variant: str, metric_name: str) -> List[tuple[float, float, float]]:
    selected = []
    for row in rows:
        if row.get("variant", "").strip() != variant:
            continue
        fraction = float(row["fraction"])
        mean, std = metric_value(row, metric_name)
        if math.isfinite(mean) and math.isfinite(std):
            selected.append((fraction, mean, std))
    return sorted(selected, key=lambda item: item[0])


def y_axis_label(metric_name: str, property_name: str) -> str:
    if metric_name == "mae":
        return f"MAE ({PROPERTY_UNITS[property_name]})"
    if metric_name == "mape":
        return "MAPE (%)"
    return r"$R^2$"


def plot_metric(
    rows_by_property: Mapping[str, Sequence[dict]],
    metric_name: str,
    variants: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    """Create a five-panel learning-curve figure and save PNG/PDF copies."""
    config = METRIC_CONFIG[metric_name]
    figure, axes = plt.subplots(3, 2, figsize=(12.5, 13.0), squeeze=False)
    axes_flat = axes.flatten()

    for index, property_name in enumerate(PROPERTY_ORDER):
        axis = axes_flat[index]
        property_rows = rows_by_property[property_name]
        plotted = {}
        lower_values = []
        upper_values = []

        for variant in variants:
            values = rows_for_variant(property_rows, variant, metric_name)
            if not values:
                continue
            plotted[variant] = values
            lower_values.extend(mean - std for _, mean, std in values)
            upper_values.extend(mean + std for _, mean, std in values)

        for variant in variants:
            values = plotted.get(variant)
            if not values:
                continue
            x = [fraction * 100.0 for fraction, _, _ in values]
            means = [mean for _, mean, _ in values]
            stds = [std for _, _, std in values]
            color = VARIANT_COLORS[variant]
            linestyle, marker = VARIANT_STYLES[variant]
            axis.fill_between(
                x,
                [mean - std for mean, std in zip(means, stds)],
                [mean + std for mean, std in zip(means, stds)],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            axis.plot(
                x,
                means,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0 if variant == "jrmpnn_full" else 1.5,
                markersize=5,
                label=VARIANT_LABELS[variant],
                zorder=3,
            )

        if not lower_values:
            axis.set_visible(False)
            continue

        y_min = min(lower_values)
        y_max = max(upper_values)
        span = y_max - y_min
        pad = 0.08 * span if span > 0 else max(abs(y_max) * 0.08, 1.0)
        if config["lower_is_better"]:
            axis.set_ylim(max(0.0, y_min - pad), y_max + pad)
        else:
            axis.set_ylim(min(-0.02, y_min - pad), min(1.02, y_max + pad))

        axis.set_title(PROPERTY_LABELS[property_name], loc="left", fontweight="bold")
        axis.set_xticks([10, 25, 50, 75, 100])
        axis.set_xlim(7, 103)
        axis.set_xlabel("Training fraction")
        axis.set_ylabel(y_axis_label(metric_name, property_name))
        axis.grid(axis="y", color="#bdbdbd", alpha=0.35, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes_flat[-1].set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            color=VARIANT_COLORS[variant],
            linestyle=VARIANT_STYLES[variant][0],
            marker=VARIANT_STYLES[variant][1],
            linewidth=2.0 if variant == "jrmpnn_full" else 1.5,
            markersize=5,
            label=VARIANT_LABELS[variant],
        )
        for variant in variants
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False, fontsize=11)
    figure.suptitle(f"Test Set {config['label']}", fontsize=17, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.91))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"learning_curves_{metric_name}.png",
        output_dir / f"learning_curves_{metric_name}.pdf",
    ]
    for path in outputs:
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return outputs


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=default_dir, help="Directory containing summary CSV files.")
    parser.add_argument("--output-dir", type=Path, default=default_dir / "plots", help="Directory for PNG/PDF figures.")
    parser.add_argument("--metric", choices=["mae", "mape", "r2", "all"], default="all", help="Metric to plot. Default: all.")
    parser.add_argument("--variants", nargs="+", choices=VARIANT_ORDER, default=None, help="Variants to plot. Default: every variant found.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution. Default: 300.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows_by_property = load_summary_rows(input_dir)
    variants = args.variants or available_variants(rows_by_property)
    if not variants:
        raise ValueError("No recognized model variants were found in the summary CSV files.")

    metrics = ["mae", "mape", "r2"] if args.metric == "all" else [args.metric]
    print(f"Reading summaries from: {input_dir}")
    print("Plotting variants:")
    for variant in variants:
        print(f"  - {variant}: {VARIANT_LABELS[variant]}")
    for metric_name in metrics:
        for output in plot_metric(rows_by_property, metric_name, variants, output_dir, args.dpi):
            print(f"Wrote {output}")


if __name__ == "__main__":
    main()
