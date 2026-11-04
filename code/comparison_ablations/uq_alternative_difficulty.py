#!/usr/bin/env python3
"""Compare alternative difficulty measures for adaptive conformal prediction.

The script reuses the same trained-model predictions and train/calibration/test
split, then recalculates conformal intervals using:

1. k-NN distance in the JR-MPNN embedding space;
2. k-NN distance in Morgan fingerprint space; and
3. k-NN distance in heavy-atom-count space.

The latter is a size control for the reviewer's concern that sum pooling may
make embedding distance reflect molecular size.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr


PROPERTY_ORDER = ("nbp", "mp", "tc", "pc", "vc")
PROPERTY_LABELS = {"nbp": "NBP", "mp": "MP", "tc": r"$T_c$", "pc": r"$P_c$", "vc": r"$V_c$"}
PROPERTY_UNITS = {"nbp": "K", "mp": "K", "tc": "K", "pc": "MPa", "vc": "source units"}
METHOD_ORDER = ("embedding_knn", "fingerprint_knn", "size_knn")
METHOD_LABELS = {
    "embedding_knn": "Embedding k-NN",
    "fingerprint_knn": "Fingerprint k-NN",
    "size_knn": "Heavy-atom size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="One UQ run directory, such as uq_runs_correct/k_10/seed_42.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing *_train.csv, *_calibration.csv, and *_test.csv.")
    parser.add_argument("--properties", nargs="+", choices=PROPERTY_ORDER, default=list(PROPERTY_ORDER))
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbours for fingerprint and size distances.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x, y)[0])
    return float(spearmanr(x, y)[0])


def fingerprints(smiles: list[str]) -> list:
    result = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"Could not parse SMILES: {value}")
        result.append(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048))
    return result


def heavy_atom_counts(smiles: list[str]) -> np.ndarray:
    counts = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"Could not parse SMILES: {value}")
        counts.append(molecule.GetNumHeavyAtoms())
    return np.asarray(counts, dtype=float)


def fingerprint_knn_distance(train_smiles: list[str], query_smiles: list[str], k: int) -> np.ndarray:
    train_fps = fingerprints(train_smiles)
    query_fps = fingerprints(query_smiles)
    k = min(k, len(train_fps))
    distances = []
    for query_fp in query_fps:
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, train_fps), dtype=float)
        distances.append(np.sort(1.0 - similarities)[:k].mean())
    return np.asarray(distances, dtype=float)


def size_knn_distance(train_smiles: list[str], query_smiles: list[str], k: int) -> np.ndarray:
    train_sizes = heavy_atom_counts(train_smiles)
    query_sizes = heavy_atom_counts(query_smiles)
    k = min(k, len(train_sizes))
    return np.asarray([np.sort(np.abs(train_sizes - size))[:k].mean() for size in query_sizes], dtype=float)


def add_fallback_smiles(frame: pd.DataFrame, path: Path, split: str) -> pd.DataFrame:
    if "SMILES" in frame.columns:
        return frame
    if not path.exists():
        raise FileNotFoundError(f"{split} SMILES are missing from the result file and {path} does not exist.")
    source = pd.read_csv(path)
    if len(source) != len(frame):
        raise ValueError(f"Cannot align {split} SMILES: {len(source)} source rows versus {len(frame)} result rows.")
    result = frame.copy()
    result.insert(0, "SMILES", source["SMILES"].astype(str).to_numpy())
    return result


def summarize_method(calibration: pd.DataFrame, test: pd.DataFrame, difficulty: str, cal_difficulty: np.ndarray, test_difficulty: np.ndarray, alpha: float = 0.05) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cal_error = np.abs(calibration["y_true"].to_numpy() - calibration["y_pred"].to_numpy())
    test_true = test["y_true"].to_numpy()
    test_pred = test["y_pred"].to_numpy()
    test_error = np.abs(test_true - test_pred)
    epsilon = 1e-3
    n = len(cal_error)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    scores = cal_error / (cal_difficulty + epsilon)
    q = float(np.quantile(scores, level))
    half_width = q * (test_difficulty + epsilon)
    lower = test_pred - half_width
    upper = test_pred + half_width
    covered = (test_true >= lower) & (test_true <= upper)
    widths = upper - lower
    summary = {
        "difficulty": difficulty,
        "n_calibration": n,
        "n_test": len(test),
        "coverage": float(covered.mean()),
        "mean_interval_width": float(widths.mean()),
        "median_interval_width": float(np.median(widths)),
        "p90_interval_width": float(np.quantile(widths, 0.90)),
        "calibration_quantile": q,
        "pearson_error": safe_correlation(test_difficulty, test_error, "pearson"),
        "spearman_error": safe_correlation(test_difficulty, test_error, "spearman"),
    }
    ranks = pd.Series(test_difficulty).rank(method="first")
    bins = pd.qcut(ranks, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    bin_frame = pd.DataFrame({
        "difficulty": difficulty,
        "difficulty_bin": bins.astype(str),
        "n_test": 1,
        "covered": covered.astype(int),
        "absolute_error": test_error,
        "interval_width": widths,
    })
    bin_summary = bin_frame.groupby(["difficulty", "difficulty_bin"], as_index=False, observed=False).agg(
        n_test=("n_test", "sum"),
        coverage=("covered", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        mean_interval_width=("interval_width", "mean"),
    )
    result_frame = test[["SMILES", "y_true", "y_pred"]].copy()
    result_frame["difficulty"] = difficulty
    result_frame["difficulty_value"] = test_difficulty
    result_frame["lower"] = lower
    result_frame["upper"] = upper
    result_frame["interval_width"] = widths
    result_frame["covered"] = covered.astype(int)
    return summary, bin_summary, result_frame


def plot_correlations(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 13), squeeze=False)
    for index, prop in enumerate(PROPERTY_ORDER):
        axis = axes.flat[index]
        subset = summary[summary["property"] == prop].set_index("difficulty").reindex(METHOD_ORDER)
        x = np.arange(len(METHOD_ORDER))
        axis.bar(x - 0.18, subset["pearson_error"], width=0.36, label="Pearson", color="#2166ac")
        axis.bar(x + 0.18, subset["spearman_error"], width=0.36, label="Spearman", color="#1b9e77")
        axis.set_xticks(x, [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=25, ha="right")
        axis.set_ylim(-1, 1)
        axis.set_ylabel("Correlation with absolute error")
        axis.set_title(PROPERTY_LABELS[prop], loc="left", fontweight="bold")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.3)
    axes.flat[-1].set_visible(False)
    axes.flat[0].legend(frameon=False)
    figure.suptitle("Alternative Difficulty Measures", fontsize=17, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_coverage_width(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 13), squeeze=False)
    for index, prop in enumerate(PROPERTY_ORDER):
        axis = axes.flat[index]
        subset = summary[summary["property"] == prop].set_index("difficulty").reindex(METHOD_ORDER)
        x = np.arange(len(METHOD_ORDER))
        axis.bar(x, subset["coverage"] * 100, color="#2166ac", alpha=0.85)
        axis.axhline(95, color="#d95f02", linestyle="--", linewidth=1.2)
        axis.set_ylim(0, 105)
        axis.set_ylabel("Coverage (%)")
        axis.set_xticks(x, [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=25, ha="right")
        width_axis = axis.twinx()
        width_axis.plot(x, subset["mean_interval_width"], "o-", color="#1b9e77", linewidth=1.8, label="Mean width")
        width_axis.set_ylabel(f"Mean width ({PROPERTY_UNITS[prop]})", color="#1b9e77")
        width_axis.tick_params(axis="y", labelcolor="#1b9e77")
        axis.set_title(PROPERTY_LABELS[prop], loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.3)
    axes.flat[-1].set_visible(False)
    figure.suptitle("Coverage and Interval Width by Difficulty Measure", fontsize=17, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    data_dir = (args.data_dir or Path(os.environ.get("ROOT_DIR", "~/deeplearning/new")).expanduser() / "data").expanduser().resolve()
    output_dir = (args.output_dir or run_dir / "alternative_difficulty_analysis").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    bin_frames = []

    for prop in args.properties:
        test_path = run_dir / f"jrmpnn_{prop}_conformal_results.csv"
        cal_path = run_dir / f"jrmpnn_{prop}_calibration_results.csv"
        if not test_path.exists() or not cal_path.exists():
            raise FileNotFoundError(f"Missing corrected UQ outputs for {prop}: {test_path} or {cal_path}")
        test = add_fallback_smiles(pd.read_csv(test_path), data_dir / f"{prop.upper()}_test.csv", "test")
        calibration = add_fallback_smiles(pd.read_csv(cal_path), data_dir / f"{prop.upper()}_calibration.csv", "calibration")
        train = pd.read_csv(data_dir / f"{prop.upper()}_train.csv")
        train_smiles = train["SMILES"].astype(str).tolist()
        cal_smiles = calibration["SMILES"].astype(str).tolist()
        test_smiles = test["SMILES"].astype(str).tolist()

        difficulty_values = {
            "embedding_knn": (calibration["knn_difficulty"].to_numpy(dtype=float), test["knn_difficulty"].to_numpy(dtype=float)),
            "fingerprint_knn": (fingerprint_knn_distance(train_smiles, cal_smiles, args.k), fingerprint_knn_distance(train_smiles, test_smiles, args.k)),
            "size_knn": (size_knn_distance(train_smiles, cal_smiles, args.k), size_knn_distance(train_smiles, test_smiles, args.k)),
        }
        property_results = []
        for method in METHOD_ORDER:
            summary, bins, result = summarize_method(calibration, test, method, *difficulty_values[method])
            summary["property"] = prop
            summary_rows.append(summary)
            bins.insert(0, "property", prop)
            bin_frames.append(bins)
            property_results.append(result)
        pd.concat(property_results, ignore_index=True).to_csv(output_dir / f"{prop}_alternative_difficulty_results.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary = summary[["property", "difficulty", "n_calibration", "n_test", "coverage", "mean_interval_width", "median_interval_width", "p90_interval_width", "calibration_quantile", "pearson_error", "spearman_error"]]
    summary.to_csv(output_dir / "uq_alternative_difficulty_summary.csv", index=False)
    pd.concat(bin_frames, ignore_index=True).to_csv(output_dir / "uq_alternative_difficulty_bins.csv", index=False)
    plot_correlations(summary, output_dir / "uq_alternative_difficulty_correlations")
    plot_coverage_width(summary, output_dir / "uq_alternative_difficulty_coverage_width")
    print(summary.to_string(index=False))
    print(f"Wrote alternative-difficulty outputs to {output_dir}")


if __name__ == "__main__":
    main()
