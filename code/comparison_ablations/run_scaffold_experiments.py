#!/usr/bin/env python3
"""Run the scaffold-split comparison experiment for the five properties.

Expected input layout::

    data/scaffold_splits/seed_42/{NBP,MP,TC,PC,VC}_{train,calibration,test}.csv

The calibration split is deliberately not used for neural-network early
stopping.  A whole-scaffold validation subset is made from the scaffold
training split instead, so the calibration data remain available for a later
conformal-UQ analysis.  Outputs are written to a separate scaffold-results
directory and never overwrite the random-split comparison results.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import run_comparison_ablations as base


PROPERTIES = list(base.PROPERTY_CONFIG)
DEFAULT_VARIANTS = [
    "joback_original",
    "joback_refit",
    "mpnn_control",
    "jrmpnn_full",
]


def scaffold_key(smiles: str) -> str:
    """Return a Bemis--Murcko scaffold key with a safe acyclic fallback."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return f"INVALID::{smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    if scaffold:
        return scaffold
    # RDKit returns an empty Murcko scaffold for acyclic compounds.  Use the
    # canonical molecule so these compounds are still assigned consistently.
    return f"ACYCLIC::{Chem.MolToSmiles(molecule, canonical=True)}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_features(frame: pd.DataFrame, prop: str) -> pd.DataFrame:
    result = base.add_joback_features(frame.copy(), prop)
    result["y_joback"] = result["y_joback"].fillna(result["y_joback"].mean())
    return result


def load_scaffold_frames(data_dir: Path, prop: str) -> Dict[str, pd.DataFrame]:
    stem = base.PROPERTY_CONFIG[prop]["data_stem"]
    frames: Dict[str, pd.DataFrame] = {}
    for split in ("train", "calibration", "test"):
        path = data_dir / f"{stem}_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing scaffold split file: {path}")
        frames[split] = add_features(pd.read_csv(path), prop)
        frames[split]["scaffold"] = frames[split]["SMILES"].map(scaffold_key)
    return frames


def scaffold_train_val_split(frame: pd.DataFrame, seed: int, val_fraction: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the scaffold-training rows into whole-scaffold train and val sets."""
    groups = frame["scaffold"].drop_duplicates().tolist()
    if len(groups) < 2:
        raise RuntimeError("The scaffold training split contains fewer than two scaffolds.")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    target = max(1, int(round(len(frame) * val_fraction)))
    val_groups: List[str] = []
    val_count = 0
    for group in groups:
        if val_count < target or not val_groups:
            val_groups.append(group)
            val_count += int((frame["scaffold"] == group).sum())
    # Keep at least one scaffold in the training portion.
    if len(val_groups) == len(groups):
        val_groups.pop()
    is_val = frame["scaffold"].isin(val_groups)
    return frame.loc[~is_val].reset_index(drop=True), frame.loc[is_val].reset_index(drop=True)


def save_manifest(frames: Dict[str, pd.DataFrame], prop: str, seed: int, output_dir: Path) -> None:
    rows: List[pd.DataFrame] = []
    for split, frame in frames.items():
        rows.append(frame[["SMILES", "scaffold"]].assign(property=prop, split=split, seed=seed))
    pd.concat(rows, ignore_index=True).to_csv(
        output_dir / f"scaffold_split_manifest_{prop}_seed{seed}.csv", index=False
    )


def metric_row(y_true: Sequence[float], y_pred: Sequence[float], prefix: str) -> Dict[str, float]:
    return base.metric_dict(y_true, y_pred, prefix)


def run_one(
    prop: str,
    variant: str,
    seed: int,
    frames: Dict[str, pd.DataFrame],
    target_column: str,
    group_cols: Sequence[str],
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    clip_norm: Optional[float],
) -> Dict[str, object]:
    set_seed(seed)
    train_frame, val_frame = scaffold_train_val_split(frames["train"], seed)
    test_frame = frames["test"]
    refit = base.fit_refitted_joback(frames["train"], target_column, group_cols)
    row: Dict[str, object] = {
        "property": prop,
        "variant": variant,
        "seed": seed,
        "split": "scaffold",
        "n_scaffold_train": int(frames["train"]["scaffold"].nunique()),
        "n_scaffold_calibration": int(frames["calibration"]["scaffold"].nunique()),
        "n_scaffold_test": int(frames["test"]["scaffold"].nunique()),
        "n_train": len(frames["train"]),
        "n_calibration": len(frames["calibration"]),
        "n_test": len(test_frame),
        "device": str(device),
    }

    if variant in {"joback_original", "joback_refit"}:
        y_true, y_pred, smiles = base.evaluate_baseline(test_frame, target_column, variant, refit)
        row.update(metric_row(y_true, y_pred, "test"))
        prediction_dir = output_dir / "predictions" / prop
        prediction_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"SMILES": smiles, "y_true": y_true, "y_pred": y_pred}).to_csv(
            prediction_dir / f"{variant}_seed{seed}.csv", index=False
        )
        return row

    specs = base.variant_spec(variant)
    train_graphs = base.build_graphs(train_frame, target_column, group_cols)
    val_graphs = base.build_graphs(val_frame, target_column, group_cols)
    test_graphs = base.build_graphs(test_frame, target_column, group_cols)
    if len(train_graphs) < 4 or not val_graphs or not test_graphs:
        raise RuntimeError(
            f"Too few valid graphs for {prop}: train={len(train_graphs)}, "
            f"val={len(val_graphs)}, test={len(test_graphs)}"
        )

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    config = base.PROPERTY_CONFIG[prop]
    model = base.ControlledMPNN(
        node_dim=base.n_atom_features(),
        edge_dim=base.n_bond_features(),
        hidden_dim=config["hidden_dim"],
        num_layers=config["layers"],
        target_mode=specs["target"],
        condition_mode=specs["condition"],
        pooling=specs["pooling"],
        group_count=len(group_cols),
    ).to(device)
    model = base.train_model(
        model, train_loader, val_loader, device,
        epochs=epochs, patience=patience, learning_rate=2e-4,
        weight_decay=1e-4, clip_norm=clip_norm,
    )
    y_train, pred_train, _ = base.evaluate_model(model, train_loader, device)
    y_val, pred_val, _ = base.evaluate_model(model, val_loader, device)
    y_test, pred_test, test_smiles = base.evaluate_model(model, test_loader, device)
    row.update(metric_row(y_train, pred_train, "train"))
    row.update(metric_row(y_val, pred_val, "val"))
    row.update(metric_row(y_test, pred_test, "test"))
    row.update({"target_mode": specs["target"], "condition_mode": specs["condition"], "pooling": specs["pooling"]})

    prediction_dir = output_dir / "predictions" / prop
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SMILES": test_smiles, "y_true": y_test, "y_pred": pred_test}).to_csv(
        prediction_dir / f"{variant}_seed{seed}.csv", index=False
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--property", choices=[*PROPERTIES, "all"], default="all")
    parser.add_argument("--variants", nargs="+", choices=[*base.DEFAULT_VARIANTS], default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--utils-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-gradient-clipping", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.load_dependencies(str(Path(args.utils_dir).expanduser()))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    properties = PROPERTIES if args.property == "all" else [args.property]
    clip_norm = None if args.no_gradient_clipping else 1.0
    all_rows: List[Dict[str, object]] = []

    print(f"device: {device}", flush=True)
    for prop in properties:
        for seed in args.seeds:
            print(f"\n=== {prop.upper()} | scaffold split | seed={seed} ===", flush=True)
            frames = load_scaffold_frames(Path(args.data_dir).expanduser(), prop)
            save_manifest(frames, prop, seed, output_dir)
            target_column = base.PROPERTY_CONFIG[prop]["target_column"]
            group_cols = [f"Group {i}" for i in range(1, len(base.thermo.J_BIGGS_JOBACK_SMARTS) + 1)]
            print(
                f"Rows: train={len(frames['train'])}, calibration={len(frames['calibration'])}, "
                f"test={len(frames['test'])}; test scaffolds={frames['test']['scaffold'].nunique()}",
                flush=True,
            )
            for variant in args.variants:
                print(f"Running {variant}", flush=True)
                all_rows.append(run_one(
                    prop, variant, seed, frames, target_column, group_cols,
                    output_dir, args.epochs, args.patience, args.batch_size,
                    device, clip_norm,
                ))
                pd.DataFrame(all_rows).to_csv(output_dir / "scaffold_results.csv", index=False)

    results = pd.DataFrame(all_rows)
    results.to_csv(output_dir / "scaffold_results.csv", index=False)
    if not results.empty:
        metric_columns = [c for c in results.columns if c.startswith(("train_", "val_", "test_"))]
        summary = results.groupby(["property", "variant"], as_index=False)[metric_columns].agg(["mean", "std"])
        summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
        summary.to_csv(output_dir / "scaffold_summary.csv", index=False)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    print(f"\nSaved scaffold results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
