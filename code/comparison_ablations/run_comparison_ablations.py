#!/usr/bin/env python3
"""Controlled comparison and ablation experiments for JR-MPNN.

The default data and utility paths intentionally match the existing scripts in
the repository.  The implementation uses one common neural architecture for
all learned variants so that target parameterization, conditioning, and graph
pooling can be ablated without changing unrelated training choices.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool


SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_CANDIDATES = [
    Path(os.path.expanduser("~/deeplearnig/new")),
    Path(os.path.expanduser("~/deeplearning/new")),
    Path(os.path.expanduser("~/deeplearning/residual")),
]


def _default_subdirectory(name: str) -> str:
    for root in _ROOT_CANDIDATES:
        if (root / name).exists():
            return str(root / name)
    # Preserve the user's stated layout when the script is being copied to a
    # fresh cluster checkout and the directories are not visible yet.
    return str(_ROOT_CANDIDATES[0] / name)


DEFAULT_DATA_DIR = _default_subdirectory("data")
DEFAULT_UTILS_DIR = _default_subdirectory("utils")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

# Loaded after argument parsing so --utils-dir works as documented.
atom_features = bond_features = n_atom_features = n_bond_features = None
thermo = None


PROPERTY_CONFIG = {
    "nbp": {"target_column": "NBP", "data_stem": "NBP", "hidden_dim": 128, "layers": 5},
    "mp": {"target_column": "MP", "data_stem": "MP", "hidden_dim": 128, "layers": 5},
    "tc": {"target_column": "TC", "data_stem": "TC", "hidden_dim": 64, "layers": 3},
    "pc": {"target_column": "PC", "data_stem": "PC", "hidden_dim": 64, "layers": 3},
    "vc": {"target_column": "VC", "data_stem": "VC", "hidden_dim": 64, "layers": 3},
}


def load_dependencies(utils_dir: str) -> None:
    global atom_features, bond_features, n_atom_features, n_bond_features, thermo
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    from molecular_graph import atom_features as _atom_features
    from molecular_graph import bond_features as _bond_features
    from molecular_graph import n_atom_features as _n_atom_features
    from molecular_graph import n_bond_features as _n_bond_features
    import thermo as _thermo

    atom_features = _atom_features
    bond_features = _bond_features
    n_atom_features = _n_atom_features
    n_bond_features = _n_bond_features
    thermo = _thermo

DEFAULT_VARIANTS = [
    "joback_original",
    "joback_refit",
    "mpnn_control",
    "jrmpnn_full",
    "absolute_conditioned",
    "residual_unconditioned",
    "residual_joback_only",
    "residual_sum_pool",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility is more important than peak throughput for these runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def property_joback_prediction(smiles: str, prop: str) -> float:
    """Return Joback prediction in the same units used by the current scripts."""
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return np.nan
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return np.nan
        joback = thermo.Joback(smiles)
        counts = joback.counts
        if prop == "nbp":
            value = joback.Tb(counts)
        elif prop == "mp":
            value = joback.Tm(counts)
        elif prop == "tc":
            # This deliberately matches the existing tc/jrmpnn implementation.
            value = joback.Tc(counts)
        elif prop == "pc":
            value = joback.Pc(counts, Chem.AddHs(molecule).GetNumAtoms()) / 1e6
        elif prop == "vc":
            value = joback.Vc(counts) * 1e3
        else:
            raise ValueError(f"Unsupported property: {prop}")
        return float(value) if value is not None else np.nan
    except Exception:
        return np.nan


def add_joback_features(frame: pd.DataFrame, prop: str) -> pd.DataFrame:
    result = frame.copy()
    result["y_joback"] = result["SMILES"].apply(lambda value: property_joback_prediction(value, prop))
    group_count = len(thermo.J_BIGGS_JOBACK_SMARTS)
    counts = np.zeros((len(result), group_count), dtype=np.float32)
    for row_index, smiles in enumerate(result["SMILES"]):
        try:
            group_map = thermo.Joback(smiles).counts
            for group_index in range(group_count):
                counts[row_index, group_index] = group_map.get(group_index + 1, 0)
        except Exception:
            continue
    for group_index in range(group_count):
        result[f"Group {group_index + 1}"] = counts[:, group_index]
    return result


def load_property_frames(data_dir: str, prop: str) -> Tuple[pd.DataFrame, pd.DataFrame, str, List[str]]:
    config = PROPERTY_CONFIG[prop]
    train_path = Path(data_dir) / f"{config['data_stem']}_train.csv"
    test_path = Path(data_dir) / f"{config['data_stem']}_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {test_path}. "
            "Use --data-dir to override the existing ~/deeplearning/residual/data path."
        )
    train = add_joback_features(pd.read_csv(train_path), prop)
    test = add_joback_features(pd.read_csv(test_path), prop)
    train["y_joback"] = train["y_joback"].fillna(train["y_joback"].mean())
    test["y_joback"] = test["y_joback"].fillna(train["y_joback"].mean())
    group_cols = [f"Group {i}" for i in range(1, len(thermo.J_BIGGS_JOBACK_SMARTS) + 1)]
    return train, test, config["target_column"], group_cols


def fit_refitted_joback(train: pd.DataFrame, target_column: str, group_cols: Sequence[str]) -> LinearRegression:
    """Fit a training-only linear group-contribution baseline.

    The fitted model is intentionally reported separately from the original
    Joback correlation.  It is the fair GC-only baseline requested by Reviewer 2.
    """
    x = train[list(group_cols)].to_numpy(dtype=np.float64)
    y = train[target_column].to_numpy(dtype=np.float64)
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    return model


def mol_to_graph(smiles: str, y: float, y_joback: float, groups: Sequence[float]) -> Optional[Data]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumBonds() == 0:
        return None
    node_features = [atom_features(atom) for atom in molecule.GetAtoms()]
    edge_index_list: List[List[int]] = []
    edge_features: List[torch.Tensor] = []
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index_list += [[begin, end], [end, begin]]
        features = bond_features(bond)
        edge_features += [features, features]
    return Data(
        x=torch.stack(node_features).float(),
        edge_index=torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_features).float(),
        y=torch.tensor([float(y)], dtype=torch.float32),
        yj=torch.tensor([float(y_joback)], dtype=torch.float32),
        groups=torch.tensor(groups, dtype=torch.float32),
        smiles=smiles,
    )


def build_graphs(frame: pd.DataFrame, target_column: str, group_cols: Sequence[str]) -> List[Data]:
    graphs: List[Data] = []
    for _, row in frame.iterrows():
        condition = [float(row["y_joback"])] + [float(row[col]) for col in group_cols]
        graph = mol_to_graph(row["SMILES"], float(row[target_column]), float(row["y_joback"]), condition[1:])
        if graph is not None:
            graphs.append(graph)
    return graphs


class DMPNNLayer(MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add", flow="source_to_target")
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        src, tgt = edge_index
        incoming = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        incoming.index_add_(0, tgt, edge_attr)
        reverse = torch.arange(edge_attr.size(0), device=x.device) ^ 1
        message = incoming[src] - edge_attr[reverse]
        edge_new = self.edge_mlp(torch.cat([message, x[src], edge_attr], dim=1))
        aggregate = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        aggregate.index_add_(0, tgt, edge_new)
        node_new = self.node_mlp(torch.cat([x, aggregate], dim=1))
        return node_new + x, edge_new + edge_attr


class ControlledMPNN(nn.Module):
    """One common MPNN with configurable target, conditioning, and pooling."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        target_mode: str,
        condition_mode: str,
        pooling: str,
        group_count: int,
    ):
        super().__init__()
        self.target_mode = target_mode
        self.condition_mode = condition_mode
        self.pooling = pooling
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([DMPNNLayer(hidden_dim) for _ in range(num_layers)])

        pool_multiplier = {"sum": 1, "meanmax": 2, "triple": 3}[pooling]
        condition_dim = {"none": 0, "joback": 1, "groups": group_count, "full": group_count + 1}[condition_mode]
        self.condition_dim = condition_dim
        readout_dim = pool_multiplier * hidden_dim + condition_dim
        self.fc1 = nn.Linear(readout_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, max(1, hidden_dim // 2))
        self.fc4 = nn.Linear(max(1, hidden_dim // 2), 1)
        self.skip = nn.Linear(readout_dim, hidden_dim)
        self.drop = nn.Dropout(0.2)
        self.act = nn.ReLU()

    def forward(self, data: Data):
        x = self.node_embedding(data.x.float())
        edge_attr = self.edge_embedding(data.edge_attr.float())
        for layer in self.layers:
            x, edge_attr = layer(x, data.edge_index, edge_attr)

        pooled = {
            "sum": [global_add_pool(x, data.batch)],
            "meanmax": [global_mean_pool(x, data.batch), global_max_pool(x, data.batch)],
            "triple": [global_add_pool(x, data.batch), global_mean_pool(x, data.batch), global_max_pool(x, data.batch)],
        }[self.pooling]
        graph_embedding = torch.cat(pooled, dim=-1)

        if self.condition_mode == "none":
            condition = None
        elif self.condition_mode == "joback":
            condition = data.yj.view(-1, 1)
        elif self.condition_mode == "groups":
            condition = data.groups.view(-1, self.condition_dim)
        else:
            condition = torch.cat([data.yj.view(-1, 1), data.groups.view(-1, self.condition_dim - 1)], dim=-1)
        joined = graph_embedding if condition is None else torch.cat([graph_embedding, condition], dim=-1)

        hidden = self.drop(self.act(self.fc1(joined)))
        hidden = self.drop(self.act(self.fc2(hidden) + self.skip(joined)))
        hidden = self.act(self.fc3(hidden))
        output = self.fc4(hidden).squeeze(-1)
        if self.target_mode == "residual":
            prediction = data.yj.view(-1) + output
        else:
            prediction = output
        return prediction, output, graph_embedding


def variant_spec(name: str) -> Dict[str, str]:
    specs = {
        "mpnn_control": {"target": "absolute", "condition": "none", "pooling": "sum"},
        "jrmpnn_full": {"target": "residual", "condition": "full", "pooling": "triple"},
        "absolute_conditioned": {"target": "absolute", "condition": "full", "pooling": "triple"},
        "residual_unconditioned": {"target": "residual", "condition": "none", "pooling": "triple"},
        "residual_joback_only": {"target": "residual", "condition": "joback", "pooling": "triple"},
        "residual_sum_pool": {"target": "residual", "condition": "full", "pooling": "sum"},
    }
    if name not in specs:
        raise ValueError(f"Unknown learned variant: {name}")
    return specs[name]


def metric_dict(y_true: Sequence[float], y_pred: Sequence[float], prefix: str) -> Dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        f"{prefix}_r2": float(r2_score(true, pred)),
        f"{prefix}_mae": float(mean_absolute_error(true, pred)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(true, pred))),
        f"{prefix}_mape": float(mean_absolute_percentage_error(true, pred)),
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    true_values: List[float] = []
    predictions: List[float] = []
    smiles: List[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction, _, _ = model(batch)
            true_values.extend(batch.y.view(-1).cpu().numpy().tolist())
            predictions.extend(prediction.cpu().numpy().tolist())
            smiles.extend(list(batch.smiles))
    return np.asarray(true_values), np.asarray(predictions), smiles


def train_model(
    model: ControlledMPNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    clip_norm: Optional[float],
) -> ControlledMPNN:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    loss_fn = nn.SmoothL1Loss()
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            prediction, raw_output, _ = model(batch)
            target = batch.y.view(-1) - batch.yj.view(-1) if model.target_mode == "residual" else batch.y.view(-1)
            predicted_target = raw_output if model.target_mode == "residual" else prediction
            loss = loss_fn(predicted_target, target)
            if model.target_mode == "residual":
                loss = loss + 1e-5 * (raw_output ** 2).mean()
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
        scheduler.step()

        y_val, pred_val, _ = evaluate_model(model, val_loader, device)
        val_mae = mean_absolute_error(y_val, pred_val)
        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    return model


def evaluate_baseline(
    frame: pd.DataFrame,
    target_column: str,
    model_name: str,
    refit_model: Optional[LinearRegression] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y_true = frame[target_column].to_numpy(dtype=float)
    if model_name == "joback_original":
        y_pred = frame["y_joback"].to_numpy(dtype=float)
    elif model_name == "joback_refit" and refit_model is not None:
        group_cols = [col for col in frame.columns if col.startswith("Group ")]
        y_pred = refit_model.predict(frame[group_cols].to_numpy(dtype=float))
    else:
        raise ValueError(f"Invalid baseline: {model_name}")
    return y_true, np.asarray(y_pred), frame["SMILES"].astype(str).tolist()


def run_one(
    prop: str,
    variant: str,
    seed: int,
    fraction: float,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
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
    n_subset = max(2, int(round(len(train_frame) * fraction)))
    subset_indices = np.random.default_rng(seed).choice(len(train_frame), size=n_subset, replace=False)
    subset = train_frame.iloc[np.sort(subset_indices)].reset_index(drop=True)
    refit = fit_refitted_joback(subset, target_column, group_cols)

    row: Dict[str, object] = {
        "property": prop,
        "variant": variant,
        "seed": seed,
        "fraction": fraction,
        "n_train": len(subset),
        "device": str(device),
    }

    if variant in {"joback_original", "joback_refit"}:
        y_true, y_pred, smiles = evaluate_baseline(test_frame, target_column, variant, refit)
        row.update(metric_dict(y_true, y_pred, "test"))
        row["n_test"] = len(y_true)
        pred_dir = output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"SMILES": smiles, "y_true": y_true, "y_pred": y_pred}).to_csv(
            pred_dir / f"{prop}_{variant}_seed{seed}_fraction{fraction:.2f}.csv", index=False
        )
        if variant == "joback_refit":
            parameter_dir = output_dir / "refitted_joback_parameters"
            parameter_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"group": ["intercept"] + list(group_cols), "coefficient": [refit.intercept_] + list(refit.coef_)}).to_csv(
                parameter_dir / f"{prop}_seed{seed}_fraction{fraction:.2f}.csv", index=False
            )
        return row

    specs = variant_spec(variant)
    subset_graphs = build_graphs(subset, target_column, group_cols)
    test_graphs = build_graphs(test_frame, target_column, group_cols)
    if len(subset_graphs) < 4 or len(test_graphs) == 0:
        raise RuntimeError(f"Too few valid molecular graphs for {prop} ({len(subset_graphs)} train, {len(test_graphs)} test)")

    train_idx, val_idx = train_test_split(np.arange(len(subset_graphs)), test_size=0.15, random_state=seed)
    train_graphs = [subset_graphs[index] for index in train_idx]
    val_graphs = [subset_graphs[index] for index in val_idx]
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    config = PROPERTY_CONFIG[prop]
    model = ControlledMPNN(
        node_dim=n_atom_features(),
        edge_dim=n_bond_features(),
        hidden_dim=config["hidden_dim"],
        num_layers=config["layers"],
        target_mode=specs["target"],
        condition_mode=specs["condition"],
        pooling=specs["pooling"],
        group_count=len(group_cols),
    ).to(device)
    model = train_model(
        model, train_loader, val_loader, device,
        epochs=epochs,
        patience=patience,
        learning_rate=2e-4,
        weight_decay=1e-4,
        clip_norm=clip_norm,
    )

    y_train, pred_train, _ = evaluate_model(model, train_loader, device)
    y_val, pred_val, _ = evaluate_model(model, val_loader, device)
    y_test, pred_test, test_smiles = evaluate_model(model, test_loader, device)
    row.update(metric_dict(y_train, pred_train, "train"))
    row.update(metric_dict(y_val, pred_val, "val"))
    row.update(metric_dict(y_test, pred_test, "test"))
    row.update({"target_mode": specs["target"], "condition_mode": specs["condition"], "pooling": specs["pooling"]})
    row["n_test"] = len(y_test)

    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SMILES": test_smiles, "y_true": y_test, "y_pred": pred_test}).to_csv(
        pred_dir / f"{prop}_{variant}_seed{seed}_fraction{fraction:.2f}.csv", index=False
    )
    return row


def summarize(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty:
        return
    metric_columns = [col for col in results.columns if col.startswith(("train_", "val_", "test_"))]
    summary = results.groupby(["property", "variant", "fraction"], as_index=False)[metric_columns].agg(["mean", "std"])
    summary.columns = ["_".join(column).strip("_") if isinstance(column, tuple) else column for column in summary.columns]
    summary.to_csv(output_dir / "comparison_ablations_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--property", choices=[*PROPERTY_CONFIG.keys(), "all"], default="all")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=[*DEFAULT_VARIANTS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0])
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--utils-dir", default=DEFAULT_UTILS_DIR)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-gradient-clipping", action="store_true")
    parser.add_argument("--list-variants", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_variants:
        for variant in DEFAULT_VARIANTS:
            print(variant)
        return

    # Honor an explicitly supplied utility path while retaining a local fallback
    # when this repository is run outside the original cluster checkout.
    utils_dir = Path(args.utils_dir).expanduser()
    if not utils_dir.exists() and (SCRIPT_DIR.parent / "utils").exists():
        utils_dir = SCRIPT_DIR.parent / "utils"
    load_dependencies(str(utils_dir))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    properties = list(PROPERTY_CONFIG) if args.property == "all" else [args.property]
    clip_norm = None if args.no_gradient_clipping else 1.0

    all_rows: List[Dict[str, object]] = []
    for prop in properties:
        print(f"\n=== {prop.upper()} ===", flush=True)
        train_frame, test_frame, target_column, group_cols = load_property_frames(args.data_dir, prop)
        print(f"Loaded {len(train_frame)} training and {len(test_frame)} test rows", flush=True)
        for fraction in args.fractions:
            if not 0 < fraction <= 1:
                raise ValueError(f"Fractions must be in (0, 1], got {fraction}")
            for seed in args.seeds:
                for variant in args.variants:
                    print(f"Running {prop} | {variant} | seed={seed} | fraction={fraction:.2f}", flush=True)
                    row = run_one(
                        prop, variant, seed, fraction,
                        train_frame, test_frame, target_column, group_cols,
                        output_dir, args.epochs, args.patience, args.batch_size,
                        device, clip_norm,
                    )
                    all_rows.append(row)
                    pd.DataFrame(all_rows).to_csv(output_dir / "comparison_ablations_results.csv", index=False)

    results = pd.DataFrame(all_rows)
    results.to_csv(output_dir / "comparison_ablations_results.csv", index=False)
    summarize(results, output_dir)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    print(f"\nSaved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
