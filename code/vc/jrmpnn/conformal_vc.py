#!/usr/bin/env python3
"""
Adaptive Conformal Prediction for JR-MPNN (NBP)
================================================
Split conformal prediction using k-NN distance in embedding space
as the adaptive difficulty measure.

Coverage guarantee:
    P(y ∈ C(x)) ≥ 1 - α
    holds because the calibration set is exchangeable with the test set
    and was never used during model training.

Difficulty measure:
    d(x) = mean distance to k nearest neighbours in training embedding space
    - Molecules close to training data  → narrow intervals
    - Molecules far from training data  → wide intervals

Three-way split: train / calibration / test
    Train       → train the model, fit k-NN reference
    Calibration → compute non-conformity scores → quantile q
    Test        → compute intervals using q and k-NN difficulty
"""

import warnings
warnings.filterwarnings('ignore')
import torch
torch.set_warn_always(False)
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
import thermo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (r2_score, mean_absolute_error,
                             mean_squared_error, mean_absolute_percentage_error)
import sys
import os
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
SEED = int(os.environ.get('SEED', '42'))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f'seed: {SEED}', flush=True)

# --- UTILS ---
root_dir = os.path.expanduser(os.environ.get('ROOT_DIR', '~/deeplearnig/new'))
if not os.path.isdir(root_dir):
    root_dir = os.path.expanduser('~/deeplearning/new')
utils_path = os.path.join(root_dir, 'utils')
sys.path.insert(0, utils_path)
from molecular_graph import atom_features, bond_features, n_atom_features, n_bond_features

print('running conformal_vc.py', flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}', flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

data_dir   = os.path.join(root_dir, 'data')
df_train   = pd.read_csv(os.path.join(data_dir, 'VC_train.csv'))
df_cal     = pd.read_csv(os.path.join(data_dir, 'VC_calibration.csv'))
df_test    = pd.read_csv(os.path.join(data_dir, 'VC_test.csv'))

print(f"Train: {len(df_train)} | Calibration: {len(df_cal)} | Test: {len(df_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# JOBACK
# ─────────────────────────────────────────────────────────────────────────────

def JobackVc(smiles):
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return np.nan
        J = thermo.Joback(smiles)
        val = J.Vc(J.counts)   # change to J.Tb(J.counts) if you want boiling point
        if val is None:
            return np.nan
        return round(float(val), 8) * 1e3
    except Exception:
        return np.nan

def compute_joback_pred(df):
    df = df.copy()
    df['y_joback'] = df['SMILES'].apply(JobackVc)
    return df

df_train = compute_joback_pred(df_train)
df_cal   = compute_joback_pred(df_cal)
df_test  = compute_joback_pred(df_test)

# Fill NaN with training mean only — never use cal/test statistics
joback_mean          = df_train['y_joback'].mean()
df_train['y_joback'] = df_train['y_joback'].fillna(joback_mean)
df_cal['y_joback']   = df_cal['y_joback'].fillna(joback_mean)
df_test['y_joback']  = df_test['y_joback'].fillna(joback_mean)

fg_list = thermo.J_BIGGS_JOBACK_SMARTS

def joback_func_groups(df):
    df        = df.copy()
    fg_counts = np.zeros((len(df), len(fg_list)))
    for i, smiles in enumerate(df['SMILES']):
        try:
            counts = thermo.Joback(smiles).counts
            for j in range(len(fg_list)):
                fg_counts[i, j] = counts.get(j + 1, 0)
        except Exception:
            pass
    for j in range(len(fg_list)):
        df[f'Group {j+1}'] = fg_counts[:, j]
    return df

df_train = joback_func_groups(df_train)
df_cal   = joback_func_groups(df_cal)
df_test  = joback_func_groups(df_test)

group_cols = [f'Group {i}' for i in range(1, len(fg_list) + 1)]

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def mol_to_graph(smiles, y, y_joback, c_vec):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feats                   = [atom_features(a) for a in mol.GetAtoms()]
    edge_index_list, edge_feats  = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index_list += [[i, j], [j, i]]
        bf = bond_features(bond)
        edge_feats += [bf, bf]
    if not edge_feats:
        return None
    return Data(
        x          = torch.stack(atom_feats).float(),
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
        edge_attr  = torch.stack(edge_feats).float(),
        y          = torch.tensor([float(y)],        dtype=torch.float32),
        yj         = torch.tensor([float(y_joback)], dtype=torch.float32),
        c          = torch.tensor(c_vec,             dtype=torch.float32),
    )

def build_graphs(df):
    graphs = []
    for _, row in df.iterrows():
        yj     = float(row.y_joback)
        c      = [yj] + [float(row[g]) for g in group_cols]
        d      = mol_to_graph(row.SMILES, float(row.VC), yj, c)
        if d is not None:
            graphs.append(d)
    return graphs

def valid_graph_smiles(df):
    values = []
    for smiles in df.SMILES:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None and molecule.GetNumBonds() > 0:
            values.append(smiles)
    return np.asarray(values, dtype=object)

train_graphs = build_graphs(df_train)
cal_graphs   = build_graphs(df_cal)
test_graphs  = build_graphs(df_test)
cal_smiles   = valid_graph_smiles(df_cal)
test_smiles  = valid_graph_smiles(df_test)

print(f"Graphs — Train: {len(train_graphs)} | Cal: {len(cal_graphs)} | Test: {len(test_graphs)}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (identical to jrmpnn_v4.py — forward returns y_hat, r_hat, h_G)
# ─────────────────────────────────────────────────────────────────────────────

class DMPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add', flow='source_to_target')
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + node_dim + hidden_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        x, edge_attr  = x.float(), edge_attr.float()
        edge_attr     = self.edge_update(x, edge_index, edge_attr)
        x_new         = self.node_update(x, edge_index, edge_attr)
        return x_new, edge_attr

    def edge_update(self, x, edge_index, edge_attr):
        src, tgt      = edge_index
        incoming      = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        incoming.index_add_(0, tgt, edge_attr)
        rev           = torch.arange(edge_attr.size(0), device=x.device) ^ 1
        msg           = incoming[src] - edge_attr[rev]
        new_e         = self.edge_mlp(torch.cat([msg, x[src], edge_attr], dim=1))
        return new_e + edge_attr

    def node_update(self, x, edge_index, edge_attr):
        _, col        = edge_index
        agg           = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        agg.index_add_(0, col, edge_attr)
        new_x         = self.node_mlp(torch.cat([x, agg], dim=1))
        return new_x + x


class JRMPNN(nn.Module):
    """
    Joback-Residual MPNN — triple pooling + deep readout with skip connection.
    forward() returns (y_hat, r_hat, h_G).
    h_G  [B, 3*hidden_dim]  is the pre-readout embedding used for conformal UQ.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, cond_dim, output_dim=1):
        super().__init__()
        self.cond_dim   = cond_dim
        self.hidden_dim = hidden_dim

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim,  hidden_dim)
        self.dmpnn_layers   = nn.ModuleList(
            [DMPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        in_dim    = 3 * hidden_dim + cond_dim
        self.fc1  = nn.Linear(in_dim,        hidden_dim)
        self.fc2  = nn.Linear(hidden_dim,    hidden_dim)
        self.fc3  = nn.Linear(hidden_dim,    hidden_dim // 2)
        self.fc4  = nn.Linear(hidden_dim // 2, output_dim)
        self.skip = nn.Linear(in_dim,        hidden_dim)
        self.drop = nn.Dropout(0.2)
        self.act  = nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.edge_attr.float()
        batch, c, yj = data.batch, data.c, data.yj

        x         = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        for layer in self.dmpnn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        h_G = torch.cat([
            global_add_pool(x,  batch),
            global_mean_pool(x, batch),
            global_max_pool(x,  batch),
        ], dim=-1)                                    # [B, 3*hidden_dim]

        B = h_G.size(0)
        if c.dim() == 1:
            c = c.view(B, self.cond_dim)

        h_cat = torch.cat([h_G, c], dim=-1)
        h     = self.drop(self.act(self.fc1(h_cat)))
        h     = self.drop(self.act(self.fc2(h) + self.skip(h_cat)))
        h     = self.act(self.fc3(h))
        r_hat = self.fc4(h).squeeze(-1)

        return yj.view(-1) + r_hat, r_hat, h_G       # (y_hat, r_hat, h_G)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

cond_dim = train_graphs[0].c.numel()
node_dim = n_atom_features()
edge_dim = n_bond_features()

model = JRMPNN(
    node_dim=node_dim, edge_dim=edge_dim,
    hidden_dim=64, num_layers=3, cond_dim=cond_dim,
).to(device)

checkpoint_override = os.environ.get('CHECKPOINT_PATH', '').strip()
checkpoint_path = os.path.expanduser(
    checkpoint_override or os.path.join(script_dir, 'jrmpnn_vc.pth')
)
print(f'checkpoint: {checkpoint_path}', flush=True)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("Model loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT EMBEDDINGS + PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract(model, graphs, device, batch_size=32):
    loader                       = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    y_preds, y_trues, embeddings = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_hat, _, h_G = model(batch)
            y_preds.append(y_hat.cpu().numpy().flatten())
            y_trues.append(batch.y.cpu().numpy().flatten())
            embeddings.append(h_G.cpu().numpy())
    return np.concatenate(y_preds), np.concatenate(y_trues), np.concatenate(embeddings)

print("\nExtracting embeddings...")
train_pred, train_true, train_emb = extract(model, train_graphs, device)
cal_pred,   cal_true,   cal_emb   = extract(model, cal_graphs,   device)
test_pred,  test_true,  test_emb  = extract(model, test_graphs,  device)

print(f"Embedding dim : {train_emb.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE CONFORMAL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

ALPHA        = 0.05    # miscoverage → 95% target coverage
K_NEIGHBOURS = int(os.environ.get('K_NEIGHBOURS', '10'))  # nearest neighbours for difficulty
EPSILON      = 1e-3    # prevents division by zero

# Step 1 — fit k-NN on training embeddings only
print("\nFitting k-NN on training embeddings...")
knn = NearestNeighbors(n_neighbors=K_NEIGHBOURS, metric='euclidean', n_jobs=-1)
knn.fit(train_emb)

# Step 2 — calibration difficulty: mean distance to k training neighbours
cal_difficulty = knn.kneighbors(cal_emb)[0].mean(axis=1)

# Step 3 — non-conformity scores:  s_i = |y_i - ŷ_i| / (d_i + ε)
cal_scores = np.abs(cal_true - cal_pred) / (cal_difficulty + EPSILON)

# Step 4 — quantile with finite-sample correction
n     = len(cal_scores)
level = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
q     = np.quantile(cal_scores, level)

# Standard fixed-width split-conformal baseline using the same calibration set.
cal_abs_errors = np.abs(cal_true - cal_pred)
q_fixed = np.quantile(cal_abs_errors, level)
fixed_lower = test_pred - q_fixed
fixed_upper = test_pred + q_fixed
fixed_covered = (test_true >= fixed_lower) & (test_true <= fixed_upper)
fixed_widths = fixed_upper - fixed_lower

print(f"\nConformal calibration:")
print(f"  Calibration N          : {n}")
print(f"  Quantile level         : {level:.4f}")
print(f"  Non-conformity quantile: {q:.4f}")
print(f"  Score mean / std       : {cal_scores.mean():.4f} / {cal_scores.std():.4f}")
print(f"  Fixed-width quantile   : {q_fixed:.4f}")

# Step 5 — adaptive intervals on test set
test_difficulty = knn.kneighbors(test_emb)[0].mean(axis=1)
half_width      = q * (test_difficulty + EPSILON)
lower           = test_pred - half_width
upper           = test_pred + half_width

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

covered       = (test_true >= lower) & (test_true <= upper)
empirical_cov = covered.mean()
widths        = upper - lower

r2   = r2_score(test_true, test_pred)
mae  = mean_absolute_error(test_true, test_pred)
rmse = np.sqrt(mean_squared_error(test_true, test_pred))
mape = mean_absolute_percentage_error(test_true, test_pred)

print(f"\n── Point Prediction Metrics ────────────────────────────────")
print(f"  R²   : {r2:.4f}")
print(f"  MAE  : {mae:.4f} K")
print(f"  RMSE : {rmse:.4f} K")
print(f"  MAPE : {mape:.4f}")

print(f"\n── Conformal Prediction Intervals ──────────────────────────")
print(f"  Target coverage   : {(1 - ALPHA) * 100:.1f}%")
print(f"  Empirical coverage: {empirical_cov * 100:.2f}%")
print(f"  Mean width        : {widths.mean():.2f} K")
print(f"  Median width      : {np.median(widths):.2f} K")
print(f"  Covered           : {covered.sum()} / {len(test_true)}")

# ─────────────────────────────────────────────────────────────────────────────
# TRUST SCORES  (1 = most confident, 0 = least confident)
# ─────────────────────────────────────────────────────────────────────────────

w_min, w_max = widths.min(), widths.max()
trust_scores = 1.0 - (widths - w_min) / (w_max - w_min + EPSILON)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

df_results = pd.DataFrame({
    'SMILES'        : test_smiles,
    'y_true'        : test_true,
    'y_pred'        : test_pred,
    'lower'         : lower,
    'upper'         : upper,
    'interval_width': widths,
    'knn_difficulty': test_difficulty,
    'trust_score'   : trust_scores,
    'covered'       : covered.astype(int),
    'fixed_lower'   : fixed_lower,
    'fixed_upper'   : fixed_upper,
    'fixed_interval_width': fixed_widths,
    'fixed_covered' : fixed_covered.astype(int),
})
df_results.to_csv('jrmpnn_vc_conformal_results.csv', index=False)
print("\nSaved: jrmpnn_vc_conformal_results.csv")

df_calibration_results = pd.DataFrame({
    'SMILES'          : cal_smiles,
    'y_true'          : cal_true,
    'y_pred'          : cal_pred,
    'knn_difficulty'  : cal_difficulty,
    'absolute_error'  : np.abs(cal_true - cal_pred),
})
df_calibration_results.to_csv('jrmpnn_vc_calibration_results.csv', index=False)
print("Saved: jrmpnn_vc_calibration_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams["font.family"] = "Arial"

# ── Figure 1: parity with adaptive intervals + width vs difficulty ────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sort_idx = np.argsort(test_difficulty)
sc = axes[0].scatter(
    test_true[sort_idx], test_pred[sort_idx],
    c=test_difficulty[sort_idx], cmap='viridis', alpha=0.7, s=35, zorder=3,
)
for i in range(len(test_true)):
    axes[0].plot(
        [test_true[i], test_true[i]], [lower[i], upper[i]],
        color='gray', alpha=0.25, linewidth=0.7, zorder=1,
    )
axes[0].plot([200, 1200], [200, 1200], 'k--', linewidth=1.5)
axes[0].set_xlim(200, 1200); axes[0].set_ylim(200, 1200)
axes[0].set_xlabel("Experimental NBP (K)", fontsize=14)
axes[0].set_ylabel("Predicted NBP (K)",    fontsize=14)
axes[0].set_title("Adaptive Prediction Intervals", fontsize=13)
axes[0].set_aspect('equal')
plt.colorbar(sc, ax=axes[0], label='k-NN Difficulty')
axes[0].text(
    0.05, 0.95,
    f"Coverage: {empirical_cov*100:.1f}%\n"
    f"Mean width: {widths.mean():.1f} K\n"
    f"MAE: {mae:.2f} K\n"
    f"R² = {r2:.3f}",
    transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
)

z = np.polyfit(test_difficulty, widths, 1)
x_line = np.linspace(test_difficulty.min(), test_difficulty.max(), 100)
axes[1].scatter(test_difficulty, widths, c=test_difficulty,
                cmap='viridis', alpha=0.6, s=25)
axes[1].plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=1.5, label='Linear fit')
axes[1].set_xlabel("k-NN Difficulty",               fontsize=14)
axes[1].set_ylabel("Prediction Interval Width (K)", fontsize=14)
axes[1].set_title("Interval Width vs Molecular Difficulty", fontsize=13)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig("jrmpnn_vc_conformal_intervals.png", format="png",
            dpi=500, bbox_inches="tight")
plt.close()
print("Saved: jrmpnn_vc_conformal_intervals.pdf")

# ── Figure 2: trust score distribution + trust vs error ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(trust_scores, bins=25, color='#1e488f', alpha=0.8, edgecolor='black')
axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold = 0.5')
axes[0].set_xlabel("Trust Score", fontsize=13)
axes[0].set_ylabel("Count",       fontsize=13)
axes[0].set_title("Distribution of Trust Scores", fontsize=13)
axes[0].legend(fontsize=11)

abs_errors = np.abs(test_true - test_pred)
axes[1].scatter(trust_scores, abs_errors, c='#2ca02c', alpha=0.6, s=25)
z2 = np.polyfit(trust_scores, abs_errors, 1)
xs = np.linspace(trust_scores.min(), trust_scores.max(), 100)
axes[1].plot(xs, np.poly1d(z2)(xs), 'r--', linewidth=1.5)
axes[1].set_xlabel("Trust Score", fontsize=13)
axes[1].set_ylabel("|Error| (K)", fontsize=13)
axes[1].set_title("Trust Score vs Prediction Error", fontsize=13)

plt.tight_layout()
plt.savefig("jrmpnn_vc_trust_scores.png", format="png",
            dpi=500, bbox_inches="tight")
plt.close()
print("Saved: jrmpnn_vc_trust_scores.png")

# ─────────────────────────────────────────────────────────────────────────────
# LOW-TRUST OUTLIER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

LOW_TRUST_THRESHOLD = 0.3
df_sorted = df_results.sort_values('trust_score', ascending=False)

print(f"\n── Top 10 most trustworthy predictions ─────────────────────")
print(df_sorted.head(10)[
    ['y_true', 'y_pred', 'trust_score', 'interval_width', 'knn_difficulty']
].to_string(index=False))

low_trust = df_sorted[df_sorted['trust_score'] < LOW_TRUST_THRESHOLD]
print(f"\n── Low-trust molecules (trust < {LOW_TRUST_THRESHOLD}) ──────────────────────")
print(f"  Count: {len(low_trust)}")
if len(low_trust) > 0:
    print(low_trust[
        ['y_true', 'y_pred', 'trust_score', 'interval_width', 'knn_difficulty']
    ].to_string(index=False))
    low_trust.to_csv('jrmpnn_vc_low_trust_outliers.csv', index=False)
    print("Saved: jrmpnn_vc_low_trust_outliers.csv")

print("\nDone.")
