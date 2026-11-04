#!/usr/bin/env python3
"""
JR-MPNN v4 — Joback-Residual Message Passing Neural Network (NBP)
==================================================================
Architecture:
    - Directed message passing with backtracking prevention (XOR trick)
    - Triple pooling: global_add + global_mean + global_max
    - Deep readout MLP (4 layers) with skip connection
    - Joback conditioning vector concatenated before readout
    - Residual formulation: y_hat = y_joback + r_hat

forward() returns (y_hat, r_hat, h_G)
    h_G  [B, 3*hidden_dim]  — pre-readout embedding, used by conformal_nbp.py
    for k-NN difficulty estimation.
"""

import warnings
warnings.filterwarnings('ignore')
import torch
torch.set_warn_always(False)
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import (r2_score, mean_absolute_error,
                             mean_squared_error, mean_absolute_percentage_error)
from rdkit import Chem
import thermo
import sys
import os

# --- UTILS ---
utils_path = os.path.expanduser('~/deeplearning/residual/utils')
sys.path.insert(0, utils_path)
from molecular_graph import atom_features, bond_features, n_atom_features, n_bond_features

print('running jrmpnn_v4.py', flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}', flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

data_dir = os.path.expanduser('~/deeplearning/new/data')
df_train = pd.read_csv(os.path.join(data_dir, 'NBP_train.csv'))
df_test  = pd.read_csv(os.path.join(data_dir, 'NBP_test.csv'))

# ─────────────────────────────────────────────────────────────────────────────
# JOBACK
# ─────────────────────────────────────────────────────────────────────────────

def JobackTb(smiles):
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return np.nan
        J   = thermo.Joback(smiles)
        val = J.Tb(J.counts)
        return round(float(val), 2) if val is not None else np.nan
    except Exception:
        return np.nan

def compute_joback_pred(df):
    df = df.copy()
    df['y_joback'] = df['SMILES'].apply(JobackTb)
    return df

df_train = compute_joback_pred(df_train)
df_test  = compute_joback_pred(df_test)

joback_mean          = df_train['y_joback'].mean()
df_train['y_joback'] = df_train['y_joback'].fillna(joback_mean)
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
df_test  = joback_func_groups(df_test)

group_cols = [f'Group {i}' for i in range(1, len(fg_list) + 1)]

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def mol_to_graph(smiles, y, y_joback, c_vec):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feats                  = [atom_features(a) for a in mol.GetAtoms()]
    edge_index_list, edge_feats = [], []
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
        yj = float(row.y_joback)
        c  = [yj] + [float(row[g]) for g in group_cols]
        d  = mol_to_graph(row.SMILES, float(row.NBP), yj, c)
        if d is not None:
            graphs.append(d)
    return graphs

train_data_graphs = build_graphs(df_train)
test_data_graphs  = build_graphs(df_test)

print(f"Training graphs: {len(train_data_graphs)}")
print(f"Test graphs    : {len(test_data_graphs)}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DMPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add', flow='source_to_target')
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        x, edge_attr  = x.float(), edge_attr.float()
        edge_attr     = self.edge_update(x, edge_index, edge_attr)
        x_new         = self.node_update(x, edge_index, edge_attr)
        return x_new, edge_attr

    def edge_update(self, x, edge_index, edge_attr):
        src, tgt  = edge_index
        incoming  = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        incoming.index_add_(0, tgt, edge_attr)
        rev       = torch.arange(edge_attr.size(0), device=x.device) ^ 1
        msg       = incoming[src] - edge_attr[rev]          # backtracking prevention
        new_e     = self.edge_mlp(torch.cat([msg, x[src], edge_attr], dim=1))
        return new_e + edge_attr                             # residual

    def node_update(self, x, edge_index, edge_attr):
        _, col = edge_index
        agg    = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        agg.index_add_(0, col, edge_attr)
        new_x  = self.node_mlp(torch.cat([x, agg], dim=1))
        return new_x + x                                     # residual


class JRMPNN(nn.Module):
    """
    Joback-Residual MPNN (JR-MPNN)

    Triple pooling:  global_add + global_mean + global_max → [B, 3*hidden_dim]
    Conditioning:    concatenate with Joback vector c = [y_joback, g1..g41]
    Readout:         4-layer MLP with skip connection
    Output:          y_hat = y_joback + r_hat

    forward() returns (y_hat, r_hat, h_G)
        y_hat  — final boiling point prediction
        r_hat  — learned residual correction
        h_G    — triple-pooled graph embedding [B, 3*hidden_dim]
                 used by conformal_nbp.py for k-NN difficulty estimation
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

        # Triple pooling
        h_G = torch.cat([
            global_add_pool(x,  batch),    # size-sensitive
            global_mean_pool(x, batch),    # size-invariant
            global_max_pool(x,  batch),    # feature-dominant
        ], dim=-1)                          # [B, 3*hidden_dim]

        B = h_G.size(0)
        if c.dim() == 1:
            c = c.view(B, self.cond_dim)

        # Concatenate with Joback conditioning vector
        h_cat = torch.cat([h_G, c], dim=-1)

        # Deep readout with skip connection
        h     = self.drop(self.act(self.fc1(h_cat)))
        h     = self.drop(self.act(self.fc2(h) + self.skip(h_cat)))
        h     = self.act(self.fc3(h))
        r_hat = self.fc4(h).squeeze(-1)

        y_hat = yj.view(-1) + r_hat
        return y_hat, r_hat, h_G    # h_G exposed for conformal UQ


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, scheduler=None):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch        = batch.to(device)
        optimizer.zero_grad()
        y_hat, r_hat, _ = model(batch)              # unpack 3 values

        # Loss on residuals
        r_true = batch.y.view(-1) - batch.yj.view(-1)
        loss   = loss_fn(r_hat, r_true)

        # Soft regularisation: penalise large residuals
        loss = loss + 1e-5 * (r_hat ** 2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(loader)


def evaluate_loader(model, loader):
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            y_hat, _, _ = model(batch)              # unpack 3 values
            y_preds.extend(y_hat.cpu().numpy().flatten())
            y_trues.extend(batch.y.cpu().numpy().flatten())
    return np.array(y_trues), np.array(y_preds)


def train_validate(model, train_loader, val_loader, optimizer, loss_fn,
                   epochs=800, scheduler=None):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scheduler
        )
        if (epoch + 1) % 50 == 0:
            y_true, y_pred = evaluate_loader(model, val_loader)
            val_mae = mean_absolute_error(y_true, y_pred)
            print(f"  Epoch {epoch+1:4d} | train loss: {train_loss:.4f} "
                  f"| val MAE: {val_mae:.4f} K", flush=True)
        train_losses.append(train_loss)
    return train_losses


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

cond_dim     = train_data_graphs[0].c.numel()
node_dim     = n_atom_features()
edge_dim     = n_bond_features()

hidden_size  = 128
num_layers   = 5
output_dim   = 1
lr           = 2e-4
weight_decay = 1e-4
epochs       = 800

# ─────────────────────────────────────────────────────────────────────────────
# 5-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_train_results, mae_train_results   = [], []
rmse_train_results, mape_train_results = [], []
r2_val_results, mae_val_results       = [], []
rmse_val_results, mape_val_results    = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_data_graphs), start=1):
    print(f"\nFold {fold}/5 starting...", flush=True)

    train_subset = [train_data_graphs[i] for i in train_idx]
    val_subset   = [train_data_graphs[i] for i in val_idx]

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=32, shuffle=False)

    model = JRMPNN(
        node_dim=node_dim, edge_dim=edge_dim,
        hidden_dim=hidden_size, num_layers=num_layers, cond_dim=cond_dim,
    ).to(device)

    loss_fn   = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    train_validate(model, train_loader, val_loader, optimizer, loss_fn,
                   epochs=epochs, scheduler=scheduler)

    y_train_true, y_train_pred = evaluate_loader(model, train_loader)
    y_val_true,   y_val_pred   = evaluate_loader(model, val_loader)

    r2_train_results.append(r2_score(y_train_true, y_train_pred))
    mae_train_results.append(mean_absolute_error(y_train_true, y_train_pred))
    rmse_train_results.append(np.sqrt(mean_squared_error(y_train_true, y_train_pred)))
    mape_train_results.append(mean_absolute_percentage_error(y_train_true, y_train_pred))

    r2_val_results.append(r2_score(y_val_true, y_val_pred))
    mae_val_results.append(mean_absolute_error(y_val_true, y_val_pred))
    rmse_val_results.append(np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
    mape_val_results.append(mean_absolute_percentage_error(y_val_true, y_val_pred))

    print(
        f"Fold {fold} — "
        f"train R²: {r2_train_results[-1]:.4f}, MAE: {mae_train_results[-1]:.4f}, "
        f"RMSE: {rmse_train_results[-1]:.4f}, MAPE: {mape_train_results[-1]:.4f}"
    )
    print(
        f"         val  R²: {r2_val_results[-1]:.4f}, MAE: {mae_val_results[-1]:.4f}, "
        f"RMSE: {rmse_val_results[-1]:.4f}, MAPE: {mape_val_results[-1]:.4f}"
    )

df_cv = pd.DataFrame({
    'r2_train'  : r2_train_results,   'mae_train'  : mae_train_results,
    'rmse_train': rmse_train_results, 'mape_train' : mape_train_results,
    'r2_val'    : r2_val_results,     'mae_val'    : mae_val_results,
    'rmse_val'  : rmse_val_results,   'mape_val'   : mape_val_results,
})
print("\nCross-validation results:")
print(df_cv)
print(f"\nMean val MAE  : {np.mean(mae_val_results):.4f} ± {np.std(mae_val_results):.4f} K")
print(f"Mean val RMSE : {np.mean(rmse_val_results):.4f} ± {np.std(rmse_val_results):.4f} K")
print(f"Mean val R²   : {np.mean(r2_val_results):.4f} ± {np.std(r2_val_results):.4f}")
df_cv.to_csv('jrmpnn_v4_cross_validation_results.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL MODEL — train on full training set
# ─────────────────────────────────────────────────────────────────────────────

print("\nTraining final model on full training set...", flush=True)

train_loader = DataLoader(train_data_graphs, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data_graphs,  batch_size=32, shuffle=False)

model = JRMPNN(
    node_dim=node_dim, edge_dim=edge_dim,
    hidden_dim=hidden_size, num_layers=num_layers, cond_dim=cond_dim,
).to(device)

loss_fn   = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)

train_validate(model, train_loader, test_loader, optimizer, loss_fn,
               epochs=epochs, scheduler=scheduler)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_full(model, loader):
    model.eval()
    y_preds, y_trues, y_bases = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            y_hat, _, _ = model(batch)              # unpack 3 values
            y_preds.append(y_hat.cpu().numpy().flatten())
            y_trues.append(batch.y.cpu().numpy().flatten())
            y_bases.append(batch.yj.cpu().numpy().flatten())
    return (np.concatenate(y_preds),
            np.concatenate(y_trues),
            np.concatenate(y_bases))

train_pred, train_true, y_base_train = evaluate_full(model, train_loader)
test_pred,  test_true,  y_base_test  = evaluate_full(model, test_loader)

def metrics(true, pred, label):
    r2   = r2_score(true, pred)
    mae  = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = mean_absolute_percentage_error(true, pred)
    print(f"\n{label}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : {mae:.4f} K")
    print(f"  RMSE : {rmse:.4f} K")
    print(f"  MAPE : {mape:.4f}")
    return r2, mae, rmse, mape

metrics(train_true, train_pred, "Final Train Results:")
metrics(test_true,  test_pred,  "Final Test Results:")
metrics(test_true,  y_base_test,"Joback Baseline (Test):")

torch.save(model.state_dict(), 'jrmpnn_v4_nbp.pth')
print("\nModel saved: jrmpnn_v4_nbp.pth")

# ─────────────────────────────────────────────────────────────────────────────
# PARITY PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_parity(actuals, predictions, color, filename):
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    r2   = r2_score(actuals, predictions)
    mae  = mean_absolute_error(actuals, predictions)
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, c=color, alpha=0.6, s=40)
    plt.plot([200, 1200], [200, 1200], 'k--', linewidth=2)
    plt.xlim(200, 1200); plt.ylim(200, 1200)
    plt.xticks(range(200, 1201, 100), fontsize=16)
    plt.yticks(range(200, 1201, 100), fontsize=16)
    plt.xlabel("Experimental NBP (K)", fontsize=16)
    plt.ylabel("Predicted NBP (K)",    fontsize=16)
    metrics_label = f"R$^2$ = {r2:.2f}\nMAE = {mae:.2f}\nMAPE = {mape:.2f}"
    plt.legend([metrics_label], loc="lower right", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(filename, format="pdf", dpi=500, bbox_inches="tight")
    plt.close()

#plot_parity(train_true, train_pred,  'red',     'jrmpnn_v4_train_parity.pdf')
#plot_parity(test_true,  test_pred,   '#1e488f', 'jrmpnn_v4_test_parity.pdf')
#plot_parity(test_true,  y_base_test, '#1e488f', 'joback_test_parity.pdf')
#print("Parity plots saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

pd.DataFrame({
    'y_train'    : train_true,
    'y_train_pred': train_pred,
    'joback_pred': y_base_train,
}).to_csv('jrmpnn_v4_nbp_train_pred.csv', index=False)

pd.DataFrame({
    'y_test'    : test_true,
    'y_test_pred': test_pred,
    'joback_pred': y_base_test,
}).to_csv('jrmpnn_v4_nbp_test_pred.csv', index=False)

print("Predictions saved.")
print("\nDone. Run conformal_nbp.py to generate uncertainty intervals.")