#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
import torch
torch.set_warn_always(False)
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import rdkit
from rdkit import Chem
import thermo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import sys
import os

# --- UTILS SETUP ---
utils_path = os.path.expanduser('~/deeplearning/residual/utils')
sys.path.insert(0, utils_path)
from molecular_graph import atom_features, bond_features, n_atom_features, n_bond_features
from train_validate import train_validate, train_validate_film

print('running', flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device type: {device}', flush=True)

# --- DATA LOADING ---
data_dir = os.path.expanduser('~/deeplearning/residual/data')
train_data = os.path.join(data_dir, 'NBP_train.csv')
test_data = os.path.join(data_dir, 'NBP_test.csv')

df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)

# --- JOBACK LOGIC ---
def JobackTb(smiles):
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return np.nan
        J = thermo.Joback(smiles)
        val = J.Tb(J.counts)
        if val is None:
            return np.nan
        return round(float(val), 2)
    except Exception:
        return np.nan

def compute_joback_pred(df: pd.DataFrame):
    df = df.copy()
    df['y_joback'] = df['SMILES'].apply(JobackTb)
    return df

df_train = compute_joback_pred(df_train)
df_test = compute_joback_pred(df_test)

# Fill NaN Joback predictions with training mean to avoid NaN propagation
joback_mean = df_train['y_joback'].mean()
df_train['y_joback'] = df_train['y_joback'].fillna(joback_mean)
df_test['y_joback'] = df_test['y_joback'].fillna(joback_mean)

fg_list = thermo.J_BIGGS_JOBACK_SMARTS

def joback_func_groups(df: pd.DataFrame):
    df = df.copy()
    fg_counts = np.zeros((len(df), len(fg_list)))
    for i, smiles in enumerate(df['SMILES']):
        try:
            joback = thermo.Joback(smiles)
            counts = joback.counts
            for j in range(len(fg_list)):
                fg_counts[i, j] = counts.get(j + 1, 0)
        except Exception:
            pass

    for j in range(len(fg_list)):
        df[f'Group {j+1}'] = fg_counts[:, j]
    return df

df_train = joback_func_groups(df_train)
df_test = joback_func_groups(df_test)

group_cols = [f'Group {i}' for i in range(1, len(fg_list) + 1)]

# --- GRAPH CONVERSION ---
def mol_to_graph(smiles, y, y_joback, c_vec):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feature_list = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_index_list, edge_features_list = [], []

    for bond in mol.GetBonds():
        edge_index_list.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index_list.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_features_list.append(bond_features(bond))
        edge_features_list.append(bond_features(bond))

    if not edge_features_list:
        return None

    return Data(
        x=torch.stack(atom_feature_list).to(torch.float32),
        edge_index=torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_features_list).to(torch.float32),
        y=torch.tensor([float(y)], dtype=torch.float32),
        yj=torch.tensor([float(y_joback)], dtype=torch.float32),
        c=torch.tensor(c_vec, dtype=torch.float32),
    )

# --- MODEL CLASSES ---

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
        self.residual = True

    def forward(self, x, edge_index, edge_attr):
        x = x.float()
        edge_attr = edge_attr.float()
        edge_attr = self.edge_update(x, edge_index, edge_attr)
        x_new = self.node_update(x, edge_index, edge_attr)
        return x_new, edge_attr

    def edge_update(self, x, edge_index, edge_attr):
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]

        incoming_messages = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        incoming_messages.index_add_(0, tgt_nodes, edge_attr)

        total_input_to_src = incoming_messages[src_nodes]

        reverse_indices = torch.arange(edge_attr.size(0), device=x.device) ^ 1
        reverse_edge_attr = edge_attr[reverse_indices]

        messages_excluding_reverse = total_input_to_src - reverse_edge_attr

        x_source = x[edge_index[0]]
        edge_input = torch.cat([messages_excluding_reverse, x_source, edge_attr], dim=1)
        new_edge_attr = self.edge_mlp(edge_input)

        if self.residual:
            new_edge_attr = new_edge_attr + edge_attr

        return new_edge_attr

    def node_update(self, x, edge_index, edge_attr):
        row, col = edge_index
        node_messages = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        node_messages.index_add_(0, col, edge_attr)

        node_input = torch.cat([x, node_messages], dim=1)
        new_x = self.node_mlp(node_input)

        if self.residual:
            new_x = new_x + x

        return new_x


class JRMPNN(nn.Module):
    """
    Joback-Residual MPNN (JR-MPNN)

    Learns a residual correction over Joback group contribution predictions.
    Three pooling operations (sum, mean, max) are concatenated to form a
    richer molecular representation, which is then concatenated with the
    Joback conditioning vector and passed through a deeper readout MLP
    with a skip connection.

    Pooling:  global_add + global_mean + global_max → 3 * hidden_dim
    Readout:  [3*hidden_dim + cond_dim] → hidden → hidden → hidden//2 → 1
    Output:   y_hat = y_joback + r_hat  (physics-informed residual)
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, cond_dim, output_dim=1):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        self.dmpnn_layers = nn.ModuleList(
            [DMPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Triple pooling produces 3 * hidden_dim; append cond_dim for full input
        in_dim = 3 * hidden_dim + cond_dim

        # Deeper readout MLP
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)

        # Skip connection: direct path from full input to second hidden layer
        self.skip = nn.Linear(in_dim, hidden_dim)

        self.drop = nn.Dropout(0.2)
        self.act  = nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.edge_attr.float()
        batch, c, yj = data.batch, data.c, data.yj

        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        for layer in self.dmpnn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        # Triple pooling: captures total signal, average signal, and dominant signal
        h_add  = global_add_pool(x, batch=batch)   # sensitive to molecule size
        h_mean = global_mean_pool(x, batch=batch)  # size-invariant average
        h_max  = global_max_pool(x, batch=batch)   # most activated feature per dim
        h_G = torch.cat([h_add, h_mean, h_max], dim=-1)  # [B, 3 * hidden_dim]

        B = h_G.size(0)
        if c.dim() == 1:
            c = c.view(B, self.cond_dim)

        # Concatenate graph embedding with Joback conditioning vector
        h_cat = torch.cat([h_G, c], dim=-1)  # [B, 3*hidden_dim + cond_dim]

        # Layer 1
        h = self.act(self.fc1(h_cat))
        h = self.drop(h)
        # Layer 2 + skip connection from full concatenated input
        h = self.act(self.fc2(h) + self.skip(h_cat))
        h = self.drop(h)
        # Layer 3 -> output
        h = self.act(self.fc3(h))
        r_hat = self.fc4(h).squeeze(-1)

        # Final prediction = Joback baseline + learned residual
        y_hat = yj.view(-1) + r_hat
        return y_hat, r_hat


# --- DATA PREP ---
train_data_graphs = []
for _, row in df_train.iterrows():
    y = float(row.NBP)
    yj = float(row.y_joback)
    groups = [float(row[g]) for g in group_cols]
    c = [yj] + groups
    d = mol_to_graph(row.SMILES, y, yj, c)
    if d is not None:
        train_data_graphs.append(d)

test_data_graphs = []
for _, row in df_test.iterrows():
    y = float(row.NBP)
    yj = float(row.y_joback)
    groups = [float(row[g]) for g in group_cols]
    c = [yj] + groups
    d = mol_to_graph(row.SMILES, y, yj, c)
    if d is not None:
        test_data_graphs.append(d)

print("Training data length: ", len(train_data_graphs))
print("Test data length: ", len(test_data_graphs))

# --- HYPERPARAMETERS ---
cond_dim = train_data_graphs[0].c.numel()
node_dim = n_atom_features()
edge_dim = n_bond_features()

hidden_size = 128
num_layers = 5
output_dim = 1
lr = 2e-4
weight_decay = 1e-4
epochs = 800

# --- CROSS VALIDATION ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_train_results, mae_train_results, mse_train_results, rmse_train_results, mape_train_results = [], [], [], [], []
r2_val_results, mae_val_results, mse_val_results, rmse_val_results, mape_val_results = [], [], [], [], []

fold = 1
for train_index, val_index in kf.split(train_data_graphs):
    print(f"fold {fold} starting...")

    train_subset = [train_data_graphs[i] for i in train_index]
    val_subset = [train_data_graphs[i] for i in val_index]

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    model = JRMPNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_size,
        num_layers=num_layers,
        cond_dim=cond_dim,
    ).to(device)

    loss_fn = nn.SmoothL1Loss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #train_losses, val_losses = train_validate_film(model, train_loader, val_loader, optimizer, loss_fn, epochs=800)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_losses, val_losses = train_validate_film(
    model, train_loader, val_loader, optimizer, loss_fn, epochs=epochs, scheduler=scheduler)

    model.eval()
    with torch.no_grad():
        y_train_pred, y_train_true = [], []
        for batch in train_loader:
            batch = batch.to(device)
            y_hat, r_hat = model(batch)
            y_train_pred.extend(y_hat.cpu().numpy().flatten())
            y_train_true.extend(batch.y.cpu().numpy().flatten())

        r2_train = r2_score(y_train_true, y_train_pred)
        mae_train = mean_absolute_error(y_train_true, y_train_pred)
        mse_train = mean_squared_error(y_train_true, y_train_pred)
        rmse_train = np.sqrt(mse_train)
        mape_train = mean_absolute_percentage_error(y_train_true, y_train_pred)

        y_val_pred, y_val_true = [], []
        for batch in val_loader:
            batch = batch.to(device)
            y_hat, r_hat = model(batch)
            y_val_pred.extend(y_hat.cpu().numpy().flatten())
            y_val_true.extend(batch.y.cpu().numpy().flatten())

        r2_val = r2_score(y_val_true, y_val_pred)
        mae_val = mean_absolute_error(y_val_true, y_val_pred)
        mse_val = mean_squared_error(y_val_true, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        mape_val = mean_absolute_percentage_error(y_val_true, y_val_pred)

    r2_train_results.append(r2_train); mae_train_results.append(mae_train)
    mse_train_results.append(mse_train); rmse_train_results.append(rmse_train); mape_train_results.append(mape_train)
    r2_val_results.append(r2_val); mae_val_results.append(mae_val)
    mse_val_results.append(mse_val); rmse_val_results.append(rmse_val); mape_val_results.append(mape_val)

    print(
        f"fold {fold} completed. "
        f"train r2: {r2_train:.4f}, mae: {mae_train:.4f}, rmse: {rmse_train:.4f}, mape: {mape_train:.4f}"
    )
    print(
        f"val   r2: {r2_val:.4f}, mae: {mae_val:.4f}, rmse: {rmse_val:.4f}, mape: {mape_val:.4f}\n"
    )
    fold += 1

df_results = pd.DataFrame({
    "r2_train": r2_train_results,
    "mae_train": mae_train_results,
    "mse_train": mse_train_results,
    "rmse_train": rmse_train_results,
    "mape_train": mape_train_results,
    "r2_val": r2_val_results,
    "mae_val": mae_val_results,
    "mse_val": mse_val_results,
    "rmse_val": rmse_val_results,
    "mape_val": mape_val_results,
})
print("cross-validation results:")
print(df_results)
df_results.to_csv('jrmpnn_v4_cross_validation_results.csv', index=False)

# --- FINAL EVALUATION ---
train_loader = DataLoader(train_data_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(test_data_graphs, batch_size=32, shuffle=False)

def evaluate_model(model, data_loader, device):
    model.eval()
    y_pred, y_true, y_base = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            y_hat, r_hat = model(batch)
            y_pred.append(y_hat.cpu().numpy().flatten())
            y_true.append(batch.y.cpu().numpy().flatten())
            y_base.append(batch.yj.cpu().numpy().flatten())
    return np.concatenate(y_pred), np.concatenate(y_true), np.concatenate(y_base)

train_predictions, train_actuals, y_base_train = evaluate_model(model, train_loader, device)
test_predictions, test_actuals, y_base_test = evaluate_model(model, val_loader, device)

r2_train = r2_score(train_actuals, train_predictions)
r2_test = r2_score(test_actuals, test_predictions)
mae_train = mean_absolute_error(train_actuals, train_predictions)
mae_test = mean_absolute_error(test_actuals, test_predictions)
mse_train = mean_squared_error(train_actuals, train_predictions)
mse_test = mean_squared_error(test_actuals, test_predictions)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
mape_train = mean_absolute_percentage_error(train_actuals, train_predictions)
mape_test = mean_absolute_percentage_error(test_actuals, test_predictions)

# Joback baseline for direct comparison
r2_joback_test = r2_score(test_actuals, y_base_test)
mae_joback_test = mean_absolute_error(test_actuals, y_base_test)
rmse_joback_test = np.sqrt(mean_squared_error(test_actuals, y_base_test))
mape_joback_test = mean_absolute_percentage_error(test_actuals, y_base_test)

print("\nFinal Train Results:")
print(f"  R2: {r2_train:.4f}, MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, MAPE: {mape_train:.4f}")
print("\nFinal Test Results:")
print(f"  R2: {r2_test:.4f}, MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, MAPE: {mape_test:.4f}")
print("\nJoback Baseline Test Results:")
print(f"  R2: {r2_joback_test:.4f}, MAE: {mae_joback_test:.4f}, RMSE: {rmse_joback_test:.4f}, MAPE: {mape_joback_test:.4f}")

torch.save(model.state_dict(), "jrmpnn_v4_nbp.pth")

# --- PLOTTING ---
def plot_parity(actuals, predictions, color, filename):
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, c=color, alpha=0.6, s=40)
    plt.plot([200, 1200], [200, 1200], 'k--', linewidth=2)
    plt.xlim(200, 1200); plt.ylim(200, 1200)
    plt.xticks(range(200, 1201, 100), fontsize=16)
    plt.yticks(range(200, 1201, 100), fontsize=16)
    plt.xlabel("Experimental NBP (K)", fontsize=16)
    plt.ylabel("Predicted NBP (K)", fontsize=16)
    metrics_label = f"R$^2$ = {r2:.2f}\nMAE = {mae:.2f}\nMAPE = {mape:.2f}"
    plt.legend([metrics_label], loc="lower right", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(filename, format="pdf", dpi=500, bbox_inches="tight")
    plt.close()

# Uncomment to generate plots
# plot_parity(train_actuals, train_predictions, 'red',     'jrmpnn_v4_train_parity.pdf')
# plot_parity(test_actuals,  test_predictions,  '#1e488f', 'jrmpnn_v4_test_parity.pdf')
# plot_parity(train_actuals, y_base_train,      'red',     'joback_train_parity.pdf')
# plot_parity(test_actuals,  y_base_test,       '#1e488f', 'joback_test_parity.pdf')

# --- SAVE PREDICTIONS ---
df_train_predictions = pd.DataFrame({
    'y_train': train_actuals,
    'y_train_pred': train_predictions,
    'joback_pred': y_base_train
})
df_train_predictions.to_csv('jrmpnn_v4_nbp_train_pred.csv', index=False)

df_test_predictions = pd.DataFrame({
    'y_test': test_actuals,
    'y_test_pred': test_predictions,
    'joback_pred': y_base_test
})
df_test_predictions.to_csv('jrmpnn_v4_nbp_test_pred.csv', index=False)