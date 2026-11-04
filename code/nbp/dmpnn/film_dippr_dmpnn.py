#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
import torch
torch.set_warn_always(False)
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
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
        
def compute_joback_pred(df:pd.DataFrame):
    df['y_joback'] = df['SMILES'].apply(JobackTb)
    return df

df_train = compute_joback_pred(df_train)
df_test = compute_joback_pred(df_test)

fg_list = thermo.J_BIGGS_JOBACK_SMARTS

def joback_func_groups(df: pd.DataFrame):
    fg_counts = np.zeros((len(df), len(fg_list)))
    for i, smiles in enumerate(df['SMILES']):
        try:
            joback = thermo.Joback(smiles)
            counts = joback.counts
            for j, fg in enumerate(fg_list):
                fg_counts[i, j] = counts.get(j+1, 0)
        except:
            pass # Handle rdkit errors gracefully
            
    fg_name_map = {}
    for i, fg in enumerate(fg_list):
        fg_name_map[fg[1]] = f'Group{i+1}'
    df = df.rename(columns=fg_name_map)
    
    for j, fg in enumerate(fg_list):
        col_name = f'Group {j+1}'
        df[col_name] = fg_counts[:, j]
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
        # IMPORTANT: Edges must be added in pairs (u,v) then (v,u) for the D-MPNN logic to work!
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

class FiLM(nn.Module):
    def __init__(self, dim, cond_dim, gamma_scale=0.5, beta_clip=3.0):
        super().__init__()
        self.gamma = nn.Sequential(nn.Linear(cond_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.beta  = nn.Sequential(nn.Linear(cond_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        
        nn.init.zeros_(self.gamma[-1].weight); nn.init.zeros_(self.gamma[-1].bias)
        nn.init.zeros_(self.beta[-1].weight); nn.init.zeros_(self.beta[-1].bias)
        
        self.gamma_scale = gamma_scale
        self.beta_clip = beta_clip
    
    def forward(self, h, c):
        g = 1.0 + torch.tanh(self.gamma(c)) * self.gamma_scale
        b = torch.clamp(self.beta(c), -self.beta_clip, self.beta_clip)
        return g * h + b

class DMPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add', flow='source_to_target')
        
        # FIX 1: Input dimension is 3 * hidden_dim because we concatenate:
        # [Message_sum, Source_Node, Edge_Feature]
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
        # Vectorized Backtracking Prevention
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]
        
        # A. Sum all incoming messages to target
        incoming_messages = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        incoming_messages.index_add_(0, tgt_nodes, edge_attr)
        
        # B. Get inputs for source nodes
        total_input_to_src = incoming_messages[src_nodes]
        
        # C. Subtract reverse edge
        # Relies on edges being added in pairs (0,1), (2,3)...
        reverse_indices = torch.arange(edge_attr.size(0), device=x.device) ^ 1
        reverse_edge_attr = edge_attr[reverse_indices]
        
        messages_excluding_reverse = total_input_to_src - reverse_edge_attr
        
        # D. MLP
        row = edge_index[0]
        x_source = x[row]
        
        # Concatenate: Message + Source Node + Original Edge
        edge_input = torch.cat([messages_excluding_reverse, x_source, edge_attr], dim=1)
        
        new_edge_attr = self.edge_mlp(edge_input)
        
        if self.residual and new_edge_attr.size(1) == edge_attr.size(1):
             new_edge_attr = new_edge_attr + edge_attr
        
        return new_edge_attr

    def node_update(self, x, edge_index, edge_attr):
        row, col = edge_index
        node_messages = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        node_messages.index_add_(0, col, edge_attr)
        
        node_input = torch.cat([x, node_messages], dim=1)
        new_x = self.node_mlp(node_input)
        
        if self.residual and new_x.size(1) == x.size(1):
            new_x = new_x + x
        return new_x

class DMPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, cond_dim, output_dim=1):
        super().__init__()
        self.cond_dim = cond_dim
        # Embedding layers project input features to hidden_dim
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # Note: We pass hidden_dim for all dimensions because inputs are already projected
        self.dmpnn_layers = nn.ModuleList(
            [DMPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.film = FiLM(hidden_dim, cond_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.edge_attr.float()
        batch, c, yj = data.batch, data.c, data.yj
        
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        for layer in self.dmpnn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        
        # Global Pooling
        h_G = global_add_pool(x, batch=batch)
        B = h_G.size(0)
        
        # Handle conditioning vector shape
        if c.dim() == 1:
            c = c.view(B, self.cond_dim)
        elif c.dim() == 2 and c.size(0) != B and c.size(1) == self.cond_dim:
            c = c.t().contiguous()
            
        # FiLM Modulation
        h_G = self.film(h_G, c)
        
        # Residual Prediction
        r_hat = self.readout(h_G).squeeze(-1)
        
        # Final = Baseline + Residual
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

# --- TRAINING LOOP ---
cond_dim = train_data_graphs[0].c.numel()
node_dim = n_atom_features() # Use function directly for consistency
edge_dim = n_bond_features()

hidden_size = 32
num_layers = 4
output_dim = 1
lr = 2e-4
weight_decay = 1e-4

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Metrics containers...
r2_train_results, mae_train_results, mse_train_results, rmse_train_results, mape_train_results = [], [], [], [], []
r2_val_results, mae_val_results, mse_val_results, rmse_val_results, mape_val_results = [], [], [], [], []

fold = 1
for train_index, val_index in kf.split(train_data_graphs):
    print(f"fold {fold} starting...")

    train_subset = [train_data_graphs[i] for i in train_index]
    val_subset = [train_data_graphs[i] for i in val_index]

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    model = DMPNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_size,
        num_layers=num_layers,
        cond_dim=cond_dim,
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss() # Kept your loss function preference

    # Assuming train_validate_film is defined in your utils
    train_losses, val_losses = train_validate_film(model, train_loader, val_loader, optimizer, loss_fn, epochs=800)
    
    model.eval()
    with torch.no_grad():
        # Evaluation Logic (kept identical to your snippet)
        y_train_pred, y_train_true = [], []
        for batch in train_loader:
            batch = batch.to(device)
            y_hat, r_hat = model(batch)
            y_train_pred.extend(y_hat.cpu().numpy().flatten())
            y_train_true.extend(batch.y.cpu().numpy().flatten())

        # ... (Metrics calculation same as your snippet) ...
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

    print(f"fold {fold} completed. val r2: {r2_val:.4f}, mae:{mae_val:.4f}")
    fold += 1

# ... (Saving results and plotting code remains the same) ...
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
print(df_results)
df_results.to_csv('film_dmpnn_cross_validation_results.csv', index=False)

train_loader = DataLoader(train_data_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(test_data_graphs, batch_size=32, shuffle=False)

def evaluate_model(model, data_loader, device):
    model.eval()
    y_pred, y_true, y_base = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            y_hat, r_hat = model(batch)
            outputs = y_hat.cpu().numpy().flatten()
            y_pred.append(outputs)
            y_true.append(batch.y.cpu().numpy().flatten())
            y_base.append(batch.yj.cpu().numpy().flatten())
    
    return np.concatenate(y_pred), np.concatenate(y_true), np.concatenate(y_base)

# Then use it like this:
train_predictions, train_actuals, y_base_train = evaluate_model(model, train_loader, device)
test_predictions, test_actuals, y_base_test = evaluate_model(model, val_loader, device)

# use y_true first
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

torch.save(model.state_dict(), "film_dmpnn_nbp.pth")

def plot_parity_test(actuals, predictions):
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, c='#1e488f', alpha=0.6, s=40)
    plt.plot([200, 1201], [200, 1201], 'k--', linewidth=2)
    plt.xlim(200, 1200); plt.ylim(200, 1200)
    plt.xticks(range(200, 1200, 100), fontsize=16)
    plt.yticks(range(200, 1200, 100), fontsize=16)
    plt.xlabel("Experimental TC (K)", fontsize=16)
    plt.ylabel("Predicted TC (K)", fontsize=16)
    metrics_label = f"R$^2$ = {r2:.2f}\nMAE = {mae:.2f}\nMAPE = {mape:.2f}"
    plt.legend([metrics_label], loc="lower right", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig("film_dmpnn_test_parity_tc.pdf", format="pdf", dpi=500, bbox_inches="tight")
    #plt.show()

def plot_parity_train(actuals, predictions):
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, c='red', alpha=0.6, s=40)
    plt.plot([200, 1200], [200, 1200], 'k--', linewidth=2)
    plt.xlim(200, 1201); plt.ylim(200, 1201)
    plt.xticks(range(200, 1200, 100), fontsize=16)
    plt.yticks(range(200, 1200, 100), fontsize=16)
    plt.xlabel("Experimental TC (K)", fontsize=16)
    plt.ylabel("Predicted TC (K)", fontsize=16)
    metrics_label = f"R$^2$ = {r2:.2f}\nMAE = {mae:.2f}\nMAPE = {mape:.2f}"
    plt.legend([metrics_label], loc="lower right", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig("film_dmpnn_train_parity_tc.pdf", format="pdf", dpi=500, bbox_inches="tight")
    #plt.show()
    
def plot_parity_test_base(actuals, predictions):
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, c='#1e488f', alpha=0.6, s=40)
    plt.plot([200, 1200], [200, 1200], 'k--', linewidth=2)
    plt.xlim(200, 1200); plt.ylim(200, 1200)
    plt.xticks(range(200, 1201, 100), fontsize=16)
    plt.yticks(range(200, 1201, 100), fontsize=16)
    plt.xlabel("Experimental TC (K)", fontsize=16)
    plt.ylabel("Predicted TC (K)", fontsize=16)
    metrics_label = f"R$^2$ = {r2:.2f}\nMAE = {mae:.2f}\nMAPE = {mape:.2f}"
    plt.legend([metrics_label], loc="lower right", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig("joback_test_parity_tc.pdf", format="pdf", dpi=500, bbox_inches="tight")
    #plt.show()
    
def plot_parity_train_base(actuals, predictions):
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, c='red', alpha=0.6, s=40)
    plt.plot([200, 1200], [200, 1200], 'k--', linewidth=2)
    plt.xlim(200, 1200); plt.ylim(200, 1200)
    plt.xticks(range(200, 1201, 100), fontsize=16)
    plt.yticks(range(200, 1201, 100), fontsize=16)
    plt.xlabel("Experimental TC (K)", fontsize=16)
    plt.ylabel("Predicted TC (K)", fontsize=16)
    metrics_label = f"R$^2$ = {r2:.2f}\nMAE = {mae:.2f}\nMAPE = {mape:.2f}"
    plt.legend([metrics_label], loc="lower right", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig("joback_train_parity_tc.pdf", format="pdf", dpi=500, bbox_inches="tight")
    #plt.show()

#plot_parity_train(train_actuals, train_predictions)
#plot_parity_test(test_actuals, test_predictions)
#plot_parity_train_base(train_actuals, y_base_train)
#plot_parity_test_base(test_actuals, y_base_test)

df_train_predictions = pd.DataFrame({'y_train': train_actuals, 'y_train_pred': train_predictions, 'joback_pred': y_base_train})
df_train_predictions.to_csv('film_dmpnn_nbp_train_pred.csv', index=False)

df_test_predictions = pd.DataFrame({'y_test': test_actuals, 'y_test_pred': test_predictions,'joback': y_base_test})
df_test_predictions.to_csv('film_dmpnn_nbp_test_pred.csv', index=False)  

