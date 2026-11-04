#!/usr/bin/env python3
"""
Compute calibration quantile for Melting Point (Tm)
Run this script to generate the quantile for conformal prediction
"""

import warnings
warnings.filterwarnings('ignore')
import torch
torch.set_warn_always(False)
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
import thermo
import pandas as pd
import numpy as np
import json
import sys
import os
utils_path = os.path.expanduser('~/deeplearning/residual/utils')
sys.path.insert(0, utils_path)
from molecular_graph import atom_features, bond_features, n_atom_features, n_bond_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
PROPERTY_NAME = 'Vc'
PROPERTY_COL = 'VC'  # experimental melting point
data_dir = os.path.expanduser('~/deeplearning/residual/data')
CALIBRATION_DATA_FILE = os.path.join(data_dir, 'VC_calibration.csv')  # CALIBRATION CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'film_dmpnn_vc.pth')

# Model hyperparameters (update if different)
HIDDEN_DIM = 32
NUM_LAYERS = 3
GAMMA_SCALE = 0.5
BETA_CLIP = 3.0

# ============================================================================
# Model Classes
# ============================================================================

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
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + node_dim, hidden_dim),
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
        edge_attr = self.edge_update(x, edge_index, edge_attr)
        x_new = self.node_update(x, edge_index, edge_attr)
        return x_new, edge_attr

    def edge_update(self, x, edge_index, edge_attr):
        row, col = edge_index
        num_nodes = x.size(0)
        x_source = x[row]
        node_incoming_messages = torch.zeros(num_nodes, edge_attr.size(1), 
                                            device=edge_attr.device, dtype=edge_attr.dtype)
        node_incoming_messages.index_add_(0, col, edge_attr)
        edge_messages = node_incoming_messages[row]
        edge_input = torch.cat([edge_messages, x_source], dim=1)
        new_edge_attr = self.edge_mlp(edge_input)
        if self.residual and new_edge_attr.size(1) == edge_attr.size(1):
            new_edge_attr = new_edge_attr + edge_attr
        return new_edge_attr

    def node_update(self, x, edge_index, edge_attr):
        row, col = edge_index
        node_messages = torch.zeros(x.size(0), edge_attr.size(1), 
                                    device=x.device, dtype=x.dtype)
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
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
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
        
        h_G = global_add_pool(x, batch=batch)
        B = h_G.size(0)
        
        if c.dim() == 1:
            c = c.view(B, self.cond_dim)
        elif c.dim() == 2 and c.size(0) != B and c.size(1) == self.cond_dim:
            c = c.t().contiguous()
            
        h_G = self.film(h_G, c)
        r_hat = self.readout(h_G).squeeze(-1)
        y_hat = yj.view(-1) + r_hat
        return y_hat, h_G

# ============================================================================
# Helper Functions
# ============================================================================

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

def get_joback_vc(smiles):
    try:
        J = thermo.Joback(smiles)
        val = J.Vc(J.counts)
        return val if val is not None else None
    except:
        return None

def get_joback_groups(smiles):
    try:
        J = thermo.Joback(smiles)
        fg_list = thermo.J_BIGGS_JOBACK_SMARTS
        groups = [J.counts.get(i+1, 0) for i in range(len(fg_list))]
        return groups
    except:
        return None

def get_embedding_norms(model, loader):
    y_true, y_pred, norms = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_hat, h_G = model(batch)
            y_true.extend(batch.y.cpu().numpy().flatten())
            y_pred.extend(y_hat.cpu().numpy().flatten())
            norms.extend(h_G.norm(dim=1).cpu().numpy().flatten())
    return np.array(y_true), np.array(y_pred), np.array(norms)

# ============================================================================
# Main Script
# ============================================================================

def main():
    print("="*70)
    print(f"Computing Calibration Quantile for {PROPERTY_NAME}")
    print("="*70)
    
    # Load calibration data
    print(f"\n1. Loading calibration data from {CALIBRATION_DATA_FILE}...")
    df = pd.read_csv(CALIBRATION_DATA_FILE)
    print(f"   Loaded {len(df)} molecules")
    
    # Compute Joback predictions
    print("\n2. Computing Joback predictions...")
    joback_preds = []
    group_counts_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        yj = get_joback_vc(smiles)
        groups = get_joback_groups(smiles)
        
        if yj is not None and groups is not None:
            joback_preds.append(yj)
            group_counts_list.append(groups)
            valid_indices.append(idx)
    
    df = df.iloc[valid_indices].reset_index(drop=True)
    print(f"   Valid molecules: {len(df)}")
    
    # Create graphs
    print("\n3. Creating molecular graphs...")
    graphs = []
    for idx, row in df.iterrows():
        y = float(row[PROPERTY_COL])
        yj = float(joback_preds[idx])
        groups = group_counts_list[idx]
        c = [yj] + groups
        
        graph = mol_to_graph(row['SMILES'], y, yj, c)
        if graph is not None:
            graphs.append(graph)
    
    print(f"   Created {len(graphs)} graphs for calibration")
    
    # Load model
    print(f"\n4. Loading model from {MODEL_FILE}...")
    cond_dim = len(graphs[0].c)
    node_dim = graphs[0].x.size(1)
    edge_dim = graphs[0].edge_attr.size(1)
    
    model = DMPNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        cond_dim=cond_dim,
        output_dim=1
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    print("   Model loaded successfully")
    
    # Compute predictions
    print("\n5. Computing predictions and embedding norms...")
    cal_loader = DataLoader(graphs, batch_size=32, shuffle=False)
    y_cal_true, y_cal_pred, cal_norms = get_embedding_norms(model, cal_loader)
    
    # Compute quantile
    print("\n6. Computing 95% quantile...")
    scaled_residuals = np.abs(y_cal_true - y_cal_pred) / (cal_norms + 1e-6)
    quantile = float(np.quantile(scaled_residuals, 0.95))
    
    print(f"\n{'='*70}")
    print(f"RESULT: {PROPERTY_NAME} quantile = {quantile:.6f}")
    print(f"{'='*70}")
    
    # Save to file
    output_file = f'quantile_{PROPERTY_NAME.lower()}.json'
    with open(output_file, 'w') as f:
        json.dump({PROPERTY_NAME: quantile}, f, indent=2)
    
    print(f"\n✓ Saved to {output_file}")
    print("\nNext steps:")
    print("  1. Run the other 4 scripts (Tb, Tc, Pc, Vc)")
    print("  2. Combine all 5 quantiles into calibration_quantiles.json")

if __name__ == "__main__":
    main()
