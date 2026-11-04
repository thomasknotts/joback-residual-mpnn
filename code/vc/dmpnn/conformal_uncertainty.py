#!/usr/bin/env python3
#import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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
from molecular_graph import atom_features, bond_features, get_bond_pair, n_atom_features, n_bond_features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from train_validate import train_validate_film

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = 'data/TC_train.csv'
test_data = 'data/TC_test.csv'

df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)

# compute Joback predictions for all compounds
def JobackTb(smiles):
    J = thermo.Joback(smiles)
    ans = np.round(J.Tc(J.counts), 3)
    return ans

df_train['y_joback'] = df_train['SMILES'].apply(JobackTb)
df_test['y_joback'] = df_test['SMILES'].apply(JobackTb)
fg_list = thermo.J_BIGGS_JOBACK_SMARTS

for j, fg in enumerate(fg_list):
    df_train[f'Group {j+1}'] = 0
    df_test[f'Group {j+1}'] = 0
group_cols = [f'Group {i}' for i in range(1, len(fg_list)+1)]

# mol graph
def mol_to_graph(smiles, y, y_joback, c_vec):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    atom_feature_list = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_index_list, edge_features_list = [], []

    for bond in mol.GetBonds():
        edge_index_list.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index_list.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_features_list.append(bond_features(bond))
        edge_features_list.append(bond_features(bond))

    if not edge_features_list:
        print(f"Warning no bonds found for SMILES {smiles}")
        return None

    return Data(
        x=torch.stack(atom_feature_list).to(torch.float32),
        edge_index=torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_features_list).to(torch.float32),
        y=torch.tensor([float(y)], dtype=torch.float32),                  # experimental value
        yj=torch.tensor([float(y_joback)], dtype=torch.float32),          # Joback baseline
        c=torch.tensor(c_vec, dtype=torch.float32),                       # conditioning vector       
    )

# FiLM for readout conditioning
class FiLM(nn.Module):
    def __init__(self, dim, cond_dim, gamma_scale=0.5, beta_clip=3.0):
        super().__init__()
        self.gamma = nn.Sequential(nn.Linear(cond_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.beta  = nn.Sequential(nn.Linear(cond_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        # start near identity
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
        # update edge hidden states via edge-to-edge message passing
        row, col = edge_index          # row = source, col = target
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        # get source node features for each edge
        x_source = x[row]
        
        # create a mapping: for each node, aggregate all incoming edge features
        node_incoming_messages = torch.zeros(num_nodes, edge_attr.size(1), device=edge_attr.device, dtype=edge_attr.dtype)
        node_incoming_messages.index_add_(0, col, edge_attr)
        
        # for each edge, get the aggregated incoming messages to its source node
        edge_messages = node_incoming_messages[row]
        
        # combine source node features with aggregated edge messages
        edge_input = torch.cat([edge_messages, x_source], dim=1)
        
        # update edge representations
        new_edge_attr = self.edge_mlp(edge_input)
        
        # residual connection
        if self.residual and new_edge_attr.size(1) == edge_attr.size(1):
            new_edge_attr = new_edge_attr + edge_attr
            
        return new_edge_attr

    def node_update(self, x, edge_index, edge_attr):
        # aggregate edge messages to update node representations
        row, col = edge_index
        
        # aggregate messages from incoming edges
        node_messages = torch.zeros(x.size(0), edge_attr.size(1), device=x.device, dtype=x.dtype)
        node_messages.index_add_(0, col, edge_attr)
        
        # combine with current node features
        node_input = torch.cat([x, node_messages], dim=1)
        
        # update node representations
        new_x = self.node_mlp(node_input)
        
        # residual connection
        if self.residual and new_x.size(1) == x.size(1):
            new_x = new_x + x
            
        return new_x

class DMPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, cond_dim):
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
        
        h_G = global_add_pool(x, batch=batch)                # [B, hidden_dim]
        B = h_G.size(0)
        
        # reshape c to [B, cond_dim]
        if c.dim() == 1:
            # c was concatenated into length B*cond_dim
            c = c.view(B, self.cond_dim)
        elif c.dim() == 2 and c.size(0) != B and c.size(1) == self.cond_dim:
            # rare case if c came in as [cond_dim, B]
            c = c.t().contiguous()
            
        h_G = self.film(h_G, c)                           # FiLM(h, c) with matching shapes
        r_hat = self.readout(h_G).squeeze(-1)
        y_hat = yj.view(-1) + r_hat
        return y_hat, h_G
    
# Load trained model
hidden_size = 32
num_layers = 3
output_dim = 1
train_graphs, test_graphs = [], []


for _, row in df_train.iterrows():
    y = float(row.TC)
    yj = float(row.y_joback)
    groups = [float(row[g]) for g in group_cols]
    c = [yj] + groups
    d = mol_to_graph(row.SMILES, y, yj, c)
    if d is not None:
        train_graphs.append(d)
        
for _, row in df_test.iterrows():
    y = float(row.TC)
    yj = float(row.y_joback)
    groups = [float(row[g]) for g in group_cols]
    c = [yj] + groups
    d = mol_to_graph(row.SMILES, y, yj, c)
    if d is not None:
        test_graphs.append(d)
        
cond_dim = train_graphs[0].c.numel()
node_dim = train_graphs[0].x.size(1)
edge_dim = train_graphs[0].edge_attr.size(1)

model = DMPNN(node_dim, edge_dim, hidden_size, num_layers, cond_dim).to(device)
model.load_state_dict(torch.load('film_dmpnn_tc.pth', map_location=device))
model.eval()

# Global Conformal Prediction
np.random.seed(42)
n_train = len(train_graphs)
cal_size = int(0.2 * n_train)
idx = np.random.permutation(n_train)
cal_idx, _ = idx[:cal_size], idx[cal_size:]
cal_subset = [train_graphs[i] for i in cal_idx]
cal_loader = DataLoader(cal_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

def evaluate(model, loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_hat, _ = model(batch)
            y_true.extend(batch.y.cpu().numpy().flatten())
            y_pred.extend(y_hat.cpu().numpy().flatten())
    return np.array(y_true), np.array(y_pred)

y_cal_true, y_cal_pred = evaluate(model, cal_loader)
y_test_true, y_test_pred = evaluate(model, test_loader)

#residuals = np.abs(y_cal_true - y_cal_pred)
#q = np.quantile(residuals, 0.95)
#lower = y_test_pred - q
#upper = y_test_pred + q
#coverage = np.mean((y_test_true >= lower) & (y_test_true <= upper))
#print(f"Global conformal coverage: {coverage*100:.2f}%")

#df_global = pd.DataFrame({
#    'y_true': y_test_true, 'y_pred': y_test_pred,
#    'lower': lower, 'upper': upper
#})
#df_global.to_csv('film_dmpnn_tc_test_conformal.csv', index=False)

# Adaptive (Feature-dependent) Conformal Prediction
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

y_cal_true, y_cal_pred, cal_norms = get_embedding_norms(model, cal_loader)
y_test_true, y_test_pred, test_norms = get_embedding_norms(model, test_loader)

scaled_residuals = np.abs(y_cal_true - y_cal_pred) / (cal_norms + 1e-6)
q = np.quantile(scaled_residuals, 0.95)
lower_ad = y_test_pred - q * test_norms
upper_ad = y_test_pred + q * test_norms
coverage_ad = np.mean((y_test_true >= lower_ad) & (y_test_true <= upper_ad))
print(f"Adaptive conformal coverage: {coverage_ad*100:.2f}%")

df_adaptive = pd.DataFrame({
    'y_true': y_test_true, 'y_pred': y_test_pred,
    'lower': lower_ad, 'upper': upper_ad,
    'embedding_norm': test_norms
})
df_adaptive.to_csv('film_dmpnn_tc_test_adaptive_conformal.csv', index=False)

# Plot Results
def plot_conformal(y_true, y_pred, lower, upper, title, fname):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(6,6))
    plt.errorbar(y_true, y_pred,
                 yerr=[y_pred-lower, upper-y_pred],
                 fmt='o', alpha=0.5, ecolor='k', capsize=2)
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
    plt.savefig(fname, dpi=500, bbox_inches="tight")
    plt.show()
    

#plot_conformal(y_test_true, y_test_pred, lower, upper,
#               "Global Conformal Intervals (95%)",
#               "film_dmpnn_tc_global_conformal_parity.pdf")
plot_conformal(y_test_true, y_test_pred, lower_ad, upper_ad,
               "Adaptive Conformal Intervals (95%)",
               "film_dmpnn_tc_adaptive_conformal_parity.pdf")
print("All conformal analyses completed successfully.")

# ==========================================================
# Compute Trust Scores
# ==========================================================
print("Computing molecular trust scores...")

# Compute interval widths
widths_ad = upper_ad - lower_ad
width_min, width_max = widths_ad.min(), widths_ad.max()

# Normalize 0–1 (1 = most confident)
trust_scores = 1 - (widths_ad - width_min) / (width_max - width_min)

# Optional: exponential trust formulation
# avg_width = np.mean(widths_ad)
# trust_scores = np.exp(-widths_ad / avg_width)

df_adaptive['interval_width'] = widths_ad
df_adaptive['trust_score'] = trust_scores

# Sort from most → least trustworthy
df_adaptive_sorted = df_adaptive.sort_values(by='trust_score', ascending=False)
df_adaptive_sorted.to_csv('film_dmpnn_tc_test_trust_scores.csv', index=False)

# ==========================================================
# Visualization: Trust Distribution
# ==========================================================
plt.figure(figsize=(6,4))
plt.hist(trust_scores, bins=20, color='#1e488f', alpha=0.8)
plt.xlabel("Trust Score")
plt.ylabel("Count")
plt.title("Distribution of Model Trustworthiness (Adaptive Conformal)")
plt.tight_layout()
plt.show()
plt.savefig("film_dmpnn_tc_trust_distribution.pdf", dpi=500, bbox_inches="tight")

# Scatter trust vs. error
abs_errors = np.abs(y_test_true - y_test_pred)
plt.figure(figsize=(6,4))
plt.scatter(trust_scores, abs_errors, c='#d62728', alpha=0.7)
plt.xlabel("Trust Score")
plt.ylabel("|Error| (K)")
plt.title("Trust Score vs. Prediction Error")
plt.tight_layout()
plt.show()
plt.savefig("film_dmpnn_tc_trust_vs_error.pdf", dpi=500, bbox_inches="tight")

print("Trust score analysis completed successfully.")

# ==========================================================
# Rank molecules by trust score
# ==========================================================
top_n = 10
low_threshold = 0.3

# Sort descending by trust
df_adaptive_sorted = df_adaptive.sort_values(by='trust_score', ascending=False)

print(f"\nTop {top_n} most trustworthy predictions:")
print(df_adaptive_sorted.head(top_n)[['y_true', 'y_pred', 'trust_score', 'interval_width']])

print(f"\nLow-trust molecules (trust < {low_threshold}):")
outliers = df_adaptive_sorted[df_adaptive_sorted['trust_score'] < low_threshold]
print(outliers[['y_true', 'y_pred', 'trust_score', 'interval_width']])

outliers.to_csv('film_dmpnn_low_trust_outliers.csv', index=False)


            