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
utils_path = os.path.expanduser('~/deeplearning/residual/utils')
sys.path.insert(0, utils_path)
from molecular_graph import atom_features, bond_features, get_bond_pair, n_atom_features, n_bond_features
from train_validate import train_validate, train_validate_film, plot_loss_curves


print('running', flush=True)

print('pytorch version: ', torch.__version__)
print('rdkit version: ', rdkit.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'device type: ', {device}, flush=True)

data_dir = os.path.expanduser('~/deeplearning/residual/data')
train_data = os.path.join(data_dir, 'VC_train.csv')
test_data = os.path.join(data_dir, 'VC_test.csv')

df = pd.read_csv(train_data)
df2 = pd.read_csv(test_data)

def mol_to_graph(smiles, y):
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
        y=torch.tensor([y], dtype=torch.float32)  # <- ensure float32 targets
    )

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

    def forward(self, x, edge_index, edge_attr):
        x = x.float()            # <- keep model in float32
        edge_attr = edge_attr.float()
        edge_attr = self.edge_update(x, edge_index, edge_attr)
        x = self.node_update(x, edge_index, edge_attr)
        return x, edge_attr

    def edge_update(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_source = x[row]

        edge_messages = []
        for edge_idx in range(edge_index.size(1)):
            src = edge_index[0, edge_idx]
            mask = edge_index[1] == src
            if mask.any():
                incoming_edges = edge_attr[mask]
                agg_message = incoming_edges.sum(dim=0)
            else:
                agg_message = torch.zeros_like(edge_attr[edge_idx])
            edge_messages.append(agg_message)

        edge_messages = torch.stack(edge_messages)
        edge_input = torch.cat([edge_messages, x_source], dim=1)
        new_edge_attr = self.edge_mlp(edge_input)
        if new_edge_attr.size(1) == edge_attr.size(1):
            new_edge_attr = new_edge_attr + edge_attr
        return new_edge_attr

    def node_update(self, x, edge_index, edge_attr):
        row, col = edge_index
        node_messages = torch.zeros(x.size(0), edge_attr.size(1), device=x.device, dtype=x.dtype)
        node_messages.index_add_(0, col, edge_attr)
        node_input = torch.cat([x, node_messages], dim=1)
        new_x = self.node_mlp(node_input)
        if new_x.size(1) == x.size(1):
            new_x = new_x + x
        return new_x

class DMPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        self.dmpnn_layers = nn.ModuleList(
            [DMPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x.float(), data.edge_index, data.edge_attr.float()
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        for layer in self.dmpnn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        x = global_add_pool(x, batch=data.batch)
        return self.readout(x)

train_data_graphs = [mol_to_graph(row.SMILES, row.VC) for _, row in df.iterrows()]
train_data_graphs = [data for data in train_data_graphs if data is not None]

val_data_graphs = [mol_to_graph(row.SMILES, row.VC) for _, row in df2.iterrows()]
val_data_graphs = [data for data in val_data_graphs if data is not None]

print("Training data length: ", len(train_data_graphs))
print("Test data length: ", len(val_data_graphs))

hidden_size = 32
num_layers = 3
output_dim = 1
lr = 0.0037

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

    model = DMPNN(
        node_dim=n_atom_features(),
        edge_dim=n_bond_features(),
        hidden_dim=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = train_validate(model, train_loader, val_loader, optimizer, criterion, epochs=800)
    #plot_loss_curves(train_losses, val_losses)

    model.eval()
    with torch.no_grad():
        y_train_pred, y_train_true = [], []
        for batch in train_loader:
            batch = batch.to(device)
            outputs = model(batch).cpu().numpy().flatten()
            y_train_pred.extend(outputs)
            y_train_true.extend(batch.y.cpu().numpy().flatten())

        r2_train = r2_score(y_train_true, y_train_pred)
        mae_train = mean_absolute_error(y_train_true, y_train_pred)
        mse_train = mean_squared_error(y_train_true, y_train_pred)
        rmse_train = np.sqrt(mse_train)  # <- fixed
        mape_train = mean_absolute_percentage_error(y_train_true, y_train_pred)

        y_val_pred, y_val_true = [], []
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch).cpu().numpy().flatten()
            y_val_pred.extend(outputs)
            y_val_true.extend(batch.y.cpu().numpy().flatten())

        r2_val = r2_score(y_val_true, y_val_pred)
        mae_val = mean_absolute_error(y_val_true, y_val_pred)
        mse_val = mean_squared_error(y_val_true, y_val_pred)
        rmse_val = np.sqrt(mse_val)  # <- fixed
        mape_val = mean_absolute_percentage_error(y_val_true, y_val_pred)

    r2_train_results.append(r2_train); mae_train_results.append(mae_train)
    mse_train_results.append(mse_train); rmse_train_results.append(rmse_train); mape_train_results.append(mape_train)
    r2_val_results.append(r2_val); mae_val_results.append(mae_val)
    mse_val_results.append(mse_val); rmse_val_results.append(rmse_val); mape_val_results.append(mape_val)

    print(
        f"fold {fold} completed. "
        f"train r2: {r2_train:.4f}, mae:{mae_train:.4f}, mse: {mse_train:.4f}, rmse: {rmse_train:.4f}, mape: {mape_train:.4f}"
    )
    print(
        f"val r2: {r2_val:.4f}, mae:{mae_val:.4f}, mse: {mse_val:.4f}, rmse: {rmse_val:.4f}, mape: {mape_val:.4f}\n"
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
df_results.to_csv('dmpnn_cross_validation_results.csv', index=False)

train_loader = DataLoader(train_data_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data_graphs, batch_size=32, shuffle=False)

def evaluate_model(model, data_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch)
            predictions.append(out.cpu().numpy().flatten())
            actuals.append(batch.y.cpu().numpy().flatten())
    return np.concatenate(predictions), np.concatenate(actuals)  # <- fixed

# evaluate training and test
train_predictions, train_actuals = evaluate_model(model, train_loader)
test_predictions, test_actuals = evaluate_model(model, val_loader)

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

torch.save(model.state_dict(), "dmpnn_vc.pth")

def plot_parity_test(actuals, predictions):
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, c='#1e488f', alpha=0.6, s=40)
    plt.plot([200, 1200], [200, 1200], 'k--', linewidth=2)
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
    plt.savefig("dmpnn_test_parity_tc.pdf", format="pdf", dpi=500, bbox_inches="tight")
#    plt.show()

def plot_parity_train(actuals, predictions):
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, alpha=0.6, c='red', s=40)
    plt.plot([200, 1200], [200, 1200], 'k--', linewidth=2)
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
    plt.savefig("dmpnn_train_parity_tc.pdf", format="pdf", dpi=500, bbox_inches="tight")
#    plt.show()

#plot_parity_train(train_actuals, train_predictions)
#plot_parity_test(test_actuals, test_predictions)

df_train_predictions = pd.DataFrame({'y_train': train_actuals, 'y_train_pred': train_predictions})
df_train_predictions.to_csv('dmpnn_vc_train_pred.csv', index=False)

df_test_predictions = pd.DataFrame({'y_test': test_actuals, 'y_test_pred': test_predictions})
df_test_predictions.to_csv('dmpnn_vc_test_pred.csv', index=False)  
