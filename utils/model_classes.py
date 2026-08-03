"""
JR-MPNN: Joback-Residual Message Passing Neural Network
Architecture: Triple pooling + deep readout MLP with skip connection
"""

import torch
import torch.nn as nn
from torch_geometric.nn import (MessagePassing, global_add_pool,
                                 global_mean_pool, global_max_pool)


class DMPNNLayer(MessagePassing):
    """DMPNN layer with XOR backtracking prevention and residual connections."""
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
        x, edge_attr = x.float(), edge_attr.float()
        edge_attr = self.edge_update(x, edge_index, edge_attr)
        x_new = self.node_update(x, edge_index, edge_attr)
        return x_new, edge_attr

    def edge_update(self, x, edge_index, edge_attr):
        src, tgt = edge_index
        incoming = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        incoming.index_add_(0, tgt, edge_attr)
        rev = torch.arange(edge_attr.size(0), device=x.device) ^ 1
        msg = incoming[src] - edge_attr[rev]          # backtracking prevention
        new_e = self.edge_mlp(torch.cat([msg, x[src], edge_attr], dim=1))
        return new_e + edge_attr                       # residual

    def node_update(self, x, edge_index, edge_attr):
        _, col = edge_index
        agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device)
        agg.index_add_(0, col, edge_attr)
        new_x = self.node_mlp(torch.cat([x, agg], dim=1))
        return new_x + x                               # residual


class JRMPNN(nn.Module):
    """
    Joback-Residual MPNN
    - Triple pooling: global_add + global_mean + global_max
    - Joback conditioning vector concatenated before readout
    - 4-layer MLP readout with skip connection
    - Residual output: y_hat = y_joback + r_hat

    forward() returns (y_hat, h_G)
        h_G  [B, 3*hidden_dim] — graph embedding for uncertainty estimation
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

        in_dim = 3 * hidden_dim + cond_dim
        self.fc1  = nn.Linear(in_dim,          hidden_dim)
        self.fc2  = nn.Linear(hidden_dim,       hidden_dim)
        self.fc3  = nn.Linear(hidden_dim,       hidden_dim // 2)
        self.fc4  = nn.Linear(hidden_dim // 2,  output_dim)
        self.skip = nn.Linear(in_dim,           hidden_dim)
        self.drop = nn.Dropout(0.2)
        self.act  = nn.ReLU()

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        batch = data.batch
        c = data.c
        yj = data.yj

        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        for layer in self.dmpnn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        # Triple pooling
        h_G = torch.cat([
            global_add_pool(x,  batch),
            global_mean_pool(x, batch),
            global_max_pool(x,  batch),
        ], dim=-1)                                     # [B, 3*hidden_dim]

        B = h_G.size(0)
        if c.dim() == 1:
            c = c.view(B, self.cond_dim)

        h_cat = torch.cat([h_G, c], dim=-1)

        h = self.drop(self.act(self.fc1(h_cat)))
        h = self.drop(self.act(self.fc2(h) + self.skip(h_cat)))
        h = self.act(self.fc3(h))
        r_hat = self.fc4(h).squeeze(-1)

        y_hat = yj.view(-1) + r_hat
        return y_hat, h_G
