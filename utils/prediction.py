"""
Prediction utilities for JR-MPNN web application
"""

import torch
import thermo
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data

from utils.molecular_graph import atom_features, bond_features


# ── Property configs ──────────────────────────────────────────────────────────
PROPERTY_CONFIG = {
    'Tm': {
        'full_name': 'Melting Point',
        'unit': 'K',
        'hidden_dim': 128,
        'num_layers': 5,
        'model_file': 'jrmpnn_mp.pth',
        'joback_fn': 'Tm',
    },
    'Tb': {
        'full_name': 'Normal Boiling Point',
        'unit': 'K',
        'hidden_dim': 128,
        'num_layers': 5,
        'model_file': 'jrmpnn_tb.pth',
        'joback_fn': 'Tb',
    },
    'Tc': {
        'full_name': 'Critical Temperature',
        'unit': 'K',
        'hidden_dim': 64,
        'num_layers': 3,
        'model_file': 'jrmpnn_tc.pth',
        'joback_fn': 'Tc',
    },
    'Pc': {
        'full_name': 'Critical Pressure',
        'unit': 'MPa',
        'hidden_dim': 32,
        'num_layers': 3,
        'model_file': 'jrmpnn_pc.pth',
        'joback_fn': 'Pc',
    },
    'Vc': {
        'full_name': 'Critical Volume',
        'unit': 'm³/kmol',
        'hidden_dim': 64,
        'num_layers': 3,
        'model_file': 'jrmpnn_vc.pth',
        'joback_fn': 'Vc',
    },
}

# Calibration quantiles (α=0.05, 95% coverage)
# These are approximate values; replace with your actual calibrated quantiles
CALIBRATION_QUANTILES = {
    'Tm': 0.9326,
    'Tb': 0.1810,
    'Tc': 0.5741,
    'Pc': 0.0250,
    'Vc': 0.0044,
}


def validate_smiles(smiles: str):
    if not smiles or not isinstance(smiles, str):
        return False, "SMILES string is empty or invalid."
    smiles = smiles.strip()
    if not smiles:
        return False, "SMILES string is empty."
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES string — could not parse molecule."
    return True, None


def get_joback_prediction(smiles: str, property_name: str):
    """Return Joback baseline for the given property."""
    try:
        J = thermo.Joback(smiles)
        fn = PROPERTY_CONFIG[property_name]['joback_fn']
        if fn == 'Tm':
            val = J.Tm(J.counts)
        elif fn == 'Tb':
            val = J.Tb(J.counts)
        elif fn == 'Tc':
            val = J.Tc(J.counts)
        elif fn == 'Pc':
            mol = Chem.MolFromSmiles(smiles)
            atom_count = Chem.AddHs(mol).GetNumAtoms()
            val = J.Pc(J.counts, atom_count) / 1e6
        elif fn == 'Vc':
            val = J.Vc(J.counts)
            if val is not None:
                val = float(val) * 1e3
        else:
            return None
        return round(float(val), 4) if val is not None else None
    except Exception:
        return None


def get_joback_groups(smiles: str):
    """Return Joback functional group count vector (length 41)."""
    try:
        J = thermo.Joback(smiles)
        fg_list = thermo.J_BIGGS_JOBACK_SMARTS
        return [int(J.counts.get(i + 1, 0)) for i in range(len(fg_list))]
    except Exception:
        return [0] * 41


def mol_to_graph(smiles: str, y_joback: float, groups: list):
    """Convert SMILES string to a PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feat_list = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_index_list, edge_feat_list = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index_list += [[i, j], [j, i]]
        bf = bond_features(bond)
        edge_feat_list += [bf, bf]

    if not edge_feat_list:
        return None

    c_vec = [float(y_joback)] + [float(g) for g in groups]

    return Data(
        x          = torch.stack(atom_feat_list).float(),
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
        edge_attr  = torch.stack(edge_feat_list).float(),
        y          = torch.tensor([0.0], dtype=torch.float32),
        yj         = torch.tensor([float(y_joback)], dtype=torch.float32),
        c          = torch.tensor(c_vec, dtype=torch.float32),
    )


def predict_property(model, graph, property_name: str, device):
    """
    Run inference and return prediction dict.
    Uncertainty interval: ŷ ± q × ‖h_G‖  (embedding-norm proxy for kNN difficulty)
    """
    model.eval()
    graph = graph.to(device)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        y_pred, h_G = model(graph)

    prediction   = float(y_pred.item())
    emb_norm     = float(h_G.norm(dim=1).item())
    q            = CALIBRATION_QUANTILES[property_name]
    half_width   = q * emb_norm

    return {
        'prediction':   round(prediction, 3),
        'lower':        round(max(0.0, prediction - half_width), 3),
        'upper':        round(prediction + half_width, 3),
        'uncertainty':  round(half_width, 3),
        'emb_norm':     round(emb_norm, 3),
    }
