"""
JR-MPNN Thermophysical Property Predictor — Flask Application
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import torch
from flask import Flask, render_template, request, jsonify
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.model_classes import JRMPNN
from utils.molecular_graph import n_atom_features, n_bond_features
from utils.prediction import (
    PROPERTY_CONFIG, validate_smiles, get_joback_prediction,
    get_joback_groups, mol_to_graph, predict_property,
)

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_models = {}
MODELS_DIR = os.path.join(BASE_DIR, 'models')
NODE_DIM = n_atom_features()
EDGE_DIM = n_bond_features()
COND_DIM = 42


def load_models():
    global _models
    for prop, cfg in PROPERTY_CONFIG.items():
        path = os.path.join(MODELS_DIR, cfg['model_file'])
        if not os.path.exists(path):
            print(f"  [WARNING] Model not found: {path}")
            continue
        model = JRMPNN(
            node_dim=NODE_DIM, edge_dim=EDGE_DIM,
            hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'],
            cond_dim=COND_DIM,
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        _models[prop] = model
        print(f"  Loaded {prop}  (hidden={cfg['hidden_dim']}, layers={cfg['num_layers']})")
    print(f"  Device: {device} | Loaded: {list(_models.keys())}")


def smiles_to_svg(smiles, width=420, height=300):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.bondLineWidth = 1.8
    opts.padding = 0.14
    try:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    except Exception:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace("rect style='opacity:1.0;fill:#FFFFFF",
                      "rect style='opacity:0.0;fill:#FFFFFF")
    return svg


@app.route('/')
def index():
    return render_template('index.html',
                           properties=PROPERTY_CONFIG,
                           loaded=list(_models.keys()))


@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.get_json() or {}
    smiles = data.get('smiles', '').strip()
    valid, err = validate_smiles(smiles)
    if not valid:
        return jsonify({'success': False, 'error': err}), 400
    svg = smiles_to_svg(smiles)
    if svg is None:
        return jsonify({'success': False, 'error': 'Could not render molecule.'}), 400
    mol = Chem.MolFromSmiles(smiles)
    from rdkit.Chem import rdMolDescriptors
    info = {
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'atoms': mol.GetNumAtoms(),
        'bonds': mol.GetNumBonds(),
        'rings': mol.GetRingInfo().NumRings(),
    }
    return jsonify({'success': True, 'svg': svg, 'info': info})


@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.get_json() or {}
    smiles = data.get('smiles', '').strip()
    valid, err = validate_smiles(smiles)
    if not valid:
        return jsonify({'success': False, 'error': err}), 400
    groups = get_joback_groups(smiles)
    results = {}
    for prop in PROPERTY_CONFIG:
        if prop not in _models:
            results[prop] = {'error': 'model not loaded'}
            continue
        y_joback = get_joback_prediction(smiles, prop)
        if y_joback is None:
            results[prop] = {'error': 'Joback failed'}
            continue
        graph = mol_to_graph(smiles, y_joback, groups)
        if graph is None:
            results[prop] = {'error': 'graph build failed'}
            continue
        try:
            r = predict_property(_models[prop], graph, prop, device)
            cfg = PROPERTY_CONFIG[prop]
            results[prop] = {
                'full_name': cfg['full_name'],
                'unit': cfg['unit'],
                'prediction': r['prediction'],
                'lower': r['lower'],
                'upper': r['upper'],
                'joback': round(y_joback, 3),
                'residual': round(r['prediction'] - y_joback, 3),
            }
        except Exception as e:
            results[prop] = {'error': str(e)}
    return jsonify({'success': True, 'results': results})


@app.route('/batch', methods=['POST'])
def batch():
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    prop = data.get('property', '').strip()
    if not smiles_list or len(smiles_list) > 500:
        return jsonify({'success': False, 'error': 'Provide 1–500 SMILES strings.'}), 400
    if prop not in PROPERTY_CONFIG:
        return jsonify({'success': False, 'error': f'Unknown property: {prop}'}), 400
    if prop not in _models:
        return jsonify({'success': False, 'error': f'Model for {prop} not loaded.'}), 503
    results = []
    for entry in smiles_list:
        smiles = entry.get('smiles', '').strip()
        name = entry.get('name', smiles[:24])
        valid, err = validate_smiles(smiles)
        if not valid:
            results.append({'name': name, 'smiles': smiles, 'error': err})
            continue
        y_joback = get_joback_prediction(smiles, prop)
        if y_joback is None:
            results.append({'name': name, 'smiles': smiles, 'error': 'Joback failed'})
            continue
        graph = mol_to_graph(smiles, y_joback, get_joback_groups(smiles))
        if graph is None:
            results.append({'name': name, 'smiles': smiles, 'error': 'graph build failed'})
            continue
        try:
            r = predict_property(_models[prop], graph, prop, device)
            results.append({
                'name': name, 'smiles': smiles,
                'prediction': r['prediction'], 'lower': r['lower'],
                'upper': r['upper'], 'joback': round(y_joback, 3),
                'unit': PROPERTY_CONFIG[prop]['unit'],
            })
        except Exception as e:
            results.append({'name': name, 'smiles': smiles, 'error': str(e)})
    return jsonify({'success': True, 'results': results,
                    'property': prop,
                    'full_name': PROPERTY_CONFIG[prop]['full_name'],
                    'unit': PROPERTY_CONFIG[prop]['unit']})


if __name__ == '__main__':
    print("\nLoading JR-MPNN models...")
    load_models()
    print("\nStarting server on http://0.0.0.0:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
