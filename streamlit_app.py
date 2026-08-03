"""JR-MPNN Thermophysical Property Predictor — Streamlit Application"""

import os, sys, warningswarnings.filterwarnings("ignore")

import torchimport streamlit as stimport pandas as pdfrom rdkit import Chemfrom rdkit.Chem.Draw import rdMolDraw2D

BASE_DIR = os.path.dirname(os.path.abspath(file))sys.path.insert(0, BASE_DIR)

from utils.model_classes import JRMPNNfrom utils.molecular_graph import n_atom_features, n_bond_featuresfrom utils.prediction import (PROPERTY_CONFIG, validate_smiles, get_joback_prediction,get_joback_groups, mol_to_graph, predict_property,)

st.set_page_config(page_title="JR-MPNN Property Predictor",page_icon="⚗",layout="wide",initial_sidebar_state="collapsed",)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')MODELS_DIR = os.path.join(BASE_DIR, 'models')NODE_DIM   = n_atom_features()EDGE_DIM   = n_bond_features()COND_DIM   = 42

── CSS ───────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --page-gutter: max(32px, calc((100vw - 1400px) / 2));
}

/* ── Hide Streamlit chrome ─────────────────────────────────── */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
.stDeployButton,
[data-testid="collapsedControl"],
#MainMenu { display: none !important; }
footer { visibility: hidden !important; }

/* ── Base ──────────────────────────────────────────────────── */
html, body { font-family: 'Inter', sans-serif !important; }
.stApp { background: #FFFFFF !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Custom header ─────────────────────────────────────────── */
.jr-header {
    background: #002E5D;
    padding: 14px var(--page-gutter);
    border-bottom: 1px solid #1a3f6f;
    margin-bottom: 0;
}
.jr-header h1 {
    font-family: 'Inter', sans-serif;
    font-size: 28px;
    font-weight: 400;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.2px;
    line-height: 1;
}
.jr-header p {
    font-family: 'DM Mono', 'Fira Code', monospace;
    font-size: 14px;
    color: rgba(255,255,255,0.8);
    margin: 4px 0 0;
    letter-spacing: 0.3px;
}

/* ── Tabs — fused with header ──────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #002E5D !important;
    padding: 10px var(--page-gutter) 0 var(--page-gutter) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: none !important;
    border: none !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    border-radius: 6px 6px 0 0 !important;
    transition: background 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #2e3340 !important;
    color: #ffffff !important;
}
.stTabs [aria-selected="true"] {
    background: #2e3340 !important;
    color: #ffffff !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
    padding: 32px var(--page-gutter) 64px !important;
    background: #ffffff !important;
}

/* ── Input widgets ─────────────────────────────────────────── */
.stTextInput label, .stSelectbox label, .stFileUploader label {
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    color: #0d0d0d !important;
    letter-spacing: 0 !important;
}
.stTextInput input {
    font-family: 'DM Mono', 'Fira Code', monospace !important;
    font-size: 15px !important;
    background: #f1f3f5 !important;
    border: 1px solid #ced4da !important;
    border-radius: 6px !important;
    color: #0d0d0d !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 3px rgba(45,212,191,0.1) !important;
}
.stSelectbox [data-baseweb="select"] > div:first-child {
    background: #f1f3f5 !important;
    border: 1px solid #ced4da !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploadDropzone"] {
    background: #f1f3f5 !important;
    border: 1px dashed #ced4da !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #0d9488 !important;
    background: rgba(45,212,191,0.06) !important;
}

/* ── Buttons ───────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: #002E5D !important;
    color: #ffffff !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background: #0047BA !important;
    color: #ffffff !important;
    border: none !important;
}
.stButton > button[kind="secondary"] {
    background: #e9ecef !important;
    color: #0d0d0d !important;
    border: 1px solid #ced4da !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #0d9488 !important;
    color: #0d9488 !important;
}
.stDownloadButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    background: transparent !important;
    border: 1px solid #002E5D !important;
    color: #343a40 !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
}
.stDownloadButton > button:hover {
    background: #f1f3f5 !important;
    color: #0d0d0d !important;
}

/* ── Jr custom card ────────────────────────────────────────── */
.jr-card {
    background: #ffffff;
    border: 1px solid #002E5D;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
    margin-bottom: 16px;
}
.jr-card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid #002E5D;
}
.jr-card-icon {
    font-size: 16px;
    color: #2dd4bf;
    width: 32px; height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(45,212,191,0.12);
    border: 1px solid rgba(45,212,191,0.2);
    border-radius: 6px;
    flex-shrink: 0;
}
.jr-card h2 {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: #0d0d0d;
    margin: 0;
    letter-spacing: 0.2px;
}

/* ── Molecule badges ───────────────────────────────────────── */
.mol-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }
.mol-badge {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    padding: 3px 8px;
    border-radius: 3px;
    background: #e9ecef;
    border: 1px solid #002E5D;
    color: #343a40;
}

/* ── Molecule SVG wrap ─────────────────────────────────────── */
.mol-svg-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    background: #f1f3f5;
    border-radius: 6px;
    padding: 12px;
    min-height: 240px;
}
.mol-svg-wrap svg { max-width: 100%; height: auto; }

/* ── Divider ───────────────────────────────────────────────── */
.jr-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #6c757d;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    letter-spacing: 0.5px;
    margin: 8px 0;
}
.jr-divider::before, .jr-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #002E5D;
}

/* ── Idle state ────────────────────────────────────────────── */
.idle-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 500px;
    color: #6c757d;
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    text-align: center;
    line-height: 1.7;
}
.idle-state strong { color: #343a40; }

/* ── Property results grid ─────────────────────────────────── */
.results-all {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
.prop-card {
    background: #ffffff;
    border: 2px solid #002E5D;
    border-radius: 10px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
    animation: slideUp 0.35s ease both;
}
.prop-card:hover { border-color: #ced4da; transform: translateY(-2px); }
.prop-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.prop-card-header { margin-bottom: 10px; }
.prop-name {
    font-family: 'DM Mono', 'Fira Code', monospace;
    font-size: 18px;
    font-weight: 600;
    color: #0d0d0d;
}
.prop-fullname {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    color: #343a40;
    margin-top: 2px;
}
.prop-value {
    font-family: 'DM Mono', 'Fira Code', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #0d0d0d;
    margin-bottom: 14px;
    line-height: 1.2;
}
.prop-value span {
    font-size: 14px;
    font-weight: 400;
    color: #6c757d;
    margin-left: 4px;
}
.ci-bar-wrap { margin-bottom: 14px; }
.ci-label {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #6c757d;
    margin-bottom: 6px;
}
.ci-bar {
    height: 6px;
    background: #e9ecef;
    border-radius: 3px;
    position: relative;
    overflow: visible;
}
.ci-fill {
    position: absolute;
    top: 0; height: 100%;
    border-radius: 3px;
    background: #002E5D;
}
.ci-marker {
    position: absolute;
    top: -3px;
    width: 2px; height: 12px;
    background: #2dd4bf;
    border-radius: 1px;
    box-shadow: 0 0 6px #2dd4bf;
}
.prop-meta {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #6c757d;
}
.prop-meta .meta-val { color: #002E5D; }
.residual-pos { color: #34d399 !important; }
.residual-neg { color: #f87171 !important; }
.prop-card.error-card::before { background: #f87171; }
.prop-card.error-card .prop-value { font-size: 13px; color: #f87171; }

/* ── Batch info ────────────────────────────────────────────── */
.batch-info {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    color: #343a40;
    background: #e9ecef;
    border: 1px solid #002E5D;
    border-radius: 6px;
    padding: 12px 16px;
    line-height: 1.6;
    margin-bottom: 20px;
}
.batch-info code {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #2dd4bf;
    background: rgba(45,212,191,0.1);
    padding: 1px 5px;
    border-radius: 3px;
}

/* ── About card ────────────────────────────────────────────── */
.about-card {
    background: #ffffff;
    border: 1px solid #002E5D;
    border-radius: 14px;
    padding: 36px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
    max-width: 860px;
}
.about-title {
    font-family: 'DM Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    color: #0d0d0d;
    margin: 0 0 20px;
}
.about-sub {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: #0d0d0d;
    letter-spacing: 0.3px;
    margin: 28px 0 14px;
}
.about-body {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
    color: #343a40;
    line-height: 1.75;
    margin-bottom: 10px;
}
.about-body strong { color: #0d0d0d; }
.equation {
    font-family: 'DM Mono', monospace;
    font-size: 18px;
    color: #21242d;
    background: #f1f3f5;
    padding: 14px 20px;
    border-radius: 0 6px 6px 0;
    margin: 18px 0;
}
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 16px 0;
}
.feature-item {
    background: #f1f3f5;
    border: 1px solid #002E5D;
    border-radius: 10px;
    padding: 16px;
}
.feature-tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
    color: #38bdf8;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    padding: 3px 9px;
    border-radius: 3px;
    margin-bottom: 10px;
}
.feature-item p {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: #343a40;
    line-height: 1.65;
    margin: 0;
}
.about-table {
    width: 100%;
    border-collapse: collapse;
    background: #f1f3f5;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 12px;
}
.about-table th {
    text-align: left;
    padding: 10px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #6c757d;
    background: #e9ecef;
    border-bottom: 1px solid #002E5D;
}
.about-table td {
    padding: 10px 16px;
    border-bottom: 1px solid #dee2e6;
    color: #343a40;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
}
.about-table tr:last-child td { border-bottom: none; }
.about-table code {
    font-family: 'DM Mono', monospace;
    color: #2dd4bf;
    background: rgba(45,212,191,0.1);
    padding: 2px 6px;
    border-radius: 3px;
}
.status-ok   { color: #34d399; font-family: 'DM Mono', monospace; }
.status-miss { color: #f87171; font-family: 'DM Mono', monospace; }

/* ── Animations ────────────────────────────────────────────── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: none; }
}

/* ── Responsive page gutters ───────────────────────────────── */
@media (max-width: 700px) {
    :root {
        --page-gutter: 18px;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 24px !important;
        padding-bottom: 40px !important;
    }
}

/* ── Dataframe overrides ───────────────────────────────────── */
[data-testid="stDataFrame"] iframe {
    border-radius: 10px;
}
</style>

"""

HEADER_HTML = """

<div class="jr-header">
  <div>
    <h1>Thermophysical Property Predictor</h1>
    <p>Joback Residual · Message Passing Neural Network</p>
  </div>
</div>
"""

── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")def load_models():models = {}for prop, cfg in PROPERTY_CONFIG.items():path = os.path.join(MODELS_DIR, cfg['model_file'])if not os.path.exists(path):continuemodel = JRMPNN(node_dim=NODE_DIM, edge_dim=EDGE_DIM,hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'],cond_dim=COND_DIM,).to(DEVICE)model.load_state_dict(torch.load(path, map_location=DEVICE))model.eval()models[prop] = modelreturn models

── Helpers ───────────────────────────────────────────────────────────────────

def smiles_to_svg(smiles: str, width=420, height=300) -> str | None:mol = Chem.MolFromSmiles(smiles)if mol is None:return Nonedrawer = rdMolDraw2D.MolDraw2DSVG(width, height)opts = drawer.drawOptions()opts.addAtomIndices = Falseopts.bondLineWidth  = 1.8opts.padding        = 0.14try:rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)except Exception:drawer.DrawMolecule(mol)drawer.FinishDrawing()svg = drawer.GetDrawingText()svg = svg.replace("rect style='opacity:1.0;fill:#FFFFFF","rect style='opacity:0.0;fill:#FFFFFF")return svg

def run_single(smiles: str, models: dict):valid, err = validate_smiles(smiles)if not valid:return None, errgroups  = get_joback_groups(smiles)results = {}for prop in PROPERTY_CONFIG:if prop not in models:continuey_joback = get_joback_prediction(smiles, prop)if y_joback is None:results[prop] = {'error': 'Joback method failed'}continuegraph = mol_to_graph(smiles, y_joback, groups)if graph is None:results[prop] = {'error': 'Graph build failed'}continuetry:r   = predict_property(models[prop], graph, prop, DEVICE)cfg = PROPERTY_CONFIG[prop]results[prop] = {'full_name':  cfg['full_name'],'unit':       cfg['unit'],'prediction': r['prediction'],'lower':      r['lower'],'upper':      r['upper'],'joback':     round(y_joback, 3),'residual':   round(r['prediction'] - y_joback, 4 if prop == 'Vc' else 3),}except Exception as e:results[prop] = {'error': str(e)}return results, None

def run_batch(entries: list, prop: str, models: dict) -> list:unit = PROPERTY_CONFIG[prop]['unit']rows = []for entry in entries:smiles = entry.get('smiles', '').strip()name   = entry.get('name', smiles[:30])valid, err = validate_smiles(smiles)if not valid:rows.append({'Name': name, 'SMILES': smiles, 'Prediction': None,'Lower 95%': None, 'Upper 95%': None,'Joback': None, 'Unit': unit,'Status': f'Error: {err}'})continuey_joback = get_joback_prediction(smiles, prop)if y_joback is None:rows.append({'Name': name, 'SMILES': smiles, 'Prediction': None,'Lower 95%': None, 'Upper 95%': None,'Joback': None, 'Unit': unit, 'Status': 'Joback failed'})continuegraph = mol_to_graph(smiles, y_joback, get_joback_groups(smiles))if graph is None:rows.append({'Name': name, 'SMILES': smiles, 'Prediction': None,'Lower 95%': None, 'Upper 95%': None,'Joback': None, 'Unit': unit, 'Status': 'Graph build failed'})continuetry:r = predict_property(models[prop], graph, prop, DEVICE)rows.append({'Name': name, 'SMILES': smiles,'Prediction': r['prediction'],'Lower 95%':  r['lower'],'Upper 95%':  r['upper'],'Joback':     round(y_joback, 3),'Unit':       unit, 'Status': 'OK',})except Exception as e:rows.append({'Name': name, 'SMILES': smiles, 'Prediction': None,'Lower 95%': None, 'Upper 95%': None,'Joback': None, 'Unit': unit, 'Status': f'Error: {e}'})return rows

def build_prop_card(prop: str, data: dict, delay_ms: int = 0) -> str:cfg = PROPERTY_CONFIG[prop]if 'error' in data:return f"""

<div class="prop-card error-card" style="animation-delay:{delay_ms}ms">
  <div class="prop-card-header">
    <div class="prop-name">{prop}</div>
    <div class="prop-fullname">{cfg['full_name']}</div>
  </div>
  <div class="prop-value">Error</div>
  <div class="prop-meta" style="color:#f87171">{data['error']}</div>
</div>"""

r        = data
rng      = r['upper'] - r['lower']
pad      = rng * 0.3
bar_min  = r['lower'] - pad
bar_max  = r['upper'] + pad
bar_rng  = bar_max - bar_min

fill_left   = (r['lower']      - bar_min) / bar_rng * 100
fill_right  = 100 - (r['upper'] - bar_min) / bar_rng * 100
marker_left = (r['prediction'] - bar_min) / bar_rng * 100

res       = r['residual']
res_cls   = 'residual-pos' if res >= 0 else 'residual-neg'
res_sign  = '+' if res >= 0 else ''
res_fmt   = '.4f' if prop == 'Vc' else '.2f'
val_fmt   = '.4f' if prop == 'Vc' else '.3f'

return f"""

<div class="prop-card" style="animation-delay:{delay_ms}ms">
  <div class="prop-card-header">
    <div class="prop-name">{prop}</div>
    <div class="prop-fullname">{cfg['full_name']}</div>
  </div>
  <div class="prop-value">{r['prediction']:{val_fmt}} <span>{cfg['unit']}</span></div>
  <div class="ci-bar-wrap">
    <div class="ci-label">
      <span>{r['lower']:{val_fmt}}</span>
      <span>95% CI</span>
      <span>{r['upper']:{val_fmt}}</span>
    </div>
    <div class="ci-bar">
      <div class="ci-fill" style="left:{fill_left:.1f}%;right:{fill_right:.1f}%"></div>
      <div class="ci-marker" style="left:calc({marker_left:.1f}% - 1px)"></div>
    </div>
  </div>
  <div class="prop-meta">
    <span>Joback: <span class="meta-val">{r['joback']:{val_fmt}}</span></span>
    <span>&#x394;: <span class="{res_cls}">{res_sign}{res:{res_fmt}}</span></span>
  </div>
</div>"""

── Main ──────────────────────────────────────────────────────────────────────

def main():models = load_models()loaded = list(models.keys())

for key in ('results', 'show_mode', 'show_prop'):
    if key not in st.session_state:
        st.session_state[key] = None

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(HEADER_HTML, unsafe_allow_html=True)

if not loaded:
    st.error("No model weights found in `models/`. Copy your `.pth` files there and restart.")

tab_single, tab_batch, tab_about = st.tabs(
    ["Single Molecule", "Batch", "About"]
)

# ── Single Molecule ───────────────────────────────────────────────────────
with tab_single:
    col_left, col_right = st.columns([5, 7], gap="large")

    with col_left:
        # Card header
        st.markdown("""

<div style="display:flex;align-items:center;gap:12px;
            margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid #002E5D;">
  <span class="jr-card-icon">&#x25B3;</span>
  <span style="font-family:'Inter',sans-serif;font-size:18px;
               font-weight:600;color:#0d0d0d;">Molecule Input</span>
</div>""", unsafe_allow_html=True)

        smiles = st.text_input(
            "SMILES String",
            placeholder="e.g.  CCO   or   c1ccccc1",
        )

        predict_all = st.button(
            "Predict All Properties", type="primary",
            use_container_width=True, disabled=not loaded,
        )

        st.markdown(
            '<div class="jr-divider"><span>or predict a single property</span></div>',
            unsafe_allow_html=True,
        )

        available = {
            k: f"{k} — {v['full_name']}"
            for k, v in PROPERTY_CONFIG.items() if k in loaded
        }
        prop_sel = st.selectbox(
            "Property",
            [""] + list(available.keys()),
            format_func=lambda k: "— select —" if k == "" else available.get(k, k),
        )

        predict_one = st.button(
            "Predict Selected", type="secondary",
            use_container_width=True,
            disabled=(not loaded or not prop_sel),
        )

        # Run inference
        if predict_all:
            if not smiles:
                st.warning("Please enter a SMILES string.")
            else:
                with st.spinner("Running inference…"):
                    results, err = run_single(smiles, models)
                if err:
                    st.error(err)
                else:
                    st.session_state.results   = results
                    st.session_state.show_mode = 'all'
                    st.session_state.show_prop = None

        elif predict_one:
            if not smiles:
                st.warning("Please enter a SMILES string.")
            else:
                with st.spinner("Running inference…"):
                    results, err = run_single(smiles, models)
                if err:
                    st.error(err)
                else:
                    st.session_state.results   = results
                    st.session_state.show_mode = 'single'
                    st.session_state.show_prop = prop_sel

        # Molecule visualisation
        if smiles:
            valid, verr = validate_smiles(smiles)
            if valid:
                svg = smiles_to_svg(smiles)
                if svg:
                    st.markdown(f"""

<div class="jr-card">
  <div class="jr-card-header">
    <span class="jr-card-icon">&#x25C8;</span>
    <h2>Structure</h2>
  </div>
  <div class="mol-svg-wrap">{svg}</div>
</div>""", unsafe_allow_html=True)
                else:
                    st.warning(verr)

    with col_right:
        if st.session_state.results and st.session_state.show_mode:
            ORDER       = ['Tm', 'Tb', 'Tc', 'Pc', 'Vc']
            show_props  = (
                [p for p in ORDER if p in st.session_state.results]
                if st.session_state.show_mode == 'all'
                else ([st.session_state.show_prop]
                      if st.session_state.show_prop in st.session_state.results
                      else [])
            )
            cards_html = ''.join(
                build_prop_card(p, st.session_state.results[p], i * 60)
                for i, p in enumerate(show_props)
            )
            grid_class = 'results-all' if st.session_state.show_mode == 'all' else ''
            st.markdown(
                f'<div class="{grid_class}">{cards_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""

<div class="idle-state">
  <p>Enter a SMILES string and click<br>
  <strong>Predict All Properties</strong> to begin.</p>
</div>""", unsafe_allow_html=True)

# ── Batch ─────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("""

<div class="batch-info">
  Upload a CSV with a <code>SMILES</code> column (optional <code>Name</code> column).
  Maximum 500 molecules per batch.
</div>""", unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2, gap="large")
    with col_b1:
        b_available = {
            k: f"{k} — {v['full_name']}"
            for k, v in PROPERTY_CONFIG.items() if k in loaded
        }
        batch_prop = st.selectbox(
            "Property",
            list(b_available.keys()),
            format_func=lambda k: b_available[k],
            key="batch_prop",
        ) if b_available else None

    with col_b2:
        uploaded = st.file_uploader("CSV File", type="csv")

    col_run, col_dl = st.columns([1, 3])
    with col_run:
        run_batch_btn = st.button(
            "Run Batch", type="primary",
            disabled=(batch_prop is None or uploaded is None),
        )

    if run_batch_btn and uploaded and batch_prop:
        df_in      = pd.read_csv(uploaded)
        smiles_col = next(
            (c for c in df_in.columns if c.upper() == 'SMILES'), None
        )
        if smiles_col is None:
            st.error("CSV must contain a **SMILES** column.")
        else:
            name_col = next(
                (c for c in df_in.columns if c.lower() == 'name'), None
            )
            entries = [
                {'smiles': str(row[smiles_col]),
                 **(({'name': str(row[name_col])}) if name_col else {})}
                for _, row in df_in.iterrows()
            ]
            if len(entries) > 500:
                st.warning(f"Truncating to first 500 rows ({len(entries)} provided).")
                entries = entries[:500]

            with st.spinner(f"Running batch ({len(entries)} molecules)…"):
                rows = run_batch(entries, batch_prop, models)

            df_out = pd.DataFrame(rows)
            n_ok   = (df_out['Status'] == 'OK').sum()
            n_err  = len(df_out) - n_ok
            st.success(f"Done — {n_ok} succeeded, {n_err} failed.")
            st.dataframe(df_out, use_container_width=True, hide_index=True)

            csv_bytes = df_out.to_csv(index=False).encode()
            st.download_button(
                "⤓  Download CSV",
                data=csv_bytes,
                file_name=f"jrmpnn_{batch_prop.lower()}_predictions.csv",
                mime="text/csv",
            )

# ── About ─────────────────────────────────────────────────────────────────
with tab_about:
    status_rows = ''.join(
        f"""<tr>

  <td><code>{prop}</code></td>
  <td>{cfg['full_name']}</td>
  <td>{cfg['unit']}</td>
  <td>{cfg['hidden_dim']}</td>
  <td>{cfg['num_layers']}</td>
  <td class="{'status-ok' if prop in loaded else 'status-miss'}">
    {'&#x2713; Loaded' if prop in loaded else '&#x2717; Not found'}
  </td>
</tr>"""
            for prop, cfg in PROPERTY_CONFIG.items()
        )

    st.markdown(f"""

<div class="about-card">
  <h2 class="about-title">JR-MPNN Architecture</h2>
  <p class="about-body">
    The <strong>Joback-Residual Message Passing Neural Network (JR-MPNN)</strong> is a hybrid
    graph neural network that combines classical Joback group-contribution estimates with a
    data-driven residual correction. Predictions follow the form:
  </p>
  <div class="equation">
    &#375; = y<sub>Joback</sub> + r&#x0302;(G, c)
  </div>
  <p class="about-body">
    where <strong>G</strong> is the molecular graph and <strong>c</strong> is the Joback conditioning
    vector [y<sub>Joback</sub>, g<sub>1</sub>, &hellip;, g<sub>41</sub>].
  </p>

  <h3 class="about-sub">Key Design Choices</h3>
  <div class="feature-grid">
    <div class="feature-item">
      <span class="feature-tag">Triple Pooling</span>
      <p>Global add + mean + max pooling concatenated into a single embedding &mdash; captures size,
      density, and dominant features simultaneously.</p>
    </div>
    <div class="feature-item">
      <span class="feature-tag">XOR Backtracking Prevention</span>
      <p>Directed message passing prevents messages from immediately reversing along the same bond,
      improving long-range communication.</p>
    </div>
    <div class="feature-item">
      <span class="feature-tag">Joback Conditioning</span>
      <p>Functional group counts concatenated directly into the readout MLP, anchoring predictions
      to interpretable thermodynamic baselines.</p>
    </div>
    <div class="feature-item">
      <span class="feature-tag">Conformal UQ</span>
      <p>Adaptive prediction intervals via embedding-norm difficulty estimation, providing 95%
      marginal coverage guarantees.</p>
    </div>
  </div>

  <h3 class="about-sub">Data &amp; Training</h3>
  <p class="about-body">
    All models trained on DIPPR 801 data with 5-fold cross-validation. Conformal calibration
    performed on a held-out calibration split (never seen during training). The model is intended
    for organic compounds within the DIPPR chemical space.
  </p>

  <h3 class="about-sub">Models</h3>
  <table class="about-table">
    <thead>
      <tr>
        <th>Property</th><th>Full Name</th><th>Unit</th>
        <th>Hidden Dim</th><th>Layers</th><th>Status</th>
      </tr>
    </thead>
    <tbody>{status_rows}</tbody>
  </table>
</div>""", unsafe_allow_html=True)

if name == 'main':main()
