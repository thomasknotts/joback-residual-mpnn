# JR-MPNN Thermophysical Property Predictor

Streamlit web application for predicting five thermophysical properties using the
**Joback-Residual Message Passing Neural Network (JR-MPNN)**.

---

## The Residual Framework

JR-MPNN is a hybrid model that combines a classical group-contribution method with a
graph neural network. Rather than predicting a property directly, the model learns to
correct the error made by the Joback method:

```
ŷ = y_Joback + r̂(G, c)
```

- **y_Joback** — baseline estimate from the Joback group-contribution method
- **r̂(G, c)** — residual correction learned by the GNN from the molecular graph G
  and conditioning vector c

The conditioning vector **c** = [y_Joback, g₁, …, g₄₁] concatenates the Joback
baseline with the 41 Joback functional group counts. This anchors the network's
readout MLP to the same group-contribution decomposition that underpins the baseline,
letting the GNN focus on what Joback gets wrong rather than learning thermophysical
properties from scratch.

### Why residual learning?

Joback predictions are fast and interpretable but systematically biased for molecules
outside its training distribution. A GNN trained purely end-to-end must learn both
the global thermodynamic trends and the local corrections simultaneously, which
increases data requirements. By targeting the residual, JR-MPNN exploits the Joback
estimate as a strong prior and only needs to capture the structured error — improving
generalisation, especially in low-data regimes.

---

## Architecture

### Message passing — DMPNNLayer

Each layer uses directed message passing with two key features:

**XOR backtracking prevention** — when aggregating messages along an edge (i→j),
the contribution of the reverse edge (j→i) is subtracted from the neighbourhood
sum before it is processed. This prevents a message from immediately "bouncing back"
along the same bond, effectively enforcing a longer information path through the graph.

**Residual connections** — both the edge update and node update add the new
representation to the previous one, stabilising training in deeper networks.

### Graph embedding — triple pooling

After all message-passing layers, the node representations are pooled three ways and
concatenated:

```
h_G = [global_add_pool(x) ‖ global_mean_pool(x) ‖ global_max_pool(x)]
```

This gives a single graph-level vector that captures complementary aspects of the
molecule: total size (sum), average density (mean), and dominant local features (max).

### Readout MLP

The graph embedding h_G is concatenated with the conditioning vector c and passed
through a four-layer MLP with a skip connection. The output is the predicted residual
r̂, which is added to y_Joback to give the final property estimate.

### Uncertainty quantification — conformal prediction

Prediction intervals are constructed using the embedding norm ‖h_G‖ as a proxy for
input difficulty. The interval half-width is:

```
half-width = q × ‖h_G‖
```

where q is a scalar calibration quantile computed on a held-out calibration set
(never seen during training). This provides **95% marginal coverage** guarantees
under the exchangeability assumption.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, follow the official installation guide at
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 2. Place model weights

Copy your `.pth` files into the `models/` folder:

```
models/
  jrmpnn_mp.pth    # Melting point
  jrmpnn_tb.pth    # Normal boiling point
  jrmpnn_tc.pth    # Critical temperature
  jrmpnn_pc.pth    # Critical pressure
  jrmpnn_vc.pth    # Critical volume
```

### 3. (Optional) Update calibration quantiles

Edit `utils/prediction.py` → `CALIBRATION_QUANTILES` with your calibrated q values.

### 4. Run

```bash
streamlit run streamlit_app.py
```

---

## Batch prediction

The Batch tab accepts a CSV file with the following format:

| SMILES | Name |
|--------|------|
| CCO | Ethanol |
| c1ccccc1 | Benzene |

The `Name` column is optional. Maximum 500 molecules per batch. Results can be
downloaded as a CSV from the app.

---

## Project structure

```
for_st/
├── streamlit_app.py        # Streamlit application
├── utils/
│   ├── model_classes.py    # JR-MPNN architecture (DMPNNLayer, JRMPNN)
│   ├── molecular_graph.py  # Atom/bond featurisation
│   └── prediction.py       # Inference utilities, conformal intervals
├── models/                 # .pth weight files (you supply these)
├── requirements.txt
└── README.md
```
