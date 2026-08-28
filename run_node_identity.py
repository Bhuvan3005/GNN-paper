# -*- coding: utf-8 -*-
"""
run_node_identity.py
====================
Does the feature-node graph matter once nodes are IDENTIFIABLE?

Diagnosis
---------
With 1-D node features, GCNConv(1, H) computes

    h_i = W * ( sum_j alpha_ij x_j ) + b ,   W in R^{H x 1}

i.e. every node embedding is the SAME vector W scaled by a scalar. All 13
node embeddings are collinear -> effective rank exactly 1 (confirmed
empirically: table_representation_rank.csv, conv1 = 1.00 +/- 0.00).

Two consequences:
  (1) the network cannot distinguish `age` from `chol` -- weights are
      shared and aggregation is permutation-equivariant, so feature
      identity is discarded;
  (2) message passing can only average scalars, so "high cholesterol FOR
      THIS AGE" is not representable.

Under these conditions the topology cannot contribute much, and the
observed null result is a statement about 1-D node features rather than
about feature-node graphs.

Fix under test
--------------
Give each node a learnable identity embedding e_i in R^d, so the node
input becomes [value_i ; e_i]. Node embeddings may then span up to
min(13, 1+d) dimensions and message passing mixes *identified* features.

Design (2 x 3 factorial, 3 seeds x 5 folds = 15 runs per cell)
--------------------------------------------------------------
  node encoding : scalar (current)  |  scalar + identity (d=8)
  topology      : no graph | corr+MST | fully connected

The quantity of interest is the GRAPH GAP

    Delta = AUC(corr+MST) - AUC(no graph)

measured under each encoding, with a paired Wilcoxon test over the 15
matched (seed, fold) pairs. Hypothesis: the gap is ~0 for scalar nodes
and materially positive once nodes are identifiable.

Nothing here changes the graph construction, tau, the CV protocol, or
the evaluation. Only the node input representation changes.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from scipy.stats import wilcoxon

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from pipeline import (
    FEATURES, N_FEATURES, NUMERIC_COLS, TAU, DEVICE,
    load_data, set_seed, TrainConfig, build_edge_index,
    train_model, predict_probs, best_threshold, metrics_from, _aggregate,
)

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123]
N_SPLITS = 5
ID_DIM = 8
TOPOLOGIES = ["none", "corr_mst", "fully_connected"]
TOPO_LABEL = {"none": "No graph", "corr_mst": "Corr+MST",
              "fully_connected": "Fully connected"}


# ----------------------------------------------------------------------
# Graphs carrying an explicit node index
# ----------------------------------------------------------------------
def make_graphs_id(X_df, y_ser, edge_index):
    # read the pipeline globals dynamically so the reduced 8-feature
    # external setting (which rebinds P.FEATURES) works unchanged
    import pipeline as _P
    graphs = []
    Xv = X_df[_P.FEATURES].values.astype(np.float32)
    yv = y_ser.values.astype(np.float32)
    nid = torch.arange(_P.N_FEATURES, dtype=torch.long)
    for i in range(len(X_df)):
        graphs.append(Data(
            x=torch.tensor(Xv[i], dtype=torch.float).view(-1, 1),
            edge_index=edge_index,
            y=torch.tensor([yv[i]], dtype=torch.float),
            node_id=nid.clone(),          # concatenated correctly by PyG
        ))
    return graphs


# ----------------------------------------------------------------------
# Model: identical to the paper's GCN except for the node input
# ----------------------------------------------------------------------
class GCN_ID(nn.Module):
    """Two-layer GCN. id_dim=0 reproduces the paper's scalar-node model."""

    def __init__(self, hidden=32, dropout=0.30, id_dim=0, n_nodes=None,
                 operator="gcn"):
        super().__init__()
        import pipeline as _P
        n_nodes = n_nodes if n_nodes is not None else _P.N_FEATURES
        self.id_dim = id_dim
        self.operator = operator
        in_ch = 1 + id_dim
        if id_dim > 0:
            self.emb = nn.Embedding(n_nodes, id_dim)
            nn.init.normal_(self.emb.weight, std=0.1)
        # GCNConv averages self and neighbours together (symmetric
        # normalisation); SAGEConv keeps a separate self-transform, so node
        # identity survives message passing.
        Conv = GCNConv if operator == "gcn" else SAGEConv
        self.conv1 = Conv(in_ch, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = Conv(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.linear = nn.Linear(hidden, 1)
        self.dropout = dropout

    def _node_input(self, data):
        x = data.x
        if self.id_dim > 0:
            x = torch.cat([x, self.emb(data.node_id)], dim=1)
        return x

    def forward(self, data):
        ei = data.edge_index
        batch = data.batch if getattr(data, "batch", None) is not None \
            else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        x = self._node_input(data)
        x = F.relu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, ei)))
        x = global_mean_pool(x, batch)
        return self.linear(x).view(-1)

    @torch.no_grad()
    def node_embeddings(self, data):
        """Layer-1 pre-BN/ReLU node matrix, for effective-rank probing."""
        return self.conv1(self._node_input(data), data.edge_index)


def eff_rank(H):
    s = torch.linalg.svdvals(H.double())
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    return float((s.sum() ** 2) / (s ** 2).sum())


# ----------------------------------------------------------------------
# One CV run of a (topology, id_dim) cell
# ----------------------------------------------------------------------
def run_cell(X, y, topology, id_dim, seed):
    set_seed(seed)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_metrics, ranks = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr_full, y_tr_full = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15,
            stratify=y_tr_full, random_state=seed)

        sc = MinMaxScaler()
        X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()
        X_tr[NUMERIC_COLS] = sc.fit_transform(X_tr[NUMERIC_COLS])
        X_val[NUMERIC_COLS] = sc.transform(X_val[NUMERIC_COLS])
        X_te[NUMERIC_COLS] = sc.transform(X_te[NUMERIC_COLS])

        edge_index, _ = build_edge_index(
            X_tr, topology=topology, threshold=TAU, seed=seed + fold)

        g_tr = make_graphs_id(X_tr, y_tr, edge_index)
        g_val = make_graphs_id(X_val, y_val, edge_index)
        g_te = make_graphs_id(X_te, y_te, edge_index)

        set_seed(seed + fold)
        model = GCN_ID(hidden=32, dropout=0.30, id_dim=id_dim)
        model = train_model(model, g_tr, g_val, TrainConfig())

        vprob, vtrue = predict_probs(model, g_val)
        thr = best_threshold(vprob, vtrue)
        tprob, ttrue = predict_probs(model, g_te)
        fold_metrics.append(metrics_from(ttrue, (tprob >= thr).astype(int), tprob))

        model.eval()
        ranks.append(np.mean([
            eff_rank(model.node_embeddings(g.to(DEVICE))) for g in g_te[:20]]))

    return fold_metrics, float(np.mean(ranks))


def main():
    X, y = load_data()
    print("2x3 factorial: node encoding x topology "
          f"({len(SEEDS)} seeds x {N_SPLITS} folds = {len(SEEDS)*N_SPLITS} runs/cell)")
    print("=" * 84)

    cells = {}     # (id_dim, topology) -> list of fold metric dicts
    rank_of = {}

    for id_dim in (0, ID_DIM):
        tag = "scalar" if id_dim == 0 else f"identity(d={id_dim})"
        for topo in TOPOLOGIES:
            folds, rk = [], []
            for seed in SEEDS:
                f, r = run_cell(X, y, topo, id_dim, seed)
                folds.extend(f)
                rk.append(r)
            cells[(id_dim, topo)] = folds
            rank_of[(id_dim, topo)] = float(np.mean(rk))
            s = _aggregate(folds)
            print(f"{tag:<16} {TOPO_LABEL[topo]:<17} "
                  f"AUC={s['ROC-AUC'][0]:.4f}  F1={s['F1'][0]:.4f}  "
                  f"MCC={s['MCC'][0]:.4f}  rank(conv1)={rank_of[(id_dim, topo)]:.2f}")

    # ---------------- main table ----------------
    rows = []
    for id_dim in (0, ID_DIM):
        tag = "Scalar node (current)" if id_dim == 0 else f"+ Identity embedding (d={ID_DIM})"
        for topo in TOPOLOGIES:
            s = _aggregate(cells[(id_dim, topo)])
            rows.append({
                "Node encoding": tag,
                "Topology": TOPO_LABEL[topo],
                "Effective rank (conv1)": round(rank_of[(id_dim, topo)], 2),
                "ROC-AUC": f"{s['ROC-AUC'][0]:.4f} ± {s['ROC-AUC'][1]:.4f}",
                "F1": f"{s['F1'][0]:.4f} ± {s['F1'][1]:.4f}",
                "MCC": f"{s['MCC'][0]:.4f} ± {s['MCC'][1]:.4f}",
                "Accuracy": f"{s['Accuracy'][0]:.4f} ± {s['Accuracy'][1]:.4f}",
            })
    df = pd.DataFrame(rows)
    df.to_csv("results/table_node_identity.csv", index=False)

    # ---------------- graph gap, paired over (seed, fold) ----------------
    gap_rows = []
    for id_dim in (0, ID_DIM):
        tag = "Scalar node (current)" if id_dim == 0 else f"+ Identity embedding (d={ID_DIM})"
        for contrast, other in (("Corr+MST − No graph", "none"),
                                ("Corr+MST − Fully connected", "fully_connected")):
            a = np.array([m["ROC-AUC"] for m in cells[(id_dim, "corr_mst")]])
            b = np.array([m["ROC-AUC"] for m in cells[(id_dim, other)]])
            try:
                _, p = wilcoxon(a, b)
            except ValueError:
                p = float("nan")
            gap_rows.append({
                "Node encoding": tag,
                "Contrast": contrast,
                "ΔROC-AUC": round(float(a.mean() - b.mean()), 4),
                "Wilcoxon p (15 paired folds)": round(float(p), 4),
                "Wins / 15": int((a > b).sum()),
            })
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv("results/table_node_identity_gap.csv", index=False)

    print("\n" + "=" * 84)
    print("FACTORIAL RESULT")
    print(df.to_string(index=False))
    print("\nGRAPH GAP (does topology matter under each encoding?)")
    print(gaps.to_string(index=False))
    print("\nSaved: results/table_node_identity.csv, results/table_node_identity_gap.csv")


if __name__ == "__main__":
    main()
