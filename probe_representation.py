# -*- coding: utf-8 -*-
"""
probe_representation.py
=======================
What does message passing actually build on top of scalar feature nodes?

With one scalar per node, the first GCN layer maps R^1 -> R^32 per node and
then aggregates, so every node embedding is a scaled copy of the same weight
vector: the layer-1 representation is essentially rank-1 in DIRECTION, and
nodes differ only in magnitude. Whether a genuinely multi-directional
representation emerges at layer 2 is an empirical question, and it bears
directly on how strongly we may describe the embeddings as
"context-dependent".

We measure the participation-ratio effective rank
    r_eff = (sum_i sigma_i)^2 / sum_i sigma_i^2
of the 13 x 32 node-embedding matrix, on a TRAINED model, averaged over
test patients. r_eff = 1 means all node embeddings point the same way;
r_eff = k means roughly k independent directions are in use.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pipeline import (
    load_data, set_seed, GCN, GCNConfig, TrainConfig, train_model,
    make_graphs, build_edge_index, NUMERIC_COLS, TAU, DEVICE,
)

SEED = 42


def eff_rank(H):
    s = torch.linalg.svdvals(H.double())
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    return float((s.sum() ** 2) / (s ** 2).sum())


def build(topology="corr_mst"):
    X, y = load_data()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    sc = MinMaxScaler()
    X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()
    X_tr[NUMERIC_COLS] = sc.fit_transform(X_tr[NUMERIC_COLS])
    X_val[NUMERIC_COLS] = sc.transform(X_val[NUMERIC_COLS])
    X_te[NUMERIC_COLS] = sc.transform(X_te[NUMERIC_COLS])
    ei, _ = build_edge_index(X_tr, topology=topology, threshold=TAU)
    g_tr = make_graphs(X_tr, y_tr, ei)
    g_val = make_graphs(X_val, y_val, ei)
    g_te = make_graphs(X_te, y_te, ei)
    set_seed(SEED)
    model = train_model(GCN(GCNConfig()), g_tr, g_val, TrainConfig())
    return model, g_te, ei


@torch.no_grad()
def layer_ranks(model, graphs):
    model.eval().to(DEVICE)
    out = {"conv1 (pre BN/ReLU)": [], "layer 1 (BN+ReLU)": [], "layer 2": []}
    for g in graphs:
        g = g.to(DEVICE)
        h1 = model.convs[0](g.x, g.edge_index)
        h1b = torch.relu(model.bns[0](h1))
        h2 = torch.relu(model.bns[1](model.convs[1](h1b, g.edge_index)))
        out["conv1 (pre BN/ReLU)"].append(eff_rank(h1))
        out["layer 1 (BN+ReLU)"].append(eff_rank(h1b))
        out["layer 2"].append(eff_rank(h2))
    return out


def main():
    rows = []
    for topo, label in [("corr_mst", "Corr+MST (Ours)"), ("none", "No graph")]:
        model, g_te, _ = build(topo)
        r = layer_ranks(model, g_te)
        for stage, vals in r.items():
            rows.append({
                "Topology": label,
                "Stage": stage,
                "Effective rank (mean ± std)":
                    f"{np.mean(vals):.2f} ± {np.std(vals):.2f}",
                "Max possible": 13,
            })
        print(f"{label}: " + "  ".join(
            f"{k}={np.mean(v):.2f}" for k, v in r.items()))

    df = pd.DataFrame(rows)
    df.to_csv("results/table_representation_rank.csv", index=False)
    print()
    print(df.to_string(index=False))
    print("\nSaved: results/table_representation_rank.csv")


if __name__ == "__main__":
    main()
