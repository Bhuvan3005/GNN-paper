# -*- coding: utf-8 -*-
"""
run_graphstats.py
=================
Task 5 (partial): graph-statistics table and hyperparameter table.

Graph statistics are reported as mean +/- std across the 5 CV folds
(each fold's correlation+MST graph is built on that fold's inner-train
data), plus a canonical full-data reference graph. This documents the
topology reproducibly.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from pipeline import (
    FEATURES, N_FEATURES, NUMERIC_COLS, CATEGORICAL_COLS, TAU,
    load_data, _corr_mst_edge_set,
)

os.makedirs("results", exist_ok=True)
SEED = 42


def graph_stats(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    # Fiedler value (algebraic connectivity) on unweighted graph
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eig = np.sort(np.linalg.eigvalsh(L))
    fiedler = float(eig[1]) if len(eig) > 1 else 0.0
    connected = nx.is_connected(G)
    return {
        "Nodes": n,
        "Edges": e,
        "Density": round(nx.density(G), 4),
        "Avg Degree": round(float(np.mean(degrees)), 4),
        "Avg Path Length": round(nx.average_shortest_path_length(G), 4) if connected else np.nan,
        "Diameter": nx.diameter(G) if connected else np.nan,
        "Clustering Coeff": round(nx.average_clustering(G), 4),
        "Fiedler Value": round(fiedler, 4),
    }


def build_fold_graph(X_tr):
    corr = X_tr[FEATURES].corr().values
    G = _corr_mst_edge_set(corr, TAU)
    return nx.relabel_nodes(G, {i: FEATURES[i] for i in range(N_FEATURES)})


def main():
    X, y = load_data()

    # ---- per-fold graph statistics ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    per_fold = []
    for fold, (tr_idx, _) in enumerate(skf.split(X, y)):
        X_tr_full, y_tr_full = X.iloc[tr_idx], y.iloc[tr_idx]
        X_tr, _, _, _ = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15,
            stratify=y_tr_full, random_state=SEED)
        X_tr = X_tr.copy()
        sc = MinMaxScaler()
        X_tr[NUMERIC_COLS] = sc.fit_transform(X_tr[NUMERIC_COLS])
        per_fold.append(graph_stats(build_fold_graph(X_tr)))

    keys = list(per_fold[0].keys())
    rows = []
    for k in keys:
        vals = np.array([f[k] for f in per_fold], dtype=float)
        if k in ("Nodes",):
            rows.append({"Statistic": k, "Value (mean ± std)": f"{int(vals[0])}"})
        else:
            rows.append({"Statistic": k,
                         "Value (mean ± std)": f"{np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}"})
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv("results/table_graph_statistics.csv", index=False)
    print("GRAPH STATISTICS (mean ± std across 5 folds)")
    print(stats_df.to_string(index=False))

    # ---- canonical full-data reference graph ----
    Xc = X.copy()
    sc = MinMaxScaler()
    Xc[NUMERIC_COLS] = sc.fit_transform(Xc[NUMERIC_COLS])
    G_full = build_fold_graph(Xc)
    full_stats = graph_stats(G_full)
    pd.DataFrame([{"Statistic": k, "Value": v} for k, v in full_stats.items()]) \
        .to_csv("results/table_graph_statistics_fulldata.csv", index=False)
    print("\nCanonical full-data graph:", full_stats)

    # ---- hyperparameter table ----
    hp = [
        ("Node feature dim", "1 (scalar clinical value)"),
        ("# Nodes", f"{N_FEATURES} (clinical features)"),
        ("Continuous features (scaled)", str(len(NUMERIC_COLS))),
        ("Categorical features (raw)", str(len(CATEGORICAL_COLS))),
        ("Correlation metric", "Pearson"),
        ("Correlation threshold τ", "0.15"),
        ("Graph augmentation", "Minimum Spanning Tree ($d = 1 - |r|$)"),
        ("GCN layers", "2"),
        ("Hidden dimension", "32"),
        ("Normalization", "BatchNorm1d"),
        ("Activation", "ReLU"),
        ("Dropout", "0.30"),
        ("Readout", "Global mean pooling"),
        ("Output head", "Linear(32→1) + sigmoid"),
        ("Loss", "BCEWithLogits (balanced)"),
        ("Optimizer", "Adam"),
        ("Learning rate", "1e-3"),
        ("Weight decay", "1e-4"),
        ("Gradient clipping", "max-norm 1.0"),
        ("Batch size", "32"),
        ("Max epochs", "150"),
        ("Early stopping patience", "25 (val loss)"),
        ("Decision threshold", "tuned on inner-validation (F1)"),
        ("Cross-validation", "5-fold Stratified"),
        ("Inner val split", "15% of train fold (stratified)"),
        ("Scaling", "MinMax (continuous cols), fit in-fold"),
        ("Framework", "PyTorch 2.x + PyG 2.7, CPU"),
        ("Seeds", "42, 7, 123"),
    ]
    hp_df = pd.DataFrame(hp, columns=["Hyperparameter", "Value"])
    hp_df.to_csv("results/table_hyperparameters.csv", index=False)
    print("\nSaved: results/table_graph_statistics.csv, "
          "table_graph_statistics_fulldata.csv, table_hyperparameters.csv")


if __name__ == "__main__":
    main()
