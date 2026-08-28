# -*- coding: utf-8 -*-
"""
run_knn_topology.py
===================
Graph-construction study: adaptive per-node sparsity (k-NN on |Pearson|)
versus the global correlation cut-off tau.

Scientific question
-------------------
The full model keeps an edge when |r| >= tau, a single GLOBAL cut-off. A
reviewer will reasonably ask why a per-node rule was not used instead: k-NN
links every feature to its k strongest partners, guaranteeing a MINIMUM
degree of k so that weakly-correlated features are never left near-isolated
by the cut-off. This script answers that directly.

It also quantifies WHEN the MST augmentation actually fires, which explains
the near-zero MST ablation delta reported in the main ablation table.

All configurations share the identical (seed x fold) partitions, so fold
metrics are PAIRED and compared with the Wilcoxon signed-rank test, matching
the convention used in run_learnable_stats.py.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from pipeline import (
    load_data, GCNConfig, TrainConfig, run_gcn_cv, build_edge_index,
    TAU, NUMERIC_COLS, N_FEATURES, _aggregate,
)

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123]
N_SPLITS = 5
KEYS = ["Accuracy", "F1", "ROC-AUC", "MCC", "Recall", "Specificity"]

# label -> topology string
CONFIGS = {
    "Corr+MST, τ=0.15 (Ours)": "corr_mst",
    "Corr only, τ=0.15": "corr_only",
    "k-NN, k=2": "knn2",
    "k-NN, k=3": "knn3",
    "k-NN, k=4": "knn4",
}

COMPARISONS = [
    ("Corr+MST, τ=0.15 (Ours)", "k-NN, k=2"),
    ("Corr+MST, τ=0.15 (Ours)", "k-NN, k=3"),
    ("Corr+MST, τ=0.15 (Ours)", "k-NN, k=4"),
]

TAU_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


# ----------------------------------------------------------------------
# Structural characterisation
# ----------------------------------------------------------------------
def fold_train_frames(seed=42):
    """Yield the per-fold inner-train frames used for graph construction."""
    X, y = load_data()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr_idx, _ in skf.split(X, y):
        X_tr, _, _, _ = train_test_split(
            X.iloc[tr_idx], y.iloc[tr_idx], test_size=0.15,
            stratify=y.iloc[tr_idx], random_state=seed)
        X_tr = X_tr.copy()
        X_tr[NUMERIC_COLS] = MinMaxScaler().fit_transform(X_tr[NUMERIC_COLS])
        yield X_tr


def graph_shape(topology, threshold=TAU):
    """Mean edges / degree / connectivity of a topology across folds."""
    edges, mindeg, conn = [], [], []
    for X_tr in fold_train_frames():
        ei, _ = build_edge_index(X_tr, topology=topology, threshold=threshold)
        G = nx.Graph()
        G.add_nodes_from(range(N_FEATURES))
        G.add_edges_from(ei.t().tolist())
        edges.append(G.number_of_edges())
        mindeg.append(min(d for _, d in G.degree()))
        conn.append(nx.is_connected(G))
    return {
        "Edges": f"{np.mean(edges):.1f} ± {np.std(edges):.1f}",
        "Min degree": int(np.min(mindeg)),
        "Connected folds": f"{sum(conn)}/{len(conn)}",
    }


def mst_activation_table():
    """How many bridges the MST actually adds, as a function of tau."""
    rows = []
    for tau in TAU_GRID:
        n_only, n_mst, conn = [], [], []
        for X_tr in fold_train_frames():
            e1, _ = build_edge_index(X_tr, topology="corr_only", threshold=tau)
            e2, _ = build_edge_index(X_tr, topology="corr_mst", threshold=tau)
            G = nx.Graph()
            G.add_nodes_from(range(N_FEATURES))
            G.add_edges_from(e1.t().tolist())
            n_only.append(e1.shape[1] // 2)
            n_mst.append(e2.shape[1] // 2)
            conn.append(nx.is_connected(G))
        rows.append({
            "τ": tau,
            "Threshold-graph edges": round(float(np.mean(n_only)), 1),
            "After MST": round(float(np.mean(n_mst)), 1),
            "MST bridges added": round(float(np.mean(n_mst) - np.mean(n_only)), 1),
            "Connected before MST": f"{sum(conn)}/{len(conn)}",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Predictive comparison
# ----------------------------------------------------------------------
def fold_metrics(topology):
    out = []
    X, y = DATA
    for seed in SEEDS:
        _, folds = run_gcn_cv(
            X, y, gcfg=GCNConfig(), tcfg=TrainConfig(),
            topology=topology, threshold=TAU,
            n_splits=N_SPLITS, seed=seed, tune_threshold=True)
        out.extend(folds)
    return out


def main():
    global DATA
    DATA = load_data()

    # ---- 1. MST activation analysis ----
    print("=" * 78)
    print("WHEN DOES THE MST ACTUALLY FIRE?")
    print("=" * 78)
    mst_df = mst_activation_table()
    mst_df.to_csv("results/table_mst_activation.csv", index=False)
    print(mst_df.to_string(index=False))

    # ---- 2. Structural characterisation ----
    print("\n" + "=" * 78)
    print("TOPOLOGY STRUCTURE (across 5 folds)")
    print("=" * 78)
    struct_rows = []
    for label, topo in CONFIGS.items():
        struct_rows.append({"Construction": label, **graph_shape(topo)})
    struct_df = pd.DataFrame(struct_rows)
    struct_df.to_csv("results/table_knn_structure.csv", index=False)
    print(struct_df.to_string(index=False))

    # ---- 3. Predictive comparison ----
    print("\n" + "=" * 78)
    print(f"PREDICTIVE COMPARISON ({len(SEEDS)} seeds × {N_SPLITS} folds)")
    print("=" * 78)
    store, rows = {}, []
    for label, topo in CONFIGS.items():
        folds = fold_metrics(topo)
        store[label] = folds
        s = _aggregate(folds)
        row = {"Construction": label}
        row.update({k: f"{s[k][0]:.4f} ± {s[k][1]:.4f}" for k in KEYS})
        rows.append(row)
        print(f"{label:<26} F1={s['F1'][0]:.4f}  AUC={s['ROC-AUC'][0]:.4f}  "
              f"MCC={s['MCC'][0]:.4f}")

    knn_df = pd.DataFrame(rows)
    knn_df.to_csv("results/table_knn_topology.csv", index=False)

    # ---- 4. Paired significance ----
    print("\n" + "=" * 78)
    print("PAIRED SIGNIFICANCE (Wilcoxon signed-rank, 15 paired folds)")
    print("=" * 78)
    stat_rows = []
    for a, b in COMPARISONS:
        for metric in ["F1", "ROC-AUC", "MCC"]:
            va = np.array([m[metric] for m in store[a]])
            vb = np.array([m[metric] for m in store[b]])
            try:
                _, p = wilcoxon(va, vb)
            except ValueError:
                p = np.nan
            stat_rows.append({
                "Comparison": f"{a} vs {b}",
                "Metric": metric,
                "Mean A": round(float(va.mean()), 4),
                "Mean B": round(float(vb.mean()), 4),
                "Δ (A−B)": round(float(va.mean() - vb.mean()), 4),
                "Wilcoxon p": round(float(p), 4),
                "Significant (α=0.05)": "yes" if p < 0.05 else "no",
            })
    stat_df = pd.DataFrame(stat_rows)
    # 9 paired tests are run over the same folds, so raw p-values are
    # optimistic. Holm-Bonferroni controls the family-wise error rate.
    from statsmodels.stats.multitest import multipletests
    reject, p_adj, _, _ = multipletests(
        stat_df["Wilcoxon p"].values, alpha=0.05, method="holm")
    stat_df["Holm p"] = np.round(p_adj, 4)
    stat_df["Significant (Holm)"] = np.where(reject, "yes", "no")
    stat_df = stat_df.drop(columns=["Significant (α=0.05)"])
    stat_df.to_csv("results/table_knn_stats.csv", index=False)
    print(stat_df.to_string(index=False))

    print("\nSaved: results/table_mst_activation.csv, table_knn_structure.csv, "
          "table_knn_topology.csv, table_knn_stats.csv")


if __name__ == "__main__":
    main()
