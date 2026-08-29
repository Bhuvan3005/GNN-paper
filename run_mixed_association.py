# -*- coding: utf-8 -*-
"""
run_mixed_association.py
=========================
Robustness check: does the graph topology change materially, and does
predictive performance change materially, if edges are built from a
mixed-type association measure instead of raw Pearson correlation?

Scientific question
--------------------
Pearson correlation is computed over all 13 features including four
integer-coded nominal/ordinal columns (cp, restecg, slope, thal). Pearson
implicitly treats those integer codes as evenly spaced and ordered, which
is not a valid assumption for a nominal code (e.g. cp = 0,1,2,3 has no
inherent ordering). This script replaces the association measure with one
that is valid for mixed continuous/categorical data -- Cramer's V
(categorical-categorical), the correlation ratio eta (continuous-
categorical), and Pearson |r| (continuous-continuous), see
pipeline.compute_mixed_association -- while keeping every other element of
the methodology identical: same tau=0.15 threshold rule, same MST
connectivity augmentation, same fold-specific (train-only) computation,
same 5-fold x 3-seed protocol, same GCN architecture.

If results are statistically indistinguishable from the Pearson-based
graph, that is direct evidence the topology is not an artefact of treating
nominal codes as ordered scalars.
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
from statsmodels.stats.multitest import multipletests

from pipeline import (
    load_data, GCNConfig, TrainConfig, run_gcn_cv, build_edge_index,
    TAU, NUMERIC_COLS, N_FEATURES, _aggregate,
)

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123]
N_SPLITS = 5
KEYS = ["Accuracy", "F1", "ROC-AUC", "MCC", "Recall", "Specificity"]

CONFIGS = {
    "Corr+MST, \u03c4=0.15 (Ours)": "corr_mst",
    "Mixed-assoc+MST, \u03c4=0.15": "mixed_mst",
    "Mixed-assoc only, \u03c4=0.15": "mixed_only",
}

COMPARISONS = [
    ("Corr+MST, \u03c4=0.15 (Ours)", "Mixed-assoc+MST, \u03c4=0.15"),
]


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
        "Edges": f"{np.mean(edges):.1f} \u00b1 {np.std(edges):.1f}",
        "Min degree": int(np.min(mindeg)),
        "Connected folds": f"{sum(conn)}/{len(conn)}",
    }


def edge_overlap(topo_a, topo_b, threshold=TAU):
    """Jaccard overlap between the two topologies' edge sets, averaged
    across folds -- quantifies how much the graph itself changes."""
    jac = []
    for X_tr in fold_train_frames():
        ea, _ = build_edge_index(X_tr, topology=topo_a, threshold=threshold)
        eb, _ = build_edge_index(X_tr, topology=topo_b, threshold=threshold)
        Sa = set(map(tuple, map(sorted, ea.t().tolist())))
        Sb = set(map(tuple, map(sorted, eb.t().tolist())))
        union = len(Sa | Sb)
        jac.append(len(Sa & Sb) / union if union else 1.0)
    return float(np.mean(jac)), float(np.std(jac))


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

    # ---- 1. Structural characterisation ----
    print("=" * 78)
    print("TOPOLOGY STRUCTURE (across 5 folds)")
    print("=" * 78)
    struct_rows = []
    for label, topo in CONFIGS.items():
        struct_rows.append({"Construction": label, **graph_shape(topo)})
    struct_df = pd.DataFrame(struct_rows)
    struct_df.to_csv("results/table_mixed_association_structure.csv", index=False)
    print(struct_df.to_string(index=False))

    mean_jac, std_jac = edge_overlap("corr_mst", "mixed_mst")
    print(f"\nEdge-set Jaccard overlap (Corr+MST vs Mixed-assoc+MST): "
          f"{mean_jac:.3f} \u00b1 {std_jac:.3f}")
    pd.DataFrame([{"Comparison": "Corr+MST vs Mixed-assoc+MST",
                    "Jaccard overlap (mean)": round(mean_jac, 3),
                    "Jaccard overlap (std)": round(std_jac, 3)}]).to_csv(
        "results/table_mixed_association_overlap.csv", index=False)

    # ---- 2. Predictive comparison ----
    print("\n" + "=" * 78)
    print(f"PREDICTIVE COMPARISON ({len(SEEDS)} seeds \u00d7 {N_SPLITS} folds)")
    print("=" * 78)
    store, rows = {}, []
    for label, topo in CONFIGS.items():
        folds = fold_metrics(topo)
        store[label] = folds
        s = _aggregate(folds)
        row = {"Construction": label}
        row.update({k: f"{s[k][0]:.4f} \u00b1 {s[k][1]:.4f}" for k in KEYS})
        rows.append(row)
        print(f"{label:<26} F1={s['F1'][0]:.4f}  AUC={s['ROC-AUC'][0]:.4f}  "
              f"MCC={s['MCC'][0]:.4f}")

    assoc_df = pd.DataFrame(rows)
    assoc_df.to_csv("results/table_mixed_association.csv", index=False)

    # ---- 3. Paired significance ----
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
                "\u0394 (A\u2212B)": round(float(va.mean() - vb.mean()), 4),
                "Wilcoxon p": round(float(p), 4),
            })
    stat_df = pd.DataFrame(stat_rows)
    # 3 paired tests over the same folds -> Holm-Bonferroni family-wise
    # error control, matching the convention in run_knn_topology.py.
    reject, p_adj, _, _ = multipletests(
        stat_df["Wilcoxon p"].values, alpha=0.05, method="holm")
    stat_df["Holm p"] = np.round(p_adj, 4)
    stat_df["Significant (Holm)"] = np.where(reject, "yes", "no")
    stat_df.to_csv("results/table_mixed_association_stats.csv", index=False)
    print(stat_df.to_string(index=False))

    print("\nSaved: results/table_mixed_association_structure.csv, "
          "table_mixed_association_overlap.csv, table_mixed_association.csv, "
          "table_mixed_association_stats.csv")


if __name__ == "__main__":
    main()
