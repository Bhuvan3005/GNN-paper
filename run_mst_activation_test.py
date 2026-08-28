# -*- coding: utf-8 -*-
"""
run_mst_activation_test.py
==========================
Does the MST augmentation restore CONNECTIVITY or improve ACCURACY?

The main ablation compares Corr vs Corr+MST at tau = 0.15, where the
thresholded graph is already connected in every fold, so the MST adds no
edges and the two conditions are the same graph. That comparison therefore
cannot test the MST at all.

This script runs the comparison at a threshold where the MST is ACTIVE.
At tau = 0.20 the thresholded graph is disconnected in 5/5 folds and the MST
supplies ~1.8 bridges per fold. A 2 x 2 design isolates the effect:

                       tau = 0.15 (MST inert)   tau = 0.20 (MST active)
    Corr only                A                          C
    Corr + MST               B                          D

  A vs B  -> null by construction (identical graphs); a sanity check.
  C vs D  -> the real test of what the MST contributes.

Prediction if the MST is a robustness mechanism rather than a performance
booster: D restores connectivity relative to C, but D ~ C in accuracy.

All four configurations share the identical (seed x fold) partitions, so
fold metrics are PAIRED (Wilcoxon signed-rank, Holm-corrected).
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
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from pipeline import (
    load_data, GCNConfig, TrainConfig, run_gcn_cv, build_edge_index,
    NUMERIC_COLS, N_FEATURES, _aggregate,
)

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123]
N_SPLITS = 5
KEYS = ["Accuracy", "F1", "ROC-AUC", "MCC", "Recall", "Specificity"]

# label -> (topology, tau)
CONFIGS = {
    "Corr only, τ=0.15":  ("corr_only", 0.15),
    "Corr+MST, τ=0.15":   ("corr_mst",  0.15),
    "Corr only, τ=0.20":  ("corr_only", 0.20),
    "Corr+MST, τ=0.20":   ("corr_mst",  0.20),
}

COMPARISONS = [
    ("Corr only, τ=0.15", "Corr+MST, τ=0.15",
     "MST inert (identical graphs) — sanity check"),
    ("Corr only, τ=0.20", "Corr+MST, τ=0.20",
     "MST active (+1.8 bridges) — the real test"),
]


def connectivity_profile(topology, tau, seed=42):
    """Edges / components / isolated-ish nodes per fold."""
    X, y = load_data()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    edges, comps, conn = [], [], []
    for tr_idx, _ in skf.split(X, y):
        X_tr, _, _, _ = train_test_split(
            X.iloc[tr_idx], y.iloc[tr_idx], test_size=0.15,
            stratify=y.iloc[tr_idx], random_state=seed)
        X_tr = X_tr.copy()
        X_tr[NUMERIC_COLS] = MinMaxScaler().fit_transform(X_tr[NUMERIC_COLS])
        ei, _ = build_edge_index(X_tr, topology=topology, threshold=tau)
        G = nx.Graph()
        G.add_nodes_from(range(N_FEATURES))
        G.add_edges_from(ei.t().tolist())
        edges.append(G.number_of_edges())
        comps.append(nx.number_connected_components(G))
        conn.append(nx.is_connected(G))
    return {
        "Edges": f"{np.mean(edges):.1f} ± {np.std(edges):.1f}",
        "Components": f"{np.mean(comps):.1f}",
        "Connected folds": f"{sum(conn)}/{len(conn)}",
    }


def fold_metrics(topology, tau):
    out = []
    X, y = DATA
    for seed in SEEDS:
        _, folds = run_gcn_cv(
            X, y, gcfg=GCNConfig(), tcfg=TrainConfig(),
            topology=topology, threshold=tau,
            n_splits=N_SPLITS, seed=seed, tune_threshold=True)
        out.extend(folds)
    return out


def main():
    global DATA
    DATA = load_data()

    # ---- 1. Connectivity: what the MST actually changes ----
    print("=" * 78)
    print("CONNECTIVITY PROFILE")
    print("=" * 78)
    conn_rows = []
    for label, (topo, tau) in CONFIGS.items():
        conn_rows.append({"Configuration": label,
                          **connectivity_profile(topo, tau)})
    conn_df = pd.DataFrame(conn_rows)
    conn_df.to_csv("results/table_mst_connectivity.csv", index=False)
    print(conn_df.to_string(index=False))

    # ---- 2. Predictive performance ----
    print("\n" + "=" * 78)
    print(f"PREDICTIVE PERFORMANCE ({len(SEEDS)} seeds × {N_SPLITS} folds)")
    print("=" * 78)
    store, rows = {}, []
    for label, (topo, tau) in CONFIGS.items():
        folds = fold_metrics(topo, tau)
        store[label] = folds
        s = _aggregate(folds)
        row = {"Configuration": label}
        row.update({k: f"{s[k][0]:.4f} ± {s[k][1]:.4f}" for k in KEYS})
        rows.append(row)
        print(f"{label:<22} F1={s['F1'][0]:.4f}  AUC={s['ROC-AUC'][0]:.4f}  "
              f"MCC={s['MCC'][0]:.4f}")
    perf_df = pd.DataFrame(rows)
    perf_df.to_csv("results/table_mst_tau_comparison.csv", index=False)

    # ---- 3. Paired significance, Holm-corrected ----
    print("\n" + "=" * 78)
    print("PAIRED SIGNIFICANCE (Wilcoxon, 15 matched folds, Holm-corrected)")
    print("=" * 78)
    stat_rows = []
    for a, b, note in COMPARISONS:
        for metric in ["F1", "ROC-AUC", "MCC"]:
            va = np.array([m[metric] for m in store[a]])
            vb = np.array([m[metric] for m in store[b]])
            try:
                _, p = wilcoxon(va, vb)
            except ValueError:      # all-zero differences => identical
                p = 1.0
            stat_rows.append({
                "Comparison": f"{a} vs {b}",
                "Regime": note,
                "Metric": metric,
                "Mean A": round(float(va.mean()), 4),
                "Mean B": round(float(vb.mean()), 4),
                "Δ (B−A)": round(float(vb.mean() - va.mean()), 4),
                "Wilcoxon p": round(float(p), 4),
            })
    stat_df = pd.DataFrame(stat_rows)
    reject, p_adj, _, _ = multipletests(
        stat_df["Wilcoxon p"].values, alpha=0.05, method="holm")
    stat_df["Holm p"] = np.round(p_adj, 4)
    stat_df["Significant (Holm)"] = np.where(reject, "yes", "no")
    stat_df.to_csv("results/table_mst_tau_stats.csv", index=False)
    print(stat_df.to_string(index=False))

    print("\nSaved: results/table_mst_connectivity.csv, "
          "table_mst_tau_comparison.csv, table_mst_tau_stats.csv")


if __name__ == "__main__":
    main()
