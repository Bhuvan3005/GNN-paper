# -*- coding: utf-8 -*-
"""
run_topology_significance.py
============================
Does message passing over scalar feature nodes actually help?

The ablation table reports topology variants as point estimates, but the
central claim of the feature-node formulation---that propagating over
clinically meaningful feature relations is what buys the performance---was
never tested for significance. This script does that directly.

All four topologies are run over the identical (seed x fold) partitions, so
fold metrics are PAIRED. Wilcoxon signed-rank, Holm-corrected over the
family of nine tests (3 comparisons x 3 metrics).

  Corr+MST (Ours)   vs  No graph          -> does propagation help at all?
  Corr+MST (Ours)   vs  Fully connected   -> must the topology be selective?
  Corr+MST (Ours)   vs  Random graph      -> must the edges be the correlated ones?

It also records the stable feature neighbourhoods (edges present in all 5
folds), which are what any mechanistic claim about "context" must rest on.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from pipeline import (
    load_data, GCNConfig, TrainConfig, run_gcn_cv, build_edge_index,
    FEATURES, NUMERIC_COLS, N_FEATURES, TAU, _aggregate,
)

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123]
N_SPLITS = 5
KEYS = ["Accuracy", "F1", "ROC-AUC", "MCC"]

CONFIGS = {
    "Corr+MST (Ours)": "corr_mst",
    "No graph": "none",
    "Fully connected": "fully_connected",
    "Random graph": "random",
}

COMPARISONS = [
    ("Corr+MST (Ours)", "No graph", "Does propagation help at all?"),
    ("Corr+MST (Ours)", "Fully connected", "Must the topology be selective?"),
    ("Corr+MST (Ours)", "Random graph", "Must the edges be the correlated ones?"),
]


def stable_neighbourhoods(seed=42):
    """Neighbours present in ALL folds, i.e. not an artefact of one split."""
    X, y = load_data()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    nb = defaultdict(Counter)
    for tr_idx, _ in skf.split(X, y):
        X_tr, _, _, _ = train_test_split(
            X.iloc[tr_idx], y.iloc[tr_idx], test_size=0.15,
            stratify=y.iloc[tr_idx], random_state=seed)
        X_tr = X_tr.copy()
        X_tr[NUMERIC_COLS] = MinMaxScaler().fit_transform(X_tr[NUMERIC_COLS])
        ei, _ = build_edge_index(X_tr, topology="corr_mst", threshold=TAU)
        G = nx.Graph()
        G.add_nodes_from(range(N_FEATURES))
        G.add_edges_from(ei.t().tolist())
        for i in range(N_FEATURES):
            for n in G.neighbors(i):
                nb[FEATURES[i]][FEATURES[n]] += 1
    rows = []
    for f in FEATURES:
        stable = [k for k, v in nb[f].most_common() if v == N_SPLITS]
        rows.append({
            "Feature": f,
            "Stable degree": len(stable),
            "Neighbours in all folds": ", ".join(stable) if stable else "---",
        })
    return pd.DataFrame(rows)


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

    print("=" * 78)
    print("STABLE FEATURE NEIGHBOURHOODS (present in all 5 folds)")
    print("=" * 78)
    nb_df = stable_neighbourhoods()
    nb_df.to_csv("results/table_feature_neighbourhoods.csv", index=False)
    print(nb_df.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"TOPOLOGY PERFORMANCE ({len(SEEDS)} seeds × {N_SPLITS} folds)")
    print("=" * 78)
    store, rows = {}, []
    for label, topo in CONFIGS.items():
        folds = fold_metrics(topo)
        store[label] = folds
        s = _aggregate(folds)
        row = {"Topology": label}
        row.update({k: f"{s[k][0]:.4f} ± {s[k][1]:.4f}" for k in KEYS})
        rows.append(row)
        print(f"{label:<20} F1={s['F1'][0]:.4f}  AUC={s['ROC-AUC'][0]:.4f}  "
              f"MCC={s['MCC'][0]:.4f}")
    pd.DataFrame(rows).to_csv("results/table_topology_significance_perf.csv",
                              index=False)

    print("\n" + "=" * 78)
    print("PAIRED SIGNIFICANCE (Wilcoxon, 15 matched folds, Holm-corrected)")
    print("=" * 78)
    stat_rows = []
    for a, b, q in COMPARISONS:
        for metric in ["F1", "ROC-AUC", "MCC"]:
            va = np.array([m[metric] for m in store[a]])
            vb = np.array([m[metric] for m in store[b]])
            try:
                _, p = wilcoxon(va, vb)
            except ValueError:
                p = 1.0
            stat_rows.append({
                "Comparison": f"{a} vs {b}",
                "Question": q,
                "Metric": metric,
                "Ours": round(float(va.mean()), 4),
                "Other": round(float(vb.mean()), 4),
                "Δ (Ours−Other)": round(float(va.mean() - vb.mean()), 4),
                "Wilcoxon p": round(float(p), 4),
            })
    stat_df = pd.DataFrame(stat_rows)
    rej, padj, _, _ = multipletests(stat_df["Wilcoxon p"].values,
                                    alpha=0.05, method="holm")
    stat_df["Holm p"] = np.round(padj, 4)
    stat_df["Significant (Holm)"] = np.where(rej, "yes", "no")
    stat_df.to_csv("results/table_topology_significance.csv", index=False)
    print(stat_df.drop(columns=["Question"]).to_string(index=False))

    print("\nSaved: results/table_feature_neighbourhoods.csv, "
          "table_topology_significance_perf.csv, table_topology_significance.csv")


if __name__ == "__main__":
    main()
