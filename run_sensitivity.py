# -*- coding: utf-8 -*-
"""
run_sensitivity.py
==================
Task 5 (partial): sensitivity analysis + threshold-selection table.

Sweeps the Pearson correlation threshold tau over a grid and reports the
full-model CV performance at each value (mean +/- std pooled over
seed x fold), justifying the tau = 0.15 choice. Also records the
resulting mean edge count so the accuracy/sparsity trade-off is visible.
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

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from pipeline import (
    FEATURES, N_FEATURES, NUMERIC_COLS, load_data, GCNConfig, TrainConfig,
    run_gcn_cv, _aggregate, _corr_mst_edge_set,
)

os.makedirs("results", exist_ok=True)
SEEDS = [42, 7, 123]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]
KEYS = ["Accuracy", "F1", "ROC-AUC", "MCC"]


def mean_edges_at(X, y, tau, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    counts = []
    for tr_idx, _ in skf.split(X, y):
        X_tr, _, _, _ = train_test_split(
            X.iloc[tr_idx], y.iloc[tr_idx], test_size=0.15,
            stratify=y.iloc[tr_idx], random_state=seed)
        X_tr = X_tr.copy()
        sc = MinMaxScaler(); X_tr[NUMERIC_COLS] = sc.fit_transform(X_tr[NUMERIC_COLS])
        corr = X_tr[FEATURES].corr().values
        counts.append(_corr_mst_edge_set(corr, tau).number_of_edges())
    return float(np.mean(counts))


def main():
    X, y = load_data()
    rows = []
    for tau in THRESHOLDS:
        all_folds = []
        for seed in SEEDS:
            _, folds = run_gcn_cv(
                X, y, GCNConfig(), TrainConfig(), topology="corr_mst",
                threshold=tau, n_splits=5, seed=seed, tune_threshold=True)
            all_folds.extend(folds)
        s = _aggregate(all_folds)
        row = {"tau": tau, "Mean edges": round(mean_edges_at(X, y, tau), 1)}
        row.update({k: f"{s[k][0]:.4f} +/- {s[k][1]:.4f}" for k in KEYS})
        rows.append(row)
        print(f"tau={tau:.2f}  edges={row['Mean edges']:>5}  "
              f"F1={s['F1'][0]:.4f}  AUC={s['ROC-AUC'][0]:.4f}  MCC={s['MCC'][0]:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv("results/table_threshold_sensitivity.csv", index=False)
    print("\nSaved: results/table_threshold_sensitivity.csv")


if __name__ == "__main__":
    main()
