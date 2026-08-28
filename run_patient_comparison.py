# -*- coding: utf-8 -*-
"""
run_patient_comparison.py
=========================
Empirical comparison of the two GNN formulations for clinical tabular data:

    feature-node graph (ours)   vs   patient-similarity graph (conventional)

Both are evaluated under the IDENTICAL 5-fold stratified CV protocol,
repeated over three seeds, with in-fold scaling and inner-validation
threshold tuning.

Stage 1 -- k sweep. The patient-similarity baseline is given a fair
           search over k in {5, 10, 15, 20} so the comparison cannot be
           accused of strawmanning it.
Stage 2 -- head-to-head. The best-k patient model is compared against the
           feature-node model on pooled out-of-fold predictions using
           McNemar's test (thresholded predictions) and DeLong's test
           (correlated ROC curves).
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from pipeline import (
    load_data, GCNConfig, TrainConfig, run_gcn_cv, TAU, _aggregate,
)
from patient_graph import run_patient_gcn_cv
from run_external_stats import delong_roc_test

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123]
K_GRID = [5, 10, 15, 20]
METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MCC", "Specificity"]


def pooled_patient(X, y, k):
    """Pool fold metrics across seeds for one k."""
    folds = []
    for sd in SEEDS:
        _, f = run_patient_gcn_cv(X, y, k=k, n_splits=5, seed=sd)
        folds.extend(f)
    return _aggregate(folds)


def main():
    X, y = load_data()
    print("=" * 78)
    print("FEATURE-NODE  vs  PATIENT-SIMILARITY  GNN")
    print(f"Identical protocol: 5-fold stratified CV x seeds {SEEDS}")
    print("=" * 78)

    # ---------------- Stage 1: k sweep ----------------
    print(f"\n[1] Patient-similarity k sweep over {K_GRID} ...")
    sweep_rows, sweep_summ = [], {}
    for k in K_GRID:
        s = pooled_patient(X, y, k)
        sweep_summ[k] = s
        sweep_rows.append({"k": k,
                           **{m: f"{s[m][0]:.4f} ± {s[m][1]:.4f}" for m in METRIC_KEYS}})
        print(f"    k={k:<3} Acc={s['Accuracy'][0]:.4f}  F1={s['F1'][0]:.4f}  "
              f"AUC={s['ROC-AUC'][0]:.4f}  MCC={s['MCC'][0]:.4f}")

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv("results/table_patient_knn_sweep.csv", index=False)

    best_k = max(K_GRID, key=lambda k: sweep_summ[k]["ROC-AUC"][0])
    print(f"\n    Best k by mean ROC-AUC: k={best_k}")

    # ---------------- Stage 2: head-to-head ----------------
    print(f"\n[2] Head-to-head (feature-node vs patient-similarity k={best_k}) ...")

    feat_folds, pat_folds = [], []
    feat_oof, pat_oof = {}, {}
    for sd in SEEDS:
        _, ff, fprob, fpred, ftrue = run_gcn_cv(
            X, y, GCNConfig(), TrainConfig(), topology="corr_mst",
            threshold=TAU, n_splits=5, seed=sd, tune_threshold=True, return_oof=True)
        feat_folds.extend(ff)
        feat_oof[sd] = (fprob, fpred, ftrue)

        _, pf, pprob, ppred, ptrue = run_patient_gcn_cv(
            X, y, k=best_k, n_splits=5, seed=sd, return_oof=True)
        pat_folds.extend(pf)
        pat_oof[sd] = (pprob, ppred, ptrue)

    feat_s, pat_s = _aggregate(feat_folds), _aggregate(pat_folds)

    comp = pd.DataFrame([
        {"Formulation": "Feature-node graph (Ours)",
         **{m: f"{feat_s[m][0]:.4f} ± {feat_s[m][1]:.4f}" for m in METRIC_KEYS}},
        {"Formulation": f"Patient-similarity graph (k={best_k})",
         **{m: f"{pat_s[m][0]:.4f} ± {pat_s[m][1]:.4f}" for m in METRIC_KEYS}},
    ])
    comp.to_csv("results/table_patient_vs_feature.csv", index=False)

    print("\n" + comp[["Formulation", "Accuracy", "F1", "ROC-AUC", "MCC"]]
          .to_string(index=False))

    # ---- statistics on pooled OOF, per seed then averaged ----
    stat_rows = []
    for sd in SEEDS:
        fprob, fpred, true = feat_oof[sd]
        pprob, ppred, _ = pat_oof[sd]
        fc, pc = (fpred == true).astype(int), (ppred == true).astype(int)
        b = int(np.sum((fc == 0) & (pc == 1)))
        c = int(np.sum((fc == 1) & (pc == 0)))
        tbl = [[int(np.sum((fc == 1) & (pc == 1))), c],
               [b, int(np.sum((fc == 0) & (pc == 0)))]]
        mc = mcnemar(tbl, exact=(b + c) < 25)
        auc_f, auc_p, dl_p = delong_roc_test(true, fprob, pprob)
        stat_rows.append({
            "Seed": sd,
            "AUC (feature-node)": round(float(auc_f), 4),
            "AUC (patient-sim)": round(float(auc_p), 4),
            "ΔAUC": round(float(auc_f - auc_p), 4),
            "b": b, "c": c,
            "McNemar p": round(float(mc.pvalue), 4),
            "DeLong p": f"{float(dl_p):.4g}",
        })
    stats = pd.DataFrame(stat_rows)
    stats.to_csv("results/table_patient_vs_feature_stats.csv", index=False)

    print("\nStatistical comparison (pooled OOF, per seed):")
    print(stats.to_string(index=False))
    print("\nSaved: results/table_patient_knn_sweep.csv, "
          "table_patient_vs_feature.csv, table_patient_vs_feature_stats.csv")


if __name__ == "__main__":
    main()
