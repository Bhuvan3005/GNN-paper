# -*- coding: utf-8 -*-
"""
run_patient_external.py
=======================
External transport of the CONVENTIONAL patient-similarity GNN, under the
identical protocol used for the feature-node model in `run_external.py`.

Why this matters
----------------
The paper argues that the feature-node formulation is *inductive* whereas
patient-similarity GNNs are *transductive*. This script turns that
architectural argument into a measurement: both formulations are trained
on Cleveland only and transported to Hungarian / Switzerland / VA.

Transport protocol (strict)
---------------------------
  * Scaler, model weights and decision threshold: fit on Cleveland only.
  * Feature-node model : the Cleveland feature graph is reused directly --
                         each external patient is scored independently.
  * Patient-sim model  : a NEW k-NN graph must be constructed over the
                         external cohort's own patients before any
                         prediction can be made. This is an inherent cost
                         of the transductive formulation and is reported
                         as such: the model cannot score an isolated
                         patient, it needs a cohort.
  * Each external cohort is touched exactly once, for scoring.
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pipeline as P
from patient_graph import (
    knn_edge_index, train_patient_gcn, node_probs,
)
from run_external import (
    EXT_FEATURES, EXT_CATEGORICAL, EXT_NUMERIC, COHORTS,
    patch_pipeline_features, load_cohort, complete_cases,
)
from run_external_stats import delong_roc_test

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123, 2024, 5]
METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MCC", "Specificity"]
BEST_K_FILE = "results/table_patient_knn_sweep.csv"


def pick_best_k(default=10):
    """Reuse the k selected on Cleveland (never re-tuned on external data)."""
    try:
        df = pd.read_csv(BEST_K_FILE)
        aucs = df["ROC-AUC"].astype(str).str.split("±").str[0].astype(float)
        return int(df.loc[aucs.idxmax(), "k"])
    except Exception:
        return default


def transport_patient_once(X_dev, y_dev, ext_data, k, seed):
    """Train the patient-similarity GCN on Cleveland, score each cohort once."""
    idx = np.arange(len(X_dev))
    inner_tr, inner_val = train_test_split(
        idx, test_size=0.15, stratify=y_dev, random_state=seed)

    # Cleveland scaler, fit on inner-train only
    Xs = X_dev[EXT_FEATURES].copy()
    scaler = MinMaxScaler().fit(Xs.iloc[inner_tr][EXT_NUMERIC])
    Xs[EXT_NUMERIC] = scaler.transform(Xs[EXT_NUMERIC])
    Xv = Xs.values.astype(np.float32)

    ei_dev = knn_edge_index(Xv, k=k)
    x_dev = torch.tensor(Xv, dtype=torch.float)
    y_t = torch.tensor(y_dev.values.astype(np.float32), dtype=torch.float)

    n = len(X_dev)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[inner_tr] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[inner_val] = True

    model = train_patient_gcn(x_dev, ei_dev, y_t, train_mask, val_mask, seed=seed)

    probs_dev = node_probs(model, x_dev, ei_dev)
    thr = P.best_threshold(probs_dev[inner_val],
                           y_dev.values.astype(int)[inner_val])   # frozen

    out = {}
    for name, (Xe, ye) in ext_data.items():
        Xe_s = Xe[EXT_FEATURES].copy()
        Xe_s[EXT_NUMERIC] = scaler.transform(Xe_s[EXT_NUMERIC])   # Cleveland scaler
        Xe_v = Xe_s.values.astype(np.float32)
        # a cohort graph must be built before any external prediction
        ei_ext = knn_edge_index(Xe_v, k=k)
        pe = node_probs(model, torch.tensor(Xe_v, dtype=torch.float), ei_ext)
        out[name] = (P.metrics_from(ye.values, (pe >= thr).astype(int), pe), pe)
    return out


def main():
    patch_pipeline_features(EXT_FEATURES, EXT_CATEGORICAL)
    k = pick_best_k()
    print("=" * 78)
    print("EXTERNAL TRANSPORT — patient-similarity GNN (conventional)")
    print(f"k = {k} (selected on Cleveland, never re-tuned externally)")
    print(f"Seeds: {SEEDS}")
    print("=" * 78)

    dev_raw, dev_y = load_cohort("processed.cleveland.data")
    X_dev, y_dev, n_dev = complete_cases(dev_raw, dev_y, EXT_FEATURES)
    print(f"\nCleveland development set: {n_dev} complete cases")

    ext_data = {}
    for name, fname in COHORTS.items():
        d_raw, y_raw = load_cohort(fname)
        Xe, ye, n_c = complete_cases(d_raw, y_raw, EXT_FEATURES)
        if n_c > 0 and ye.nunique() >= 2:
            ext_data[name] = (Xe, ye)

    print(f"\nTransporting over {len(SEEDS)} seeds ...")
    runs, prob_store = [], {n: [] for n in ext_data}
    for sd in SEEDS:
        r = transport_patient_once(X_dev, y_dev, ext_data, k, sd)
        runs.append({n: v[0] for n, v in r.items()})
        for n, v in r.items():
            prob_store[n].append(v[1])

    rows = []
    for name in ext_data:
        agg = {m: [r[name][m] for r in runs] for m in METRIC_KEYS}
        rows.append({"Cohort": name,
                     "Model": f"Patient-similarity GNN (k={k})",
                     "n": len(ext_data[name][1]),
                     **{m: f"{np.mean(agg[m]):.4f} ± {np.std(agg[m]):.4f}"
                        for m in METRIC_KEYS}})
    pat_df = pd.DataFrame(rows)
    pat_df.to_csv("results/table_patient_external.csv", index=False)

    print("\n" + "=" * 78)
    print("PATIENT-SIMILARITY GNN — EXTERNAL RESULTS")
    print("=" * 78)
    print(pat_df[["Cohort", "Model", "n", "ROC-AUC", "F1", "MCC"]].to_string(index=False))

    # ---- compare against the feature-node external AUCs ----
    try:
        feat = pd.read_csv("results/table_external_validation.csv")
        feat = feat[feat["Model"] == "GCN (Ours)"]
        cmp_rows = []
        for name in ext_data:
            fa = float(feat[feat["Cohort"] == name]["ROC-AUC"]
                       .iloc[0].split("±")[0])
            pa = float(pat_df[pat_df["Cohort"] == name]["ROC-AUC"]
                       .iloc[0].split("±")[0])
            cmp_rows.append({"Cohort": name,
                             "AUC feature-node": round(fa, 4),
                             "AUC patient-sim": round(pa, 4),
                             "ΔAUC": round(fa - pa, 4)})
        cmp = pd.DataFrame(cmp_rows)
        cmp.to_csv("results/table_patient_vs_feature_external.csv", index=False)
        print("\nFeature-node vs patient-similarity on external cohorts:")
        print(cmp.to_string(index=False))
    except Exception as e:
        print(f"\n(comparison table skipped: {e})")

    np.savez("results/patient_external_probs.npz",
             **{n: np.mean(np.stack(v), axis=0) for n, v in prob_store.items()})
    print("\nSaved: results/table_patient_external.csv, "
          "table_patient_vs_feature_external.csv")


if __name__ == "__main__":
    main()
