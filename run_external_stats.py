# -*- coding: utf-8 -*-
"""
run_external_stats.py
======================
Statistical testing of the external-validation result.

The multi-seed run showed the feature-node GCN leading ROC-AUC on all
three external cohorts. Because that is a headline claim, we test it
formally rather than relying on seed variance alone:

  * DeLong's test for two CORRELATED ROC curves (same patients scored by
    both models) -> p-value on the AUC difference.
  * McNemar's test on the thresholded predictions.

Implementation of the fast DeLong estimator follows Sun & Xu (2014),
"Fast Implementation of DeLong's Algorithm for Comparing the Areas Under
Correlated Receiver Operating Characteristic Curves".
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pipeline as P
from run_external import (
    EXT_FEATURES, EXT_CATEGORICAL, EXT_NUMERIC, COHORTS, SEED,
    patch_pipeline_features, load_cohort, complete_cases,
)

os.makedirs("results", exist_ok=True)


# ----------------------------------------------------------------------
# Fast DeLong (Sun & Xu 2014)
# ----------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    """Return (auc_a, auc_b, p_value) for H0: AUC_a == AUC_b."""
    y_true = np.asarray(y_true)
    order = (-y_true).argsort(kind="mergesort")      # positives first
    label_1_count = int(y_true.sum())
    preds = np.vstack((np.asarray(prob_a)[order], np.asarray(prob_b)[order]))
    aucs, cov = _fast_delong(preds, label_1_count)
    l = np.array([[1, -1]])
    var = float(l.dot(cov).dot(l.T))
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan")
    z = float(aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2.0 * (1.0 - scipy.stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p


# ----------------------------------------------------------------------
def fit_and_score(X_dev, y_dev, ext_data, seed=SEED):
    """Fit all three models on Cleveland; return per-cohort probs/preds."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=seed)
    scaler = MinMaxScaler()
    X_tr = X_tr.copy(); X_val = X_val.copy()
    X_tr[EXT_NUMERIC] = scaler.fit_transform(X_tr[EXT_NUMERIC])
    X_val[EXT_NUMERIC] = scaler.transform(X_val[EXT_NUMERIC])

    edge_index, _ = P.build_edge_index(X_tr, topology="corr_mst",
                                       threshold=P.TAU, seed=seed)
    P.set_seed(seed)
    model = P.train_model(P.GCN(P.GCNConfig()),
                          P.make_graphs(X_tr, y_tr, edge_index),
                          P.make_graphs(X_val, y_val, edge_index),
                          P.TrainConfig())
    vprob, vtrue = P.predict_probs(model, P.make_graphs(X_val, y_val, edge_index))
    thr = P.best_threshold(vprob, vtrue)

    sc_b = StandardScaler().fit(X_tr[EXT_FEATURES])
    Xtr_b, Xval_b = sc_b.transform(X_tr[EXT_FEATURES]), sc_b.transform(X_val[EXT_FEATURES])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=seed).fit(Xtr_b, y_tr)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=seed).fit(Xtr_b, y_tr)
    thr_lr = P.best_threshold(lr.predict_proba(Xval_b)[:, 1], y_val.values)
    thr_rf = P.best_threshold(rf.predict_proba(Xval_b)[:, 1], y_val.values)

    scored = {}
    for name, (Xe, ye) in ext_data.items():
        Xe_s = Xe.copy()
        Xe_s[EXT_NUMERIC] = scaler.transform(Xe_s[EXT_NUMERIC])
        prob_g, true = P.predict_probs(model, P.make_graphs(Xe_s, ye, edge_index))
        Xe_b = sc_b.transform(Xe[EXT_FEATURES])
        scored[name] = {
            "y": true,
            "GCN (Ours)": (prob_g, (prob_g >= thr).astype(int)),
            "Logistic Regression": (lr.predict_proba(Xe_b)[:, 1],
                                    (lr.predict_proba(Xe_b)[:, 1] >= thr_lr).astype(int)),
            "Random Forest": (rf.predict_proba(Xe_b)[:, 1],
                              (rf.predict_proba(Xe_b)[:, 1] >= thr_rf).astype(int)),
        }
    return scored


def main():
    patch_pipeline_features(EXT_FEATURES, EXT_CATEGORICAL)
    dev_raw, dev_y = load_cohort("processed.cleveland.data")
    X_dev, y_dev, _ = complete_cases(dev_raw, dev_y, EXT_FEATURES)

    ext_data = {}
    for name, fname in COHORTS.items():
        d_raw, y_raw = load_cohort(fname)
        Xe, ye, n = complete_cases(d_raw, y_raw, EXT_FEATURES)
        if n > 0 and ye.nunique() >= 2:
            ext_data[name] = (Xe, ye)

    print("Fitting on Cleveland (seed %d) and scoring external cohorts...\n" % SEED)
    scored = fit_and_score(X_dev, y_dev, ext_data)

    rows = []
    for cohort, d in scored.items():
        y = d["y"]
        g_prob, g_pred = d["GCN (Ours)"]
        for other in ["Logistic Regression", "Random Forest"]:
            o_prob, o_pred = d[other]
            auc_g, auc_o, p_delong = delong_roc_test(y, g_prob, o_prob)
            gc, oc = (g_pred == y).astype(int), (o_pred == y).astype(int)
            b = int(np.sum((gc == 0) & (oc == 1)))
            c = int(np.sum((gc == 1) & (oc == 0)))
            tbl = [[int(np.sum((gc == 1) & (oc == 1))), c],
                   [b, int(np.sum((gc == 0) & (oc == 0)))]]
            mc = mcnemar(tbl, exact=(b + c) < 25)
            rows.append({
                "Cohort": cohort,
                "Comparison": f"GCN vs {other}",
                "AUC (GCN)": round(auc_g, 4),
                "AUC (other)": round(auc_o, 4),
                "ΔAUC": round(auc_g - auc_o, 4),
                "DeLong p": f"{p_delong:.2e}" if p_delong < 1e-3 else round(p_delong, 4),
                "McNemar p": f"{mc.pvalue:.2e}" if mc.pvalue < 1e-3 else round(float(mc.pvalue), 4),
            })
            print(f"{cohort:14} GCN vs {other:20} "
                  f"ΔAUC={auc_g-auc_o:+.4f}  DeLong p={p_delong:.4g}  "
                  f"McNemar p={mc.pvalue:.4g}")

    df = pd.DataFrame(rows)
    df.to_csv("results/table_external_stats.csv", index=False)
    print("\nSaved: results/table_external_stats.csv")


if __name__ == "__main__":
    main()
