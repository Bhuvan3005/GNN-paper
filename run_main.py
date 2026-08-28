# -*- coding: utf-8 -*-
"""
run_main.py
===========
Task 1 (fair evaluation, reproducibility) + Task 4 (statistical validation).

Produces, under ONE identical 5-fold Stratified CV protocol:
  * GCN full-model out-of-fold (OOF) results (mean +/- std across folds),
  * fair baselines (LR, RF, GBM, MLP) with in-fold StandardScaler,
  * pooled OOF metrics for every model,
  * McNemar (pooled OOF) and Wilcoxon signed-rank (per-fold) tests
    for GCN vs LR and GCN vs RF.

All tables are written to ./results as CSV.
"""

import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from pipeline import (
    FEATURES, NUMERIC_COLS, TAU, load_data, set_seed,
    GCNConfig, TrainConfig, run_gcn_cv, metrics_from, best_threshold, _aggregate,
)

import os
os.makedirs("results", exist_ok=True)

SEED = 42                       # representative seed used for pooled-OOF McNemar
SEEDS = [42, 7, 123]            # repeated cross-validation
N_SPLITS = 5                    # 5-fold stratified CV (per methodology)
METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MCC", "Specificity"]


# ----------------------------------------------------------------------
# Baselines under the SAME fold splits, with in-fold scaling
# ----------------------------------------------------------------------
def baseline_oof(X, y, clf_factory, seed=SEED, n_splits=N_SPLITS, tune_threshold=True):
    """Return (fold_metrics, oof_prob, oof_pred, oof_true) using the same
    StratifiedKFold splits as the GCN, StandardScaler fit in-fold, and an
    inner-validation decision-threshold tuned identically to the GCN."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.full(len(X), np.nan)
    oof_pred = np.full(len(X), np.nan)
    oof_true = y.values.astype(int)
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr_full, y_tr_full = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15,
            stratify=y_tr_full, random_state=seed)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_tr[FEATURES])
        Xval = scaler.transform(X_val[FEATURES])
        Xte = scaler.transform(X_te[FEATURES])

        clf = clf_factory()
        clf.fit(Xtr, y_tr.values)
        vprob = clf.predict_proba(Xval)[:, 1]
        thr = best_threshold(vprob, y_val.values) if tune_threshold else 0.5

        tprob = clf.predict_proba(Xte)[:, 1]
        tpred = (tprob >= thr).astype(int)
        oof_prob[te_idx] = tprob
        oof_pred[te_idx] = tpred
        fold_metrics.append(metrics_from(y_te.values.astype(int), tpred, tprob))

    from pipeline import _aggregate
    return _aggregate(fold_metrics), fold_metrics, oof_prob, oof_pred.astype(int), oof_true


def gcn_multiseed(X, y):
    """Run the GCN over all SEEDS at N_SPLITS-fold CV.
    Returns pooled summary, flat per-(seed,fold) F1 list, and the seed-SEED
    OOF predictions (a complete 303-sample partition) for McNemar."""
    all_folds, fold_f1 = [], []
    ref = (None, None, None)
    for sd in SEEDS:
        _, folds, prob, pred, true = run_gcn_cv(
            X, y, GCNConfig(), TrainConfig(), topology="corr_mst",
            threshold=TAU, n_splits=N_SPLITS, seed=sd,
            tune_threshold=True, return_oof=True, verbose=False)
        all_folds.extend(folds)
        fold_f1.extend([m["F1"] for m in folds])
        if sd == SEED:
            ref = (pred, true, prob)
    return _aggregate(all_folds), fold_f1, ref


def baseline_multiseed(X, y, fac):
    all_folds, fold_f1 = [], []
    ref = (None, None, None)
    for sd in SEEDS:
        _, folds, prob, pred, true = baseline_oof(X, y, fac, seed=sd, n_splits=N_SPLITS)
        all_folds.extend(folds)
        fold_f1.extend([m["F1"] for m in folds])
        if sd == SEED:
            ref = (pred, true, prob)
    return _aggregate(all_folds), fold_f1, ref


def main():
    X, y = load_data()
    print(f"Dataset: {len(X)} patients, class balance = {dict(y.value_counts())}")
    print(f"Protocol: {N_SPLITS}-fold Stratified CV, repeated over seeds {SEEDS}")
    print("=" * 78)

    results = {}          # name -> summary dict
    oof_store = {}        # name -> dict(pred,true,prob,f1)

    # ---- GCN full model ----
    print(f"Training GCN full model ({N_SPLITS}-fold x {len(SEEDS)} seeds)...")
    gsum, gf1, (gpred, gtrue, gprob) = gcn_multiseed(X, y)
    results["GCN (Ours)"] = gsum
    oof_store["GCN (Ours)"] = {"pred": gpred, "true": gtrue, "prob": gprob, "f1": gf1}

    # ---- Baselines ----
    factories = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, random_state=SEED),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED,
            n_jobs=1, verbosity=0),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=200, max_depth=-1, num_leaves=15,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, verbosity=-1, min_child_samples=5),
    }
    for name, fac in factories.items():
        print(f"Training {name} ({N_SPLITS}-fold x {len(SEEDS)} seeds)...")
        s, f1, (pred, true, prob) = baseline_multiseed(X, y, fac)
        results[name] = s
        oof_store[name] = {"pred": pred, "true": true, "prob": prob, "f1": f1}

    # ---- Comparison table ----
    rows = []
    for name, s in results.items():
        row = {"Model": name}
        row.update({k: f"{s[k][0]:.4f} ± {s[k][1]:.4f}" for k in METRIC_KEYS})
        rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv("results/table_model_comparison.csv", index=False)
    print("\n" + "=" * 78)
    print(f"MODEL COMPARISON ({N_SPLITS}-fold CV x {len(SEEDS)} seeds, mean ± std)")
    print("=" * 78)
    print(comp.to_string(index=False))

    # also store pooled-OOF point metrics (seed-SEED complete partition)
    pooled_rows = []
    for name, d in oof_store.items():
        m = metrics_from(d["true"], d["pred"], d["prob"])
        pooled_rows.append({"Model": name, **{k: round(m[k], 4) for k in METRIC_KEYS}})
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv("results/table_pooled_oof.csv", index=False)

    # ---- Statistical tests: GCN vs LR, GCN vs RF ----
    print("\n" + "=" * 78)
    print("STATISTICAL SIGNIFICANCE (GCN vs baselines)")
    print("=" * 78)
    gcn_pred = oof_store["GCN (Ours)"]["pred"]
    true = oof_store["GCN (Ours)"]["true"]
    gcn_correct = (gcn_pred == true).astype(int)
    gcn_f1_folds = oof_store["GCN (Ours)"]["f1"]

    stat_rows = []
    for other in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]:
        opred = oof_store[other]["pred"]
        ocorrect = (opred == true).astype(int)
        # McNemar contingency on the seed-SEED pooled OOF partition (all 303)
        n01 = int(np.sum((gcn_correct == 0) & (ocorrect == 1)))  # GCN wrong, other right
        n10 = int(np.sum((gcn_correct == 1) & (ocorrect == 0)))  # GCN right, other wrong
        table = [[int(np.sum((gcn_correct == 1) & (ocorrect == 1))), n10],
                 [n01, int(np.sum((gcn_correct == 0) & (ocorrect == 0)))]]
        mc = mcnemar(table, exact=(n01 + n10) < 25)
        # Wilcoxon on per-(seed,fold) F1 (30 aligned pairs)
        ofolds = oof_store[other]["f1"]
        try:
            w_stat, w_p = wilcoxon(gcn_f1_folds, ofolds)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        stat_rows.append({
            "Comparison": f"GCN vs {other}",
            "McNemar_b(GCN-wrong,other-right)": n01,
            "McNemar_c(GCN-right,other-wrong)": n10,
            "McNemar_stat": round(float(mc.statistic), 4),
            "McNemar_p": round(float(mc.pvalue), 4),
            "Wilcoxon_p(foldF1)": round(float(w_p), 4),
        })
        print(f"GCN vs {other}: McNemar p={mc.pvalue:.4f} "
              f"(b={n01}, c={n10}), Wilcoxon(foldF1,n={len(gcn_f1_folds)}) p={w_p:.4f}")

    stat = pd.DataFrame(stat_rows)
    stat.to_csv("results/table_statistical_tests.csv", index=False)

    # persist raw OOF for reuse
    np.savez("results/oof_predictions.npz",
             **{f"{k}_pred": v["pred"] for k, v in oof_store.items()},
             **{f"{k}_prob": v["prob"] for k, v in oof_store.items()},
             true=true)
    with open("results/main_summary.json", "w") as f:
        json.dump({name: {k: list(map(float, s[k])) for k in METRIC_KEYS}
                   for name, s in results.items()}, f, indent=2)
    print("\nSaved: results/table_model_comparison.csv, table_pooled_oof.csv, "
          "table_statistical_tests.csv, main_summary.json")


if __name__ == "__main__":
    main()
