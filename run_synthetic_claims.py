# -*- coding: utf-8 -*-
"""
run_synthetic_claims.py
=======================
Simulation study: are the paper's claims detectable at n = 10,000?

THIS IS NOT EXTERNAL VALIDATION. The generator is fitted to Cleveland,
so agreement here shows the claims are consistent with the estimated
dependence structure at large n -- it cannot show the real population
has that structure.

Two data-generating processes are used:
  additive     -- no interactions; a linear model is Bayes-optimal, so
                  this scenario asks whether the graph prior COSTS
                  anything when no feature interactions exist.
  interaction  -- multiplicative terms along the Pearson+MST edges, so
                  this scenario asks whether it GAINS when they do.

Claims tested
  C1  selective sparse topology  >  fully connected / no graph / random
  C2  MST supplies connectivity, not accuracy  (corr_mst ~= corr_only)
  C3  GCN is at parity with LR / RF in-distribution
  C4  the graph prior transports better under covariate shift
"""

from __future__ import annotations

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

import torch

from pipeline import (
    FEATURES, NUMERIC_COLS, TAU, DEVICE, set_seed,
    GCNConfig, TrainConfig, GCN, train_model, make_graphs,
    build_edge_index, predict_probs, best_threshold, metrics_from, _aggregate,
)
from run_external_stats import delong_roc_test

os.makedirs("results", exist_ok=True)

N = 10_000
SEEDS = [42, 7]
TOPOLOGIES = ["corr_mst", "corr_only", "random", "fully_connected", "none"]
LABEL = {
    "corr_mst": "Corr+MST (Ours)",
    "corr_only": "Corr only (no MST)",
    "random": "Random graph",
    "fully_connected": "Fully connected",
    "none": "No graph",
}
# larger n converges faster; keep the architecture identical to the paper
TCFG = TrainConfig(epochs=40, patience=8, batch_size=256)


def load_syn(tag):
    df = pd.read_csv(f"synthetic_{tag}_{N}.csv")
    return df[FEATURES].copy(), df["target"].astype(int).copy()


# ----- disk cache: these CV runs cost minutes each -----
import pickle
CACHE_DIR = "results/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def cached(key, fn):
    path = os.path.join(CACHE_DIR, key + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    val = fn()
    with open(path, "wb") as f:
        pickle.dump(val, f)
    return val


# ----------------------------------------------------------------------
# GCN cross-validation returning per-fold AUC and pooled OOF probabilities
# ----------------------------------------------------------------------
def gcn_cv(X, y, topology, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(X), np.nan)
    folds = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_trf, y_trf = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_trf, y_trf, test_size=0.15, stratify=y_trf, random_state=seed)

        sc = MinMaxScaler()
        X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
        X_tr[NUMERIC_COLS] = sc.fit_transform(X_tr[NUMERIC_COLS])
        X_val[NUMERIC_COLS] = sc.transform(X_val[NUMERIC_COLS])
        X_te[NUMERIC_COLS] = sc.transform(X_te[NUMERIC_COLS])

        ei, _ = build_edge_index(X_tr, topology=topology,
                                 threshold=TAU, seed=seed + fold)
        g_tr = make_graphs(X_tr, y_tr, ei)
        g_val = make_graphs(X_val, y_val, ei)
        g_te = make_graphs(X_te, y_te, ei)

        set_seed(seed + fold)
        model = train_model(GCN(GCNConfig()), g_tr, g_val, TCFG)

        vp, vt = predict_probs(model, g_val)
        thr = best_threshold(vp, vt)
        tp, tt = predict_probs(model, g_te)
        oof[te_idx] = tp
        folds.append(metrics_from(tt, (tp >= thr).astype(int), tp))
    return folds, oof


def sk_cv(X, y, factory, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(X), np.nan)
    folds = []
    for tr_idx, te_idx in skf.split(X, y):
        X_trf, y_trf = X.iloc[tr_idx], y.iloc[tr_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_trf, y_trf, test_size=0.15, stratify=y_trf, random_state=seed)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_tr[FEATURES]); Xval = sc.transform(X_val[FEATURES])
        Xte = sc.transform(X.iloc[te_idx][FEATURES])
        clf = factory().fit(Xtr, y_tr.values)
        thr = best_threshold(clf.predict_proba(Xval)[:, 1], y_val.values)
        p = clf.predict_proba(Xte)[:, 1]
        oof[te_idx] = p
        folds.append(metrics_from(y.iloc[te_idx].values, (p >= thr).astype(int), p))
    return folds, oof


BASELINES = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000,
                                                      class_weight="balanced"),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                    class_weight="balanced",
                                                    random_state=42),
}


# ----------------------------------------------------------------------
# in-distribution claim tests
# ----------------------------------------------------------------------
def run_dgp(tag):
    X, y = load_syn(tag)
    print(f"\n{'=' * 78}\nDGP = {tag}   n={len(X)}   prevalence={y.mean():.4f}\n{'=' * 78}")

    fold_auc, oof_store, summaries = {}, {}, {}

    for topo in TOPOLOGIES:
        aucs, all_folds, last_oof = [], [], None
        for seed in SEEDS:
            folds, oof = cached(f"{tag}_{topo}_{seed}",
                                lambda: gcn_cv(X, y, topo, seed))
            all_folds.extend(folds)
            aucs.extend([f["ROC-AUC"] for f in folds])
            last_oof = oof
        fold_auc[LABEL[topo]] = aucs
        oof_store[LABEL[topo]] = last_oof
        summaries[LABEL[topo]] = _aggregate(all_folds)
        print(f"  {LABEL[topo]:<22} AUC={np.mean(aucs):.4f}  "
              f"F1={summaries[LABEL[topo]]['F1'][0]:.4f}")

    for name, fac in BASELINES.items():
        aucs, all_folds, last_oof = [], [], None
        for seed in SEEDS:
            folds, oof = sk_cv(X, y, fac, seed)
            all_folds.extend(folds)
            aucs.extend([f["ROC-AUC"] for f in folds])
            last_oof = oof
        fold_auc[name] = aucs
        oof_store[name] = last_oof
        summaries[name] = _aggregate(all_folds)
        print(f"  {name:<22} AUC={np.mean(aucs):.4f}  "
              f"F1={summaries[name]['F1'][0]:.4f}")

    # results table
    keys = ["Accuracy", "F1", "ROC-AUC", "MCC", "Recall", "Specificity"]
    rows = [{"Model": k, **{m: f"{v[m][0]:.4f} ± {v[m][1]:.4f}" for m in keys}}
            for k, v in summaries.items()]
    pd.DataFrame(rows).to_csv(f"results/table_synthetic_{tag}_performance.csv",
                              index=False)

    # paired tests vs Corr+MST
    ref = LABEL["corr_mst"]
    stat_rows = []
    for other in [k for k in fold_auc if k != ref]:
        a, b = np.array(fold_auc[ref]), np.array(fold_auc[other])
        try:
            _, wp = wilcoxon(a, b)
        except ValueError:
            wp = float("nan")
        _, _, dl = delong_roc_test(y.values, oof_store[ref], oof_store[other])
        stat_rows.append({
            "Comparison": f"{ref} vs {other}",
            "ΔAUC (fold mean)": round(float(a.mean() - b.mean()), 4),
            "Wilcoxon p": round(float(wp), 5),
            "DeLong p (pooled OOF)": f"{dl:.3g}",
            "Wins/n": f"{int((a > b).sum())}/{len(a)}",
        })
    pd.DataFrame(stat_rows).to_csv(f"results/table_synthetic_{tag}_stats.csv",
                                   index=False)
    print("\n  paired contrasts vs Corr+MST:")
    for r in stat_rows:
        print(f"    {r['Comparison']:<48} ΔAUC={r['ΔAUC (fold mean)']:+.4f}  "
              f"Wilcoxon p={r['Wilcoxon p']}  DeLong p={r['DeLong p (pooled OOF)']}")
    return summaries, fold_auc


# ----------------------------------------------------------------------
# C4: transport under covariate shift
# ----------------------------------------------------------------------
def run_transport():
    print(f"\n{'=' * 78}\nC4  TRANSPORT UNDER COVARIATE SHIFT\n{'=' * 78}")
    X_src, y_src = load_syn("interaction")
    df_t = pd.read_csv(f"synthetic_shifted_{N}.csv")
    X_tgt, y_tgt = df_t[FEATURES].copy(), df_t["target"].astype(int)

    rows, probs = [], {}
    for topo in ["corr_mst", "corr_only", "none", "fully_connected"]:
        aucs = []
        for seed in SEEDS:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_src, y_src, test_size=0.15, stratify=y_src, random_state=seed)
            sc = MinMaxScaler()
            X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_tgt.copy()
            X_tr[NUMERIC_COLS] = sc.fit_transform(X_tr[NUMERIC_COLS])
            X_val[NUMERIC_COLS] = sc.transform(X_val[NUMERIC_COLS])
            X_te[NUMERIC_COLS] = sc.transform(X_te[NUMERIC_COLS])
            ei, _ = build_edge_index(X_tr, topology=topo, threshold=TAU, seed=seed)
            set_seed(seed)
            model = train_model(GCN(GCNConfig()),
                                make_graphs(X_tr, y_tr, ei),
                                make_graphs(X_val, y_val, ei), TCFG)
            p, t = predict_probs(model, make_graphs(X_te, y_tgt, ei))
            aucs.append(roc_auc_score(t, p))
            probs[(topo, seed)] = p
        rows.append({"Model": LABEL[topo],
                     "External AUC": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"})
        print(f"  {LABEL[topo]:<22} shifted-cohort AUC={np.mean(aucs):.4f}")

    for name, fac in BASELINES.items():
        aucs = []
        for seed in SEEDS:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_src, y_src, test_size=0.15, stratify=y_src, random_state=seed)
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_tr[FEATURES])
            clf = fac().fit(Xtr, y_tr.values)
            p = clf.predict_proba(sc.transform(X_tgt[FEATURES]))[:, 1]
            aucs.append(roc_auc_score(y_tgt.values, p))
            probs[(name, seed)] = p
        rows.append({"Model": name,
                     "External AUC": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"})
        print(f"  {name:<22} shifted-cohort AUC={np.mean(aucs):.4f}")

    # DeLong on the first seed's transported probabilities
    s0 = SEEDS[0]
    stat = []
    for other in ["none", "fully_connected", "Logistic Regression", "Random Forest"]:
        key = (other, s0)
        _, _, dl = delong_roc_test(y_tgt.values, probs[("corr_mst", s0)], probs[key])
        stat.append({"Comparison": f"Corr+MST vs {LABEL.get(other, other)}",
                     "DeLong p": f"{dl:.3g}"})
    pd.DataFrame(rows).to_csv("results/table_synthetic_transport.csv", index=False)
    pd.DataFrame(stat).to_csv("results/table_synthetic_transport_stats.csv",
                              index=False)
    print("  DeLong (seed %d):" % s0)
    for s in stat:
        print(f"    {s['Comparison']:<44} p={s['DeLong p']}")


if __name__ == "__main__":
    run_dgp("additive")
    run_dgp("interaction")
    run_transport()
    print("\nSaved: results/table_synthetic_*.csv")
