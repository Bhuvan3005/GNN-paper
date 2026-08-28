# -*- coding: utf-8 -*-
"""
run_dataset_summary.py
=======================
Dataset description + feature descriptives + runtime/memory footprint,
computed on the correct UCI Cleveland dataset (303 patients, 13 features)
as fixed by the project methodology. Replaces earlier tables that were
mistakenly computed on a different combined Statlog+Cleveland+Hungary+VA
file (1190 samples, 11 features) — that file and its derived tables have
been removed from the project.
"""

import os
import sys
import time
import tracemalloc

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

from pipeline import (
    CSV_PATH, FEATURES, NUMERIC_COLS, CATEGORICAL_COLS, CLINICAL_TOP,
    load_data, GCNConfig, TrainConfig, run_gcn_cv, best_threshold,
)

os.makedirs("results", exist_ok=True)
SEED = 42


def dataset_summary():
    df = pd.read_csv(CSV_PATH)
    X, y = load_data()
    dup = int(df.duplicated().sum())
    rows = [
        ("Dataset file", CSV_PATH),
        ("Source", "UCI Machine Learning Repository — Cleveland Heart Disease"),
        ("Samples", len(df)),
        ("Duplicate rows", dup),
        ("Input features", len(FEATURES)),
        ("Target column", "target"),
        ("Classes", "[0, 1]"),
        ("Class distribution", f"0: {int((y==0).sum())}  |  1: {int((y==1).sum())}"),
        ("Class balance (%)", f"{100*(y==0).mean():.1f} / {100*(y==1).mean():.1f}"),
        ("Missing values (NaN)", int(df[FEATURES].isna().sum().sum())),
        ("Continuous features", f"{len(NUMERIC_COLS)}: {NUMERIC_COLS}"),
        ("Categorical features", f"{len(CATEGORICAL_COLS)}: {CATEGORICAL_COLS}"),
        ("Clinical risk factors (XAI ref. set)", f"{len(CLINICAL_TOP)}: {CLINICAL_TOP}"),
    ]
    out = pd.DataFrame(rows, columns=["Property", "Value"])
    out.to_csv("results/table_dataset_summary.csv", index=False)
    print("DATASET SUMMARY")
    print(out.to_string(index=False))
    return df, X, y


def feature_descriptives(df):
    rows = []
    for f in FEATURES:
        col = df[f]
        rows.append({
            "Feature": f,
            "Type": "categorical" if f in CATEGORICAL_COLS else "continuous",
            "Unique": int(col.nunique()),
            "Min": float(col.min()),
            "Max": float(col.max()),
            "Mean": round(float(col.mean()), 3),
            "Std": round(float(col.std()), 3),
            "Missing": int(col.isna().sum()),
        })
    out = pd.DataFrame(rows)
    out.to_csv("results/table_feature_descriptives.csv", index=False)
    print("\nFEATURE DESCRIPTIVES")
    print(out.to_string(index=False))


def runtime_memory(X, y):
    """Wall-clock time and peak memory for one 5-fold CV pass of each model."""
    results = []

    # GCN
    tracemalloc.start()
    t0 = time.perf_counter()
    run_gcn_cv(X, y, GCNConfig(), TrainConfig(), n_splits=5, seed=SEED)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results.append(("GCN (Ours)", elapsed, elapsed / 5, peak / 1e6))

    # Baselines
    factories = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, random_state=SEED),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, fac in factories.items():
        tracemalloc.start()
        t0 = time.perf_counter()
        for tr_idx, te_idx in skf.split(X, y):
            X_tr_full, y_tr_full = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te = X.iloc[te_idx]
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr_full, y_tr_full, test_size=0.15,
                stratify=y_tr_full, random_state=SEED)
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_tr[FEATURES])
            Xval = scaler.transform(X_val[FEATURES])
            Xte = scaler.transform(X_te[FEATURES])
            clf = fac()
            clf.fit(Xtr, y_tr.values)
            vprob = clf.predict_proba(Xval)[:, 1]
            thr = best_threshold(vprob, y_val.values)
            _ = (clf.predict_proba(Xte)[:, 1] >= thr).astype(int)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append((name, elapsed, elapsed / 5, peak / 1e6))

    out = pd.DataFrame(results, columns=[
        "Model", "Train+eval time (s)", "Time / fold (s)", "Peak memory (MB)"])
    out["Train+eval time (s)"] = out["Train+eval time (s)"].round(2)
    out["Time / fold (s)"] = out["Time / fold (s)"].round(2)
    out["Peak memory (MB)"] = out["Peak memory (MB)"].round(2)
    out.to_csv("results/table_runtime_memory.csv", index=False)
    print("\nRUNTIME & MEMORY (5-fold CV, single seed, CPU)")
    print(out.to_string(index=False))


def main():
    df, X, y = dataset_summary()
    feature_descriptives(df)
    runtime_memory(X, y)
    print("\nSaved: results/table_dataset_summary.csv, "
          "results/table_feature_descriptives.csv, "
          "results/table_runtime_memory.csv")


if __name__ == "__main__":
    main()
