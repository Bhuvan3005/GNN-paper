# -*- coding: utf-8 -*-
"""
run_external.py
================
External (out-of-distribution) validation on the three non-Cleveland UCI
heart-disease cohorts: Hungarian, Switzerland, and Long Beach VA.

WHY A REDUCED FEATURE SET
-------------------------
The paper's primary model uses all 13 Cleveland features. That model
CANNOT be validated externally: the non-Cleveland cohorts systematically
do not record several of those variables. Complete-case counts for the
full 13-feature vector are:

    Hungarian 1/294 | Switzerland 0/123 | VA 1/200

because `ca` (95-99% missing), `thal` (42-90%), and `slope` (14-65%) are
largely unrecorded outside Cleveland, and Switzerland's `chol` column is
100% zero-encoded (a documented sentinel, not a real measurement).

We therefore define a TRANSPORTABLE feature set of the 8 variables that
are reliably recorded in all four cohorts:

    age, sex, cp, trestbps, restecg, thalach, exang, oldpeak

and retrain the identical architecture on Cleveland restricted to those
8 features. Dropping `chol` (Cleveland target-corr +0.085) and `fbs`
(+0.025) costs almost nothing predictively; dropping `ca` (+0.460),
`thal` (+0.516) and `slope` (+0.339) is a genuine loss that is forced by
data availability, not chosen.

DESIGN (strict — no peeking at external data)
---------------------------------------------
  * Scaler, correlation+MST graph, model weights and the decision
    threshold are ALL fit on Cleveland only.
  * Each external cohort is touched exactly once, for scoring.
  * Baselines (LR, RF) are transported under the identical protocol.

This lets us decompose the performance drop into two separate causes:
  (a) feature reduction   -> 13-feature vs 8-feature Cleveland CV
  (b) cohort shift        -> 8-feature Cleveland CV vs external
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

import pipeline as P   # patched below for the reduced feature set

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

os.makedirs("results", exist_ok=True)
SEED = 42

RAW_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Transportable 8-feature set (see module docstring)
EXT_FEATURES = ["age", "sex", "cp", "trestbps", "restecg",
                "thalach", "exang", "oldpeak"]
EXT_CATEGORICAL = ["sex", "cp", "restecg", "exang"]
EXT_NUMERIC = [c for c in EXT_FEATURES if c not in EXT_CATEGORICAL]

COHORTS = {
    "Hungarian": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}

METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MCC", "Specificity"]
SEEDS = [42, 7, 123, 2024, 5]


def patch_pipeline_features(features, categorical):
    """Point the shared pipeline at the reduced feature set.

    pipeline.build_edge_index / make_graphs read these module-level
    globals, so rebinding them here reuses the exact tested code paths
    with an 8-node graph instead of a 13-node one. The architecture is
    unchanged (1 scalar per node)."""
    P.FEATURES = list(features)
    P.N_FEATURES = len(features)
    P.CATEGORICAL_COLS = list(categorical)
    P.NUMERIC_COLS = [c for c in features if c not in categorical]


def load_cohort(fname):
    """Load a raw UCI cohort, apply sentinel->NaN, binarise the target."""
    d = pd.read_csv(f"data_external/{fname}", names=RAW_COLS, na_values="?")
    # documented physiologically-impossible sentinels
    d.loc[d["chol"] == 0, "chol"] = np.nan
    d.loc[d["trestbps"] == 0, "trestbps"] = np.nan
    y = (d["num"] > 0).astype(int)
    return d, y


def complete_cases(d, y, features):
    m = d[features].notna().all(axis=1)
    return d.loc[m, features].reset_index(drop=True), y.loc[m].reset_index(drop=True), int(m.sum())


def transport_once(X_dev, y_dev, ext_data, seed):
    """Fit on Cleveland with one seed, score every external cohort once.
    Returns {(cohort, model): metrics_dict}."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=seed)

    scaler = MinMaxScaler()
    X_tr = X_tr.copy(); X_val = X_val.copy()
    X_tr[EXT_NUMERIC] = scaler.fit_transform(X_tr[EXT_NUMERIC])
    X_val[EXT_NUMERIC] = scaler.transform(X_val[EXT_NUMERIC])

    edge_index, _ = P.build_edge_index(X_tr, topology="corr_mst",
                                       threshold=P.TAU, seed=seed)
    g_tr = P.make_graphs(X_tr, y_tr, edge_index)
    g_val = P.make_graphs(X_val, y_val, edge_index)

    P.set_seed(seed)
    model = P.GCN(P.GCNConfig())
    model = P.train_model(model, g_tr, g_val, P.TrainConfig())
    vprob, vtrue = P.predict_probs(model, g_val)
    thr = P.best_threshold(vprob, vtrue)           # frozen

    sc_b = StandardScaler().fit(X_tr[EXT_FEATURES])
    Xtr_b, Xval_b = sc_b.transform(X_tr[EXT_FEATURES]), sc_b.transform(X_val[EXT_FEATURES])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=seed).fit(Xtr_b, y_tr)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=seed).fit(Xtr_b, y_tr)
    thr_lr = P.best_threshold(lr.predict_proba(Xval_b)[:, 1], y_val.values)
    thr_rf = P.best_threshold(rf.predict_proba(Xval_b)[:, 1], y_val.values)

    out = {}
    for name, (Xe, ye) in ext_data.items():
        Xe_s = Xe.copy()
        Xe_s[EXT_NUMERIC] = scaler.transform(Xe_s[EXT_NUMERIC])   # Cleveland scaler
        g_ext = P.make_graphs(Xe_s, ye, edge_index)               # Cleveland graph
        prob, true = P.predict_probs(model, g_ext)
        out[(name, "GCN (Ours)")] = P.metrics_from(true, (prob >= thr).astype(int), prob)

        Xe_b = sc_b.transform(Xe[EXT_FEATURES])
        for mdl, t, label in [(lr, thr_lr, "Logistic Regression"),
                              (rf, thr_rf, "Random Forest")]:
            pb = mdl.predict_proba(Xe_b)[:, 1]
            out[(name, label)] = P.metrics_from(ye.values, (pb >= t).astype(int), pb)
    return out


def main():
    patch_pipeline_features(EXT_FEATURES, EXT_CATEGORICAL)
    print("=" * 78)
    print(f"EXTERNAL VALIDATION — transportable {len(EXT_FEATURES)}-feature model")
    print(f"Features: {EXT_FEATURES}")
    print(f"Seeds: {SEEDS}")
    print("=" * 78)

    # ---------------- Cleveland (development cohort) ----------------
    dev_raw, dev_y = load_cohort("processed.cleveland.data")
    X_dev, y_dev, n_dev = complete_cases(dev_raw, dev_y, EXT_FEATURES)
    print(f"\nCleveland (development): {n_dev} complete cases, "
          f"disease prevalence {100*y_dev.mean():.1f}%")

    # (a) internal CV of the reduced model -> isolates feature-reduction cost
    print(f"\n[1] Internal 5-fold CV of the reduced {len(EXT_FEATURES)}-feature "
          f"model (Cleveland), over {len(SEEDS)} seeds...")
    int_folds = []
    for sd in SEEDS:
        _, folds = P.run_gcn_cv(
            X_dev, y_dev, P.GCNConfig(), P.TrainConfig(),
            topology="corr_mst", threshold=P.TAU, n_splits=5, seed=sd,
            tune_threshold=True)
        int_folds.extend(folds)
    internal = P._aggregate(int_folds)
    print("    " + "  ".join(f"{k}={internal[k][0]:.3f}±{internal[k][1]:.3f}"
                             for k in ["Accuracy", "F1", "ROC-AUC", "MCC"]))

    # ---------------- Load external cohorts once ----------------
    ext_data, cohort_info = {}, []
    for name, fname in COHORTS.items():
        d_raw, y_raw = load_cohort(fname)
        n_total = len(d_raw)
        prev_total = float(y_raw.mean())
        Xe, ye, n_complete = complete_cases(d_raw, y_raw, EXT_FEATURES)
        evaluable = n_complete > 0 and ye.nunique() >= 2
        if evaluable:
            ext_data[name] = (Xe, ye)
        n_pos = int(ye.sum()) if n_complete else 0
        n_neg = n_complete - n_pos
        cohort_info.append((
            name, n_total, n_complete,
            n_total - n_complete,
            round(100 * (n_total - n_complete) / n_total, 1) if n_total else float("nan"),
            prev_total,
            float(ye.mean()) if n_complete else float("nan"),
            n_pos, n_neg,
            "evaluated" if evaluable else "not evaluable"))

    # ---------------- Repeat transport over seeds ----------------
    print(f"\n[2] Fitting on Cleveland and transporting, {len(SEEDS)} seeds...")
    runs = [transport_once(X_dev, y_dev, ext_data, sd) for sd in SEEDS]

    rows = []
    for name in ext_data:
        for label in ["GCN (Ours)", "Logistic Regression", "Random Forest"]:
            agg = {k: [r[(name, label)][k] for r in runs] for k in METRIC_KEYS}
            rows.append({
                "Cohort": name, "Model": label, "n": len(ext_data[name][1]),
                **{k: f"{np.mean(agg[k]):.4f} ± {np.std(agg[k]):.4f}" for k in METRIC_KEYS}})
    ext_df = pd.DataFrame(rows)
    ext_df.to_csv("results/table_external_validation.csv", index=False)

    ref = pd.DataFrame([{
        "Setting": f"Cleveland internal CV ({len(EXT_FEATURES)}-feature)",
        **{k: f"{internal[k][0]:.4f} ± {internal[k][1]:.4f}" for k in METRIC_KEYS}}])
    ref.to_csv("results/table_external_reference.csv", index=False)

    info = pd.DataFrame(cohort_info, columns=[
        "Cohort", "Total records", f"Complete cases ({len(EXT_FEATURES)} features)",
        "Missing (dropped)", "Missing %",
        "Prevalence (all records)", "Prevalence (complete cases)",
        "Positive (n)", "Negative (n)", "Status"])
    for col in ["Prevalence (all records)", "Prevalence (complete cases)"]:
        info[col] = info[col].apply(lambda v: "n/a" if pd.isna(v) else f"{100*v:.1f}%")
    info.to_csv("results/table_external_cohorts.csv", index=False)

    print("\n" + "=" * 78)
    print(f"EXTERNAL VALIDATION RESULTS (mean ± std over {len(SEEDS)} seeds)")
    print("=" * 78)
    print(ext_df[["Cohort", "Model", "n", "ROC-AUC", "F1", "MCC", "Specificity"]]
          .to_string(index=False))
    print("\nCohort availability:")
    print(info.to_string(index=False))
    print("\nSaved: results/table_external_validation.csv, "
          "table_external_cohorts.csv, table_external_reference.csv")


if __name__ == "__main__":
    main()
