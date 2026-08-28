# -*- coding: utf-8 -*-
"""
run_external_bootstrap.py
=========================
Bootstrap confidence intervals for the external-validation AUCs.

WHY THIS IS NECESSARY
---------------------
`run_external.py` reports external AUC as "mean +/- std over 5 seeds".
That spread measures only *model-initialisation* variability. It says
nothing about *sampling* uncertainty, which dominates on these cohorts:
Switzerland contributes only 8 negative cases and VA only 30. Reporting
e.g. "0.744 +/- 0.008" therefore understates the true uncertainty by a
large factor and is the single most attackable number in the manuscript.

WHAT THIS COMPUTES
------------------
A nested bootstrap that propagates BOTH sources of variation. For each of
B replicates:
    1. draw a random seed s (model-initialisation uncertainty),
    2. draw a stratified resample of the cohort's patients with
       replacement, preserving the per-class counts (sampling
       uncertainty),
    3. evaluate every model on that resample.

Because step 2 uses the SAME resampled patients for every model, the
resulting Delta-AUC intervals are *paired*, which is the correct way to
express a between-model difference.

Reported: 95% percentile CIs for each model's AUC, and for Delta-AUC
against the feature-node GCN. A Delta-AUC interval containing zero means
the advantage is not established on that cohort, regardless of the
point estimate.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pipeline as P
from run_external import (
    EXT_FEATURES, EXT_CATEGORICAL, COHORTS,
    patch_pipeline_features, load_cohort, complete_cases,
)
from run_external_stats import fit_and_score

os.makedirs("results", exist_ok=True)

SEEDS = [42, 7, 123, 2024, 5]
B = 2000
RNG = np.random.default_rng(42)
MODELS = ["GCN (Ours)", "Logistic Regression", "Random Forest"]


def stratified_resample(y, rng):
    """Resample indices with replacement, preserving per-class counts."""
    idx = []
    for cls in (0, 1):
        pool = np.flatnonzero(y == cls)
        if len(pool):
            idx.append(rng.choice(pool, size=len(pool), replace=True))
    return np.concatenate(idx)


def main():
    patch_pipeline_features(EXT_FEATURES, EXT_CATEGORICAL)
    print("=" * 78)
    print("BOOTSTRAP CONFIDENCE INTERVALS — external validation")
    print(f"B = {B} replicates, nested over {len(SEEDS)} seeds, stratified")
    print("=" * 78)

    dev_raw, dev_y = load_cohort("processed.cleveland.data")
    X_dev, y_dev, _ = complete_cases(dev_raw, dev_y, EXT_FEATURES)

    ext_data = {}
    for name, fname in COHORTS.items():
        d_raw, y_raw = load_cohort(fname)
        Xe, ye, n_c = complete_cases(d_raw, y_raw, EXT_FEATURES)
        if n_c > 0 and ye.nunique() >= 2:
            ext_data[name] = (Xe, ye)

    # ---- fit once per seed, cache per-patient probabilities ----
    print(f"\nFitting on Cleveland for {len(SEEDS)} seeds ...")
    scored_by_seed = []
    for sd in SEEDS:
        scored_by_seed.append(fit_and_score(X_dev, y_dev, ext_data, seed=sd))
    print("done.\n")

    auc_rows, delta_rows = [], []

    for cohort in ext_data:
        y_true = np.asarray(scored_by_seed[0][cohort]["y"]).astype(int)
        n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())

        boot = {m: np.empty(B) for m in MODELS}
        boot_delta = {m: np.empty(B) for m in MODELS[1:]}

        for b in range(B):
            s = RNG.integers(len(SEEDS))                 # model uncertainty
            idx = stratified_resample(y_true, RNG)       # sampling uncertainty
            yb = y_true[idx]
            if yb.min() == yb.max():                     # degenerate draw
                for m in MODELS:
                    boot[m][b] = np.nan
                for m in MODELS[1:]:
                    boot_delta[m][b] = np.nan
                continue
            aucs = {}
            for m in MODELS:
                prob = np.asarray(scored_by_seed[s][cohort][m][0])
                aucs[m] = roc_auc_score(yb, prob[idx])
                boot[m][b] = aucs[m]
            for m in MODELS[1:]:
                boot_delta[m][b] = aucs["GCN (Ours)"] - aucs[m]   # paired

        for m in MODELS:
            v = boot[m][~np.isnan(boot[m])]
            lo, hi = np.percentile(v, [2.5, 97.5])
            auc_rows.append({
                "Cohort": cohort, "Model": m,
                "n (pos/neg)": f"{n_pos}/{n_neg}",
                "AUC": f"{np.mean(v):.3f}",
                "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                "CI width": round(float(hi - lo), 3),
            })

        for m in MODELS[1:]:
            v = boot_delta[m][~np.isnan(boot_delta[m])]
            lo, hi = np.percentile(v, [2.5, 97.5])
            excl = "yes" if (lo > 0 or hi < 0) else "no"
            delta_rows.append({
                "Cohort": cohort,
                "Comparison": f"GCN − {m}",
                "ΔAUC": f"{np.mean(v):+.3f}",
                "95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                "Excludes zero": excl,
            })

    auc_df = pd.DataFrame(auc_rows)
    delta_df = pd.DataFrame(delta_rows)
    auc_df.to_csv("results/table_external_bootstrap_auc.csv", index=False)
    delta_df.to_csv("results/table_external_bootstrap_delta.csv", index=False)

    print("EXTERNAL AUC WITH BOOTSTRAP 95% CI")
    print(auc_df.to_string(index=False))
    print("\nPAIRED ΔAUC vs THE FEATURE-NODE GCN")
    print(delta_df.to_string(index=False))
    print("\nSaved: results/table_external_bootstrap_auc.csv, "
          "results/table_external_bootstrap_delta.csv")


if __name__ == "__main__":
    main()
