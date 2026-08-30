# -*- coding: utf-8 -*-
"""
Robustness check for the pooled Delta-AUC interval.

WHY: the primary run returned GCN - Random Forest = +0.050
[+0.004, +0.135] (stratified). A lower bound 0.004 above zero is within
Monte Carlo error of a B=2000 percentile bootstrap, so "excludes zero"
may be an artefact of the particular RNG draw. This script re-runs the
pooled bootstrap at larger B across several independent RNG seeds and
reports the spread of the lower bound, plus the bootstrap-tail mass at
or below zero.

Decision rule: report "excludes zero" ONLY if the lower bound stays
positive across every RNG seed at large B.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from _fastauc import fast_auc as roc_auc_score
import pipeline as P
from run_external import (EXT_FEATURES, EXT_CATEGORICAL, COHORTS,
                          patch_pipeline_features, load_cohort, complete_cases)
from run_external_stats import fit_and_score
from run_external_bootstrap import stratified_resample, SEEDS, MODELS

B_BIG = 20000
RNG_SEEDS = [42, 7, 2024, 999, 12345]

def pooled_draws(scored, cohorts, rng, n_boot):
    y = {c: np.asarray(scored[0][c]["y"]).astype(int) for c in cohorts}
    n = {c: len(y[c]) for c in cohorts}
    tot = sum(n.values())
    out = {m: np.full(n_boot, np.nan) for m in MODELS[1:]}
    for b in range(n_boot):
        s = rng.integers(len(SEEDS))
        idx = {c: stratified_resample(y[c], rng) for c in cohorts}
        a = {}
        for m in MODELS:
            num = 0.0
            for c in cohorts:
                yb = y[c][idx[c]]
                pb = np.asarray(scored[s][c][m][0])[idx[c]]
                num += n[c] * roc_auc_score(yb, pb)
            a[m] = num / tot
        for m in MODELS[1:]:
            out[m][b] = a["GCN (Ours)"] - a[m]
    return out

def main():
    patch_pipeline_features(EXT_FEATURES, EXT_CATEGORICAL)
    dev_raw, dev_y = load_cohort("processed.cleveland.data")
    X_dev, y_dev, _ = complete_cases(dev_raw, dev_y, EXT_FEATURES)
    ext = {}
    for name, fn in COHORTS.items():
        d, yv = load_cohort(fn)
        Xe, ye, nc = complete_cases(d, yv, EXT_FEATURES)
        if nc > 0 and ye.nunique() >= 2:
            ext[name] = (Xe, ye)
    print(f"fitting {len(SEEDS)} seeds ...", flush=True)
    scored = [fit_and_score(X_dev, y_dev, ext, seed=sd) for sd in SEEDS]
    print("done\n", flush=True)

    rows = []
    for rs in RNG_SEEDS:
        d = pooled_draws(scored, list(ext), np.random.default_rng(rs), B_BIG)
        for m in MODELS[1:]:
            v = d[m][~np.isnan(d[m])]
            lo, hi = np.percentile(v, [2.5, 97.5])
            rows.append({"rng_seed": rs, "comparison": f"GCN - {m}",
                         "mean": round(float(v.mean()), 4),
                         "lo95": round(float(lo), 4), "hi95": round(float(hi), 4),
                         "P(delta<=0)": round(float((v <= 0).mean()), 4),
                         "excludes_0": "yes" if lo > 0 else "no"})
        print(f"  rng {rs} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/table_pooled_robustness.csv", index=False)
    print(f"\nB = {B_BIG} per seed\n")
    print(df.to_string(index=False))
    print("\nVERDICT")
    for m in MODELS[1:]:
        s = df[df.comparison == f"GCN - {m}"]
        stable = (s.excludes_0 == "yes").all()
        print(f"  GCN vs {m:20s} lo95 range [{s.lo95.min():+.4f}, {s.lo95.max():+.4f}]"
              f"  tail mass {s['P(delta<=0)'].min():.4f}-{s['P(delta<=0)'].max():.4f}"
              f"  -> {'STABLE, excludes zero' if stable else 'UNSTABLE, flips across seeds'}")

if __name__ == "__main__":
    main()
