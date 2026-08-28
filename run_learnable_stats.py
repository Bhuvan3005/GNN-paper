# -*- coding: utf-8 -*-
"""
run_learnable_stats.py
======================
Paired significance testing for the learnable-topology study.

Each configuration is run over the identical (seed x fold) partitions, so
fold-level metrics are PAIRED. We use the Wilcoxon signed-rank test over the
15 paired folds:
  * fixed Corr+MST      vs  learned (lambda1=1, lambda2=1)
  * learned (1,1)       vs  learned (0,0)   -> does the prior anchor matter?
  * learned (1,1)       vs  learned (1,0)   -> does L_conn matter?
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pipeline import load_data, GCNConfig, TrainConfig, run_gcn_cv, TAU

os.makedirs("results", exist_ok=True)
SEEDS = [42, 7, 123]

CONFIGS = {
    "Fixed Corr+MST": ("corr_mst", {}),
    "Learned (λ1=1, λ2=1)": ("corr_mst_learned",
                             {"lambda_l1": 1.0, "lambda_conn": 1.0}),
    "Learned (λ1=0, λ2=0)": ("corr_mst_learned",
                             {"lambda_l1": 0.0, "lambda_conn": 0.0}),
    "Learned (λ1=1, λ2=0)": ("corr_mst_learned",
                             {"lambda_l1": 1.0, "lambda_conn": 0.0}),
}

COMPARISONS = [
    ("Fixed Corr+MST", "Learned (λ1=1, λ2=1)",
     "Does end-to-end refinement beat the fixed prior?"),
    ("Learned (λ1=1, λ2=1)", "Learned (λ1=0, λ2=0)",
     "Does the Pearson anchor (λ1) matter?"),
    ("Learned (λ1=1, λ2=1)", "Learned (λ1=1, λ2=0)",
     "Does the connectivity term (λ2) matter?"),
]


def fold_metrics(topology, kwargs):
    out = []
    for seed in SEEDS:
        _, folds = run_gcn_cv(
            load_data()[0], load_data()[1],
            gcfg=GCNConfig(**kwargs), tcfg=TrainConfig(),
            topology=topology, threshold=TAU, n_splits=5, seed=seed,
            tune_threshold=True)
        out.extend(folds)
    return out


def main():
    print("Collecting paired fold metrics (3 seeds x 5 folds = 15 pairs)...")
    store = {}
    for name, (topo, kw) in CONFIGS.items():
        store[name] = fold_metrics(topo, kw)
        print(f"  {name}: done")

    rows = []
    for a, b, question in COMPARISONS:
        for metric in ["F1", "ROC-AUC", "MCC"]:
            va = np.array([m[metric] for m in store[a]])
            vb = np.array([m[metric] for m in store[b]])
            try:
                stat, p = wilcoxon(va, vb)
            except ValueError:
                stat, p = np.nan, np.nan
            rows.append({
                "Comparison": f"{a} vs {b}",
                "Question": question,
                "Metric": metric,
                f"Mean A": round(float(va.mean()), 4),
                f"Mean B": round(float(vb.mean()), 4),
                "Δ (A−B)": round(float(va.mean() - vb.mean()), 4),
                "Wilcoxon p": round(float(p), 4),
                "Significant (α=0.05)": "yes" if p < 0.05 else "no",
            })
    df = pd.DataFrame(rows)
    df.to_csv("results/table_learnable_stats.csv", index=False)
    print("\nPAIRED SIGNIFICANCE (Wilcoxon signed-rank, 15 paired folds)")
    print(df.to_string(index=False))
    print("\nSaved: results/table_learnable_stats.csv")


if __name__ == "__main__":
    main()
