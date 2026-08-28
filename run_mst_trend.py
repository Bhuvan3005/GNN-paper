# -*- coding: utf-8 -*-
"""
run_mst_trend.py
================
Ordered-alternative test of the connectivity hypothesis.

The hypothesis under test is ORDINAL, not a set of independent pairwise
claims:

    more connected graph  ->  better external transport
    corr_mst (1 comp)  >  corr_only (2.8 comp)  >  none (8 comp)

Testing that with nine pairwise DeLong tests and a Holm correction is
underpowered and answers the wrong question. Page's trend test is the
appropriate statistic: it tests H1 (a pre-specified monotone ordering)
against H0 (no ordering) using all seeds jointly.

`fully_connected` is deliberately EXCLUDED from the trend: it is
connected (1 component) but non-selective, so it is not a point on the
connectivity axis. It is reported separately as the selectivity control.

Reads results/mst_transport_probs.npz (written by run_mst_transport.py);
no model is retrained.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import page_trend_test, wilcoxon
from sklearn.metrics import roc_auc_score

COHORTS = ["Hungarian", "Switzerland", "VA Long Beach"]
# ordered from LEAST to MOST connected, as Page's test expects ascending
ORDER = ["none", "corr_only", "corr_mst"]
NICE = {"none": "No graph (8 comp)",
        "corr_only": "Corr only (2.8 comp)",
        "corr_mst": "Corr+MST (1 comp)",
        "fully_connected": "Fully connected (1 comp, non-selective)"}


def per_seed_aucs(z, topo, cohort):
    probs = z[f"{topo}|{cohort}|prob"]      # [n_seeds, n_patients]
    true = z[f"{topo}|{cohort}|true"]
    return np.array([roc_auc_score(true, p) for p in probs])


def main():
    z = np.load("results/mst_transport_probs.npz")

    print("=" * 76)
    print("ORDERED-ALTERNATIVE TEST — does connectivity predict transport?")
    print("H1: AUC(no graph) < AUC(corr only) < AUC(corr+MST)")
    print("=" * 76)

    rows = []
    pooled_matrix = []

    for c in COHORTS:
        mat = np.column_stack([per_seed_aucs(z, t, c) for t in ORDER])
        pooled_matrix.append(mat)
        # Page's test wants rows = blocks (seeds), cols = ordered conditions
        try:
            res = page_trend_test(mat, ranked=False)
            L, p = res.statistic, res.pvalue
        except Exception as e:
            L, p = float("nan"), float("nan")

        n_seeds = mat.shape[0]
        perfect = int(np.sum((mat[:, 0] < mat[:, 1]) & (mat[:, 1] < mat[:, 2])))
        rows.append({
            "Cohort": c,
            "AUC no-graph": f"{mat[:, 0].mean():.4f}",
            "AUC corr-only": f"{mat[:, 1].mean():.4f}",
            "AUC corr+MST": f"{mat[:, 2].mean():.4f}",
            "Monotone seeds": f"{perfect}/{n_seeds}",
            "Page L": round(float(L), 1) if np.isfinite(L) else "n/a",
            "Page p": f"{p:.4g}" if np.isfinite(p) else "n/a",
        })
        print(f"\n{c}:")
        print(f"  mean AUC  none={mat[:,0].mean():.4f}  "
              f"corr_only={mat[:,1].mean():.4f}  corr_mst={mat[:,2].mean():.4f}")
        print(f"  seeds with perfect monotone ordering: {perfect}/{n_seeds}")
        print(f"  Page's trend test: L={L:.1f}, p={p:.4g}")

    # ---- pooled across cohorts (blocks = seed x cohort) ----
    pooled = np.vstack(pooled_matrix)
    res = page_trend_test(pooled, ranked=False)
    n_blocks = pooled.shape[0]
    perfect = int(np.sum((pooled[:, 0] < pooled[:, 1]) & (pooled[:, 1] < pooled[:, 2])))
    print("\n" + "-" * 76)
    print(f"POOLED (blocks = {n_blocks} seed x cohort):")
    print(f"  perfect monotone ordering in {perfect}/{n_blocks} blocks")
    print(f"  Page's trend test: L={res.statistic:.1f}, p={res.pvalue:.4g}")
    rows.append({
        "Cohort": "POOLED (all cohorts)",
        "AUC no-graph": f"{pooled[:, 0].mean():.4f}",
        "AUC corr-only": f"{pooled[:, 1].mean():.4f}",
        "AUC corr+MST": f"{pooled[:, 2].mean():.4f}",
        "Monotone seeds": f"{perfect}/{n_blocks}",
        "Page L": round(float(res.statistic), 1),
        "Page p": f"{res.pvalue:.4g}",
    })

    df = pd.DataFrame(rows)
    df.to_csv("results/table_mst_trend.csv", index=False)

    # ---- selectivity control: connected but non-selective ----
    print("\n" + "=" * 76)
    print("SELECTIVITY CONTROL — fully connected is also 1 component")
    print("=" * 76)
    sel_rows = []
    for c in COHORTS:
        a = per_seed_aucs(z, "corr_mst", c)
        b = per_seed_aucs(z, "fully_connected", c)
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = float("nan")
        sel_rows.append({
            "Cohort": c,
            "AUC Corr+MST": f"{a.mean():.4f}",
            "AUC Fully connected": f"{b.mean():.4f}",
            "ΔAUC": f"{a.mean() - b.mean():+.4f}",
            "Wilcoxon p (seeds)": f"{p:.4g}",
        })
        print(f"  {c:<16} Corr+MST={a.mean():.4f}  FC={b.mean():.4f}  "
              f"Δ={a.mean()-b.mean():+.4f}  p={p:.4g}")
    pd.DataFrame(sel_rows).to_csv("results/table_mst_selectivity.csv", index=False)

    print("\nSaved: results/table_mst_trend.csv, results/table_mst_selectivity.csv")


if __name__ == "__main__":
    main()
