# -*- coding: utf-8 -*-
"""
apply_holm.py
=============
Uniform multiple-comparison control across every significance table in the
manuscript, so that statistical rigour is consistent throughout.

Holm--Bonferroni controls the family-wise error rate. Applying it requires
declaring what the FAMILY is, which is a judgement, not a mechanical step.
The families used here are declared explicitly below:

  table_statistical_tests
      GCN vs each baseline. Family = the set of baseline comparisons,
      corrected separately for each test type (McNemar, Wilcoxon).

  table_learnable_stats
      3 topology comparisons x 3 metrics = 9 hypotheses, one family.

  table_conv_stats
      GCN vs each alternative operator; family per test type.

  table_external_stats
      Cohort x comparison; family per test type.

  table_knn_stats
      Already corrected in run_knn_topology.py (9 hypotheses).

  table_node_identity_gap_sweep
      Identity-embedding dimension sweep (d=0,2,4,8,16,32) x 2 topology
      contrasts (Corr+MST-vs-No-graph, Corr+MST-vs-Fully-connected) = 12
      distinct hypotheses, one family: does the graph-gap sign reversal
      seen at d=8 in the original single-point test generalise across
      dimension.

  table_patient_vs_feature_stats
      DELIBERATELY NOT CORRECTED. Its rows are three SEEDS of the same
      single hypothesis (feature-node vs patient-similarity), i.e.
      replicates, not distinct hypotheses. Holm across replicates would
      inflate the p-values of one hypothesis and is statistically wrong.
      A note column records this instead.

Running this script is idempotent: it recomputes the adjusted columns from
the raw p-values every time.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

RESULTS = "results"

# table -> list of raw p-value columns, each forming its own family
FAMILIES = {
    "table_statistical_tests": ["McNemar_p", "Wilcoxon_p(foldF1)"],
    "table_learnable_stats": ["Wilcoxon p"],
    "table_conv_stats": ["DeLong p (median)", "McNemar p (median)"],
    "table_external_stats": ["DeLong p", "McNemar p"],
    "table_node_identity_gap_sweep": ["Wilcoxon p (15 paired folds)"],
}

# tables whose repeated rows are replicates of ONE hypothesis
NOT_A_FAMILY = {
    "table_patient_vs_feature_stats":
        "rows are seed replicates of a single hypothesis, not a test family",
}

# stale uncorrected verdict columns to drop once Holm is present
DROP_IF_PRESENT = ["Significant (α=0.05)", "Significant (alpha=0.05)"]


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    adj = np.full(p.shape, np.nan)
    rej = np.zeros(p.shape, dtype=bool)
    if ok.sum() == 0:
        return adj, rej
    r, pa, _, _ = multipletests(p[ok], alpha=0.05, method="holm")
    adj[ok], rej[ok] = pa, r
    return adj, rej


def main():
    for name, pcols in FAMILIES.items():
        path = os.path.join(RESULTS, f"{name}.csv")
        if not os.path.exists(path):
            print(f"skip (missing): {name}")
            continue
        df = pd.read_csv(path)

        for c in DROP_IF_PRESENT:
            if c in df.columns:
                df = df.drop(columns=c)

        for pcol in pcols:
            if pcol not in df.columns:
                print(f"  ! {name}: no column '{pcol}'")
                continue
            adj, rej = holm(df[pcol].values)
            base = pcol.replace(" p", "").replace("_p", "").strip()
            df[f"Holm p ({base})" if len(pcols) > 1 else "Holm p"] = np.round(adj, 4)
            df[f"Sig. Holm ({base})" if len(pcols) > 1 else "Significant (Holm)"] = \
                np.where(rej, "yes", "no")

        df.to_csv(path, index=False)
        n = len(df)
        print(f"corrected: {name}  ({n} tests, families={pcols})")

    for name, why in NOT_A_FAMILY.items():
        path = os.path.join(RESULTS, f"{name}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["Multiplicity"] = "per-seed replicate (no correction)"
        df.to_csv(path, index=False)
        print(f"annotated (not corrected): {name} — {why}")


if __name__ == "__main__":
    main()
