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


def pooled_bootstrap(scored_by_seed, ext_data, rng, n_boot=B):
    """
    Joint bootstrap across ALL external cohorts -> one paired interval.

    WHY A JOINT BOOTSTRAP AND NOT A COMBINATION OF THE PER-COHORT ONES
    -----------------------------------------------------------------
    The three per-cohort intervals above cannot be combined post hoc (e.g.
    by inverse-variance / random-effects meta-analysis) without an
    independence assumption that is false here: all three cohorts are
    scored by the SAME fitted models, so their errors are correlated
    through the shared fit. Treating them as independent understates the
    variance of the combined estimate. The only honest combination
    resamples the cohorts jointly, which is what this does.

    Concretely: within a replicate the seed s is drawn ONCE and used for
    every cohort, mirroring deployment (one fitted model transported to
    all sites). That shared draw is precisely the correlation a post-hoc
    combination cannot represent.

    TWO ESTIMANDS, BOTH REPORTED
    ----------------------------
    (a) STRATIFIED (primary). AUC is computed WITHIN each cohort on the
        resample, then combined as a complete-case-weighted mean. This is
        the transport claim: how well does the model separate cases from
        controls *at a site*.

    (b) CONCATENATED (secondary, diagnostic only). One ROC over all
        patients merged into a single ranking. Do NOT headline this:
        cohort prevalence runs from 36% (Hungarian) to 93%
        (Switzerland), so a model whose scores merely shift with cohort
        is rewarded for discriminating *sites* rather than *patients*.
        Reported so the gap between (a) and (b) is visible rather than
        hidden.

    Resampling is stratified by cohort AND class: each cohort contributes
    its own patients with per-class counts preserved, so Switzerland's 8
    negatives stay 8 negatives and cannot vanish from a replicate.
    """
    cohort_names = list(ext_data.keys())
    y_by_cohort = {c: np.asarray(scored_by_seed[0][c]["y"]).astype(int)
                   for c in cohort_names}
    n_by_cohort = {c: len(y_by_cohort[c]) for c in cohort_names}
    total_n = sum(n_by_cohort.values())

    strat = {m: np.full(n_boot, np.nan) for m in MODELS}
    concat = {m: np.full(n_boot, np.nan) for m in MODELS}
    strat_d = {m: np.full(n_boot, np.nan) for m in MODELS[1:]}
    concat_d = {m: np.full(n_boot, np.nan) for m in MODELS[1:]}

    for b in range(n_boot):
        s = rng.integers(len(SEEDS))          # ONE model draw for all cohorts
        idx_by_cohort = {c: stratified_resample(y_by_cohort[c], rng)
                         for c in cohort_names}

        per_model_strat, per_model_concat = {}, {}
        for m in MODELS:
            num, ys, ps = 0.0, [], []
            ok = True
            for c in cohort_names:
                idx = idx_by_cohort[c]
                yb = y_by_cohort[c][idx]
                pb = np.asarray(scored_by_seed[s][c][m][0])[idx]
                if yb.min() == yb.max():      # cannot happen under stratification
                    ok = False
                    break
                num += n_by_cohort[c] * roc_auc_score(yb, pb)
                ys.append(yb)
                ps.append(pb)
            if not ok:
                break
            per_model_strat[m] = num / total_n
            per_model_concat[m] = roc_auc_score(np.concatenate(ys),
                                               np.concatenate(ps))
        if len(per_model_strat) != len(MODELS):
            continue

        for m in MODELS:
            strat[m][b] = per_model_strat[m]
            concat[m][b] = per_model_concat[m]
        for m in MODELS[1:]:
            strat_d[m][b] = per_model_strat["GCN (Ours)"] - per_model_strat[m]
            concat_d[m][b] = per_model_concat["GCN (Ours)"] - per_model_concat[m]

    def summarise(store, kind, is_delta):
        out = []
        for m in (MODELS[1:] if is_delta else MODELS):
            v = store[m][~np.isnan(store[m])]
            lo, hi = np.percentile(v, [2.5, 97.5])
            row = {"Cohort": "Pooled external cohorts", "Estimand": kind,
                   "n (patients)": total_n, "B (valid)": int(len(v))}
            if is_delta:
                row.update({
                    "Comparison": f"GCN \u2212 {m}",
                    "\u0394AUC": f"{np.mean(v):+.3f}",
                    "95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                    "Excludes zero": "yes" if (lo > 0 or hi < 0) else "no",
                })
            else:
                row.update({
                    "Model": m,
                    "AUC": f"{np.mean(v):.3f}",
                    "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                    "CI width": round(float(hi - lo), 3),
                })
            out.append(row)
        return out

    auc_rows = (summarise(strat, "stratified", False)
                + summarise(concat, "concatenated", False))
    delta_rows = (summarise(strat_d, "stratified", True)
                  + summarise(concat_d, "concatenated", True))
    return pd.DataFrame(auc_rows), pd.DataFrame(delta_rows)


DELTA_COL = "\u0394AUC"


def emit_latex_rows(delta_df):
    """Print the two table_external_bootstrap_delta rows ready to paste."""
    prim = delta_df[delta_df["Estimand"] == "stratified"]
    print("\n% --- paste into panel (b) of tab:extbootdelta ---")
    for _, r in prim.iterrows():
        comp = r["Comparison"].replace("\u2212", "$-$")
        point = r[DELTA_COL].lstrip("+")
        ci = r["95% CI"]
        excl = r["Excludes zero"]
        print("Pooled external cohorts & {} & {} & {} & {} \\\\".format(
            comp, point, ci, excl))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled", action="store_true",
                    help="also run the joint bootstrap across all external "
                         "cohorts and emit the pooled rows for "
                         "tab:extbootdelta")
    args = ap.parse_args()

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

    if args.pooled:
        print("\n" + "=" * 78)
        print("JOINT POOLED BOOTSTRAP — all external cohorts resampled together")
        print(f"B = {B}, seed shared across cohorts within each replicate")
        print("=" * 78)
        p_auc, p_delta = pooled_bootstrap(scored_by_seed, ext_data, RNG)
        p_auc.to_csv("results/table_external_pooled_auc.csv", index=False)
        p_delta.to_csv("results/table_external_pooled_delta.csv", index=False)
        print("\nPOOLED AUC")
        print(p_auc.to_string(index=False))
        print("\nPOOLED PAIRED \u0394AUC vs THE FEATURE-NODE GCN")
        print(p_delta.to_string(index=False))
        emit_latex_rows(p_delta)
        print("\nSaved: results/table_external_pooled_auc.csv, "
              "results/table_external_pooled_delta.csv")
        print("\nNOTE: the 'stratified' rows are the ones to report. The "
              "'concatenated' rows are diagnostic only -- they let a model "
              "profit from between-cohort prevalence differences.")


if __name__ == "__main__":
    main()
