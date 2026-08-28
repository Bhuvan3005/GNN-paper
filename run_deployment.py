# -*- coding: utf-8 -*-
"""
run_deployment.py
=================
Empirical evidence for the DEPLOYMENT pillar of the paper.

The thesis of this work is that the feature-node representation earns its
place through representation, deployability and transportability rather
than through accuracy or architectural novelty. Accuracy and transport
are already measured elsewhere (`run_main.py`, `run_external*.py`). This
module supplies the deployment evidence, which was previously asserted
rather than shown.

It produces four things:

  A. INDUCTIVE SINGLE-PATIENT INFERENCE
     Scores patients strictly one at a time, with no cohort present, and
     verifies the result is numerically identical to batch scoring. This
     is the operational property that distinguishes the feature-node
     model from a patient-similarity GNN, which cannot score an isolated
     individual at all. Per-patient latency is measured.

  B. CALIBRATION
     Brier score and Expected Calibration Error (ECE), internally
     (out-of-fold) and on each external cohort. A model offered as a
     clinical decision aid must be calibrated, not merely discriminative.

  C. NET BENEFIT / DECISION CURVE
     Net benefit across clinically plausible risk thresholds,
     NB = TP/n - (FP/n) * pt/(1-pt), against treat-all and treat-none.
     This is the standard clinical-utility analysis (TRIPOD+AI).

  D. FOOTPRINT
     Trainable parameters and serialised model size.
"""

import os
import sys
import time
import json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import brier_score_loss

import pipeline as P
from pipeline import (
    FEATURES, NUMERIC_COLS, TAU, load_data, set_seed,
    GCNConfig, TrainConfig, GCN, train_model, build_edge_index,
    make_graphs, predict_probs, best_threshold,
)

os.makedirs("results", exist_ok=True)
SCRATCH = "results"
SEED = 42
SEEDS = [42, 7, 123]
THRESHOLDS = np.arange(0.05, 0.55, 0.05)     # clinically plausible risk band


# ----------------------------------------------------------------------
# Calibration metrics
# ----------------------------------------------------------------------
def expected_calibration_error(y_true, prob, n_bins=10):
    """Standard equal-width-binning ECE."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(prob)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (prob > lo) & (prob <= hi) if i > 0 else (prob >= lo) & (prob <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y_true[m].mean() - prob[m].mean())
    return float(ece)


def net_benefit(y_true, prob, pt):
    """Net benefit of the model at risk threshold pt."""
    y_true = np.asarray(y_true, dtype=int)
    pred = (prob >= pt).astype(int)
    n = len(y_true)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return tp / n - (fp / n) * (pt / (1.0 - pt))


def net_benefit_all(y_true, pt):
    """Net benefit of the treat-all strategy."""
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    tp, fp = int(y_true.sum()), int((1 - y_true).sum())
    return tp / n - (fp / n) * (pt / (1.0 - pt))


# ----------------------------------------------------------------------
# Build one deployable Cleveland model
# ----------------------------------------------------------------------
def build_deployable(seed=SEED):
    X, y = load_data()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed)
    scaler = MinMaxScaler()
    X_tr, X_val = X_tr.copy(), X_val.copy()
    X_tr[NUMERIC_COLS] = scaler.fit_transform(X_tr[NUMERIC_COLS])
    X_val[NUMERIC_COLS] = scaler.transform(X_val[NUMERIC_COLS])

    edge_index, _ = build_edge_index(X_tr, topology="corr_mst",
                                     threshold=TAU, seed=seed)
    set_seed(seed)
    model = train_model(GCN(GCNConfig()),
                        make_graphs(X_tr, y_tr, edge_index),
                        make_graphs(X_val, y_val, edge_index),
                        TrainConfig())
    vprob, vtrue = predict_probs(model, make_graphs(X_val, y_val, edge_index))
    thr = best_threshold(vprob, vtrue)
    return model, scaler, edge_index, thr, (X_val, y_val)


# ----------------------------------------------------------------------
# A. Inductive single-patient inference
# ----------------------------------------------------------------------
def single_patient_inference(model, scaler, edge_index, X_raw):
    """Score patients one at a time with NO cohort context."""
    model.eval()
    Xs = X_raw.copy()
    Xs[NUMERIC_COLS] = scaler.transform(Xs[NUMERIC_COLS])
    vals = Xs[FEATURES].values.astype(np.float32)

    from torch_geometric.data import Data
    probs, latencies = [], []
    with torch.no_grad():
        for i in range(len(vals)):
            t0 = time.perf_counter()
            x = torch.tensor(vals[i], dtype=torch.float).view(-1, 1)
            d = Data(x=x, edge_index=edge_index)
            d.batch = torch.zeros(x.size(0), dtype=torch.long)
            p = torch.sigmoid(model(d)).view(-1).item()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            probs.append(p)
    return np.array(probs), np.array(latencies)


def main():
    print("=" * 78)
    print("DEPLOYMENT CHARACTERISATION")
    print("=" * 78)

    model, scaler, edge_index, thr, (X_val, y_val) = build_deployable()

    # ---------------- A. single-patient inductive inference -------------
    print("\n[A] Inductive single-patient inference (no cohort present)")
    probs_single, lat = single_patient_inference(model, scaler, edge_index, X_val)

    Xs = X_val.copy()
    Xs[NUMERIC_COLS] = scaler.transform(Xs[NUMERIC_COLS])
    probs_batch, _ = predict_probs(model, make_graphs(Xs, y_val, edge_index))
    max_dev = float(np.max(np.abs(probs_single - probs_batch)))

    print(f"    patients scored individually : {len(probs_single)}")
    print(f"    max |single - batch| deviation: {max_dev:.2e}")
    print(f"    latency  mean {lat.mean():.3f} ms | median {np.median(lat):.3f} ms"
          f" | p95 {np.percentile(lat,95):.3f} ms")
    print(f"    cohort graph required        : NO")

    deploy_rows = [
        ("Scores an isolated patient", "Yes (inductive)"),
        ("Cohort graph required at inference", "No"),
        ("Patients scored individually", f"{len(probs_single)}"),
        ("Max deviation vs batch scoring", f"{max_dev:.2e}"),
        ("Latency per patient, mean", f"{lat.mean():.3f} ms"),
        ("Latency per patient, median", f"{np.median(lat):.3f} ms"),
        ("Latency per patient, p95", f"{np.percentile(lat,95):.3f} ms"),
    ]

    # ---------------- D. footprint --------------------------------------
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    path = os.path.join(SCRATCH, "_deploy_model.pt")
    torch.save(model.state_dict(), path)
    size_kb = os.path.getsize(path) / 1024.0
    os.remove(path)
    deploy_rows += [
        ("Trainable parameters", f"{n_par:,}"),
        ("Serialised model size", f"{size_kb:.1f} KB"),
        ("Hardware", "CPU only"),
    ]
    print(f"\n[D] Footprint: {n_par:,} params, {size_kb:.1f} KB, CPU only")

    pd.DataFrame(deploy_rows, columns=["Property", "Value"]).to_csv(
        "results/table_deployment.csv", index=False)

    # ---------------- B. calibration ------------------------------------
    print("\n[B] Calibration (internal out-of-fold, 3 seeds)")
    from run_main import baseline_oof
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from pipeline import run_gcn_cv

    X, y = load_data()
    cal_rows = []

    gb, ge = [], []
    for sd in SEEDS:
        _, _, prob, _, true = run_gcn_cv(
            X, y, GCNConfig(), TrainConfig(), topology="corr_mst",
            threshold=TAU, n_splits=5, seed=sd, return_oof=True)
        gb.append(brier_score_loss(true, prob))
        ge.append(expected_calibration_error(true, prob))
    cal_rows.append({"Setting": "Cleveland (OOF)", "Model": "GCN (Ours)",
                     "Brier": f"{np.mean(gb):.4f} ± {np.std(gb):.4f}",
                     "ECE": f"{np.mean(ge):.4f} ± {np.std(ge):.4f}"})
    print(f"    GCN  Brier={np.mean(gb):.4f}  ECE={np.mean(ge):.4f}")

    for label, fac in [
        ("Logistic Regression",
         lambda: LogisticRegression(max_iter=1000, class_weight="balanced",
                                    random_state=SEED)),
        ("Random Forest",
         lambda: RandomForestClassifier(n_estimators=300,
                                        class_weight="balanced",
                                        random_state=SEED)),
    ]:
        bs, es = [], []
        for sd in SEEDS:
            _, _, prob, _, true = baseline_oof(X, y, fac, seed=sd)
            bs.append(brier_score_loss(true, prob))
            es.append(expected_calibration_error(true, prob))
        cal_rows.append({"Setting": "Cleveland (OOF)", "Model": label,
                         "Brier": f"{np.mean(bs):.4f} ± {np.std(bs):.4f}",
                         "ECE": f"{np.mean(es):.4f} ± {np.std(es):.4f}"})
        print(f"    {label:<20} Brier={np.mean(bs):.4f}  ECE={np.mean(es):.4f}")

    # external calibration
    print("\n    External cohorts:")
    from run_external import (
        EXT_FEATURES, EXT_CATEGORICAL, COHORTS,
        patch_pipeline_features, load_cohort, complete_cases,
    )
    from run_external_stats import fit_and_score

    saved = (list(P.FEATURES), list(P.CATEGORICAL_COLS))
    patch_pipeline_features(EXT_FEATURES, EXT_CATEGORICAL)
    dev_raw, dev_y = load_cohort("processed.cleveland.data")
    X_dev, y_dev, _ = complete_cases(dev_raw, dev_y, EXT_FEATURES)
    ext_data = {}
    for name, fn in COHORTS.items():
        d, yr = load_cohort(fn)
        Xe, ye, nc = complete_cases(d, yr, EXT_FEATURES)
        if nc > 0 and ye.nunique() >= 2:
            ext_data[name] = (Xe, ye)

    scored = [fit_and_score(X_dev, y_dev, ext_data, seed=s) for s in SEEDS]
    dca_records = []
    for cohort in ext_data:
        yt = np.asarray(scored[0][cohort]["y"]).astype(int)
        for mdl in ["GCN (Ours)", "Logistic Regression", "Random Forest"]:
            bs = [brier_score_loss(yt, s[cohort][mdl][0]) for s in scored]
            es = [expected_calibration_error(yt, s[cohort][mdl][0]) for s in scored]
            cal_rows.append({"Setting": cohort, "Model": mdl,
                             "Brier": f"{np.mean(bs):.4f} ± {np.std(bs):.4f}",
                             "ECE": f"{np.mean(es):.4f} ± {np.std(es):.4f}"})
            print(f"    {cohort:<14} {mdl:<20} Brier={np.mean(bs):.4f}  "
                  f"ECE={np.mean(es):.4f}")
            prob_mean = np.mean([s[cohort][mdl][0] for s in scored], axis=0)
            for pt in THRESHOLDS:
                dca_records.append({"Cohort": cohort, "Model": mdl,
                                    "Threshold": round(float(pt), 2),
                                    "Net benefit": round(net_benefit(yt, prob_mean, pt), 4)})
        for pt in THRESHOLDS:
            dca_records.append({"Cohort": cohort, "Model": "Treat all",
                                "Threshold": round(float(pt), 2),
                                "Net benefit": round(net_benefit_all(yt, pt), 4)})

    P.FEATURES, P.CATEGORICAL_COLS = saved[0], saved[1]
    P.N_FEATURES = len(saved[0])
    P.NUMERIC_COLS = [c for c in saved[0] if c not in saved[1]]

    pd.DataFrame(cal_rows).to_csv("results/table_calibration.csv", index=False)

    # ---------------- C. decision curve ---------------------------------
    dca = pd.DataFrame(dca_records)
    dca.to_csv("results/table_decision_curve.csv", index=False)

    piv = (dca[dca["Threshold"].isin([0.10, 0.20, 0.30, 0.40, 0.50])]
           .pivot_table(index=["Cohort", "Model"], columns="Threshold",
                        values="Net benefit"))
    piv.columns = [f"pt={c:.2f}" for c in piv.columns]
    piv.reset_index().to_csv("results/table_decision_curve_summary.csv", index=False)

    print("\n[C] Decision curve (net benefit) computed at "
          f"{len(THRESHOLDS)} thresholds per cohort")
    print(piv.round(4).to_string())

    print("\nSaved: results/table_deployment.csv, table_calibration.csv, "
          "table_decision_curve.csv, table_decision_curve_summary.csv")


if __name__ == "__main__":
    main()
