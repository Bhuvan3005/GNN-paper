# -*- coding: utf-8 -*-
"""
run_xai.py
==========
Task 3: publication-quality explainability evaluation.

Three complementary explainers are compared (GNNExplainer is KEPT, not
replaced):
  * GNNExplainer            (learned node-feature masks)
  * Integrated Gradients    (Captum, axiomatic attribution)
  * Saliency                (Captum, |gradient|)

Quantitative metrics (all in [0,1], higher = better unless noted):
  Fidelity+  comprehensiveness: prob drop when top-k features removed
  Fidelity-  sufficiency: 1 - prob drop when only top-k kept
  Sparsity   1 - (#selected / total)
  Stability  1 - mean|Δexplanation| under small input noise
  Sensitivity(↓) mean|Δexplanation| under noise  (lower = better)
  Deletion-AUC(↓) area under prob curve while deleting MoRF features
  Insertion-AUC   area under prob curve while inserting MoRF features
  Clinical-Agreement  fraction of top-k features that are clinical risk factors

Cross-method agreement:
  Top-k Jaccard and Spearman rank correlation between every method pair.
"""

import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # allow unicode metric names on Windows
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from captum.attr import IntegratedGradients, Saliency

from pipeline import (
    FEATURES, N_FEATURES, NUMERIC_COLS, TAU, CLINICAL_TOP, DEVICE,
    load_data, set_seed, GCNConfig, TrainConfig, GCN, train_model,
    make_graphs, build_edge_index, best_threshold, predict_probs,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
SEED = 42
TOP_K = 5
N_SAMPLES = 40


# ----------------------------------------------------------------------
# Prob wrapper: forward(x) -> P(disease) for a single fixed-topology graph
# ----------------------------------------------------------------------
class ProbWrapper(torch.nn.Module):
    def __init__(self, model, edge_index):
        super().__init__()
        self.model = model
        self.edge_index = edge_index

    def forward(self, x, edge_index=None):
        # GNNExplainer calls forward(x, edge_index); Captum calls forward(x).
        ei = edge_index if edge_index is not None else self.edge_index
        data = Data(x=x, edge_index=ei)
        data.batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return torch.sigmoid(self.model(data)).view(-1, 1)


def _prob(model, x, edge_index):
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    with torch.no_grad():
        return float(torch.sigmoid(model(data)).view(-1).item())


# ----------------------------------------------------------------------
# Explainers  -> normalized non-negative score vector of length 13
# ----------------------------------------------------------------------
def explain_gnn(model, graph, edge_index):
    wrapper = ProbWrapper(model, edge_index).to(DEVICE).eval()
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification",
                          task_level="graph", return_type="probs"),
    )
    expl = explainer(graph.x, edge_index)
    s = np.abs(expl.node_mask.detach().cpu().numpy()).flatten()
    return s / (s.sum() + 1e-12)


def explain_ig(model, graph, edge_index):
    wrapper = ProbWrapper(model, edge_index).to(DEVICE).eval()
    ig = IntegratedGradients(wrapper)
    x = graph.x.clone().requires_grad_(True)
    attr = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=64)
    s = np.abs(attr.detach().cpu().numpy()).flatten()
    return s / (s.sum() + 1e-12)


def explain_saliency(model, graph, edge_index):
    wrapper = ProbWrapper(model, edge_index).to(DEVICE).eval()
    sal = Saliency(wrapper)
    x = graph.x.clone().requires_grad_(True)
    attr = sal.attribute(x)
    s = np.abs(attr.detach().cpu().numpy()).flatten()
    return s / (s.sum() + 1e-12)


EXPLAINERS = {
    "GNNExplainer": explain_gnn,
    "Integrated Gradients": explain_ig,
    "Saliency": explain_saliency,
}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def fidelity_plus(model, graph, scores, edge_index, k=TOP_K):
    """Prob drop when the top-k features are removed (masked to 0)."""
    base = _prob(model, graph.x, edge_index)
    top = np.argsort(scores)[-k:]
    x = graph.x.clone(); x[top] = 0.0
    return abs(base - _prob(model, x, edge_index))


def fidelity_minus(model, graph, scores, edge_index, k=TOP_K):
    """Sufficiency: 1 - prob drop when ONLY the top-k are kept."""
    base = _prob(model, graph.x, edge_index)
    top = set(np.argsort(scores)[-k:].tolist())
    x = graph.x.clone()
    for i in range(N_FEATURES):
        if i not in top:
            x[i] = 0.0
    return 1.0 - abs(base - _prob(model, x, edge_index))


def sparsity(scores):
    sel = int(np.sum(scores > scores.mean()))
    return 1.0 - sel / N_FEATURES


def stability_and_sensitivity(explain_fn, model, graph, edge_index,
                              trials=5, noise=0.01):
    base = explain_fn(model, graph, edge_index)
    devs = []
    for _ in range(trials):
        g = graph.clone()
        g.x = graph.x + torch.randn_like(graph.x) * noise
        devs.append(np.mean(np.abs(explain_fn(model, g, edge_index) - base)))
    sens = float(np.mean(devs))
    return max(0.0, 1.0 - sens), sens


def deletion_insertion(model, graph, scores, edge_index):
    """MoRF deletion & insertion curves; return (del_auc, ins_auc)."""
    order = np.argsort(scores)[::-1]                 # most relevant first
    # deletion: start from full input, remove features one by one
    x = graph.x.clone()
    del_curve = [_prob(model, x, edge_index)]
    for idx in order:
        x = x.clone(); x[idx] = 0.0
        del_curve.append(_prob(model, x, edge_index))
    # insertion: start from empty, add features one by one
    x = torch.zeros_like(graph.x)
    ins_curve = [_prob(model, x, edge_index)]
    for idx in order:
        x = x.clone(); x[idx] = graph.x[idx]
        ins_curve.append(_prob(model, x, edge_index))
    xs = np.linspace(0, 1, len(del_curve))
    return float(np.trapz(del_curve, xs)), float(np.trapz(ins_curve, xs)), \
        np.array(del_curve), np.array(ins_curve)


def clinical_agreement(scores, k=TOP_K):
    top = [FEATURES[i] for i in np.argsort(scores)[-k:]]
    return len(set(top) & set(CLINICAL_TOP)) / k


# ----------------------------------------------------------------------
# Build a final full model on a stratified 70/15/15 split
# ----------------------------------------------------------------------
def build_final_model():
    X, y = load_data()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    scaler = MinMaxScaler()
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    X_tr[NUMERIC_COLS] = scaler.fit_transform(X_tr[NUMERIC_COLS])
    X_val[NUMERIC_COLS] = scaler.transform(X_val[NUMERIC_COLS])
    X_te[NUMERIC_COLS] = scaler.transform(X_te[NUMERIC_COLS])

    edge_index, _ = build_edge_index(X_tr, topology="corr_mst", threshold=TAU, seed=SEED)
    g_tr = make_graphs(X_tr, y_tr, edge_index)
    g_val = make_graphs(X_val, y_val, edge_index)
    g_te = make_graphs(X_te, y_te, edge_index)

    set_seed(SEED)
    model = GCN(GCNConfig())
    model = train_model(model, g_tr, g_val, TrainConfig())
    return model, g_te, edge_index


def main():
    model, test_graphs, edge_index = build_final_model()
    disease = [g for g in test_graphs if int(g.y.item()) == 1]
    samples = (disease + [g for g in test_graphs if int(g.y.item()) == 0])[:N_SAMPLES]
    print(f"XAI on {len(samples)} test patients "
          f"({len(disease)} disease-positive available)")
    print("=" * 78)

    rows = {m: {k: [] for k in
                ["Fidelity+", "Fidelity-", "Sparsity", "Stability",
                 "Sensitivity(↓)", "Deletion-AUC(↓)", "Insertion-AUC",
                 "Clinical-Agreement"]}
            for m in EXPLAINERS}
    score_store = {m: [] for m in EXPLAINERS}
    del_curves = {m: [] for m in EXPLAINERS}
    ins_curves = {m: [] for m in EXPLAINERS}

    for i, g in enumerate(samples):
        g = g.to(DEVICE)
        for mname, fn in EXPLAINERS.items():
            try:
                s = fn(model, g, edge_index)
            except Exception as e:
                print(f"  [{mname}] sample {i} failed: {e}")
                continue
            score_store[mname].append(s)
            rows[mname]["Fidelity+"].append(fidelity_plus(model, g, s, edge_index))
            rows[mname]["Fidelity-"].append(fidelity_minus(model, g, s, edge_index))
            rows[mname]["Sparsity"].append(sparsity(s))
            stab, sens = stability_and_sensitivity(fn, model, g, edge_index)
            rows[mname]["Stability"].append(stab)
            rows[mname]["Sensitivity(↓)"].append(sens)
            dauc, iauc, dc, ic = deletion_insertion(model, g, s, edge_index)
            rows[mname]["Deletion-AUC(↓)"].append(dauc)
            rows[mname]["Insertion-AUC"].append(iauc)
            del_curves[mname].append(dc)
            ins_curves[mname].append(ic)
            rows[mname]["Clinical-Agreement"].append(clinical_agreement(s))
        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(samples)}")

    # ---- metric table ----
    metric_names = list(next(iter(rows.values())).keys())
    table = pd.DataFrame(
        {"Metric": metric_names,
         **{m: [round(float(np.mean(rows[m][k])), 4) for k in metric_names]
            for m in EXPLAINERS}})
    table.to_csv("results/table_xai_metrics.csv", index=False)
    print("\nXAI METRIC COMPARISON (mean over samples)")
    print(table.to_string(index=False))

    # ---- cross-method agreement ----
    agr_rows = []
    names = list(EXPLAINERS)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            m1, m2 = names[a], names[b]
            jac, rho = [], []
            for s1, s2 in zip(score_store[m1], score_store[m2]):
                t1 = set(np.argsort(s1)[-TOP_K:]); t2 = set(np.argsort(s2)[-TOP_K:])
                jac.append(len(t1 & t2) / len(t1 | t2))
                rho.append(spearmanr(s1, s2).correlation)
            agr_rows.append({"Method pair": f"{m1} vs {m2}",
                             "Top-k Jaccard": round(float(np.mean(jac)), 4),
                             "Spearman rank corr": round(float(np.nanmean(rho)), 4)})
    agr = pd.DataFrame(agr_rows)
    agr.to_csv("results/table_xai_agreement.csv", index=False)
    print("\nCROSS-METHOD AGREEMENT")
    print(agr.to_string(index=False))

    # ---- deletion / insertion figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"GNNExplainer": "#2ECC71", "Integrated Gradients": "#3498DB",
              "Saliency": "#E74C3C"}
    for m in EXPLAINERS:
        if not del_curves[m]:
            continue
        dc = np.mean(np.stack(del_curves[m]), axis=0)
        ic = np.mean(np.stack(ins_curves[m]), axis=0)
        xs = np.linspace(0, 1, len(dc))
        axes[0].plot(xs, dc, "o-", ms=3, color=colors[m], label=m)
        axes[1].plot(xs, ic, "o-", ms=3, color=colors[m], label=m)
    axes[0].set_title("Deletion curve (MoRF)  — lower is better")
    axes[1].set_title("Insertion curve (MoRF) — higher is better")
    for ax in axes:
        ax.set_xlabel("Fraction of features perturbed")
        ax.set_ylabel("P(disease)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/xai_deletion_insertion.png", dpi=150, bbox_inches="tight")
    print("\nSaved: results/table_xai_metrics.csv, results/table_xai_agreement.csv, "
          "figures/xai_deletion_insertion.png")


if __name__ == "__main__":
    main()
