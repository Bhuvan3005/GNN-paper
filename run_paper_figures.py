# -*- coding: utf-8 -*-
"""
run_paper_figures.py
=====================
Generates every figure referenced by paper/main.tex that does not yet
exist in figures/. Reuses already-computed result CSVs (ablation,
threshold sensitivity, XAI metrics, OOF predictions) wherever possible;
retrains a single final model only where a genuinely new artifact is
needed (loss curve, embeddings, per-feature attribution vectors).
"""

import os
import re
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

from pipeline import (
    FEATURES, N_FEATURES, NUMERIC_COLS, CLINICAL_TOP, TAU, DEVICE,
    load_data, set_seed, GCNConfig, TrainConfig, GCN, train_model,
    make_graphs, build_edge_index, best_threshold, predict_probs,
    _corr_mst_edge_set,
)

os.makedirs("figures", exist_ok=True)
SEED = 42
PALETTE = {"GCN (Ours)": "#2ECC71", "Logistic Regression": "#3498DB",
          "Random Forest": "#9B59B6", "Gradient Boosting": "#F39C12",
          "MLP": "#E74C3C", "XGBoost": "#16A085", "LightGBM": "#D35400"}
MODEL_ORDER = ["GCN (Ours)", "Logistic Regression", "Random Forest",
              "Gradient Boosting", "MLP", "XGBoost", "LightGBM"]


def parse_ms(s):
    """'0.8109 ± 0.0652' or '0.8109 +/- 0.0652' -> (mean, std)."""
    m = re.match(r"\s*([\-\d.]+)\s*(?:\xb1|\+/-)\s*([\-\d.]+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (float(s), 0.0)


# ======================================================================
# 1. Schematic figures (no data dependency)
# ======================================================================
def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.2))
    steps = ["Raw\nCleveland\nCSV", "In-fold\nMin-Max\nscaling",
            "Pearson\ncorrelation\n(train only)", "Threshold\n$\\tau=0.15$\n+ MST",
            "Patient\ngraphs\n(13 nodes)", "2-layer\nGCN\ntraining",
            "Inner-val\nthreshold\ntuning", "Held-out\nfold\nevaluation"]
    n = len(steps)
    xs = np.linspace(0.5, n - 0.5, n)
    for x, s in zip(xs, steps):
        box = FancyBboxPatch((x - 0.42, 0.3), 0.84, 0.5,
                             boxstyle="round,pad=0.02", linewidth=1.3,
                             edgecolor="#2C3E50", facecolor="#EAF2F8")
        ax.add_patch(box)
        ax.text(x, 0.55, s, ha="center", va="center", fontsize=8.3)
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + 0.42, 0.55), (xs[i + 1] - 0.42, 0.55),
                                     arrowstyle="-|>", mutation_scale=14, color="#2C3E50"))
    ax.annotate("all steps to the left of the dashed line are refit inside\n"
               "every cross-validation fold to prevent leakage",
               xy=(xs[3] + 0.1, 0.28), xytext=(xs[3] - 1.5, -0.15),
               fontsize=8, color="#7B241C", ha="left")
    ax.axvline(xs[6] - 0.5, ls="--", color="#B03A2E", lw=1, ymin=0.05, ymax=0.95)
    ax.set_xlim(0, n); ax.set_ylim(-0.35, 1.0); ax.axis("off")
    plt.tight_layout()
    plt.savefig("figures/fig01_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close()


def fig07_architecture():
    fig, ax = plt.subplots(figsize=(13, 3.0))
    blocks = ["Input\n$x\\in\\mathbb{R}^{13\\times1}$", "GCNConv\n(1$\\to$32)",
             "BatchNorm\n+ ReLU", "Dropout\n0.30", "GCNConv\n(32$\\to$32)",
             "BatchNorm\n+ ReLU", "Global\nMean Pool", "Linear\n(32$\\to$1)",
             "Sigmoid\n$\\hat{y}$"]
    n = len(blocks)
    xs = np.linspace(0.5, n - 0.5, n)
    colors = ["#D5DBDB", "#AED6F1", "#A9DFBF", "#F9E79F", "#AED6F1",
             "#A9DFBF", "#F5B7B1", "#D2B4DE", "#D5DBDB"]
    for x, s, c in zip(xs, blocks, colors):
        box = FancyBboxPatch((x - 0.42, 0.3), 0.84, 0.5,
                             boxstyle="round,pad=0.02", linewidth=1.2,
                             edgecolor="#2C3E50", facecolor=c)
        ax.add_patch(box)
        ax.text(x, 0.55, s, ha="center", va="center", fontsize=8.3)
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + 0.42, 0.55), (xs[i + 1] - 0.42, 0.55),
                                     arrowstyle="-|>", mutation_scale=14, color="#2C3E50"))
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.axis("off")
    plt.tight_layout()
    plt.savefig("figures/fig07_architecture.png", dpi=220, bbox_inches="tight")
    plt.close()


# ======================================================================
# 2. Dataset / correlation / graph figures
# ======================================================================
def data_and_graph():
    X, y = load_data()
    Xs = X.copy()
    sc = MinMaxScaler()
    Xs[NUMERIC_COLS] = sc.fit_transform(Xs[NUMERIC_COLS])
    corr = Xs[FEATURES].corr().values
    G = _corr_mst_edge_set(corr, TAU)
    G = nx.relabel_nodes(G, {i: FEATURES[i] for i in range(N_FEATURES)})
    return X, y, Xs, corr, G


def fig02_dataset_distribution(X, y):
    fig, axes = plt.subplots(3, 5, figsize=(18, 9))
    axes = axes.flatten()
    for i, f in enumerate(FEATURES):
        ax = axes[i]
        for cls, color, label in [(0, "#3498DB", "No disease"), (1, "#E74C3C", "Disease")]:
            vals = X.loc[y == cls, f]
            ax.hist(vals, bins=15, alpha=0.55, color=color, label=label, density=True)
        ax.set_title(f, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(FEATURES), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig("figures/fig02_dataset_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig03_correlation_heatmap(corr):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(N_FEATURES)); ax.set_yticks(range(N_FEATURES))
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(FEATURES, fontsize=8)
    for i in range(N_FEATURES):
        for j in range(N_FEATURES):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson $r$")
    ax.set_title("Feature Correlation Matrix (Scaled Training Data)", fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/fig03_correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()


def fig04_feature_graph(G, corr):
    pos = nx.spring_layout(G, seed=SEED, k=1.1)
    fig, ax = plt.subplots(figsize=(8, 7))
    idx = {f: i for i, f in enumerate(FEATURES)}
    for u, v, d in G.edges(data=True):
        r = corr[idx[u], idx[v]]
        color = "#C0392B" if r > 0 else "#2980B9"
        ax.plot(*zip(pos[u], pos[v]), color=color, alpha=0.6, lw=1 + 3 * abs(r))
    degs = dict(G.degree())
    sizes = [400 + degs[n] * 140 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#5DADE2",
                           edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
    legend = [mpatches.Patch(color="#C0392B", label="Positive correlation"),
             mpatches.Patch(color="#2980B9", label="Negative correlation")]
    ax.legend(handles=legend, fontsize=9, loc="lower left")
    ax.set_title(f"Canonical Feature Graph ($\\tau={TAU}$, {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges)", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("figures/fig04_feature_graph.png", dpi=180, bbox_inches="tight")
    plt.close()


def fig05_spectral(G):
    Gp = G.copy()
    for u, v, d in Gp.edges(data=True):
        d["weight"] = 1.0
    L = nx.laplacian_matrix(Gp).toarray().astype(float)
    eig = np.sort(np.linalg.eigvalsh(L))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(eig)), eig, color="#5DADE2", edgecolor="black")
    ax.axhline(eig[1], color="#E74C3C", ls="--", label=f"Fiedler $\\lambda_2$={eig[1]:.3f}")
    ax.set_xlabel("Eigenvalue index"); ax.set_ylabel("Eigenvalue")
    ax.set_title("Laplacian Spectrum of the Feature Graph", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig05_spectral.png", dpi=180, bbox_inches="tight")
    plt.close()


def fig06_degree_centrality(G):
    n = G.number_of_nodes()
    cent = {v: G.degree(v) / (n - 1) for v in G.nodes()}
    items = sorted(cent.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#58D68D", edgecolor="black")
    ax.set_xlabel("Degree centrality")
    ax.set_title("Feature Degree Centrality", fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/fig06_degree_centrality.png", dpi=180, bbox_inches="tight")
    plt.close()


# ======================================================================
# 3. Retrain final model once: loss curve, embeddings, XAI arrays
# ======================================================================
def build_final_model_with_history():
    X, y = load_data()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    sc = MinMaxScaler()
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    X_tr[NUMERIC_COLS] = sc.fit_transform(X_tr[NUMERIC_COLS])
    X_val[NUMERIC_COLS] = sc.transform(X_val[NUMERIC_COLS])
    X_te[NUMERIC_COLS] = sc.transform(X_te[NUMERIC_COLS])
    edge_index, _ = build_edge_index(X_tr, topology="corr_mst", threshold=TAU, seed=SEED)
    g_tr = make_graphs(X_tr, y_tr, edge_index)
    g_val = make_graphs(X_val, y_val, edge_index)
    g_te = make_graphs(X_te, y_te, edge_index)

    set_seed(SEED)
    model = GCN(GCNConfig()).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    from torch_geometric.loader import DataLoader
    loader = DataLoader(g_tr, batch_size=32, shuffle=True)
    vloader = DataLoader(g_val, batch_size=256, shuffle=False)

    train_hist, val_hist = [], []
    best_val, best_wts, no_improve, stop_epoch = float("inf"), None, 0, None
    for epoch in range(150):
        model.train()
        tloss, tn = 0.0, 0
        for batch in loader:
            opt.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tloss += loss.item() * batch.num_graphs; tn += batch.num_graphs
        train_hist.append(tloss / tn)

        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in vloader:
                vloss += criterion(model(batch), batch.y.view(-1)).item() * batch.num_graphs
                vn += batch.num_graphs
        vloss /= vn
        val_hist.append(vloss)
        if vloss < best_val - 1e-5:
            best_val, best_wts, no_improve = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
        if no_improve >= 25:
            stop_epoch = epoch
            break
    if best_wts:
        model.load_state_dict(best_wts)
    return model, g_tr, g_val, g_te, train_hist, val_hist, stop_epoch, edge_index


def fig08_loss(train_hist, val_hist, stop_epoch):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train_hist, label="Train loss", color="#3498DB")
    ax.plot(val_hist, label="Inner-validation loss", color="#E74C3C")
    if stop_epoch is not None:
        ax.axvline(stop_epoch - 25, color="gray", ls="--", label="Early-stopping epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
    ax.set_title("Training Curve of the Final Model", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig08_loss.png", dpi=180, bbox_inches="tight")
    plt.close()


def get_embeddings(model, graphs):
    """Pooled graph embedding (before the final linear layer)."""
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for g in graphs:
            x, ei = g.x, g.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long)
            for k, conv in enumerate(model.convs):
                x = conv(x, ei)
                x = model.bns[k](x)
                x = F.relu(x)
            pooled = model.pool(x, batch)
            embs.append(pooled.numpy().flatten())
            labels.append(int(g.y.item()))
    return np.array(embs), np.array(labels)


def fig19_embeddings(model, test_graphs):
    embs, labels = get_embeddings(model, test_graphs)
    pca = PCA(n_components=2, random_state=SEED).fit_transform(embs)
    perp = min(30, max(5, len(embs) // 3))
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perp, init="pca").fit_transform(embs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, proj, title in [(axes[0], pca, "PCA"), (axes[1], tsne, "t-SNE")]:
        for cls, color, lab in [(0, "#3498DB", "No disease"), (1, "#E74C3C", "Disease")]:
            m = labels == cls
            ax.scatter(proj[m, 0], proj[m, 1], c=color, label=lab, alpha=0.75, edgecolor="k", s=40)
        ax.set_title(f"{title} of GCN Graph Embeddings", fontsize=11)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig19_embeddings.png", dpi=160, bbox_inches="tight")
    plt.close()


# ======================================================================
# 4. Performance / OOF-derived figures
# ======================================================================
def fig13_performance_bars():
    df = pd.read_csv("results/table_model_comparison.csv")
    metrics = ["Accuracy", "F1", "ROC-AUC", "MCC"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df)); width = 0.2
    for i, m in enumerate(metrics):
        means, stds = zip(*[parse_ms(v) for v in df[m]])
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=m)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["Model"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison (5-fold CV x 3 seeds, mean $\\pm$ std)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig13_performance_bars.png", dpi=170, bbox_inches="tight")
    plt.close()


def fig10_roc_pr():
    npz = np.load("results/oof_predictions.npz")
    true = npz["true"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name in MODEL_ORDER:
        prob = npz[f"{name}_prob"]
        fpr, tpr, _ = roc_curve(true, prob)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=PALETTE[name])
        prec, rec, _ = precision_recall_curve(true, prob)
        axes[1].plot(rec, prec, label=name, color=PALETTE[name])
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Pooled Out-of-Fold ROC Curves", fontsize=11); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Pooled Out-of-Fold Precision-Recall Curves", fontsize=11); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/fig10_roc_pr.png", dpi=160, bbox_inches="tight")
    plt.close()


def fig12_confusion():
    npz = np.load("results/oof_predictions.npz")
    true, pred = npz["true"], npz["GCN (Ours)_pred"]
    cm = confusion_matrix(true, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No disease", "Disease"]); ax.set_yticklabels(["No disease", "Disease"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("GCN Confusion Matrix (pooled OOF)", fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/fig12_confusion.png", dpi=170, bbox_inches="tight")
    plt.close()


def fig28_calibration_probdist():
    npz = np.load("results/oof_predictions.npz")
    true, prob = npz["true"], npz["GCN (Ours)_prob"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    frac_pos, mean_pred = calibration_curve(true, prob, n_bins=8)
    axes[0].plot(mean_pred, frac_pos, "o-", color="#2ECC71")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("Mean predicted probability"); axes[0].set_ylabel("Observed frequency")
    axes[0].set_title("Calibration Curve (GCN)", fontsize=11)
    axes[1].hist(prob[true == 0], bins=20, alpha=0.6, color="#3498DB", label="No disease")
    axes[1].hist(prob[true == 1], bins=20, alpha=0.6, color="#E74C3C", label="Disease")
    axes[1].set_xlabel("Predicted P(disease)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Predicted Probability Distribution by Class", fontsize=11)
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig28_calibration_probdist.png", dpi=160, bbox_inches="tight")
    plt.close()


def fig30_error_analysis():
    npz = np.load("results/oof_predictions.npz")
    true, pred, prob = npz["true"], npz["GCN (Ours)_pred"], npz["GCN (Ours)_prob"]
    correct = pred == true
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(prob[correct], bins=20, alpha=0.65, color="#2ECC71", label="Correct")
    ax.hist(prob[~correct], bins=20, alpha=0.65, color="#E74C3C", label="Misclassified")
    ax.axvline(0.5, color="gray", ls="--")
    ax.set_xlabel("Predicted P(disease)"); ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence: Correct vs. Misclassified", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig30_error_analysis.png", dpi=170, bbox_inches="tight")
    plt.close()


# ======================================================================
# 5. Ablation / threshold figures (parsed from existing CSVs)
# ======================================================================
def _bar_from_ablation(rows_df, fname, title):
    metrics = ["F1", "ROC-AUC", "MCC"]
    fig, ax = plt.subplots(figsize=(max(8, len(rows_df) * 1.1), 5.5))
    x = np.arange(len(rows_df)); width = 0.25
    for i, m in enumerate(metrics):
        means, stds = zip(*[parse_ms(v) for v in rows_df[m]])
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=m)
    ax.set_xticks(x + width); ax.set_xticklabels(rows_df["Experiment"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=160, bbox_inches="tight")
    plt.close()


def fig17_18_20_21_22_ablation():
    df = pd.read_csv("results/table_ablation.csv")
    _bar_from_ablation(df, "fig17_ablation.png", "Complete Ablation Study")

    topo = df[df["Experiment"].isin(["Full (Corr+MST)", "w/o MST (Corr only)",
                                     "Random graph", "Fully connected", "No graph (indep. nodes)"])]
    _bar_from_ablation(topo, "fig18_topology.png", "Graph-Topology Ablation")

    hid = df[df["Experiment"].isin(["Full (Corr+MST)", "Hidden 16", "Hidden 64"])]
    _bar_from_ablation(hid, "fig20_hidden.png", "Hidden-Dimension Sensitivity")

    drop = df[df["Experiment"].isin(["Full (Corr+MST)", "Dropout 0.0", "Dropout 0.5"])]
    _bar_from_ablation(drop, "fig21_dropout.png", "Dropout Sensitivity")

    pool = df[df["Experiment"].isin(["Full (Corr+MST)", "Max pooling", "Add pooling"])]
    _bar_from_ablation(pool, "fig22_pooling.png", "Readout-Pooling Sensitivity")


def fig14_threshold():
    df = pd.read_csv("results/table_threshold_sensitivity.csv")
    tau_col = "tau" if "tau" in df.columns else df.columns[0]
    f1_m, f1_s = zip(*[parse_ms(v) for v in df["F1"]])
    auc_m, auc_s = zip(*[parse_ms(v) for v in df["ROC-AUC"]])
    mcc_m, mcc_s = zip(*[parse_ms(v) for v in df["MCC"]])
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax2.bar(df[tau_col], df["Mean edges"], width=0.02, alpha=0.25, color="gray", label="Mean edges")
    ax1.errorbar(df[tau_col], f1_m, yerr=f1_s, marker="o", label="F1", color="#3498DB")
    ax1.errorbar(df[tau_col], auc_m, yerr=auc_s, marker="s", label="ROC-AUC", color="#2ECC71")
    ax1.errorbar(df[tau_col], mcc_m, yerr=mcc_s, marker="^", label="MCC", color="#E74C3C")
    ax1.axvline(0.15, color="black", ls="--", alpha=0.5, label="$\\tau=0.15$ (selected)")
    ax1.set_xlabel("Correlation threshold $\\tau$"); ax1.set_ylabel("Score")
    ax2.set_ylabel("Mean edge count")
    ax1.set_title("Threshold Sensitivity Analysis", fontsize=11)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, fontsize=8, loc="lower center")
    plt.tight_layout()
    plt.savefig("figures/fig14_threshold.png", dpi=170, bbox_inches="tight")
    plt.close()


# ======================================================================
# 6. XAI figures (need per-feature attribution vectors -> recompute)
# ======================================================================
def xai_figures(model, test_graphs, edge_index):
    from run_xai import EXPLAINERS, explain_gnn, explain_ig, explain_saliency
    samples = test_graphs[:40]
    all_scores = {m: [] for m in EXPLAINERS}
    for g in samples:
        for name, fn in EXPLAINERS.items():
            try:
                all_scores[name].append(fn(model, g, edge_index))
            except Exception:
                pass

    # fig23: mean feature importance per method
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(N_FEATURES); width = 0.25
    for i, (name, color) in enumerate(zip(EXPLAINERS, ["#2ECC71", "#3498DB", "#E74C3C"])):
        mean_s = np.mean(np.stack(all_scores[name]), axis=0)
        ax.bar(x + i * width, mean_s, width, label=name, color=color, alpha=0.85)
    ax.set_xticks(x + width); ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean normalised attribution")
    ax.set_title("Mean Feature Attribution by Method (n=40 test patients)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig23_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    # fig25: single disease-positive patient explanation
    disease = [g for g in samples if int(g.y.item()) == 1]
    g0 = disease[0] if disease else samples[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name, color in zip(axes, EXPLAINERS, ["#2ECC71", "#3498DB", "#E74C3C"]):
        s = EXPLAINERS[name](model, g0, edge_index)
        order = np.argsort(s)
        ax.barh(np.array(FEATURES)[order], s[order], color=color, alpha=0.85)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Attribution")
    plt.suptitle("Single-Patient Attribution (Disease-Positive)", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/fig25_example_explanations.png", dpi=160, bbox_inches="tight")
    plt.close()

    # fig27: cross-method agreement heatmap (Spearman)
    names = list(EXPLAINERS)
    mat = np.eye(len(names))
    for a in range(len(names)):
        for b in range(len(names)):
            if a == b:
                continue
            rhos = [spearmanr(s1, s2).correlation
                   for s1, s2 in zip(all_scores[names[a]], all_scores[names[b]])]
            mat[a, b] = np.nanmean(rhos)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman $\\rho$")
    ax.set_title("Cross-Method Explanation Agreement", fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/fig27_agreement_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close()


def fig29_radar():
    df = pd.read_csv("results/table_xai_metrics.csv")
    labels = df["Metric"].tolist()
    methods = [c for c in df.columns if c != "Metric"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = ["#2ECC71", "#3498DB", "#E74C3C"]
    for m, c in zip(methods, colors):
        vals = df[m].tolist(); vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, color=c, label=m)
        ax.fill(angles, vals, alpha=0.12, color=c)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("XAI Metric Radar", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig29_radar.png", dpi=160, bbox_inches="tight")
    plt.close()


def main():
    print("Schematic figures...")
    fig01_pipeline(); fig07_architecture()

    print("Dataset / graph figures...")
    X, y, Xs, corr, G = data_and_graph()
    fig02_dataset_distribution(X, y)
    fig03_correlation_heatmap(corr)
    fig04_feature_graph(G, corr)
    fig05_spectral(G)
    fig06_degree_centrality(G)

    print("Performance / OOF figures...")
    fig13_performance_bars()
    fig10_roc_pr()
    fig12_confusion()
    fig28_calibration_probdist()
    fig30_error_analysis()

    print("Ablation / sensitivity figures...")
    fig17_18_20_21_22_ablation()
    fig14_threshold()

    print("Retraining final model for loss curve / embeddings / XAI figures...")
    model, g_tr, g_val, g_te, train_hist, val_hist, stop_epoch, edge_index = build_final_model_with_history()
    fig08_loss(train_hist, val_hist, stop_epoch)
    fig19_embeddings(model, g_te)

    print("XAI figures (this recomputes attributions on 40 patients)...")
    xai_figures(model, g_te, edge_index)
    fig29_radar()

    print("\nAll figures written to figures/")


if __name__ == "__main__":
    main()
