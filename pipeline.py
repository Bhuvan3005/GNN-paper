# -*- coding: utf-8 -*-
"""
pipeline.py
============
Reproducible core for the feature-node GCN heart-disease study
(UCI Cleveland). This module encodes the exact methodology of the
paper and is imported by the stage runners:

    run_main.py      -> CV of the full model + fair baselines + stats
    run_ablation.py  -> redesigned ablation study
    run_xai.py       -> multi-method XAI evaluation

Methodology preserved (NOT redesigned):
  * Feature-node graph: 13 nodes = 13 clinical features, 1 scalar / node.
  * Pearson correlation graph, fixed threshold tau = 0.15.
  * MST augmentation over distance = 1 - |r| to guarantee connectivity.
  * Fold-specific graph construction: correlation + MST are computed on
    the TRAINING portion of each fold only (no leakage).
  * Two-layer GCN, global mean pooling, sigmoid head.
  * 5-fold Stratified Cross-Validation.
  * Validation-based decision-threshold optimization (inner split).

The only structural change vs. the original Colab export is that the
missing training core is implemented here so results come from genuine
held-out folds instead of the training set.
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx

import torch
# CRITICAL for speed: the feature graphs are tiny (~11 nodes), so PyTorch's
# default intra-op parallelism (one thread per core, 18 here) spends far more
# time dispatching threads than computing. Capping threads avoids that
# oversubscription and makes each training ~500x faster on this workload.
torch.set_num_threads(min(4, os.cpu_count() or 4))
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, global_mean_pool, global_max_pool, global_add_pool,
)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Dataset schema — fixed, per the mandated methodology (NOT data-driven).
# UCI Cleveland only: 303 patients, 13 clinical features.
# ----------------------------------------------------------------------
CSV_PATH = "Heart_disease_cleveland_new.csv"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]
N_FEATURES = len(FEATURES)                       # 13

# Preserve the original design: MinMax-scale only the continuous columns;
# leave categorical / ordinal columns as raw integers.
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "slope", "ca", "thal", "exang"]
NUMERIC_COLS = [c for c in FEATURES if c not in CATEGORICAL_COLS]  # age,trestbps,chol,thalach,oldpeak

TAU = 0.15                                        # correlation threshold (fixed)
DEVICE = torch.device("cpu")                      # CPU for full reproducibility

# Clinically established high-risk factors for Cleveland, used by the
# clinical-agreement XAI metric (ref: Detrano et al. 1989; standard
# cardiology risk factors). Documented so the metric is not circular.
CLINICAL_TOP = ["cp", "thalach", "exang", "oldpeak", "ca", "thal"]


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
# Weighted scatter-aggregation over the dense learned topology parallelises its
# float summation, which makes the forward pass non-reproducible run-to-run
# (verified: identical seeds, differing predictions). Deterministic kernels cost
# a little speed and make every reported number exactly reproducible.
torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_data(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)
    X = df[FEATURES].copy()
    y = df["target"].astype(int).copy()
    return X, y


# ----------------------------------------------------------------------
# Fold-specific graph construction
# ----------------------------------------------------------------------
def _complete_corr_graph(corr: np.ndarray):
    """Complete graph weighted by distance = 1 - |r| (input to the MST)."""
    n = corr.shape[0]
    Gc = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            Gc.add_edge(i, j, weight=1.0 - abs(corr[i, j]))
    return Gc


def _corr_mst_edge_set(corr: np.ndarray, threshold: float):
    """Return the undirected edge set of the (correlation + MST) graph."""
    n = corr.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= threshold:
                G.add_edge(i, j)
    # MST bridges over the complete graph, distance = 1 - |r|
    mst = nx.minimum_spanning_tree(_complete_corr_graph(corr))
    for u, v in mst.edges():
        if not G.has_edge(u, v):
            G.add_edge(u, v)
    return G


def build_edge_index(
    X_train_scaled: pd.DataFrame,
    topology: str = "corr_mst",
    threshold: float = TAU,
    seed: int = 0,
):
    """
    Build a shared (within-fold) edge_index from the fold's TRAINING data.

    topology:
      'corr_mst'         -> correlation (|r|>=tau) + MST bridges  (full model)
      'corr_only'        -> correlation edges only (no MST bridges)
      'random'           -> random graph with the same #edges as corr_mst
      'fully_connected'  -> complete graph on 13 nodes
      'none'             -> no edges (GCNConv self-loops => independent nodes)
      'knn<k>'           -> each feature linked to its k most-correlated
                            features (adaptive per-node sparsity instead of a
                            single global cut-off); undirected union
      'knn<k>_mst'       -> the above plus MST bridges, matching the full
                            model's connectivity guarantee
    """
    corr = X_train_scaled[FEATURES].corr().values
    n = N_FEATURES

    if topology in ("corr_mst", "corr_only"):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) >= threshold:
                    G.add_edge(i, j)
        if topology == "corr_mst":
            G = _corr_mst_edge_set(corr, threshold)

    elif topology == "fully_connected":
        G = nx.complete_graph(n)

    elif topology == "random":
        target = _corr_mst_edge_set(corr, threshold).number_of_edges()
        rng = np.random.default_rng(seed)
        possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
        idx = rng.choice(len(possible), size=min(target, len(possible)), replace=False)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for k in idx:
            G.add_edge(*possible[k])

    elif topology == "none":
        G = nx.Graph()
        G.add_nodes_from(range(n))

    elif topology.startswith("knn"):
        # 'knn3' / 'knn3_mst': connect every feature to its k strongest
        # |Pearson| partners. Unlike the global threshold tau, this fixes a
        # MINIMUM degree per node, so weakly-correlated features are never
        # left isolated by the cut-off.
        spec = topology[3:]
        use_mst = spec.endswith("_mst")
        if use_mst:
            spec = spec[: -len("_mst")]
        k = int(spec)
        if not 1 <= k <= n - 1:
            raise ValueError(f"knn k must be in [1, {n - 1}], got {k}")

        A = np.abs(corr).astype(float).copy()
        np.fill_diagonal(A, -np.inf)          # never select self-loops
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in np.argsort(A[i])[::-1][:k]:
                G.add_edge(i, int(j))          # undirected union (symmetrised)

        if use_mst:
            for u, v in nx.minimum_spanning_tree(
                    _complete_corr_graph(corr)).edges():
                if not G.has_edge(u, v):
                    G.add_edge(u, v)

    else:
        raise ValueError(f"unknown topology: {topology}")

    edges = list(G.edges())
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index, corr


def build_complete_pairs():
    """Canonical ordering of the P = N(N-1)/2 undirected node pairs, plus the
    directed edge_index that lists all pairs forward then all pairs reversed.
    The two halves share the same pair ordering, so a length-P weight vector
    `w` maps onto the directed edges as cat([w, w])."""
    pairs = [(i, j) for i in range(N_FEATURES) for j in range(i + 1, N_FEATURES)]
    src = [i for i, j in pairs] + [j for i, j in pairs]
    dst = [j for i, j in pairs] + [i for i, j in pairs]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    pair_index = torch.tensor([[i for i, j in pairs], [j for i, j in pairs]],
                              dtype=torch.long)                      # [2, P]
    return edge_index, pair_index


def build_adaptive_prior(X_train_scaled: pd.DataFrame, threshold: float = TAU):
    """
    Prior adjacency A0 for the learnable-graph model.

    A0[p] = |r_ij| if the pair (i,j) belongs to the correlation+MST support,
            0      otherwise.

    Learning therefore *starts* from exactly the fixed Pearson+MST graph used
    by the rest of the paper; the L1 term below anchors it there.
    """
    corr = X_train_scaled[FEATURES].corr().values
    support = _corr_mst_edge_set(corr, threshold)
    edge_index, pair_index = build_complete_pairs()
    a0 = torch.tensor(
        [abs(corr[i, j]) if support.has_edge(i, j) else 0.0
         for i, j in zip(pair_index[0].tolist(), pair_index[1].tolist())],
        dtype=torch.float)
    return edge_index, pair_index, a0, corr


def make_graphs(X_df: pd.DataFrame, y_ser: pd.Series, edge_index: torch.Tensor):
    """One PyG graph per patient; node i == FEATURES[i]."""
    graphs = []
    Xv = X_df[FEATURES].values.astype(np.float32)
    yv = y_ser.values.astype(np.float32)
    for i in range(len(X_df)):
        x = torch.tensor(Xv[i], dtype=torch.float).view(-1, 1)   # [13, 1]
        y = torch.tensor([yv[i]], dtype=torch.float)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


# ----------------------------------------------------------------------
# Model (configurable for ablation; default == the paper's full model)
# ----------------------------------------------------------------------
@dataclass
class GCNConfig:
    hidden: int = 32
    n_layers: int = 2
    dropout: float = 0.30
    pool: str = "mean"        # mean | max | add
    use_bn: bool = True
    # --- learnable-graph extension (topology='corr_mst_learned') ---
    lambda_l1: float = 1.0    # anchors learned A to the Pearson prior A0
    lambda_conn: float = 1.0  # keeps every feature connected (participation)
    d_min: float = 1.0        # minimum total incident weight per node
    lr_graph: float = 1e-2    # separate LR for the edge parameters


_POOLS = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool}


class GCN(nn.Module):
    """Two-layer GCN (default). Outputs a logit; sigmoid applied at inference."""

    def __init__(self, cfg: GCNConfig, in_channels: int = 1):
        super().__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dims = [in_channels] + [cfg.hidden] * cfg.n_layers
        for k in range(cfg.n_layers):
            self.convs.append(GCNConv(dims[k], dims[k + 1]))
            self.bns.append(nn.BatchNorm1d(cfg.hidden) if cfg.use_bn else nn.Identity())
        self.linear = nn.Linear(cfg.hidden, 1)
        self.pool = _POOLS[cfg.pool]

    def forward(self, data):
        x, ei = data.x, data.edge_index
        batch = data.batch if getattr(data, "batch", None) is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for k, conv in enumerate(self.convs):
            x = conv(x, ei)
            x = self.bns[k](x)
            x = F.relu(x)
            if k < len(self.convs) - 1:
                x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        x = self.pool(x, batch)
        return self.linear(x).view(-1)           # logits


class AdaptiveGCN(nn.Module):
    """
    Feature-node GCN with a *learnable* weighted topology.

    The adjacency is parameterised as A = sigmoid(theta) over all N(N-1)/2
    undirected pairs and is initialised so that A = A0, the fixed Pearson+MST
    graph. Correlation therefore acts as a structural PRIOR that is refined by
    end-to-end optimisation, rather than as a hard preprocessing decision.

    Training objective:

        L = L_cls  +  lambda1 * || A - A0 ||_1  +  lambda2 * L_conn

      * L_cls   : binary cross-entropy (with logits)
      * L1 term : keeps learned edges close to the Pearson prior, preventing
                  the topology from drifting to an arbitrary graph
      * L_conn  : sum_i ReLU(d_min - deg_i)^2, deg_i = sum_j A_ij, which
                  preserves the participation guarantee that the MST provides
                  in the fixed-graph model (no feature is dropped from
                  message passing)
    """

    def __init__(self, cfg: GCNConfig, a0: torch.Tensor,
                 pair_index: torch.Tensor, in_channels: int = 1):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("a0", a0.clone())
        self.register_buffer("pair_index", pair_index.clone())
        # initialise theta so that sigmoid(theta) == A0 (clamped for finiteness)
        init = a0.clamp(0.01, 0.99)
        self.theta = nn.Parameter(torch.log(init / (1.0 - init)))

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dims = [in_channels] + [cfg.hidden] * cfg.n_layers
        for k in range(cfg.n_layers):
            self.convs.append(GCNConv(dims[k], dims[k + 1]))
            self.bns.append(nn.BatchNorm1d(cfg.hidden) if cfg.use_bn else nn.Identity())
        self.linear = nn.Linear(cfg.hidden, 1)
        self.pool = _POOLS[cfg.pool]

    # -- learned adjacency ------------------------------------------------
    def edge_weights(self) -> torch.Tensor:
        """A in [0,1]^P over the canonical undirected pair ordering."""
        return torch.sigmoid(self.theta)

    def degrees(self) -> torch.Tensor:
        w = self.edge_weights()
        deg = torch.zeros(N_FEATURES, device=w.device, dtype=w.dtype)
        deg = deg.index_add(0, self.pair_index[0], w)
        deg = deg.index_add(0, self.pair_index[1], w)
        return deg

    def graph_penalty(self):
        """lambda1 * ||A - A0||_1 + lambda2 * L_conn  (returns total, parts)."""
        w = self.edge_weights()
        l1 = (w - self.a0).abs().mean()
        slack = torch.relu(self.cfg.d_min - self.degrees())
        conn = slack.pow(2).mean()
        total = self.cfg.lambda_l1 * l1 + self.cfg.lambda_conn * conn
        return total, float(l1.detach()), float(conn.detach())

    def graph_params(self):
        return [self.theta]

    def other_params(self):
        return [p for n, p in self.named_parameters() if n != "theta"]

    # -- forward ----------------------------------------------------------
    def forward(self, data):
        x, ei = data.x, data.edge_index
        batch = data.batch if getattr(data, "batch", None) is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1

        w = self.edge_weights()
        ew = torch.cat([w, w])                      # forward + reversed halves
        ew = ew.repeat(n_graphs)                    # one copy per batched graph
        if ew.numel() != ei.size(1):                # safety: shape must match
            raise RuntimeError(
                f"edge weight/index mismatch: {ew.numel()} vs {ei.size(1)}")

        for k, conv in enumerate(self.convs):
            x = conv(x, ei, ew)
            x = self.bns[k](x)
            x = F.relu(x)
            if k < len(self.convs) - 1:
                x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        x = self.pool(x, batch)
        return self.linear(x).view(-1)


# ----------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 150
    patience: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512       # near/full-batch: minimal Python overhead on tiny graphs


def train_model(model, train_graphs, val_graphs, tcfg: TrainConfig):
    model.to(DEVICE)
    adaptive = hasattr(model, "graph_penalty")
    if adaptive:
        # theta is a topology parameter, not a weight: exclude it from weight
        # decay (decay would drag every edge toward sigmoid(0) = 0.5).
        opt = torch.optim.Adam(
            [{"params": model.other_params(), "lr": tcfg.lr,
              "weight_decay": tcfg.weight_decay},
             {"params": model.graph_params(), "lr": model.cfg.lr_graph,
              "weight_decay": 0.0}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr,
                               weight_decay=tcfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()           # balanced BCE (numerically stable)
    loader = DataLoader(train_graphs, batch_size=tcfg.batch_size, shuffle=True)
    vloader = DataLoader(val_graphs, batch_size=256, shuffle=False)

    best_val, best_wts, no_improve = float("inf"), None, 0
    for epoch in range(tcfg.epochs):
        model.train()
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            if adaptive:                          # + lambda1*L1 + lambda2*Lconn
                loss = loss + model.graph_penalty()[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for batch in vloader:
                batch = batch.to(DEVICE)
                vloss += criterion(model(batch), batch.y.view(-1)).item() * batch.num_graphs
                n += batch.num_graphs
        vloss /= max(n, 1)
        if vloss < best_val - 1e-5:
            best_val, best_wts, no_improve = vloss, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
        if no_improve >= tcfg.patience:
            break
    if best_wts:
        model.load_state_dict(best_wts)
    return model


@torch.no_grad()
def predict_probs(model, graphs):
    model.eval().to(DEVICE)
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    probs, true = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        p = torch.sigmoid(model(batch)).cpu().numpy()
        probs.append(p)
        true.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(probs), np.concatenate(true)


def best_threshold(probs, true, lo=0.10, hi=0.70, step=0.01):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        f1 = f1_score(true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def metrics_from(true, pred, prob):
    cm = confusion_matrix(true, pred, labels=[0, 1])
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return {
        "Accuracy": accuracy_score(true, pred),
        "Precision": precision_score(true, pred, zero_division=0),
        "Recall": recall_score(true, pred, zero_division=0),
        "F1": f1_score(true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(true, prob) if len(set(true)) > 1 else float("nan"),
        "MCC": matthews_corrcoef(true, pred),
        "Specificity": TN / (TN + FP) if (TN + FP) else 0.0,
    }


# ----------------------------------------------------------------------
# One full CV run of a GCN configuration
# ----------------------------------------------------------------------
def run_gcn_cv(
    X, y,
    gcfg: GCNConfig = GCNConfig(),
    tcfg: TrainConfig = TrainConfig(),
    topology: str = "corr_mst",
    threshold: float = TAU,
    n_splits: int = 5,
    seed: int = 42,
    tune_threshold: bool = True,
    return_oof: bool = False,
    verbose: bool = False,
    learned_graphs: list | None = None,
):
    """
    5-fold Stratified CV. For each fold:
      * inner train/val split (85/15, stratified) for early stopping
        and decision-threshold selection,
      * MinMax scaler fit on inner-train numeric columns,
      * fold-specific edge_index from inner-train correlations,
      * train GCN, tune threshold on inner-val, evaluate on the held-out
        test fold -> out-of-fold (OOF) predictions.
    """
    set_seed(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics = []
    oof_prob = np.full(len(X), np.nan)
    oof_pred = np.full(len(X), np.nan)
    oof_true = y.values.astype(int)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr_full, y_tr_full = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15,
            stratify=y_tr_full, random_state=seed,
        )

        # in-fold scaling (fit on inner-train only)
        scaler = MinMaxScaler()
        X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
        X_tr[NUMERIC_COLS] = scaler.fit_transform(X_tr[NUMERIC_COLS])
        X_val[NUMERIC_COLS] = scaler.transform(X_val[NUMERIC_COLS])
        X_te[NUMERIC_COLS] = scaler.transform(X_te[NUMERIC_COLS])

        # fold-specific graph from inner-train correlations
        if topology == "corr_mst_learned":
            # complete edge support; weights initialised at the Pearson+MST
            # prior A0 and refined end-to-end under the L1 + connectivity terms
            edge_index, pair_index, a0, _ = build_adaptive_prior(X_tr, threshold)
        else:
            edge_index, _ = build_edge_index(X_tr, topology=topology,
                                             threshold=threshold, seed=seed + fold)

        g_tr = make_graphs(X_tr, y_tr, edge_index)
        g_val = make_graphs(X_val, y_val, edge_index)
        g_te = make_graphs(X_te, y_te, edge_index)

        set_seed(seed + fold)                    # per-fold init reproducibility
        model = AdaptiveGCN(gcfg, a0, pair_index) if topology == "corr_mst_learned" \
            else GCN(gcfg)
        model = train_model(model, g_tr, g_val, tcfg)
        if topology == "corr_mst_learned" and learned_graphs is not None:
            learned_graphs.append(model.edge_weights().detach().cpu().numpy())

        vprob, vtrue = predict_probs(model, g_val)
        thr = best_threshold(vprob, vtrue) if tune_threshold else 0.5

        tprob, ttrue = predict_probs(model, g_te)
        tpred = (tprob >= thr).astype(int)

        oof_prob[te_idx] = tprob
        oof_pred[te_idx] = tpred
        fold_metrics.append(metrics_from(ttrue, tpred, tprob))
        if verbose:
            print(f"  fold {fold}: thr={thr:.2f} "
                  f"F1={fold_metrics[-1]['F1']:.3f} AUC={fold_metrics[-1]['ROC-AUC']:.3f}")

    summary = _aggregate(fold_metrics)
    if return_oof:
        return summary, fold_metrics, oof_prob, oof_pred.astype(int), oof_true
    return summary, fold_metrics


def _aggregate(fold_metrics):
    keys = list(fold_metrics[0].keys())
    return {k: (float(np.nanmean([m[k] for m in fold_metrics])),
                float(np.nanstd([m[k] for m in fold_metrics]))) for k in keys}


def fmt_mean_std(summary, keys=None):
    keys = keys or list(summary.keys())
    return {k: f"{summary[k][0]:.4f} ± {summary[k][1]:.4f}" for k in keys}
