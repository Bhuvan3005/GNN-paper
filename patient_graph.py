# -*- coding: utf-8 -*-
"""
patient_graph.py
================
The CONVENTIONAL patient-similarity GNN formulation, implemented as an
empirical baseline for the feature-node model.

Motivation
----------
The paper's central claim is that a *feature-node* graph (one small graph
per patient; nodes = clinical features) is preferable to the conventional
*patient-similarity* graph (one cohort-level graph; nodes = patients).
Until now that contrast was argued qualitatively. This module supplies
the missing empirical comparison.

Formulation
-----------
  * Nodes            : patients (one cohort-level graph)
  * Node features    : the patient's 13 scaled clinical values
  * Edges            : symmetric k-nearest-neighbour graph in feature
                       space (Euclidean), built from FEATURES ONLY --
                       never from labels
  * Task             : transductive node classification
  * Model            : 2-layer GCN, identical hidden width / dropout /
                       optimizer / early stopping as the feature-node model
  * No pooling       : each node emits its own logit

Leakage control
---------------
The scaler is fit on inner-train patients only. The k-NN graph is built
over all patients' *feature vectors*, which is inherent and standard for
transductive semi-supervised node classification (cf. Planetoid): test
node FEATURES participate in message passing, test node LABELS never do.
The loss is masked to inner-train nodes; early stopping and the decision
threshold use inner-validation nodes only; metrics come from the held-out
test fold. This transductive requirement is itself a property we report:
unlike the feature-node model, it needs a cohort graph at inference time.

Fairness to the baseline
------------------------
k is swept over {5, 10, 15, 20} rather than fixed, so the comparison
cannot be accused of strawmanning the conventional approach.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph

import pipeline as P
from pipeline import DEVICE, set_seed, metrics_from, best_threshold, _aggregate


# ----------------------------------------------------------------------
# k-NN patient graph (features only — no labels)
# ----------------------------------------------------------------------
def knn_edge_index(X_scaled: np.ndarray, k: int = 10) -> torch.Tensor:
    n = X_scaled.shape[0]
    k_eff = min(k, max(1, n - 1))
    A = kneighbors_graph(X_scaled, n_neighbors=k_eff, mode="connectivity",
                         metric="euclidean", include_self=False)
    A = A.maximum(A.T)                      # symmetrise
    coo = A.tocoo()
    return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)


# ----------------------------------------------------------------------
# Patient-node GCN (node classification; no graph pooling)
# ----------------------------------------------------------------------
class PatientGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32,
                 n_layers: int = 2, dropout: float = 0.30, use_bn: bool = True):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dims = [in_channels] + [hidden] * n_layers
        for i in range(n_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(hidden) if use_bn else nn.Identity())
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.linear(x).view(-1)      # logits, one per patient node


# ----------------------------------------------------------------------
# Transductive training with masked loss
# ----------------------------------------------------------------------
def train_patient_gcn(x, edge_index, y, train_mask, val_mask,
                      hidden=32, n_layers=2, dropout=0.30,
                      epochs=300, patience=40, lr=1e-3, weight_decay=1e-4,
                      seed=42):
    set_seed(seed)
    model = PatientGCN(x.size(1), hidden, n_layers, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    x, edge_index, y = x.to(DEVICE), edge_index.to(DEVICE), y.to(DEVICE)
    best_val, best_wts, no_improve = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = crit(out[train_mask], y[train_mask])      # loss on train nodes only
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = crit(model(x, edge_index)[val_mask], y[val_mask]).item()
        if vloss < best_val - 1e-5:
            best_val, no_improve = vloss, 0
            best_wts = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_wts:
        model.load_state_dict(best_wts)
    return model


@torch.no_grad()
def node_probs(model, x, edge_index):
    model.eval()
    return torch.sigmoid(model(x.to(DEVICE), edge_index.to(DEVICE))).cpu().numpy()


# ----------------------------------------------------------------------
# 5-fold stratified CV, matching the feature-node protocol exactly
# ----------------------------------------------------------------------
def run_patient_gcn_cv(X, y, k=10, hidden=32, n_layers=2, dropout=0.30,
                       n_splits=5, seed=42, tune_threshold=True,
                       return_oof=False, verbose=False):
    set_seed(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    features = list(P.FEATURES)
    numeric = list(P.NUMERIC_COLS)

    fold_metrics = []
    oof_prob = np.full(len(X), np.nan)
    oof_pred = np.full(len(X), np.nan)
    oof_true = y.values.astype(int)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        # inner train/val split, identical ratio & seed to the feature-node model
        inner_tr, inner_val = train_test_split(
            tr_idx, test_size=0.15, stratify=y.iloc[tr_idx], random_state=seed)

        # scaler fit on inner-train only, applied to everyone
        Xs = X[features].copy()
        scaler = MinMaxScaler().fit(Xs.iloc[inner_tr][numeric])
        Xs[numeric] = scaler.transform(Xs[numeric])
        Xv = Xs.values.astype(np.float32)

        edge_index = knn_edge_index(Xv, k=k)      # features only, no labels
        x = torch.tensor(Xv, dtype=torch.float)
        yt = torch.tensor(y.values.astype(np.float32), dtype=torch.float)

        n = len(X)
        train_mask = torch.zeros(n, dtype=torch.bool); train_mask[inner_tr] = True
        val_mask = torch.zeros(n, dtype=torch.bool); val_mask[inner_val] = True

        model = train_patient_gcn(x, edge_index, yt, train_mask, val_mask,
                                  hidden=hidden, n_layers=n_layers,
                                  dropout=dropout, seed=seed + fold)

        probs = node_probs(model, x, edge_index)
        thr = best_threshold(probs[inner_val], oof_true[inner_val]) if tune_threshold else 0.5

        tprob = probs[te_idx]
        tpred = (tprob >= thr).astype(int)
        oof_prob[te_idx], oof_pred[te_idx] = tprob, tpred
        fold_metrics.append(metrics_from(oof_true[te_idx], tpred, tprob))
        if verbose:
            print(f"  fold {fold}: thr={thr:.2f} F1={fold_metrics[-1]['F1']:.3f} "
                  f"AUC={fold_metrics[-1]['ROC-AUC']:.3f}")

    summary = _aggregate(fold_metrics)
    if return_oof:
        return summary, fold_metrics, oof_prob, oof_pred.astype(int), oof_true
    return summary, fold_metrics
