# -*- coding: utf-8 -*-
"""
conv_variants.py
================
Alternative graph-convolution operators for the feature-node formulation.

Scope
-----
This is an ABLATION over the message-passing operator only. The research
methodology is unchanged: same feature-node graph (13 nodes, Pearson
tau=0.15 + MST, rebuilt per fold from training data only), same depth,
hidden width, dropout, readout, optimizer, early stopping, threshold
tuning and CV protocol. Only the convolution is swapped, which answers
the standard reviewer question "why GCN and not an attention-based or
more expressive operator?".

Operators
---------
  gcn   GCNConv     spectral convolution with symmetric normalisation (ours)
  gat   GATConv     multi-head attention over neighbours (4 heads x 8 dims)
  sage  SAGEConv    mean aggregation with a skip-style root weight
  gin   GINConv     sum aggregation + MLP (maximally expressive under WL)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv,
    global_mean_pool, global_max_pool, global_add_pool,
)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

import pipeline as P
from pipeline import (
    set_seed, build_edge_index, make_graphs, train_model, predict_probs,
    best_threshold, metrics_from, _aggregate, TrainConfig, TAU,
)

_POOLS = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool}


@dataclass
class ConvConfig:
    conv: str = "gcn"          # gcn | gat | sage | gin
    hidden: int = 32
    n_layers: int = 2
    dropout: float = 0.30
    pool: str = "mean"
    use_bn: bool = True
    heads: int = 4             # GAT only


def _make_conv(kind: str, in_dim: int, out_dim: int, heads: int):
    if kind == "gcn":
        return GCNConv(in_dim, out_dim)
    if kind == "gat":
        assert out_dim % heads == 0, "hidden must be divisible by heads"
        return GATConv(in_dim, out_dim // heads, heads=heads, concat=True)
    if kind == "sage":
        return SAGEConv(in_dim, out_dim)
    if kind == "gin":
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                            nn.Linear(out_dim, out_dim))
        return GINConv(mlp)
    raise ValueError(f"unknown conv: {kind}")


class ConvGNN(nn.Module):
    """Feature-node GNN with a swappable convolution operator."""

    def __init__(self, cfg: ConvConfig, in_channels: int = 1):
        super().__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dims = [in_channels] + [cfg.hidden] * cfg.n_layers
        for k in range(cfg.n_layers):
            self.convs.append(_make_conv(cfg.conv, dims[k], dims[k + 1], cfg.heads))
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
        return self.linear(x).view(-1)


def run_conv_cv(X, y, ccfg: ConvConfig, tcfg: TrainConfig = TrainConfig(),
                topology="corr_mst", threshold=TAU, n_splits=5, seed=42,
                tune_threshold=True, return_oof=False, verbose=False):
    """Identical protocol to pipeline.run_gcn_cv; only the operator differs."""
    set_seed(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    features, numeric = list(P.FEATURES), list(P.NUMERIC_COLS)

    fold_metrics = []
    oof_prob = np.full(len(X), np.nan)
    oof_pred = np.full(len(X), np.nan)
    oof_true = y.values.astype(int)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr_full, y_tr_full = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15,
            stratify=y_tr_full, random_state=seed)

        scaler = MinMaxScaler()
        X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()
        X_tr[numeric] = scaler.fit_transform(X_tr[numeric])
        X_val[numeric] = scaler.transform(X_val[numeric])
        X_te[numeric] = scaler.transform(X_te[numeric])

        edge_index, _ = build_edge_index(X_tr, topology=topology,
                                         threshold=threshold, seed=seed + fold)
        g_tr = make_graphs(X_tr, y_tr, edge_index)
        g_val = make_graphs(X_val, y_val, edge_index)
        g_te = make_graphs(X_te, y_te, edge_index)

        set_seed(seed + fold)
        model = train_model(ConvGNN(ccfg), g_tr, g_val, tcfg)

        vprob, vtrue = predict_probs(model, g_val)
        thr = best_threshold(vprob, vtrue) if tune_threshold else 0.5
        tprob, ttrue = predict_probs(model, g_te)
        tpred = (tprob >= thr).astype(int)

        oof_prob[te_idx], oof_pred[te_idx] = tprob, tpred
        fold_metrics.append(metrics_from(ttrue, tpred, tprob))
        if verbose:
            print(f"  fold {fold}: F1={fold_metrics[-1]['F1']:.3f} "
                  f"AUC={fold_metrics[-1]['ROC-AUC']:.3f}")

    summary = _aggregate(fold_metrics)
    if return_oof:
        return summary, fold_metrics, oof_prob, oof_pred.astype(int), oof_true
    return summary, fold_metrics


def count_params(ccfg: ConvConfig) -> int:
    return sum(p.numel() for p in ConvGNN(ccfg).parameters() if p.requires_grad)
