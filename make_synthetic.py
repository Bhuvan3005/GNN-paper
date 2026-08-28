# -*- coding: utf-8 -*-
"""
make_synthetic.py
=================
Gaussian-copula synthetic cohort generator for the simulation study.

PURPOSE AND LIMITS
------------------
This is a POWER / SIMULATION study, NOT external validation. The generator
is fitted to Cleveland, so testing on its output cannot confirm that the
real population has this structure. What it CAN answer is:

    "If the population had the dependence structure estimated from
     Cleveland, would the paper's claims be detectable at n = 10,000
     rather than n = 303?"

Method
------
1. Each real feature is mapped to normal scores via its empirical CDF.
2. The latent Gaussian correlation matrix is estimated from those scores
   (projected to the nearest PSD matrix if required).
3. n samples are drawn from that multivariate normal and mapped back
   through the inverse empirical CDF, so every synthetic marginal is a
   resample of the real marginal (categorical levels stay valid).
4. Labels are drawn from one of two data-generating processes:

   'additive'    : logit = b0 + z @ beta
                   Purely additive. A linear model is Bayes-optimal here,
                   so the GCN cannot win; this scenario tests whether the
                   graph prior COSTS anything when no interactions exist.

   'interaction' : logit = b0 + z @ beta + sum_{(i,j) in E} w_ij z_i z_j
                   Multiplicative terms along the Pearson+MST edge set.
                   This scenario tests whether the graph prior GAINS when
                   feature interactions genuinely exist.

   beta is taken from a logistic regression fitted to the real cohort, so
   the additive part is realistic. The interaction weights and intercept
   are calibrated to hold prevalence near the real value (~0.459).

A covariate-shifted variant is also provided to probe the transport claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline import FEATURES, CATEGORICAL_COLS, TAU, load_data, _corr_mst_edge_set

REAL_PREVALENCE = None          # set from data at fit time


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _nearest_psd(A: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone and restore unit diagonal."""
    A = (A + A.T) / 2.0
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 1e-6, None)
    A = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(A))
    return A / np.outer(d, d)


def _normal_scores(x: np.ndarray) -> np.ndarray:
    """Empirical-CDF transform to standard normal (ties averaged)."""
    n = len(x)
    ranks = pd.Series(x).rank(method="average").values
    return norm.ppf(ranks / (n + 1.0))


def _inverse_ecdf(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Map uniforms back onto the empirical distribution of `ref`."""
    return np.quantile(ref, np.clip(u, 1e-6, 1 - 1e-6), method="inverted_cdf")


# ----------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------
class CopulaGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.columns_ = list(X.columns)
        self.ref_ = {c: X[c].values.astype(float) for c in self.columns_}

        Z = np.column_stack([_normal_scores(X[c].values.astype(float))
                             for c in self.columns_])
        self.latent_corr_ = _nearest_psd(np.corrcoef(Z, rowvar=False))

        # additive coefficients from a logistic fit on the real cohort
        self.scaler_ = StandardScaler().fit(X[self.columns_].values.astype(float))
        Xs = self.scaler_.transform(X[self.columns_].values.astype(float))
        lr = LogisticRegression(max_iter=2000).fit(Xs, y.values)
        self.beta_ = lr.coef_.ravel()
        self.prevalence_ = float(y.mean())

        # interaction support = Pearson + MST edges on the real data
        corr = X[self.columns_].corr().values
        G = _corr_mst_edge_set(corr, TAU)
        self.edges_ = list(G.edges())
        return self

    # ---- sampling -----------------------------------------------------
    def _sample_X(self, n: int, rng, shift: float = 0.0) -> pd.DataFrame:
        L = np.linalg.cholesky(self.latent_corr_)
        Zs = rng.standard_normal((n, len(self.columns_))) @ L.T
        if shift:
            # covariate shift: translate the latent field, which moves the
            # marginals while leaving the dependence structure intact
            Zs = Zs + shift
        U = norm.cdf(Zs)
        cols = {}
        for k, c in enumerate(self.columns_):
            vals = _inverse_ecdf(U[:, k], self.ref_[c])
            if c in CATEGORICAL_COLS:
                vals = np.round(vals)
            cols[c] = vals
        return pd.DataFrame(cols)[self.columns_]

    def _labels(self, Xdf: pd.DataFrame, dgp: str, rng,
                inter_scale: float = 0.65) -> pd.Series:
        Xs = self.scaler_.transform(Xdf.values.astype(float))
        logit = Xs @ self.beta_

        if dgp == "interaction":
            rng_w = np.random.default_rng(12345)          # fixed interaction field
            inter = np.zeros(len(Xdf))
            for (i, j) in self.edges_:
                w = rng_w.normal(0.0, 1.0)
                inter += w * Xs[:, i] * Xs[:, j]
            inter /= (np.std(inter) + 1e-12)
            logit = logit + inter_scale * np.std(logit) * inter
        elif dgp != "additive":
            raise ValueError(dgp)

        # calibrate intercept to the real prevalence
        lo, hi = -20.0, 20.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if np.mean(1 / (1 + np.exp(-(logit + mid)))) < self.prevalence_:
                lo = mid
            else:
                hi = mid
        logit = logit + (lo + hi) / 2
        p = 1 / (1 + np.exp(-logit))
        return pd.Series((rng.random(len(p)) < p).astype(int), name="target")

    def sample(self, n: int, dgp: str = "additive", seed: int | None = None,
               shift: float = 0.0):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        Xdf = self._sample_X(n, rng, shift=shift)
        ydf = self._labels(Xdf, dgp, rng)
        return Xdf, ydf


# ----------------------------------------------------------------------
# fidelity table (real vs synthetic)
# ----------------------------------------------------------------------
def fidelity_table(X_real, y_real, X_syn, y_syn):
    rows = []
    for c in FEATURES:
        r, s = X_real[c].astype(float), X_syn[c].astype(float)
        rows.append({
            "Feature": c,
            "Real mean": round(r.mean(), 3), "Syn mean": round(s.mean(), 3),
            "Real std": round(r.std(), 3), "Syn std": round(s.std(), 3),
            "Real min": round(r.min(), 2), "Syn min": round(s.min(), 2),
            "Real max": round(r.max(), 2), "Syn max": round(s.max(), 2),
            "|Δmean|/std": round(abs(r.mean() - s.mean()) / (r.std() + 1e-9), 3),
        })
    rows.append({
        "Feature": "target (prevalence)",
        "Real mean": round(float(y_real.mean()), 3),
        "Syn mean": round(float(y_syn.mean()), 3),
        "Real std": "", "Syn std": "", "Real min": "", "Syn min": "",
        "Real max": "", "Syn max": "", "|Δmean|/std": "",
    })
    return pd.DataFrame(rows)


def corr_fidelity(X_real, X_syn):
    cr = X_real[FEATURES].corr().values
    cs = X_syn[FEATURES].corr().values
    iu = np.triu_indices(len(FEATURES), k=1)
    return {
        "Mean |Δr|": round(float(np.mean(np.abs(cr[iu] - cs[iu]))), 4),
        "Max |Δr|": round(float(np.max(np.abs(cr[iu] - cs[iu]))), 4),
        "Frobenius ||Δ||": round(float(np.linalg.norm(cr - cs)), 4),
        "Corr of corrs": round(float(np.corrcoef(cr[iu], cs[iu])[0, 1]), 4),
    }


if __name__ == "__main__":
    import sys, os
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs("results", exist_ok=True)

    N = 10_000
    X_real, y_real = load_data()
    gen = CopulaGenerator(seed=42).fit(X_real, y_real)

    for dgp in ("additive", "interaction"):
        Xs, ys = gen.sample(N, dgp=dgp, seed=42)
        out = Xs.copy(); out["target"] = ys.values
        out.to_csv(f"synthetic_{dgp}_{N}.csv", index=False)
        print(f"[{dgp}] n={len(out)}  prevalence={ys.mean():.4f}")
        if dgp == "additive":
            ft = fidelity_table(X_real, y_real, Xs, ys)
            ft.to_csv("results/table_synthetic_fidelity.csv", index=False)
            print(ft.to_string(index=False))
            cf = corr_fidelity(X_real, Xs)
            pd.DataFrame([cf]).to_csv("results/table_synthetic_corr_fidelity.csv",
                                      index=False)
            print("\nCorrelation-structure fidelity:", cf)

    # shifted cohort for the transport probe
    Xsh, ysh = gen.sample(N, dgp="interaction", seed=7, shift=0.5)
    sh = Xsh.copy(); sh["target"] = ysh.values
    sh.to_csv(f"synthetic_shifted_{N}.csv", index=False)
    print(f"[shifted] n={len(sh)}  prevalence={ysh.mean():.4f}")
    print("\nSaved synthetic cohorts + results/table_synthetic_fidelity.csv")
