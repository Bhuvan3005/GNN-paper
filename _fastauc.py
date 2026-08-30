import numpy as np
def fast_auc(y, p):
    """Rank-based AUC with correct tie handling (== sklearn roc_auc_score)."""
    order = np.argsort(p, kind="mergesort")
    p_s, y_s = p[order], y[order]
    n = len(p_s)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and p_s[j] == p_s[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    npos = y_s.sum()
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return (ranks[y_s == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
