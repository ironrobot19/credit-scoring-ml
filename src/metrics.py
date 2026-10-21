"""ROC-AUC, Gini, KS, PSI and illustrative loan-economics metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def calculate_metrics(y, p) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    auc = float(roc_auc_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)
    return {"ROC-AUC": auc, "Gini": 2 * auc - 1, "KS": float(np.max(tpr - fpr))}


def expected_loss(pd_values, loan_amount, lgd=.45):
    """EL = PD × LGD × EAD; assume EAD = initial loan for education only."""
    return np.asarray(pd_values) * lgd * np.asarray(loan_amount)


def bank_profit(y, probability, amount, threshold, good_return=.15, lgd=.45) -> float:
    """HYPOTHETICAL 15% return on good approvals, -45% principal on bad approvals."""
    y, p, amount = np.asarray(y), np.asarray(probability), np.asarray(amount)
    return float(np.where(y == 0, good_return * amount, -lgd * amount)[p < threshold].sum())


def psi(expected, actual, bins=10) -> float:
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp, act = exp[np.isfinite(exp)], act[np.isfinite(act)]
    if not len(exp) or not len(act):
        return float("nan")
    edges = np.unique(np.quantile(exp, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        # Identical constants => no evidence of drift when both are identical.
        return 0.0 if len(np.unique(act)) == 1 and act[0] == exp[0] else float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(exp, bins=edges)[0] / len(exp)
    q = np.histogram(act, bins=edges)[0] / len(act)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))
