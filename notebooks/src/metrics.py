
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def calculate_metrics(y, p):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)

    return {
        "ROC-AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": np.max(tpr - fpr)
    }

def expected_loss(pd_values, loan_amount, lgd=0.45):
    return np.asarray(pd_values) * lgd * np.asarray(loan_amount)

def bank_profit(y, p, amounts, threshold):
    approved = np.asarray(p) < threshold
    profit = np.where(
        np.asarray(y) == 0,
        np.asarray(amounts) * 0.15,
        -np.asarray(amounts) * 0.45
    )
    return profit[approved].sum()
