
import numpy as np
import lightgbm as lgb
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression

def train_models(X_train, y_train, X_valid, y_valid):
    features = list(X_train.columns)

    binning = BinningProcess(variable_names=features)
    binning.fit(X_train, y_train)

    train_woe = binning.transform(X_train, metric="woe")
    valid_woe = binning.transform(X_valid, metric="woe")

    baseline = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    baseline.fit(train_woe, y_train)

    boosting = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )

    boosting.fit(
        train_woe, y_train,
        eval_set=[(valid_woe, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    return baseline, boosting, binning
