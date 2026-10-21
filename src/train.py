"""Fit a train-only WoE scorecard and an LGBM model with temporal validation and Platt scaling."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml
from sklearn.linear_model import LogisticRegression
from .features import FEATURES, feature_matrix
from .data import split_validation
from .metrics import calculate_metrics, bank_profit

SEED = 42


class QuantileWoE:
    """Numeric train-only quantile binning with add-0.5 WoE smoothing and a missing bin.

    The output values are ln(P(bin|non-default) / P(bin|default)); fit learns
    boundaries and event rates from TRAIN alone (never from validation/test).
    """
    def __init__(self, n_bins=8, smoothing=0.5):
        self.n_bins = n_bins
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y) -> "QuantileWoE":
        X, y = pd.DataFrame(X), np.asarray(y, dtype=int)
        self.features_ = list(X.columns)
        self.bins_, self.mapping_ = {}, {}
        for col in self.features_:
            values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            values[~np.isfinite(values)] = np.nan
            valid = values[np.isfinite(values)]
            cuts = np.unique(np.quantile(valid, np.linspace(0, 1, self.n_bins + 1)[1:-1])) if len(valid) else np.array([])
            groups = np.searchsorted(cuts, values, side="right").astype(int)
            groups[~np.isfinite(values)] = len(cuts) + 1
            k = len(cuts) + 2
            n_good = np.bincount(groups, weights=(y == 0), minlength=k)
            n_bad = np.bincount(groups, weights=(y == 1), minlength=k)
            good_share = (n_good + self.smoothing) / (n_good.sum() + k * self.smoothing)
            bad_share = (n_bad + self.smoothing) / (n_bad.sum() + k * self.smoothing)
            self.bins_[col] = cuts
            self.mapping_[col] = np.log(good_share / bad_share)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        if set(self.features_) - set(X.columns):
            raise ValueError("В новых данных не хватает признаков WoE")
        out = {}
        for col in self.features_:
            values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            values[~np.isfinite(values)] = np.nan
            cuts = self.bins_[col]
            ids = np.searchsorted(cuts, values, side="right").astype(int)
            ids[~np.isfinite(values)] = len(cuts) + 1
            out[col] = self.mapping_[col][ids]
        return pd.DataFrame(out, index=X.index)


def baseline_stage(artifacts="artifacts") -> dict:
    root = Path(artifacts)
    root.mkdir(parents=True, exist_ok=True)
    train = pd.read_pickle(root / "train.pkl")
    val = pd.read_pickle(root / "valid.pkl")
    Xt, Xv = feature_matrix(train), feature_matrix(val)
    y_train, y_valid = train.target.to_numpy(), val.target.to_numpy()
    # Train-only boundaries and WoE event distribution. No val/test labels inside the transformation.
    binner = QuantileWoE().fit(Xt, y_train)
    Xt_w, Xv_w = binner.transform(Xt), binner.transform(Xv)
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    baseline.fit(Xt_w, y_train)
    splits = split_validation(val)
    tune = splits["tune"]
    p_tune = baseline.predict_proba(binner.transform(feature_matrix(tune)))[:, 1]
    stats = calculate_metrics(tune.target, p_tune)
    joblib.dump({"features": FEATURES, "binner": binner, "baseline": baseline}, root / "baseline.pkl")
    print("Неделя 2: WoE + логистическая регрессия, TUNE:", {k: round(v, 4) for k,v in stats.items()})
    return stats


class PlattCalibration:
    """Fit probability calibration solely on a chronological calibration period."""
    def fit(self, raw_prob, y) -> "PlattCalibration":
        p = np.clip(np.asarray(raw_prob, dtype=float), 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        self.model_ = LogisticRegression(max_iter=1000, random_state=SEED)
        self.model_.fit(z, np.asarray(y))
        return self

    def predict(self, raw_prob) -> np.ndarray:
        p = np.clip(np.asarray(raw_prob, dtype=float), 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        return self.model_.predict_proba(z)[:, 1]


def boost_stage(artifacts="artifacts", config_path="config.yaml", *, n_estimators=1000) -> dict:
    root = Path(artifacts)
    b = joblib.load(root / "baseline.pkl")
    train = pd.read_pickle(root / "train.pkl")
    val = pd.read_pickle(root / "valid.pkl")
    sections = split_validation(val)
    binner = b["binner"]
    Xt_w = binner.transform(feature_matrix(train))
    X_tune = binner.transform(feature_matrix(sections["tune"]))
    y_train = train.target.to_numpy()
    y_tune = sections["tune"].target.to_numpy()
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=.03, num_leaves=31,
        min_child_samples=50, colsample_bytree=.85, random_state=SEED,
        n_jobs=2, verbosity=-1,
    )
    model.fit(Xt_w, y_train, eval_set=[(X_tune, y_tune)], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])

    cal = sections["calibration"]
    cal_raw = model.predict_proba(binner.transform(feature_matrix(cal)))[:, 1]
    calibrator = PlattCalibration().fit(cal_raw, cal.target)
    tune_raw = model.predict_proba(X_tune)[:, 1]
    baseline_prob = b["baseline"].predict_proba(X_tune)[:, 1]
    result = {"baseline_tune": calculate_metrics(y_tune, baseline_prob),
              "boosting_tune": calculate_metrics(y_tune, tune_raw)}

    # Policy threshold uses separate late-validation window, not tune/calibration/test.
    policy = sections["policy"]
    p_pol = calibrator.predict(model.predict_proba(binner.transform(feature_matrix(policy)))[:, 1])
    # No threshold below 2% or above 80% for illustrative exercise.
    thresholds = np.round(np.arange(.02, .801, .02), 2)
    policy_profit = np.array([
        bank_profit(policy.target, p_pol, policy.loan_amnt, t)
        for t in thresholds
    ])
    decision_threshold = float(thresholds[np.argmax(policy_profit)])
    bundle = {"model": model, "binner": binner, "calibrator": calibrator,
              "features": FEATURES, "threshold": decision_threshold,
              "policy_assumptions": {"return_good": .15, "LGD": .45, "EAD": "loan_amnt"}}
    joblib.dump(bundle, root / "model_bundle.pkl")
    parameters = {"model": "LightGBM", "random_state": SEED,
                  "dataset": "Historical LendingClub accepted_2007_to_2018Q4",
                  "target_proxy": "Charged Off vs Fully Paid (NOT exact 90+ days)",
                  "best_iteration": int(model.best_iteration_ or n_estimators),
                  "model_parameters": model.get_params(),
                  "calibration": "Platt sigmoid, separate chronological validation segment",
                  "threshold": decision_threshold,
                  "validation_segments": {
                      key: [str(frame.issue_d.min().date()), str(frame.issue_d.max().date())]
                      for key, frame in sections.items()
                  },
                  "financial_assumptions": {"good_credit_return": .15, "LGD": .45, "EAD": "original loan"}}
    with open(config_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(parameters, stream, allow_unicode=True, sort_keys=False)
    print("Неделя 3: validation без утечки:", result)
    print("Порог (только POLICY-часть validation):", decision_threshold)
    return result


def load_bundle(path="model.pkl") -> dict:
    return joblib.load(path)


def predict(df: pd.DataFrame, bundle_path="model.pkl") -> pd.Series:
    """Return calibrated P(Charged Off) for application-like data, not 90+ default."""
    bundle = load_bundle(bundle_path)
    W = bundle["binner"].transform(feature_matrix(df))
    raw = bundle["model"].predict_proba(W)[:, 1]
    p = bundle["calibrator"].predict(raw)
    return pd.Series(p, index=df.index, name="p_charged_off")
