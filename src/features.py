"""Origination-time credit application features (LendingClub historical sample)."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_NUMERIC = [
    "loan_amnt", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "open_acc", "pub_rec",
    "revol_bal", "total_acc", "revol_util"
]
BASE_FEATURES = [
    "loan_amnt", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "open_acc", "pub_rec",
    "revol_bal", "total_acc"
]
NEW_FEATURES = [
    "income_monthly", "loan_to_income", "payment_to_income", "debt_to_income",
    "fico_average", "fico_spread", "credit_per_account", "account_activity",
    "delinq_per_account", "revolving_utilization"
]
FEATURES = BASE_FEATURES + NEW_FEATURES

ECONOMIC_MEANING = {
    "loan_amnt": "Запрошенная сумма кредита, USD",
    "installment": "Месячный платёж по предложенному кредиту, USD",
    "annual_inc": "Заявленный годовой доход, USD",
    "dti": "Долговая нагрузка из анкеты, %",
    "delinq_2yrs": "Число прежних просрочек за два года",
    "fico_range_low": "Нижняя граница исходного кредитного рейтинга FICO",
    "fico_range_high": "Верхняя граница исходного рейтинга FICO",
    "open_acc": "Число открытых кредитных счетов",
    "pub_rec": "Число негативных публичных записей",
    "revol_bal": "Остаток по револьверным кредитам, USD",
    "total_acc": "Общее число кредитных счетов",
    "revol_util": "Использование револьверного лимита",
    "income_monthly": "Оценка ежемесячного дохода: годовой / 12",
    "loan_to_income": "Отношение размера кредита к годовому доходу",
    "payment_to_income": "Доля платежа по новому кредиту в месячном доходе",
    "debt_to_income": "Долговая нагрузка в долях единицы",
    "fico_average": "Средняя оценка исходного рейтинга FICO",
    "fico_spread": "Ширина исходного диапазона FICO",
    "credit_per_account": "Средний револьверный долг на открытый счёт",
    "account_activity": "Доля открытых счетов среди всех счетов",
    "delinq_per_account": "Частота прошлых просрочек на один счёт",
    "revolving_utilization": "Доля использования револьверного кредита",
}


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    for col in RAW_NUMERIC:
        if col not in d:
            raise ValueError(f"Нет нужного столбца: {col}")
        if col == "revol_util":
            d[col] = pd.to_numeric(d[col].astype(str).str.replace("%", "", regex=False), errors="coerce")
        else:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d["income_monthly"] = d["annual_inc"] / 12
    d["loan_to_income"] = d["loan_amnt"] / (d["annual_inc"] + 1)
    d["payment_to_income"] = d["installment"] / (d["income_monthly"] + 1)
    d["debt_to_income"] = d["dti"] / 100
    d["fico_average"] = (d["fico_range_low"] + d["fico_range_high"]) / 2
    d["fico_spread"] = d["fico_range_high"] - d["fico_range_low"]
    d["credit_per_account"] = d["revol_bal"] / (d["open_acc"] + 1)
    d["account_activity"] = d["open_acc"] / (d["total_acc"] + 1)
    d["delinq_per_account"] = d["delinq_2yrs"] / (d["total_acc"] + 1)
    d["revolving_utilization"] = d["revol_util"] / 100
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan)
    return d


def feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    return create_features(data)[FEATURES]
