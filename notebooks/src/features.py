
import pandas as pd

def create_features(data):
    d = data.copy()

    d["income_monthly"] = d["annual_inc"] / 12
    d["loan_to_income"] = d["loan_amnt"] / (d["annual_inc"] + 1)
    d["payment_to_income"] = d["installment"] / (d["income_monthly"] + 1)
    d["debt_to_income"] = d["dti"] / 100
    d["fico_average"] = (d["fico_range_low"] + d["fico_range_high"]) / 2
    d["fico_spread"] = d["fico_range_high"] - d["fico_range_low"]
    d["credit_per_account"] = d["revol_bal"] / (d["open_acc"] + 1)
    d["account_activity"] = d["open_acc"] / (d["total_acc"] + 1)
    d["delinq_per_account"] = d["delinq_2yrs"] / (d["total_acc"] + 1)
    d["revolving_utilization"] = pd.to_numeric(
        d["revol_util"].astype(str).str.replace("%", ""),
        errors="coerce"
    ) / 100

    return d
