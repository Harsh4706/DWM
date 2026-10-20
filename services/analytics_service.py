from __future__ import annotations

import pandas as pd


def filter_expenses(frame: pd.DataFrame, start_date, end_date, categories=None, payment_methods=None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame[frame["expense_date"].between(start_date, end_date)].copy()
    if categories:
        result = result[result["category"].isin(categories)]
    if payment_methods:
        result = result[result["payment_method"].isin(payment_methods)]
    return result


def summarize_expenses(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"total": 0.0, "count": 0, "average": 0.0, "highest": 0.0, "top_category": "None"}
    category_totals = frame.groupby("category")["amount"].sum()
    return {
        "total": float(frame["amount"].sum()),
        "count": int(len(frame)),
        "average": float(frame["amount"].mean()),
        "highest": float(frame["amount"].max()),
        "top_category": str(category_totals.idxmax()),
    }


def monthly_totals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["month", "amount"])
    result = frame.assign(month=pd.to_datetime(frame["expense_date"]).dt.to_period("M")).groupby("month", as_index=False)["amount"].sum()
    result["month"] = result["month"].astype(str)
    return result
