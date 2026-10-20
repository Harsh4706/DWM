from __future__ import annotations

from datetime import date

import pandas as pd

CATEGORIES = [
    "Food", "Travel", "Shopping", "Education", "Entertainment",
    "Bills and Utilities", "Healthcare", "Rent", "Other",
]
PAYMENT_METHODS = ["UPI", "Cash", "Debit Card", "Credit Card", "Net Banking", "Other"]
REQUIRED_IMPORT_COLUMNS = {"title", "amount", "category", "expense_date", "payment_method"}


def validate_expense(title: str, amount: float, category: str, expense_date: date, payment_method: str) -> list[str]:
    errors: list[str] = []
    if not title.strip():
        errors.append("Title is required.")
    if amount <= 0:
        errors.append("Amount must be greater than zero.")
    if category not in CATEGORIES:
        errors.append("Choose a valid category.")
    if payment_method not in PAYMENT_METHODS:
        errors.append("Choose a valid payment method.")
    if not isinstance(expense_date, date):
        errors.append("Choose a valid date.")
    return errors


def validate_import_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    missing = sorted(REQUIRED_IMPORT_COLUMNS - set(frame.columns))
    if missing:
        return pd.DataFrame(), pd.DataFrame(), [f"Missing required columns: {', '.join(missing)}"]

    candidate = frame.copy()
    candidate["amount"] = pd.to_numeric(candidate["amount"], errors="coerce")
    candidate["expense_date"] = pd.to_datetime(candidate["expense_date"], errors="coerce").dt.date
    candidate["title"] = candidate["title"].fillna("").astype(str).str.strip()
    candidate["category"] = candidate["category"].fillna("").astype(str)
    candidate["payment_method"] = candidate["payment_method"].fillna("").astype(str)
    if "notes" not in candidate:
        candidate["notes"] = ""
    candidate["notes"] = candidate["notes"].fillna("").astype(str)

    valid_mask = (
        candidate["title"].ne("")
        & candidate["amount"].gt(0)
        & candidate["expense_date"].notna()
        & candidate["category"].isin(CATEGORIES)
        & candidate["payment_method"].isin(PAYMENT_METHODS)
    )
    return candidate.loc[valid_mask].reset_index(drop=True), candidate.loc[~valid_mask].reset_index(drop=True), []
