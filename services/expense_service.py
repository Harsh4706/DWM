from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from database.db import DB_PATH, get_connection

COLUMNS = ["id", "title", "amount", "category", "expense_date", "payment_method", "notes", "created_at"]


def list_expenses(db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as connection:
        frame = pd.read_sql_query("SELECT * FROM expenses ORDER BY expense_date DESC, id DESC", connection)
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    frame["expense_date"] = pd.to_datetime(frame["expense_date"]).dt.date
    return frame


def add_expense(title: str, amount: float, category: str, expense_date: date, payment_method: str, notes: str = "", db_path: Path = DB_PATH) -> int:
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO expenses (title, amount, category, expense_date, payment_method, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (title.strip(), float(amount), category, expense_date.isoformat(), payment_method, notes.strip()),
        )
        return int(cursor.lastrowid)


def update_expense(expense_id: int, title: str, amount: float, category: str, expense_date: date, payment_method: str, notes: str = "", db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as connection:
        connection.execute(
            """UPDATE expenses SET title = ?, amount = ?, category = ?, expense_date = ?, payment_method = ?, notes = ? WHERE id = ?""",
            (title.strip(), float(amount), category, expense_date.isoformat(), payment_method, notes.strip(), expense_id),
        )


def delete_expense(expense_id: int, db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as connection:
        connection.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))


def import_expenses(frame: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    rows = [
        (row.title, float(row.amount), row.category, row.expense_date.isoformat(), row.payment_method, row.notes)
        for row in frame.itertuples(index=False)
    ]
    with get_connection(db_path) as connection:
        connection.executemany(
            "INSERT INTO expenses (title, amount, category, expense_date, payment_method, notes) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)
