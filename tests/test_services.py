from datetime import date

import pandas as pd
import pytest

from database.db import init_db
from services.analytics_service import summarize_expenses
from services.clustering_service import elbow_scores, run_clustering, suggested_clusters
from services.expense_service import add_expense, delete_expense, list_expenses, update_expense
from utils.validators import validate_import_frame


def test_crud_persists_records(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path)
    expense_id = add_expense("Lunch", 250, "Food", date(2026, 1, 2), "UPI", db_path=db_path)
    assert list_expenses(db_path).iloc[0]["title"] == "Lunch"
    update_expense(expense_id, "Team lunch", 300, "Food", date(2026, 1, 2), "Cash", db_path=db_path)
    assert list_expenses(db_path).iloc[0]["amount"] == 300
    delete_expense(expense_id, db_path=db_path)
    assert list_expenses(db_path).empty


def test_summary_uses_actual_values():
    frame = pd.DataFrame({"amount": [100.0, 500.0], "category": ["Food", "Travel"]})
    summary = summarize_expenses(frame)
    assert summary == {"total": 600.0, "count": 2, "average": 300.0, "highest": 500.0, "top_category": "Travel"}


def test_import_validation_reports_invalid_rows():
    source = pd.DataFrame({"title": ["Valid", ""], "amount": [100, -2], "category": ["Food", "Food"], "expense_date": ["2026-01-01", "bad"], "payment_method": ["UPI", "UPI"]})
    valid, invalid, errors = validate_import_frame(source)
    assert not errors
    assert len(valid) == 1
    assert len(invalid) == 1


def test_clustering_returns_summary():
    frame = pd.DataFrame({"title": ["A", "B", "C", "D"], "amount": [10, 20, 100, 120], "expense_date": [date.today()] * 4})
    labeled, summary = run_clustering(frame, 2)
    assert labeled["cluster"].nunique() == 2
    assert summary["transactions"].sum() == 4
    assert summary["total_amount"].sum() == 250


def test_clustering_rejects_identical_amounts():
    frame = pd.DataFrame({"amount": [10, 10, 10]})
    with pytest.raises(ValueError, match="distinct"):
        run_clustering(frame, 2)


def test_five_clusters_have_ordered_amount_descriptions():
    frame = pd.DataFrame({"amount": [10, 20, 30, 40, 50]})
    _, summary = run_clustering(frame, 5)
    assert summary["description"].tolist() == [
        "Very low-amount transactions",
        "Lower-amount transactions",
        "Medium-amount transactions",
        "Higher-amount transactions",
        "Very high-amount transactions",
    ]


def test_context_clustering_adds_quality_metrics_and_centers():
    frame = pd.DataFrame({
        "amount": [10, 20, 100, 120],
        "category": ["Food", "Food", "Travel", "Travel"],
        "expense_date": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 4)],
    })
    labeled, summary = run_clustering(frame, 2, include_context=True)
    assert "cluster_name" in labeled
    assert "center_amount" in summary
    assert "share_of_spending" in summary
    assert summary.attrs["silhouette_score"] is not None


def test_elbow_scores_return_recommendation():
    frame = pd.DataFrame({"amount": [10, 20, 30, 100, 110, 120]})
    scores = elbow_scores(frame, max_clusters=5, include_context=False)
    assert scores["clusters"].tolist() == [2, 3, 4, 5]
    assert suggested_clusters(scores) in {2, 3, 4, 5}
