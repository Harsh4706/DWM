from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from database.db import init_db
from services.analytics_service import filter_expenses, monthly_totals, summarize_expenses
from services.clustering_service import elbow_scores, run_clustering, suggested_clusters
from services.expense_service import add_expense, delete_expense, import_expenses, list_expenses, update_expense
from utils.validators import CATEGORIES, PAYMENT_METHODS, validate_expense, validate_import_frame

st.set_page_config(page_title="SmartSpend", page_icon="₹", layout="wide", initial_sidebar_state="expanded")
init_db()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#172321; --muted:#526660; --mint:#dff4e9; --green:#1f8066; --green-dark:#155b49; --line:#d5e3dc; --paper:#f7fbf8; --white:#ffffff; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] { color: var(--ink); font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .stApp { background: radial-gradient(circle at 85% 0%, #e6f7ed 0, var(--paper) 28rem); }
    [data-testid="stMainBlockContainer"] { max-width: 1380px; padding-top: 2.5rem; padding-bottom: 4rem; }
    [data-testid="stSidebar"] { background: #eaf6ef; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.25rem; }
    [data-testid="stSidebar"] * { color: var(--ink) !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: var(--green) !important; }
    [data-testid="stMetric"] { background: rgba(255,255,255,.88); border: 1px solid var(--line); border-radius: 10px; padding: 16px; box-shadow: 0 8px 24px rgba(24, 70, 54, .04); }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { color: var(--green) !important; }
    .hero { background: linear-gradient(120deg, #dff4e9, #f7fbf8 72%); border: 1px solid var(--line); border-radius: 12px; padding: 30px 34px; margin-bottom: 24px; box-shadow: 0 14px 34px rgba(24, 70, 54, .06); }
    .hero h1 { color: var(--ink) !important; font-size: clamp(1.8rem, 3vw, 2.7rem); margin: 8px 0 10px; }
    .eyebrow { text-transform: uppercase; color: var(--green); font-size: .75rem; font-weight: 700; letter-spacing: .1em; }
    .muted { color: var(--muted) !important; }
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--ink) !important; font-weight: 600; }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stDateInput"] input { background: var(--white) !important; color: var(--ink) !important; border-color: var(--line) !important; }
    [data-baseweb="select"] span, [data-baseweb="input"] input, [data-testid="stDateInput"] input { color: var(--ink) !important; }
    [data-baseweb="popover"] * { color: var(--ink) !important; }
    [data-testid="stTabs"] button { color: var(--muted) !important; font-weight: 600; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--green) !important; }
    .stButton > button { min-height: 2.65rem; border-radius: 7px; font-weight: 600; color: var(--ink); border-color: var(--line); background: var(--white); transition: border-color .2s ease, background .2s ease; }
    .stButton > button:hover { border-color: var(--green); color: var(--green-dark); }
    .stButton > button[kind="primary"] { color: white; background: var(--green); border-color: var(--green); }
    .stButton > button[kind="primary"]:hover { color: white; background: var(--green-dark); border-color: var(--green-dark); }
    [data-testid="stRadio"] label { padding: .65rem .7rem; border-radius: 7px; }
    [data-testid="stRadio"] label:hover { background: rgba(31,128,102,.09); }
    [data-testid="stExpander"] { border-color: var(--line); background: rgba(255,255,255,.55); }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); background: rgba(255,255,255,.62); }
    [data-testid="stAlert"] { color: var(--ink) !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value: float) -> str:
    return f"₹{value:,.2f}"


def hero(title: str, subtitle: str, eyebrow: str = "SMARTSPEND") -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><div class="muted">{subtitle}</div></div>', unsafe_allow_html=True)


def date_filter(frame: pd.DataFrame, key: str) -> tuple[date, date]:
    if frame.empty:
        today = date.today()
        return today - timedelta(days=30), today
    minimum = min(frame["expense_date"])
    maximum = max(frame["expense_date"])
    selected = st.date_input("Date range", value=(minimum, maximum), min_value=minimum, max_value=maximum, key=key)
    if isinstance(selected, tuple) and len(selected) == 2:
        return selected
    return minimum, maximum


def dashboard(frame: pd.DataFrame) -> None:
    hero("A clearer view of your money", "Track everyday decisions, then use clustering to see the shape of your spending.")
    with st.container(border=True):
        start, end = date_filter(frame, "dashboard_dates")
        left, right = st.columns(2)
        categories = left.multiselect("Categories", CATEGORIES, key="dashboard_categories")
        methods = right.multiselect("Payment methods", PAYMENT_METHODS, key="dashboard_methods")
    filtered = filter_expenses(frame, start, end, categories, methods)
    summary = summarize_expenses(filtered)
    cards = st.columns(5)
    cards[0].metric("Total spent", money(summary["total"]))
    cards[1].metric("Transactions", summary["count"])
    cards[2].metric("Average", money(summary["average"]))
    cards[3].metric("Highest", money(summary["highest"]))
    cards[4].metric("Top category", summary["top_category"])
    if filtered.empty:
        st.info("No expenses match this view. Add your first expense from Manage Expenses.")
        return

    first, second = st.columns(2)
    category_totals = filtered.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    with first:
        st.subheader("Spending by category")
        st.plotly_chart(px.bar(category_totals, x="category", y="amount", color_discrete_sequence=["#1f8066"], labels={"amount": "Amount (INR)", "category": "Category"}), use_container_width=True)
    with second:
        st.subheader("Expense distribution")
        st.plotly_chart(px.pie(category_totals, names="category", values="amount", hole=.58, color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True)
    st.subheader("Monthly trend")
    trend = monthly_totals(filtered)
    st.plotly_chart(px.line(trend, x="month", y="amount", markers=True, labels={"month": "Month", "amount": "Amount (INR)"}, color_discrete_sequence=["#1f8066"]), use_container_width=True)
    st.subheader("Recent transactions")
    st.dataframe(filtered[["expense_date", "title", "category", "payment_method", "amount"]].head(8), hide_index=True, use_container_width=True, column_config={"amount": st.column_config.NumberColumn("Amount", format="₹ %.2f")})


def manage_expenses(frame: pd.DataFrame) -> None:
    hero("Manage expenses", "Add, correct, remove, or move your records in and out of SmartSpend.")
    add_tab, edit_tab, import_tab = st.tabs(["Add expense", "Edit or delete", "Import CSV"])
    with add_tab:
        with st.form("add_expense_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            title = col1.text_input("Title or description")
            amount = col2.number_input("Amount (INR)", min_value=0.0, step=50.0, format="%.2f")
            category = col1.selectbox("Category", CATEGORIES)
            expense_date = col2.date_input("Date", value=date.today())
            payment_method = col1.selectbox("Payment method", PAYMENT_METHODS)
            notes = col2.text_input("Notes (optional)")
            submitted = st.form_submit_button("Add expense", type="primary")
        if submitted:
            errors = validate_expense(title, amount, category, expense_date, payment_method)
            if errors:
                for error in errors: st.error(error)
            else:
                add_expense(title, amount, category, expense_date, payment_method, notes)
                st.success("Expense added.")
                st.rerun()
    with edit_tab:
        if frame.empty:
            st.info("There are no transactions to edit yet.")
        else:
            selected_id = st.selectbox("Choose a transaction", frame["id"].tolist(), format_func=lambda value: f"#{value} · {frame.loc[frame['id'] == value, 'title'].iloc[0]}")
            selected = frame.loc[frame["id"] == selected_id].iloc[0]
            with st.form("edit_expense_form"):
                col1, col2 = st.columns(2)
                title = col1.text_input("Title", value=selected["title"])
                amount = col2.number_input("Amount (INR)", min_value=0.0, value=float(selected["amount"]), step=50.0)
                category = col1.selectbox("Category", CATEGORIES, index=CATEGORIES.index(selected["category"]))
                expense_date = col2.date_input("Date", value=selected["expense_date"])
                payment_method = col1.selectbox("Payment method", PAYMENT_METHODS, index=PAYMENT_METHODS.index(selected["payment_method"]))
                notes = col2.text_input("Notes", value=selected["notes"] or "")
                save = st.form_submit_button("Save changes", type="primary")
            if save:
                errors = validate_expense(title, amount, category, expense_date, payment_method)
                if errors:
                    for error in errors: st.error(error)
                else:
                    update_expense(selected_id, title, amount, category, expense_date, payment_method, notes)
                    st.success("Expense updated.")
                    st.rerun()
            if st.button("Delete selected expense", type="secondary"):
                st.session_state["confirm_delete"] = selected_id
            if st.session_state.get("confirm_delete") == selected_id:
                st.warning("This permanently removes the selected transaction.")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Confirm delete", type="primary"):
                    delete_expense(selected_id)
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()
                if cancel_col.button("Cancel"):
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()
        st.subheader("All transactions")
        search = st.text_input("Search by title", key="manage_search")
        shown = frame[frame["title"].str.contains(search, case=False, na=False)] if search else frame
        st.dataframe(shown, hide_index=True, use_container_width=True, column_config={"amount": st.column_config.NumberColumn("Amount", format="₹ %.2f")})
    with import_tab:
        st.subheader("Move your data")
        export = frame.drop(columns=["created_at"], errors="ignore").copy()
        st.download_button("Download expenses CSV", export.to_csv(index=False).encode("utf-8"), "smartspend-expenses.csv", "text/csv", disabled=export.empty)
        st.divider()
        upload = st.file_uploader("Upload a CSV", type="csv")
        if upload:
            source = pd.read_csv(upload)
            valid, invalid, errors = validate_import_frame(source)
            if errors:
                for error in errors: st.error(error)
            else:
                st.write("Preview of valid rows")
                st.dataframe(valid, hide_index=True, use_container_width=True)
                if not invalid.empty: st.warning(f"{len(invalid)} row(s) will not be imported because they failed validation.")
                if st.button(f"Import {len(valid)} valid row(s)", type="primary", disabled=valid.empty):
                    import_expenses(valid)
                    st.success("CSV records imported.")
                    st.rerun()


def analysis(frame: pd.DataFrame) -> None:
    hero("Find the shape of your spending", "Compare amount patterns, test the right number of groups, and understand what each cluster contains.", "MACHINE LEARNING")
    if len(frame) < 2:
        st.info("Add at least two expenses to begin clustering.")
        return
    start, end = date_filter(frame, "analysis_dates")
    selected = filter_expenses(frame, start, end)
    if len(selected) < 2:
        st.info("This date range contains fewer than two expenses. Widen the range to analyze a pattern.")
        return
    controls_left, controls_right = st.columns([1.15, 1], gap="large")
    with controls_left:
        include_context = st.toggle("Use context-aware clustering", value=True, help="Adds day of week and category frequency alongside amount. All features are normalized before clustering.")
        st.caption("Context-aware mode uses amount, day of week, and how often each category appears. Amount-only mode is the simplest baseline.")
    with controls_right:
        if len(selected) == 2:
            clusters = 2
            st.info("Two transactions selected, so two clusters are used.")
        else:
            clusters = st.slider("Number of clusters", min_value=2, max_value=min(5, len(selected)), value=min(3, len(selected)))
    with st.expander("Choose a cluster count with the Elbow Method", expanded=False):
        st.write("The elbow chart compares model compactness for each possible cluster count. The suggested value is where adding another group starts to provide less improvement.")
        if len(selected) < 3:
            st.info("Add at least three transactions to calculate an elbow curve.")
        else:
            try:
                scores = elbow_scores(selected, max_clusters=5, include_context=include_context)
                recommendation = suggested_clusters(scores)
                elbow_col, elbow_note = st.columns([1.6, 1])
                with elbow_col:
                    st.plotly_chart(px.line(scores, x="clusters", y="inertia", markers=True, labels={"clusters": "Number of clusters (k)", "inertia": "Within-cluster error"}, color_discrete_sequence=["#1f8066"]), use_container_width=True, config={"displayModeBar": False})
                with elbow_note:
                    st.metric("Suggested k", recommendation)
                    st.caption("Use this as a starting point, then compare the cluster summaries for interpretability.")
            except ValueError as error:
                st.warning(str(error))
    st.caption("StandardScaler normalizes every selected feature before K-Means assigns groups. Cluster names describe measured ranges, not financial judgments.")
    if st.button("Run analysis", type="primary"):
        try:
            labeled, summary = run_clustering(selected, clusters, include_context=include_context)
            st.session_state["cluster_result"] = (labeled, summary)
        except ValueError as error:
            st.error(str(error))
    if "cluster_result" not in st.session_state:
        st.info("Choose the number of groups and run the analysis.")
        return
    labeled, summary = st.session_state["cluster_result"]
    cards = st.columns(4)
    cards[0].metric("Clusters", len(summary))
    cards[1].metric("Amount analyzed", money(float(labeled["amount"].sum())))
    cards[2].metric("Average transaction", money(float(labeled["amount"].mean())))
    silhouette = summary.attrs.get("silhouette_score")
    cards[3].metric("Silhouette score", "Not available" if silhouette is None else f"{silhouette:.2f}", help="Higher values generally mean cleaner separation. This score is descriptive, not a guarantee of financial insight.")
    chart_col, bar_col = st.columns(2)
    with chart_col:
        st.subheader("Transactions by cluster")
        jittered = labeled.assign(transaction_number=range(1, len(labeled) + 1))
        st.plotly_chart(px.scatter(jittered, x="transaction_number", y="amount", color="cluster_name", hover_data=["title", "category", "amount"], labels={"transaction_number": "Transaction order", "amount": "Amount (INR)", "cluster_name": "Group"}, color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True, config={"displayModeBar": False})
    with bar_col:
        st.subheader("Share of spending")
        st.plotly_chart(px.bar(summary, x="description", y="share_of_spending", color="description", labels={"share_of_spending": "Share (%)", "description": "Amount group"}, text_auto=".1f", color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True, config={"displayModeBar": False})
    st.subheader("Cluster summary")
    st.dataframe(summary.drop(columns=["cluster"], errors="ignore"), hide_index=True, use_container_width=True, column_config={"average_amount": st.column_config.NumberColumn("Average", format="₹ %.2f"), "minimum_amount": st.column_config.NumberColumn("Minimum", format="₹ %.2f"), "maximum_amount": st.column_config.NumberColumn("Maximum", format="₹ %.2f"), "total_amount": st.column_config.NumberColumn("Total", format="₹ %.2f"), "share_of_spending": st.column_config.NumberColumn("Share", format="%.1f%%"), "center_amount": st.column_config.NumberColumn("Center amount", format="₹ %.2f")})
    st.subheader("Labeled transactions")
    st.dataframe(labeled[["expense_date", "title", "category", "amount", "cluster"]], hide_index=True, use_container_width=True)
    largest = summary.iloc[-1]
    most_common = summary.loc[summary["transactions"].idxmax()]
    st.info(f"K-Means insight: the largest average-amount group is {largest['description'].lower()} with an average of {money(largest['average_amount'])}. Most transactions fall in the {most_common['description'].lower()} group.")


frame = list_expenses()
with st.sidebar:
    st.markdown("# SmartSpend")
    st.markdown('<div class="muted">Personal expense analyzer</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    page = st.radio("Navigate", ["Dashboard", "Manage Expenses", "Spending Analysis"], label_visibility="collapsed")
    st.divider()
    st.metric("Recorded expenses", len(frame))
    st.markdown('<div class="muted">Local-first · SQLite powered</div>', unsafe_allow_html=True)

if page == "Dashboard": dashboard(frame)
elif page == "Manage Expenses": manage_expenses(frame)
else: analysis(frame)
