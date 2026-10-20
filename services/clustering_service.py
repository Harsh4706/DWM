from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _cluster_descriptions(cluster_count: int) -> list[str]:
    descriptions = {
        2: ["Lower-amount transactions", "Higher-amount transactions"],
        3: ["Lower-amount transactions", "Mid-range transactions", "Higher-amount transactions"],
        4: ["Very low-amount transactions", "Lower-amount transactions", "Higher-amount transactions", "Very high-amount transactions"],
        5: ["Very low-amount transactions", "Lower-amount transactions", "Medium-amount transactions", "Higher-amount transactions", "Very high-amount transactions"],
    }
    return descriptions[cluster_count]


def _feature_frame(frame: pd.DataFrame, include_context: bool) -> tuple[pd.DataFrame, list[str]]:
    features = pd.DataFrame({"amount": frame["amount"].astype(float)})
    if include_context:
        required = {"expense_date", "category"}
        if not required.issubset(frame.columns):
            raise ValueError("Context-aware clustering requires expense dates and categories.")
        dates = pd.to_datetime(frame["expense_date"])
        category_frequency = frame["category"].map(frame["category"].value_counts()).astype(float)
        features["day_of_week"] = dates.dt.dayofweek.astype(float)
        features["category_frequency"] = category_frequency
    return features, list(features.columns)


def _fit_elbow(frame: pd.DataFrame, clusters: int, include_context: bool) -> float:
    features, _ = _feature_frame(frame, include_context)
    scaled = StandardScaler().fit_transform(features)
    return float(KMeans(n_clusters=clusters, random_state=42, n_init=10).fit(scaled).inertia_)


def elbow_scores(frame: pd.DataFrame, max_clusters: int = 5, include_context: bool = True) -> pd.DataFrame:
    if len(frame) < 3:
        raise ValueError("At least three transactions are required for an elbow analysis.")
    upper = min(max_clusters, len(frame) - 1)
    values = [{"clusters": clusters, "inertia": _fit_elbow(frame, clusters, include_context)} for clusters in range(2, upper + 1)]
    return pd.DataFrame(values)


def suggested_clusters(scores: pd.DataFrame) -> int:
    if len(scores) < 3:
        return int(scores.loc[scores["inertia"].idxmin(), "clusters"])
    x = scores["clusters"].to_numpy(dtype=float)
    y = scores["inertia"].to_numpy(dtype=float)
    points = np.column_stack((x, y))
    start, end = points[0], points[-1]
    direction = end - start
    distances = np.abs(direction[0] * (start[1] - points[:, 1]) - direction[1] * (start[0] - points[:, 0])) / np.linalg.norm(direction)
    return int(scores.iloc[int(np.argmax(distances))]["clusters"])


def run_clustering(frame: pd.DataFrame, clusters: int, include_context: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) < clusters:
        raise ValueError(f"At least {clusters} transactions are required.")
    if not np.isfinite(frame["amount"].to_numpy(dtype=float)).all():
        raise ValueError("Transaction amounts contain invalid values.")
    if frame["amount"].nunique() < clusters:
        raise ValueError("There are not enough distinct amounts for these clusters.")

    result = frame.copy().reset_index(drop=True)
    features, feature_names = _feature_frame(result, include_context)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    result["cluster"] = model.fit_predict(scaled)
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=feature_names)
    centers["cluster"] = range(len(centers))
    summary = result.groupby("cluster", as_index=False).agg(
        transactions=("amount", "size"), average_amount=("amount", "mean"),
        minimum_amount=("amount", "min"), maximum_amount=("amount", "max"), total_amount=("amount", "sum"),
    ).sort_values("average_amount").reset_index(drop=True)
    summary["share_of_spending"] = summary["total_amount"] / summary["total_amount"].sum() * 100
    summary = summary.merge(centers.rename(columns={"amount": "center_amount", "day_of_week": "center_day_of_week", "category_frequency": "center_category_frequency"}), on="cluster", how="left")
    description_by_cluster = dict(zip(summary["cluster"], _cluster_descriptions(len(summary))))
    summary["description"] = summary["cluster"].map(description_by_cluster)
    summary = summary.sort_values("average_amount").reset_index(drop=True)
    result["cluster_name"] = result["cluster"].map(description_by_cluster)
    summary.attrs["silhouette_score"] = float(silhouette_score(scaled, result["cluster"])) if len(summary) > 1 and len(result) > clusters else None
    summary.attrs["inertia"] = float(model.inertia_)
    summary.attrs["feature_names"] = feature_names
    return result, summary
