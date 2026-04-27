"""
Evaluation metrics for drug recommendation system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def precision_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Precision@K: proportion of relevant items in top-K recommendations."""
    if k <= 0 or not recommended:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Recall@K: proportion of relevant items retrieved in top-K."""
    if not relevant or k <= 0:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant)


def hit_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Hit@K: 1 if at least one relevant item in top-K, else 0."""
    if k <= 0 or not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    relevant_set = set(relevant)
    return 1.0 if any(item in relevant_set for item in top_k) else 0.0


def reciprocal_rank(recommended: list[str], relevant: list[str]) -> float:
    """Reciprocal rank of the first relevant item."""
    if not recommended or not relevant:
        return 0.0
    relevant_set = set(relevant)
    for rank, item in enumerate(recommended, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def dcg_at_k(recommended: list[str], relevance_scores: dict[str, float], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    if k <= 0 or not recommended:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        rel = relevance_scores.get(item, 0.0)
        dcg += rel / np.log2(i + 1)
    return dcg


def ndcg_at_k(recommended: list[str], relevance_scores: dict[str, float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    dcg = dcg_at_k(recommended, relevance_scores, k)
    ideal_items = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    ideal_recommended = [item for item, _ in ideal_items]
    idcg = dcg_at_k(ideal_recommended, relevance_scores, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_single_query(
    recommended: list[str],
    relevant: list[str],
    relevance_scores: dict[str, float] | None = None,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate a single query with multiple metrics."""
    if k_values is None:
        k_values = [5, 10, 20]

    metrics = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(recommended, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(recommended, relevant, k)
        metrics[f"hit@{k}"] = hit_at_k(recommended, relevant, k)
        if relevance_scores:
            metrics[f"ndcg@{k}"] = ndcg_at_k(recommended, relevance_scores, k)

    metrics["mrr"] = reciprocal_rank(recommended, relevant)
    return metrics


def evaluate_batch(
    results: list[dict],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Evaluate a batch of queries.

    Args:
        results: List of dicts with keys:
            - 'recommended': list of recommended drug names
            - 'relevant': list of relevant drug names
            - 'relevance_scores': dict of {drug_name: score} (optional)
        k_values: List of K values to evaluate

    Returns:
        Dict of averaged metrics
    """
    if not results:
        return {}

    all_metrics = []
    for result in results:
        metrics = evaluate_single_query(
            recommended=result["recommended"],
            relevant=result["relevant"],
            relevance_scores=result.get("relevance_scores"),
            k_values=k_values,
        )
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)
    return df.mean().to_dict()
