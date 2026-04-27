"""Evaluation module for drug recommendation system."""

try:  # lazy — depends on openai, which is optional
    from .generate_eval_dataset_llm import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass
from .metrics import (
    dcg_at_k,
    evaluate_batch,
    evaluate_single_query,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

from .DrugRecommenderService import DrugRecommendationService

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "hit_at_k",
    "reciprocal_rank",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_single_query",
    "evaluate_batch",
    "generate_eval_dataset_llm_main",
    "DrugRecommenderService",
]
