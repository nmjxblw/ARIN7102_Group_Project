"""
Reusable package exports for the embedding / retrieval pipeline.

Keep this package lightweight so notebook imports do not eagerly pull
heavy model dependencies unless they are explicitly needed.
"""

from .evaluation import EVALUATION_SCOPE_NOTE
from .recommendation_pipeline import (
    baseline_filter,
    build_query_text,
    build_semantic_text,
    clean_text,
    normalize_symptom_name,
    parse_list_like,
    prepare_drug_dataframe,
    query_drugs_hybrid,
    safe_cosine_scores,
)
from .cross_encoder_reranker import CrossEncoderReranker
from .dual_recall_pipeline import DualRecallDrugRecommender, PipelineTrace, WeightedLabel, parse_weighted_labels

__all__ = [
    "EVALUATION_SCOPE_NOTE",
    "baseline_filter",
    "build_query_text",
    "build_semantic_text",
    "clean_text",
    "normalize_symptom_name",
    "parse_list_like",
    "prepare_drug_dataframe",
    "query_drugs_hybrid",
    "safe_cosine_scores",
    "CrossEncoderReranker",
    "DualRecallDrugRecommender",
    "PipelineTrace",
    "WeightedLabel",
    "parse_weighted_labels",
]
