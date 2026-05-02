"""Dual-recall drug recommendation pipeline.

Pipeline stages:
1) Semantic recall from raw symptom description with MedBERT embeddings.
2) Label-confidence recall from disease/symptom labels.
3) Weighted fusion of recall results.
4) Cross-encoder reranking.
5) Business-factor adjustment (rating/reviews/price if present).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Import pipeline config (project root level)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from static_module.pipeline_config import cfg

from .cross_encoder_reranker import CrossEncoderReranker
from .recommendation_pipeline import (
    build_query_text,
    normalize_symptom_name,
    prepare_drug_dataframe,
    safe_cosine_scores,
)


@dataclass
class PipelineTrace:
    """Data tracing / instrumentation for a single recommend() call."""

    # Timing (seconds)
    time_semantic_recall: float = 0.0
    time_label_recall: float = 0.0
    time_fuse: float = 0.0
    time_rerank: float = 0.0
    time_total: float = 0.0

    # Candidate counts
    count_semantic_candidates: int = 0
    count_label_candidates: int = 0
    count_fused_candidates: int = 0
    count_final: int = 0

    # Score distributions (top-k results)
    score_semantic_mean: float = 0.0
    score_label_mean: float = 0.0
    score_cross_encoder_mean: float = 0.0
    score_business_mean: float = 0.0
    score_final_mean: float = 0.0
    score_final_max: float = 0.0
    score_final_min: float = 0.0

    # Input metadata
    query_length: int = 0
    num_disease_labels: int = 0
    num_symptom_labels: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "time_semantic_recall_ms": round(self.time_semantic_recall * 1000, 2),
            "time_label_recall_ms": round(self.time_label_recall * 1000, 2),
            "time_fuse_ms": round(self.time_fuse * 1000, 2),
            "time_rerank_ms": round(self.time_rerank * 1000, 2),
            "time_total_ms": round(self.time_total * 1000, 2),
            "count_semantic_candidates": self.count_semantic_candidates,
            "count_label_candidates": self.count_label_candidates,
            "count_fused_candidates": self.count_fused_candidates,
            "count_final": self.count_final,
            "score_semantic_mean": round(self.score_semantic_mean, 4),
            "score_label_mean": round(self.score_label_mean, 4),
            "score_cross_encoder_mean": round(self.score_cross_encoder_mean, 4),
            "score_business_mean": round(self.score_business_mean, 4),
            "score_final_mean": round(self.score_final_mean, 4),
            "score_final_max": round(self.score_final_max, 4),
            "score_final_min": round(self.score_final_min, 4),
            "query_length": self.query_length,
            "num_disease_labels": self.num_disease_labels,
            "num_symptom_labels": self.num_symptom_labels,
        }


@dataclass(frozen=True)
class WeightedLabel:
    name: str
    confidence: float


def _min_max_normalize(values: pd.Series) -> pd.Series:
    if len(values) == 0:
        return values.astype(float)
    values = values.astype(float)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-12:
        return pd.Series(np.ones(len(values), dtype=np.float32), index=values.index)
    return (values - min_value) / (max_value - min_value)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def parse_weighted_labels(items: Iterable[Any], normalize_symptom: bool = False) -> list[WeightedLabel]:
    parsed: list[WeightedLabel] = []
    for item in items:
        name = ""
        confidence = 0.0
        if isinstance(item, dict):
            if "name" in item:
                name = str(item.get("name", "")).strip()
                confidence = _to_float(item.get("confidence"), default=0.0)
            elif "label" in item:
                name = str(item.get("label", "")).strip()
                confidence = _to_float(item.get("confidence"), default=0.0)
            elif len(item) == 1:
                key = next(iter(item))
                name = str(key).strip()
                confidence = _to_float(item[key], default=0.0)
        if not name:
            continue
        if normalize_symptom:
            name = normalize_symptom_name(name)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        parsed.append(WeightedLabel(name=name, confidence=confidence))
    return parsed


def _build_confidence_map(labels: list[WeightedLabel]) -> dict[str, float]:
    confidence_map: dict[str, float] = {}
    for item in labels:
        confidence_map[item.name] = max(confidence_map.get(item.name, 0.0), item.confidence)
    return confidence_map


def _build_combined_query(
    symptom_text: str, diseases: list[WeightedLabel], symptoms: list[WeightedLabel]
) -> str:
    """Dynamically concatenate symptom_text, disease labels, and symptom labels.

    Used by both semantic_recall and cross-encoder reranking to build a richer
    query from all available inputs.
    """
    disease_terms = [x.name for x in diseases]
    symptom_terms = [x.name for x in symptoms]
    label_text = build_query_text(disease_terms, symptom_terms)
    symptom_text = str(symptom_text or "").strip()
    if symptom_text and label_text != "[MASK]":
        return f"{symptom_text} {label_text}"
    if symptom_text:
        return symptom_text
    return label_text


class DualRecallDrugRecommender:
    def __init__(
        self,
        df: pd.DataFrame,
        drug_embeddings: np.ndarray,
        embedding_engine,
        cross_encoder_reranker: CrossEncoderReranker,
    ):
        if len(df) != drug_embeddings.shape[0]:
            raise ValueError(
                "df and drug_embeddings should align by row index. "
                f"Got len(df)={len(df)}, len(drug_embeddings)={len(drug_embeddings)}"
            )
        if "semantic_text" not in df.columns or "disease_key_list" not in df.columns:
            self.df = prepare_drug_dataframe(df)
        else:
            self.df = df.copy()
        self.drug_embeddings = np.asarray(drug_embeddings, dtype=np.float32)
        self.embedding_engine = embedding_engine
        self.cross_encoder = cross_encoder_reranker

    def semantic_recall(self, symptom_text: str, top_k: int = 300) -> pd.DataFrame:
        query_text = str(symptom_text or "").strip() or "[MASK]"
        query_vec = self.embedding_engine.encode_query(query_text).astype(np.float32)

        # Support both legacy 3D (N,2,768) and current 2D (N,768) embeddings
        if self.drug_embeddings.ndim == 3:
            emb_2d = self.drug_embeddings[:, 1, :]  # View 1: disease_description
        else:
            emb_2d = self.drug_embeddings
        semantic_scores = safe_cosine_scores(emb_2d, query_vec)
        candidates = self.df.copy()
        candidates["semantic_score_raw"] = semantic_scores
        candidates = candidates.sort_values("semantic_score_raw", ascending=False).head(top_k).copy()
        candidates["semantic_score"] = _min_max_normalize(candidates["semantic_score_raw"])
        return candidates

    def label_recall(
        self,
        disease_items: Iterable[Any],
        symptom_items: Iterable[Any],
        top_k: int = 300,
        disease_weight: float | None = None,
        symptom_weight: float | None = None,
    ) -> pd.DataFrame:
        if disease_weight is None:
            disease_weight = cfg.label_disease_weight
        if symptom_weight is None:
            symptom_weight = cfg.label_symptom_weight
        diseases = parse_weighted_labels(disease_items, normalize_symptom=False)
        symptoms = parse_weighted_labels(symptom_items, normalize_symptom=True)
        disease_conf = _build_confidence_map(diseases)
        symptom_conf = _build_confidence_map(symptoms)

        def score_row(row: pd.Series) -> tuple[float, float]:
            disease_overlap = float(
                sum(disease_conf.get(str(name).strip(), 0.0) for name in row["disease_key_list"])
            )
            symptom_overlap = float(
                sum(
                    symptom_conf.get(normalize_symptom_name(name), 0.0)
                    for name in row["symptom_list_normalized"]
                )
            )
            return disease_overlap, symptom_overlap

        scores = self.df.apply(score_row, axis=1, result_type="expand")
        scores.columns = ["disease_conf_overlap", "symptom_conf_overlap"]
        candidates = pd.concat([self.df.copy(), scores], axis=1)
        candidates = candidates[
            (candidates["disease_conf_overlap"] > 0) | (candidates["symptom_conf_overlap"] > 0)
        ].copy()
        if len(candidates) == 0:
            return candidates
        candidates["label_score_raw"] = (
            disease_weight * candidates["disease_conf_overlap"]
            + symptom_weight * candidates["symptom_conf_overlap"]
        )
        candidates = candidates.sort_values("label_score_raw", ascending=False).head(top_k).copy()
        candidates["label_score"] = _min_max_normalize(candidates["label_score_raw"])
        return candidates

    def fuse_recalls(
        self,
        semantic_candidates: pd.DataFrame,
        label_candidates: pd.DataFrame,
        semantic_weight: float = 0.5,
        label_weight: float = 0.5,
        top_k: int = 300,
    ) -> pd.DataFrame:
        semantic_scores = (
            semantic_candidates[["semantic_score"]].copy()
            if len(semantic_candidates) > 0
            else pd.DataFrame(columns=["semantic_score"])
        )
        label_scores = (
            label_candidates[
                [
                    "label_score",
                    "disease_conf_overlap",
                    "symptom_conf_overlap",
                ]
            ].copy()
            if len(label_candidates) > 0
            else pd.DataFrame(
                columns=["label_score", "disease_conf_overlap", "symptom_conf_overlap"]
            )
        )

        fused = self.df.copy()
        fused = fused.join(semantic_scores, how="left")
        fused = fused.join(label_scores, how="left")

        fused["semantic_score"] = fused["semantic_score"].fillna(0.0)
        fused["label_score"] = fused["label_score"].fillna(0.0)
        fused["disease_conf_overlap"] = fused["disease_conf_overlap"].fillna(0.0)
        fused["symptom_conf_overlap"] = fused["symptom_conf_overlap"].fillna(0.0)
        fused = fused[
            (fused["semantic_score"] > 0) | (fused["label_score"] > 0)
        ].copy()
        if len(fused) == 0:
            return fused
        fused["recall_fused_score"] = (
            semantic_weight * fused["semantic_score"]
            + label_weight * fused["label_score"]
        )
        fused = fused.sort_values("recall_fused_score", ascending=False).head(top_k).copy()
        return fused

    def _business_score(self, candidates: pd.DataFrame) -> pd.Series:
        rating_score = (
            _min_max_normalize(candidates["avg_rating"].fillna(0.0))
            if "avg_rating" in candidates.columns
            else pd.Series(np.zeros(len(candidates), dtype=np.float32), index=candidates.index)
        )
        reviews_score = (
            _min_max_normalize(np.log1p(candidates["total_reviews"].fillna(0.0)))
            if "total_reviews" in candidates.columns
            else pd.Series(np.zeros(len(candidates), dtype=np.float32), index=candidates.index)
        )

        if "price" in candidates.columns:
            price_raw = pd.to_numeric(candidates["price"], errors="coerce").fillna(np.nan)
            if price_raw.notna().any():
                price_score = 1.0 - _min_max_normalize(price_raw.fillna(price_raw.max()))
            else:
                price_score = pd.Series(np.zeros(len(candidates), dtype=np.float32), index=candidates.index)
        else:
            price_score = pd.Series(np.zeros(len(candidates), dtype=np.float32), index=candidates.index)

        return (cfg.biz_weight_rating * rating_score
                + cfg.biz_weight_reviews * reviews_score
                + cfg.biz_weight_price * price_score)

    def rerank(
        self,
        fused_candidates: pd.DataFrame,
        symptom_text: str,
        disease_items: Iterable[Any],
        symptom_items: Iterable[Any],
        top_k: int = 20,
        final_weight_recall: float | None = None,
        final_weight_cross_encoder: float | None = None,
        final_weight_business: float | None = None,
    ) -> pd.DataFrame:
        if final_weight_recall is None:
            final_weight_recall = cfg.final_weight_recall
        if final_weight_cross_encoder is None:
            final_weight_cross_encoder = cfg.final_weight_cross_encoder
        if final_weight_business is None:
            final_weight_business = cfg.final_weight_business
        if len(fused_candidates) == 0:
            return fused_candidates

        diseases = parse_weighted_labels(disease_items, normalize_symptom=False)
        symptoms = parse_weighted_labels(symptom_items, normalize_symptom=True)
        query_text = _build_combined_query(symptom_text, diseases, symptoms)
        cross_scores = self.cross_encoder.score_pairs(
            query=query_text,
            candidate_texts=fused_candidates["semantic_text"].tolist(),
        )
        reranked = fused_candidates.copy()
        reranked["cross_encoder_score"] = cross_scores
        reranked["cross_encoder_score"] = _min_max_normalize(reranked["cross_encoder_score"])
        reranked["business_score"] = self._business_score(reranked)
        reranked["final_score"] = (
            final_weight_recall * reranked["recall_fused_score"]
            + final_weight_cross_encoder * reranked["cross_encoder_score"]
            + final_weight_business * reranked["business_score"]
        )
        reranked = reranked.sort_values("final_score", ascending=False).head(top_k).copy()
        return reranked

    def recommend(
        self,
        symptom_text: str,
        disease_items: Iterable[Any],
        symptom_items: Iterable[Any],
        recall_top_k_each: int | None = None,
        fused_top_k: int | None = None,
        top_k: int | None = None,
        recall_weight_semantic: float | None = None,
        recall_weight_label: float | None = None,
        enable_trace: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, PipelineTrace]:
        if recall_top_k_each is None:
            recall_top_k_each = cfg.recall_top_k_each
        if fused_top_k is None:
            fused_top_k = cfg.fused_top_k
        if top_k is None:
            top_k = cfg.top_k
        if recall_weight_semantic is None:
            recall_weight_semantic = cfg.recall_weight_semantic
        if recall_weight_label is None:
            recall_weight_label = cfg.recall_weight_label
        trace = PipelineTrace() if enable_trace else None
        t_total_start = time.perf_counter()

        # Materialize iterables so they can be reused
        disease_items_list = list(disease_items)
        symptom_items_list = list(symptom_items)

        if trace:
            trace.query_length = len(str(symptom_text or "").strip())
            trace.num_disease_labels = len(disease_items_list)
            trace.num_symptom_labels = len(symptom_items_list)

        # --- Build enriched query (symptom_text + disease/symptom labels) ---
        diseases_parsed = parse_weighted_labels(disease_items_list, normalize_symptom=False)
        symptoms_parsed = parse_weighted_labels(symptom_items_list, normalize_symptom=True)
        enriched_query = _build_combined_query(symptom_text, diseases_parsed, symptoms_parsed)

        # --- LLM query expansion (optional) ---
        if cfg.enable_llm_query_expansion:
            from .query_expander import expand_query_with_llm
            enriched_query = expand_query_with_llm(enriched_query)

        # --- Semantic recall ---
        t0 = time.perf_counter()
        semantic_candidates = self.semantic_recall(symptom_text=enriched_query, top_k=recall_top_k_each)
        if trace:
            trace.time_semantic_recall = time.perf_counter() - t0
            trace.count_semantic_candidates = len(semantic_candidates)

        # --- Label recall ---
        t0 = time.perf_counter()
        label_candidates = self.label_recall(
            disease_items=disease_items_list,
            symptom_items=symptom_items_list,
            top_k=recall_top_k_each,
        )
        if trace:
            trace.time_label_recall = time.perf_counter() - t0
            trace.count_label_candidates = len(label_candidates)

        # --- Fusion ---
        t0 = time.perf_counter()
        fused_candidates = self.fuse_recalls(
            semantic_candidates=semantic_candidates,
            label_candidates=label_candidates,
            semantic_weight=recall_weight_semantic,
            label_weight=recall_weight_label,
            top_k=fused_top_k,
        )
        if trace:
            trace.time_fuse = time.perf_counter() - t0
            trace.count_fused_candidates = len(fused_candidates)

        # --- Rerank ---
        t0 = time.perf_counter()
        result = self.rerank(
            fused_candidates=fused_candidates,
            symptom_text=symptom_text,
            disease_items=disease_items_list,
            symptom_items=symptom_items_list,
            top_k=top_k,
        )
        if trace:
            trace.time_rerank = time.perf_counter() - t0
            trace.count_final = len(result)
            trace.time_total = time.perf_counter() - t_total_start
            # Score stats from final result
            if len(result) > 0:
                trace.score_semantic_mean = float(result["semantic_score"].mean()) if "semantic_score" in result.columns else 0.0
                trace.score_label_mean = float(result["label_score"].mean()) if "label_score" in result.columns else 0.0
                trace.score_cross_encoder_mean = float(result["cross_encoder_score"].mean())
                trace.score_business_mean = float(result["business_score"].mean())
                trace.score_final_mean = float(result["final_score"].mean())
                trace.score_final_max = float(result["final_score"].max())
                trace.score_final_min = float(result["final_score"].min())

        if trace:
            return result, trace
        return result
