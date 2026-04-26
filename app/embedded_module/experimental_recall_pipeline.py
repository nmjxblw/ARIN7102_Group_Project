"""Experimental local drug recall pipeline.

This module is intentionally separate from the production FastAPI recommender.
It implements the candidate-union and deterministic scoring plan used for
offline ablation experiments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .drug_ranker import LocalDrugRanker
from .drug_recall_index import DrugRecallIndex
from .label_adapter import NormalizedLabel, confidence_map, parse_labels
from .recommendation_pipeline import safe_cosine_scores


ABLATION_MODES = (
    "label_idf_only",
    "bm25_only",
    "dense_only",
    "label_bm25",
    "label_bm25_dense",
    "candidate_union",
    "candidate_union_no_prior",
    "candidate_union_no_bm25",
    "candidate_union_no_prior_no_bm25",
    "local_ranker",
    "label_core_rerank",
    "verified_learned_rerank",
    "verified_xgb_ranker",
)

MODE_STAGE_MAP = {
    "label_idf_only": ("disease", "strict", "symptom"),
    "bm25_only": ("bm25",),
    "dense_only": ("dense",),
    "label_bm25": ("disease", "strict", "symptom", "bm25"),
    "label_bm25_dense": ("disease", "strict", "symptom", "bm25", "dense"),
    "candidate_union": ("disease", "strict", "symptom", "bm25", "dense", "prior"),
    "candidate_union_no_prior": ("disease", "strict", "symptom", "bm25", "dense"),
    "candidate_union_no_bm25": ("disease", "strict", "symptom", "dense", "prior"),
    "candidate_union_no_prior_no_bm25": ("disease", "strict", "symptom", "dense"),
    "local_ranker": ("disease", "strict", "symptom", "bm25", "dense", "prior"),
    "label_core_rerank": ("disease", "strict", "symptom"),
    "verified_learned_rerank": ("disease", "strict", "symptom"),
    "verified_xgb_ranker": ("disease", "strict", "symptom"),
}


@dataclass
class ExperimentalTrace:
    normalized_diseases: list[dict] = field(default_factory=list)
    normalized_symptoms: list[dict] = field(default_factory=list)
    candidate_counts: dict[str, int] = field(default_factory=dict)
    final_union_size: int = 0
    embedding_manifest: dict = field(default_factory=dict)
    fallback_mode: bool = False
    ranker_used: bool = False
    ranker_model_type: str = ""
    ranker_train_source: str = ""
    mode: str = "candidate_union"
    stage_candidate_names: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self, *, include_candidates: bool = False) -> dict:
        payload = asdict(self)
        if not include_candidates:
            payload.pop("stage_candidate_names", None)
        return payload


class ExperimentalDrugRecallPipeline:
    def __init__(
        self,
        index: DrugRecallIndex,
        *,
        ranker: LocalDrugRanker | None = None,
        query_encoder: Any | None = None,
    ):
        self.index = index
        self.ranker = ranker
        self.query_encoder = query_encoder

    def recommend(
        self,
        symptom_text: str,
        disease_items: list[dict] | list[Any] | None,
        symptom_items: list[dict] | list[Any] | None,
        *,
        top_k: int = 20,
        pool_size: int = 1000,
        mode: str = "candidate_union",
        return_trace: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, ExperimentalTrace]:
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unsupported ablation mode: {mode}")

        diseases = parse_labels(disease_items, kind="disease")
        symptoms = parse_labels(symptom_items, kind="symptom")
        query_text = self._build_query_text(symptom_text, diseases, symptoms)

        stage_scores = self._collect_candidates(
            query_text=query_text,
            diseases=diseases,
            symptoms=symptoms,
            pool_size=pool_size,
            mode=mode,
        )
        selected_rows = self._rows_for_mode(stage_scores, mode=mode, pool_size=pool_size)
        features = self._build_feature_frame(
            row_ids=selected_rows,
            stage_scores=stage_scores,
            diseases=diseases,
            symptoms=symptoms,
            mode=mode,
        )

        if len(features) == 0:
            result = pd.DataFrame()
        else:
            result = self._score_features(features, mode=mode)
            result = result.sort_values("final_score", ascending=False).head(top_k).copy()

        trace = self._build_trace(
            diseases=diseases,
            symptoms=symptoms,
            stage_scores=stage_scores,
            selected_rows=selected_rows,
            mode=mode,
        )
        learned_weight_modes = {"local_ranker", "verified_learned_rerank", "verified_xgb_ranker"}
        trace.ranker_used = bool(
            mode in learned_weight_modes
            and self.ranker
            and self.ranker.is_ready
            and self._ranker_matches_mode(mode)
        )
        trace.fallback_mode = bool(mode in learned_weight_modes and not trace.ranker_used)
        if self.ranker:
            trace.ranker_model_type = self.ranker.model_type
            trace.ranker_train_source = str(self.ranker.metadata.get("train_source", ""))

        if return_trace:
            return result, trace
        return result

    def _ranker_matches_mode(self, mode: str) -> bool:
        if not self.ranker or not self.ranker.is_ready:
            return False
        if mode == "verified_learned_rerank":
            return self.ranker.model_type == "logreg"
        if mode == "verified_xgb_ranker":
            return self.ranker.model_type == "xgb_ranker"
        return True

    def _collect_candidates(
        self,
        *,
        query_text: str,
        diseases: list[NormalizedLabel],
        symptoms: list[NormalizedLabel],
        pool_size: int,
        mode: str,
    ) -> dict[str, dict[int, float]]:
        disease_conf = confidence_map(diseases)
        symptom_conf = confidence_map(symptoms)
        stage_scores: dict[str, dict[int, float]] = {
            "disease": {},
            "strict": {},
            "symptom": {},
            "bm25": {},
            "dense": {},
            "prior": {},
        }

        disease_rows_by_label = {
            label.name: self.index.disease_to_rows.get(label.name, set())
            for label in diseases
        }
        symptom_rows_by_label = {
            label.name: self.index.symptom_to_rows.get(label.name, set())
            for label in symptoms
        }

        for disease, rows in disease_rows_by_label.items():
            idf = self.index.disease_idf.get(disease, 1.0)
            confidence = disease_conf.get(disease, 0.0)
            for row_id in rows:
                stage_scores["disease"][row_id] = max(
                    stage_scores["disease"].get(row_id, 0.0),
                    confidence * idf,
                )

        for symptom, rows in symptom_rows_by_label.items():
            idf = self.index.symptom_idf.get(symptom, 1.0)
            confidence = symptom_conf.get(symptom, 0.0)
            for row_id in rows:
                stage_scores["symptom"][row_id] = stage_scores["symptom"].get(row_id, 0.0) + confidence * idf

        disease_union = set().union(*disease_rows_by_label.values()) if disease_rows_by_label else set()
        symptom_union = set().union(*symptom_rows_by_label.values()) if symptom_rows_by_label else set()
        for row_id in disease_union & symptom_union:
            stage_scores["strict"][row_id] = (
                stage_scores["disease"].get(row_id, 0.0)
                + stage_scores["symptom"].get(row_id, 0.0)
            )

        active_stages = MODE_STAGE_MAP.get(mode, ())

        if "bm25" in active_stages:
            bm25_scores = self.index.bm25.score(query_text, top_k=min(pool_size, self.index.row_count))
            stage_scores["bm25"] = {int(idx): float(score) for idx, score in bm25_scores.items()}

        if "dense" in active_stages:
            dense_scores = self._dense_scores(query_text, top_k=min(pool_size, self.index.row_count))
            stage_scores["dense"] = dense_scores

        if "prior" in active_stages:
            seed_rows = set()
            for stage in ("disease", "strict", "symptom", "bm25", "dense"):
                seed_rows.update(stage_scores[stage].keys())
            stage_scores["prior"] = self._prior_expansion(seed_rows, diseases, limit_per_group=30)

        for stage, scores in list(stage_scores.items()):
            if len(scores) > pool_size:
                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:pool_size]
                stage_scores[stage] = dict(ranked)
        return stage_scores

    def _dense_scores(self, query_text: str, *, top_k: int) -> dict[int, float]:
        if self.index.embeddings is None or self.query_encoder is None:
            return {}
        query_vec = self.query_encoder.encode_query(query_text).astype(np.float32)
        raw_scores = safe_cosine_scores(self.index.embeddings, query_vec)
        if len(raw_scores) == 0:
            return {}
        top_indices = np.argsort(raw_scores)[::-1][:top_k]
        return {int(idx): float(raw_scores[idx]) for idx in top_indices}

    def _prior_expansion(
        self,
        seed_rows: set[int],
        diseases: list[NormalizedLabel],
        *,
        limit_per_group: int,
    ) -> dict[int, float]:
        prior_rows: dict[int, float] = {}
        for row_id in seed_rows:
            for generic in self.index.row_generics[row_id]:
                for other_id in self.index.generic_to_rows.get(generic, set()):
                    prior_rows[other_id] = max(prior_rows.get(other_id, 0.0), 0.85)
            for drug_class in self.index.row_classes[row_id]:
                for other_id in self.index.class_to_rows.get(drug_class, set()):
                    prior_rows[other_id] = max(prior_rows.get(other_id, 0.0), 0.55)
            drug_name = str(self.index.df.at[row_id, "drug_name"]).strip().lower()
            for other_id in self.index.related_to_rows.get(drug_name, set()):
                prior_rows[other_id] = max(prior_rows.get(other_id, 0.0), 0.75)

        for disease in diseases:
            for row_id in self.index.top_quality_rows_for_disease(disease.name, limit=limit_per_group):
                prior_rows[row_id] = max(prior_rows.get(row_id, 0.0), float(self.index.quality_prior.loc[row_id]))
        return prior_rows

    def _rows_for_mode(
        self,
        stage_scores: dict[str, dict[int, float]],
        *,
        mode: str,
        pool_size: int,
    ) -> list[int]:
        rows: set[int] = set()
        for stage in MODE_STAGE_MAP[mode]:
            rows.update(stage_scores[stage].keys())
        if not rows:
            return []

        seed_score = defaultdict(float)
        for row_id in rows:
            for stage in MODE_STAGE_MAP[mode]:
                seed_score[row_id] += stage_scores[stage].get(row_id, 0.0)
        ranked = sorted(rows, key=lambda row_id: seed_score[row_id], reverse=True)
        return ranked[:pool_size]

    def _build_feature_frame(
        self,
        *,
        row_ids: list[int],
        stage_scores: dict[str, dict[int, float]],
        diseases: list[NormalizedLabel],
        symptoms: list[NormalizedLabel],
        mode: str,
    ) -> pd.DataFrame:
        if not row_ids:
            return pd.DataFrame()

        disease_conf = confidence_map(diseases)
        symptom_conf = confidence_map(symptoms)
        total_symptom_conf = sum(symptom_conf.values()) or 1.0
        max_disease_idf = max(self.index.disease_idf.values(), default=1.0)

        rows = []
        for row_id in row_ids:
            row = self.index.df.loc[row_id]
            row_diseases = set(row.get("disease_key_list", []))
            row_symptoms = set(row.get("symptom_list_normalized", []))
            matched_diseases = sorted(row_diseases & set(disease_conf))
            matched_symptoms = sorted(row_symptoms & set(symptom_conf))

            disease_idf_score = sum(
                disease_conf[name] * self.index.disease_idf.get(name, 1.0)
                for name in matched_diseases
            )
            symptom_idf_score = sum(
                symptom_conf[name] * self.index.symptom_idf.get(name, 1.0)
                for name in matched_symptoms
            )
            disease_conf_overlap = sum(disease_conf[name] for name in matched_diseases)
            symptom_conf_overlap = sum(symptom_conf[name] for name in matched_symptoms)
            has_specific_input_disease = any(label.name != "others" for label in diseases)
            row_is_others_only = row_diseases == {"others"} or ("others" in row_diseases and not matched_diseases)
            others_penalty = 0.18 if has_specific_input_disease and row_is_others_only else 0.0

            stage_hits = {stage: int(row_id in scores) for stage, scores in stage_scores.items()}
            rows.append(
                {
                    "row_id": row_id,
                    "drug_name": row.get("drug_name", ""),
                    "final_score": 0.0,
                    "deterministic_score": 0.0,
                    "ranker_score": 0.0,
                    "label_idf_score": disease_idf_score + symptom_idf_score,
                    "disease_idf_score": disease_idf_score,
                    "symptom_idf_score": symptom_idf_score,
                    "symptom_coverage": symptom_conf_overlap / total_symptom_conf,
                    "disease_specificity": (disease_idf_score / max_disease_idf) if disease_idf_score else 0.0,
                    "bm25_score": stage_scores["bm25"].get(row_id, 0.0),
                    "dense_score": stage_scores["dense"].get(row_id, 0.0),
                    "quality_prior": float(self.index.quality_prior.loc[row_id]),
                    "disease_conf_overlap": disease_conf_overlap,
                    "symptom_conf_overlap": symptom_conf_overlap,
                    "others_penalty": others_penalty,
                    "stage_hits": ",".join(stage for stage, hit in stage_hits.items() if hit),
                    "matched_diseases": ",".join(matched_diseases),
                    "matched_symptoms": ",".join(matched_symptoms),
                    "avg_rating": row.get("avg_rating", np.nan),
                    "total_reviews": row.get("total_reviews", np.nan),
                    "evidence": self._evidence(matched_diseases, matched_symptoms, stage_hits),
                    "mode": mode,
                    **{f"stage_{stage}": hit for stage, hit in stage_hits.items()},
                }
            )

        features = pd.DataFrame(rows).set_index("row_id", drop=False)
        for column in ("label_idf_score", "bm25_score", "dense_score"):
            features[column] = self._normalize_column(features[column])
        features["disease_idf_score"] = self._normalize_column(features["disease_idf_score"])
        features["symptom_idf_score"] = self._normalize_column(features["symptom_idf_score"])
        return features

    def _score_features(self, features: pd.DataFrame, *, mode: str) -> pd.DataFrame:
        scored = features.copy()

        # Force features to 0 when their stage is excluded from the candidate pool
        no_bm25_modes = {
            "candidate_union_no_bm25",
            "candidate_union_no_prior_no_bm25",
            "label_core_rerank",
            "verified_learned_rerank",
            "verified_xgb_ranker",
        }
        no_prior_modes = {
            "candidate_union_no_prior",
            "candidate_union_no_prior_no_bm25",
            "label_core_rerank",
            "verified_learned_rerank",
            "verified_xgb_ranker",
        }
        no_dense_modes = {
            "label_core_rerank",
            "verified_learned_rerank",
            "verified_xgb_ranker",
        }
        if mode in no_bm25_modes:
            scored["bm25_score"] = 0.0
        if mode in no_prior_modes:
            scored["stage_prior"] = 0.0
        if mode in no_dense_modes:
            scored["dense_score"] = 0.0

        dense_weight = 0.15 if scored["dense_score"].max() > 0 else 0.05
        label_weight = 0.40 + (0.15 - dense_weight)
        scored["deterministic_score"] = (
            label_weight * scored["label_idf_score"]
            + 0.20 * scored["symptom_coverage"]
            + 0.15 * scored["bm25_score"]
            + dense_weight * scored["dense_score"]
            + 0.10 * scored["quality_prior"]
            + 0.06 * scored["stage_strict"]
            + 0.03 * scored["stage_disease"]
            + 0.02 * scored["stage_prior"]
            - scored["others_penalty"]
        )

        if mode == "bm25_only":
            scored["final_score"] = scored["bm25_score"]
        elif mode == "dense_only":
            scored["final_score"] = scored["dense_score"]
        elif mode == "label_idf_only":
            scored["final_score"] = scored["label_idf_score"] + 0.15 * scored["symptom_coverage"]
        elif mode == "label_bm25":
            scored["final_score"] = 0.70 * scored["label_idf_score"] + 0.30 * scored["bm25_score"]
        elif mode == "label_bm25_dense":
            scored["final_score"] = (
                0.60 * scored["label_idf_score"]
                + 0.25 * scored["bm25_score"]
                + 0.15 * scored["dense_score"]
            )
        elif mode == "local_ranker" and self.ranker and self.ranker.is_ready:
            scored["ranker_score"] = self.ranker.predict(scored)
            scored["final_score"] = 0.80 * scored["ranker_score"] + 0.20 * scored["deterministic_score"]
        elif mode == "verified_learned_rerank":
            if self._ranker_matches_mode(mode):
                scored["ranker_score"] = self.ranker.predict(scored)
                scored["final_score"] = scored["ranker_score"]
            else:
                scored["final_score"] = (
                    0.50 * scored["label_idf_score"]
                    + 0.20 * scored["symptom_coverage"]
                    + 0.10 * scored["quality_prior"]
                    + 0.06 * scored["stage_strict"]
                    + 0.03 * scored["stage_disease"]
                    - scored["others_penalty"]
                )
        elif mode == "verified_xgb_ranker":
            if self._ranker_matches_mode(mode):
                scored["ranker_score"] = self.ranker.predict(scored)
                scored["final_score"] = scored["ranker_score"]
            else:
                scored["final_score"] = (
                    0.50 * scored["label_idf_score"]
                    + 0.20 * scored["symptom_coverage"]
                    + 0.10 * scored["quality_prior"]
                    + 0.06 * scored["stage_strict"]
                    + 0.03 * scored["stage_disease"]
                    - scored["others_penalty"]
                )
        elif mode == "label_core_rerank":
            scored["final_score"] = (
                0.50 * scored["label_idf_score"]
                + 0.20 * scored["symptom_coverage"]
                + 0.10 * scored["quality_prior"]
                + 0.06 * scored["stage_strict"]
                + 0.03 * scored["stage_disease"]
                - scored["others_penalty"]
            )
        else:
            scored["final_score"] = scored["deterministic_score"]
        return scored

    def _build_trace(
        self,
        *,
        diseases: list[NormalizedLabel],
        symptoms: list[NormalizedLabel],
        stage_scores: dict[str, dict[int, float]],
        selected_rows: list[int],
        mode: str,
    ) -> ExperimentalTrace:
        selected_set = set(selected_rows)

        # Determine which stages are actually part of this mode's candidate pool.
        active_stages = set(MODE_STAGE_MAP.get(mode, ()))

        # filtered_scores: only include rows that are both in selected_rows AND
        # belong to a stage that is part of this mode's candidate pool.
        # This ensures stage_hit=0 for stages excluded by the ablation.
        filtered_scores: dict[str, dict[int, float]] = {}
        for stage, scores in stage_scores.items():
            if stage in active_stages:
                filtered_scores[stage] = {rid: s for rid, s in scores.items() if rid in selected_set}
            else:
                filtered_scores[stage] = {}  # excluded by ablation — empty

        stage_candidate_names = {}
        for stage, scores in filtered_scores.items():
            if scores:
                names = self.index.df.loc[list(scores.keys()), "drug_name"].astype(str).tolist()
            else:
                names = []
            stage_candidate_names[stage] = names

        return ExperimentalTrace(
            normalized_diseases=[asdict(label) for label in diseases],
            normalized_symptoms=[asdict(label) for label in symptoms],
            candidate_counts={stage: len(scores) for stage, scores in filtered_scores.items()},
            final_union_size=len(selected_rows),
            embedding_manifest=self.index.embedding_manifest.to_dict(),
            mode=mode,
            stage_candidate_names=stage_candidate_names,
        )

    def _build_query_text(
        self,
        symptom_text: str,
        diseases: list[NormalizedLabel],
        symptoms: list[NormalizedLabel],
    ) -> str:
        disease_text = " ".join(label.name.replace("_", " ") for label in diseases)
        symptom_text_labels = " ".join(label.name for label in symptoms)
        return " ".join(part for part in [symptom_text or "", disease_text, symptom_text_labels] if part).strip()

    def _normalize_column(self, values: pd.Series) -> pd.Series:
        if len(values) == 0:
            return values
        values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo < 1e-12:
            return pd.Series(np.where(values > 0, 1.0, 0.0), index=values.index)
        return (values - lo) / (hi - lo)

    def _evidence(
        self,
        matched_diseases: list[str],
        matched_symptoms: list[str],
        stage_hits: dict[str, int],
    ) -> str:
        parts = []
        if matched_diseases:
            parts.append("disease=" + "|".join(matched_diseases[:3]))
        if matched_symptoms:
            parts.append("symptom=" + "|".join(matched_symptoms[:5]))
        hit_names = [name for name, hit in stage_hits.items() if hit]
        if hit_names:
            parts.append("stage=" + "|".join(hit_names))
        return "; ".join(parts)
