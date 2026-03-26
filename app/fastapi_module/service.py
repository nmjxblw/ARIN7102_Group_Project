from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from embedded_module import (
    CrossEncoderReranker,
    DualRecallDrugRecommender,
    prepare_drug_dataframe,
)
from embedded_module.drug_embedding_engine import DrugEmbeddingEngine


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_drug_table_path() -> Path:
    root = _repo_root()
    candidate_paths = [
        root / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1_structured.csv",
        root / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1.csv",
        root / "data" / "enhanced_drug_table_v1.csv",
        root.parent / "data" / "enhanced_drug_table_v1.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return candidate_paths[0]


def _default_embedding_path() -> Path:
    root = _repo_root()
    return root / "drug_comprehensive_embeddings.npy"


class DrugRecommendationService:
    """
    Lazily initialized business service for dual-recall + cross-encoder ranking.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._initialized = False

        self._engine: DrugEmbeddingEngine | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._pipeline: DualRecallDrugRecommender | None = None

    def _build_components(self):
        table_path = Path(os.getenv("DRUG_TABLE_PATH", str(_default_drug_table_path())))
        embedding_path = Path(os.getenv("DRUG_EMBEDDING_PATH", str(_default_embedding_path())))
        cross_model = os.getenv(
            "CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        medbert_name = os.getenv(
            "MEDBERT_MODEL_NAME", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        )
        auto_build = os.getenv("AUTO_BUILD_EMBEDDINGS", "false").lower() == "true"

        if not table_path.exists():
            raise FileNotFoundError(f"Drug table file not found: {table_path}")

        df = pd.read_csv(table_path)
        # Always rebuild structured columns to ensure list-like fields are parsed
        # correctly even when loaded from a CSV artifact.
        df = prepare_drug_dataframe(df)

        engine = DrugEmbeddingEngine(model_name=medbert_name)
        if embedding_path.exists():
            embeddings = engine.load(str(embedding_path))
        else:
            if not auto_build:
                raise FileNotFoundError(
                    f"Embedding file not found: {embedding_path}. "
                    "Set AUTO_BUILD_EMBEDDINGS=true to auto-generate."
                )
            embeddings = engine.encode(df["semantic_text"].tolist())
            engine.save(embeddings, str(embedding_path))

        reranker = CrossEncoderReranker(model_name=cross_model)
        pipeline = DualRecallDrugRecommender(
            df=df,
            drug_embeddings=np.asarray(embeddings, dtype=np.float32),
            embedding_engine=engine,
            cross_encoder_reranker=reranker,
        )

        self._engine = engine
        self._reranker = reranker
        self._pipeline = pipeline
        self._initialized = True

    def ensure_ready(self):
        if self._initialized:
            return
        with self._lock:
            if not self._initialized:
                self._build_components()

    @property
    def pipeline(self) -> DualRecallDrugRecommender:
        self.ensure_ready()
        assert self._pipeline is not None
        return self._pipeline

    def recommend(
        self,
        symptom_text: str,
        diseases: list[dict[str, float] | dict[str, Any]],
        symptoms: list[dict[str, float] | dict[str, Any]],
        top_k: int,
        recall_top_k_each: int,
        fused_top_k: int,
        recall_weight_semantic: float,
        recall_weight_label: float,
    ) -> pd.DataFrame:
        total_weight = recall_weight_semantic + recall_weight_label
        if total_weight <= 0:
            recall_weight_semantic = 0.5
            recall_weight_label = 0.5
            total_weight = 1.0

        return self.pipeline.recommend(
            symptom_text=symptom_text,
            disease_items=diseases,
            symptom_items=symptoms,
            recall_top_k_each=recall_top_k_each,
            fused_top_k=fused_top_k,
            top_k=top_k,
            recall_weight_semantic=recall_weight_semantic / total_weight,
            recall_weight_label=recall_weight_label / total_weight,
        )


_service_singleton: DrugRecommendationService | None = None
_service_lock = threading.RLock()


def get_recommendation_service() -> DrugRecommendationService:
    global _service_singleton
    if _service_singleton is not None:
        return _service_singleton
    with _service_lock:
        if _service_singleton is None:
            _service_singleton = DrugRecommendationService()
        return _service_singleton
