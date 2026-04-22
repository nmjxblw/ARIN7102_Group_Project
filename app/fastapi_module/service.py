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
    PipelineTrace,
    prepare_drug_dataframe,
)
from embedded_module.drug_embedding_engine import DrugEmbeddingEngine
from deployment_module.bert_main import preload as bert_preload, predict_with_preload as bert_predict_with_preload


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
    Integrates BERT multi-task classifier for automatic disease/symptom prediction.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._initialized = False

        self._engine: DrugEmbeddingEngine | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._pipeline: DualRecallDrugRecommender | None = None

        # BERT classifier components (lazy-loaded)
        self._bert_lock = threading.RLock()
        self._bert_loaded = False
        self._bert_device: Any = None
        self._bert_tokenizer: Any = None
        self._bert_model: Any = None
        self._bert_mlb_d: Any = None
        self._bert_mlb_s: Any = None
        self._bert_medians: dict = {}

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

    def _load_bert(self):
        """Lazily load BERT multi-task classifier for disease/symptom prediction."""
        if self._bert_loaded:
            return
        with self._bert_lock:
            if self._bert_loaded:
                return
            bert_model_path = os.getenv(
                "TRAINED_BERT_SAVE_PATH",
                str(Path(__file__).resolve().parents[1] / "deployment_module" / "trained_bert"),
            )
            self._bert_device, self._bert_tokenizer, self._bert_model, \
                self._bert_mlb_d, self._bert_mlb_s, self._bert_medians = bert_preload(bert_model_path)
            self._bert_loaded = True

    def predict_labels(self, symptom_text: str) -> dict[str, Any]:
        """Use BERT to predict disease and symptom labels from free text.

        Returns dict with keys: diseases, symptoms, need_first_aid
        Each disease/symptom entry has format {"name": ..., "confidence": ...}
        """
        self._load_bert()
        return bert_predict_with_preload(
            text=symptom_text,
            tokenizer=self._bert_tokenizer,
            inference_device=self._bert_device,
            inference_model=self._bert_model,
            mlb_d=self._bert_mlb_d,
            mlb_s=self._bert_mlb_s,
            medians=self._bert_medians,
        )

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
        diseases: list[dict[str, float] | dict[str, Any]] | None = None,
        symptoms: list[dict[str, float] | dict[str, Any]] | None = None,
        top_k: int | None = None,
        recall_top_k_each: int | None = None,
        fused_top_k: int | None = None,
        recall_weight_semantic: float | None = None,
        recall_weight_label: float | None = None,
        use_bert_prediction: bool = False,
        enable_trace: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, PipelineTrace]:
        if diseases is None:
            diseases = []
        if symptoms is None:
            symptoms = []

        # If BERT prediction is requested, auto-predict labels from symptom_text
        if use_bert_prediction and symptom_text.strip():
            bert_result = self.predict_labels(symptom_text)
            diseases = bert_result.get("diseases", [])
            symptoms = bert_result.get("symptoms", [])

        if recall_weight_semantic is not None and recall_weight_label is not None:
            total_weight = recall_weight_semantic + recall_weight_label
            if total_weight <= 0:
                recall_weight_semantic = None
                recall_weight_label = None
            else:
                recall_weight_semantic = recall_weight_semantic / total_weight
                recall_weight_label = recall_weight_label / total_weight

        return self.pipeline.recommend(
            symptom_text=symptom_text,
            disease_items=diseases,
            symptom_items=symptoms,
            recall_top_k_each=recall_top_k_each,
            fused_top_k=fused_top_k,
            top_k=top_k,
            recall_weight_semantic=recall_weight_semantic,
            recall_weight_label=recall_weight_label,
            enable_trace=enable_trace,
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
