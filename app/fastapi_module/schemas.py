from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class LabelWithConfidence(BaseModel):
    name: str = Field(..., description="Label name, e.g. common_cold / fever")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score in [0, 1]")


class RecommendRequest(BaseModel):
    symptom_text: str = Field(..., description="User free-text symptom description")
    diseases: list[dict[str, float] | LabelWithConfidence] = Field(
        default_factory=list,
        description="Disease labels with confidence. Supports {'name':..., 'confidence':...} "
        "or {'disease_name': 0.8} format.",
    )
    symptoms: list[dict[str, float] | LabelWithConfidence] = Field(
        default_factory=list,
        description="Symptom labels with confidence. Supports {'name':..., 'confidence':...} "
        "or {'symptom_name': 0.8} format.",
    )
    top_k: int = Field(10, ge=1, le=100)
    recall_top_k_each: int = Field(300, ge=20, le=2000)
    fused_top_k: int = Field(300, ge=20, le=3000)
    recall_weight_semantic: float = Field(0.5, ge=0.0, le=1.0)
    recall_weight_label: float = Field(0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> "RecommendRequest":
        total = self.recall_weight_semantic + self.recall_weight_label
        if total <= 0:
            raise ValueError("recall weights sum must be > 0")
        return self


class CandidateScore(BaseModel):
    drug_name: str
    final_score: float
    recall_fused_score: float
    cross_encoder_score: float
    business_score: float
    semantic_score: float
    label_score: float
    disease_conf_overlap: float
    symptom_conf_overlap: float
    avg_rating: float | None = None
    total_reviews: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    count: int
    results: list[CandidateScore]
