from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .schemas import CandidateScore, RecommendRequest, RecommendResponse
from .service import get_recommendation_service


router = APIRouter(prefix="/v1/recommendation", tags=["recommendation"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/drugs", response_model=RecommendResponse)
def recommend_drugs(payload: RecommendRequest) -> RecommendResponse:
    service = get_recommendation_service()
    trace_data = None
    try:
        result = service.recommend(
            symptom_text=payload.symptom_text,
            diseases=[item.model_dump() if hasattr(item, "model_dump") else item for item in payload.diseases],
            symptoms=[item.model_dump() if hasattr(item, "model_dump") else item for item in payload.symptoms],
            top_k=payload.top_k,
            recall_top_k_each=payload.recall_top_k_each,
            fused_top_k=payload.fused_top_k,
            recall_weight_semantic=payload.recall_weight_semantic,
            recall_weight_label=payload.recall_weight_label,
            use_bert_prediction=payload.use_bert_prediction,
            enable_trace=payload.enable_trace,
        )
        if payload.enable_trace:
            result_df, trace_obj = result
            trace_data = trace_obj.to_dict()
        else:
            result_df = result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"recommendation failed: {exc}") from exc

    results: list[CandidateScore] = []
    for _, row in result_df.iterrows():
        base_fields = {
            "drug_name",
            "final_score",
            "recall_fused_score",
            "cross_encoder_score",
            "business_score",
            "semantic_score",
            "label_score",
            "disease_conf_overlap",
            "symptom_conf_overlap",
            "avg_rating",
            "total_reviews",
        }
        extra = {k: v for k, v in row.to_dict().items() if k not in base_fields}
        results.append(
            CandidateScore(
                drug_name=str(row.get("drug_name", "")),
                final_score=float(row.get("final_score", 0.0)),
                recall_fused_score=float(row.get("recall_fused_score", 0.0)),
                cross_encoder_score=float(row.get("cross_encoder_score", 0.0)),
                business_score=float(row.get("business_score", 0.0)),
                semantic_score=float(row.get("semantic_score", 0.0)),
                label_score=float(row.get("label_score", 0.0)),
                disease_conf_overlap=float(row.get("disease_conf_overlap", 0.0)),
                symptom_conf_overlap=float(row.get("symptom_conf_overlap", 0.0)),
                avg_rating=(
                    None
                    if row.get("avg_rating") is None
                    else float(row.get("avg_rating"))
                    if str(row.get("avg_rating")) != "nan"
                    else None
                ),
                total_reviews=(
                    None
                    if row.get("total_reviews") is None
                    else float(row.get("total_reviews"))
                    if str(row.get("total_reviews")) != "nan"
                    else None
                ),
                extra=extra,
            )
        )

    return RecommendResponse(count=len(results), results=results, trace=trace_data)
