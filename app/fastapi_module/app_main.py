from __future__ import annotations

from fastapi import FastAPI

from .router import router as recommendation_router


app = FastAPI(
    title="Drug Recommendation API",
    version="1.0.0",
    description="Dual-recall drug recommendation service with MedBERT recall and cross-encoder reranking.",
)
app.include_router(recommendation_router)

