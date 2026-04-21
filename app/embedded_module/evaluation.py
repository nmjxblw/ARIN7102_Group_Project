"""
Batch evaluation helpers for the drug retrieval pipeline.

Important evaluation boundary:
This is label-based retrieval accuracy, not real clinical prescription
accuracy. The current benchmark data provides question text plus disease and
symptom labels, but it does not provide a gold-standard drug answer for each
case. As a result, evaluation should be interpreted as "does the retriever rank
drugs whose stored disease/symptom labels match the query labels?" rather than
"did the system prescribe the clinically correct medication?"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .recommendation_pipeline import (
        baseline_filter,
        build_query_text,
        prepare_drug_dataframe,
        query_drugs_hybrid,
        safe_cosine_scores,
    )
except ImportError:  # pragma: no cover
    from recommendation_pipeline import (  # type: ignore
        baseline_filter,
        build_query_text,
        prepare_drug_dataframe,
        query_drugs_hybrid,
        safe_cosine_scores,
    )


EVALUATION_SCOPE_NOTE = (
    "This evaluation reports label-based retrieval accuracy, not real clinical "
    "prescription accuracy."
)

DEFAULT_TOP_KS = (1, 3, 5, 10)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_questions_json() -> Path:
    return _repo_root() / "app/dataset_module/bert_training_dataset/medical_questions_dataset.json"


def default_drug_csv() -> Path:
    repo_root = _repo_root()
    external = repo_root.parent / "data/enhanced_drug_table_v1.csv"
    internal = repo_root / "data/enhanced_drug_table_v1.csv"
    return external if external.exists() else internal


def default_embedding_npy() -> Path:
    repo_root = _repo_root()
    candidates = [
        repo_root / "interaction/drug_comprehensive_embeddings.npy",
        repo_root / "drug_comprehensive_embeddings.npy",
        repo_root.parent / "data/drug_semantic_embeddings.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_questions(questions_json: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(questions_json).read_text())
    if not isinstance(data, list):
        raise ValueError("questions_json should contain a JSON list")
    if limit is not None:
        return data[:limit]
    return data


def load_drug_table(drug_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(drug_csv)
    return prepare_drug_dataframe(df)


def load_embeddings(embedding_npy: str | Path) -> np.ndarray:
    embeddings = np.load(embedding_npy)
    if embeddings.ndim != 2:
        raise ValueError(f"embedding matrix should be 2D, got shape={embeddings.shape}")
    return embeddings


def _normalize_query_labels(question: dict[str, Any]) -> tuple[list[str], list[str]]:
    diseases = [str(x).strip() for x in question.get("disease", []) if str(x).strip()]
    symptoms = [str(x).strip() for x in question.get("symptoms", []) if str(x).strip()]
    return diseases, symptoms


def _annotate_relevance(rows: pd.DataFrame, diseases: list[str], symptoms: list[str]) -> pd.DataFrame:
    diseases_set = {str(x).strip() for x in diseases}
    symptoms_set = {str(x).strip().lower().replace("_", " ") for x in symptoms}
    result = rows.copy()

    result["eval_disease_hit"] = result["disease_key_list"].apply(
        lambda xs: int(bool(set(xs) & diseases_set))
    )
    result["eval_symptom_hit"] = result["symptom_list_normalized"].apply(
        lambda xs: int(bool(set(xs) & symptoms_set))
    )
    result["eval_loose_hit"] = (
        (result["eval_disease_hit"] > 0) | (result["eval_symptom_hit"] > 0)
    ).astype(int)
    result["eval_strict_hit"] = (
        (result["eval_disease_hit"] > 0) & (result["eval_symptom_hit"] > 0)
    ).astype(int)
    result["eval_relevance"] = (
        2 * result["eval_strict_hit"] + 1 * (result["eval_loose_hit"] - result["eval_strict_hit"])
    )
    return result


def retrieve_baseline(
    df: pd.DataFrame,
    diseases: list[str],
    symptoms: list[str],
    top_k: int,
) -> pd.DataFrame:
    result = baseline_filter(df, diseases, symptoms).head(top_k).copy()
    result["retrieval_method"] = "baseline"
    return result


def retrieve_pure_knn(
    df: pd.DataFrame,
    drug_embeddings: np.ndarray,
    engine: Any,
    diseases: list[str],
    symptoms: list[str],
    top_k: int,
) -> pd.DataFrame:
    query_text = build_query_text(diseases, symptoms)
    query_vec = engine.encode_query(query_text).astype(np.float32)
    scores = safe_cosine_scores(drug_embeddings, query_vec)
    result = df.copy()
    result["knn_score"] = scores
    result = result.sort_values(by=["knn_score"], ascending=[False]).head(top_k).copy()
    result["retrieval_method"] = "pure_knn"
    return result


def retrieve_hybrid(
    df: pd.DataFrame,
    drug_embeddings: np.ndarray,
    engine: Any,
    diseases: list[str],
    symptoms: list[str],
    top_k: int,
    require_disease_match: bool = True,
) -> pd.DataFrame:
    result = query_drugs_hybrid(
        df=df,
        drug_embeddings=drug_embeddings,
        engine=engine,
        disease_labels=diseases,
        symptom_terms=symptoms,
        top_k=top_k,
        require_disease_match=require_disease_match,
        debug=False,
    ).head(top_k).copy()
    result["retrieval_method"] = "hybrid"
    return result


def _dcg(scores: list[int], top_k: int) -> float:
    total = 0.0
    for rank, score in enumerate(scores[:top_k], start=1):
        total += float(score) / math.log2(rank + 1)
    return total


def _ideal_relevance_count(row: pd.Series) -> int:
    return int(row["eval_relevance"])


def _per_query_metrics(
    retrieved: pd.DataFrame,
    full_df: pd.DataFrame,
    top_ks: tuple[int, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    retrieved = retrieved.reset_index(drop=True)

    full_relevance = full_df["eval_relevance"].sort_values(ascending=False).tolist()
    loose_total = int(full_df["eval_loose_hit"].sum())
    strict_total = int(full_df["eval_strict_hit"].sum())

    for k in top_ks:
        top = retrieved.head(k)
        loose_hits = top["eval_loose_hit"].tolist()
        strict_hits = top["eval_strict_hit"].tolist()
        relevance_scores = top["eval_relevance"].tolist()

        metrics[f"hit@{k}_loose"] = float(any(loose_hits))
        metrics[f"hit@{k}_strict"] = float(any(strict_hits))
        metrics[f"precision@{k}_loose"] = float(sum(loose_hits) / k)
        metrics[f"precision@{k}_strict"] = float(sum(strict_hits) / k)
        metrics[f"recall@{k}_loose"] = float(sum(loose_hits) / loose_total) if loose_total else 0.0
        metrics[f"recall@{k}_strict"] = (
            float(sum(strict_hits) / strict_total) if strict_total else 0.0
        )

        dcg = _dcg(relevance_scores, k)
        idcg = _dcg(full_relevance, k)
        metrics[f"ndcg@{k}"] = float(dcg / idcg) if idcg > 0 else 0.0

    loose_rr = 0.0
    strict_rr = 0.0
    for rank, row in enumerate(retrieved.itertuples(index=False), start=1):
        if not loose_rr and row.eval_loose_hit:
            loose_rr = 1.0 / rank
        if not strict_rr and row.eval_strict_hit:
            strict_rr = 1.0 / rank
        if loose_rr and strict_rr:
            break

    metrics["mrr_loose"] = loose_rr
    metrics["mrr_strict"] = strict_rr
    return metrics


def evaluate_questions(
    questions: list[dict[str, Any]],
    df: pd.DataFrame,
    method: str = "baseline",
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS,
    top_k_max: int | None = None,
    embeddings: np.ndarray | None = None,
    engine: Any | None = None,
    require_disease_match: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    supported = {"baseline", "pure_knn", "hybrid"}
    if method not in supported:
        raise ValueError(f"method should be one of {sorted(supported)}, got {method}")

    if top_k_max is None:
        top_k_max = max(top_ks)

    if method in {"pure_knn", "hybrid"} and embeddings is None:
        raise ValueError(f"{method} evaluation requires embeddings")
    if method in {"pure_knn", "hybrid"} and engine is None:
        raise ValueError(f"{method} evaluation requires an embedding engine")
    if embeddings is not None and len(df) != len(embeddings):
        raise ValueError(
            f"drug table rows ({len(df)}) and embedding rows ({len(embeddings)}) do not match"
        )

    query_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, float | int | str]] = []

    for question_id, question in enumerate(questions):
        diseases, symptoms = _normalize_query_labels(question)

        if method == "baseline":
            retrieved = retrieve_baseline(df, diseases, symptoms, top_k=top_k_max)
        elif method == "pure_knn":
            retrieved = retrieve_pure_knn(
                df=df,
                drug_embeddings=embeddings,
                engine=engine,
                diseases=diseases,
                symptoms=symptoms,
                top_k=top_k_max,
            )
        else:
            retrieved = retrieve_hybrid(
                df=df,
                drug_embeddings=embeddings,
                engine=engine,
                diseases=diseases,
                symptoms=symptoms,
                top_k=top_k_max,
                require_disease_match=require_disease_match,
            )

        annotated_full = _annotate_relevance(df, diseases, symptoms)
        annotated_retrieved = _annotate_relevance(retrieved, diseases, symptoms)

        query_record = {
            "question_id": question_id,
            "question": question.get("question", ""),
            "disease": diseases,
            "symptoms": symptoms,
            "need_first_aid": int(question.get("need_first_aid", 0) or 0),
            "method": method,
            "retrieved_count": int(len(annotated_retrieved)),
            "relevant_loose_total": int(annotated_full["eval_loose_hit"].sum()),
            "relevant_strict_total": int(annotated_full["eval_strict_hit"].sum()),
        }
        query_record.update(_per_query_metrics(annotated_retrieved, annotated_full, top_ks))
        query_rows.append(query_record)

        for rank, row in enumerate(annotated_retrieved.itertuples(index=False), start=1):
            metrics_rows.append(
                {
                    "question_id": question_id,
                    "method": method,
                    "rank": rank,
                    "drug_name": row.drug_name,
                    "eval_disease_hit": int(row.eval_disease_hit),
                    "eval_symptom_hit": int(row.eval_symptom_hit),
                    "eval_loose_hit": int(row.eval_loose_hit),
                    "eval_strict_hit": int(row.eval_strict_hit),
                    "eval_relevance": int(_ideal_relevance_count(pd.Series(row._asdict()))),
                    "knn_score": float(getattr(row, "knn_score", np.nan)),
                    "avg_rating": float(getattr(row, "avg_rating", np.nan)),
                }
            )

    return pd.DataFrame(query_rows), pd.DataFrame(metrics_rows)


def summarize_query_metrics(query_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in query_metrics.columns
        if col not in {"question_id", "question", "disease", "symptoms", "method"}
        and pd.api.types.is_numeric_dtype(query_metrics[col])
    ]
    summary = query_metrics[numeric_cols].mean().to_frame(name="value")
    summary.index.name = "metric"
    return summary.reset_index()


def _build_engine(model_name: str | None = None) -> Any:
    try:
        from .drug_embedding_engine import DrugEmbeddingEngine
    except ImportError:  # pragma: no cover
        from drug_embedding_engine import DrugEmbeddingEngine  # type: ignore

    if model_name:
        return DrugEmbeddingEngine(model_name=model_name)
    return DrugEmbeddingEngine()


def run_evaluation(
    method: str = "baseline",
    questions_json: str | Path | None = None,
    drug_csv: str | Path | None = None,
    embedding_npy: str | Path | None = None,
    model_name: str | None = None,
    limit: int | None = None,
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS,
    require_disease_match: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    questions_json = Path(questions_json or default_questions_json())
    drug_csv = Path(drug_csv or default_drug_csv())

    questions = load_questions(questions_json, limit=limit)
    df = load_drug_table(drug_csv)

    embeddings = None
    engine = None
    if method in {"pure_knn", "hybrid"}:
        embedding_path = Path(embedding_npy or default_embedding_npy())
        embeddings = load_embeddings(embedding_path)
        engine = _build_engine(model_name=model_name)

    query_metrics, ranked_results = evaluate_questions(
        questions=questions,
        df=df,
        method=method,
        top_ks=top_ks,
        embeddings=embeddings,
        engine=engine,
        require_disease_match=require_disease_match,
    )
    summary = summarize_query_metrics(query_metrics)
    return query_metrics, ranked_results, summary


def _parse_top_ks(value: str) -> tuple[int, ...]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    top_ks = tuple(sorted({int(x) for x in parts}))
    if not top_ks:
        raise ValueError("top_ks cannot be empty")
    return top_ks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate drug retrieval from JSON questions.")
    parser.add_argument(
        "--method",
        choices=["baseline", "pure_knn", "hybrid"],
        default="baseline",
        help="Retrieval method to evaluate.",
    )
    parser.add_argument(
        "--questions-json",
        default=str(default_questions_json()),
        help="Path to medical_questions_dataset.json.",
    )
    parser.add_argument(
        "--drug-csv",
        default=str(default_drug_csv()),
        help="Path to enhanced_drug_table_v1.csv.",
    )
    parser.add_argument(
        "--embedding-npy",
        default=str(default_embedding_npy()),
        help="Path to precomputed embedding matrix for pure_knn or hybrid.",
    )
    parser.add_argument("--model-name", default=None, help="Optional embedding model override.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument(
        "--top-ks",
        default="1,3,5,10",
        help="Comma-separated K values for metric reporting.",
    )
    parser.add_argument(
        "--allow-symptom-fallback",
        action="store_true",
        help="For hybrid, allow fallback to non-disease matches when strict disease match is empty.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    top_ks = _parse_top_ks(args.top_ks)
    query_metrics, ranked_results, summary = run_evaluation(
        method=args.method,
        questions_json=args.questions_json,
        drug_csv=args.drug_csv,
        embedding_npy=args.embedding_npy,
        model_name=args.model_name,
        limit=args.limit,
        top_ks=top_ks,
        require_disease_match=not args.allow_symptom_fallback,
    )

    print(EVALUATION_SCOPE_NOTE)
    print()
    print("Summary metrics")
    print(summary.to_string(index=False))
    print()
    print(f"Queries evaluated: {len(query_metrics)}")
    print(f"Ranked rows collected: {len(ranked_results)}")


if __name__ == "__main__":
    main()
