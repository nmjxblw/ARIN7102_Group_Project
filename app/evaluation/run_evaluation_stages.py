"""
Ablation evaluation: test each pipeline stage independently.

Modes (--mode):
  semantic  — semantic recall only (PubMedBERT cosine similarity)
  label     — label recall only (disease/symptom confidence overlap)
  fusion    — semantic + label recall → weighted fusion (no cross-encoder)
  full      — full pipeline: fusion → cross-encoder + business score

Usage:
  python -m app.evaluation.run_evaluation_stages \
    --eval-dataset data/eval_dataset_verified.json \
    --mode semantic --k-values 5 10 20 \
    --output data/eval_results_semantic.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from evaluation.metrics import evaluate_batch
from fastapi_module.service import get_recommendation_service

MODES = ("semantic", "label", "fusion", "full")


def _recommend_by_mode(pipeline, query: dict, top_k: int, mode: str) -> list[str]:
    symptom_text = query["symptom_text"]
    diseases = query["diseases"]
    symptoms = query["symptoms"]

    if mode == "semantic":
        df = pipeline.semantic_recall(symptom_text=symptom_text, top_k=top_k)
        df = df.sort_values("semantic_score_raw", ascending=False)

    elif mode == "label":
        df = pipeline.label_recall(disease_items=diseases, symptom_items=symptoms, top_k=top_k)
        if len(df) == 0:
            return []
        df = df.sort_values("label_score_raw", ascending=False)

    elif mode == "fusion":
        sem = pipeline.semantic_recall(symptom_text=symptom_text, top_k=300)
        lbl = pipeline.label_recall(disease_items=diseases, symptom_items=symptoms, top_k=300)
        df = pipeline.fuse_recalls(sem, lbl, top_k=top_k)
        if len(df) == 0:
            return []
        df = df.sort_values("recall_fused_score", ascending=False)

    else:  # full
        df = pipeline.recommend(
            symptom_text=symptom_text,
            disease_items=diseases,
            symptom_items=symptoms,
            recall_top_k_each=300,
            fused_top_k=300,
            top_k=top_k,
        )

    return df["drug_name"].head(top_k).tolist()


def run_evaluation(
    eval_dataset_path: Path,
    k_values: list[int],
    mode: str,
    output_path: Path | None = None,
) -> dict[str, float]:
    with open(eval_dataset_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    service = get_recommendation_service()
    service.ensure_ready()
    pipeline = service._pipeline

    results = []
    for i, query in enumerate(eval_data):
        print(f"[{i+1}/{len(eval_data)}] {query['query_id']}")
        try:
            recommended = _recommend_by_mode(pipeline, query, top_k=max(k_values), mode=mode)
            results.append({
                "query_id": query["query_id"],
                "recommended": recommended,
                "relevant": query["relevant_drugs"],
                "relevance_scores": query.get("relevance_scores", {}),
            })
        except Exception as e:
            print(f"  Error: {e}")
            continue

    metrics = evaluate_batch(results, k_values=k_values)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"mode": mode, "metrics": metrics, "num_queries": len(results)}, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation by pipeline stage.")
    parser.add_argument("--eval-dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True, help="Pipeline stage to evaluate")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    metrics = run_evaluation(args.eval_dataset, args.k_values, args.mode, args.output)

    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS  [{args.mode}]")
    print("=" * 50)
    for name, value in sorted(metrics.items()):
        print(f"{name:20s}: {value:.4f}")


if __name__ == "__main__":
    main()
