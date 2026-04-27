"""
Run evaluation on the drug recommendation system.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from evaluation_module.metrics import evaluate_batch
from fastapi_module.service import get_recommendation_service


def run_evaluation(
    eval_dataset_path: Path,
    k_values: list[int],
    output_path: Path | None = None,
) -> dict[str, float]:
    """Run evaluation on the dataset."""
    with open(eval_dataset_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    service = get_recommendation_service()
    service.ensure_ready()

    results = []
    for i, query in enumerate(eval_data):
        print(f"[{i+1}/{len(eval_data)}] Evaluating query: {query['query_id']}")

        try:
            result_df = service.recommend(
                symptom_text=query["symptom_text"],
                diseases=query["diseases"],
                symptoms=query["symptoms"],
                top_k=max(k_values),
                recall_top_k_each=300,
                fused_top_k=300,
                recall_weight_semantic=0.5,
                recall_weight_label=0.5,
            )

            recommended = result_df["drug_name"].tolist()
            results.append(
                {
                    "query_id": query["query_id"],
                    "recommended": recommended,
                    "relevant": query["relevant_drugs"],
                    "relevance_scores": query.get("relevance_scores", {}),
                }
            )
        except Exception as e:
            print(f"Error on query {query['query_id']}: {e}")
            continue

    metrics = evaluate_batch(results, k_values=k_values)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "num_queries": len(results),
                    "k_values": k_values,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dataset", type=Path, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metrics = run_evaluation(args.eval_dataset, args.k_values, args.output)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric_name, value in sorted(metrics.items()):
        print(f"{metric_name:20s}: {value:.4f}")


if __name__ == "__main__":
    main()
