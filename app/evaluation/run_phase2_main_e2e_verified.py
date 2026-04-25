"""
Phase2 end-to-end evaluator on the verified set.

Entrypoint:

    python -m app.evaluation.run_phase2_main_e2e_verified [OPTIONS]

Pipeline (fixed):
    symptom_text
        -> deployment_module.predict()                # BERT multi-task classifier
        -> run_phase2_final_recommendation helpers    # normalize + Phase2FinalRecommender
        -> flattened drug_name list
        -> evaluation.metrics.evaluate_batch()
        -> artifacts/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"

# Keep this evaluator runnable without requiring the long env prefix used by
# `app/__main__.py`. Only fill gaps; never override user-provided values.
_RUNTIME_DEFAULTS = {
    "BERT_TRAINING_DATASET_FOLDER": "app/dataset_module/bert_training_dataset",
    "DRUGS_TRAINING_DATASET_FOLDER": "app/dataset_module/drugs_training_dataset",
    "BERT_FOLDER": "app/deployment_module/clinicalbert_local",
    "TRAINED_BERT_SAVE_PATH": "app/deployment_module/trained_bert",
    "CLINICAL_BERT": "app/deployment_module/clinicalbert_local",
    "DATABASE_FILE": "app/database_module/database.db",
    "CHAT_HISTORY_DIR": "app/remote_llm_module/chat_histories",
}
for _key, _value in _RUNTIME_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from deployment_module.bert_main import (
    preload as bert_preload,
    predict_with_preload as bert_predict_with_preload,
)
from evaluation.metrics import evaluate_batch, evaluate_single_query
import evaluation.run_phase2_final_recommendation as phase2_module

DEFAULT_EVAL_DATASET = REPO_ROOT / "data" / "eval_dataset_verified.json"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase2_main_e2e_verified"
DEFAULT_K_VALUES = [5, 10, 20]


def _trained_bert_dir() -> Path:
    return REPO_ROOT / os.environ["TRAINED_BERT_SAVE_PATH"]


def _validate_runtime_inputs(
    *,
    eval_dataset_path: Path,
    table_path: Path,
    half1_path: Path,
    half2_path: Path,
) -> None:
    if not eval_dataset_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {eval_dataset_path}")

    bert_dir = _trained_bert_dir()
    if not bert_dir.exists():
        raise FileNotFoundError(f"Trained BERT directory not found: {bert_dir}")
    if not (
        (bert_dir / "model.safetensors").exists()
        or (bert_dir / "pytorch_model.bin").exists()
    ):
        raise FileNotFoundError(
            f"No BERT weight file found in {bert_dir}; expected model.safetensors or pytorch_model.bin"
        )

    for path, label in (
        (table_path, "Phase2 table"),
        (half1_path, "Phase2 half1 data"),
        (half2_path, "Phase2 half2 data"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def _build_phase2_recommender(
    *,
    table_path: Path,
    half1_path: Path,
    half2_path: Path,
) -> phase2_module.Phase2FinalRecommender:
    df = pd.read_csv(table_path)
    index = phase2_module.DrugRecallIndex(
        df=df,
        embedding_path=None,
    )
    return phase2_module.Phase2FinalRecommender(
        index=index,
        half_data_paths=[half1_path, half2_path],
        table_path=table_path,
    )


def _phase2_predict_from_text(
    *,
    symptom_text: str,
    query_index: int,
    bert_runtime: tuple[Any, Any, Any, Any, Any, dict[str, Any]],
    recommender: phase2_module.Phase2FinalRecommender,
    top_k_recall: int,
    top_k_per_disease: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    (
        inference_device,
        tokenizer,
        inference_model,
        mlb_d,
        mlb_s,
        medians,
    ) = bert_runtime
    bert_output = bert_predict_with_preload(
        text=symptom_text,
        tokenizer=tokenizer,
        inference_device=inference_device,
        inference_model=inference_model,
        mlb_d=mlb_d,
        mlb_s=mlb_s,
        medians=medians,
    )
    # Preserve the raw user text for phase2 query construction. The deployment
    # BERT output does not carry it forward by default.
    bert_payload = dict(bert_output)
    bert_payload["sentence"] = symptom_text

    query = phase2_module._normalize_bert_output(bert_payload)
    query["query_index"] = query_index

    phase2_result = recommender.recommend_query(
        query=query,
        top_k_recall=top_k_recall,
        top_k_per_disease=top_k_per_disease,
    )
    flat = phase2_module._build_flat_recommendations(phase2_result)
    phase2_output = {
        "input": query,
        "disease_results": phase2_result.get("disease_results", []),
        "recommendations": flat,
    }
    recommended = [str(item.get("drug_name", "")) for item in flat if item.get("drug_name")]
    return bert_output, phase2_output, recommended


def run_phase2_e2e_verified(
    *,
    eval_dataset_path: Path,
    artifact_dir: Path,
    k_values: list[int],
    top_k_recall: int,
    top_k_per_disease: int,
    table_path: Path,
    half1_path: Path,
    half2_path: Path,
    limit: int | None = None,
) -> dict[str, float]:
    _validate_runtime_inputs(
        eval_dataset_path=eval_dataset_path,
        table_path=table_path,
        half1_path=half1_path,
        half2_path=half2_path,
    )

    with open(eval_dataset_path, encoding="utf-8") as fh:
        eval_data: list[dict[str, Any]] = json.load(fh)
    if limit is not None:
        eval_data = eval_data[:limit]

    total = len(eval_data)
    print(f"[Phase2 E2E] Loaded {total} queries from {eval_dataset_path}")
    print("[Phase2 E2E] Initializing Phase2FinalRecommender ...")
    recommender = _build_phase2_recommender(
        table_path=table_path,
        half1_path=half1_path,
        half2_path=half2_path,
    )
    print("[Phase2 E2E] Preloading BERT classifier ...")
    bert_runtime = bert_preload(_trained_bert_dir())
    print("[Phase2 E2E] Recommender and BERT ready.\n")

    batch_results: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    t_start = time.time()
    n_ok = 0
    n_err = 0

    for i, query in enumerate(eval_data):
        query_id = str(query.get("query_id", f"q{i:04d}"))
        symptom_text = str(query.get("symptom_text", ""))
        relevant_drugs = [str(item) for item in query.get("relevant_drugs", [])]
        relevance_scores_raw = query.get("relevance_scores", {})
        relevance_scores = (
            relevance_scores_raw
            if isinstance(relevance_scores_raw, dict) and relevance_scores_raw
            else None
        )

        print(f"[{i + 1:>4}/{total}] {query_id}  text={symptom_text[:60]!r}")

        try:
            bert_output, phase2_output, recommended = _phase2_predict_from_text(
                symptom_text=symptom_text,
                query_index=i,
                bert_runtime=bert_runtime,
                recommender=recommender,
                top_k_recall=top_k_recall,
                top_k_per_disease=top_k_per_disease,
            )
        except Exception as exc:
            print(f"  x ERROR: {exc}")
            n_err += 1
            continue

        per_query_metrics = evaluate_single_query(
            recommended=recommended,
            relevant=relevant_drugs,
            relevance_scores=relevance_scores,
            k_values=k_values,
        )

        batch_results.append(
            {
                "query_id": query_id,
                "recommended": recommended,
                "relevant": relevant_drugs,
                "relevance_scores": relevance_scores,
            }
        )
        per_query_rows.append(
            {
                "query_id": query_id,
                "symptom_text": symptom_text,
                "predicted_diseases": json.dumps(
                    bert_output.get("diseases", []),
                    ensure_ascii=False,
                ),
                "predicted_symptoms": json.dumps(
                    bert_output.get("symptoms", []),
                    ensure_ascii=False,
                ),
                "recommended": "|".join(recommended),
                "relevant": "|".join(relevant_drugs),
                **{key: round(value, 6) for key, value in per_query_metrics.items()},
            }
        )
        prediction_records.append(
            {
                "query_id": query_id,
                "symptom_text": symptom_text,
                "bert_output": bert_output,
                "phase2_output": phase2_output,
                "recommended": recommended,
                "relevant_drugs": relevant_drugs,
            }
        )
        n_ok += 1

    elapsed = time.time() - t_start
    print(
        f"\n[Phase2 E2E] Done - ok={n_ok}, err={n_err}, total={total}, "
        f"elapsed={elapsed:.1f}s ({elapsed / max(n_ok, 1):.2f}s/query)\n"
    )

    if n_ok == 0:
        raise RuntimeError("All queries failed; no metrics to compute.")

    agg_metrics = evaluate_batch(
        [
            {
                "recommended": item["recommended"],
                "relevant": item["relevant"],
                "relevance_scores": item["relevance_scores"],
            }
            for item in batch_results
        ],
        k_values=k_values,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "eval_dataset": str(eval_dataset_path),
        "num_queries_total": total,
        "num_queries_evaluated": n_ok,
        "num_queries_failed": n_err,
        "k_values": k_values,
        "top_k_recall": top_k_recall,
        "top_k_per_disease": top_k_per_disease,
        "elapsed_seconds": round(elapsed, 2),
        "phase2_assets": {
            "table_path": str(table_path),
            "half1_path": str(half1_path),
            "half2_path": str(half2_path),
            "trained_bert_dir": str(_trained_bert_dir()),
        },
        "metrics": {key: round(value, 6) for key, value in agg_metrics.items()},
    }
    metrics_path = artifact_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, ensure_ascii=False)

    csv_path = artifact_dir / "per_query_results.csv"
    if per_query_rows:
        fieldnames = list(per_query_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query_rows)

    jsonl_path = artifact_dir / "predictions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for record in prediction_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Phase2 E2E] metrics.json -> {metrics_path}")
    print(f"[Phase2 E2E] per_query_results.csv -> {csv_path}  ({len(per_query_rows)} rows)")
    print(f"[Phase2 E2E] predictions.jsonl -> {jsonl_path}  ({len(prediction_records)} lines)")
    print("\n" + "=" * 60)
    print("PHASE2 E2E EVALUATION  [verified-set]")
    print("=" * 60)
    for metric_name, value in sorted(agg_metrics.items()):
        print(f"  {metric_name:22s}: {value:.4f}")
    print("=" * 60)

    return agg_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pure phase2 end-to-end evaluation on eval_dataset_verified.json.\n"
            "Pipeline: symptom_text -> BERT -> Phase2FinalRecommender -> metrics"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-dataset",
        type=Path,
        default=DEFAULT_EVAL_DATASET,
        metavar="PATH",
        help=f"Path to verified eval dataset JSON (default: {DEFAULT_EVAL_DATASET})",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        metavar="DIR",
        help=f"Output directory for artifacts (default: {DEFAULT_ARTIFACT_DIR})",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        metavar="K",
        help=f"K values for P/R/Hit/NDCG@K (default: {DEFAULT_K_VALUES})",
    )
    parser.add_argument(
        "--top-k-recall",
        type=int,
        default=phase2_module.DEFAULT_TOP_K_RECALL,
        metavar="N",
        help=(
            "Phase2 candidate recall Top-K per disease "
            f"(default: {phase2_module.DEFAULT_TOP_K_RECALL})"
        ),
    )
    parser.add_argument(
        "--top-k-per-disease",
        type=int,
        default=phase2_module.DEFAULT_TOP_K_PER_DISEASE,
        metavar="N",
        help=(
            "Phase2 final top-k kept per disease "
            f"(default: {phase2_module.DEFAULT_TOP_K_PER_DISEASE})"
        ),
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=phase2_module.DEFAULT_TABLE_PATH,
        metavar="PATH",
        help=f"Structured drug table path (default: {phase2_module.DEFAULT_TABLE_PATH})",
    )
    parser.add_argument(
        "--half1-path",
        type=Path,
        default=phase2_module.DEFAULT_HALF1_PATH,
        metavar="PATH",
        help=f"Half1 drug data path (default: {phase2_module.DEFAULT_HALF1_PATH})",
    )
    parser.add_argument(
        "--half2-path",
        type=Path,
        default=phase2_module.DEFAULT_HALF2_PATH,
        metavar="PATH",
        help=f"Half2 drug data path (default: {phase2_module.DEFAULT_HALF2_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N queries (smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    k_values = sorted(set(args.k_values))

    print("=" * 60)
    print("run_phase2_main_e2e_verified")
    print(f"  eval_dataset      : {args.eval_dataset}")
    print(f"  artifact_dir      : {args.artifact_dir}")
    print(f"  k_values          : {k_values}")
    print(f"  top_k_recall      : {args.top_k_recall}")
    print(f"  top_k_per_disease : {args.top_k_per_disease}")
    print(f"  table_path        : {args.table_path}")
    print(f"  half1_path        : {args.half1_path}")
    print(f"  half2_path        : {args.half2_path}")
    print(f"  limit             : {args.limit}")
    print("=" * 60 + "\n")

    run_phase2_e2e_verified(
        eval_dataset_path=args.eval_dataset,
        artifact_dir=args.artifact_dir,
        k_values=k_values,
        top_k_recall=args.top_k_recall,
        top_k_per_disease=args.top_k_per_disease,
        table_path=args.table_path,
        half1_path=args.half1_path,
        half2_path=args.half2_path,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
