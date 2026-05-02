"""Run ablations for the experimental local drug recall pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path.cwd()  # Assuming this script is run from the repo root; adjust if not
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# Import directly from module files to avoid __init__.py cascading torch load
from embedded_module.drug_ranker import LocalDrugRanker
from embedded_module.drug_recall_index import DrugRecallIndex
from embedded_module.experimental_recall_pipeline import (
    ABLATION_MODES,
    ExperimentalDrugRecallPipeline,
)
import embedded_module.label_adapter as label_adapter_module
from evaluation_module.metrics import evaluate_batch, evaluate_single_query

parse_labels = label_adapter_module.parse_labels


DEFAULT_TABLE = (
    REPO_ROOT
    / "match_data_preprocessing"
    / "data"
    / "enhanced_drug_table_v1_structured.csv"
)
DEFAULT_EVAL = REPO_ROOT / "data" / "eval_dataset_verified.json"
DEFAULT_EMBEDDINGS = REPO_ROOT / "drug_comprehensive_embeddings.npy"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "exp_drug_recall"
DEFAULT_WEAK_TRAIN = (
    REPO_ROOT
    / "app"
    / "dataset_module"
    / "drugs_training_dataset"
    / "eval_dataset_llm_v2.json"
)


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normal_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _stage_hit_rates(trace, relevant: list[str]) -> dict[str, float]:
    relevant_set = {_normal_name(name) for name in relevant}
    rates = {}
    for stage, names in trace.stage_candidate_names.items():
        stage_set = {_normal_name(name) for name in names}
        rates[f"stage_hit_{stage}"] = 1.0 if relevant_set & stage_set else 0.0
    rates["stage_hit_union"] = 1.0 if any(value for value in rates.values()) else 0.0
    return rates


def _maybe_load_ranker(path: Path | None) -> LocalDrugRanker | None:
    if path is None or not path.exists():
        return None
    return LocalDrugRanker.load(path)


def _maybe_load_query_encoder(enable_dense: bool, model_name: str | None):
    if not enable_dense:
        return None
    from embedded_module.drug_embedding_engine import DrugEmbeddingEngine

    return (
        DrugEmbeddingEngine(model_name=model_name)
        if model_name
        else DrugEmbeddingEngine()
    )


def train_ranker_from_weak_data(
    *,
    pipeline: ExperimentalDrugRecallPipeline,
    weak_data_path: Path,
    output_path: Path,
    limit: int,
    pool_size: int,
) -> LocalDrugRanker:
    weak_data = _load_json(weak_data_path)[:limit]
    feature_batches = []
    label_batches = []

    for i, query in enumerate(weak_data, start=1):
        if i % 100 == 0:
            print(f"[ranker train] {i}/{len(weak_data)}")

        disease_labels = parse_labels(query.get("diseases", []), kind="disease")
        symptom_labels = parse_labels(query.get("symptoms", []), kind="symptom")
        query_text = pipeline._build_query_text(
            query.get("sentence", ""), disease_labels, symptom_labels
        )  # noqa: SLF001
        stage_scores = pipeline._collect_candidates(  # noqa: SLF001
            query_text=query_text,
            diseases=disease_labels,
            symptoms=symptom_labels,
            pool_size=pool_size,
        )
        row_ids = pipeline._rows_for_mode(
            stage_scores, mode="candidate_union", pool_size=pool_size
        )  # noqa: SLF001
        features = pipeline._build_feature_frame(  # noqa: SLF001
            row_ids=row_ids,
            stage_scores=stage_scores,
            diseases=disease_labels,
            symptoms=symptom_labels,
            mode="candidate_union",
        )
        if len(features) == 0:
            continue
        scored = pipeline._score_features(
            features, mode="candidate_union"
        )  # noqa: SLF001

        positive_mask = (scored["disease_conf_overlap"] > 0) & (
            (scored["symptom_conf_overlap"] > 0)
            | (scored["symptom_coverage"] >= 0.25)
            | (scored["stage_strict"] > 0)
        )
        positives = scored[positive_mask].head(20)
        negatives = (
            scored[~positive_mask]
            .sort_values("deterministic_score", ascending=False)
            .head(max(20, len(positives) * 3))
        )
        if len(positives) == 0 or len(negatives) == 0:
            continue

        feature_batches.append(pd.concat([positives, negatives], axis=0))
        label_batches.append(
            np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
        )

    if not feature_batches:
        raise ValueError("No weak training samples were generated for the ranker")

    features_all = pd.concat(feature_batches, axis=0, ignore_index=True)
    labels_all = np.concatenate(label_batches).astype(int)
    ranker = LocalDrugRanker().fit(features_all, labels_all)
    ranker.save(output_path)
    print(
        f"Ranker trained on {len(labels_all)} weak candidate rows and saved to: {output_path}"
    )
    return ranker


def run_evaluation(
    *,
    table_path: Path,
    eval_dataset_path: Path,
    eval_kind: str,
    half_grouping: str,
    half_extra_datasets: list[Path],
    embedding_path: Path | None,
    artifact_dir: Path,
    modes: list[str],
    k_values: list[int],
    pool_size: int,
    ranker_path: Path | None,
    enable_dense: bool,
    model_name: str | None,
    limit: int | None,
    train_ranker: bool,
    weak_data_path: Path,
    train_limit: int,
    train_pool_size: int,
) -> dict:
    df = pd.read_csv(table_path)
    embeddings = (
        np.load(embedding_path) if embedding_path and embedding_path.exists() else None
    )
    index = DrugRecallIndex(df, embedding_path=embedding_path, embeddings=embeddings)
    query_encoder = _maybe_load_query_encoder(enable_dense, model_name)
    ranker = _maybe_load_ranker(ranker_path)
    pipeline = ExperimentalDrugRecallPipeline(
        index, ranker=ranker, query_encoder=query_encoder
    )

    if train_ranker:
        if ranker_path is None:
            raise ValueError("--train-ranker requires a --ranker output path")
        ranker = train_ranker_from_weak_data(
            pipeline=pipeline,
            weak_data_path=weak_data_path,
            output_path=ranker_path,
            limit=train_limit,
            pool_size=min(pool_size, train_pool_size),
        )
        pipeline.ranker = ranker

    if eval_kind == "half":
        from evaluation_module.half_data_adapter import convert_half_datasets

        half_paths = [eval_dataset_path, *half_extra_datasets]
        eval_data = convert_half_datasets(
            half_paths,
            grouping=half_grouping,
            table_path=table_path,
        )
        allowed_modes = {"label_idf_only", "label_core_rerank"}
        if any(m not in allowed_modes for m in modes):
            raise ValueError(f"Half eval only supports {allowed_modes}")
    else:
        eval_data = _load_json(eval_dataset_path)
    if limit is not None:
        eval_data = eval_data[:limit]

    artifact_dir.mkdir(parents=True, exist_ok=True)
    per_query_rows = []
    trace_path = artifact_dir / "stage_trace.jsonl"
    metrics_by_mode = {}

    with open(trace_path, "w", encoding="utf-8") as trace_file:
        for mode in modes:
            print(f"\nMode: {mode}")
            batch_results = []
            stage_rows = []
            for i, query in enumerate(eval_data, start=1):
                print(f"[{i}/{len(eval_data)}] {query['query_id']}")
                result_df, trace = pipeline.recommend(
                    symptom_text=query.get("symptom_text", ""),
                    disease_items=query.get("diseases", []),
                    symptom_items=query.get("symptoms", []),
                    top_k=max(k_values),
                    pool_size=pool_size,
                    mode=mode,
                    return_trace=True,
                )
                recommended = (
                    result_df["drug_name"].astype(str).tolist()
                    if len(result_df)
                    else []
                )
                relevant = query.get("relevant_drugs", [])
                relevance_scores = query.get("relevance_scores", {})
                single_metrics = evaluate_single_query(
                    recommended=recommended,
                    relevant=relevant,
                    relevance_scores=relevance_scores,
                    k_values=k_values,
                )
                stage_metrics = _stage_hit_rates(trace, relevant)
                stage_rows.append(stage_metrics)
                batch_results.append(
                    {
                        "query_id": query["query_id"],
                        "recommended": recommended,
                        "relevant": relevant,
                        "relevance_scores": relevance_scores,
                    }
                )
                per_query_rows.append(
                    {
                        "mode": mode,
                        "query_id": query["query_id"],
                        "recommended": " | ".join(recommended),
                        "relevant": " | ".join(relevant),
                        "candidate_union_size": trace.final_union_size,
                        **trace.candidate_counts,
                        **single_metrics,
                        **stage_metrics,
                    }
                )
                trace_file.write(
                    json.dumps(
                        {
                            "mode": mode,
                            "query_id": query["query_id"],
                            "trace": trace.to_dict(include_candidates=False),
                            "top_recommended": recommended[:20],
                            "relevant": relevant,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            metrics = evaluate_batch(batch_results, k_values=k_values)
            if stage_rows:
                stage_df = pd.DataFrame(stage_rows)
                metrics.update(stage_df.mean().to_dict())
            metrics_by_mode[mode] = metrics

    per_query_df = pd.DataFrame(per_query_rows)
    per_query_df.to_csv(artifact_dir / "per_query_results.csv", index=False)
    with open(artifact_dir / "asset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(index.embedding_manifest.to_dict(), f, indent=2)
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_dataset": str(eval_dataset_path),
                "half_extra_datasets": (
                    [str(path) for path in half_extra_datasets]
                    if eval_kind == "half"
                    else []
                ),
                "eval_kind": eval_kind,
                "half_grouping": half_grouping if eval_kind == "half" else None,
                "num_queries": len(eval_data),
                "k_values": k_values,
                "modes": modes,
                "metrics": metrics_by_mode,
            },
            f,
            indent=2,
        )
    return metrics_by_mode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate experimental drug recall ablations."
    )
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--eval-dataset", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--eval-kind", choices=["verified", "half"], default="verified")
    parser.add_argument(
        "--half-grouping",
        choices=["row", "disease", "disease_symptom"],
        default="disease",
        help="How to convert half JSON when --eval-kind half.",
    )
    parser.add_argument(
        "--half-extra-dataset",
        action="append",
        type=Path,
        default=[],
        help="Additional half JSON split(s) to merge before grouped half evaluation.",
    )
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    # Aside-plan default batch: 6 ablation modes (no local_ranker in default run)
    DEFAULT_BATCH = [
        "label_idf_only",
        "label_bm25",
        "candidate_union_no_prior",
        "candidate_union_no_bm25",
        "candidate_union_no_prior_no_bm25",
        "candidate_union",
    ]
    parser.add_argument(
        "--modes", nargs="+", choices=ABLATION_MODES, default=DEFAULT_BATCH
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument(
        "--ranker", type=Path, default=DEFAULT_ARTIFACT_DIR / "ranker.joblib"
    )
    parser.add_argument("--enable-dense", action="store_true")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-ranker", action="store_true")
    parser.add_argument("--weak-data", type=Path, default=DEFAULT_WEAK_TRAIN)
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--train-pool-size", type=int, default=300)
    args = parser.parse_args()

    metrics = run_evaluation(
        table_path=args.table,
        eval_dataset_path=args.eval_dataset,
        eval_kind=args.eval_kind,
        half_grouping=args.half_grouping,
        half_extra_datasets=args.half_extra_dataset,
        embedding_path=args.embeddings,
        artifact_dir=args.artifact_dir,
        modes=args.modes,
        k_values=args.k_values,
        pool_size=args.pool_size,
        ranker_path=args.ranker,
        enable_dense=args.enable_dense,
        model_name=args.model_name,
        limit=args.limit,
        train_ranker=args.train_ranker,
        weak_data_path=args.weak_data,
        train_limit=args.train_limit,
        train_pool_size=args.train_pool_size,
    )

    print("\n" + "=" * 64)
    print("EXPERIMENTAL DRUG RECALL RESULTS")
    print("=" * 64)
    for mode, values in metrics.items():
        print(f"\n[{mode}]")
        for key, value in sorted(values.items()):
            print(f"{key:24s}: {value:.4f}")


if __name__ == "__main__":
    main()
