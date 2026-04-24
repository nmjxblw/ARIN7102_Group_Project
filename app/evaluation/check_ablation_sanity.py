#!/usr/bin/env python3
"""Sanity-check abnormal gains in the no-bm25 ablations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from embedded_module.drug_recall_index import DrugRecallIndex
from embedded_module.experimental_recall_pipeline import ExperimentalDrugRecallPipeline
from embedded_module.label_adapter import parse_labels
from evaluation.metrics import evaluate_batch


DEFAULT_TABLE = REPO_ROOT / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1_structured.csv"
DEFAULT_EVAL = REPO_ROOT / "data" / "eval_dataset_verified.json"
DEFAULT_EMBEDDINGS = REPO_ROOT / "drug_comprehensive_embeddings.npy"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "exp_drug_recall"


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_pipeline(table_path: Path, embedding_path: Path | None) -> tuple[pd.DataFrame, ExperimentalDrugRecallPipeline]:
    df = pd.read_csv(table_path)
    embeddings = None
    if embedding_path and embedding_path.exists():
        embeddings = np.load(embedding_path)
    index = DrugRecallIndex(df, embedding_path=embedding_path, embeddings=embeddings)
    pipeline = ExperimentalDrugRecallPipeline(index, query_encoder=None)
    return df, pipeline


def _query_parts(pipeline: ExperimentalDrugRecallPipeline, query: dict) -> tuple[list, list, str]:
    diseases = parse_labels(query.get("diseases", []), kind="disease")
    symptoms = parse_labels(query.get("symptoms", []), kind="symptom")
    query_text = pipeline._build_query_text(query["symptom_text"], diseases, symptoms)  # noqa: SLF001
    return diseases, symptoms, query_text


def _mode_rows(
    pipeline: ExperimentalDrugRecallPipeline,
    *,
    query: dict,
    pool_size: int,
) -> tuple[dict[str, dict[int, float]], dict[str, set[int]]]:
    diseases, symptoms, query_text = _query_parts(pipeline, query)
    stage_scores = pipeline._collect_candidates(  # noqa: SLF001
        query_text=query_text,
        diseases=diseases,
        symptoms=symptoms,
        pool_size=pool_size,
    )
    rows = {
        "label_idf_only": set(pipeline._rows_for_mode(stage_scores, mode="label_idf_only", pool_size=pool_size)),  # noqa: SLF001
        "candidate_union": set(pipeline._rows_for_mode(stage_scores, mode="candidate_union", pool_size=pool_size)),  # noqa: SLF001
        "candidate_union_no_bm25": set(
            pipeline._rows_for_mode(stage_scores, mode="candidate_union_no_bm25", pool_size=pool_size)  # noqa: SLF001
        ),
        "candidate_union_no_prior_no_bm25": set(
            pipeline._rows_for_mode(stage_scores, mode="candidate_union_no_prior_no_bm25", pool_size=pool_size)  # noqa: SLF001
        ),
    }
    return stage_scores, rows


def _summarize_stage_scales(per_query_stage_values: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
    summary = {}
    for stage, values in per_query_stage_values.items():
        if not values:
            continue
        summary[stage] = {
            "queries_with_stage": len(values),
            "mean_max": round(float(np.mean([item["max"] for item in values])), 4),
            "mean_p95": round(float(np.mean([item["p95"] for item in values])), 4),
            "mean_median": round(float(np.mean([item["median"] for item in values])), 4),
        }
    return summary


def _pool_sweep(
    pipeline: ExperimentalDrugRecallPipeline,
    *,
    queries: list[dict],
    pool_sizes: list[int],
) -> dict[str, dict[str, float]]:
    sweep = {}
    for pool_size in pool_sizes:
        batch_results = []
        for query in queries:
            result = pipeline.recommend(
                symptom_text=query["symptom_text"],
                disease_items=query.get("diseases", []),
                symptom_items=query.get("symptoms", []),
                top_k=20,
                pool_size=pool_size,
                mode="candidate_union",
            )
            recommended = result["drug_name"].astype(str).tolist() if len(result) else []
            batch_results.append(
                {
                    "query_id": query["query_id"],
                    "recommended": recommended,
                    "relevant": query.get("relevant_drugs", []),
                    "relevance_scores": query.get("relevance_scores", {}),
                }
            )
        metrics = evaluate_batch(batch_results, k_values=[20])
        sweep[str(pool_size)] = {
            "hit@20": round(float(metrics["hit@20"]), 4),
            "recall@20": round(float(metrics["recall@20"]), 4),
            "mrr": round(float(metrics["mrr"]), 4),
            "precision@20": round(float(metrics["precision@20"]), 4),
        }
    return sweep


def build_sanity_report(
    *,
    table_path: Path = DEFAULT_TABLE,
    eval_dataset_path: Path = DEFAULT_EVAL,
    embedding_path: Path | None = DEFAULT_EMBEDDINGS,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    pool_size: int = 1000,
) -> dict:
    df, pipeline = _build_pipeline(table_path, embedding_path)
    queries = _load_json(eval_dataset_path)
    existing_metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]

    stage_scale_values = {stage: [] for stage in ("disease", "strict", "symptom", "bm25", "prior")}
    same_label_core = 0
    cap_overflow_queries = 0
    bm25_excludes_label_rows_queries = 0
    excluded_label_rows_total = 0
    relevant_excluded_queries = 0
    relevant_exclusion_examples = []
    label_core_sizes = []
    union_pre_cap_sizes = []
    union_selected_sizes = []

    for query in queries:
        stage_scores, rows = _mode_rows(pipeline, query=query, pool_size=pool_size)
        label_core_rows = rows["candidate_union_no_prior_no_bm25"]
        label_only_rows = rows["label_idf_only"]
        union_rows = rows["candidate_union"]

        same_label_core += int(label_core_rows == label_only_rows)
        label_core_sizes.append(len(label_core_rows))
        union_selected_sizes.append(len(union_rows))

        union_pre_cap_rows = set().union(
            stage_scores["disease"],
            stage_scores["strict"],
            stage_scores["symptom"],
            stage_scores["bm25"],
            stage_scores["dense"],
            stage_scores["prior"],
        )
        union_pre_cap_sizes.append(len(union_pre_cap_rows))
        if len(union_pre_cap_rows) > pool_size:
            cap_overflow_queries += 1

        missing_label_rows = label_core_rows - union_rows
        if missing_label_rows:
            bm25_excludes_label_rows_queries += 1
            excluded_label_rows_total += len(missing_label_rows)
            missing_names = {str(df.loc[row_id, "drug_name"]) for row_id in missing_label_rows}
            relevant_overlap = sorted(set(query.get("relevant_drugs", [])) & missing_names)
            if relevant_overlap:
                relevant_excluded_queries += 1
                if len(relevant_exclusion_examples) < 10:
                    relevant_exclusion_examples.append(
                        {
                            "query_id": query["query_id"],
                            "missing_label_rows": len(missing_label_rows),
                            "relevant_excluded": relevant_overlap,
                        }
                    )

        for stage in stage_scale_values:
            values = list(stage_scores[stage].values())
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            stage_scale_values[stage].append(
                {
                    "max": float(arr.max()),
                    "p95": float(np.percentile(arr, 95)),
                    "median": float(np.median(arr)),
                }
            )

    row_count = len(df)
    pool_sizes = [pool_size, 1500, 2000, 3000, row_count]
    pool_sweep = _pool_sweep(pipeline, queries=queries, pool_sizes=pool_sizes)

    conclusions = []
    if same_label_core == len(queries):
        conclusions.append(
            "label_idf_only and candidate_union_no_prior_no_bm25 use the same candidate rows on all queries; "
            "their performance gap comes from deterministic reranking, not extra recall sources."
        )
    if cap_overflow_queries == len(queries) and bm25_excludes_label_rows_queries > 0:
        conclusions.append(
            "candidate_union overflows the 1000-row cap on every query, and BM25 participation causes label-core "
            "rows to be dropped before final scoring."
        )
    sweep_base = pool_sweep[str(pool_size)]
    sweep_full = pool_sweep[str(row_count)]
    if sweep_full["hit@20"] > sweep_base["hit@20"] or sweep_full["recall@20"] > sweep_base["recall@20"]:
        conclusions.append(
            f"candidate_union improves from hit@20={sweep_base['hit@20']:.4f} / recall@20={sweep_base['recall@20']:.4f} "
            f"at pool_size={pool_size} to hit@20={sweep_full['hit@20']:.4f} / recall@20={sweep_full['recall@20']:.4f} "
            f"at pool_size={row_count}; the current 1000-row cap is interacting badly with raw stage-score selection."
        )
    if stage_scale_values["bm25"]:
        stage_summary = _summarize_stage_scales(stage_scale_values)
        bm25_mean_max = stage_summary["bm25"]["mean_max"]
        strict_mean_max = stage_summary["strict"]["mean_max"]
        if bm25_mean_max > strict_mean_max * 2:
            conclusions.append(
                f"BM25 raw scores operate on a much larger scale than strict label scores "
                f"(mean per-query max {bm25_mean_max:.4f} vs {strict_mean_max:.4f}), "
                "so summing raw stage scores before cap selection is numerically biased toward BM25."
            )

    report = {
        "eval_dataset": str(eval_dataset_path),
        "num_queries": len(queries),
        "pool_size_checked": pool_size,
        "baseline_metrics": {
            mode: {
                key: round(float(existing_metrics[mode][key]), 4)
                for key in ("hit@20", "recall@20", "mrr", "precision@20")
            }
            for mode in (
                "label_idf_only",
                "candidate_union",
                "candidate_union_no_prior",
                "candidate_union_no_bm25",
                "candidate_union_no_prior_no_bm25",
            )
        },
        "candidate_set_checks": {
            "same_label_idf_vs_no_prior_no_bm25_queries": same_label_core,
            "same_label_idf_vs_no_prior_no_bm25_ratio": round(same_label_core / len(queries), 4),
            "mean_label_core_size": round(float(np.mean(label_core_sizes)), 2),
            "mean_union_selected_size": round(float(np.mean(union_selected_sizes)), 2),
            "mean_union_pre_cap_size": round(float(np.mean(union_pre_cap_sizes)), 2),
            "cap_overflow_union_queries": cap_overflow_queries,
            "bm25_excludes_label_rows_queries": bm25_excludes_label_rows_queries,
            "excluded_label_rows_total": excluded_label_rows_total,
            "relevant_excluded_queries": relevant_excluded_queries,
        },
        "stage_raw_score_scales": _summarize_stage_scales(stage_scale_values),
        "candidate_union_pool_sweep": pool_sweep,
        "relevant_exclusion_examples": relevant_exclusion_examples,
        "conclusions": conclusions,
    }
    return report


def _write_markdown(report: dict, output_path: Path) -> None:
    baseline = report["baseline_metrics"]
    checks = report["candidate_set_checks"]
    lines = [
        "# Ablation Sanity Check\n\n",
        f"- Eval dataset: `{report['eval_dataset']}`\n",
        f"- Queries: `{report['num_queries']}`\n",
        f"- Checked pool size: `{report['pool_size_checked']}`\n\n",
        "## Baseline Metrics\n\n",
        "| Mode | hit@20 | recall@20 | mrr | precision@20 |\n",
        "|---|---|---|---|---|\n",
    ]
    for mode, metrics in baseline.items():
        lines.append(
            f"| `{mode}` | {metrics['hit@20']:.4f} | {metrics['recall@20']:.4f} | "
            f"{metrics['mrr']:.4f} | {metrics['precision@20']:.4f} |\n"
        )

    lines.extend(
        [
            "\n## Candidate-Set Checks\n\n",
            f"- `label_idf_only` and `candidate_union_no_prior_no_bm25` have identical candidate sets on "
            f"`{checks['same_label_idf_vs_no_prior_no_bm25_queries']}/{report['num_queries']}` queries.\n",
            f"- Mean label-core candidate count: `{checks['mean_label_core_size']}`\n",
            f"- Mean `candidate_union` selected count: `{checks['mean_union_selected_size']}`\n",
            f"- Mean pre-cap union size: `{checks['mean_union_pre_cap_size']}`\n",
            f"- `candidate_union` exceeds the 1000-row cap on `{checks['cap_overflow_union_queries']}/{report['num_queries']}` queries.\n",
            f"- BM25 participation excludes label-core rows on `{checks['bm25_excludes_label_rows_queries']}/{report['num_queries']}` queries.\n",
            f"- Total label-core rows dropped by `candidate_union`: `{checks['excluded_label_rows_total']}`\n",
            f"- Queries where dropped rows contain relevant drugs: `{checks['relevant_excluded_queries']}`\n",
            "\n## Raw Stage Score Scale\n\n",
            "| Stage | queries | mean max | mean p95 | mean median |\n",
            "|---|---|---|---|---|\n",
        ]
    )

    for stage, summary in report["stage_raw_score_scales"].items():
        lines.append(
            f"| `{stage}` | {summary['queries_with_stage']} | {summary['mean_max']:.4f} | "
            f"{summary['mean_p95']:.4f} | {summary['mean_median']:.4f} |\n"
        )

    lines.extend(
        [
            "\n## Candidate Union Pool Sweep\n\n",
            "| pool_size | hit@20 | recall@20 | mrr | precision@20 |\n",
            "|---|---|---|---|---|\n",
        ]
    )
    for pool_size, metrics in report["candidate_union_pool_sweep"].items():
        lines.append(
            f"| `{pool_size}` | {metrics['hit@20']:.4f} | {metrics['recall@20']:.4f} | "
            f"{metrics['mrr']:.4f} | {metrics['precision@20']:.4f} |\n"
        )

    if report["relevant_exclusion_examples"]:
        lines.append("\n## Relevant Exclusion Examples\n\n")
        lines.append("| query_id | dropped label rows | relevant drugs excluded |\n")
        lines.append("|---|---|---|\n")
        for example in report["relevant_exclusion_examples"]:
            lines.append(
                f"| `{example['query_id']}` | {example['missing_label_rows']} | "
                f"{', '.join(example['relevant_excluded'])} |\n"
            )

    lines.append("\n## Conclusions\n\n")
    for conclusion in report["conclusions"]:
        lines.append(f"- {conclusion}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    artifact_dir = DEFAULT_ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report = build_sanity_report()

    json_path = artifact_dir / "ablation_sanity_check.json"
    md_path = artifact_dir / "ablation_sanity_check.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_markdown(report, md_path)

    print(f"Written: {json_path}")
    print(f"Written: {md_path}")
    print("\nConclusions:")
    for line in report["conclusions"]:
        print(f"- {line}")


if __name__ == "__main__":
    main()
