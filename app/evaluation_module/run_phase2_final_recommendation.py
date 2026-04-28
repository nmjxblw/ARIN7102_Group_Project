import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from app.embedded_module.drug_recall_index import DrugRecallIndex
from app.embedded_module.phase2_final_recommender import Phase2FinalRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE_PATH = (
    REPO_ROOT
    / "match_data_preprocessing"
    / "data"
    / "enhanced_drug_table_v1_structured.csv"
)
DEFAULT_HALF1_PATH = (
    REPO_ROOT
    / "app"
    / "dataset_module"
    / "drugs_training_dataset"
    / "drug_data_half_1.json"
)
DEFAULT_HALF2_PATH = (
    REPO_ROOT
    / "app"
    / "dataset_module"
    / "drugs_training_dataset"
    / "drug_data_half_2.json"
)
DEFAULT_TOP_K_RECALL = 20
DEFAULT_TOP_K_PER_DISEASE = 3

_RECOMMENDER_CACHE: Phase2FinalRecommender | None = None


def _confidence(value: Any, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _normalize_label_items(items: Any) -> list[dict]:
    normalized = []
    for item in items or []:
        if isinstance(item, str):
            label = item.strip()
            confidence = 1.0
        elif isinstance(item, dict):
            if "label" in item:
                label = str(item.get("label") or "").strip()
                confidence = _confidence(item.get("confidence"), default=1.0)
            elif "name" in item:
                label = str(item.get("name") or "").strip()
                confidence = _confidence(item.get("confidence"), default=1.0)
            elif len(item) == 1:
                label = str(next(iter(item.keys())) or "").strip()
                confidence = _confidence(next(iter(item.values())), default=1.0)
            else:
                continue
        else:
            continue

        if label:
            normalized.append({"label": label, "confidence": confidence})
    return normalized


def _normalize_bert_output(bert_output: dict) -> dict:
    if not isinstance(bert_output, dict):
        raise TypeError("bert_output must be a dict")

    return {
        "query_index": 0,
        "diseases": _normalize_label_items(bert_output.get("diseases", [])),
        "symptoms": _normalize_label_items(bert_output.get("symptoms", [])),
        "need_first_aid": bert_output.get("need_first_aid", 0),
        "sentence": str(bert_output.get("sentence") or ""),
    }


def _get_recommender() -> Phase2FinalRecommender:
    global _RECOMMENDER_CACHE
    if _RECOMMENDER_CACHE is None:
        import pandas as pd

        df = pd.read_csv(DEFAULT_TABLE_PATH)
        index = DrugRecallIndex(
            df=df,
            embedding_path=None,
        )
        _RECOMMENDER_CACHE = Phase2FinalRecommender(
            index=index,
            half_data_paths=[DEFAULT_HALF1_PATH, DEFAULT_HALF2_PATH],
            table_path=DEFAULT_TABLE_PATH,
            phase2_mode=os.getenv("PHASE2_MODE", "xgb_ranker"),
            ranker_path=os.getenv(
                "PHASE2_RANKER_PATH",
                str(
                    REPO_ROOT
                    / "artifacts/exp_drug_recall/phase2_e2e_xgb_train/ranker.joblib"
                ),
            ),
        )
    return _RECOMMENDER_CACHE


def _build_flat_recommendations(result: dict) -> list[dict]:
    flat_results = []
    global_rank = 1
    for d_res in result.get("disease_results", []):
        for top_drug in d_res.get("final_top3", []):
            flat_item = {
                "query_index": result.get("query_index", 0),
                "disease": d_res.get("disease", ""),
                "disease_confidence": d_res.get("disease_confidence", 1.0),
                "drug_name": top_drug.get("drug_name", ""),
                "disease_rank": top_drug.get("disease_rank", global_rank),
                "global_display_rank": global_rank,
                "phase2_rank": top_drug.get("phase2_rank"),
                "selection_source": top_drug.get("selection_source", ""),
            }
            if "phase2_score" in top_drug:
                flat_item["phase2_score"] = top_drug["phase2_score"]
            if "half_disease_confidence" in top_drug:
                flat_item["half_disease_confidence"] = top_drug[
                    "half_disease_confidence"
                ]
            flat_results.append(flat_item)
            global_rank += 1
    return flat_results


def predict(bert_output: dict) -> dict:
    query = _normalize_bert_output(bert_output)
    recommender = _get_recommender()
    result = recommender.recommend_query(
        query=query,
        top_k_recall=DEFAULT_TOP_K_RECALL,
        top_k_per_disease=DEFAULT_TOP_K_PER_DISEASE,
    )
    return {
        "input": query,
        "disease_results": result.get("disease_results", []),
        "recommendations": _build_flat_recommendations(result),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Phase II Final Recommendation Pipeline"
    )
    parser.add_argument(
        "--input-json", type=str, help="Path to input JSON file with queries"
    )
    parser.add_argument(
        "--diseases",
        type=str,
        nargs="+",
        help="Space-separated list of diseases for debug query",
    )
    parser.add_argument(
        "--symptoms",
        type=str,
        nargs="+",
        default=[],
        help="Space-separated list of symptoms for debug query",
    )
    parser.add_argument("--output-json", type=str, help="Path to save the output JSON")
    parser.add_argument(
        "--top-k-recall", type=int, default=20, help="Phase II recall Top K limit"
    )
    parser.add_argument(
        "--top-k-per-disease", type=int, default=3, help="Final Top K limit per disease"
    )
    parser.add_argument("--table-path", type=str, default=str(DEFAULT_TABLE_PATH))
    parser.add_argument("--half1-path", type=str, default=str(DEFAULT_HALF1_PATH))
    parser.add_argument("--half2-path", type=str, default=str(DEFAULT_HALF2_PATH))
    return parser.parse_args()


def main():
    args = parse_args()

    queries = []
    if args.input_json:
        with open(args.input_json, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                queries = [data]
            else:
                queries = data
    elif args.diseases:
        query = {
            "query_index": 0,
            "diseases": [{"label": d, "confidence": 1.0} for d in args.diseases],
            "symptoms": [{"label": s, "confidence": 1.0} for s in args.symptoms],
            "sentence": "Debug query",
        }
        queries = [query]
    else:
        logger.error("Must provide either --input-json or --diseases")
        return

    import pandas as pd

    logger.debug(f"Loading drug table from {args.table_path}...")
    df = pd.read_csv(args.table_path)

    logger.debug("Loading DrugRecallIndex...")
    index = DrugRecallIndex(
        df=df,
        embedding_path=None,  # Not used in label_core_rerank
    )

    logger.debug("Initializing Phase2FinalRecommender...")
    recommender = Phase2FinalRecommender(
        index=index,
        half_data_paths=[args.half1_path, args.half2_path],
        table_path=args.table_path,
    )

    all_grouped_results = []
    all_flat_results = []

    logger.debug(f"Processing {len(queries)} queries...")
    for q_idx, query in enumerate(queries):
        if "query_index" not in query:
            query["query_index"] = q_idx

        res = recommender.recommend_query(
            query=query,
            top_k_recall=args.top_k_recall,
            top_k_per_disease=args.top_k_per_disease,
        )

        all_grouped_results.append(res)

        all_flat_results.extend(_build_flat_recommendations(res))

    output = {"queries": all_grouped_results, "recommendations": all_flat_results}

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        logger.debug(f"Results saved to {out_path}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
