import argparse
import json
import logging
from pathlib import Path

from app.embedded_module.drug_recall_index import DrugRecallIndex
from app.embedded_module.phase2_final_recommender import Phase2FinalRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase II Final Recommendation Pipeline")
    parser.add_argument("--input-json", type=str, help="Path to input JSON file with queries")
    parser.add_argument("--diseases", type=str, nargs="+", help="Space-separated list of diseases for debug query")
    parser.add_argument("--symptoms", type=str, nargs="+", default=[], help="Space-separated list of symptoms for debug query")
    parser.add_argument("--output-json", type=str, help="Path to save the output JSON")
    parser.add_argument("--top-k-recall", type=int, default=20, help="Phase II recall Top K limit")
    parser.add_argument("--top-k-per-disease", type=int, default=3, help="Final Top K limit per disease")
    parser.add_argument("--table-path", type=str, default=str(REPO_ROOT / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1_structured.csv"))
    parser.add_argument("--half1-path", type=str, default=str(REPO_ROOT / "app" / "dataset_module" / "drugs_training_dataset" / "drug_data_half_1.json"))
    parser.add_argument("--half2-path", type=str, default=str(REPO_ROOT / "app" / "dataset_module" / "drugs_training_dataset" / "drug_data_half_2.json"))
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
            "sentence": "Debug query"
        }
        queries = [query]
    else:
        logger.error("Must provide either --input-json or --diseases")
        return

    import pandas as pd
    logger.info(f"Loading drug table from {args.table_path}...")
    df = pd.read_csv(args.table_path)
    
    logger.info("Loading DrugRecallIndex...")
    index = DrugRecallIndex(
        df=df,
        embedding_path=None,  # Not used in label_core_rerank
    )
    
    logger.info("Initializing Phase2FinalRecommender...")
    recommender = Phase2FinalRecommender(
        index=index,
        half_data_paths=[args.half1_path, args.half2_path],
        table_path=args.table_path
    )

    all_grouped_results = []
    all_flat_results = []
    
    logger.info(f"Processing {len(queries)} queries...")
    for q_idx, query in enumerate(queries):
        if "query_index" not in query:
            query["query_index"] = q_idx
            
        res = recommender.recommend_query(
            query=query,
            top_k_recall=args.top_k_recall,
            top_k_per_disease=args.top_k_per_disease
        )
        
        all_grouped_results.append(res)
        
        # Build flat view for this query
        global_rank = 1
        for d_res in res["disease_results"]:
            for top_drug in d_res["final_top3"]:
                flat_item = {
                    "query_index": res["query_index"],
                    "disease": d_res["disease"],
                    "disease_confidence": d_res["disease_confidence"],
                    "drug_name": top_drug["drug_name"],
                    "disease_rank": top_drug["disease_rank"],
                    "global_display_rank": global_rank,
                    "phase2_rank": top_drug["phase2_rank"],
                    "selection_source": top_drug["selection_source"]
                }
                # add phase2_score and half_disease_confidence if available
                if "phase2_score" in top_drug:
                    flat_item["phase2_score"] = top_drug["phase2_score"]
                if "half_disease_confidence" in top_drug:
                    flat_item["half_disease_confidence"] = top_drug["half_disease_confidence"]
                    
                all_flat_results.append(flat_item)
                global_rank += 1

    output = {
        "queries": all_grouped_results,
        "recommendations": all_flat_results
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {out_path}")
    else:
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
