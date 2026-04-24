"""
Test the recommendation pipeline with a single query from eval_dataset_verified.json.
Run from project root: python test_pipeline_single.py
"""
import sys
import json
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
os.chdir(APP_ROOT)

# Load .env from app directory
from dotenv import load_dotenv
load_dotenv(APP_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Patch DEEPSEEK_API_KEY if not set (not needed for pipeline test)
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "dummy_for_pipeline_test"

from fastapi_module.service import get_recommendation_service

def main():
    # Test case from eval_dataset_verified.json
    test_case = {
        "query_id": "eval_0011",
        "symptom_text": "I have big, painful bumps on my skin that leave scars. How can I treat this?",
        "diseases": [{"name": "acne", "confidence": 1.0}],
        "symptoms": [
            {"name": "nodal_skin_eruptions", "confidence": 1.0},
            {"name": "scurring", "confidence": 1.0},
        ],
        "relevant_drugs": ["doxycycline", "clindamycin", "sulfamethoxazole / trimethoprim"],
    }

    print("=" * 60)
    print("Pipeline Single-Query Test")
    print("=" * 60)
    print(f"Query: {test_case['symptom_text']}")
    print(f"Diseases: {test_case['diseases']}")
    print(f"Symptoms: {test_case['symptoms']}")
    print(f"Expected drugs: {test_case['relevant_drugs']}")
    print()

    print("Initializing service...")
    service = get_recommendation_service()
    service.ensure_ready()
    print("Service ready.\n")

    print("Running recommendation with trace enabled...")
    result_df, trace = service.recommend(
        symptom_text=test_case["symptom_text"],
        diseases=test_case["diseases"],
        symptoms=test_case["symptoms"],
        top_k=10,
        recall_top_k_each=300,
        fused_top_k=300,
        recall_weight_semantic=0.5,
        recall_weight_label=0.5,
        enable_trace=True,
    )

    print("\n" + "=" * 60)
    print("TOP-10 RECOMMENDED DRUGS")
    print("=" * 60)
    cols = ["drug_name", "final_score", "semantic_score", "label_score", "cross_encoder_score", "business_score"]
    available_cols = [c for c in cols if c in result_df.columns]
    print(result_df[available_cols].to_string(index=False))

    # Check hits
    recommended = result_df["drug_name"].tolist()
    relevant_set = set(test_case["relevant_drugs"])
    hits = [d for d in recommended if d in relevant_set]
    print(f"\n--- Hit Analysis ---")
    print(f"Relevant drugs in top-10: {hits}")
    print(f"Recall@10: {len(hits)}/{len(relevant_set)} = {len(hits)/len(relevant_set):.2f}")

    print("\n" + "=" * 60)
    print("PIPELINE TRACE")
    print("=" * 60)
    print(json.dumps(trace.to_dict(), indent=2))


if __name__ == "__main__":
    main()
