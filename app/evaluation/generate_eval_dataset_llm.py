"""
Generate evaluation dataset using LLM to create natural language queries.
"""

from __future__ import annotations
import sys
import os
import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .llm_client import LLMClient
from static_module import DRUGS_TRAINING_DATASET_FOLDER


def parse_list_field(value) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s or s == "[]":
        return []
    s = s.strip("[]").replace("'", "").replace('"', "")
    return [x.strip() for x in s.split(",") if x.strip()]


def build_prompt(drug_name: str, diseases: list[str], symptoms: list[str]) -> str:
    """Build prompt for LLM to generate patient self-description."""
    disease_text = ", ".join(diseases[:2]) if diseases else "general health issues"
    symptom_text = ", ".join(symptoms[:4]) if symptoms else "discomfort"

    prompt = f"""You are a patient who needs the medication "{drug_name}".

This medication is typically used for: {disease_text}
Common symptoms it treats: {symptom_text}

Write a brief, natural self-description (1-2 sentences) of how you feel and why you might need this medication. Write from first-person perspective as if you're describing your symptoms to a doctor or pharmacist.

Examples:
- "I have a severe headache and fever that started yesterday, and I feel very weak."
- "My throat is extremely sore and I'm having trouble swallowing. I also have a mild fever."
- "I've been experiencing chest pain and shortness of breath when I exercise."

Your description:"""

    return prompt


def generate_query_with_llm(
    llm_client: LLMClient,
    drug_name: str,
    diseases: list[str],
    symptoms: list[str],
    retry: int = 3,
) -> str | None:
    """Generate natural language query using LLM."""
    prompt = build_prompt(drug_name, diseases, symptoms)

    for attempt in range(retry):
        try:
            response = llm_client.generate(prompt, temperature=0.8, max_tokens=150)
            if response and len(response) > 10:
                return response
        except Exception as e:
            print(f"  LLM error (attempt {attempt+1}/{retry}): {e}")
            if attempt < retry - 1:
                time.sleep(2)

    return None


def generate_eval_samples_llm(
    df: pd.DataFrame,
    llm_client: LLMClient,
    num_samples: int,
    min_diseases: int = 1,
    min_symptoms: int = 1,
    confidence_range: tuple[float, float] = (0.5, 0.75),
) -> list[dict]:
    """Generate evaluation samples using LLM for natural queries."""
    samples = []

    df["disease_list"] = df["matched_disease_keys"].apply(parse_list_field)
    df["symptom_list"] = df["matched_symptoms"].apply(parse_list_field)

    candidates = df[
        (df["disease_list"].str.len() >= min_diseases)
        & (df["symptom_list"].str.len() >= min_symptoms)
    ].copy()

    if len(candidates) == 0:
        raise ValueError("No drugs meet the criteria")

    sampled = candidates.sample(n=min(num_samples, len(candidates)), random_state=42)

    for idx, (_, row) in enumerate(
        tqdm(sampled.iterrows(), total=len(sampled), desc="Generating")
    ):
        diseases = row["disease_list"]
        symptoms = row["symptom_list"]
        drug_name = str(row.get("drug_name", "")).strip()

        if not drug_name:
            continue

        symptom_text = generate_query_with_llm(
            llm_client, drug_name, diseases, symptoms
        )

        if not symptom_text:
            print(f"  Skipping {drug_name}: LLM generation failed")
            continue

        min_conf, max_conf = confidence_range
        disease_items = [
            {"name": d, "confidence": round(random.uniform(min_conf, max_conf), 2)}
            for d in diseases[:3]
        ]
        symptom_items = [
            {"name": s, "confidence": round(random.uniform(min_conf, max_conf), 2)}
            for s in symptoms[:4]
        ]

        samples.append(
            {
                "query_id": f"eval_{idx+1:04d}",
                "symptom_text": symptom_text,
                "diseases": disease_items,
                "symptoms": symptom_items,
                "relevant_drugs": [drug_name],
                "relevance_scores": {drug_name: 3},
            }
        )

    return samples


def generate_eval_dataset_llm_main():
    parser = argparse.ArgumentParser()
    input_csv_default: Path = (
        Path.cwd()
        / DRUGS_TRAINING_DATASET_FOLDER
        / r"enhanced_drug_table_v1_structured.csv"
    )
    if not input_csv_default.exists():
        print(f"Warning: Default input CSV not found at {input_csv_default}")
        raise
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=input_csv_default,
    )
    output_json_default: Path = (
        Path.cwd() / DRUGS_TRAINING_DATASET_FOLDER / "eval_dataset_llm.json"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=output_json_default,
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--min-diseases", type=int, default=1)
    parser.add_argument("--min-symptoms", type=int, default=1)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default="deepseek-v3-2-251201")
    parser.add_argument("--confidence-min", type=float, default=0.5)
    parser.add_argument("--confidence-max", type=float, default=0.75)
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    llm_client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    df = pd.read_csv(args.input_csv)
    samples = generate_eval_samples_llm(
        df,
        llm_client,
        num_samples=args.num_samples,
        min_diseases=args.min_diseases,
        min_symptoms=args.min_symptoms,
        confidence_range=(args.confidence_min, args.confidence_max),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(samples)} evaluation samples")
    print(f"Saved to: {args.output_json}")


if __name__ == "__main__":
    generate_eval_dataset_llm_main()
