"""
Generate evaluation dataset from drug table.

For each drug, create queries based on its indications and symptoms,
then mark that drug as the ground truth.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


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


def generate_query_text(diseases: list[str], symptoms: list[str]) -> str:
    parts = []
    if symptoms:
        symptom_phrase = ", ".join(symptoms[:3])
        parts.append(f"I have {symptom_phrase}")
    if diseases and random.random() > 0.3:
        parts.append(f"related to {diseases[0]}")
    return ". ".join(parts) if parts else "I need medication"


def generate_eval_samples(
    df: pd.DataFrame,
    num_samples: int,
    min_diseases: int = 1,
    min_symptoms: int = 1,
) -> list[dict]:
    samples = []

    df["disease_list"] = df["matched_disease_keys"].apply(parse_list_field)
    df["symptom_list"] = df["matched_symptoms"].apply(parse_list_field)

    candidates = df[
        (df["disease_list"].str.len() >= min_diseases) &
        (df["symptom_list"].str.len() >= min_symptoms)
    ].copy()

    if len(candidates) == 0:
        raise ValueError("No drugs meet the criteria")

    sampled = candidates.sample(n=min(num_samples, len(candidates)), random_state=42)

    for idx, (_, row) in enumerate(sampled.iterrows()):
        diseases = row["disease_list"]
        symptoms = row["symptom_list"]
        drug_name = str(row.get("drug_name", "")).strip()

        if not drug_name:
            continue

        disease_items = [
            {"name": d, "confidence": round(random.uniform(0.7, 0.95), 2)}
            for d in diseases[:3]
        ]
        symptom_items = [
            {"name": s, "confidence": round(random.uniform(0.65, 0.92), 2)}
            for s in symptoms[:4]
        ]

        symptom_text = generate_query_text(diseases, symptoms)

        samples.append({
            "query_id": f"eval_{idx+1:04d}",
            "symptom_text": symptom_text,
            "diseases": disease_items,
            "symptoms": symptom_items,
            "relevant_drugs": [drug_name],
            "relevance_scores": {drug_name: 3}
        })

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--min-diseases", type=int, default=1)
    parser.add_argument("--min-symptoms", type=int, default=1)
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    samples = generate_eval_samples(
        df,
        num_samples=args.num_samples,
        min_diseases=args.min_diseases,
        min_symptoms=args.min_symptoms,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(samples)} evaluation samples")
    print(f"Saved to: {args.output_json}")


if __name__ == "__main__":
    main()
