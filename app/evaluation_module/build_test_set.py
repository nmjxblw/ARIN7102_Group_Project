"""
Stage 1: Build candidate drug-recall test set from local data only.

This script intentionally does not use an LLM for `symptom_text`. It prefers
real patient-style sentences from `eval_dataset_llm_v2.json` and only falls
back to a simple template when no suitable sentence exists for a sampled
combination.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DRUG_TABLE_CSV = (
    REPO_ROOT
    / "match_data_preprocessing"
    / "data"
    / "enhanced_drug_table_v1_structured.csv"
)
BERT_OUTPUT_JSON = (
    REPO_ROOT
    / "app"
    / "dataset_module"
    / "drugs_training_dataset"
    / "eval_dataset_llm_v2.json"
)


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


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def generate_symptom_text_template(diseases: list[str], symptoms: list[str]) -> str:
    disease_phrase = ", ".join(diseases[:3]) if diseases else "some discomfort"
    symptom_phrase = (
        ", ".join(s.replace("_", " ") for s in symptoms[:5])
        if symptoms
        else "general malaise"
    )
    return (
        f"I've been having {symptom_phrase}. "
        f"I'm worried it might be related to {disease_phrase}."
    )


def row_symptom_text(row: dict) -> str:
    return row["sentence"] or generate_symptom_text_template(
        row["diseases"], row["symptoms"]
    )


def load_excluded_symptom_texts(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Exclude input not found: {path}")
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            text = str(row.get("symptom_text", "")).strip()
            if text:
                excluded.add(text)
    return excluded


def load_drug_table() -> pd.DataFrame:
    df = pd.read_csv(DRUG_TABLE_CSV)
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")
    df["total_reviews"] = pd.to_numeric(df["total_reviews"], errors="coerce").fillna(0)
    df["disease_key_list_parsed"] = df["disease_key_list"].apply(parse_list_field)
    df["symptom_list_parsed"] = df["symptom_list"].apply(parse_list_field)
    df["score"] = df["avg_rating"] * np.log1p(df["total_reviews"])
    df = df[df["disease_key_list_parsed"].map(bool)].copy()
    df = df[df["disease_key_list_parsed"].apply(lambda xs: any(x != "others" for x in xs))]
    return df


def load_query_rows() -> list[dict]:
    with open(BERT_OUTPUT_JSON, encoding="utf-8") as f:
        raw_rows = json.load(f)

    rows: list[dict] = []
    for idx, row in enumerate(raw_rows):
        disease_items = [
            {
                "label": d.get("label", "").strip(),
                "confidence": d.get("confidence"),
            }
            for d in row.get("diseases", [])
            if d.get("label")
        ]
        symptom_items = [
            {
                "label": s.get("label", "").strip(),
                "confidence": s.get("confidence"),
            }
            for s in row.get("symptoms", [])
            if s.get("label")
        ]
        diseases = dedupe_keep_order([d["label"] for d in disease_items if d["label"]])
        diseases = [d for d in diseases if d != "others"]
        symptoms = dedupe_keep_order([s["label"] for s in symptom_items if s["label"]])
        sentence = str(row.get("sentence", "")).strip()
        if not diseases:
            continue
        rows.append(
            {
                "source_idx": idx,
                "diseases": diseases,
                "symptoms": symptoms,
                "sentence": sentence,
                "disease_items": disease_items,
                "symptom_items": symptom_items,
            }
        )
    return rows


def build_disease_drug_index(drug_table: pd.DataFrame) -> dict[str, list[dict]]:
    """Index rated drugs by disease, ordered by business score descending."""
    rated = drug_table[drug_table["avg_rating"].notna()].copy()
    rated = rated.sort_values("score", ascending=False)

    disease_to_drugs: dict[str, list[dict]] = defaultdict(list)
    for row in rated.itertuples(index=False):
        diseases = [d for d in row.disease_key_list_parsed if d != "others"]
        symptoms = set(row.symptom_list_parsed)
        entry = {
            "drug_name": row.drug_name,
            "diseases": diseases,
            "symptoms": symptoms,
            "score": float(row.score) if pd.notna(row.score) else 0.0,
        }
        for disease in diseases:
            disease_to_drugs[disease].append(entry)
    return disease_to_drugs


def select_relevant_drugs(
    diseases: list[str],
    symptoms: list[str],
    disease_to_drugs: dict[str, list[dict]],
    per_disease: int = 3,
) -> tuple[list[str], dict[str, int]] | None:
    symptom_set = set(symptoms)
    chosen: list[str] = []
    scores: dict[str, int] = {}
    limit = max(3, min(9, per_disease * max(1, len(diseases))))

    for disease_index, disease in enumerate(diseases):
        picked_for_disease = 0
        disease_pool = list(disease_to_drugs.get(disease, []))
        if symptom_set:
            disease_pool.sort(
                key=lambda drug: (
                    len(drug["symptoms"] & symptom_set),
                    drug["score"],
                    -len(drug["diseases"]),
                ),
                reverse=True,
            )
        for drug in disease_pool:
            name = str(drug["drug_name"]).strip()
            if not name or name in scores:
                continue
            if symptom_set and not (drug["symptoms"] & symptom_set):
                continue
            chosen.append(name)
            scores[name] = 3 if disease_index == 0 and picked_for_disease == 0 else 2
            picked_for_disease += 1
            if picked_for_disease >= per_disease or len(chosen) >= limit:
                break
        if len(chosen) >= limit:
            break

    if len(chosen) < 3:
        return None
    return chosen[:limit], {name: scores[name] for name in chosen[:limit]}


def finalize_case(
    idx: int,
    case: dict,
    rng: random.Random,
    confidence_range: tuple[float, float],
    query_id_start: int = 1,
) -> dict:
    low, high = confidence_range
    diseases = [
        {"name": d["name"], "confidence": round(d["confidence"], 2)}
        for d in case["diseases"]
    ]
    symptoms = [
        {"name": s["name"], "confidence": round(s["confidence"], 2)}
        for s in case["symptoms"]
    ]

    if not diseases:
        diseases = [
            {"name": d, "confidence": round(rng.uniform(low, high), 2)}
            for d in case["disease_names"]
        ]
    if not symptoms:
        symptoms = [
            {"name": s, "confidence": round(rng.uniform(low, high), 2)}
            for s in case["symptom_names"]
        ]

    return {
        "query_id": f"eval_{idx + query_id_start:04d}",
        "symptom_text": case["symptom_text"],
        "diseases": diseases,
        "symptoms": symptoms,
        "relevant_drugs": case["relevant_drugs"],
        "relevance_scores": case["relevance_scores"],
    }


def query_to_candidate(
    row: dict,
    disease_to_drugs: dict[str, list[dict]],
    rng: random.Random,
    confidence_range: tuple[float, float],
    per_disease: int,
) -> dict | None:
    selected = select_relevant_drugs(
        row["diseases"],
        row["symptoms"],
        disease_to_drugs,
        per_disease=per_disease,
    )
    if selected is None:
        return None

    relevant_drugs, relevance_scores = selected
    low, high = confidence_range

    def _resolve_confidence(value) -> float:
        return float(value) if value is not None else rng.uniform(low, high)

    disease_items = [
        {"name": d["label"], "confidence": _resolve_confidence(d.get("confidence"))}
        for d in row.get("disease_items", [])
        if d["label"] != "others"
    ]
    symptom_items = [
        {"name": s["label"], "confidence": _resolve_confidence(s.get("confidence"))}
        for s in row.get("symptom_items", [])
    ]

    return {
        "source_idx": row["source_idx"],
        "disease_names": row["diseases"],
        "symptom_names": row["symptoms"],
        "diseases": disease_items,
        "symptoms": symptom_items,
        "symptom_text": row_symptom_text(row),
        "relevant_drugs": relevant_drugs,
        "relevance_scores": relevance_scores,
    }


def build_query_buckets(query_rows: list[dict]) -> dict[str, list[dict]]:
    buckets = {"single": [], "double": [], "triple_plus": []}
    for row in query_rows:
        if not row["diseases"]:
            continue
        count = len(row["diseases"])
        key = "single" if count == 1 else "double" if count == 2 else "triple_plus"
        buckets[key].append(row)
    return buckets


def collect_bucket_cases(
    bucket_rows: list[dict],
    bucket_name: str,
    target_count: int,
    disease_to_drugs: dict[str, list[dict]],
    rng: random.Random,
    confidence_range: tuple[float, float],
    per_disease: int,
    used_symptom_texts: set[str],
) -> list[dict]:
    rows = list(bucket_rows)
    rng.shuffle(rows)
    cases: list[dict] = []
    used_sources: set[int] = set()

    for row in rows:
        if len(cases) >= target_count:
            break
        candidate = query_to_candidate(
            row,
            disease_to_drugs,
            rng,
            confidence_range,
            per_disease,
        )
        if candidate is None:
            continue
        if candidate["symptom_text"] in used_symptom_texts:
            continue
        if candidate["source_idx"] in used_sources:
            continue
        used_sources.add(candidate["source_idx"])
        used_symptom_texts.add(candidate["symptom_text"])
        cases.append(candidate)

    if len(cases) >= target_count:
        return cases

    # Fallback: synthesize from remaining disease combinations but keep the same
    # local-label matching rules for relevant drugs.
    combo_to_rows: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        combo_to_rows[tuple(row["diseases"])].append(row)

    combo_items = list(combo_to_rows.items())
    rng.shuffle(combo_items)
    while len(cases) < target_count and combo_items:
        added_this_round = 0
        for combo, combo_rows in combo_items:
            if len(cases) >= target_count:
                break
            base_row = rng.choice(combo_rows)
            synthetic_row = dict(base_row)
            synthetic_row["sentence"] = base_row["sentence"] or generate_symptom_text_template(
                base_row["diseases"], base_row["symptoms"]
            )
            candidate = query_to_candidate(
                synthetic_row,
                disease_to_drugs,
                rng,
                confidence_range,
                per_disease,
            )
            if candidate is None:
                continue
            if candidate["symptom_text"] in used_symptom_texts:
                continue
            used_symptom_texts.add(candidate["symptom_text"])
            cases.append(candidate)
            added_this_round += 1
        if added_this_round == 0:
            break
        if len(cases) < target_count and not cases:
            raise ValueError(f"Unable to build any valid {bucket_name} cases.")
        if len(cases) < target_count and len(combo_items) == 1:
            break

    if len(cases) < target_count:
        raise ValueError(
            f"Unable to build enough {bucket_name} cases: "
            f"need {target_count}, got {len(cases)}"
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: build candidate test set")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--single-count", type=int, default=150)
    parser.add_argument("--double-count", type=int, default=100)
    parser.add_argument("--triple-plus-count", type=int, default=50)
    parser.add_argument("--drugs-per-disease", type=int, default=3)
    parser.add_argument("--confidence-min", type=float, default=0.55)
    parser.add_argument("--confidence-max", type=float, default=0.85)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "eval_dataset_candidates.json",
    )
    parser.add_argument(
        "--exclude-input",
        action="append",
        type=Path,
        default=[],
        help="Existing candidate JSON file(s); generated cases with matching symptom_text are skipped.",
    )
    parser.add_argument(
        "--query-id-start",
        type=int,
        default=1,
        help="First numeric suffix for generated query_id values.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.single_count + args.double_count + args.triple_plus_count != args.num_samples:
        raise ValueError(
            "single-count + double-count + triple-plus-count must equal num-samples"
        )

    rng = random.Random(args.seed)
    confidence_range = (args.confidence_min, args.confidence_max)

    print("[build_test_set] loading drug table…")
    drug_table = load_drug_table()
    disease_to_drugs = build_disease_drug_index(drug_table)
    print(f"[build_test_set] disease pool: {len(disease_to_drugs)} diseases")

    print("[build_test_set] loading query rows…")
    query_rows = load_query_rows()
    excluded_texts = load_excluded_symptom_texts(args.exclude_input)
    if excluded_texts:
        before_count = len(query_rows)
        query_rows = [
            row for row in query_rows
            if row_symptom_text(row) not in excluded_texts
        ]
        print(
            f"[build_test_set] excluded {before_count - len(query_rows)} rows "
            f"by symptom_text from {len(args.exclude_input)} file(s)"
        )
    buckets = build_query_buckets(query_rows)
    print(
        "[build_test_set] query buckets:",
        {
            "single": len(buckets["single"]),
            "double": len(buckets["double"]),
            "triple_plus": len(buckets["triple_plus"]),
        },
    )

    cases: list[dict] = []
    used_symptom_texts: set[str] = set()
    cases += collect_bucket_cases(
        buckets["single"],
        "single",
        args.single_count,
        disease_to_drugs,
        rng,
        confidence_range,
        args.drugs_per_disease,
        used_symptom_texts,
    )
    cases += collect_bucket_cases(
        buckets["double"],
        "double",
        args.double_count,
        disease_to_drugs,
        rng,
        confidence_range,
        args.drugs_per_disease,
        used_symptom_texts,
    )
    cases += collect_bucket_cases(
        buckets["triple_plus"],
        "triple_plus",
        args.triple_plus_count,
        disease_to_drugs,
        rng,
        confidence_range,
        args.drugs_per_disease,
        used_symptom_texts,
    )
    rng.shuffle(cases)

    finalized = [
        finalize_case(idx, case, rng, confidence_range, args.query_id_start)
        for idx, case in enumerate(cases[: args.num_samples])
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(finalized, f, indent=2, ensure_ascii=False)

    d_counts = Counter(len(c["diseases"]) for c in finalized)
    r_counts = Counter(len(c["relevant_drugs"]) for c in finalized)
    print(f"[build_test_set] wrote {len(finalized)} cases → {args.output}")
    print(f"[build_test_set] diseases-per-query: {dict(sorted(d_counts.items()))}")
    print(f"[build_test_set] relevant_drugs-per-query: {dict(sorted(r_counts.items()))}")


if __name__ == "__main__":
    main()
