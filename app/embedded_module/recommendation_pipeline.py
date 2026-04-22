"""
Reusable helpers for the drug recommendation experiments.

This module keeps notebook logic focused on orchestration and display,
while the core preparation / retrieval utilities live in plain Python.
"""

from __future__ import annotations

import ast
from typing import Iterable

import numpy as np
import pandas as pd


def parse_list_like(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [text]


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def normalize_symptom_name(symptom) -> str:
    return str(symptom).strip().lower().replace("_", " ")


def safe_cosine_scores(candidate_vecs, query_vec, debug: bool = False) -> np.ndarray:
    candidate_vecs = np.asarray(candidate_vecs, dtype=np.float32)
    query_vec = np.asarray(query_vec, dtype=np.float32)

    if candidate_vecs.ndim != 2:
        raise ValueError(f"candidate_vecs should be 2D, got shape={candidate_vecs.shape}")
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    if query_vec.ndim != 2:
        raise ValueError(f"query_vec should be 2D, got shape={query_vec.shape}")

    c_norm = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
    q_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)

    zero_c = int((c_norm == 0).sum())
    zero_q = int((q_norm == 0).sum())
    if debug and (zero_c > 0 or zero_q > 0):
        print(f"[debug] zero-norm vectors detected: candidates={zero_c}, queries={zero_q}")

    c_norm = np.where(c_norm == 0, 1.0, c_norm)
    q_norm = np.where(q_norm == 0, 1.0, q_norm)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scores = ((candidate_vecs / c_norm) @ (query_vec / q_norm).T).flatten()

    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _base_info_text(row: pd.Series) -> str:
    """Build base info part: drug name, ingredient, class, diseases, symptoms."""
    parts: list[str] = []
    if clean_text(row.get("drug_name")):
        parts.append(f"Drug: {clean_text(row.get('drug_name'))}.")
    if clean_text(row.get("generic_name")):
        parts.append(f"Ingredient: {clean_text(row.get('generic_name'))}.")
    if clean_text(row.get("drug_classes")):
        parts.append(f"Class: {clean_text(row.get('drug_classes'))}.")
    if row.get("disease_key_list"):
        parts.append("Diseases: " + ", ".join(row["disease_key_list"]) + ".")
    if row.get("symptom_list_normalized"):
        parts.append("Symptoms: " + ", ".join(row["symptom_list_normalized"][:15]) + ".")
    return " ".join(parts) if parts else ""


def build_semantic_text(row: pd.Series) -> str:
    """Build base semantic text (no description). Used for cross-encoder reranking."""
    return _base_info_text(row) or "[MASK]"


def build_semantic_text_mcd(row: pd.Series) -> str:
    """Build semantic text: base info + medical_condition_description.

    medical_condition_description 偏口语/患者视角，包含通俗症状描述。
    """
    base = _base_info_text(row)
    mcd = clean_text(row.get("medical_condition_description"))
    if mcd:
        return f"{base} Description: {mcd}" if base else f"Description: {mcd}"
    return base or "[MASK]"


def build_semantic_text_dd(row: pd.Series) -> str:
    """Build semantic text: base info + disease_description.

    disease_description 偏医学/临床视角，包含专业病理描述。
    """
    base = _base_info_text(row)
    dd = clean_text(row.get("disease_description"))
    if dd:
        return f"{base} Description: {dd}" if base else f"Description: {dd}"
    return base or "[MASK]"


def prepare_drug_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["disease_key_list"] = result["matched_disease_keys"].apply(parse_list_like)
    result["symptom_list"] = result["matched_symptoms"].apply(parse_list_like)
    result["symptom_list_normalized"] = result["symptom_list"].apply(
        lambda items: [normalize_symptom_name(x) for x in items]
    )
    result["semantic_text"] = result.apply(build_semantic_text, axis=1)
    result["semantic_text_dd"] = result.apply(build_semantic_text_dd, axis=1)
    return result


def build_query_text(disease_labels: Iterable[str], symptom_terms: Iterable[str]) -> str:
    """Build query text from labels, matching the concise drug text format."""
    parts = []
    disease_labels = [str(x).strip() for x in disease_labels if str(x).strip()]
    symptom_terms = [normalize_symptom_name(x) for x in symptom_terms if str(x).strip()]
    if disease_labels:
        parts.append("Diseases: " + ", ".join(disease_labels) + ".")
    if symptom_terms:
        parts.append("Symptoms: " + ", ".join(symptom_terms) + ".")
    return " ".join(parts) if parts else "[MASK]"


def baseline_filter(df: pd.DataFrame, disease_labels, symptom_terms) -> pd.DataFrame:
    disease_labels = {str(x).strip() for x in disease_labels}
    symptom_terms = {normalize_symptom_name(x) for x in symptom_terms}

    def match_row(row: pd.Series):
        disease_overlap = len(set(row["disease_key_list"]) & disease_labels)
        symptom_overlap = len(set(row["symptom_list_normalized"]) & symptom_terms)
        return disease_overlap, symptom_overlap

    scores = df.apply(match_row, axis=1, result_type="expand")
    scores.columns = ["disease_overlap", "symptom_overlap"]
    result = pd.concat([df.copy(), scores], axis=1)
    result = result[(result["disease_overlap"] > 0) | (result["symptom_overlap"] > 0)].copy()
    result["avg_rating_filled"] = result["avg_rating"].fillna(0)
    return result.sort_values(
        by=["disease_overlap", "symptom_overlap", "avg_rating_filled"],
        ascending=[False, False, False],
    )


def query_drugs_hybrid(
    df: pd.DataFrame,
    drug_embeddings: np.ndarray,
    engine,
    disease_labels,
    symptom_terms,
    top_k: int = 10,
    require_disease_match: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    candidates = baseline_filter(df, disease_labels, symptom_terms)
    if require_disease_match:
        candidates = candidates[candidates["disease_overlap"] > 0].copy()

    if len(candidates) == 0:
        if debug:
            print("[debug] no strict disease match found, fallback to baseline candidates")
        candidates = baseline_filter(df, disease_labels, symptom_terms)
        if len(candidates) == 0:
            return pd.DataFrame()

    candidate_indices = candidates.index.tolist()
    candidate_vecs = drug_embeddings[candidate_indices].copy().astype(np.float32)
    query_vec = engine.encode_query(build_query_text(disease_labels, symptom_terms)).astype(np.float32)
    candidates = candidates.copy()
    candidates["knn_score"] = safe_cosine_scores(candidate_vecs, query_vec, debug=debug)
    candidates["avg_rating_filled"] = candidates["avg_rating"].fillna(0)
    candidates = candidates.sort_values(
        by=["disease_overlap", "symptom_overlap", "knn_score", "avg_rating_filled"],
        ascending=[False, False, False, False],
    )
    return candidates
