from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _file_prefix(input_path: Path | str) -> str:
    return "half1" if "half_1" in str(input_path) else "half2"


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()[:80]


def _compact_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _label_name(item: Any) -> str:
    if isinstance(item, dict):
        if "name" in item:
            return str(item["name"]).strip()
        if item:
            return str(next(iter(item.keys()))).strip()
        return ""
    return str(item).strip()


def _label_confidence(item: Any) -> float:
    if isinstance(item, dict):
        if "confidence" in item:
            value = item["confidence"]
        elif item:
            value = next(iter(item.values()))
        else:
            value = 1.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _parse_labels(items: list[Any]) -> list[dict]:
    labels = []
    for item in items or []:
        name = _label_name(item)
        if not name:
            continue
        labels.append({"name": name, "confidence": _label_confidence(item)})
    return labels


def _load_table_name_map(table_path: Path | str | None) -> dict[str, str]:
    """Map half-data drug spelling to the drug table's display spelling."""
    if table_path is None:
        return {}
    path = Path(table_path)
    if not path.exists():
        raise FileNotFoundError(f"Drug table not found: {path}")

    name_map: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("drug_name", "")).strip()
            if not name:
                continue
            variants = {
                name.lower(),
                name.lower().replace(" ", "_"),
                name.lower().replace(" / ", "_").replace("/", "_").replace(" ", "_"),
                _compact_name(name),
            }
            for variant in variants:
                name_map.setdefault(variant, name)
    return name_map


def _canonical_drug_name(drug_name: str, table_name_map: dict[str, str]) -> str | None:
    raw = str(drug_name).strip()
    if not raw:
        return None
    if not table_name_map:
        return raw
    candidates = [
        raw.lower(),
        raw.lower().replace("_", " "),
        raw.lower().replace("_", " / "),
        _compact_name(raw),
    ]
    for candidate in candidates:
        if candidate in table_name_map:
            return table_name_map[candidate]
    return None


def _label_payload(name: str, confidence: float = 1.0) -> dict:
    return {"name": name, "confidence": round(float(confidence), 2)}


def _row_level_cases(
    data: list[dict],
    *,
    prefix: str,
    table_name_map: dict[str, str],
    include_others: bool,
) -> list[dict]:
    converted = []
    for i, row in enumerate(data, start=1):
        diseases = [
            label for label in _parse_labels(row.get("diseases", []))
            if include_others or label["name"] != "others"
        ]
        symptoms = _parse_labels(row.get("symptoms", []))
        drug_name = _canonical_drug_name(row.get("drug_name", ""), table_name_map)
        relevant_drugs = [drug_name] if drug_name else []
        converted.append(
            {
                "query_id": f"{prefix}_{i:06d}",
                "symptom_text": "",
                "diseases": diseases,
                "symptoms": symptoms,
                "relevant_drugs": relevant_drugs,
                "relevance_scores": {name: 3 for name in relevant_drugs},
            }
        )
    return converted


def _grouped_cases(
    data: list[dict],
    *,
    prefix: str,
    grouping: str,
    table_name_map: dict[str, str],
    include_others: bool,
) -> list[dict]:
    groups: dict[tuple[str, ...], dict] = defaultdict(
        lambda: {
            "disease_conf": 0.0,
            "symptom_conf": 0.0,
            "drugs": set(),
            "source_rows": 0,
            "unmapped_drugs": set(),
        }
    )

    for row in data:
        canonical_drug = _canonical_drug_name(row.get("drug_name", ""), table_name_map)
        diseases = [
            label for label in _parse_labels(row.get("diseases", []))
            if include_others or label["name"] != "others"
        ]
        symptoms = _parse_labels(row.get("symptoms", []))
        if not diseases:
            continue

        for disease in diseases:
            if grouping == "disease":
                keys = [(disease["name"],)]
            elif grouping == "disease_symptom":
                keys = [
                    (disease["name"], symptom["name"])
                    for symptom in symptoms
                    if symptom["name"]
                ]
            else:
                raise ValueError(f"Unsupported half grouping: {grouping}")

            for key in keys:
                group = groups[key]
                group["source_rows"] += 1
                group["disease_conf"] = max(group["disease_conf"], disease["confidence"])
                if grouping == "disease_symptom":
                    symptom_conf = next(
                        (s["confidence"] for s in symptoms if s["name"] == key[1]),
                        1.0,
                    )
                    group["symptom_conf"] = max(group["symptom_conf"], symptom_conf)
                if canonical_drug:
                    group["drugs"].add(canonical_drug)
                else:
                    raw_name = str(row.get("drug_name", "")).strip()
                    if raw_name:
                        group["unmapped_drugs"].add(raw_name)

    converted = []
    for idx, (key, group) in enumerate(sorted(groups.items()), start=1):
        disease = key[0]
        symptom = key[1] if len(key) > 1 else None
        relevant_drugs = sorted(group["drugs"])
        if not relevant_drugs:
            continue
        query_id = f"{prefix}_{grouping}_{_safe_id('_'.join(key))}_{idx:04d}"
        symptom_text = (
            f"I have {symptom.replace('_', ' ')} related to {disease.replace('_', ' ')}."
            if symptom
            else f"Drugs for {disease.replace('_', ' ')}."
        )
        converted.append(
            {
                "query_id": query_id,
                "symptom_text": symptom_text,
                "diseases": [_label_payload(disease, group["disease_conf"] or 1.0)],
                "symptoms": (
                    [_label_payload(symptom, group["symptom_conf"] or 1.0)]
                    if symptom
                    else []
                ),
                "relevant_drugs": relevant_drugs,
                "relevance_scores": {name: 3 for name in relevant_drugs},
                "source_rows": group["source_rows"],
                "unmapped_drug_count": len(group["unmapped_drugs"]),
            }
        )
    return converted


def convert_half_data(
    input_path: Path | str,
    *,
    grouping: str = "disease",
    table_path: Path | str | None = None,
    include_others: bool = False,
) -> list[dict]:
    """Convert filtered half JSON into evaluator cases.

    `row` preserves the old weak-regression behavior: one half row has exactly
    one relevant drug. `disease` and `disease_symptom` are query-level views
    that merge all half drugs for the same disease, or disease+symptom pair.
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    prefix = _file_prefix(input_path)
    table_name_map = _load_table_name_map(table_path)

    if grouping == "row":
        return _row_level_cases(
            data,
            prefix=prefix,
            table_name_map=table_name_map,
            include_others=include_others,
        )
    if grouping in {"disease", "disease_symptom"}:
        return _grouped_cases(
            data,
            prefix=prefix,
            grouping=grouping,
            table_name_map=table_name_map,
            include_others=include_others,
        )
    raise ValueError("grouping must be one of: row, disease, disease_symptom")


def convert_half_datasets(
    input_paths: list[Path | str],
    *,
    grouping: str = "disease",
    table_path: Path | str | None = None,
    include_others: bool = False,
) -> list[dict]:
    """Convert one or more half JSON files.

    For grouped evaluation, multiple split files must be merged before grouping;
    otherwise drugs in the other split become false negatives for the same
    disease.
    """
    if not input_paths:
        return []
    if len(input_paths) == 1:
        return convert_half_data(
            input_paths[0],
            grouping=grouping,
            table_path=table_path,
            include_others=include_others,
        )

    table_name_map = _load_table_name_map(table_path)
    if grouping == "row":
        converted = []
        for input_path in input_paths:
            with open(input_path, encoding="utf-8") as f:
                data = json.load(f)
            converted.extend(
                _row_level_cases(
                    data,
                    prefix=_file_prefix(input_path),
                    table_name_map=table_name_map,
                    include_others=include_others,
                )
            )
        return converted

    combined = []
    for input_path in input_paths:
        with open(input_path, encoding="utf-8") as f:
            combined.extend(json.load(f))
    if grouping in {"disease", "disease_symptom"}:
        return _grouped_cases(
            combined,
            prefix="half_all",
            grouping=grouping,
            table_name_map=table_name_map,
            include_others=include_others,
        )
    raise ValueError("grouping must be one of: row, disease, disease_symptom")
