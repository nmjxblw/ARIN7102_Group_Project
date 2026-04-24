"""Normalize disease and symptom labels from BERT-style inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class NormalizedLabel:
    name: str
    confidence: float


_SPACE_RE = re.compile(r"\s+")
_DISEASE_SEPARATOR_RE = re.compile(r"[\s\-]+")
_SYMPTOM_SEPARATOR_RE = re.compile(r"[_\-]+")


def _clip_confidence(value: Any, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def normalize_disease_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _SPACE_RE.sub(" ", text)
    text = _DISEASE_SEPARATOR_RE.sub("_", text)
    return text.strip("_")


def normalize_symptom_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = _SYMPTOM_SEPARATOR_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text


def _parse_one(item: Any, *, kind: str) -> NormalizedLabel | None:
    if isinstance(item, str):
        raw_name = item
        confidence = 1.0
    elif isinstance(item, dict):
        if "name" in item:
            raw_name = item.get("name")
            confidence = _clip_confidence(item.get("confidence"), default=1.0)
        elif "label" in item:
            raw_name = item.get("label")
            confidence = _clip_confidence(item.get("confidence"), default=1.0)
        elif len(item) == 1:
            raw_name = next(iter(item))
            confidence = _clip_confidence(item.get(raw_name), default=1.0)
        else:
            return None
    else:
        return None

    name = normalize_disease_label(raw_name) if kind == "disease" else normalize_symptom_label(raw_name)
    if not name:
        return None
    return NormalizedLabel(name=name, confidence=confidence)


def parse_labels(items: Iterable[Any] | None, *, kind: str) -> list[NormalizedLabel]:
    if kind not in {"disease", "symptom"}:
        raise ValueError(f"Unsupported label kind: {kind}")
    if not items:
        return []

    best_confidence: dict[str, float] = {}
    for item in items:
        parsed = _parse_one(item, kind=kind)
        if parsed is None:
            continue
        best_confidence[parsed.name] = max(best_confidence.get(parsed.name, 0.0), parsed.confidence)

    labels = [NormalizedLabel(name=name, confidence=conf) for name, conf in best_confidence.items()]
    labels.sort(key=lambda x: (-x.confidence, x.name))

    if kind == "disease" and len(labels) > 1:
        labels = [label for label in labels if label.name != "others"]
    return labels


def confidence_map(labels: Iterable[NormalizedLabel]) -> dict[str, float]:
    result: dict[str, float] = {}
    for label in labels:
        result[label.name] = max(result.get(label.name, 0.0), label.confidence)
    return result
