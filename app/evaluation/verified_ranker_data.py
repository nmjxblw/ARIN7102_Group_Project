"""Helpers for verified-supervised ranker training data."""

from __future__ import annotations

import json
import random
from pathlib import Path


def build_verified_split_manifest(
    queries: list[dict],
    *,
    dataset_path: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    query_ids = [str(query["query_id"]) for query in queries]
    shuffled = list(query_ids)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = max(n_total - n_train - n_val, 0)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    return {
        "dataset_path": str(dataset_path),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": 1.0 - train_ratio - val_ratio,
        },
        "counts": {
            "all": n_total,
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "remainder_check": n_test,
        },
        "query_ids": {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    }


def filter_queries_by_split(queries: list[dict], manifest: dict, split: str) -> list[dict]:
    if split == "all":
        return list(queries)
    allowed = set(manifest["query_ids"][split])
    return [query for query in queries if str(query["query_id"]) in allowed]


def maybe_filter_verified_eval_split(
    queries: list[dict],
    *,
    eval_dataset_path: Path,
    verified_dataset_path: Path,
    manifest: dict | None,
    split: str,
) -> list[dict]:
    if manifest is None or split == "all":
        return list(queries)
    if eval_dataset_path.resolve() != verified_dataset_path.resolve():
        return list(queries)
    return filter_queries_by_split(queries, manifest, split)


def write_split_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
