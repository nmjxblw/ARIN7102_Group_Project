"""
Drug-drug interaction (DDI) lookup helper.

Loads `match_data_preprocessing/data/db_drug_interactions.csv` (~191k rows) once
and provides case-insensitive pair lookup for use as RAG context in LLM prompts.

The interactions CSV uses generic/brand drug names, while the enhanced drug
table has both `drug_name` (product) and `generic_name` (ingredient). Lookups
therefore try several name forms per drug.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path

import pandas as pd


def _default_csv_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "match_data_preprocessing"
        / "data"
        / "db_drug_interactions.csv"
    )


class DDIIndex:
    """In-memory DDI lookup indexed by lowercased drug-pair tuple."""

    def __init__(self, csv_path: Path | None = None):
        path = Path(csv_path) if csv_path else _default_csv_path()
        if not path.exists():
            raise FileNotFoundError(f"DDI csv not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        # Store both directions so order doesn't matter at lookup.
        self._pair_to_desc: dict[tuple[str, str], list[str]] = {}
        for a, b, desc in zip(df["Drug 1"], df["Drug 2"], df["Interaction Description"]):
            if not isinstance(a, str) or not isinstance(b, str):
                continue
            key = _canonical_pair(a, b)
            self._pair_to_desc.setdefault(key, []).append(str(desc).strip())
        self._known_names: set[str] = {
            name.strip().lower()
            for col in ("Drug 1", "Drug 2")
            for name in df[col].dropna().astype(str)
        }

    def has_drug(self, name: str) -> bool:
        return bool(name) and name.strip().lower() in self._known_names

    def lookup_pair(self, drug_a: str, drug_b: str) -> list[str]:
        if not drug_a or not drug_b:
            return []
        key = _canonical_pair(drug_a, drug_b)
        return list(self._pair_to_desc.get(key, []))

    def lookup_many(
        self,
        drugs: list[str],
        max_descriptions_per_pair: int = 2,
    ) -> list[tuple[str, str, str]]:
        """Return (drug_a, drug_b, description) rows for all pairs that hit."""
        seen: set[tuple[str, str]] = set()
        results: list[tuple[str, str, str]] = []
        for a, b in combinations(drugs, 2):
            key = _canonical_pair(a, b)
            if key in seen:
                continue
            seen.add(key)
            descs = self.lookup_pair(a, b)
            for desc in descs[:max_descriptions_per_pair]:
                results.append((a, b, desc))
        return results


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    a_low = a.strip().lower()
    b_low = b.strip().lower()
    return (a_low, b_low) if a_low <= b_low else (b_low, a_low)


@lru_cache(maxsize=1)
def get_default_index() -> DDIIndex:
    """Singleton accessor — loads the 191k-row CSV once per process."""
    return DDIIndex()


def format_ddi_context(
    drugs: list[str],
    index: DDIIndex | None = None,
    max_rows: int = 20,
) -> str:
    """Produce a compact text block to append to an LLM prompt."""
    if len(drugs) < 2:
        return "No drug pairs to check (single drug)."
    idx = index or get_default_index()
    rows = idx.lookup_many(drugs)
    if not rows:
        return "No entries found in DDI database for these drug pairs."
    lines = [f"- {a} + {b}: {desc}" for a, b, desc in rows[:max_rows]]
    suffix = ""
    if len(rows) > max_rows:
        suffix = f"\n(...{len(rows) - max_rows} more pair-entries omitted)"
    return "Known drug-drug interactions (from DrugBank-derived CSV):\n" + "\n".join(lines) + suffix
