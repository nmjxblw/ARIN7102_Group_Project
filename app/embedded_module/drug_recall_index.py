"""Local indexes for the experimental drug recall pipeline."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .label_adapter import normalize_disease_label, normalize_symptom_label
from .recommendation_pipeline import clean_text, parse_list_like, prepare_drug_dataframe


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _norm_key(value: object) -> str:
    text = clean_text(value).lower()
    return text


def _split_text_values(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = parse_list_like(text)
    if len(parsed) != 1 or parsed[0] != text:
        return parsed
    return [part.strip() for part in re.split(r"[,;/|]", text) if part.strip()]


def _parse_related_drugs(value: object) -> list[str]:
    if pd.isna(value):
        return []
    names = []
    for part in str(value).split("|"):
        part = part.strip()
        if not part:
            continue
        name = part.split(":", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _normalize_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    if len(numeric) == 0:
        return numeric
    lo = float(numeric.min())
    hi = float(numeric.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(numeric), dtype=np.float32), index=numeric.index)
    return (numeric - lo) / (hi - lo)


@dataclass
class EmbeddingManifest:
    path: str | None = None
    available: bool = False
    row_count: int = 0
    ndim: int = 0
    dim: int = 0
    dtype: str = ""
    row_aligned: bool = False
    projected_view: int | None = None
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BM25Index:
    postings: dict[str, dict[int, int]]
    doc_lengths: np.ndarray
    avg_doc_length: float
    doc_count: int
    k1: float = 1.5
    b: float = 0.75

    def idf(self, token: str) -> float:
        doc_freq = len(self.postings.get(token, {}))
        if doc_freq == 0:
            return 0.0
        return math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def score(self, query: str, top_k: int | None = None) -> pd.Series:
        tokens = tokenize(query)
        if not tokens:
            return pd.Series(dtype=float)

        scores: dict[int, float] = defaultdict(float)
        query_terms = Counter(tokens)
        for token, query_tf in query_terms.items():
            posting = self.postings.get(token)
            if not posting:
                continue
            idf = self.idf(token)
            for row_id, tf in posting.items():
                length = self.doc_lengths[row_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * length / self.avg_doc_length)
                scores[row_id] += idf * (tf * (self.k1 + 1.0) / denom) * query_tf

        result = pd.Series(scores, dtype=float)
        if len(result) == 0:
            return result
        result = result.sort_values(ascending=False)
        if top_k is not None:
            result = result.head(top_k)
        return result


class DrugRecallIndex:
    """In-memory inverted indexes and reusable statistics over the drug table."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        embedding_path: str | Path | None = None,
        embeddings: np.ndarray | None = None,
    ):
        self.df = prepare_drug_dataframe(df)
        self.df = self.df.reset_index(drop=True)
        self.row_count = len(self.df)

        self.disease_to_rows: dict[str, set[int]] = defaultdict(set)
        self.symptom_to_rows: dict[str, set[int]] = defaultdict(set)
        self.generic_to_rows: dict[str, set[int]] = defaultdict(set)
        self.class_to_rows: dict[str, set[int]] = defaultdict(set)
        self.related_to_rows: dict[str, set[int]] = defaultdict(set)

        self.row_generics: list[set[str]] = []
        self.row_classes: list[set[str]] = []
        self.row_related: list[set[str]] = []

        self.disease_idf: dict[str, float] = {}
        self.symptom_idf: dict[str, float] = {}
        self.quality_prior = self._build_quality_prior()
        self.bm25 = self._build_indexes()
        self.embeddings, self.embedding_manifest = self._load_embeddings(
            embedding_path=embedding_path,
            embeddings=embeddings,
        )

    def _build_quality_prior(self) -> pd.Series:
        rating = _normalize_series(self.df.get("avg_rating", pd.Series(np.zeros(self.row_count))))
        reviews = _normalize_series(np.log1p(pd.to_numeric(self.df.get("total_reviews", 0), errors="coerce").fillna(0.0)))
        return (0.65 * rating + 0.35 * reviews).astype(float)

    def _build_indexes(self) -> BM25Index:
        disease_doc_freq: Counter[str] = Counter()
        symptom_doc_freq: Counter[str] = Counter()
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        doc_lengths = np.zeros(self.row_count, dtype=np.float32)

        for row_id, row in self.df.iterrows():
            diseases = {
                normalize_disease_label(item)
                for item in row.get("disease_key_list", [])
                if normalize_disease_label(item)
            }
            symptoms = {
                normalize_symptom_label(item)
                for item in row.get("symptom_list_normalized", [])
                if normalize_symptom_label(item)
            }
            generics = {_norm_key(item) for item in _split_text_values(row.get("generic_name")) if _norm_key(item)}
            classes = {_norm_key(item) for item in _split_text_values(row.get("drug_classes")) if _norm_key(item)}
            related = {_norm_key(item) for item in _parse_related_drugs(row.get("related_drugs")) if _norm_key(item)}

            self.row_generics.append(generics)
            self.row_classes.append(classes)
            self.row_related.append(related)

            for disease in diseases:
                self.disease_to_rows[disease].add(row_id)
            for symptom in symptoms:
                self.symptom_to_rows[symptom].add(row_id)
            for generic in generics:
                self.generic_to_rows[generic].add(row_id)
            for drug_class in classes:
                self.class_to_rows[drug_class].add(row_id)
            for related_name in related:
                self.related_to_rows[related_name].add(row_id)

            disease_doc_freq.update(diseases)
            symptom_doc_freq.update(symptoms)

            text = self._row_text(row)
            counts = Counter(tokenize(text))
            doc_lengths[row_id] = sum(counts.values()) or 1
            for token, tf in counts.items():
                postings[token][row_id] = tf

        self.disease_idf = self._build_idf(disease_doc_freq)
        self.symptom_idf = self._build_idf(symptom_doc_freq)
        avg_length = float(doc_lengths.mean()) if len(doc_lengths) else 1.0
        return BM25Index(
            postings=dict(postings),
            doc_lengths=doc_lengths,
            avg_doc_length=max(avg_length, 1.0),
            doc_count=self.row_count,
        )

    def _build_idf(self, doc_freq: Counter[str]) -> dict[str, float]:
        return {
            key: math.log((self.row_count + 1.0) / (freq + 1.0)) + 1.0
            for key, freq in doc_freq.items()
        }

    def _row_text(self, row: pd.Series) -> str:
        fields = [
            row.get("drug_name", ""),
            row.get("generic_name", ""),
            row.get("drug_classes", ""),
            row.get("brand_names", ""),
            row.get("related_drugs", ""),
            " ".join(row.get("disease_key_list", [])),
            " ".join(row.get("symptom_list_normalized", [])),
            row.get("disease_description", ""),
            row.get("medical_condition_description", ""),
            row.get("semantic_text", ""),
            row.get("semantic_text_mcd", ""),
            row.get("semantic_text_dd", ""),
        ]
        return " ".join(clean_text(value) for value in fields if clean_text(value))

    def _load_embeddings(
        self,
        *,
        embedding_path: str | Path | None,
        embeddings: np.ndarray | None,
    ) -> tuple[np.ndarray | None, EmbeddingManifest]:
        manifest = EmbeddingManifest(path=str(embedding_path) if embedding_path else None)
        if embeddings is None and embedding_path is not None:
            path = Path(embedding_path)
            if path.exists():
                embeddings = np.load(path)
            else:
                manifest.message = f"Embedding file not found: {path}"
                return None, manifest

        if embeddings is None:
            manifest.message = "No embedding asset supplied"
            return None, manifest

        arr = np.asarray(embeddings, dtype=np.float32)
        manifest.available = True
        manifest.row_count = int(arr.shape[0]) if arr.ndim >= 1 else 0
        manifest.ndim = int(arr.ndim)
        manifest.dtype = str(arr.dtype)

        if arr.ndim == 3:
            manifest.projected_view = 1
            arr = arr[:, 1, :]
        if arr.ndim != 2:
            manifest.message = f"Unsupported embedding shape: {embeddings.shape}"
            return None, manifest

        manifest.dim = int(arr.shape[1])
        manifest.row_aligned = arr.shape[0] == self.row_count
        if not manifest.row_aligned:
            manifest.message = f"Embedding rows {arr.shape[0]} do not match table rows {self.row_count}"
            return None, manifest

        manifest.message = "ok"
        return arr, manifest

    def rows_for_diseases(self, diseases: Iterable[str]) -> set[int]:
        rows: set[int] = set()
        for disease in diseases:
            rows.update(self.disease_to_rows.get(disease, set()))
        return rows

    def rows_for_symptoms(self, symptoms: Iterable[str]) -> set[int]:
        rows: set[int] = set()
        for symptom in symptoms:
            rows.update(self.symptom_to_rows.get(symptom, set()))
        return rows

    def top_quality_rows_for_disease(self, disease: str, limit: int) -> list[int]:
        rows = list(self.disease_to_rows.get(disease, set()))
        if not rows:
            return []
        return (
            self.quality_prior.loc[rows]
            .sort_values(ascending=False)
            .head(limit)
            .index.astype(int)
            .tolist()
        )
