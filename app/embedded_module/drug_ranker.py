"""Optional local learning-to-rank wrapper for experimental recall."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "label_idf_score",
    "disease_idf_score",
    "symptom_idf_score",
    "symptom_coverage",
    "disease_specificity",
    "bm25_score",
    "dense_score",
    "quality_prior",
    "disease_conf_overlap",
    "symptom_conf_overlap",
    "stage_disease",
    "stage_strict",
    "stage_symptom",
    "stage_bm25",
    "stage_dense",
    "stage_prior",
    "others_penalty",
]


@dataclass
class LocalDrugRanker:
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))
    model: object | None = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def fit(self, features: pd.DataFrame, labels: np.ndarray) -> "LocalDrugRanker":
        from sklearn.ensemble import HistGradientBoostingClassifier

        x = self._frame_to_matrix(features)
        y = np.asarray(labels, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("Ranker training requires both positive and negative samples")
        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=42,
        )
        model.fit(x, y)
        self.model = model
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Ranker model has not been loaded or trained")
        x = self._frame_to_matrix(features)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)
        return np.asarray(self.model.predict(x), dtype=float)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise ValueError("Cannot save an empty ranker")
        payload = {"feature_columns": self.feature_columns, "model": self.model}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "LocalDrugRanker":
        payload = joblib.load(path)
        return cls(
            feature_columns=list(payload["feature_columns"]),
            model=payload["model"],
        )

    def _frame_to_matrix(self, features: pd.DataFrame) -> np.ndarray:
        frame = features.copy()
        for column in self.feature_columns:
            if column not in frame.columns:
                frame[column] = 0.0
        return (
            frame[self.feature_columns]
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )
