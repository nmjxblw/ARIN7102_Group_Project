"""Optional local learning-to-rank wrapper for experimental recall."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

VERIFIED_LEARNED_FEATURE_COLUMNS = [
    "label_idf_score",
    "disease_idf_score",
    "symptom_idf_score",
    "symptom_coverage",
    "disease_specificity",
    "quality_prior",
    "disease_conf_overlap",
    "symptom_conf_overlap",
    "stage_disease",
    "stage_strict",
    "stage_symptom",
    "others_penalty",
]


@dataclass
class LocalDrugRanker:
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))
    model: object | None = None
    model_type: str = "gbdt"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        *,
        model_type: str = "gbdt",
        metadata: dict[str, Any] | None = None,
        group: list[int] | np.ndarray | None = None,
    ) -> "LocalDrugRanker":
        x = self._frame_to_matrix(features)
        y = np.asarray(labels)
        if len(np.unique(y)) < 2:
            raise ValueError("Ranker training requires both positive and negative samples")

        if model_type == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=42,
                            solver="liblinear",
                            C=0.1,
                        ),
                    ),
                ]
            )
        elif model_type == "gbdt":
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=42,
            )
        elif model_type == "xgb_ranker":
            try:
                from xgboost import XGBRanker
            except ImportError as exc:
                raise ImportError(
                    "xgboost is required for model_type='xgb_ranker'. "
                    "Install it in the active environment before training."
                ) from exc

            if group is None:
                raise ValueError("XGBRanker training requires non-empty group sizes")
            group_sizes = np.asarray(group, dtype=int)
            if len(group_sizes) == 0 or group_sizes.sum() != len(y):
                raise ValueError("XGBRanker group sizes must sum to the number of training rows")
            if np.any(group_sizes <= 0):
                raise ValueError("XGBRanker group sizes must all be positive")

            model = XGBRanker(
                objective="rank:ndcg",
                learning_rate=0.05,
                n_estimators=200,
                max_depth=6,
                min_child_weight=1.0,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                tree_method="hist",
                ndcg_exp_gain=False,
            )
        else:
            raise ValueError(f"Unsupported ranker model_type: {model_type}")
        if model_type == "logreg":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(x, y)
        elif model_type == "xgb_ranker":
            model.fit(x, y, group=group_sizes, verbose=False)
        else:
            model.fit(x, y)
        self.model = model
        self.model_type = model_type
        self.metadata = dict(metadata or {})
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Ranker model has not been loaded or trained")
        x = self._frame_to_matrix(features)
        if self.model_type == "xgb_ranker":
            return np.asarray(self.model.predict(x), dtype=float)
        if hasattr(self.model, "predict_proba"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)
        return np.asarray(self.model.predict(x), dtype=float)

    def save(self, path: str | Path, *, weights_path: str | Path | None = None) -> None:
        if self.model is None:
            raise ValueError("Cannot save an empty ranker")
        payload = {
            "feature_columns": self.feature_columns,
            "model": self.model,
            "model_type": self.model_type,
            "metadata": self.metadata,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)
        if weights_path is not None:
            self.export_explanation(weights_path)

    @classmethod
    def load(cls, path: str | Path) -> "LocalDrugRanker":
        payload = joblib.load(path)
        return cls(
            feature_columns=list(payload["feature_columns"]),
            model=payload["model"],
            model_type=str(payload.get("model_type") or "gbdt"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def export_explanation(self, path: str | Path) -> None:
        explanation = self.explain()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(explanation, f, indent=2)

    def explain(self) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("Ranker model has not been loaded or trained")

        explanation: dict[str, Any] = {
            "model_type": self.model_type,
            "feature_columns": list(self.feature_columns),
            "metadata": dict(self.metadata),
        }

        if self.model_type == "logreg" and hasattr(self.model, "named_steps"):
            classifier = self.model.named_steps["classifier"]
            coef = np.asarray(classifier.coef_[0], dtype=float)
            intercept = float(classifier.intercept_[0])
            ranked = sorted(
                (
                    {
                        "feature": feature,
                        "coefficient": float(weight),
                        "abs_coefficient": float(abs(weight)),
                    }
                    for feature, weight in zip(self.feature_columns, coef, strict=False)
                ),
                key=lambda item: item["abs_coefficient"],
                reverse=True,
            )
            explanation.update(
                {
                    "intercept": intercept,
                    "coefficients": {feature: float(weight) for feature, weight in zip(self.feature_columns, coef, strict=False)},
                    "top_features": ranked,
                }
            )
            return explanation

        if self.model_type == "xgb_ranker" and hasattr(self.model, "feature_importances_"):
            importances = np.asarray(self.model.feature_importances_, dtype=float)
            ranked = sorted(
                (
                    {
                        "feature": feature,
                        "importance": float(weight),
                    }
                    for feature, weight in zip(self.feature_columns, importances, strict=False)
                ),
                key=lambda item: item["importance"],
                reverse=True,
            )
            explanation.update(
                {
                    "feature_importances": {
                        feature: float(weight)
                        for feature, weight in zip(self.feature_columns, importances, strict=False)
                    },
                    "top_features": ranked,
                }
            )
            return explanation

        explanation["note"] = "Model does not expose linear coefficients for direct weight export."
        return explanation

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
