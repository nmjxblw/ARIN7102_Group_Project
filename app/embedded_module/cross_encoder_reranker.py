"""
Cross-encoder reranker used in second-stage drug ranking.

The reranker scores (query, candidate_text) pairs and returns a scalar score
for each pair. This module is intentionally model-agnostic and supports any
sequence-classification checkpoint that can be loaded with Hugging Face
Transformers.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
try:
    import torch  # pyright: ignore[reportMissingImports]
    from transformers import (  # pyright: ignore[reportMissingImports]
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _convert_logits_to_scores(self, logits: torch.Tensor) -> np.ndarray:
        logits = logits.detach().cpu()
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits).squeeze(-1).numpy()
        probs = torch.softmax(logits, dim=-1)
        if probs.shape[-1] > 1:
            return probs[:, 1].numpy()
        return probs.squeeze(-1).numpy()

    def score_pairs(self, query: str, candidate_texts: Iterable[str]) -> np.ndarray:
        candidate_list = [str(text) if text is not None else "" for text in candidate_texts]
        if not candidate_list:
            return np.array([], dtype=np.float32)

        all_scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(candidate_list), self.batch_size):
                batch_candidates = candidate_list[start : start + self.batch_size]
                batch_queries = [query] * len(batch_candidates)
                encoded = self.tokenizer(
                    batch_queries,
                    batch_candidates,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                assert outputs.logits is not None, "Cross-encoder output logits is None"
                batch_scores = self._convert_logits_to_scores(outputs.logits)
                all_scores.append(batch_scores.astype(np.float32))
        return np.concatenate(all_scores, axis=0)
