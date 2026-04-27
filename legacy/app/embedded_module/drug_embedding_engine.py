"""
drug_embedding_engine.py
使用语义编码模型对药物的语义文本进行 Embedding,
供后续 Faiss KNN 检索与 CrossEncoder 重排使用.

支持模型:
  - pritamdeka/S-PubMedBert-MS-MARCO (推荐, 医学检索专用)
  - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract (旧模型)
  - sentence-transformers/all-MiniLM-L6-v2 (轻量备选)

支持 pooling 策略: cls / mean
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch  # pyright: ignore[reportMissingImports]
from transformers import AutoModel, AutoTokenizer  # pyright: ignore[reportMissingImports]
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]

# Import pipeline config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import cfg

DEFAULT_MODEL_NAME = cfg.medbert_model_name
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = cfg.embedding_max_length
DEFAULT_POOLING = cfg.embedding_pooling
DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class DrugEmbeddingEngine:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        pooling: str = DEFAULT_POOLING,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling.strip().lower()  # 'cls' or 'mean'
        self.local_files_only = local_files_only

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[DrugEmbeddingEngine] 设备: {self.device}")
        print(f"[DrugEmbeddingEngine] 正在加载模型: {self.model_name} ...")
        print(f"[DrugEmbeddingEngine] max_length={self.max_length}, pooling={self.pooling}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        ).to(self.device)
        self.model.eval()
        print("[DrugEmbeddingEngine] 模型加载完成")

    def _pool(self, outputs, attention_mask) -> np.ndarray:
        """Extract sentence-level embeddings from model outputs."""
        if self.pooling == "mean":
            # Mean pooling: average all token embeddings weighted by attention mask
            token_embeddings = outputs.last_hidden_state  # (B, seq_len, dim)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return (sum_embeddings / sum_mask).cpu().numpy()
        else:
            # CLS pooling: take the [CLS] token vector
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def encode(self, texts: list[str]) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding drugs"):
                batch = texts[i : i + self.batch_size]
                batch = [
                    str(t).strip() if pd.notna(t) and str(t).strip() else "[MASK]"
                    for t in batch
                ]

                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                outputs = self.model(**encoded)
                pooled = self._pool(outputs, encoded["attention_mask"])
                all_embeddings.append(pooled)

        embeddings = np.vstack(all_embeddings)
        print(f"[DrugEmbeddingEngine] 编码完成: shape = {embeddings.shape}")
        return embeddings

    def encode_query(self, query_text: str) -> np.ndarray:
        return self.encode([query_text])

    @staticmethod
    def save(embeddings: np.ndarray, filename: str, output_dir: str = DEFAULT_OUTPUT_DIR):
        filename_path = Path(filename)
        if filename_path.is_absolute():
            path = str(filename_path)
        else:
            path = os.path.join(output_dir, filename)
        np.save(path, embeddings)
        print(f"[DrugEmbeddingEngine] 已保存: {path}  (shape={embeddings.shape})")
        return path

    @staticmethod
    def load(filename: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> np.ndarray:
        filename_path = Path(filename)
        if filename_path.is_absolute():
            path = str(filename_path)
        else:
            path = os.path.join(output_dir, filename)
        embeddings = np.load(path)
        print(f"[DrugEmbeddingEngine] 已加载: {path}  (shape={embeddings.shape})")
        return embeddings
