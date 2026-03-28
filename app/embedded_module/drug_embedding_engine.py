"""
drug_embedding_engine.py
使用 PubMedBERT 对药物的"语义文本"(drug_name + symptoms + disease_desc)
进行 768 维 Embedding，供后续 Faiss KNN 检索与传统 ML 重排使用。
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch  # pyright: ignore[reportMissingImports]
from transformers import AutoModel, AutoTokenizer  # pyright: ignore[reportMissingImports]
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]


DEFAULT_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256
DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class DrugEmbeddingEngine:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print("[DrugEmbeddingEngine] 模型加载完成")

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
                cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_vectors)

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
