"""
drug_knn_retriever.py
基于 Faiss 的 KNN 药物召回引擎。
"""

import numpy as np
import pandas as pd

try:
    import faiss  # pyright: ignore[reportMissingImports]
except ImportError:
    raise ImportError("请先安装 faiss: pip install faiss-cpu  (或 faiss-gpu 如果有 CUDA)")


class DrugKNNRetriever:
    def __init__(self, drug_embeddings: np.ndarray, drug_df: pd.DataFrame):
        self.drug_df = drug_df.copy().reset_index(drop=True)
        self.n_drugs, self.dim = drug_embeddings.shape

        self.embeddings = drug_embeddings.copy().astype(np.float32)
        faiss.normalize_L2(self.embeddings)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

        print("[DrugKNNRetriever] Faiss 索引构建完成")
        print(f"  药物总数: {self.n_drugs}")
        print(f"  向量维度: {self.dim}")
        print("  索引类型: IndexFlatIP (余弦相似度)")

    def retrieve(self, query_vector: np.ndarray, top_k: int = 20) -> pd.DataFrame:
        q = query_vector.copy().astype(np.float32)
        faiss.normalize_L2(q)

        distances, indices = self.index.search(q, top_k)
        result = self.drug_df.iloc[indices[0]].copy()
        result["similarity_score"] = distances[0]
        return result.sort_values("similarity_score", ascending=False).reset_index(drop=True)

    def query(self, query_text: str, embedding_engine, top_k: int = 20) -> pd.DataFrame:
        query_vec = embedding_engine.encode_query(query_text)
        return self.retrieve(query_vec, top_k)

    def query_from_labels(
        self,
        disease_labels: list[str],
        symptom_terms: list[str],
        embedding_engine,
        top_k: int = 20,
    ) -> pd.DataFrame:
        parts = []
        if disease_labels:
            parts.append("Diseases: " + ", ".join(disease_labels) + ".")
        if symptom_terms:
            parts.append("Symptoms: " + ", ".join(symptom_terms) + ".")
        query_text = " ".join(parts) if parts else "[MASK]"

        print(f"[KNN Query] {query_text}")
        return self.query(query_text, embedding_engine, top_k)
