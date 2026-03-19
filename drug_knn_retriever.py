"""
drug_knn_retriever.py  —— Member C 核心模块 (Phase 2)
基于 Faiss 的 KNN 药物召回引擎。

核心传统 ML 算法: K-Nearest Neighbors (KNN)
= 老师重点考察的无监督 / 基于距离度量的经典机器学习方法。

用法:
    from drug_knn_retriever import DrugKNNRetriever
    retriever = DrugKNNRetriever(drug_embeddings, df)
    results = retriever.query("fever headache cough", engine, top_k=10)
"""

import numpy as np
import pandas as pd

try:
    import faiss  # pyright: ignore[reportMissingImports]
except ImportError:
    raise ImportError(
        "请先安装 faiss: pip install faiss-cpu  (或 faiss-gpu 如果有 CUDA)"
    )


class DrugKNNRetriever:
    """
    药物 KNN 召回引擎。

    底层算法: Faiss IndexFlatIP (内积索引)
    归一化后的内积 = 余弦相似度 (Cosine Similarity)
    这就是传统机器学习中最经典的 KNN 检索。
    """

    def __init__(self, drug_embeddings: np.ndarray, drug_df: pd.DataFrame):
        """
        Args:
            drug_embeddings: shape = (n_drugs, 768), 从 Block 7 生成的特征矩阵
            drug_df: 原始药物 DataFrame (enhanced_drug_table)
        """
        self.drug_df = drug_df.copy().reset_index(drop=True)
        self.n_drugs, self.dim = drug_embeddings.shape

        # L2 归一化 → 内积就变成了余弦相似度
        self.embeddings = drug_embeddings.copy().astype(np.float32)
        faiss.normalize_L2(self.embeddings)

        # 构建 Faiss 索引
        self.index = faiss.IndexFlatIP(self.dim)  # 内积 = 余弦相似度
        self.index.add(self.embeddings)

        print(f"[DrugKNNRetriever] Faiss 索引构建完成 ✅")
        print(f"  药物总数: {self.n_drugs}")
        print(f"  向量维度: {self.dim}")
        print(f"  索引类型: IndexFlatIP (余弦相似度)")

    # ----------------------------------------------------------
    # 核心检索方法
    # ----------------------------------------------------------
    def retrieve(self, query_vector: np.ndarray, top_k: int = 20) -> pd.DataFrame:
        """
        用一个已编码好的查询向量去 Faiss 库里做 KNN 检索。

        Args:
            query_vector: shape = (1, 768), 已通过 engine.encode_query() 编码
            top_k: 返回最相似的几款药

        Returns:
            DataFrame: 候选药物列表, 附带 similarity_score 列
        """
        # 归一化查询向量
        q = query_vector.copy().astype(np.float32)
        faiss.normalize_L2(q)

        # KNN 搜索
        distances, indices = self.index.search(q, top_k)

        # 组装结果
        result = self.drug_df.iloc[indices[0]].copy()
        result["similarity_score"] = distances[0]
        result = result.sort_values("similarity_score", ascending=False)

        return result.reset_index(drop=True)

    # ----------------------------------------------------------
    # 端到端查询 (文本 → 向量 → KNN → 召回)
    # ----------------------------------------------------------
    def query(self, query_text: str, embedding_engine, top_k: int = 20) -> pd.DataFrame:
        """
        端到端查询接口: 直接输入症状/疾病文本, 返回推荐药物。

        Args:
            query_text: 如 "symptoms: fever, headache, cough; disease: common cold"
            embedding_engine: DrugEmbeddingEngine 实例 (用于编码查询文本)
            top_k: 返回最相似的几款药

        Returns:
            DataFrame: 候选药物列表
        """
        query_vec = embedding_engine.encode_query(query_text)
        return self.retrieve(query_vec, top_k)

    # ----------------------------------------------------------
    # 批量查询 (多个疾病 + 多个症状 → 自动组装文本)
    # ----------------------------------------------------------
    def query_from_labels(
        self,
        disease_labels: list[str],
        symptom_terms: list[str],
        embedding_engine,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """
        接收小队一传过来的结构化病症标签, 自动拼装成查询文本并检索。

        Args:
            disease_labels: 如 ["common_cold", "bronchial_asthma"]
            symptom_terms: 如 ["fever", "headache", "cough"]
            embedding_engine: DrugEmbeddingEngine 实例
            top_k: 返回最相似的几款药

        Returns:
            DataFrame: 候选药物列表, 含 similarity_score
        """
        # 组装查询文本 (和 Block 5 的 semantic_text 结构对齐)
        parts = []
        if disease_labels:
            parts.append("Related diseases: " + ", ".join(disease_labels) + ".")
        if symptom_terms:
            parts.append("Target symptoms: " + ", ".join(symptom_terms) + ".")
        query_text = " ".join(parts) if parts else "[MASK]"

        print(f"[KNN Query] {query_text}")
        return self.query(query_text, embedding_engine, top_k)
