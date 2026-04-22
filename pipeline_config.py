"""
Pipeline Configuration - 集中配置读取模块

从 pipeline_config.env 加载所有推荐管道参数。
其他模块通过 `from pipeline_config import cfg` 获取配置。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 pipeline_config.env（项目根目录）
_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _PROJECT_ROOT / "pipeline_config.env"
if _CONFIG_PATH.exists():
    load_dotenv(_CONFIG_PATH, override=False)

# 加载根目录 .env（提供 OPENAI_API_KEY 等回退配置）
_ROOT_ENV = _PROJECT_ROOT / ".env"
if _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV, override=False)


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


class PipelineConfig:
    """推荐管道参数配置（从环境变量读取，支持 pipeline_config.env）"""

    # ── 召回阶段 ──────────────────────────────────────────
    @property
    def recall_top_k_each(self) -> int:
        """每个召回通道保留的候选数"""
        return _int("RECALL_TOP_K_EACH", 200)

    @property
    def fused_top_k(self) -> int:
        """融合后保留的候选数（送入精排）"""
        return _int("FUSED_TOP_K", 200)

    @property
    def recall_weight_semantic(self) -> float:
        """语义召回在融合中的权重"""
        return _float("RECALL_WEIGHT_SEMANTIC", 0.5)

    @property
    def recall_weight_label(self) -> float:
        """标签召回在融合中的权重"""
        return _float("RECALL_WEIGHT_LABEL", 0.5)

    # ── 标签召回内部 ──────────────────────────────────────
    @property
    def label_disease_weight(self) -> float:
        """标签召回中疾病匹配的权重"""
        return _float("LABEL_DISEASE_WEIGHT", 0.55)

    @property
    def label_symptom_weight(self) -> float:
        """标签召回中症状匹配的权重"""
        return _float("LABEL_SYMPTOM_WEIGHT", 0.45)

    # ── 精排最终得分 ──────────────────────────────────────
    @property
    def final_weight_recall(self) -> float:
        """最终得分中召回融合分的权重"""
        return _float("FINAL_WEIGHT_RECALL", 0.35)

    @property
    def final_weight_cross_encoder(self) -> float:
        """最终得分中 CrossEncoder 分的权重"""
        return _float("FINAL_WEIGHT_CROSS_ENCODER", 0.50)

    @property
    def final_weight_business(self) -> float:
        """最终得分中业务因子的权重"""
        return _float("FINAL_WEIGHT_BUSINESS", 0.15)

    # ── 业务评分内部 ──────────────────────────────────────
    @property
    def biz_weight_rating(self) -> float:
        """业务评分中平均评分的权重"""
        return _float("BIZ_WEIGHT_RATING", 0.55)

    @property
    def biz_weight_reviews(self) -> float:
        """业务评分中评论数的权重"""
        return _float("BIZ_WEIGHT_REVIEWS", 0.30)

    @property
    def biz_weight_price(self) -> float:
        """业务评分中价格的权重"""
        return _float("BIZ_WEIGHT_PRICE", 0.15)

    # ── 输出 ─────────────────────────────────────────────
    @property
    def top_k(self) -> int:
        """默认返回的药物数量"""
        return _int("TOP_K", 10)

    # ── 模型配置 ──────────────────────────────────────────
    @property
    def medbert_model_name(self) -> str:
        """语义编码模型名称"""
        return os.getenv("MEDBERT_MODEL_NAME", "pritamdeka/S-PubMedBert-MS-MARCO")

    @property
    def embedding_max_length(self) -> int:
        """编码最大 token 长度"""
        return _int("EMBEDDING_MAX_LENGTH", 512)

    @property
    def embedding_pooling(self) -> str:
        """向量池化策略: 'cls' 或 'mean'"""
        return os.getenv("EMBEDDING_POOLING", "mean").strip().lower()

    # ── LLM 术语扩展 ──────────────────────────────────────────
    @property
    def enable_llm_query_expansion(self) -> bool:
        """是否启用 LLM 医学术语扩展"""
        return os.getenv("ENABLE_LLM_QUERY_EXPANSION", "false").strip().lower() == "true"

    @property
    def llm_api_key(self) -> str:
        """LLM API Key（优先专用配置，回退到 OPENAI_API_KEY）"""
        return os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    @property
    def llm_base_url(self) -> str:
        """LLM Base URL（优先专用配置，回退到 OPENAI_BASE_URL）"""
        return os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def llm_model(self) -> str:
        """LLM 模型名称（优先专用配置，回退到 OPENAI_MODEL）"""
        return os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    def summary(self) -> dict:
        """返回所有配置的字典形式（用于日志/调试）"""
        return {
            "recall_top_k_each": self.recall_top_k_each,
            "fused_top_k": self.fused_top_k,
            "recall_weight_semantic": self.recall_weight_semantic,
            "recall_weight_label": self.recall_weight_label,
            "label_disease_weight": self.label_disease_weight,
            "label_symptom_weight": self.label_symptom_weight,
            "final_weight_recall": self.final_weight_recall,
            "final_weight_cross_encoder": self.final_weight_cross_encoder,
            "final_weight_business": self.final_weight_business,
            "biz_weight_rating": self.biz_weight_rating,
            "biz_weight_reviews": self.biz_weight_reviews,
            "biz_weight_price": self.biz_weight_price,
            "top_k": self.top_k,
            "medbert_model_name": self.medbert_model_name,
            "embedding_max_length": self.embedding_max_length,
            "embedding_pooling": self.embedding_pooling,
            "enable_llm_query_expansion": self.enable_llm_query_expansion,
            "llm_model": self.llm_model,
        }


# 全局单例
cfg = PipelineConfig()
