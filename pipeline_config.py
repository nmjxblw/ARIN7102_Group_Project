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
        }


# 全局单例
cfg = PipelineConfig()
