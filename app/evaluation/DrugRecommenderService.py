import logging
from pathlib import Path
from typing import Any
import pandas as pd


from singleton_module import SingletonMeta


from embedded_module.drug_recall_index import DrugRecallIndex
from embedded_module.phase2_final_recommender import Phase2FinalRecommender

logger = logging.getLogger(__name__)


class DrugRecommendationService(metaclass=SingletonMeta):
    def __init__(self):
        """
        初始化方法只会在第一次实例化时运行一次。
        在这里完成所有耗时的加载操作。
        """
        logger.info("Initializing DrugRecommendationService (Singleton)...")


        self.repo_root = Path(__file__).resolve().parents[2]
        self.table_path = self.repo_root / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1_structured.csv"
        self.half1_path = self.repo_root / "app" / "dataset_module" / "drugs_training_dataset" / "drug_data_half_1.json"
        self.half2_path = self.repo_root / "app" / "dataset_module" / "drugs_training_dataset" / "drug_data_half_2.json"

        self.default_top_k_recall = 20
        self.default_top_k_per_disease = 3

        # 2. 加载核心组件
        logger.info(f"Loading drug table from {self.table_path}...")
        self.df = pd.read_csv(self.table_path)

        logger.info("Building DrugRecallIndex...")
        self.index = DrugRecallIndex(df=self.df, embedding_path=None)

        logger.info("Setting up Phase2FinalRecommender...")
        self.recommender = Phase2FinalRecommender(
            index=self.index,
            half_data_paths=[self.half1_path, self.half2_path],
            table_path=self.table_path,
        )
        logger.info("DrugRecommendationService initialized successfully.")


    def _confidence(self, value: Any, default: float = 1.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, parsed))

    def _normalize_label_items(self, items: Any) -> list[dict]:
        normalized = []
        for item in items or []:
            label, confidence = None, 1.0
            if isinstance(item, str):
                label = item.strip()
            elif isinstance(item, dict):
                if "label" in item:
                    label = str(item.get("label") or "").strip()
                    confidence = self._confidence(item.get("confidence"))
                elif "name" in item:
                    label = str(item.get("name") or "").strip()
                    confidence = self._confidence(item.get("confidence"))
                elif len(item) == 1:
                    label = str(next(iter(item.keys())) or "").strip()
                    confidence = self._confidence(next(iter(item.values())))

            if label:
                normalized.append({"label": label, "confidence": confidence})
        return normalized

    def _normalize_bert_output(self, bert_output: dict) -> dict:
        if not isinstance(bert_output, dict):
            raise TypeError("bert_output must be a dict")
        return {
            "query_index": 0,
            "diseases": self._normalize_label_items(bert_output.get("diseases", [])),
            "symptoms": self._normalize_label_items(bert_output.get("symptoms", [])),
            "need_first_aid": bert_output.get("need_first_aid", 0),
            "sentence": str(bert_output.get("sentence") or ""),
        }

    def _build_flat_recommendations(self, result: dict) -> list[dict]:
        flat_results = []
        global_rank = 1
        for d_res in result.get("disease_results", []):
            for top_drug in d_res.get("final_top3", []):
                flat_item = {
                    #"query_index": result.get("query_index", 0),
                    "disease": d_res.get("disease", ""),
                    #"disease_confidence": d_res.get("disease_confidence", 1.0),
                    "drug_name": top_drug.get("drug_name", ""),
                    #"disease_rank": top_drug.get("disease_rank", global_rank),
                    #"global_display_rank": global_rank,
                    #"phase2_rank": top_drug.get("phase2_rank"),
                    #"selection_source": top_drug.get("selection_source", ""),
                    "final_confidence": top_drug.get("half_disease_confidence", 0.0)
                }
                # if "phase2_score" in top_drug:
                #     flat_item["phase2_score"] = top_drug["phase2_score"]
                flat_results.append(flat_item)
                global_rank += 1
        return flat_results



    def predict(self, bert_output: dict,flat_out=False) -> dict:
        """外部调用主入口"""
        query = self._normalize_bert_output(bert_output)


        result = self.recommender.recommend_query(
            query=query,
            top_k_recall=self.default_top_k_recall,
            top_k_per_disease=self.default_top_k_per_disease,
        )
        if flat_out == False:
            return {
                "input": query,
                "disease_results": result.get("disease_results", []),
                "recommendations": self._build_flat_recommendations(result),
            }
        else:
            return {"recommendations":self._build_flat_recommendations(result)}

