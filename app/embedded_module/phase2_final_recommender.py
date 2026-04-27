from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.embedded_module.drug_ranker import LocalDrugRanker
from app.embedded_module.experimental_recall_pipeline import ExperimentalDrugRecallPipeline
from app.embedded_module.drug_recall_index import DrugRecallIndex

class Phase2FinalRecommender:
    def __init__(
        self,
        index: DrugRecallIndex,
        half_data_paths: list[Path | str],
        table_path: Path | str,
        *,
        phase2_mode: str = "label_core_rerank",
        ranker_path: Path | str | None = None,
    ):
        self.index = index
        self.phase2_mode = phase2_mode
        self.ranker_path = Path(ranker_path) if ranker_path else None
        self.ranker = LocalDrugRanker.load(self.ranker_path) if self.ranker_path else None
        self.pipeline = ExperimentalDrugRecallPipeline(index=index, ranker=self.ranker)
        
        self.half_pool: dict[str, dict[str, float]] = defaultdict(dict)
        
        from app.evaluation.half_data_adapter import _load_table_name_map, _canonical_drug_name, _parse_labels
        
        table_name_map = _load_table_name_map(table_path)
        
        for path in half_data_paths:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for row in data:
                canonical_drug = _canonical_drug_name(row.get("drug_name", ""), table_name_map)
                if not canonical_drug:
                    continue
                diseases = _parse_labels(row.get("diseases", []))
                for disease in diseases:
                    d_name = disease["name"]
                    d_conf = disease["confidence"]
                    current_conf = self.half_pool[d_name].get(canonical_drug, 0.0)
                    self.half_pool[d_name][canonical_drug] = max(current_conf, d_conf)

    def recommend_query(self, query: dict, top_k_recall: int = 20, top_k_per_disease: int = 3) -> dict:
        query_index = query.get("query_index", 0)
        sentence = query.get("sentence", "")
        diseases = query.get("diseases", [])
        symptoms = query.get("symptoms", [])
        
        disease_results = []
        
        # To handle multi-disease properly, we iterate each disease
        for d in diseases:
            d_label = d.get("label") or d.get("name") or ""
            if not d_label:
                continue
                
            d_conf = d.get("confidence", 1.0)
            
            # The pipeline handles normalization, so passing the raw label dict is fine
            mapped_d = [{"label": d_label, "confidence": d_conf}]
            mapped_s = symptoms # pass all symptoms as they are shared
            
            df = self.pipeline.recommend(
                symptom_text=sentence,
                disease_items=mapped_d,
                symptom_items=mapped_s,
                top_k=top_k_recall,
                pool_size=1000, # Use a reasonable default size
                mode=self.phase2_mode,
                return_trace=False
            )
            
            phase2_candidates = []
            if not df.empty:
                for rank, (idx, row) in enumerate(df.iterrows(), start=1):
                    drug_name = str(row.get("drug_name", ""))
                    score = float(row.get("final_score", 0.0))
                    matched_diseases_str = str(row.get("matched_diseases", ""))
                    matched_symptoms_str = str(row.get("matched_symptoms", ""))
                    matched_diseases = matched_diseases_str.split(",") if matched_diseases_str else []
                    matched_symptoms = matched_symptoms_str.split(",") if matched_symptoms_str else []
                    
                    phase2_candidates.append({
                        "drug_name": drug_name,
                        "phase2_rank": rank,
                        "phase2_score": score,
                        "matched_diseases": matched_diseases,
                        "matched_symptoms": matched_symptoms
                    })
            
            # Normalization of disease label to access half_pool
            from app.embedded_module.label_adapter import normalize_disease_label
            normalized_d_label = normalize_disease_label(d_label)
            
            pool_for_disease = self.half_pool.get(normalized_d_label, {})
            
            # Count the actual number of half confirmed in the *entire* Top20 phase2 candidates
            half_confirmed_count = sum(1 for c in phase2_candidates if c["drug_name"] in pool_for_disease)
            
            final_top = []
            
            # Pass 1: half_confirmed
            for c in phase2_candidates:
                if c["drug_name"] in pool_for_disease:
                    new_c = dict(c)
                    new_c["selection_source"] = "half_confirmed"
                    new_c["half_disease_confidence"] = pool_for_disease[c["drug_name"]]
                    final_top.append(new_c)
                if len(final_top) == top_k_per_disease:
                    break
                    
            # Pass 2: fallback
            if len(final_top) < top_k_per_disease:
                for c in phase2_candidates:
                    if not any(fc["drug_name"] == c["drug_name"] for fc in final_top):
                        new_c = dict(c)
                        new_c["selection_source"] = "phase2_fallback"
                        new_c["half_disease_confidence"] = 0.0
                        final_top.append(new_c)
                    if len(final_top) == top_k_per_disease:
                        break
            
            for idx, item in enumerate(final_top, start=1):
                item["disease_rank"] = idx
                
            disease_results.append({
                "disease": d_label,
                "disease_confidence": d_conf,
                "phase2_top20_count": len(phase2_candidates),
                "half_confirmed_in_top20": half_confirmed_count,
                "final_top3": final_top
            })
            
        return {
            "query_index": query_index,
            "sentence": sentence,
            "shared_symptoms": symptoms,
            "disease_results": disease_results
        }
