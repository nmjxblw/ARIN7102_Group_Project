import os
import sys
import json
import re
from pathlib import Path
import pandas as pd


DRUG_RAW_DATASET_FILE: Path = (
    Path(__file__).parent / "drugs_training_dataset" / "enhanced_drug_table_v1.json"
)
DRUG_DISEASE_MAPPING_FILE: Path = (
    Path(__file__).parent / "drugs_training_dataset" / "drug_disease_mapping.json"
)
SENTENCES_DATASET_FILE: Path = (
    Path(__file__).parent / "bert_training_dataset" / "generated_medical_dataset.json"
)
DRUG_OUTPUT_FILE: Path = (
    Path(__file__).parent / "drugs_training_dataset" / "drug_training_dataset.json"
)


def merge_sentences_to_drug_dataset():
    """
    将生成的句子数据合并到药物数据集中

    Returns:
        None

    Example:
        >>> merge_sentences_to_drug_dataset()
        将生成的句子数据合并到药物数据集中，并保存到 drug_training_dataset.json 文件中
    """
    if not DRUG_RAW_DATASET_FILE.exists():
        raise FileNotFoundError(f"药物原始数据文件 {DRUG_RAW_DATASET_FILE} 不存在")
    if not SENTENCES_DATASET_FILE.exists():
        raise FileNotFoundError(f"生成的句子数据文件 {SENTENCES_DATASET_FILE} 不存在")

    drug_disease_dict: dict[str, dict[str, list[str]]] = {}
    with open(DRUG_RAW_DATASET_FILE, "r", encoding="utf-8") as f:
        drug_raw_data: list[dict] = json.load(f)
        for entry in drug_raw_data:
            diseases = entry.get("matched_disease_keys", ["others"])
            symptoms = entry.get("matched_symptoms", [])
            drug_names: list[str] = str(entry.get("drug_name", "")).split("/")
            drug_names = [drug_name.strip() for drug_name in drug_names]
            fit: bool = True
            for disease in diseases:
                if disease in ["others"] or len(symptoms) == 0:
                    fit = False
                    break
            if not fit:
                continue
            for drug_name in drug_names:
                if drug_name not in drug_disease_dict:
                    drug_disease_dict[drug_name] = {"diseases": [], "symptoms": []}
                disease_set: set = set(drug_disease_dict[drug_name]["diseases"])
                symptoms_set: set = set(drug_disease_dict[drug_name]["symptoms"])
                disease_set.update(diseases)
                symptoms_set.update(symptoms)
                drug_disease_dict[drug_name] = {
                    "diseases": list(disease_set),
                    "symptoms": list(symptoms_set),
                }
    print(f"已整合完成药物-疾病字典，包含 {len(drug_disease_dict)} 个药物")
    print(f"样本：{list(drug_disease_dict.items())[:5]}")
    # return
    with open(DRUG_DISEASE_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(drug_disease_dict, f, ensure_ascii=False, indent=4)
    # return
    with open(SENTENCES_DATASET_FILE, "r", encoding="utf-8") as f:
        sentences_data: list[dict] = json.load(f)
    print(f"已加载完成生成的句子数据，包含 {len(sentences_data)} 条记录")
    print(f"样本：{sentences_data[:2]}")
    merge_dataset = []
    for sentence_entry in sentences_data:
        sentence_entry["drugs"] = []
        diseases = sentence_entry.get("disease", ["others"])
        symptoms = sentence_entry.get("symptoms", [])
        if "others" in diseases or len(symptoms) == 0:
            continue
        for drug_name, drug_info in drug_disease_dict.items():
            if any(disease in drug_info["diseases"] for disease in diseases) and any(
                symptom in drug_info["symptoms"] for symptom in symptoms
            ):
                sentence_entry["drugs"].append(drug_name)
        merge_dataset.append(sentence_entry)

    print(f"已合并完成数据集，包含 {len(merge_dataset)} 条记录")
    print(f"样本：{merge_dataset[:2]}")
    with open(DRUG_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merge_dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    merge_sentences_to_drug_dataset()
