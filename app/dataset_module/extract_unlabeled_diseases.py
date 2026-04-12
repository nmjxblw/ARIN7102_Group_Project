"""
从原始数据集中提取未被标签化的疾病-症状
"""

import os
import sys
import json
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from static_module import DRUGS_TRAINING_DATASET_FOLDER, BERT_TRAINING_DATASET_FOLDER
from utility_module import logger

DRUGS_ORIGIN_DATASET_JSON: Path = (
    Path(DRUGS_TRAINING_DATASET_FOLDER) / "enhanced_drug_table_v1.json"
)
""" 原始数据集的路径"""

SAVE_JSON: Path = (
    Path(BERT_TRAINING_DATASET_FOLDER) / "extracted_unlabeled_diseases.json"
)
""" 提取后的数据集的保存路径"""


def extract_unlabeled_diseases():
    if not DRUGS_ORIGIN_DATASET_JSON.exists():
        raise FileNotFoundError(
            f"原始数据集文件 {DRUGS_ORIGIN_DATASET_JSON} 不存在，无法提取疾病-症状数据"
        )
    drug_data: list[dict] = []
    with open(DRUGS_ORIGIN_DATASET_JSON, "r", encoding="utf-8") as f:
        drug_data = json.load(f)
    if len(drug_data) == 0:
        raise ValueError(
            f"原始数据集文件 {DRUGS_ORIGIN_DATASET_JSON} 中没有数据，无法提取疾病-症状数据"
        )
    unlabeled_disease_pairs: list[dict] = []
    for entry in tqdm(drug_data, desc="提取未标签化疾病-症状数据"):
        matched_disease_keys = entry.get("matched_disease_keys", ["others"])
        if "others" not in matched_disease_keys:
            continue
        original_conditions = entry.get("original_conditions", ["others"])
        if "others" in original_conditions:
            continue
        drug_name = entry.get("drug_name", "")
        if not drug_name:
            continue
        matched_symptoms = entry.get("matched_symptoms", [])
        unlabeled_disease_pairs.append(
            {
                "drug": drug_name,
                "diseases": [
                    str(disease).lower().strip().replace(" ", "_")
                    for disease in original_conditions
                ],
                "symptoms": matched_symptoms,
            }
        )
    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(unlabeled_disease_pairs, f, ensure_ascii=False, indent=4)
    logger.debug(
        f"提取疾病-症状数据完成，共 {len(unlabeled_disease_pairs)} 条记录，存放在 {SAVE_JSON}"
    )
