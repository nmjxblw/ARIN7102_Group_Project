"""生成疾病-症状数据"""

import os
import sys
import json
import re
from pathlib import Path
import pandas as pd


MODULE_DIR: Path = Path(__file__).parent


def load_disease_with_symptoms() -> dict[str, dict[str, str | list[str]]]:
    """
    读取对应的疾病-症状数据

    Returns:
        disease_symptoms_dict(dict[str, dict[str, str | list[str]]]): 疾病-症状数据

    Example:
        >>> load_disease_with_symptoms()
        {
            "disease": {
                "symptoms": ["symptom1", "symptom2"], # 症状列表
                "description": "疾病描述" # 疾病描述
                ...
            }
        }
    """
    disease_symptoms_description_dict: dict[str, dict[str, str | list[str]]] = {}
    disease_description_file_name = Path(
        MODULE_DIR,
        "disease-symptom-description-dataset",
        "symptom_Description_cleaned.csv",
    )
    if not disease_description_file_name.exists():
        raise FileNotFoundError(
            f"疾病描述数据文件 {disease_description_file_name} 不存在"
        )

    disease_description_df = pd.read_csv(disease_description_file_name)
    for idx, row in disease_description_df.iterrows():
        disease_list: list = row.filter(like="Disease").dropna().tolist()
        if len(disease_list) == 0:
            continue
        disease: str = (
            str(disease_list[0])
            .strip()
            .replace(" (", "(")
            .replace(") ", ")")
            .replace(" ", "_")
            .lower()
        )
        description: str = str(row["Description"]).strip()
        if disease not in disease_symptoms_description_dict:
            disease_symptoms_description_dict[disease] = {
                "symptoms": [],
                "description": description,
            }
        else:
            # 只存一遍
            continue
    disease_names = list(disease_symptoms_description_dict.keys())
    disease_symptoms_file_name = Path(
        MODULE_DIR, "disease-symptom-description-dataset", "dataset_cleaned.csv"
    )
    if not disease_symptoms_file_name.exists():
        raise FileNotFoundError(
            f"疾病-症状数据文件 {disease_symptoms_file_name} 不存在"
        )
    disease_symptom_df = pd.read_csv(disease_symptoms_file_name)
    for idx, row in disease_symptom_df.iterrows():
        disease_list: list = row.filter(like="Disease").dropna().tolist()
        if len(disease_list) == 0:
            continue
        disease: str = (
            str(disease_list[0])
            .strip()
            .replace(" (", "(")
            .replace(") ", ")")
            .replace(" ", "_")
            .lower()
        )
        symptoms: list = row.filter(like="Symptom").dropna().tolist()
        for symptom in symptoms:
            symptom = str(symptom).strip().replace(" ", "").lower()
            if disease not in disease_symptoms_description_dict:
                disease_symptoms_description_dict[disease] = {
                    "symptoms": [symptom],
                    "description": "",
                }
            elif isinstance(
                disease_symptoms_description_dict[disease]["symptoms"], list
            ):
                if (
                    symptom
                    not in disease_symptoms_description_dict[disease]["symptoms"]
                ):
                    disease_symptoms_description_dict[disease]["symptoms"].append(  # type: ignore
                        symptom
                    )

            else:
                # 添加新的病症
                disease_symptoms_description_dict[disease]["symptoms"] = [symptom]
    json_file_name = Path(
        MODULE_DIR, "disease-symptom-description-dataset", "disease_symptoms_dict.json"
    )
    with open(json_file_name, "w", encoding="utf-8") as f:
        json.dump(disease_symptoms_description_dict, f, ensure_ascii=False, indent=4)
    return disease_symptoms_description_dict


if __name__ == "__main__":
    load_disease_with_symptoms()
