from pathlib import Path
import json
import os

from static_module import BERT_TRAINING_DATASET_FOLDER
from utility_module import logger


def merge_raw_datasets() -> None:
    full_data = []  # 存储去重后的字典列表
    seen = set()  # 存储已见字典的哈希表示（JSON字符串）

    for root, dirs, files in os.walk(BERT_TRAINING_DATASET_FOLDER):
        for filename in files:
            if filename.startswith("generated_medical_dataset") and filename.endswith(
                ".json"
            ):
                filepath = Path(root, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data: list[dict] = json.load(f)
                    for item in data:
                        # 将字典转换为可哈希的JSON字符串（排序键以确保一致性）
                        item_hash = json.dumps(item, sort_keys=True)
                        if item_hash not in seen:
                            seen.add(item_hash)
                            full_data.append(item)

    output_path = Path(BERT_TRAINING_DATASET_FOLDER, "generated_medical_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=4)

    logger.debug(f"合并原始数据完成，共 {len(full_data)} 条记录，存放在 {output_path}")
