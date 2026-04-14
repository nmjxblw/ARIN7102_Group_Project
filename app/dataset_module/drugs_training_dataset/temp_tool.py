import json
import os
from pathlib import Path

if __name__ == "__main__":
    file_path = Path.cwd() / "eval_dataset_llm_v2.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"数据集中共有 {len(data)} 条记录。")
    data.sort(key=lambda x: x.get("sentence", ""))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"数据集已按句子排序并保存到 {file_path}")
