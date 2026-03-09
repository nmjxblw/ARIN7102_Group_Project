from optparse import Option
import os
import sys
import json
from typing import Optional, Callable
import pandas as pd
from pathlib import Path
import ast
import re


def parse_array_string(val):
    """解析数组字符串"""
    if pd.isna(val):
        return val

    if not isinstance(val, str):
        return val

    val = val.strip()

    # 空字符串返回空列表
    if val == "":
        return []

    # 尝试用json.loads解析
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        # JSON解析失败，尝试其他方法
        pass

    # 尝试用ast.literal_eval解析
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # 如果都失败，返回原始值
        return val


def parse_related_drugs(val: str) -> dict[str, str]:
    if pd.isna(val):
        return {}
    if not isinstance(val, str):
        return {}
    list_of_pairs = val.split(" | ")
    return {
        name: url
        for name, url in (pair.split(": ") for pair in list_of_pairs if ": " in pair)
    }


def parse_disease_description(val: str) -> dict[str, str]:
    if pd.isna(val):
        return {}
    if not isinstance(val, str):
        return {}
    list_of_pairs = val.split(" | ")
    # [disease] description
    pattern: re.Pattern = re.compile(r"\[(?P<disease>.*?)\]\s*(?P<description>.*)")
    result = {}
    for pair in list_of_pairs:
        if pair is not None:
            match = pattern.match(pair)
            if match:
                disease, description = match.groupdict().values()
                result[disease] = description
    return result


def csv_to_json(
    csv_file_path: Path,
    convert_policy: Optional[dict[str, Callable]] = {},
    json_file_path: Optional[Path] = None,
) -> None:
    """将CSV文件转换为JSON格式"""
    if not csv_file_path.exists():
        raise FileNotFoundError(f"CSV文件 {csv_file_path} 不存在")

    # 读取CSV文件
    df = pd.read_csv(
        csv_file_path,
        encoding="utf-8",
        converters=convert_policy,
    ).fillna("")

    # 将DataFrame转换为字典列表
    data_dict_list = df.to_dict(orient="records")

    # 将字典列表写入JSON文件
    if json_file_path is None:
        json_file_path = csv_file_path.with_suffix(".json")

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(data_dict_list, json_file, ensure_ascii=False, indent=4)

    print(f"已成功将CSV文件 {csv_file_path} 转换为JSON文件 {json_file_path}")


def convert_all_csv_to_json():
    """将当前目录下的所有CSV文件转换为JSON格式"""
    for root, dirs, files in os.walk(Path.cwd() / "dataset_module"):
        for filename in files:
            if filename.endswith(".csv"):
                csv_file_path = Path(root) / filename
                csv_to_json(
                    csv_file_path,
                    convert_policy={
                        "related_drugs": parse_related_drugs,
                        "original_conditions": parse_array_string,
                        "matched_disease_keys": parse_array_string,
                        "matched_symptoms": parse_array_string,
                        "symptom_severity": parse_array_string,
                        "disease_description": parse_disease_description,
                    },
                )  # Pass an empty convert_policy


if __name__ == "__main__":
    convert_all_csv_to_json()
