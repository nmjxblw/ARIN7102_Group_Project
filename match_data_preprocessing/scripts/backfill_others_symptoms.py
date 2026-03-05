"""
将 LLM 生成的 others 症状回填到 enhanced_drug_table.csv

功能:
    1. 读取 others_condition_symptoms.json (LLM 生成的 condition→symptoms 映射)
    2. 读取 enhanced_drug_table.csv
    3. 对 matched_disease_keys 为 ["others"] 的药物:
       - 根据 original_conditions 查找对应症状
       - 回填 matched_symptoms 和 symptom_severity
    4. 保存更新后的 enhanced_drug_table.csv

使用方式:
    python backfill_others_symptoms.py

    可选参数:
    --mapping   症状映射文件路径 (默认: ../data/others_condition_symptoms.json)
    --input     输入CSV路径 (默认: ../data/enhanced_drug_table.csv)
    --output    输出CSV路径 (默认: 覆盖输入文件)
    --dry-run   仅预览不保存
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PREPROCESS_DIR = PROJECT_ROOT / "match_data_preprocessing"
DATA_DIR = PREPROCESS_DIR / "data"
DEFAULT_MAPPING = DATA_DIR / "others_condition_symptoms.json"
DEFAULT_TABLE = DATA_DIR / "enhanced_drug_table.csv"

# DS8: 症状严重度
DS8_PATH = PROJECT_ROOT / "app" / "dataset_module" / "disease-symptom-description-dataset" / "Symptom-severity_cleaned.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 LLM 生成的 others 症状回填到 enhanced_drug_table.csv"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=str(DEFAULT_MAPPING),
        help="症状映射 JSON 文件路径",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_TABLE),
        help="输入 enhanced_drug_table.csv 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 CSV 路径 (默认覆盖输入文件)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览, 不保存",
    )
    return parser.parse_args()


def load_symptom_severity_map() -> dict:
    """从 DS8 加载 symptom → weight 映射"""
    if not DS8_PATH.exists():
        print(f"[WARN] DS8 文件不存在: {DS8_PATH}")
        return {}
    ds8 = pd.read_csv(DS8_PATH)
    ds8.columns = [c.strip() for c in ds8.columns]
    severity_map = {}
    for _, row in ds8.iterrows():
        symptom = str(row.get("Symptom", "")).strip()
        weight = row.get("weight", 1)
        if symptom and symptom.lower() != "nan":
            try:
                severity_map[symptom] = int(weight)
            except (ValueError, TypeError):
                severity_map[symptom] = 1
    # 也建立小写映射用于模糊匹配
    severity_map_lower = {}
    for k, v in severity_map.items():
        severity_map_lower[k.lower().replace(" ", "_")] = v
    return severity_map, severity_map_lower


def match_severity(symptom: str, severity_map: dict, severity_map_lower: dict) -> int:
    """尝试匹配症状的严重度权重"""
    # 精确匹配
    if symptom in severity_map:
        return severity_map[symptom]
    # 小写下划线格式匹配
    sym_norm = symptom.lower().replace(" ", "_")
    if sym_norm in severity_map_lower:
        return severity_map_lower[sym_norm]
    # 去下划线匹配
    sym_clean = symptom.lower().replace("_", " ").strip()
    for k, v in severity_map.items():
        if k.lower().replace("_", " ").strip() == sym_clean:
            return v
    return -1  # 未匹配


def main():
    args = parse_args()
    output_path = args.output or args.input

    print("=" * 60)
    print("回填 others 症状到 enhanced_drug_table.csv")
    print("=" * 60)

    # Step 1: 加载 LLM 生成的映射
    print("\n[Step 1] 加载症状映射文件 ...")
    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"[ERROR] 映射文件不存在: {mapping_path}")
        print("请先运行 generate_others_symptoms.py 生成映射文件")
        sys.exit(1)

    with open(mapping_path, "r", encoding="utf-8") as f:
        cond_symptom_map = json.load(f)
    print(f"  加载 {len(cond_symptom_map)} 条 condition->symptoms 映射")
    has_symptoms = sum(1 for v in cond_symptom_map.values() if v)
    print(f"  其中有症状的: {has_symptoms}")

    # Step 2: 加载 DS8 严重度映射
    print("\n[Step 2] 加载症状严重度 (DS8) ...")
    severity_map, severity_map_lower = load_symptom_severity_map()
    print(f"  DS8 症状数: {len(severity_map)}")

    # Step 3: 加载 enhanced_drug_table
    print("\n[Step 3] 加载 enhanced_drug_table.csv ...")
    df = pd.read_csv(args.input)
    total = len(df)
    others_mask = df["matched_disease_keys"] == '["others"]'
    others_count = others_mask.sum()
    print(f"  总药物数: {total}")
    print(f"  Others 药物数: {others_count}")

    # Step 4: 回填
    print("\n[Step 4] 回填症状 ...")
    updated_count = 0
    symptom_counts = []

    for idx in df[others_mask].index:
        original_conds_json = df.at[idx, "original_conditions"]
        try:
            conditions = json.loads(original_conds_json)
        except (json.JSONDecodeError, TypeError):
            conditions = []

        # 收集该药物所有 condition 对应的症状
        all_symptoms = set()
        for cond in conditions:
            cond_clean = cond.strip()
            symptoms = cond_symptom_map.get(cond_clean, [])
            if not symptoms:
                # 尝试不区分大小写匹配
                for k, v in cond_symptom_map.items():
                    if k.lower().strip() == cond_clean.lower():
                        symptoms = v
                        break
            all_symptoms.update(symptoms)

        if all_symptoms:
            symptoms_list = sorted(all_symptoms)
            df.at[idx, "matched_symptoms"] = json.dumps(symptoms_list, ensure_ascii=False)

            # 计算 symptom_severity
            severity = {}
            for s in symptoms_list:
                w = match_severity(s, severity_map, severity_map_lower)
                if w > 0:
                    severity[s] = w
            if severity:
                df.at[idx, "symptom_severity"] = json.dumps(severity, ensure_ascii=False)
            else:
                df.at[idx, "symptom_severity"] = "{}"

            updated_count += 1
            symptom_counts.append(len(symptoms_list))

    # Step 5: 统计与保存
    print(f"\n  回填完成:")
    print(f"    Others 药物中成功回填症状: {updated_count} / {others_count}")
    still_empty = others_count - updated_count
    print(f"    仍然无症状的 Others 药物: {still_empty}")
    if symptom_counts:
        avg_sym = sum(symptom_counts) / len(symptom_counts)
        print(f"    平均每个药物的症状数:     {avg_sym:.1f}")

    # 全表统计
    all_empty = (df["matched_symptoms"] == "[]").sum() + df["matched_symptoms"].isna().sum()
    all_nonempty = total - all_empty
    print(f"\n  全表统计:")
    print(f"    有症状的药物: {all_nonempty} / {total} ({all_nonempty/total*100:.1f}%)")
    print(f"    无症状的药物: {all_empty} / {total} ({all_empty/total*100:.1f}%)")

    if args.dry_run:
        print("\n[DRY-RUN] 预览模式, 未保存文件")
        # 显示几个样本
        print("\n--- 回填样本 ---")
        sample_idx = df[others_mask & (df["matched_symptoms"] != "[]")].index[:3]
        for idx in sample_idx:
            print(f"  Drug: {df.at[idx, 'drug_name']}")
            print(f"    Conditions: {df.at[idx, 'original_conditions']}")
            syms = df.at[idx, "matched_symptoms"]
            print(f"    Symptoms: {syms[:200]}...")
            print()
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n[DONE] 已保存至: {output_path}")
        file_size = Path(output_path).stat().st_size / 1024
        print(f"  文件大小: {file_size:.1f} KB")


if __name__ == "__main__":
    main()
