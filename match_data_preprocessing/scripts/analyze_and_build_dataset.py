"""
分析 enhanced_drug_table.json 中的 generic_name 字段，
清洗数据并构建 商品名-有效成分 配对数据集。
"""
import json
import re
import csv
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "enhanced_drug_table.json"
OUTPUT_CSV = DATA_DIR / "drug_ingredient_pairs.csv"
OUTPUT_JSON = DATA_DIR / "drug_ingredient_pairs.json"

# ============================================================
# 1. 加载数据
# ============================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

total = len(data)

# ============================================================
# 2. 统计 generic_name 空值
# ============================================================
empty_gn = sum(1 for d in data if not d.get("generic_name", "").strip())
has_gn = total - empty_gn

# ============================================================
# 3. 分离单成分 / 复方
# ============================================================
single_records = []
multi_records = []
for d in data:
    gn = d.get("generic_name", "").strip()
    if gn:
        if " and " in gn.lower() or "," in gn or " / " in gn:
            multi_records.append(d)
        else:
            single_records.append(d)

# ============================================================
# 4. 清洗函数
# ============================================================
def clean_generic_name(gn: str) -> str:
    """清洗 generic_name：
    1) 去除首尾空格
    2) 去除括号中的给药途径注释，如 (oral route), (topical) 等
    3) 统一转小写
    4) 合并多余空格
    """
    gn = gn.strip()
    gn = re.sub(r"\s*\([^)]*\)\s*", " ", gn)   # 去除 (xxx)
    gn = gn.strip().lower()
    gn = re.sub(r"\s+", " ", gn)                # 合并空格
    return gn

# ============================================================
# 5. 统计清洗后的成分分布
# ============================================================
ingredient_count = Counter()
for d in single_records:
    clean = clean_generic_name(d["generic_name"])
    ingredient_count[clean] += 1

# ============================================================
# 6. 分析需要清洗的模式
# ============================================================
route_annotations = Counter()
for d in single_records:
    gn = d["generic_name"].strip()
    matches = re.findall(r"\(([^)]+)\)", gn)
    for m in matches:
        route_annotations[m] += 1

# ============================================================
# 7. 过滤：仅保留 单成分 且 >=2 个品牌
# ============================================================
filtered_ingredients = {ing for ing, cnt in ingredient_count.items() if cnt >= 2}
filtered_records = []
for d in single_records:
    clean = clean_generic_name(d["generic_name"])
    if clean in filtered_ingredients:
        filtered_records.append({
            "drug_name": d["drug_name"].strip().lower(),
            "active_ingredient": clean,
            "drug_classes": d.get("drug_classes", "").strip(),
            "original_conditions": d.get("original_conditions", []),
        })

# 去重（同一 drug_name 可能出现多次）
seen = set()
deduped = []
for r in filtered_records:
    key = (r["drug_name"], r["active_ingredient"])
    if key not in seen:
        seen.add(key)
        deduped.append(r)
filtered_records = deduped

# 品牌数分布
brand_dist = Counter()
for ing in filtered_ingredients:
    cnt = ingredient_count[ing]
    if cnt <= 3:
        brand_dist["2-3"] += 1
    elif cnt <= 5:
        brand_dist["4-5"] += 1
    elif cnt <= 10:
        brand_dist["6-10"] += 1
    elif cnt <= 20:
        brand_dist["11-20"] += 1
    else:
        brand_dist["20+"] += 1

# ============================================================
# 8. 输出数据集
# ============================================================
# CSV
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["drug_name", "active_ingredient", "drug_classes", "original_conditions"])
    writer.writeheader()
    for r in filtered_records:
        row = dict(r)
        row["original_conditions"] = "|".join(r["original_conditions"])
        writer.writerow(row)

# JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(filtered_records, f, ensure_ascii=False, indent=2)

# ============================================================
# 9. 打印统计摘要
# ============================================================
print("=" * 60)
print("数据统计摘要")
print("=" * 60)
print(f"原始数据总记录数:          {total}")
print(f"generic_name 非空:         {has_gn} ({has_gn/total*100:.1f}%)")
print(f"generic_name 为空:         {empty_gn} ({empty_gn/total*100:.1f}%)")
print(f"单成分记录数:              {len(single_records)} ({len(single_records)/has_gn*100:.1f}% of non-empty)")
print(f"复方记录数:                {len(multi_records)} ({len(multi_records)/has_gn*100:.1f}% of non-empty)")
print(f"清洗后唯一单成分种类:      {len(ingredient_count)}")
print(f"  其中 >=2 个品牌的成分:   {len(filtered_ingredients)}")
print(f"  其中仅 1 个品牌的成分:   {len(ingredient_count) - len(filtered_ingredients)}")
print(f"\n最终数据集记录数:          {len(filtered_records)}")
print(f"最终唯一有效成分:          {len(filtered_ingredients)}")
print()
print("品牌数分布 (过滤后):")
for k in ["2-3", "4-5", "6-10", "11-20", "20+"]:
    print(f"  {k} 个品牌: {brand_dist.get(k, 0)} 种成分")
print()
print("Top 20 有效成分:")
for ing in sorted(filtered_ingredients, key=lambda x: ingredient_count[x], reverse=True)[:20]:
    print(f"  {ing}: {ingredient_count[ing]} 个品牌")
print()
print("需要清洗的括号注释 (Top 15):")
for ann, cnt in route_annotations.most_common(15):
    print(f"  ({ann}): {cnt} 条")
print()
print(f"数据集已输出至:")
print(f"  CSV:  {OUTPUT_CSV}")
print(f"  JSON: {OUTPUT_JSON}")
