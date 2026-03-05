"""
构建增强药物表 enhanced_drug_table.csv

从多个数据源合并药物信息，执行模糊匹配将适应症映射到 disease_keys.json 中的标准疾病名，
最终输出以药名为主键的增强CSV表。

使用方式:
    python build_enhanced_drug_table.py

输出:
    match_data_preprocessing/data/enhanced_drug_table.csv
"""

import os
import sys
import json
import warnings
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # ARIN7102_Group_Project
DATASET_DIR = PROJECT_ROOT / "app" / "dataset_module"
PREPROCESS_DIR = PROJECT_ROOT / "match_data_preprocessing"
OUTPUT_DIR = PREPROCESS_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 同义词映射表: 原始condition关键词 → disease_key
# ============================================================
SYNONYM_MAP = {
    # 高血压
    "high blood pressure": "hypertension",
    "hbp": "hypertension",
    "hypertension": "hypertension",
    "elevated blood pressure": "hypertension",
    "pulmonary hypertension": "hypertension",
    "pulmonary arterial hypertension": "hypertension",
    # 感冒/流感
    "flu": "common_cold",
    "influenza": "common_cold",
    "common cold": "common_cold",
    "upper respiratory": "common_cold",
    "nasal congestion": "common_cold",
    "sinusitis": "common_cold",
    "rhinitis": "common_cold",
    # 痔疮
    "piles": "dimorphic_hemmorhoids(piles)",
    "hemorrhoid": "dimorphic_hemmorhoids(piles)",
    "haemorrhoid": "dimorphic_hemmorhoids(piles)",
    # HIV/AIDS
    "hiv": "aids",
    "human immunodeficiency": "aids",
    "aids": "aids",
    "hiv infection": "aids",
    # 糖尿病
    "diabetes": "diabetes",
    "type 1 diabetes": "diabetes",
    "type 2 diabetes": "diabetes",
    "diabetic": "diabetes",
    "blood sugar": "diabetes",
    "insulin resistance": "diabetes",
    "hyperglycemia": "diabetes",
    # 肺结核
    "tuberculosis": "tuberculosis",
    "tb": "tuberculosis",
    # GERD
    "gastroesophageal reflux": "gerd",
    "acid reflux": "gerd",
    "heartburn": "gerd",
    "gerd": "gerd",
    "reflux": "gerd",
    # 偏头痛
    "migraine": "migraine",
    "migraine prevention": "migraine",
    "migraine prophylaxis": "migraine",
    # 哮喘
    "asthma": "bronchial_asthma",
    "bronchitis": "bronchial_asthma",
    "bronchial asthma": "bronchial_asthma",
    "copd": "bronchial_asthma",
    "chronic obstructive pulmonary": "bronchial_asthma",
    "bronchospasm": "bronchial_asthma",
    # 水痘
    "chicken pox": "chicken_pox",
    "chickenpox": "chicken_pox",
    "varicella": "chicken_pox",
    # 登革热
    "dengue": "dengue",
    "dengue fever": "dengue",
    # 伤寒
    "typhoid": "typhoid",
    "typhoid fever": "typhoid",
    # 疟疾
    "malaria": "malaria",
    # 肺炎
    "pneumonia": "pneumonia",
    "community-acquired pneumonia": "pneumonia",
    # 黄疸
    "jaundice": "jaundice",
    "neonatal jaundice": "jaundice",
    # 肝炎
    "hepatitis a": "hepatitis_a",
    "hepatitis b": "hepatitis_b",
    "hepatitis c": "hepatitis_c",
    "hepatitis d": "hepatitis_d",
    "hepatitis e": "hepatitis_e",
    "alcoholic hepatitis": "alcoholic_hepatitis",
    "chronic hepatitis b": "hepatitis_b",
    "chronic hepatitis c": "hepatitis_c",
    # 甲状腺
    "hypothyroid": "hypothyroidism",
    "hypothyroidism": "hypothyroidism",
    "underactive thyroid": "hypothyroidism",
    "hashimoto": "hypothyroidism",
    "hyperthyroid": "hyperthyroidism",
    "hyperthyroidism": "hyperthyroidism",
    "overactive thyroid": "hyperthyroidism",
    "graves": "hyperthyroidism",
    "thyroid": "hypothyroidism",
    # 低血糖
    "hypoglycemia": "hypoglycemia",
    "low blood sugar": "hypoglycemia",
    # 眩晕
    "vertigo": "(vertigo)paroymsal__positional_vertigo",
    "bppv": "(vertigo)paroymsal__positional_vertigo",
    "dizziness": "(vertigo)paroymsal__positional_vertigo",
    # 痤疮
    "acne": "acne",
    "acne vulgaris": "acne",
    "pimple": "acne",
    # 尿路感染
    "urinary tract infection": "urinary_tract_infection",
    "uti": "urinary_tract_infection",
    "bladder infection": "urinary_tract_infection",
    "cystitis": "urinary_tract_infection",
    # 银屑病
    "psoriasis": "psoriasis",
    "plaque psoriasis": "psoriasis",
    # 脓疱疮
    "impetigo": "impetigo",
    # 真菌感染
    "fungal": "fungal_infection",
    "fungal infection": "fungal_infection",
    "fungus": "fungal_infection",
    "ringworm": "fungal_infection",
    "yeast infection": "fungal_infection",
    "candida": "fungal_infection",
    "candidiasis": "fungal_infection",
    "tinea": "fungal_infection",
    "athlete's foot": "fungal_infection",
    "jock itch": "fungal_infection",
    "onychomycosis": "fungal_infection",
    # 过敏
    "allergy": "allergy",
    "allergic": "allergy",
    "allergies": "allergy",
    "allergic rhinitis": "allergy",
    "hay fever": "allergy",
    "anaphylaxis": "allergy",
    "hives": "allergy",
    "urticaria": "allergy",
    # 消化性溃疡
    "peptic ulcer": "peptic_ulcer_diseae",
    "stomach ulcer": "peptic_ulcer_diseae",
    "gastric ulcer": "peptic_ulcer_diseae",
    "duodenal ulcer": "peptic_ulcer_diseae",
    # 药物反应
    "drug reaction": "drug_reaction",
    "adverse drug": "drug_reaction",
    "drug allergy": "drug_reaction",
    # 胆汁淤积
    "cholestasis": "chronic_cholestasis",
    "cholestatic": "chronic_cholestasis",
    # 瘫痪/脑出血
    "paralysis": "paralysis(brain_hemorrhage)",
    "brain hemorrhage": "paralysis(brain_hemorrhage)",
    "stroke": "paralysis(brain_hemorrhage)",
    "cerebral hemorrhage": "paralysis(brain_hemorrhage)",
    "intracerebral hemorrhage": "paralysis(brain_hemorrhage)",
    # 心脏病
    "heart attack": "heart_attack",
    "myocardial infarction": "heart_attack",
    "cardiac arrest": "heart_attack",
    "angina": "heart_attack",
    "acute coronary": "heart_attack",
    # 静脉曲张
    "varicose": "varicose_veins",
    "varicose veins": "varicose_veins",
    # 颈椎病
    "cervical spondylosis": "cervical_spondylosis",
    "spondylosis": "cervical_spondylosis",
    # 关节炎
    "arthritis": "arthritis",
    "rheumatoid arthritis": "arthritis",
    "rheumatoid": "arthritis",
    "osteoarthritis": "osteoarthristis",
    "osteoarthristis": "osteoarthristis",
    "degenerative joint": "osteoarthristis",
    # 胃肠炎
    "gastroenteritis": "gastroenteritis",
    "stomach flu": "gastroenteritis",
    "food poisoning": "gastroenteritis",
    "gastritis": "gastroenteritis",
    "diarrhea": "gastroenteritis",
    "nausea": "gastroenteritis",
    "vomiting": "gastroenteritis",
    # 头痛 → migraine
    "headache": "migraine",
    "tension headache": "migraine",
    "cluster headache": "migraine",
}

# disease_key 关键词（从 key 中提取的核心词，用于包含匹配）
DISEASE_KEY_KEYWORDS = {
    "fungal_infection": ["fungal", "fungus", "mycosis"],
    "allergy": ["allergy", "allergic", "allergen"],
    "gerd": ["gerd", "reflux", "heartburn"],
    "chronic_cholestasis": ["cholestasis", "cholestatic"],
    "drug_reaction": ["drug reaction", "adverse drug"],
    "peptic_ulcer_diseae": ["peptic", "ulcer", "gastric ulcer"],
    "aids": ["aids", "hiv"],
    "diabetes": ["diabetes", "diabetic", "insulin"],
    "gastroenteritis": ["gastroenteritis", "gastritis"],
    "bronchial_asthma": ["asthma", "bronchial", "bronchitis", "copd", "bronchospasm"],
    "hypertension": ["hypertension", "high blood pressure"],
    "migraine": ["migraine"],
    "cervical_spondylosis": ["cervical spondylosis", "spondylosis"],
    "paralysis(brain_hemorrhage)": ["paralysis", "brain hemorrhage", "stroke"],
    "jaundice": ["jaundice"],
    "malaria": ["malaria"],
    "chicken_pox": ["chicken pox", "chickenpox", "varicella"],
    "dengue": ["dengue"],
    "typhoid": ["typhoid"],
    "hepatitis_a": ["hepatitis a"],
    "hepatitis_b": ["hepatitis b"],
    "hepatitis_c": ["hepatitis c"],
    "hepatitis_d": ["hepatitis d"],
    "hepatitis_e": ["hepatitis e"],
    "alcoholic_hepatitis": ["alcoholic hepatitis"],
    "tuberculosis": ["tuberculosis", "tb"],
    "common_cold": ["common cold", "cold", "influenza", "flu"],
    "pneumonia": ["pneumonia"],
    "dimorphic_hemmorhoids(piles)": ["hemorrhoid", "piles", "haemorrhoid"],
    "heart_attack": ["heart attack", "myocardial infarction", "angina"],
    "varicose_veins": ["varicose"],
    "hypothyroidism": ["hypothyroid", "underactive thyroid"],
    "hyperthyroidism": ["hyperthyroid", "overactive thyroid", "graves"],
    "hypoglycemia": ["hypoglycemia", "low blood sugar"],
    "osteoarthristis": ["osteoarthritis", "osteoarthristis"],
    "arthritis": ["arthritis", "rheumatoid"],
    "(vertigo)paroymsal__positional_vertigo": ["vertigo", "bppv"],
    "acne": ["acne", "pimple"],
    "urinary_tract_infection": ["urinary tract", "uti", "bladder infection", "cystitis"],
    "psoriasis": ["psoriasis"],
    "impetigo": ["impetigo"],
}


def load_disease_keys():
    """加载 disease_keys.json"""
    path = PREPROCESS_DIR / "disease_keys.json"
    with open(path, "r", encoding="utf-8") as f:
        keys = json.load(f)
    print(f"[INFO] 加载 disease_keys: {len(keys)} 个疾病")
    return keys


def load_ds1():
    """DS1: drug-prescription-to-disease-dataset"""
    path = DATASET_DIR / "drug-prescription-to-disease-dataset" / "final_cleaned.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"[INFO] DS1 加载完成: {len(df)} 行, 列: {list(df.columns)}")
    return df


def load_ds2():
    """DS2: drugs-side-effects-and-medical-condition"""
    path = DATASET_DIR / "drugs-side-effects-and-medical-condition" / "drugs_side_effects_drugs_com_cleaned.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"[INFO] DS2 加载完成: {len(df)} 行, 列: {list(df.columns)}")
    return df


def load_ds3():
    """DS3: drugs-related-to-common-treatments"""
    path = DATASET_DIR / "drugs-related-to-common-treatments" / "drugs_for_common_treatments_cleaned.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"[INFO] DS3 加载完成: {len(df)} 行, 列: {list(df.columns)}")
    return df


def load_ds4():
    """DS4: kuc-hackathon-winter-2018 (train + test)"""
    train_path = DATASET_DIR / "kuc-hackathon-winter-2018" / "drugsComTrain_raw_cleaned.csv"
    test_path = DATASET_DIR / "kuc-hackathon-winter-2018" / "drugsComTest_raw_cleaned.csv"
    dfs = []
    for p in [train_path, test_path]:
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            dfs.append(df)
    ds4 = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"[INFO] DS4 加载完成: {len(ds4)} 行, 列: {list(ds4.columns)}")
    return ds4


def load_ds5():
    """DS5: disease→symptoms 映射"""
    path = DATASET_DIR / "disease-symptom-description-dataset" / "dataset_cleaned.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"[INFO] DS5 加载完成: {len(df)} 行, 列: {list(df.columns)}")
    return df


def load_ds6():
    """DS6: symptom_Description (疾病描述)"""
    path = DATASET_DIR / "disease-symptom-description-dataset" / "symptom_Description_cleaned.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"[INFO] DS6 加载完成: {len(df)} 行, 列: {list(df.columns)}")
    return df


def load_ds8():
    """DS8: symptom severity (症状严重度权重)"""
    path = DATASET_DIR / "disease-symptom-description-dataset" / "Symptom-severity_cleaned.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"[INFO] DS8 加载完成: {len(df)} 行, 列: {list(df.columns)}")
    return df


def normalize_name(name):
    """标准化名称: 小写、去空格、下划线替换空格"""
    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace("__", "_")


def match_condition_to_disease_keys(condition, disease_keys):
    """
    将一个原始 condition 匹配到 disease_keys 列表中。

    匹配策略（逐层递进）:
    1. 精确匹配（标准化后）
    2. 同义词表匹配
    3. 包含匹配（关键词）
    4. 模糊字符串匹配（SequenceMatcher ≥ 0.65）
    5. 兜底 → None（不匹配）

    返回匹配到的 disease_key 或 None
    """
    if not isinstance(condition, str) or condition.strip() == "" or condition.strip().lower() == "nan":
        return None

    condition_clean = condition.strip()
    condition_lower = condition_clean.lower()
    condition_normalized = normalize_name(condition_clean)

    # 层级 1: 精确匹配
    for key in disease_keys:
        if key == "others":
            continue
        if condition_normalized == key:
            return key
        # 也尝试不带下划线的比较
        if condition_lower.replace("_", " ").replace("-", " ") == key.replace("_", " ").replace("(", "").replace(")", ""):
            return key

    # 层级 2: 同义词表
    for synonym, mapped_key in SYNONYM_MAP.items():
        if synonym in condition_lower:
            return mapped_key

    # 层级 3: 包含匹配（disease_key的关键词是否出现在condition中）
    for key, keywords in DISEASE_KEY_KEYWORDS.items():
        for kw in keywords:
            if kw in condition_lower:
                return key

    # 层级 4: 模糊匹配
    best_match = None
    best_score = 0
    condition_words = condition_lower.replace("_", " ").replace("-", " ")
    for key in disease_keys:
        if key == "others":
            continue
        key_words = key.replace("_", " ").replace("(", "").replace(")", "")
        score = SequenceMatcher(None, condition_words, key_words).ratio()
        if score > best_score:
            best_score = score
            best_match = key
    if best_score >= 0.65:
        return best_match

    # 层级 5: 未匹配
    return None


def build_drug_conditions_map(ds1, ds2, ds4):
    """
    从 DS1、DS2、DS4 中提取所有 (drug_name, condition) 对，
    按药名聚合 original_conditions。
    """
    pairs = []

    # DS1: drug, disease
    if "drug" in ds1.columns and "disease" in ds1.columns:
        p1 = ds1[["drug", "disease"]].copy()
        p1.columns = ["drug_name", "condition"]
        pairs.append(p1)

    # DS2: drug_name, medical_condition
    if "drug_name" in ds2.columns and "medical_condition" in ds2.columns:
        p2 = ds2[["drug_name", "medical_condition"]].copy()
        p2.columns = ["drug_name", "condition"]
        pairs.append(p2)

    # DS4: drugName, condition
    if "drugName" in ds4.columns and "condition" in ds4.columns:
        p4 = ds4[["drugName", "condition"]].copy()
        p4.columns = ["drug_name", "condition"]
        pairs.append(p4)

    all_pairs = pd.concat(pairs, ignore_index=True)
    all_pairs["drug_name"] = all_pairs["drug_name"].apply(
        lambda x: x.strip().lower() if isinstance(x, str) else ""
    )
    all_pairs["condition"] = all_pairs["condition"].apply(
        lambda x: x.strip() if isinstance(x, str) else ""
    )
    # 去除空值
    all_pairs = all_pairs[
        (all_pairs["drug_name"] != "") & (all_pairs["condition"] != "") &
        (all_pairs["condition"].str.lower() != "nan")
    ]
    all_pairs = all_pairs.drop_duplicates()

    print(f"[INFO] 合并后共 {len(all_pairs)} 条 (drug, condition) 对")

    # 按药名聚合
    drug_cond_map = all_pairs.groupby("drug_name")["condition"].apply(
        lambda x: list(sorted(set(x)))
    ).reset_index()
    drug_cond_map.columns = ["drug_name", "original_conditions"]

    print(f"[INFO] 共 {len(drug_cond_map)} 种药物")
    return drug_cond_map


def build_drug_attributes(ds2, ds3):
    """
    以 DS2 为底表，提取药物属性。
    同一药物多行时取 no_of_reviews 最多的行。
    DS2 缺失的字段从 DS3 补充。
    """
    attr_cols = [
        "drug_name", "generic_name", "drug_classes", "brand_names",
        "side_effects", "activity", "rx_otc", "pregnancy_category",
        "csa", "alcohol", "related_drugs", "medical_condition_description"
    ]

    ds2_copy = ds2.copy()
    ds2_copy["drug_name"] = ds2_copy["drug_name"].apply(
        lambda x: x.strip().lower() if isinstance(x, str) else ""
    )
    # 过滤空药名
    ds2_copy = ds2_copy[ds2_copy["drug_name"] != ""]

    # 确保 no_of_reviews 是数值
    ds2_copy["no_of_reviews"] = pd.to_numeric(ds2_copy["no_of_reviews"], errors="coerce").fillna(0)

    # 按 no_of_reviews 降序，取每种药物的第一行
    ds2_copy = ds2_copy.sort_values("no_of_reviews", ascending=False)
    ds2_dedup = ds2_copy.groupby("drug_name").first().reset_index()

    # 只保留需要的列
    available_cols = [c for c in attr_cols if c in ds2_dedup.columns]
    attrs = ds2_dedup[available_cols].copy()

    # DS3 补充缺失字段
    if ds3 is not None and len(ds3) > 0:
        ds3_copy = ds3.copy()
        ds3_copy["drug_name"] = ds3_copy["drug_name"].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else ""
        )
        ds3_copy = ds3_copy[ds3_copy["drug_name"] != ""]
        ds3_copy["no_of_reviews"] = pd.to_numeric(ds3_copy["no_of_reviews"], errors="coerce").fillna(0)
        ds3_dedup = ds3_copy.sort_values("no_of_reviews", ascending=False).groupby("drug_name").first().reset_index()

        for col in ["activity", "rx_otc", "pregnancy_category"]:
            if col in attrs.columns and col in ds3_dedup.columns:
                # 只补充 attrs 中为 NaN 的
                fill_map = ds3_dedup.set_index("drug_name")[col]
                mask = attrs[col].isna() | (attrs[col].astype(str).str.strip() == "") | (attrs[col].astype(str).str.lower() == "nan")
                attrs.loc[mask, col] = attrs.loc[mask, "drug_name"].map(fill_map)

    print(f"[INFO] 药物属性表: {len(attrs)} 种药物, {len(attrs.columns)} 个字段")
    return attrs


def build_ratings(ds2, ds4):
    """
    聚合评分:
    - DS2: rating, no_of_reviews (官方)
    - DS4: rating, count (用户)
    加权平均得到 avg_rating 和 total_reviews
    """
    # DS2 评分
    ds2_copy = ds2.copy()
    ds2_copy["drug_name"] = ds2_copy["drug_name"].apply(
        lambda x: x.strip().lower() if isinstance(x, str) else ""
    )
    ds2_copy["rating"] = pd.to_numeric(ds2_copy["rating"], errors="coerce")
    ds2_copy["no_of_reviews"] = pd.to_numeric(ds2_copy["no_of_reviews"], errors="coerce")
    ds2_ratings = ds2_copy.groupby("drug_name").agg(
        ds2_sum_rating_x_reviews=("rating", lambda x: (x * ds2_copy.loc[x.index, "no_of_reviews"]).sum()),
        ds2_total_reviews=("no_of_reviews", "sum")
    ).reset_index()

    # DS4 评分
    ds4_copy = ds4.copy()
    if "drugName" in ds4_copy.columns:
        ds4_copy["drug_name"] = ds4_copy["drugName"].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else ""
        )
    else:
        ds4_copy["drug_name"] = ""
    ds4_copy["rating"] = pd.to_numeric(ds4_copy["rating"], errors="coerce")
    ds4_ratings = ds4_copy[ds4_copy["drug_name"] != ""].groupby("drug_name").agg(
        ds4_avg_rating=("rating", "mean"),
        ds4_total_reviews=("rating", "count")
    ).reset_index()

    # 合并
    ratings = ds2_ratings.merge(ds4_ratings, on="drug_name", how="outer")

    # 加权平均
    def compute_weighted_avg(row):
        ds2_total = row.get("ds2_total_reviews", 0)
        ds4_total = row.get("ds4_total_reviews", 0)
        if pd.isna(ds2_total):
            ds2_total = 0
        if pd.isna(ds4_total):
            ds4_total = 0

        ds2_sum = row.get("ds2_sum_rating_x_reviews", 0)
        if pd.isna(ds2_sum):
            ds2_sum = 0

        ds4_avg = row.get("ds4_avg_rating", 0)
        if pd.isna(ds4_avg):
            ds4_avg = 0

        total = ds2_total + ds4_total
        if total == 0:
            return pd.Series({"avg_rating": np.nan, "total_reviews": 0})

        weighted = (ds2_sum + ds4_avg * ds4_total) / total
        return pd.Series({"avg_rating": round(weighted, 2), "total_reviews": int(total)})

    result = ratings.apply(compute_weighted_avg, axis=1)
    ratings = pd.concat([ratings[["drug_name"]], result], axis=1)

    print(f"[INFO] 评分聚合: {len(ratings)} 种药物有评分数据")
    return ratings


def build_disease_description_map(ds6, disease_keys):
    """
    从 DS6 构建 disease_key → description 映射
    """
    desc_map = {}
    for _, row in ds6.iterrows():
        disease = row.get("Disease", "")
        description = row.get("Description", "")
        if not isinstance(disease, str) or disease.strip() == "":
            continue
        # 标准化 DS6 的疾病名并与 disease_keys 匹配
        disease_norm = normalize_name(disease)
        for key in disease_keys:
            if key == "others":
                continue
            if disease_norm == key:
                desc_map[key] = description
                break
            # 尝试更宽松的匹配
            key_clean = key.replace("_", " ").replace("(", "").replace(")", "")
            disease_clean = disease.strip().lower()
            if disease_clean == key_clean or disease_clean.replace(" ", "_") == key:
                desc_map[key] = description
                break

    print(f"[INFO] 疾病描述映射: {len(desc_map)} / {len(disease_keys)} 个疾病有描述")
    return desc_map


def build_disease_symptom_map(ds5, disease_keys):
    """
    从 DS5 构建 disease_key → [symptoms] 映射。
    DS5 中的 Disease 名称需要匹配到 disease_keys。
    """
    # 先建立 DS5 中每种疾病的症状集合（去重取并集）
    ds5_disease_symptoms = {}
    for _, row in ds5.iterrows():
        disease = str(row.get("Disease", "")).strip()
        if not disease or disease.lower() == "nan":
            continue
        symptoms = []
        for i in range(1, 18):
            col = f"Symptom_{i}"
            if col in row.index:
                s = str(row[col]).strip()
                if s and s.lower() != "nan":
                    symptoms.append(s.strip())
        if disease not in ds5_disease_symptoms:
            ds5_disease_symptoms[disease] = set()
        ds5_disease_symptoms[disease].update(symptoms)

    # 将 DS5 疾病名匹配到 disease_keys
    key_symptom_map = {}
    for ds5_disease, symptoms in ds5_disease_symptoms.items():
        matched_key = match_condition_to_disease_keys(ds5_disease, disease_keys)
        if matched_key and matched_key != "others":
            if matched_key not in key_symptom_map:
                key_symptom_map[matched_key] = set()
            key_symptom_map[matched_key].update(symptoms)

    # 转为排序列表
    for k in key_symptom_map:
        key_symptom_map[k] = sorted(key_symptom_map[k])

    print(f"[INFO] 疾病→症状映射: {len(key_symptom_map)} 个 disease_key 有症状数据")
    for k in sorted(key_symptom_map.keys())[:5]:
        print(f"       {k}: {key_symptom_map[k][:5]}...")

    return key_symptom_map


def build_symptom_severity_map(ds8):
    """
    从 DS8 构建 symptom → weight 映射
    """
    severity_map = {}
    for _, row in ds8.iterrows():
        symptom = str(row.get("Symptom", "")).strip()
        weight = row.get("weight", 1)
        if symptom and symptom.lower() != "nan":
            try:
                severity_map[symptom] = int(weight)
            except (ValueError, TypeError):
                severity_map[symptom] = 1
    print(f"[INFO] 症状严重度映射: {len(severity_map)} 个症状有权重")
    return severity_map


def do_disease_matching(drug_cond_map, disease_keys):
    """
    对每个药物的 original_conditions 执行模糊匹配，生成 matched_disease_keys。
    """
    matched_results = []
    match_stats = {"exact": 0, "synonym": 0, "contains": 0, "fuzzy": 0, "others": 0}

    for _, row in drug_cond_map.iterrows():
        drug = row["drug_name"]
        conditions = row["original_conditions"]
        matched_keys = set()

        for cond in conditions:
            result = match_condition_to_disease_keys(cond, disease_keys)
            if result:
                matched_keys.add(result)

        if not matched_keys:
            matched_keys.add("others")
            match_stats["others"] += 1
        else:
            match_stats["exact"] += 1  # 简化统计

        matched_results.append({
            "drug_name": drug,
            "matched_disease_keys": json.dumps(sorted(matched_keys), ensure_ascii=False),
        })

    matched_df = pd.DataFrame(matched_results)

    # 统计
    total = len(matched_df)
    others_count = matched_df["matched_disease_keys"].apply(lambda x: x == '["others"]').sum()
    print(f"[INFO] 疾病匹配完成:")
    print(f"       总药物数: {total}")
    print(f"       匹配到 disease_keys 的: {total - others_count} ({(total - others_count) / total * 100:.1f}%)")
    print(f"       归入 others 的: {others_count} ({others_count / total * 100:.1f}%)")

    return matched_df


def main():
    print("=" * 60)
    print("构建增强药物表 enhanced_drug_table.csv")
    print("=" * 60)

    # Step 1: 加载数据
    print("\n--- Step 1: 加载数据源 ---")
    disease_keys = load_disease_keys()
    ds1 = load_ds1()
    ds2 = load_ds2()
    ds3 = load_ds3()
    ds4 = load_ds4()
    ds5 = load_ds5()
    ds6 = load_ds6()
    ds8 = load_ds8()

    # Step 2: 构建药物→适应症映射
    print("\n--- Step 2: 构建药物→适应症映射 ---")
    drug_cond_map = build_drug_conditions_map(ds1, ds2, ds4)

    # Step 3: 构建药物属性表
    print("\n--- Step 3: 构建药物属性表 ---")
    drug_attrs = build_drug_attributes(ds2, ds3)

    # Step 4: 聚合评分
    print("\n--- Step 4: 聚合评分 ---")
    ratings = build_ratings(ds2, ds4)

    # Step 5: 模糊匹配 disease_keys
    print("\n--- Step 5: 模糊匹配 disease_keys ---")
    matched_df = do_disease_matching(drug_cond_map, disease_keys)

    # Step 6: 构建疾病描述映射
    print("\n--- Step 6: 构建疾病描述映射 ---")
    desc_map = build_disease_description_map(ds6, disease_keys)

    # Step 6.5: 构建疾病→症状映射 和 症状严重度映射
    print("\n--- Step 6.5: 构建疾病→症状映射 ---")
    disease_symptom_map = build_disease_symptom_map(ds5, disease_keys)
    symptom_severity_map = build_symptom_severity_map(ds8)

    # Step 7: 合并所有数据
    print("\n--- Step 7: 合并所有数据 ---")

    # 从 drug_cond_map 中获取 original_conditions（转为JSON字符串）
    drug_cond_map["original_conditions"] = drug_cond_map["original_conditions"].apply(
        lambda x: json.dumps(x, ensure_ascii=False)
    )

    # 合并: attrs + conditions + matched_keys + ratings
    final = drug_attrs.merge(
        drug_cond_map[["drug_name", "original_conditions"]],
        on="drug_name", how="outer"
    )
    final = final.merge(matched_df, on="drug_name", how="left")
    final = final.merge(ratings[["drug_name", "avg_rating", "total_reviews"]], on="drug_name", how="left")

    # 补充 disease_description
    def get_disease_desc(matched_keys_json):
        if not isinstance(matched_keys_json, str):
            return ""
        try:
            keys = json.loads(matched_keys_json)
        except:
            return ""
        descs = []
        for k in keys:
            d = desc_map.get(k, "")
            if d:
                descs.append(f"[{k}] {d}")
        return " | ".join(descs)

    final["disease_description"] = final["matched_disease_keys"].apply(get_disease_desc)

    # 补充 matched_symptoms: 通过 matched_disease_keys → DS5 反推症状
    def get_matched_symptoms(matched_keys_json):
        if not isinstance(matched_keys_json, str):
            return "[]"
        try:
            keys = json.loads(matched_keys_json)
        except Exception:
            return "[]"
        all_symptoms = set()
        for k in keys:
            symptoms = disease_symptom_map.get(k, [])
            all_symptoms.update(symptoms)
        return json.dumps(sorted(all_symptoms), ensure_ascii=False)

    final["matched_symptoms"] = final["matched_disease_keys"].apply(get_matched_symptoms)

    # 补充 symptom_severity: 对 matched_symptoms 中的每个症状附加权重
    def get_symptom_severity(symptoms_json):
        if not isinstance(symptoms_json, str):
            return "{}"
        try:
            symptoms = json.loads(symptoms_json)
        except Exception:
            return "{}"
        severity = {}
        for s in symptoms:
            w = symptom_severity_map.get(s, None)
            # 尝试去首尾空格后再匹配
            if w is None:
                w = symptom_severity_map.get(s.strip(), None)
            if w is not None:
                severity[s] = w
        if not severity:
            return "{}"
        return json.dumps(severity, ensure_ascii=False)

    final["symptom_severity"] = final["matched_symptoms"].apply(get_symptom_severity)

    # 填充缺失的 matched_disease_keys
    final["matched_disease_keys"] = final["matched_disease_keys"].fillna('["others"]')
    final["original_conditions"] = final["original_conditions"].fillna("[]")

    # 替换 NaN 字符串
    str_cols = [
        "generic_name", "drug_classes", "brand_names", "side_effects",
        "activity", "rx_otc", "pregnancy_category", "csa", "alcohol",
        "related_drugs", "medical_condition_description", "disease_description"
    ]
    for col in str_cols:
        if col in final.columns:
            final[col] = final[col].fillna("").astype(str)
            final[col] = final[col].replace("nan", "")

    # 排序输出列
    output_cols = [
        "drug_name", "generic_name", "drug_classes", "brand_names",
        "side_effects", "activity", "rx_otc", "pregnancy_category",
        "csa", "alcohol", "related_drugs",
        "avg_rating", "total_reviews",
        "original_conditions", "medical_condition_description",
        "matched_disease_keys", "matched_symptoms", "symptom_severity",
        "disease_description"
    ]
    output_cols = [c for c in output_cols if c in final.columns]
    final = final[output_cols]

    # 按药名排序
    final = final.sort_values("drug_name").reset_index(drop=True)

    # 验证
    print(f"\n--- 数据验证 ---")
    print(f"总行数（药物数）: {len(final)}")
    print(f"drug_name 唯一性: {'PASS' if final['drug_name'].is_unique else 'FAIL - 存在重复'}")

    # 如果有重复，做最终去重
    if not final["drug_name"].is_unique:
        print("[WARN] 发现重复 drug_name，执行最终去重...")
        final = final.groupby("drug_name").first().reset_index()
        final = final.sort_values("drug_name").reset_index(drop=True)
        print(f"去重后: {len(final)} 种药物")

    if "avg_rating" in final.columns:
        valid_ratings = final["avg_rating"].dropna()
        print(f"有评分的药物: {len(valid_ratings)} ({len(valid_ratings)/len(final)*100:.1f}%)")
        if len(valid_ratings) > 0:
            print(f"评分范围: [{valid_ratings.min():.1f}, {valid_ratings.max():.1f}]")

    matched_count = final["matched_disease_keys"].apply(lambda x: x != '["others"]').sum()
    print(f"匹配到 disease_keys 的药物: {matched_count} ({matched_count/len(final)*100:.1f}%)")
    others_count = len(final) - matched_count
    print(f"归入 others 的药物: {others_count} ({others_count/len(final)*100:.1f}%)")

    # 统计各 disease_key 的覆盖药物数
    print(f"\n--- disease_key 覆盖统计 ---")
    key_counts = {}
    for _, row in final.iterrows():
        try:
            keys = json.loads(row["matched_disease_keys"])
        except:
            keys = ["others"]
        for k in keys:
            key_counts[k] = key_counts.get(k, 0) + 1

    for k in sorted(key_counts.keys(), key=lambda x: key_counts[x], reverse=True)[:20]:
        print(f"  {k}: {key_counts[k]} 种药物")
    if len(key_counts) > 20:
        print(f"  ... 共 {len(key_counts)} 个 disease_key 有覆盖")

    # Step 8: 保存
    output_path = OUTPUT_DIR / "enhanced_drug_table.csv"
    final.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] 增强药物表已保存至: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"   总药物数: {len(final)}")
    print(f"   总字段数: {len(final.columns)}")
    print(f"   字段列表: {list(final.columns)}")


if __name__ == "__main__":
    main()
