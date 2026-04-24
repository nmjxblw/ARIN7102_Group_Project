"""
向量匹配模块（Embedding Module）单元测试脚本
===============================================
测试内容：
  1. DrugEmbeddingEngine 编码正确性
  2. 语义文本构建（build_semantic_text）
  3. 余弦相似度计算（safe_cosine_scores）
  4. 语义召回（semantic_recall）质量诊断
  5. Query 构造方式对比实验
  6. Pooling 策略对比实验（CLS vs Mean）
  7. max_length 截断影响分析

运行方式:
  python test_embedding_module.py              # 跑全部测试
  python test_embedding_module.py --test 1     # 只跑测试1
  python test_embedding_module.py --test 5     # 只跑Query对比实验
"""
import sys
import os
import time
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ─── 环境初始化 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(APP_ROOT)

# HuggingFace 镜像 + 离线优先
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "pipeline_config.env", override=False)
load_dotenv(APP_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "dummy"

# ─── 延迟导入（测试按需加载）───────────────────────────────────
def get_engine_and_data():
    """加载模型和数据（lazy，仅在需要时调用）"""
    from embedded_module.drug_embedding_engine import DrugEmbeddingEngine
    from embedded_module.recommendation_pipeline import prepare_drug_dataframe, build_semantic_text, safe_cosine_scores

    # 加载数据
    csv_path = PROJECT_ROOT / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1_structured.csv"
    emb_path = PROJECT_ROOT / "drug_comprehensive_embeddings.npy"

    df = pd.read_csv(csv_path)
    embeddings = np.load(str(emb_path))
    print(f"[数据] 药物表: {df.shape[0]} 条, 向量: {embeddings.shape}")

    # 加载 embedding engine
    engine = DrugEmbeddingEngine()
    return engine, df, embeddings


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════
# TEST 1: 基础编码正确性
# ══════════════════════════════════════════════════════════════
def test_1_encoding_basic(engine, df, embeddings):
    separator("TEST 1: 基础编码正确性")

    # 1.1 编码维度验证
    test_texts = [
        "headache and nausea",
        "Drug name: aspirin. Related diseases: pain.",
        "",  # empty
    ]
    vecs = engine.encode(test_texts)
    print(f"输入 {len(test_texts)} 条文本 → 输出 shape: {vecs.shape}")
    assert vecs.shape == (3, 768), f"期望 (3,768), 得到 {vecs.shape}"
    print("  ✓ 维度正确 (N, 768)")

    # 1.2 向量非零
    norms = np.linalg.norm(vecs, axis=1)
    print(f"  向量范数: {norms}")
    assert all(n > 0 for n in norms), "存在零范数向量"
    print("  ✓ 所有向量非零")

    # 1.3 编码一致性（同一文本两次编码应相同）
    vec_a = engine.encode_query("test query")
    vec_b = engine.encode_query("test query")
    diff = np.abs(vec_a - vec_b).max()
    print(f"  相同文本两次编码差异: {diff:.2e}")
    assert diff < 1e-5, f"编码不一致: diff={diff}"
    print("  ✓ 编码一致性通过")

    # 1.4 预计算向量与实时编码对比
    sample_idx = 0
    sample_text = df.iloc[sample_idx]["semantic_text"]
    live_vec = engine.encode_query(sample_text)
    stored_vec = embeddings[sample_idx:sample_idx+1]
    cosine = float(np.dot(live_vec.flatten(), stored_vec.flatten()) /
                   (np.linalg.norm(live_vec) * np.linalg.norm(stored_vec)))
    print(f"  第0条药物: 实时编码 vs 预存向量 余弦相似度 = {cosine:.6f}")
    if cosine > 0.99:
        print("  ✓ 预计算向量与实时编码高度一致")
    else:
        print(f"  ⚠ 预计算向量与实时编码偏差较大 (cosine={cosine:.4f})")
    print()


# ══════════════════════════════════════════════════════════════
# TEST 2: semantic_text 构建检查
# ══════════════════════════════════════════════════════════════
def test_2_semantic_text(engine, df, embeddings):
    separator("TEST 2: semantic_text 构建检查")

    from embedded_module.recommendation_pipeline import build_semantic_text
    from transformers import AutoTokenizer
    tokenizer = engine.tokenizer

    # 统计 semantic_text 长度分布
    texts = df["semantic_text"].tolist()
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in texts[:100]]  # 抽样100条

    print(f"前100条 semantic_text token数统计:")
    print(f"  min: {min(lengths)}, max: {max(lengths)}, mean: {np.mean(lengths):.1f}, median: {np.median(lengths):.1f}")
    over_256 = sum(1 for l in lengths if l > 256)
    over_512 = sum(1 for l in lengths if l > 512)
    print(f"  超过 256 tokens: {over_256}/100 ({over_256}%)")
    print(f"  超过 512 tokens: {over_512}/100 ({over_512}%)")

    # 展示一条典型 semantic_text
    acne_drugs = df[df["semantic_text"].str.contains("acne", na=False)]
    if len(acne_drugs) > 0:
        sample = acne_drugs.iloc[0]
        text = sample["semantic_text"]
        tokens = tokenizer.encode(text, truncation=False)
        print(f"\n  示例 (acne 药物 '{sample['drug_name']}'):")
        print(f"  全文 token 数: {len(tokens)}")
        print(f"  截断到 256 后丢失: {max(0, len(tokens)-256)} tokens ({max(0, len(tokens)-256)/len(tokens)*100:.1f}%)")
        print(f"  文本前200字符: {text[:200]}...")
    print()


# ══════════════════════════════════════════════════════════════
# TEST 3: 200-case 多维度召回质量评估 (eval_dataset 标签)
# ══════════════════════════════════════════════════════════════
def test_3_cosine_distribution(engine, df, embeddings):
    separator("TEST 3: 200-case 拼接方式对比评估 (eval_dataset 标签)")

    from collections import defaultdict
    from embedded_module.recommendation_pipeline import safe_cosine_scores

    def compute_scores(emb, qv):
        """对 2D 或 3D (legacy) embeddings 计算相似度."""
        if emb.ndim == 3:
            return safe_cosine_scores(emb[:, 1, :], qv)  # View 1 only
        return safe_cosine_scores(emb, qv)

    def find_rank(scores, drug_name):
        """找到指定药物在排名中的位置."""
        idx = df[df["drug_name"].str.contains(drug_name, case=False, na=False)].index
        if len(idx) == 0:
            return None, None
        best_score = scores[idx].max()
        return int((scores > best_score).sum()) + 1, best_score

    def _label_key(d):
        return d.get("label") or d.get("name", "")

    # ── 1. 加载 eval_dataset_llm_v2.json ──
    eval_path = PROJECT_ROOT / "app" / "dataset_module" / "drugs_training_dataset" / "eval_dataset_llm_v2.json"
    print(f"  加载评估数据集: {eval_path.name}")
    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"  数据集总量: {len(eval_data)} 条")

    # ── 2. 按疾病分组 (排除 others) ──
    disease_entries = defaultdict(list)
    for e in eval_data:
        for d in e.get("diseases", []):
            lbl = _label_key(d)
            if lbl and lbl != "others":
                disease_entries[lbl].append(e)

    # ── 3. 构建 disease → 药物索引映射 ──
    disease_drug_map = {}
    for i, row in df.iterrows():
        try:
            keys = json.loads(str(row.get("matched_disease_keys", "[]")))
        except Exception:
            continue
        for k in keys:
            disease_drug_map.setdefault(k, []).append(i)

    # ── 4. 每个疾病选 1 条代表 (症状最多 & 置信度 > 0.5) ──
    test_diseases = sorted(disease_entries.keys())
    if len(test_diseases) > 40:
        test_diseases = test_diseases[:40]

    n_drugs = df.shape[0]
    print(f"  可用疾病: {len(test_diseases)}, 药物总数: {n_drugs}, embedding shape: {embeddings.shape}")
    print(f"  测试: {len(test_diseases)} 疾病 × 6 拼接方式 = {len(test_diseases)*6} case")
    print(f"  标签来源: eval_dataset_llm_v2.json (LLM 生成标签)\n")

    hdr = f"  {'疾病':<38s} {'目标药物':<22s} {'纯口语':>5s} {'口+病':>5s} {'口+症':>5s} {'口+全':>5s} {'纯标签':>5s} {'纯病名':>5s}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    summary_rows = []
    skipped = []

    for disease_key in test_diseases:
        # 找目标药物 (该疾病下评论数最多的药物)
        drug_indices = disease_drug_map.get(disease_key, [])
        if not drug_indices:
            skipped.append(disease_key)
            continue
        sub = df.iloc[drug_indices].copy()
        sub["_reviews"] = pd.to_numeric(sub["total_reviews"], errors="coerce").fillna(0)
        target_drug = df.loc[sub["_reviews"].idxmax(), "drug_name"]

        # 选代表条目: 优先选症状多的
        entries = disease_entries[disease_key]
        scored = []
        for e in entries:
            n_sym = len(e.get("symptoms", []))
            d_conf = max((d.get("confidence", 0) for d in e["diseases"] if _label_key(d) == disease_key), default=0)
            scored.append((n_sym, d_conf, e))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_entry = scored[0][2]

        oral_query = best_entry["sentence"]
        pred_diseases = best_entry.get("diseases", [])
        pred_symptoms = best_entry.get("symptoms", [])

        # 构建标签文本片段
        d_names = [_label_key(d).replace("_", " ") for d in pred_diseases if _label_key(d)]
        s_names = [_label_key(s).replace("_", " ") for s in pred_symptoms[:8] if _label_key(s)]
        disease_text = f"Diseases: {', '.join(d_names)}." if d_names else ""
        symptom_text = f"Symptoms: {', '.join(s_names)}." if s_names else ""
        full_label = f"{disease_text} {symptom_text}".strip()

        # 6 种拼接方式
        queries = [
            ("纯口语", oral_query),
            ("口+病", f"{oral_query} {disease_text}".strip()),
            ("口+症", f"{oral_query} {symptom_text}".strip()),
            ("口+全", f"{oral_query} {full_label}".strip()),
            ("纯标签", full_label or "[MASK]"),
            ("纯病名", disease_key.replace("_", " ")),
        ]

        ranks_str = []
        for qtype, query in queries:
            qv = engine.encode_query(query)
            sc = compute_scores(embeddings, qv)
            rank, score = find_rank(sc, target_drug)
            summary_rows.append({
                "disease": disease_key, "target": target_drug,
                "type": qtype, "rank": rank, "score": score,
            })
            ranks_str.append(f"{rank:>5d}" if rank else "  N/A")

        print(f"  {disease_key:<38s} {target_drug:<22s} {''.join(ranks_str)}")

    if skipped:
        print(f"\n  ⚠ 跳过 {len(skipped)} 个无匹配药物的疾病: {skipped}")

    # ── 汇总统计 ──
    valid_count = len(test_diseases) - len(skipped)
    print(f"\n  {'='*70}")
    print(f"  汇总 ({valid_count} 疾病 × 6 拼接方式 = {valid_count * 6} case)")
    print(f"  {'='*70}")
    print(f"  {'拼接方式':<10s} {'平均排名':>8s} {'中位排名':>8s} {'R@100':>7s} {'R@200':>7s} {'R@500':>7s}")
    print(f"  {'-'*55}")
    for qtype in ["纯口语", "口+病", "口+症", "口+全", "纯标签", "纯病名"]:
        rows = [r for r in summary_rows if r["type"] == qtype and r["rank"] is not None]
        if not rows:
            continue
        rks = [r["rank"] for r in rows]
        avg_r, med_r = np.mean(rks), np.median(rks)
        r100 = sum(1 for r in rks if r <= 100) / len(rks) * 100
        r200 = sum(1 for r in rks if r <= 200) / len(rks) * 100
        r500 = sum(1 for r in rks if r <= 500) / len(rks) * 100
        print(f"  {qtype:<10s} {avg_r:>7.0f}   {med_r:>7.0f}   {r100:>5.1f}%  {r200:>5.1f}%  {r500:>5.1f}%")
    print()


# ══════════════════════════════════════════════════════════════
# TEST 4: semantic_recall 端到端质量
# ══════════════════════════════════════════════════════════════
def test_4_semantic_recall_quality(engine, df, embeddings):
    separator("TEST 4: semantic_recall 端到端质量")

    from embedded_module.dual_recall_pipeline import DualRecallDrugRecommender
    from embedded_module.cross_encoder_reranker import CrossEncoderReranker

    # 创建轻量 pipeline（不需要 cross_encoder 做重排）
    cross_encoder = CrossEncoderReranker()
    pipeline = DualRecallDrugRecommender(df, embeddings, engine, cross_encoder)

    # 多条测试 case
    test_cases = [
        {
            "name": "acne case",
            "query": "I have big, painful bumps on my skin that leave scars.",
            "expected_drugs": ["doxycycline", "clindamycin", "tetracycline"],
        },
        {
            "name": "headache/migraine case",
            "query": "I have severe headaches with nausea and sensitivity to light.",
            "expected_drugs": ["sumatriptan", "ibuprofen", "excedrin"],
        },
        {
            "name": "diabetes case",
            "query": "My blood sugar is very high and I feel thirsty all the time.",
            "expected_drugs": ["metformin", "insulin", "glipizide"],
        },
    ]

    for case in test_cases:
        cands = pipeline.semantic_recall(case["query"], top_k=200)
        recalled_drugs = set(cands["drug_name"].str.lower())
        hits = [d for d in case["expected_drugs"] if d.lower() in recalled_drugs]
        print(f"  [{case['name']}]")
        print(f"    Query: {case['query'][:60]}...")
        print(f"    Recall@200: {len(hits)}/{len(case['expected_drugs'])} = {hits}")
        print(f"    Top-5 drugs: {cands['drug_name'].head(5).tolist()}")
        print(f"    Score range: [{cands['semantic_score_raw'].min():.6f}, {cands['semantic_score_raw'].max():.6f}]")
        print(f"    Score std: {cands['semantic_score_raw'].std():.6f}")
        print()


# ══════════════════════════════════════════════════════════════
# TEST 5: Query 构造方式对比实验
# ══════════════════════════════════════════════════════════════
def test_5_query_format_comparison(engine, df, embeddings):
    separator("TEST 5: Query 构造方式对比实验")

    from embedded_module.recommendation_pipeline import safe_cosine_scores

    # 同一语义意图，不同表达形式
    query_variants = {
        "用户原文": "I have big, painful bumps on my skin that leave scars. How can I treat this?",
        "纯标签（标签格式）": "Related diseases: acne. Target symptoms: nodal skin eruptions, scurring.",
        "原文+标签拼接": "I have big, painful bumps on my skin that leave scars. Related diseases: acne. Target symptoms: nodal skin eruptions, scurring.",
        "药名+疾病": "doxycycline acne treatment for skin eruptions",
        "模板化查询": "Drug for acne with symptoms: nodal skin eruptions, scurring, skin rash",
    }

    # Ground truth drugs
    gt_drugs = ["doxycycline", "clindamycin", "tetracycline"]
    gt_indices = []
    for drug in gt_drugs:
        idx = df[df["drug_name"].str.contains(drug, case=False, na=False)].index
        if len(idx) > 0:
            gt_indices.append((drug, idx[0]))

    print(f"  Ground truth drugs: {gt_drugs}")
    print(f"  Found indices: {gt_indices}\n")

    results = []
    for name, query in query_variants.items():
        query_vec = engine.encode_query(query)
        scores = safe_cosine_scores(embeddings, query_vec)

        # 计算 GT 药物的排名
        gt_ranks = []
        gt_scores_list = []
        for drug_name, idx in gt_indices:
            rank = int((scores > scores[idx]).sum()) + 1
            gt_ranks.append(rank)
            gt_scores_list.append(scores[idx])

        avg_rank = np.mean(gt_ranks)
        score_range = scores.max() - scores.min()

        results.append({
            "query_type": name,
            "avg_gt_rank": avg_rank,
            "gt_ranks": gt_ranks,
            "score_range": score_range,
            "score_std": scores.std(),
        })

        print(f"  [{name}]")
        print(f"    Query: \"{query[:70]}\"")
        print(f"    GT药物平均排名: {avg_rank:.0f} / {len(scores)}")
        print(f"    各药物排名: {dict(zip(gt_drugs, gt_ranks))}")
        print(f"    分数区分度 (range): {score_range:.6f}, std: {scores.std():.6f}")
        print()

    # 总结
    best = min(results, key=lambda x: x["avg_gt_rank"])
    worst = max(results, key=lambda x: x["avg_gt_rank"])
    print(f"  ★ 最佳 Query 格式: [{best['query_type']}] (GT平均排名 {best['avg_gt_rank']:.0f})")
    print(f"  ✗ 最差 Query 格式: [{worst['query_type']}] (GT平均排名 {worst['avg_gt_rank']:.0f})")
    print()


# ══════════════════════════════════════════════════════════════
# TEST 6: Pooling 策略对比（CLS vs Mean Pooling）
# ══════════════════════════════════════════════════════════════
def test_6_pooling_comparison(engine, df, embeddings):
    separator("TEST 6: Pooling 策略对比 (CLS vs Mean)")

    import torch
    from embedded_module.recommendation_pipeline import safe_cosine_scores

    query = "Related diseases: acne. Target symptoms: nodal skin eruptions, scurring."
    drug_texts = df["semantic_text"].head(200).tolist()  # 用前200条做对比

    # CLS pooling (当前方式)
    encoded = engine.tokenizer(
        [query], padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    encoded = {k: v.to(engine.device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = engine.model(**encoded)
    cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # Mean pooling
    attention_mask = encoded["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    mean_vec = (torch.sum(token_embeddings * input_mask_expanded, 1) /
                torch.clamp(input_mask_expanded.sum(1), min=1e-9)).cpu().numpy()

    # 对比两种 query 向量与药物的余弦相似度
    drug_vecs = embeddings[:200]  # 前200条药物的预存向量（CLS方式编码的）

    cls_scores = safe_cosine_scores(drug_vecs, cls_vec)
    mean_scores = safe_cosine_scores(drug_vecs, mean_vec)

    print(f"  Query: \"{query}\"")
    print(f"  对比范围: 前200条药物")
    print()
    print(f"  CLS pooling:")
    print(f"    分数范围: [{cls_scores.min():.6f}, {cls_scores.max():.6f}]")
    print(f"    std: {cls_scores.std():.6f}")
    print()
    print(f"  Mean pooling:")
    print(f"    分数范围: [{mean_scores.min():.6f}, {mean_scores.max():.6f}]")
    print(f"    std: {mean_scores.std():.6f}")
    print()

    # 检查排名差异
    acne_idx = df.head(200)[df.head(200)["semantic_text"].str.contains("acne", na=False)].index
    if len(acne_idx) > 0:
        best_acne_cls = cls_scores[acne_idx].max()
        best_acne_mean = mean_scores[acne_idx].max()
        rank_cls = int((cls_scores > best_acne_cls).sum()) + 1
        rank_mean = int((mean_scores > best_acne_mean).sum()) + 1
        print(f"  Acne相关药物 (在前200条中):")
        print(f"    CLS: 最高分={best_acne_cls:.6f}, 排名={rank_cls}")
        print(f"    Mean: 最高分={best_acne_mean:.6f}, 排名={rank_mean}")

    print()
    print("  注意: 药物侧向量也是CLS编码的，所以Mean pooling仅改变query侧。")
    print("  若要完整对比，需要重新编码所有药物（耗时较长）。")
    print()


# ══════════════════════════════════════════════════════════════
# TEST 7: max_length 截断影响分析
# ══════════════════════════════════════════════════════════════
def test_7_truncation_analysis(engine, df, embeddings):
    separator("TEST 7: max_length 截断影响分析")

    import torch

    # 选取一条长文本药物
    long_drugs = []
    for i, row in df.iterrows():
        text = str(row.get("semantic_text", ""))
        tokens = engine.tokenizer.encode(text, truncation=False)
        if len(tokens) > 300:
            long_drugs.append((i, row["drug_name"], len(tokens), text))
        if len(long_drugs) >= 3:
            break

    if not long_drugs:
        print("  未找到超过300 tokens的药物文本")
        return

    print(f"  找到 {len(long_drugs)} 条长文本药物:\n")

    for idx, drug_name, token_count, text in long_drugs:
        print(f"  [{drug_name}] ({token_count} tokens)")

        # 编码截断到256
        encoded_256 = engine.tokenizer(
            [text], padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        # 编码截断到512
        encoded_512 = engine.tokenizer(
            [text], padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        with torch.no_grad():
            encoded_256 = {k: v.to(engine.device) for k, v in encoded_256.items()}
            encoded_512 = {k: v.to(engine.device) for k, v in encoded_512.items()}
            out_256 = engine.model(**encoded_256)
            out_512 = engine.model(**encoded_512)

        vec_256 = out_256.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        vec_512 = out_512.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        stored_vec = embeddings[idx]

        # 比较
        cos_256_stored = float(np.dot(vec_256, stored_vec) / (np.linalg.norm(vec_256) * np.linalg.norm(stored_vec)))
        cos_512_256 = float(np.dot(vec_512, vec_256) / (np.linalg.norm(vec_512) * np.linalg.norm(vec_256)))

        print(f"    预存向量 vs 实时256: cosine = {cos_256_stored:.6f}")
        print(f"    256截断 vs 512截断: cosine = {cos_512_256:.6f}")
        print(f"    截断丢失 tokens: {token_count - 256}")

        # 看截断掉的内容
        full_tokens = engine.tokenizer.encode(text, truncation=False)
        truncated_text = engine.tokenizer.decode(full_tokens[256:], skip_special_tokens=True)
        print(f"    被截断的内容(前100字符): \"{truncated_text[:100]}...\"")
        print()


# ══════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════
ALL_TESTS = {
    1: ("基础编码正确性", test_1_encoding_basic),
    2: ("semantic_text 构建检查", test_2_semantic_text),
    3: ("200-case 召回质量评估 (eval_dataset 标签)", test_3_cosine_distribution),
    4: ("semantic_recall 端到端质量", test_4_semantic_recall_quality),
    5: ("Query 构造方式对比实验", test_5_query_format_comparison),
    6: ("Pooling 策略对比 (CLS vs Mean)", test_6_pooling_comparison),
    7: ("max_length 截断影响分析", test_7_truncation_analysis),
}


def main():
    parser = argparse.ArgumentParser(description="向量匹配模块单元测试")
    parser.add_argument("--test", type=int, nargs="*", help="指定运行的测试编号 (1-7), 不指定则全部运行")
    args = parser.parse_args()

    print("=" * 60)
    print("  向量匹配模块 (Embedding Module) 单元测试")
    print("=" * 60)

    # 加载模型和数据
    print("\n正在加载模型和数据...")
    t0 = time.time()
    engine, df, embeddings = get_engine_and_data()
    print(f"加载耗时: {time.time()-t0:.1f}s\n")

    # 确定要运行的测试
    test_ids = args.test if args.test else list(ALL_TESTS.keys())

    for tid in test_ids:
        if tid not in ALL_TESTS:
            print(f"\n⚠ 测试 {tid} 不存在，跳过")
            continue
        name, func = ALL_TESTS[tid]
        try:
            func(engine, df, embeddings)
        except Exception as e:
            print(f"\n  ✗ 测试 {tid} [{name}] 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
