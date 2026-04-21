"""召回通道对比测试：纯语义(oral) / 纯语义(enriched) / LLM扩展语义 / 纯标签 / 语义+标签

对比 5 种召回策略:
  1. 纯语义(oral)    - 仅用口语 sentence 做向量检索，不拼标签
  2. 纯语义(enriched) - 用 sentence+标签拼接后做向量检索，不走 label_recall
  3. LLM扩展语义     - enriched query 经 LLM 术语扩展后做向量检索
  4. 纯标签           - 仅用 label_recall 精确匹配，不走语义
  5. 语义+标签        - enriched 语义 + label_recall 加权融合 (生产默认)
"""
import sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(APP_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "pipeline_config.env", override=False)
load_dotenv(APP_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)
# 不覆盖 DEEPSEEK_API_KEY，使用 .env 中的真实 key

from embedded_module.drug_embedding_engine import DrugEmbeddingEngine
from embedded_module.recommendation_pipeline import prepare_drug_dataframe
from embedded_module.dual_recall_pipeline import (
    DualRecallDrugRecommender, _build_combined_query, parse_weighted_labels,
)
from embedded_module.cross_encoder_reranker import CrossEncoderReranker
from embedded_module.query_expander import expand_query_with_llm

# ── 加载数据 ──
csv_path = PROJECT_ROOT / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1_structured.csv"
emb_path = PROJECT_ROOT / "drug_comprehensive_embeddings.npy"
df_raw = pd.read_csv(csv_path)
embeddings = np.load(str(emb_path))

engine = DrugEmbeddingEngine()
df = prepare_drug_dataframe(df_raw)
print(f"药物: {len(df)}, embedding: {embeddings.shape}")

cross_encoder = CrossEncoderReranker()
pipeline = DualRecallDrugRecommender(df, embeddings, engine, cross_encoder)

# ── 验证 label 解析 ──
test_items = [{"label": "acne", "confidence": 0.9}]
parsed = parse_weighted_labels(test_items)
assert len(parsed) == 1, "label 解析失败!"
print(f"label 解析验证 OK: {[(p.name, p.confidence) for p in parsed]}\n")

# ── 加载 eval_dataset ──
eval_path = PROJECT_ROOT / "app" / "dataset_module" / "drugs_training_dataset" / "eval_dataset_llm_v2.json"
with open(eval_path, encoding="utf-8") as f:
    eval_data = json.load(f)

def _label_key(d):
    return d.get("label") or d.get("name", "")

disease_entries = defaultdict(list)
for e in eval_data:
    for d in e.get("diseases", []):
        lbl = _label_key(d)
        if lbl and lbl != "others":
            disease_entries[lbl].append(e)

disease_drug_map = {}
for i, row in df.iterrows():
    try:
        keys = json.loads(str(row.get("matched_disease_keys", "[]")))
    except Exception:
        continue
    for k in keys:
        disease_drug_map.setdefault(k, []).append(i)

test_diseases = sorted(disease_entries.keys())[:40]


def find_drug_rank(candidates_df, target_drug, sort_col="recall_fused_score"):
    if len(candidates_df) == 0:
        return 99999
    if sort_col in candidates_df.columns:
        ranked = candidates_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    else:
        ranked = candidates_df.reset_index(drop=True)
    for idx, row in ranked.iterrows():
        if target_drug.lower() in str(row["drug_name"]).lower():
            return idx + 1
    return 99999


def run_combo(pipeline, oral_query, disease_items, symptom_items,
              use_semantic=True, use_label=True, use_enriched_query=True,
              use_llm_expansion=False, top_k=200):
    """运行不同召回组合"""
    diseases_parsed = parse_weighted_labels(disease_items, normalize_symptom=False)
    symptoms_parsed = parse_weighted_labels(symptom_items, normalize_symptom=True)
    enriched_query = _build_combined_query(oral_query, diseases_parsed, symptoms_parsed)

    sem_cands = pd.DataFrame()
    label_cands = pd.DataFrame()

    if use_semantic:
        query = enriched_query if use_enriched_query else oral_query
        if use_llm_expansion:
            query = expand_query_with_llm(query)
        sem_cands = pipeline.semantic_recall(query, top_k=top_k)

    if use_label:
        label_cands = pipeline.label_recall(disease_items, symptom_items, top_k=top_k)

    # 权重
    if use_semantic and use_label:
        w_sem, w_label = 0.5, 0.5
    elif use_semantic:
        w_sem, w_label = 1.0, 0.0
    else:
        w_sem, w_label = 0.0, 1.0

    fused = pipeline.fuse_recalls(
        semantic_candidates=sem_cands,
        label_candidates=label_cands,
        semantic_weight=w_sem,
        label_weight=w_label,
        top_k=9999,
    )

    # 对纯语义通道，sort_col 是 semantic_score
    sort_col = "recall_fused_score" if "recall_fused_score" in fused.columns else "semantic_score"
    return fused, sort_col


# ── 定义 5 种组合 ──
combos = [
    # (名称, use_semantic, use_label, use_enriched_query, use_llm_expansion)
    ("纯语义(oral)",      True,  False, False, False),
    ("纯语义(enriched)",  True,  False, True,  False),
    ("LLM扩展语义",       True,  False, True,  True),
    ("纯标签",            False, True,  False, False),
    ("语义+标签",         True,  True,  True,  False),
]

summary = {name: [] for name, *_ in combos}
skipped = 0

hdr = f"  {'#':>3s} {'疾病':<36s} {'目标药物':<20s}"
for name, *_ in combos:
    hdr += f" {name:>16s}"
print(hdr)
print(f"  {'-'*len(hdr)}")

for idx, disease_key in enumerate(test_diseases, 1):
    drug_indices = disease_drug_map.get(disease_key, [])
    if not drug_indices:
        skipped += 1
        continue
    sub = df.iloc[drug_indices].copy()
    sub["_reviews"] = pd.to_numeric(sub["total_reviews"], errors="coerce").fillna(0)
    target_drug = df.loc[sub["_reviews"].idxmax(), "drug_name"]

    entries = disease_entries[disease_key]
    scored = []
    for e in entries:
        n_sym = len(e.get("symptoms", []))
        d_conf = max((d.get("confidence", 0) for d in e["diseases"] if _label_key(d) == disease_key), default=0)
        scored.append((n_sym, d_conf, e))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_entry = scored[0][2]

    oral_query = best_entry["sentence"]
    disease_items = best_entry.get("diseases", [])
    symptom_items = best_entry.get("symptoms", [])

    rank_strs = []
    for name, use_sem, use_lbl, use_enriched, use_llm in combos:
        fused, sort_col = run_combo(
            pipeline, oral_query, disease_items, symptom_items,
            use_semantic=use_sem, use_label=use_lbl, use_enriched_query=use_enriched,
            use_llm_expansion=use_llm,
        )
        rank = find_drug_rank(fused, target_drug, sort_col)
        summary[name].append(rank)
        rank_strs.append(f"{rank:>16d}")

    print(f"  {idx:>3d} {disease_key:<36s} {target_drug:<20s} {''.join(rank_strs)}")

# ── 汇总 ──
valid_count = len(test_diseases) - skipped
print(f"\n{'='*90}")
print(f"  召回通道对比 ({valid_count} 疾病)")
print(f"{'='*90}")
print(f"  {'组合':<18s} {'平均排名':>8s} {'中位排名':>8s} {'R@50':>7s} {'R@100':>7s} {'R@200':>7s} {'召回':>5s}")
print(f"  {'-'*70}")
for name, *_ in combos:
    rks = summary[name]
    if not rks:
        continue
    total = len(rks)
    found = [r for r in rks if r < 99999]
    avg_r = np.mean(found) if found else 99999
    med_r = np.median(found) if found else 99999
    r50 = sum(1 for r in rks if r <= 50) / total * 100
    r100 = sum(1 for r in rks if r <= 100) / total * 100
    r200 = sum(1 for r in rks if r <= 200) / total * 100
    found_pct = len(found) / total * 100
    print(f"  {name:<18s} {avg_r:>7.0f}   {med_r:>7.0f}   {r50:>5.1f}%  {r100:>5.1f}%  {r200:>5.1f}%  {found_pct:>4.0f}%")
