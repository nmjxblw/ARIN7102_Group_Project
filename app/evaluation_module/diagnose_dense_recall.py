#!/usr/bin/env python3
"""
Dense Recall Diagnosis — three views on the existing 3D embedding.

Follows EXP_DRUG_RECALL_ASIDE_PLAN.md exactly:
  - Input: drug_comprehensive_embeddings.npy (root), enhanced_drug_table_v1_structured.csv,
           eval_dataset_verified.json (all 191 queries)
  - Views: view0 = emb[:,0,:], view1 = emb[:,1,:], mean = (emb[:,0,:]+emb[:,1,:])/2
  - Output: artifacts/exp_drug_recall/dense_diagnosis.json
            artifacts/exp_drug_recall/dense_diagnosis.md
  - Conclusions by fixed rules (no hardcoded text)
  - No model download, no production path changes
"""
import os
import sys
import json
import math
import importlib.util
from pathlib import Path

APP_ROOT = Path.cwd()
OUT_DIR  = APP_ROOT / "artifacts" / "exp_drug_recall"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(APP_ROOT / "app"))

import numpy as np
import pandas as pd

# ── Load data ─────────────────────────────────────────────────────────────────
EMBED_PATH = APP_ROOT / "drug_comprehensive_embeddings.npy"
DRUG_CSV   = APP_ROOT / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1_structured.csv"
EVAL_JSON  = APP_ROOT / "data" / "eval_dataset_verified.json"

embed_3d = np.load(EMBED_PATH)            # (5595, 2, 768)
df_drugs = pd.read_csv(DRUG_CSV)
queries  = json.load(open(EVAL_JSON))     # all 191

print(f"Embedding: {embed_3d.shape}  (3D — two views stored)")
print(f"Drug table: {len(df_drugs)} rows")
print(f"Queries: {len(queries)}")

# ── Build three views ─────────────────────────────────────────────────────────
v0   = embed_3d[:, 0, :]                   # (5595, 768)
v1   = embed_3d[:, 1, :]                   # (5595, 768)
vmean = (embed_3d[:, 0, :] + embed_3d[:, 1, :]) / 2  # (5595, 768)

def norm(x):
    x = x.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return x / n

v0_n   = norm(v0)
v1_n   = norm(v1)
vmean_n = norm(vmean)

# drug_names aligned with embedding rows
drug_names = df_drugs["drug_name"].fillna("").tolist()
assert len(drug_names) == embed_3d.shape[0], \
    f"Drug table rows {len(drug_names)} != embedding rows {embed_3d.shape[0]}"

# ── Encode queries ─────────────────────────────────────────────────────────────
# Force offline/local-only loading so this diagnosis never downloads models.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
print("\nLoading DrugEmbeddingEngine...")
engine_module_path = APP_ROOT / "app" / "embedded_module" / "drug_embedding_engine.py"
spec = importlib.util.spec_from_file_location("drug_embedding_engine_local", engine_module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load DrugEmbeddingEngine from {engine_module_path}")
engine_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_module)
DrugEmbeddingEngine = engine_module.DrugEmbeddingEngine
try:
    engine = DrugEmbeddingEngine(local_files_only=True)
except OSError as exc:
    raise SystemExit(
        "Local model files are unavailable. Dense diagnosis is configured as a no-download pass; "
        "cache the embedding model locally before running this script."
    ) from None

query_texts = []
for q in queries:
    disease_names = [d["name"] for d in q.get("diseases", [])]
    symptom_names = [s["name"] for s in q.get("symptoms", [])]
    query_text = f"{q['symptom_text']} {' '.join(disease_names)} {' '.join(symptom_names)}".strip()
    query_texts.append(query_text)

# Batch encode all 191 queries at once
print(f"Encoding {len(query_texts)} queries...")
query_vectors = engine.encode(query_texts)   # (191, 768)
print(f"Query vectors: {query_vectors.shape}")

# ── Semantic recall for each view ─────────────────────────────────────────────
def cosine_topk(qvecs, doc_matrix, k=20):
    """qvecs: (N, D), doc_matrix: (M, D) -> (N, k) top-k indices"""
    scores = np.dot(qvecs, doc_matrix.T)   # (N, M)
    topk   = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    return topk, scores

views = {
    "view0":   (v0_n,    "emb[:, 0, :]  — drug_recall_index.py path"),
    "view1":   (v1_n,    "emb[:, 1, :]  — dual_recall_pipeline.py path (CURRENT)"),
    "vmean":   (vmean_n, "(emb[:,0,:]+emb[:,1,:])/2  — mean pooling"),
}

results = {}
per_query_data = {}   # qid -> {view: {top20, hit20, recall20, mrr}}

print("\nRunning semantic recall for each view...")
for view_name, (doc_n, note) in views.items():
    print(f"\n--- {view_name} ---")
    topk, scores = cosine_topk(query_vectors, doc_n, k=20)

    hit20_list, recall20_list, mrr_list = [], [], []
    query_results = []

    for i, q in enumerate(queries):
        qid       = q["query_id"]
        relevant  = set(q["relevant_drugs"])
        topk_idx  = topk[i]
        topk_names= [drug_names[idx] for idx in topk_idx]
        topk_scores = scores[i, topk_idx]

        hits   = len(set(topk_names) & relevant)
        recall = hits / max(len(relevant), 1)

        # MRR
        first_hit = None
        for rank, idx in enumerate(topk_idx, 1):
            if drug_names[idx] in relevant:
                first_hit = rank
                break
        mrr = 1.0 / first_hit if first_hit else 0.0

        hit20_list.append(1.0 if hits > 0 else 0.0)
        recall20_list.append(recall)
        mrr_list.append(mrr)

        qr = {
            "query_id":      qid,
            "top20_drugs":   topk_names,
            "top20_scores":  [float(s) for s in topk_scores],
            "hit@20":        1.0 if hits > 0 else 0.0,
            "recall@20":     float(recall),
            "mrr":           float(mrr),
            "relevant_count": len(relevant),
            "hit_count":     hits,
        }
        query_results.append(qr)

    avg_hit20    = float(np.mean(hit20_list))
    avg_recall20 = float(np.mean(recall20_list))
    avg_mrr      = float(np.mean(mrr_list))
    hit_queries  = sum(hit20_list)
    miss_queries = [qr for qr in query_results if qr["hit@20"] == 0.0]

    results[view_name] = {
        "note":           note,
        "hit@20":         avg_hit20,
        "recall@20":      avg_recall20,
        "mrr":            avg_mrr,
        "hit_query_count": int(hit_queries),
        "miss_query_count": int(len(miss_queries)),
        "total_queries":  len(queries),
    }
    per_query_data[view_name] = query_results

    print(f"  hit@20    : {avg_hit20:.4f}  ({int(hit_queries)}/{len(queries)} queries)")
    print(f"  recall@20 : {avg_recall20:.4f}")
    print(f"  mrr       : {avg_mrr:.4f}")
    print(f"  miss queries: {len(miss_queries)}")
    if miss_queries:
        for qr in miss_queries[:3]:
            print(f"    FAIL {qr['query_id']}: top1={qr['top20_drugs'][0] if qr['top20_drugs'] else 'NONE'}  "
                  f"relevant={qr['relevant_count']}")

# ── Apply fixed conclusion rules ───────────────────────────────────────────────
all_hit20 = {vn: r["hit@20"] for vn, r in results.items()}
best_view = max(all_hit20, key=all_hit20.get)
best_hit  = all_hit20[best_view]
worst_hit = min(all_hit20.values())
all_zero  = all(v == 0.0 for v in all_hit20.values())
any_nonzero = any(v > 0.0 for v in all_hit20.values())

conclusions = []

if all_zero:
    conclusions.append({
        "rule": "all_views_zero",
        "finding": "All three variants (view0, view1, vmean) yield hit@20 = 0. "
                   "Problem is not view selection — the issue is upstream in embedding generation or alignment.",
        "evidence": {vn: round(v, 4) for vn, v in all_hit20.items()},
    })
elif best_hit < 0.05:
    conclusions.append({
        "rule": "all_views_near_zero",
        "finding": f"All views are near zero (best={best_view} hit@20={best_hit:.4f}). "
                   "Problem is not merely view selection — likely upstream embedding or alignment issue.",
        "evidence": {vn: round(v, 4) for vn, v in all_hit20.items()},
    })
else:
    conclusions.append({
        "rule": "view_selection_matters",
        "finding": f"View {best_view} performs best (hit@20={best_hit:.4f}). "
                   "Consider fixing the view/projection strategy first.",
        "evidence": {vn: round(v, 4) for vn, v in all_hit20.items()},
    })

# Additional nuance: compare view0 vs view1
v0_hit = all_hit20.get("view0", 0.0)
v1_hit = all_hit20.get("view1", 0.0)
if abs(v0_hit - v1_hit) > 0.02:
    better = "view0" if v0_hit > v1_hit else "view1"
    conclusions.append({
        "rule": "view0_vs_view1",
        "finding": f"{better} outperforms the other by {abs(v0_hit-v1_hit):.4f} hit@20. "
                   f"Verify which projection is used by the production pipeline.",
        "evidence": {"view0_hit@20": round(v0_hit, 4), "view1_hit@20": round(v1_hit, 4)},
    })

# ── Write dense_diagnosis.json ────────────────────────────────────────────────
output_json = {
    "embedding_shape": list(embed_3d.shape),
    "embedding_path": str(EMBED_PATH),
    "query_count": len(queries),
    "views_tested": list(views.keys()),
    "results": {
        vn: {k: v for k, v in r.items() if k != "note"}
        for vn, r in results.items()
    },
    "per_query": per_query_data,
    "conclusions": conclusions,
}

with open(OUT_DIR / "dense_diagnosis.json", "w") as f:
    json.dump(output_json, f, indent=2, ensure_ascii=False)

# ── Write dense_diagnosis.md ─────────────────────────────────────────────────
md_lines = [
    "# Dense Recall Diagnosis\n\n",
    f"**Embedding**: `{EMBED_PATH}`  shape={embed_3d.shape}\n",
    f"**Query count**: {len(queries)}  (all of `eval_dataset_verified.json`)\n\n",
    "## Per-View Results\n\n",
    f"| View | hit@20 | recall@20 | mrr | hit queries | miss queries |\n",
    f"|---|---|---|---|---|---|\n",
]

for vn, r in results.items():
    flag = " ◄ BEST" if vn == best_view else (" ◄ CURRENT" if vn == "view1" else "")
    md_lines.append(
        f"| `{vn}` {flag} | {r['hit@20']:.4f} | {r['recall@20']:.4f} | "
        f"{r['mrr']:.4f} | {r['hit_query_count']} | {r['miss_query_count']} |\n"
    )

md_lines.append("\n### Notes\n")
for vn, r in results.items():
    md_lines.append(f"- **{vn}**: {r.get('note', '')}\n")

md_lines.append("\n## Conclusions (by plan rules)\n")
for c in conclusions:
    md_lines.append(f"### Rule: `{c['rule']}`\n")
    md_lines.append(f"{c['finding']}\n\n")
    md_lines.append(f"Evidence:\n```json\n{json.dumps(c['evidence'], indent=2)}\n```\n\n")

# Sample failure queries per view
md_lines.append("## Sample Failure Queries (per view)\n")
for vn in views:
    misses = [qr for qr in per_query_data[vn] if qr["hit@20"] == 0.0]
    if misses:
        md_lines.append(f"\n### {vn} failures ({len(misses)} queries)\n")
        for qr in misses[:5]:
            q = next((q0 for q0 in queries if q0["query_id"] == qr["query_id"]), {})
            top1_score = f"{qr['top20_scores'][0]:.4f}" if qr['top20_scores'] else "N/A"
            md_lines.append(
                f"- `{qr['query_id']}`  top1=`{qr['top20_drugs'][0] if qr['top20_drugs'] else 'NONE'}`  "
                f"score={top1_score}  "
                f"relevant={qr['relevant_count']}  "
                f"text=\"{q.get('symptom_text','')[:50]}\"\n"
            )

with open(OUT_DIR / "dense_diagnosis.md", "w") as f:
    f.writelines(md_lines)

print(f"\nWritten:")
print(f"  {OUT_DIR / 'dense_diagnosis.json'}")
print(f"  {OUT_DIR / 'dense_diagnosis.md'}")
