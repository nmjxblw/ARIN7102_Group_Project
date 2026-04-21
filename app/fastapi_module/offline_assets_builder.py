from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedded_module.drug_embedding_engine import DrugEmbeddingEngine
from embedded_module.recommendation_pipeline import prepare_drug_dataframe
from pipeline_config import cfg


def build_assets(
    input_csv: Path,
    output_structured_csv: Path,
    output_embeddings_npy: Path,
    model_name: str,
    force_rebuild_embeddings: bool,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_structured_csv.parent.mkdir(parents=True, exist_ok=True)
    output_embeddings_npy.parent.mkdir(parents=True, exist_ok=True)

    print(f"[offline] loading input csv: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[offline] input shape: {df.shape}")

    print("[offline] building structured dataframe...")
    structured_df = prepare_drug_dataframe(df)
    structured_df.to_csv(output_structured_csv, index=False, encoding="utf-8")
    print(f"[offline] structured dataset saved: {output_structured_csv}")

    if output_embeddings_npy.exists() and not force_rebuild_embeddings:
        print(f"[offline] embeddings already exist, skip build: {output_embeddings_npy}")
        return

    print(f"[offline] building embeddings with model: {model_name}")
    engine = DrugEmbeddingEngine(model_name=model_name)

    import numpy as np

    # 只编码 View 1 (disease_description), 输出 (N, 768)
    texts_dd = structured_df["semantic_text_dd"].tolist()
    n_drugs = len(structured_df)

    # ── 去重 + 按长度排序编码 ──
    unique_texts: dict[str, int] = {}  # text -> index in unique list
    text_idx: list[int] = []

    for text in texts_dd:
        if text not in unique_texts:
            unique_texts[text] = len(unique_texts)
        text_idx.append(unique_texts[text])

    unique_list = [""] * len(unique_texts)
    for text, idx in unique_texts.items():
        unique_list[idx] = text

    dedup_count = len(unique_list)
    saved_pct = (1 - dedup_count / n_drugs) * 100
    print(f"[offline] 去重: {n_drugs} 条文本 -> {dedup_count} 条唯一文本 (节省 {saved_pct:.1f}%)")

    # 按 token 长度排序，让短文本 batch 快速通过
    sort_keys = sorted(range(dedup_count), key=lambda k: len(unique_list[k]))
    sorted_texts = [unique_list[k] for k in sort_keys]

    print(f"[offline] 文本已按长度排序 (shortest: {len(sorted_texts[0])} chars, longest: {len(sorted_texts[-1])} chars)")
    print(f"[offline] 开始编码 {dedup_count} 条唯一文本...")
    all_embs = engine.encode(sorted_texts)  # (dedup_count, 768)

    # 按原始顺序重新排列
    all_embs_reordered = np.empty_like(all_embs)
    for new_idx, old_idx in enumerate(sort_keys):
        all_embs_reordered[old_idx] = all_embs[new_idx]

    # 组装 (N, 768) — 单向量
    emb_dim = all_embs.shape[1]
    embeddings = np.empty((n_drugs, emb_dim), dtype=np.float32)
    for i in range(n_drugs):
        embeddings[i] = all_embs_reordered[text_idx[i]]

    print(f"[offline] single-view embeddings shape: {embeddings.shape}")
    engine.save(embeddings, str(output_embeddings_npy))
    print(f"[offline] embeddings saved: {output_embeddings_npy}")


def parse_args() -> argparse.Namespace:
    repo_root = REPO_ROOT
    default_input = repo_root / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1.csv"
    default_structured = (
        repo_root / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1_structured.csv"
    )
    default_embeddings = repo_root / "drug_comprehensive_embeddings.npy"

    parser = argparse.ArgumentParser(description="Build offline structured dataset + embeddings.")
    parser.add_argument("--input-csv", type=Path, default=default_input)
    parser.add_argument("--output-structured-csv", type=Path, default=default_structured)
    parser.add_argument("--output-embeddings-npy", type=Path, default=default_embeddings)
    parser.add_argument(
        "--model-name",
        type=str,
        default=cfg.medbert_model_name,
    )
    parser.add_argument(
        "--force-rebuild-embeddings",
        action="store_true",
        help="Rebuild embeddings even if output file already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_assets(
        input_csv=args.input_csv,
        output_structured_csv=args.output_structured_csv,
        output_embeddings_npy=args.output_embeddings_npy,
        model_name=args.model_name,
        force_rebuild_embeddings=args.force_rebuild_embeddings,
    )
