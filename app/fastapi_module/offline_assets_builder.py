from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from embedded_module.drug_embedding_engine import DrugEmbeddingEngine
from embedded_module.recommendation_pipeline import prepare_drug_dataframe


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
    embeddings = engine.encode(structured_df["semantic_text"].tolist())
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
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
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
