"""Rebuild drug embeddings with new model and semantic text format."""
import sys, os
from pathlib import Path

# Set HuggingFace mirror for China network
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(APP_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "pipeline_config.env", override=False)
load_dotenv(APP_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "dummy"

from fastapi_module.offline_assets_builder import build_assets

build_assets(
    input_csv=PROJECT_ROOT / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1.csv",
    output_structured_csv=PROJECT_ROOT / "match_data_preprocessing" / "data" / "enhanced_drug_table_v1_structured.csv",
    output_embeddings_npy=PROJECT_ROOT / "drug_comprehensive_embeddings.npy",
    model_name=os.getenv("MEDBERT_MODEL_NAME", "pritamdeka/S-PubMedBert-MS-MARCO"),
    force_rebuild_embeddings=True,
)
print("\nDone! Embeddings rebuilt successfully.")
