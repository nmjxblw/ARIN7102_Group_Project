import json
import pandas as pd
from pathlib import Path

from static_module import BERT_TRAINING_DATASET_FOLDER
from utility_module import logger


def json_to_dataframe(json_file: Path) -> pd.DataFrame:
    """Convert JSON file to pandas DataFrame."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Assuming the JSON structure is a list of records
    df = pd.json_normalize(data)
    return df


def json_to_dataframe_main():
    file_full_path = (
        Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "generated_medical_dataset.json"
    )
    if not file_full_path.exists():
        raise FileNotFoundError(f"JSON 文件[{file_full_path}]不存在")
    df = json_to_dataframe(file_full_path)
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"DataFrame head:\n{df.head()}")
    output_csv_path = (
        Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "generated_medical_dataset.csv"
    )
    df.to_csv(output_csv_path, index=False)
    logger.info(f"DataFrame saved to CSV: {output_csv_path}")
