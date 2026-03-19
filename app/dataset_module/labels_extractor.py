from pathlib import Path
import json

from static_module import BERT_TRAINING_DATASET_FOLDER
from utility_module import logger

RAW_DATASET_FILE: Path = (
    Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "generated_medical_dataset.json"
)
DISEASE_LABELS_PATH: Path = (
    Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "disease_labels.json"
)

SYMPTOM_LABELS_PATH: Path = (
    Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "symptom_labels.json"
)


def extract_labels_from_dataset() -> None:

    with open(RAW_DATASET_FILE, "r", encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    disease_labels: set[str] = set()
    symptom_labels: set[str] = set()

    for entry in dataset:
        disease_labels.update(entry.get("disease", []))
        symptom_labels.update(entry.get("symptoms", []))

    with open(DISEASE_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(disease_labels), f, ensure_ascii=False, indent=4)
        logger.debug(
            f"提取疾病标签完成，共 {len(disease_labels)} 个标签，存放在 {DISEASE_LABELS_PATH}"
        )
    with open(SYMPTOM_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(symptom_labels), f, ensure_ascii=False, indent=4)
        logger.debug(
            f"提取症状标签完成，共 {len(symptom_labels)} 个标签，存放在 {SYMPTOM_LABELS_PATH}"
        )
