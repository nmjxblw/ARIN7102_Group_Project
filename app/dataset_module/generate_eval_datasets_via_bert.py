import json
from pathlib import Path
from tqdm import tqdm
from utility_module import logger
from static_module import BERT_TRAINING_DATASET_FOLDER, DRUGS_TRAINING_DATASET_FOLDER

RAW_DATASET_FILE = Path(BERT_TRAINING_DATASET_FOLDER, "generated_medical_dataset.json")
OUTPUT_DATASET_FILE = Path(DRUGS_TRAINING_DATASET_FOLDER, "eval_dataset_llm_v2.json")


def generate_eval_datasets_via_bert():
    """基于BERT模型生成评估数据集"""
    from deployment_module.bert_main import preload, predict_with_preload

    raw_data_sentences_set: set[str] = set()
    with open(RAW_DATASET_FILE, "r", encoding="utf-8") as f:
        raw_data: list[dict] = json.load(f)
        for item in raw_data:
            sentences = item.get("sentences", "")
            if sentences:
                raw_data_sentences_set.add(sentences)
    logger.debug(f"从原始数据集中提取了 {len(raw_data_sentences_set)} 条唯一的句子。")
    inference_device, tokenizer, inference_model, mlb_d, mlb_s, medians = preload()
    eval_dataset: list[dict] = []
    for sentence in tqdm(
        raw_data_sentences_set, position=0, leave=True, desc="生成评估数据集"
    ):
        prediction_result = predict_with_preload(
            sentence,
            tokenizer,
            inference_device,
            inference_model,
            mlb_d,
            mlb_s,
            medians,
        )
        prediction_result["sentence"] = sentence
        eval_dataset.append(prediction_result)
        with open(OUTPUT_DATASET_FILE, "a", encoding="utf-8") as f:
            json.dump(prediction_result, f, ensure_ascii=False, indent=4)
    logger.debug(
        f"生成评估数据集完成，共 {len(eval_dataset)} 条记录，存放在 {OUTPUT_DATASET_FILE}"
    )
