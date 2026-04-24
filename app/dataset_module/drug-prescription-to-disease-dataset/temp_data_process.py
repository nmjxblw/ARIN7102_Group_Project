import os
import sys
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
import re

FILE_PATH = Path.cwd() / "final_cleaned.csv"
SAVE_PATH = Path.cwd() / "final_processed.json"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_main():
    df = pd.read_csv(FILE_PATH)
    json_data = list(df.to_dict(orient="records"))
    merge_dict: dict[str, list] = {}
    for item in tqdm(json_data):
        drug_name = re.sub(
            r"\s+", "_", re.sub(r"\s/\s", r"|", str(item["drug"]).strip().lower())
        )
        disease_name = re.sub(
            r"\s+", "_", re.sub(r"\,\s+", r",", str(item["disease"]).strip().lower())
        )
        if drug_name not in merge_dict:
            merge_dict[drug_name] = [disease_name]
        else:
            if disease_name not in merge_dict[drug_name]:
                merge_dict[drug_name].append(disease_name)
    with open(SAVE_PATH, "w") as f:
        json.dump(merge_dict, f, indent=4)


if __name__ == "__main__":
    run_main()
