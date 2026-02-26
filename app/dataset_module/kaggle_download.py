"""Kaggle 数据下载"""

import shutil
from typing import Any, Optional
import kagglehub
import pandas as pd
import os
import subprocess
from utility_module import logger


# Download latest version
def download_and_open_datasets() -> dict[str, Optional[str]]:
    """
    下载并加载数据集，在下载完成后打开下载文件夹。

    返回
    - dict[str, Optional[str]]: 包含数据集文件路径的字典
    """
    url_dict: dict[str, Optional[str]] = {
        "manncodes/drug-prescription-to-disease-dataset": None,
        "uom190346a/disease-symptoms-and-patient-profile-dataset": None,
        "jithinanievarghese/drugs-related-to-common-treatments": None,
        "itachi9604/disease-symptom-description-dataset": None,
        "niyarrbarman/symptom2disease": None,
        "jithinanievarghese/drugs-side-effects-and-medical-condition": None,
        "jessicali9530/kuc-hackathon-winter-2018": None,
    }
    for url in url_dict.keys():
        url_dict[url] = kagglehub.dataset_download(url, force_download=True)
        try:
            # 使用浏览器打开下载文件夹
            # subprocess.run(
            #     ["explorer", str(url_dict[url])],
            #     check=True,
            # )
            shutil.copytree(
                str(url_dict[url]),
                os.path.join(os.getcwd(), "dataset_module", os.path.basename(url)),
                dirs_exist_ok=True,
            )
        except Exception as e:
            logger.error(f"复制文件夹 {url_dict[url]} 时出错: {e}")
    return url_dict
