"""Kaggle 数据下载"""

import shutil
from typing import Any, Optional
import kagglehub
import pandas as pd
import os
import subprocess
import json
from pathlib import Path
from static_module import KAGGLE_DATASET_DOWNLOAD_URLS_FILE
from utility_module import logger


# Download latest version
def download_and_open_datasets() -> dict[str, Optional[str]]:
    """
    下载并加载数据集，在下载完成后打开下载文件夹。

    返回
    - dict[str, Optional[str]]: 包含数据集文件路径的字典
    """
    urls_file_path = Path.cwd() / KAGGLE_DATASET_DOWNLOAD_URLS_FILE
    if not urls_file_path.exists():
        logger.error(f"Kaggle数据集下载URL列表文件不存在: {urls_file_path}")
        return {}
    url_dict: dict[str, Optional[str]] = json.loads(
        urls_file_path.read_text(encoding="utf-8")
    )
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
            logger.debug(
                f"已复制文件夹 {url_dict[url]} 到 dataset_module/{os.path.basename(url)}"
            )
        except Exception as e:
            logger.error(f"复制文件夹 {url_dict[url]} 时出错: {e}")
    return url_dict
