"""Kaggle 数据下载"""

from typing import Any
import kagglehub
import pandas as pd
import os
import subprocess
from utility_module import logger


# Download latest version
def load_dataset() -> dict[str, Any]:
    """加载数据集"""
    url_dict: dict[str, Any] = {
        "taeefnajib/bank-customer-complaints": None,
        "nitindatta/finance-data": None,
    }
    for url in url_dict.keys():
        url_dict[url] = kagglehub.dataset_download(url, force_download=True)
        try:
            subprocess.run(
                ["explorer", url_dict[url]],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error opening file {url_dict[url]}: {e}")
    return url_dict
