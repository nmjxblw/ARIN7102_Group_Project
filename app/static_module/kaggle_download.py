"""Kaggle 数据下载"""

from typing import Any
import kagglehub
import pandas as pd
import os


# Download latest version
def load_dataset() -> dict[str, Any]:
    """加载数据集"""
    url_dict: dict[str, Any] = {
        "taeefnajib/bank-customer-complaints": None,
        "nitindatta/finance-data": None,
    }
    for url in url_dict.keys():
        url_dict[url] = kagglehub.dataset_download(url, force_download=True)
    return url_dict
