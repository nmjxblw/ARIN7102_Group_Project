"""
数据预处理

读取该包下的数据集(.csv文件)
"""

import os
import sys
from pathlib import Path
import pandas as pd
from typing import Optional
from utility_module import logger


def load_dataset(
    file_path: Optional[list[str]],
    *,
    file_dir: os.PathLike = Path.cwd() / "dataset_module",
) -> dict[str, pd.DataFrame]:
    """
    加载数据集文件

    参数：
    - file_path: 可选的文件路径列表。如果为None，则加载目录下所有.csv文件。
    - file_dir: 数据集文件所在目录，默认为当前工作目录下的'dataset_module'目录。

    返回：
    - 包含数据集名称和对应DataFrame的字典。
    """
    datasets_dict = {}
    if file_path is None:
        file_path = []
        # 实现自动加载目录下所有数据集的功能
        for file in os.listdir(file_dir):
            if file.endswith(".csv"):
                full_path = Path(file_dir) / file
                file_path.append(full_path.name)
                try:
                    df = pd.read_csv(full_path)
                    logger.info(f"成功加载文件: {full_path}")
                    datasets_dict[file] = df
                except Exception as e:
                    logger.info(f"加载文件{full_path}失败: {e}")
                    continue
    else:
        for file in file_path:
            if file.endswith(".csv") is False:
                file += ".csv"
            full_path = Path(file_dir) / file
            try:
                df = pd.read_csv(full_path)
                logger.info(f"成功加载文件: {full_path}")
                datasets_dict[file] = df
            except Exception as e:
                logger.info(f"加载文件{full_path}失败: {e}")
                continue
    return datasets_dict


def visualize_data_frame(df: pd.DataFrame) -> None:
    """
    可视化数据集信息,以图表形式展示DataFrame的基本信息

    参数：
    - df: 需要可视化信息的DataFrame。
    """
    import matplotlib.pyplot as plt

    # TODO: 使用matplotlib绘制图表，保存在'reports/figures'目录下。
    # 1. 设置图表为UTF-8编码
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # 支持中文显示
    plt.rcParams["axes.unicode_minus"] = False  # 支持负号显示
    # 2.设置标题为""
    # 2. 对于每个DF列，进行统计。绘制圆饼图显示Cell中的值分布，百分比表示。对于频率小于1%的数据归类于Other。
