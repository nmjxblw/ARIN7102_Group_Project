"""
数据预处理

读取该包下的数据集(.csv文件)
"""

# 标准库导入
import os
import sys
from pathlib import Path
import pandas as pd
from typing import Optional
from matplotlib import pyplot as plt
import tqdm

# 本地模块导入
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


def visualize_data_frame(df: pd.DataFrame, *, title="DataFrame Visualization") -> None:
    """
    可视化数据集信息,以图表形式展示DataFrame的基本信息

    参数：
    - df: 需要可视化信息的DataFrame。
    """

    # 使用matplotlib绘制图表，保存在'reports/figures'目录下。
    # 1. 设置图表为UTF-8编码
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # 支持中文显示
    plt.rcParams["axes.unicode_minus"] = False  # 支持负号显示
    # 2.设置标题为title
    plt.title(title)
    # 3. 对于每个DF列，进行统计。绘制圆饼图显示Cell中的值分布，百分比表示。对于频率小于1%的数据归类于Other。将所有图表绘制在一张大图上。
    num_columns = len(df.columns)
    (fig, axes) = plt.subplots(
        nrows=(num_columns + 2) // 3,
        ncols=3,
        figsize=(15, 5 * ((num_columns + 2) // 3)),
    )
    axes = axes.flatten()
    progress_bar = tqdm.tqdm(
        enumerate(df.to_records()), total=df.to_records().shape[0], desc="绘制图表"
    )
    val_distribute_dict: dict[str, dict[str, int]] = {}
    for i, record in progress_bar:
        for col in df.columns:
            if col not in val_distribute_dict:
                val_distribute_dict[col] = {}
            val = str(record[col])
            if val not in val_distribute_dict[col]:
                val_distribute_dict[col][val] = 0
            val_distribute_dict[col][val] += 1
    # 将键值对填充到图表中
    for idx, (col, val_count_dict) in enumerate(val_distribute_dict.items()):
        ax = axes[idx]
        # 计算总数
        total_count = sum(val_count_dict.values())
        # 计算百分比并归类
        labels = []
        sizes = []
        other_size = 0
        for val, count in val_count_dict.items():
            percentage = (count / total_count) * 100
            if percentage < 1.0:
                other_size += count
            else:
                labels.append(f"{val} ({percentage:.1f}%)")
                sizes.append(count)
        if other_size > 0:
            labels.append(f"Other (<1%)")
            sizes.append(other_size)
        # 绘制饼图
        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 8},
        )
        ax.set_title(f"Column: {col}")
    plt.tight_layout()
    figures_dir = Path.cwd() / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / f"{title.replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()
    plt.show()
    logger.info(f"数据集可视化图表已保存至: {fig_path}")


def generate_visualize_data_frame() -> None:
    # 生成可视化数据集图表
    datasets: dict[str, pd.DataFrame] = load_dataset(None)
    for name, df in datasets.items():

        visualize_data_frame(df, title=name)
