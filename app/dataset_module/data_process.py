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
from matplotlib.patches import Rectangle
import tqdm
import json
import numpy as np

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
                    df = pd.read_csv(full_path).fillna("NaN")  # 加载数据并填充缺失值
                    logger.info(f"成功加载文件: {full_path}")
                    datasets_dict[Path(file).stem] = df
                except Exception as e:
                    logger.info(f"加载文件{full_path}失败: {e}")
                    continue
    else:
        for file in file_path:
            if file.endswith(".csv") is False:
                file += ".csv"
            full_path = Path(file_dir) / file
            try:
                df = pd.read_csv(full_path).fillna("NaN")  # 加载数据并填充缺失值
                logger.info(f"成功加载文件: {full_path}")
                datasets_dict[Path(file).stem] = df
            except Exception as e:
                logger.info(f"加载文件{full_path}失败: {e}")
                continue
    return datasets_dict


def visualize_data_frame(
    df: pd.DataFrame,
    *,
    file_name="DataFrame Visualization",
    max_categories_per_chart=10,
) -> os.PathLike:
    """
    可视化数据帧

    参数：
    - df: 需要可视化信息的DataFrame
    - file_name: 输出文件名
    - max_categories_per_chart: 每个图表最多显示的类别数
    """
    # 设置中文字体和样式 - 更彻底的配置
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "STSong"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["text.usetex"] = False
    plt.style.use("seaborn-v0_8-whitegrid")

    file_name = Path(file_name).stem

    # 计算布局
    num_rows = len(df.columns)

    # 创建图形 - 增加宽度
    fig = plt.figure(figsize=(28, 5 * num_rows))

    # 创建外部布局 - 调整高度比例
    gs = fig.add_gridspec(
        num_rows + 1, 1, hspace=0.3, height_ratios=[0.08] + [1] * num_rows
    )

    # 添加标题
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.5,
        f"{file_name}",
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="center",
    )

    progress_bar = tqdm.tqdm(enumerate(df.columns), desc="绘制图表", total=num_rows)

    for idx, col in progress_bar:
        row_idx = idx + 1

        # 为每一行创建子网格：左边饼图，右边图例
        inner_gs = gs[row_idx].subgridspec(1, 2, width_ratios=[1, 1.2], wspace=0.1)

        # 左侧：饼图
        ax_pie = fig.add_subplot(inner_gs[0])
        # 右侧：图例
        ax_legend = fig.add_subplot(inner_gs[1])
        ax_legend.axis("off")

        # 统计值分布
        value_counts = df[col].astype(str).value_counts()
        total = len(df)

        # 如果类别太多，合并小的类别
        if len(value_counts) > max_categories_per_chart:
            main_values = value_counts.head(max_categories_per_chart - 1)
            other_count = value_counts.tail(
                len(value_counts) - max_categories_per_chart + 1
            ).sum()
            main_values = pd.concat(
                [
                    main_values,
                    pd.Series(
                        {
                            f"Others ({len(value_counts)-max_categories_per_chart+1})": other_count
                        }
                    ),
                ]
            )
        else:
            main_values = value_counts

        # 计算百分比
        percentages = (main_values / total * 100).round(2)

        # 准备数据
        sizes = np.array(main_values.values)
        labels = main_values.index.tolist()

        # 创建好看的颜色
        colors = plt.cm.get_cmap("Set3")(np.linspace(0, 1, len(sizes))).tolist()

        # 绘制甜甜圈图
        wedges, texts, autotexts = ax_pie.pie(  # type: ignore
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
            autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
            pctdistance=0.75,
            textprops=dict(color="white", fontsize=11, fontweight="bold"),
        )

        # 中心添加信息
        center_text = f"Total\n{total:,}"
        ax_pie.text(
            0,
            0,
            center_text,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            style="italic",
            fontfamily="sans-serif",
        )

        # 设置子图标题
        ax_pie.set_title(
            f"{col}", fontsize=16, fontweight="bold", pad=15, fontfamily="sans-serif"
        )
        ax_pie.axis("equal")

        # 在右侧面板创建带颜色的图例
        num_labels = len(labels)
        # 根据标签数量动态调整列数

        cols = (num_labels + 9) // 10  # 每列最多显示10个标签
        cols = max(1, cols)  # 确保至少有1列

        rows_per_col = (num_labels + cols - 1) // cols  # 向上取整
        col_width = 1.0 / cols

        for i, (label, count, pct, color) in enumerate(
            zip(labels, main_values.values, percentages, colors)
        ):
            label_str = str(label).replace("_", " ")
            if len(label_str) > 20:
                label_display = label_str[:17] + "..."
            else:
                label_display = label_str

            # 计算当前项在第几列、第几行
            col_idx = i // rows_per_col
            row_idx = i % rows_per_col

            # 计算位置
            x_start = col_idx * col_width + 0.02
            y_start = 0.95
            y_step = 0.85 / rows_per_col
            y_pos = y_start - row_idx * y_step

            # 绘制颜色方块
            rect = Rectangle(
                (x_start, y_pos - 0.015),
                0.04,
                0.03,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                transform=ax_legend.transAxes,
            )
            ax_legend.add_patch(rect)

            # 添加文本 - 明确指定字体
            text_content = f"{i+1:2d}. {label_display:<18} {count:>6,} ({pct:>6.2f}%)"
            ax_legend.text(
                x_start + 0.06,
                y_pos,
                text_content,
                fontsize=9,
                verticalalignment="center",
                transform=ax_legend.transAxes,
                fontfamily="sans-serif",
            )

    # 调整布局
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3)

    # 保存文件
    figures_dir = Path.cwd() / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_path = figures_dir / f"{file_name.replace(' ', '_')}.png"

    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"可视化图表已保存至: {fig_path}")
    return fig_path


def generate_visualize_data_frame() -> list[Optional[os.PathLike]]:
    """
    生成可视化数据集图表
    """
    datasets: dict[str, pd.DataFrame] = load_dataset(None)
    fig_paths = []
    for file_name, df in datasets.items():

        fig_path = visualize_data_frame(
            df, file_name=file_name, max_categories_per_chart=30
        )
        fig_paths.append(fig_path)
    return fig_paths
