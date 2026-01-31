import os
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from utility_module import logger


class MultiColumnPieChartGenerator:
    def __init__(self, df: pd.DataFrame, max_workers=4, max_categories=20):
        """
        初始化多列饼图生成器

        参数:
        df: pandas DataFrame, 输入数据
        max_workers: int, 最大线程数
        max_categories: int, 每列最大分类数（避免过多分类导致饼图难以阅读）
        """
        self.df = df.copy()
        self.max_workers = max_workers
        self.max_categories = max_categories
        self.results = {}

    def calculate_column_percentage(self, column_name):
        """
        计算单列的百分比分布[1,3](@ref)

        参数:
        column_name: str, 列名

        返回:
        dict: 包含列名和百分比数据的字典
        """
        try:
            # 获取列数据
            col_data = self.df[column_name]

            # 处理缺失值
            col_data = col_data.dropna()

            if len(col_data) == 0:
                return {"column": column_name, "data": None, "error": "无有效数据"}

            # 计算value_counts，限制最大分类数[5](@ref)
            value_counts = col_data.value_counts(normalize=True) * 100

            # 如果分类过多，保留前max_categories个，其余合并为"其他"
            if len(value_counts) > self.max_categories:
                top_categories = value_counts.head(self.max_categories - 1)
                other_percentage = value_counts.iloc[self.max_categories - 1 :].sum()
                value_counts = pd.concat(
                    [top_categories, pd.Series([other_percentage], index=["其他"])]
                )

            return {
                "column": column_name,
                "data": value_counts,
                "total_count": len(col_data),
            }
        except Exception as e:
            return {"column": column_name, "data": None, "error": str(e)}

    def process_columns_parallel(self, columns=None):
        """
        使用多线程并行处理所有列[6,7,8](@ref)

        参数:
        columns: list, 要处理的列名列表，如果为None则处理所有列

        返回:
        dict: 处理结果
        """
        if columns is None:
            # 自动选择分类列（object类型和category类型）
            categorical_columns = self.df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            # 也可以处理唯一值较少的数值列
            numeric_columns = []
            for col in self.df.select_dtypes(include=["int64", "float64"]).columns:
                if self.df[col].nunique() <= self.max_categories:
                    numeric_columns.append(col)
            columns = categorical_columns + numeric_columns

        print(f"开始处理 {len(columns)} 列数据，使用 {self.max_workers} 个线程...")

        # 使用ThreadPoolExecutor进行多线程处理[7,8](@ref)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_column = {
                executor.submit(self.calculate_column_percentage, col): col
                for col in columns
            }

            # 使用tqdm显示进度[6](@ref)
            results = {}
            for future in tqdm(
                as_completed(future_to_column), total=len(columns), desc="处理列"
            ):
                column_name = future_to_column[future]
                try:
                    result = future.result()
                    results[column_name] = result
                except Exception as e:
                    results[column_name] = {
                        "column": column_name,
                        "data": None,
                        "error": str(e),
                    }

        self.results = results
        return results

    def create_combined_pie_charts(self, figsize=(20, 15), max_cols=4):
        """
        创建合并的多列饼图[10,11](@ref)

        参数:
        figsize: tuple, 图形大小
        max_cols: int, 每行最多显示的饼图数量

        返回:
        matplotlib.figure.Figure: 图形对象
        """
        # 过滤出有有效数据的列
        valid_results = {
            k: v
            for k, v in self.results.items()
            if v["data"] is not None and len(v["data"]) > 0
        }

        if not valid_results:
            print("没有有效数据可以绘制饼图")
            return None

        n_cols = min(max_cols, len(valid_results))
        n_rows = (len(valid_results) + n_cols - 1) // n_cols

        # 创建子图[10,11](@ref)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # 如果只有一行，确保axes是二维数组
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # 设置颜色循环
        colors = plt.cm.Set3(np.linspace(0, 1, self.max_categories))

        # 绘制每个饼图
        for idx, (col_name, result) in enumerate(valid_results.items()):
            row = idx // n_cols
            col = idx % n_cols

            ax = axes[row, col]
            data = result["data"]
            total_count = result["total_count"]

            # 绘制饼图[1,3](@ref)
            wedges, texts, autotexts = ax.pie(
                data.values,
                labels=data.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors[: len(data)],
                wedgeprops={"edgecolor": "w", "linewidth": 1},
            )

            # 设置标题
            ax.set_title(f"{col_name}\n(总数: {total_count:,})", fontsize=12, pad=10)

            # 调整标签字体大小
            for text in texts + autotexts:
                text.set_fontsize(8)

        # 隐藏空的子图
        for idx in range(len(valid_results), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        # 调整布局[10](@ref)
        plt.tight_layout(pad=3.0)
        plt.suptitle("多列数据分布饼图", fontsize=16, y=1.02)

        return fig

    def generate_summary_report(self):
        """生成处理结果摘要报告"""
        valid_count = sum(1 for v in self.results.values() if v["data"] is not None)
        error_count = sum(1 for v in self.results.values() if v.get("error"))

        print(f"\n=== 处理摘要 ===")
        print(f"总处理列数: {len(self.results)}")
        print(f"成功处理列数: {valid_count}")
        print(f"错误列数: {error_count}")

        # 显示错误信息
        for col_name, result in self.results.items():
            if result.get("error"):
                print(f"列 '{col_name}' 错误: {result['error']}")


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


# 使用示例
def process_datasets_deepseek_version():
    dfs = load_dataset(None)
    for name, df in dfs.items():
        logger.info(f"数据集 '{name}' 形状: {df.shape}")
        # 移除ID列（通过检测列中是否包含"ID"或类似标识）
        df = df.loc[:, ~df.columns.str.contains("ID", case=False)]
        # 移除日期列（通过检测列名中是否包含"date"或类似标识）
        df = df.loc[:, ~df.columns.str.contains("date", case=False)]
        # 初始化生成器
        generator = MultiColumnPieChartGenerator(df, max_workers=4, max_categories=10)

        # 处理数据
        results = generator.process_columns_parallel()

        # 生成摘要报告
        generator.generate_summary_report()

        # 绘制饼图
        logger.info("\n生成饼图...")
        fig = generator.create_combined_pie_charts(figsize=(20, 15), max_cols=3)

        if fig:
            # 保存图片
            plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
            logger.info(f"饼图已保存为 '{name}.png'")

            # 显示图片
            plt.show()
        else:
            logger.info("无法生成饼图")


if __name__ == "__main__":
    process_datasets_deepseek_version()
