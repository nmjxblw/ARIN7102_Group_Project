import os
import sys
import pandas as pd
import sqlite3
from pathlib import Path

from static_module import DATABASE_FILE
from dataset_module import load_dataset
from utility_module import logger


def build_base_database(table_name: str, data_frame: pd.DataFrame):
    """
    构建基础数据库

    Args:
        table_name (str): 数据表名称
        data_frame (pd.DataFrame): 包含数据的数据框架
    """

    data_frame = data_frame.fillna("NaN").astype(str)
    database_path = Path.cwd() / DATABASE_FILE
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    # Create table

    columns: list[str] = data_frame.columns.tolist()
    column_defs = []
    for col in columns:
        if col == "id":
            column_defs.append(f'"{col}" TEXT PRIMARY KEY')
        else:
            column_defs.append(f'"{col}" TEXT')
    create_table_sql = (
        f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(column_defs)});'
    )
    cursor.execute(create_table_sql)

    # Insert data
    for index, row in data_frame.iterrows():
        place_holders = ", ".join(["?"] * len(columns))

        insert_sql = rf'INSERT OR REPLACE INTO "{table_name}" ({", ".join([f'\"{col}\"' for col in columns])}) VALUES ({place_holders});'
        # logger.debug(f"正在插入数据 {row.to_dict()} 到表 [{table_name}]")
        cursor.execute(
            insert_sql,
            tuple(row),
        )
    conn.commit()
    conn.close()


def run_build_database():
    """构建数据库"""
    datasets: dict[str, pd.DataFrame] = load_dataset(None)
    for table_name, data_frame in datasets.items():
        build_base_database(table_name, data_frame)


if __name__ == "__main__":
    run_build_database()
