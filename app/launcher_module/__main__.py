import os
import sys
from typing import NoReturn
from utility_module import logger
import pandas as pd


def exit() -> NoReturn:
    """退出程序"""
    logger.info("Application exit.")
    return os._exit(0)


def main() -> None:
    """主程序入口"""
    from static_module import PROJECT_NAME, load_dataset

    logger.info("Launching application: %s", PROJECT_NAME)

    df: pd.DataFrame = load_dataset()
    logger.info("Dataset loaded successfully with %d records.", len(df))
    logger.debug("First 5 records:\n%s", df.head())

    exit()
