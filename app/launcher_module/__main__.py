import os
import sys
from typing import NoReturn
from utility_module import logger
import pandas as pd


def exit() -> NoReturn:
    """退出程序"""
    logger.debug("程序退出。")
    return os._exit(0)


def main() -> None:
    """主程序入口"""
    from static_module import PROJECT_NAME, load_dataset

    logger.debug("主程序运行中: %s", PROJECT_NAME)

    exit()
