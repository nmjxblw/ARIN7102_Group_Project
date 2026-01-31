"""
静态资源模块

在这里存放项目所需的静态资源，

如静态参数配置、静态类定义（JSON序列化）。
"""

from .parameters import *
from .classes import *
from .enums import *

__all__ = [
    # Parameters
    "RUNTIME_TIMESTAMP",
    "RUNTIME_TIMESTAMP_STR",
    "PROJECT_NAME",
    "DEEPSEEK_API_KEY",
    "LOG_LEVEL",
    "CHAT_HISTORY_DIR",
    "THREAD_TIMEOUT",
    # Classes
    "AppAsyncTask",
    # Enums
    "TaskStatus",
]
