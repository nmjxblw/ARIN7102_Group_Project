"""
单例模块包

所有单例管理者实例化放置于此
"""

from .singleton_meta import SingletonMeta
from .main_thread_task_manager import *
from .deepseek_manager import *

__all__ = [
    "SingletonMeta",
    "AppAsyncTaskManager",
    "deepseek_manager",
    "deepseek_manager_new"
]
