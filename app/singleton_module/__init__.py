"""
单例模块包

所有单例管理者实例化放置于此
"""

from .main_thread_task_manager import *
from .singleton_meta import SingletonMeta

__all__ = [
    "AppAsyncTaskManager",
    "SingletonMeta",
]
