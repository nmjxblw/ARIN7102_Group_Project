"""
静态资源模块

在这里存放项目所需的静态资源，

如静态参数配置、静态类定义（JSON序列化）和Kaggle数据集下载功能。
"""

from .parameters import *
from .classes import *

__all__ = ["PROJECT_NAME", "API_KEY", "LOG_LEVEL"]
