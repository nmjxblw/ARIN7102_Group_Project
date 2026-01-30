"""枚举类型定义模块"""

# 系统/第三方模块导入
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""

    PENDING = "待处理"
    RUNNING = "运行中"
    COMPLETED = "已完成"
    CANCELLED = "已取消"
    FAILED = "失败"
