"""类结构模板定义模块"""

# 系统/第三方模块导入
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import threading
from datetime import datetime

# 本地模块导入
from .enums import TaskStatus


@dataclass
class AppAsyncTask:
    """应用异步任务模板类"""

    task_id: int
    task_func: Callable
    task_name: str
    created_time: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[Exception] = None
    execution_thread: Optional[threading.Thread] = None

    def cancel(self) -> bool:
        """请求取消任务"""
        if self.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            self.cancel_event.set()
            from utility_module import logger  # 避免循环导入

            logger.info(f"任务 {self.task_id} ({self.task_name}) 取消请求已发出。")
            return True
        return False

    def is_cancelled(self) -> bool:
        """检查任务是否已被取消"""
        return self.cancel_event.is_set()
