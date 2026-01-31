"""主线程异步任务管理器模块
管理和调度在主线程中运行的异步任务。"""

# 系统/第三方模块导入
import threading
from typing import Callable, Any, Optional
from datetime import datetime

# 本地模块导入
from utility_module import logger
from .singleton_meta import SingletonMeta
from static_module import TaskStatus, AppAsyncTask


class AppAsyncTaskManager(metaclass=SingletonMeta):
    """异步任务管理器"""

    def __init__(self):
        self._tasks: dict[int, AppAsyncTask] = {}
        self._task_counter = 0
        self._manager_lock = threading.RLock()

    def create_task(self, task_func: Callable, task_name: str) -> int:
        """创建新任务，返回任务ID"""
        with self._manager_lock:
            self._task_counter = hash(datetime.now())
            task_id = self._task_counter
            task = AppAsyncTask(
                task_id=task_id, task_func=task_func, task_name=task_name
            )
            self._tasks[task_id] = task
            logger.info(f"任务 {task_id} ({task_name}) 已创建。")
            return task_id

    def submit_task(self, task_id: int) -> bool:
        """提交任务到后台线程执行"""
        with self._manager_lock:
            if task_id not in self._tasks:
                logger.warning(f"任务 {task_id} 不存在。")
                return False

            task = self._tasks[task_id]
            if task.status != TaskStatus.PENDING:
                logger.warning(f"任务 {task_id} 状态不是待处理。")
                return False

            # 启动后台线程执行任务
            task.status = TaskStatus.RUNNING
            task.execution_thread = threading.Thread(
                target=self._execute_task,
                args=(task_id,),
                daemon=True,
                name=f"AppAsyncTask-{task_id}",
            )
            task.execution_thread.start()
            logger.info(f"任务 {task_id} ({task.task_name}) 已提交执行。")
            return True

    def _execute_task(self, task_id: int) -> None:
        """在后台线程中执行任务"""
        task = self._tasks[task_id]
        try:
            # 将cancel_event作为参数传递给任务函数，允许任务检查取消信号
            # 如果任务函数不接受cancel_event参数，直接调用
            import inspect

            sig = inspect.signature(task.task_func)

            if "cancel_event" in sig.parameters:
                task.result = task.task_func(cancel_event=task.cancel_event)
            else:
                # 定期检查取消信号
                task.result = task.task_func()

            if task.is_cancelled():
                with self._manager_lock:
                    task.status = TaskStatus.CANCELLED
                logger.info(f"任务 {task_id} ({task.task_name}) 已取消。")
            else:
                with self._manager_lock:
                    task.status = TaskStatus.COMPLETED
                logger.info(f"任务 {task_id} ({task.task_name}) 已完成。")
        except Exception as e:
            with self._manager_lock:
                task.status = TaskStatus.FAILED
                task.error = e
            logger.error(f"任务 {task_id} ({task.task_name}) 执行失败: {str(e)}")

    def cancel_task(self, task_id: int) -> bool:
        """取消指定任务"""
        with self._manager_lock:
            if task_id not in self._tasks:
                logger.warning(f"任务 {task_id} 不存在。")
                return False
            return self._tasks[task_id].cancel()

    def get_task_status(self, task_id: int) -> Optional[TaskStatus]:
        """获取任务状态"""
        with self._manager_lock:
            if task_id not in self._tasks:
                return None
            return self._tasks[task_id].status

    def get_all_tasks(self) -> dict[int, AppAsyncTask]:
        """获取所有任务副本"""
        with self._manager_lock:
            return dict(self._tasks)

    def wait_task(self, task_id: int, timeout: Optional[float] = None) -> bool:
        """等待任务完成"""
        with self._manager_lock:
            if task_id not in self._tasks:
                return False
            task = self._tasks[task_id]

        if task.execution_thread:
            task.execution_thread.join(timeout=timeout)
            return (
                task.execution_thread is not None
                and not task.execution_thread.is_alive()
            )
        return False

    def remove_task(self, task_id: int) -> bool:
        """移除任务"""
        with self._manager_lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.info(f"任务 {task_id} 已移除。")
                return True
        return False
