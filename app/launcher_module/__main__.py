"""主程序模块"""

# 系统/第三方模块导入
import os
from typing import NoReturn, Callable, Optional, Any
import threading
import atexit
import queue
import re

# 本地模块导入
from static_module import TaskStatus, PROJECT_NAME, THREAD_TIMEOUT
from singleton_module import AppAsyncTaskManager, deepseek_manager
from utility_module import logger

app_running_flag: bool = True
"""应用程序运行标志"""
app_main_thread_while_loop_tasks: list[Callable] = []
""" 主线程while循环任务列表 """

_main_thread_lock = threading.RLock()
"""主线程锁"""

user_input_queue: queue.Queue[Any] = queue.Queue()
"""用户输入队列"""

# 全局任务管理器
app_async_task_manager = AppAsyncTaskManager()
""" 全局异步任务管理器单例 """


def exit() -> NoReturn:
    """退出程序"""
    logger.debug("程序退出。")
    return os._exit(0)


def submit_async_task(task_func: Callable, task_name: str = "未命名任务") -> int:
    """
    提交异步任务到任务管理器

    Args:
        task_func: 任务函数，可选参数cancel_event用于检查取消信号
        task_name: 任务名称

    Returns:
        任务ID
    """
    task_id = app_async_task_manager.create_task(task_func, task_name)
    app_async_task_manager.submit_task(task_id)
    return task_id


def process_user_input():
    """处理用户输入"""
    global app_running_flag
    user_input: str = input("用户：").strip()
    if re.match(r"^\s?(exit|quit|q)\.?$", user_input, re.IGNORECASE):
        app_running_flag = False
        return
    deepseek_manager.send(user_input)


def register_default_main_thread_tasks():
    """注册默认的主线程任务"""
    # 这里可以添加一些默认的主线程任务
    logger.debug("注册默认的主线程任务。")
    app_main_thread_while_loop_tasks.append(process_user_input)


def main_thread_task_handler():
    """主线程任务处理器"""
    logger.debug("主线程任务处理器启动。")
    global app_running_flag
    while app_running_flag:
        try:
            with _main_thread_lock:
                for task in app_main_thread_while_loop_tasks:
                    try:
                        task()
                    except Exception as e:
                        logger.error(f"主线程任务执行出错: {str(e)}")

        except KeyboardInterrupt:
            app_running_flag = False
            logger.info("检测到键盘中断，准备退出主线程任务。")


@atexit.register
def on_exit():
    """程序退出时的处理函数"""
    end_background_threads()
    logger.debug("程序已退出。")


def end_background_threads():
    """
    结束后台线程

    在其他模块中可能开启了后台线程，这里确保它们在程序退出前被正确终止。
    这有助于防止资源泄漏和确保程序的干净退出。
    """
    # 线程处理
    current_thread = threading.current_thread()  # 获取当前线程（通常是主线程）
    _thread_list: list[threading.Thread] = threading.enumerate()
    for t in _thread_list:
        if t is not current_thread:
            # 等待其他线程结束
            t.join(timeout=THREAD_TIMEOUT)
    logger.debug("结束后台线程。")


def run() -> None:
    """主程序入口"""

    logger.debug("主程序运行中: %s", PROJECT_NAME)

    register_default_main_thread_tasks()

    main_thread_task_handler()
    exit()
