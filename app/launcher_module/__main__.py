"""主程序模块"""

# 系统/第三方模块导入
import os
from typing import NoReturn, Callable, Optional, Any
import threading
import atexit
import queue

# 本地模块导入
from static_module import TaskStatus, PROJECT_NAME, THREAD_TIMEOUT
from singleton_module import AppAsyncTaskManager
from utility_module import logger
from dataset_module import *

app_running_flag: bool = True
"""应用程序运行标志"""
app_main_thread_while_loop_tasks: queue.Queue[Callable] = queue.Queue()
""" 主线程while循环任务列表 """

_main_thread_lock = threading.RLock()
"""主线程锁"""

user_input_queue: queue.Queue[Any] = queue.Queue()
"""用户输入队列"""


# 全局任务管理器
task_manager = AppAsyncTaskManager()


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
    task_id = task_manager.create_task(task_func, task_name)
    task_manager.submit_task(task_id)
    return task_id


def display_user_menu():
    """显示用户菜单"""
    print("\n" + "=" * 50)
    print("主菜单")
    print("=" * 50)
    print("0. 查看任务状态")
    print("1. 取消任务")
    print("2. 清理已完成任务")
    print("9. 退出程序")
    print("=" * 50)


def handle_user_input():
    """处理用户输入的线程"""
    global app_running_flag
    while app_running_flag:
        try:
            display_user_menu()
            user_choice = input("请输入选项 (0-2, 9): ").strip()
            user_input_queue.put(user_choice)
        except EOFError:
            # 处理管道关闭或其他EOF情况
            break
        except Exception as e:
            logger.error(f"处理用户输入时出错: {str(e)}")


def show_task_status():
    """显示所有任务状态"""
    all_tasks = task_manager.get_all_tasks()
    if not all_tasks:
        print("\n当前没有任务。")
        return

    print("\n" + "-" * 70)
    print(f"{'任务ID':<8} {'任务名称':<20} {'状态':<10} {'创建时间':<20}")
    print("-" * 70)
    for task_id, task in all_tasks.items():
        print(
            f"{task_id:<8} {task.task_name:<20} {task.status.value:<10} {task.created_time.strftime('%Y-%m-%d %H:%M:%S'):<20}"
        )
    print("-" * 70 + "\n")


def cancel_task_by_id():
    """取消指定ID的任务"""
    show_task_status()
    try:
        task_id = int(input("请输入要取消的任务ID: ").strip())
        if task_manager.cancel_task(task_id):
            print(f"任务 {task_id} 取消请求已发出。")
        else:
            print(f"无法取消任务 {task_id}，请检查任务ID或任务状态。")
    except ValueError:
        print("无效的任务ID。")


def cleanup_completed_tasks():
    """清理已完成和失败的任务"""
    all_tasks = task_manager.get_all_tasks()
    removed_count = 0
    for task_id, task in all_tasks.items():
        if task.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ]:
            if task_manager.remove_task(task_id):
                removed_count += 1
    print(f"已清理 {removed_count} 个任务。")


def process_user_input(user_input: str):
    """处理用户输入"""
    global app_running_flag

    if user_input == "0":
        show_task_status()
    elif user_input == "1":
        cancel_task_by_id()
    elif user_input == "2":
        cleanup_completed_tasks()
    elif user_input == "9":
        app_running_flag = False
        print("准备退出程序...")
        logger.info("用户选择退出程序。")
    else:
        print("无效的选项，请重新输入。")


def main_thread_task():
    """主线程任务"""
    global app_running_flag

    # 启动用户输入处理线程
    input_thread = threading.Thread(
        target=handle_user_input, daemon=True, name="UserInputHandler"
    )
    input_thread.start()
    logger.info("用户输入处理线程已启动。")

    while app_running_flag:
        try:
            # 处理队列中的任务（来自其他部分的长时间任务）
            try:
                task: Callable = app_main_thread_while_loop_tasks.get(timeout=0.5)
                task()
            except queue.Empty:
                pass

            # 处理用户输入
            try:
                user_input = user_input_queue.get(timeout=0.1)
                process_user_input(user_input)
            except queue.Empty:
                pass

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
    # 启动主线程,并设置为非守护线程
    # application_thread = threading.Thread(
    #     target=main_thread_task, args=(), kwargs={}, daemon=False
    # )
    # application_thread.start()
    # application_thread.join()
    process_datasets_deepseek_version()
    exit()
