"""DeepSeek 管理器模块"""

# 系统/第三方模块导入
import os
import sys
import queue
from typing import Any
import threading
from openai import OpenAI, resources, responses
from openai.types.chat import ChatCompletion
from pathlib import Path
import json

from torch import Stream

# 本地模块导入
from .singleton_meta import SingletonMeta
from static_module import DEEPSEEK_API_KEY, RUNTIME_TIMESTAMP, CHAT_HISTORY_DIR
from utility_module import logger


class DeepSeekManager(metaclass=SingletonMeta):
    """DeepSeek 管理器单例类"""

    def __init__(self, *, debug_mode: bool = False):
        # 声明变量
        self._initialized: bool = False
        """ 初始化标识符 """
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
        )
        """ 初始化 DeepSeek 客户端 """
        self.message_queue = queue.Queue()
        """ 消息队列 """
        self._history: list = []
        """ 对话历史记录 """
        self.history_file: Path
        """ 对话历史记录文件路径 """
        self._system_prompt: str = (
            rf"""你是一个AI/数学/统计/编程领域的专家，擅长解答各种与人工智能/数学/统计/概率学/编程相关的问题。你的回答应当简洁明了，易于理解，并且尽可能提供实用的信息和建议。在回答问题时，请确保信息的准确性和最新性。你的回答应当以文本为主，必要时可以采用Markdown格式回复。在用户没有明确要求的情况下，使用中文回答所有问题。"""
        )
        """ 系统提示语 """
        self._debug_mode: bool = debug_mode
        """ 调试模式标识符 """
        # 调用初始化函数
        self._initialize()

    def _initialize(self) -> None:
        """初始化函数"""
        if not self._initialized:
            self._load_history_from_file()
            self._deepseek_background_thread = threading.Thread(
                target=self._deepseek_background_task,
                daemon=True,
                name="DeepSeekBackgroundThread",
            )
            self._deepseek_background_thread.start()
            self._initialized = True
            logger.debug("DeepSeek 管理器已初始化并启动后台线程。")

    def _load_history_from_file(self) -> bool:
        """从文件加载对话历史记录"""
        chat_history_dir = Path.cwd() / CHAT_HISTORY_DIR
        chat_history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = (
            chat_history_dir / f"{RUNTIME_TIMESTAMP.strftime('%Y%m%d_%H%M%S')}.json"
        )
        # 首先将系统提示词添加到历史记录中
        self._history.append({"role": "system", "content": self._system_prompt})
        if self.history_file.exists():
            try:

                with open(self.history_file, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
                logger.debug(f"已从文件加载对话历史记录: {self.history_file}")
                return True
            except Exception as e:
                logger.error(f"加载对话历史记录失败: {e}")
        return False

    def _save_history_to_file(self) -> bool:
        """保存对话历史记录到文件"""
        try:
            # 保存时移除系统提示词，避免重复保存
            self._history = self._history[1:]
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self._history, f, ensure_ascii=False, indent=4)
            logger.debug(f"已保存对话历史记录到文件: {self.history_file}")
            return True
        except Exception as e:
            logger.error(f"保存对话历史记录失败: {e}")
            return False

    def send(self, message: str) -> None:
        """发送消息到 DeepSeek"""
        self.message_queue.put(message)

    def _deepseek_background_task(self):
        """DeepSeek 后台任务处理函数"""
        while True:
            try:
                if self.message_queue.empty():
                    continue
                self._history.append(
                    {
                        "role": "user",
                        "content": self.message_queue.get(block=False, timeout=1),
                    }
                )
                # 在这里处理消息，例如发送到 DeepSeek API
                response: ChatCompletion = self.client.chat.completions.create(
                    model="deepseek-reasoner", messages=self._history
                )
                if response is not None:
                    response_json_text = json.dumps(response.model_dump())
                    logger.debug(f"DeepSeek:\n {response_json_text}")
                    _response_content = response.choices[0].message.content
                    logger.info(f"DeepSeek:\n{_response_content}")
                    self._history.append(
                        {
                            "role": "assistant",
                            "content": _response_content,
                        }
                    )
                    self._save_history_to_file()
            except queue.Empty:
                continue


deepseek_manager = DeepSeekManager()
""" DeepSeek 管理器单例 """
