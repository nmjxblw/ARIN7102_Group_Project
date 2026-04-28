"""DeepSeek 管理器模块"""

# 系统/第三方模块导入
import os
import sys
import queue
from typing import Any, Optional
import threading
from openai import OpenAI, resources, responses
from openai.types.chat import ChatCompletion
from pathlib import Path
import json

from torch import Stream

# 本地模块导入
from singleton_module import SingletonMeta
from static_module import (
    DEEPSEEK_API_KEY,
    RUNTIME_TIMESTAMP,
    CHAT_HISTORY_DIR,
    DEEPSEEK_MODEL,
    DEFAULT_PROMPT_FOLDER_PATH,
)
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
        self.message_queue: queue.Queue = queue.Queue()
        """ 消息队列 """
        self._history: list = []
        """ 对话历史记录 """
        self.history_file: Path
        """ 对话历史记录文件路径 """
        self._system_prompt: str = rf"""You are a medical consultant robot."""
        """ 系统提示语 """
        self._debug_mode: bool = debug_mode
        """ 调试模式标识符 """
        if self._debug_mode:
            logger.debug("DeepSeekManager 已进入调试模式")
        self.default_prompt_folder_path: str = DEFAULT_PROMPT_FOLDER_PATH
        """ 默认提示词文件夹路径 """
        self.default_prompt: str = ""
        """ 默认提示词内容 """
        self._app_is_running: bool = True
        """ 应用程序运行标志 """
        # 调用初始化函数
        self._initialize()

    def _initialize(self) -> None:
        """初始化函数"""
        if not self._initialized:
            self._load_prompts()
            self._load_history_from_file()
            self._deepseek_background_thread = threading.Thread(
                target=self._deepseek_background_task,
                daemon=True,  # 设置为守护线程，随主程序退出而自动结束
                name="DeepSeekBackgroundThread",
            )
            self._deepseek_background_thread.start()
            self._initialized = True
            logger.debug("DeepSeek 管理器已初始化。")

    def _load_prompts(self) -> None:
        """加载提示词"""
        default_prompt_path = Path(self.default_prompt_folder_path)
        prompt_file: Path | None = None
        for root, dir, files in os.walk(default_prompt_path):
            for file in files:
                if file.endswith(".md") and file.startswith("default"):
                    prompt_file = Path(root) / file
                    break
        if not prompt_file or not prompt_file.exists():
            raise FileNotFoundError(
                f"未找到默认提示词文件，路径: {self.default_prompt_folder_path}，请确保该文件夹下存在以default开头的.md文件。"
            )

        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.default_prompt = f.read()
            if self._debug_mode:
                logger.debug(f"已加载默认提示词: {prompt_file}")
        except Exception as e:
            logger.error(f"加载默认提示词失败: {e}")

    def _load_history_from_file(self, file_path: Optional[os.PathLike] = None) -> bool:
        """从文件加载对话历史记录"""
        chat_history_dir = Path.cwd() / CHAT_HISTORY_DIR
        chat_history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = (
            chat_history_dir
            / f"{RUNTIME_TIMESTAMP.strftime('%Y%m%d_%H%M%S')if file_path is None else Path(file_path).stem}.json"
        )
        # 首先将系统提示词添加到历史记录中
        self._history.append({"role": "system", "content": self._system_prompt})
        if self.history_file.exists():
            try:

                with open(self.history_file, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
                if self._debug_mode:
                    logger.debug(f"已从文件加载对话历史记录: {self.history_file}")
                return True
            except Exception as e:
                logger.error(f"加载对话历史记录失败: {e}")
        return False

    def _save_history_to_file(self) -> bool:
        """保存对话历史记录到文件"""
        try:
            # 保存时移除系统提示词，避免重复保存
            self.saved_history = self._history[1:]
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.saved_history, f, ensure_ascii=False, indent=4)
            logger.debug(f"已保存对话历史记录到文件: {self.history_file}")
            return True
        except Exception as e:
            logger.error(f"保存对话历史记录失败: {e}")
            return False

    def _prompt_build(self, input: object) -> str:
        """构建提示词"""
        sentences: str = input.get("sentences", "")
        if not sentences:
            return ""
        pipeline_output: dict = input.get("pipeline_output", {})
        bert_output: dict = input.get("bert_output", {})
        prompt: str = self.default_prompt.format(
            sentences=sentences,
            pipeline_output=pipeline_output,
            bert_output=bert_output,
        )
        return prompt

    def send(self, input_object: object) -> None:
        """发送消息到 DeepSeek"""
        self.message_queue.put(input_object)

    def dev_send(self, string: str) -> None:
        """开发测试用发送消息到 DeepSeek"""
        if not self._debug_mode:
            return
        self.message_queue.put(string)

    def _deepseek_background_task(self):
        """DeepSeek 后台任务处理函数"""
        while self._app_is_running:
            try:
                if self.message_queue.empty():
                    continue
                input_object: object = self.message_queue.get(block=False, timeout=1)
                logger.debug(f"正在处理用户输入: {input_object}")
                if self._debug_mode:
                    input_content: str = str(input_object)
                else:
                    input_content: str = self._prompt_build(input_object)
                logger.debug(input_content)
                if not input_content:
                    continue
                self._history.append(
                    {
                        "role": "user",
                        "content": input_content,
                    }
                )
                # 在这里处理消息，例如发送到 DeepSeek API
                response: ChatCompletion = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL, messages=self._history
                )
                if response is not None:
                    response_json_text = json.dumps(
                        response.model_dump(), ensure_ascii=False, indent=4
                    )
                    # logger.debug(f"DeepSeek:\n {response_json_text}")
                    _response_content: str = str(response.choices[0].message.content)
                    # logger.debug(f"DeepSeek:\n{_response_content}")
                    if _response_content is not None:
                        self._history.append(
                            {
                                "role": "assistant",
                                "content": _response_content,
                            }
                        )
                        self._save_history_to_file()
                        from launcher_module.launcher_main import (
                            display_deepseek_response,
                        )

                        display_deepseek_response(_response_content)
            except queue.Empty:
                continue

    def set_app_running_flag(self, flag: bool) -> None:
        """设置应用程序运行标志"""
        self._app_is_running = flag
