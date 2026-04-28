# 系统/第三方模块导入
import os
import json
import asyncio
from typing import Any, Optional, Dict
from pathlib import Path
from openai import AsyncOpenAI  # 关键：改用异步客户端以适配 FastAPI

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
    """DeepSeek 管理器单例类 (异步适配 Web 版)"""

    def __init__(self, *, debug_mode: bool = False):
        self._initialized: bool = False
        # 初始化异步客户端
        self.aclient = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
        )
        self._history: list = []
        self.history_file: Path
        self._system_prompt: str = rf"""You are a medical consultant robot."""
        self._debug_mode: bool = debug_mode
        self.default_prompt_folder_path: str = DEFAULT_PROMPT_FOLDER_PATH
        self.default_prompt: str = ""
        
        if self._debug_mode:
            logger.info("DeepSeekManager 已进入调试模式")
            
        self._initialize()

    def _initialize(self) -> None:
        """初始化：加载提示词和历史记录"""
        if not self._initialized:
            self._load_prompts()
            self._load_history_from_file()
            # 注意：在 Web 模式下不再需要启动 _deepseek_background_task 线程
            self._initialized = True
            logger.debug("DeepSeek 管理器已完成异步模式初始化。")

    def _load_prompts(self) -> None:
        """(逻辑保持不变) 加载默认的 markdown 提示词模板"""
        default_prompt_path = Path(self.default_prompt_folder_path)
        prompt_file: Optional[Path] = None
        for root, dirs, files in os.walk(default_prompt_path):
            for file in files:
                if file.endswith(".md") and file.startswith("default"):
                    prompt_file = Path(root) / file
                    break
        
        if not prompt_file or not prompt_file.exists():
            logger.error("未找到提示词文件")
            return

        with open(prompt_file, "r", encoding="utf-8") as f:
            self.default_prompt = f.read()

    def _load_history_from_file(self) -> None:
        """(逻辑简化) 初始化对话历史"""
        chat_history_dir = Path.cwd() / CHAT_HISTORY_DIR
        chat_history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = chat_history_dir / f"chat_{RUNTIME_TIMESTAMP.strftime('%Y%m%d_%H%M%S')}.json"
        
        # 初始载入系统提示词
        self._history = [{"role": "system", "content": self._system_prompt}]

    def _save_history_to_file(self) -> None:
        """(逻辑保持不变) 保存记录"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self._history, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存历史记录失败: {e}")

    def _prompt_build(self, input_obj: Dict[str, Any]) -> str:
        """构建提示词内容"""
        sentences = input_obj.get("sentences", "")
        if not sentences: return ""
        
        return self.default_prompt.format(
            sentences=sentences,
            pipeline_output=input_obj.get("pipeline_output", {}),
            bert_output=input_obj.get("bert_output", {}),
        )

    async def generate_response(self, input_object: Dict[str, Any]) -> str:
        """
        供 FastAPI 调用的异步核心方法
        
        Args:
            input_object: 包含用户输入、BERT结果和推荐结果的字典
        Returns:
            AI 生成的文本回复
        """
        # 1. 组装输入
        input_content = self._prompt_build(input_object)
        if not input_content:
            return "收到空输入，无法处理。"

        # 2. 加入历史记录
        self._history.append({"role": "user", "content": input_content})

        try:
            # 3. 异步请求 DeepSeek API
            response = await self.aclient.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=self._history
            )
            
            ai_content = response.choices[0].message.content
            
            # 4. 更新回复历史并持久化
            if ai_content:
                self._history.append({"role": "assistant", "content": ai_content})
                self._save_history_to_file()
                return ai_content
            
            return "AI 未生成内容。"

        except Exception as e:
            logger.error(f"DeepSeek API 请求出错: {e}")
            return f"系统繁忙，请稍后再试。错误详情: {str(e)}"