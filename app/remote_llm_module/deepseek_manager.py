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
        # 调用初始化函数
        self._initialize()

    def _initialize(self) -> None:
        """初始化函数"""
        if not self._initialized:
            self._load_history_from_file()
            self._deepseek_background_thread = threading.Thread(
                target=self._deepseek_background_task,
                daemon=True,  # 设置为守护线程，随主程序退出而自动结束
                name="DeepSeekBackgroundThread",
            )
            self._deepseek_background_thread.start()
            self._initialized = True
            logger.debug("DeepSeek 管理器已初始化。")

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
        sentences: str = getattr(input, "sentences", "")
        if not sentences:
            return ""
        pipeline_output: dict = getattr(input, "pipeline_output", {})
        bert_output: dict = getattr(input, "bert_output", {})
        prompt: str = rf"""
# Role
You are a professional, rigorous, and empathetic AI pharmaceutical assistant. Your task is to generate a safe, clear, and easy-to-understand medication recommendation plan for users based on their illness descriptions, system-inferred symptom/disease labels, and the database-matched candidate drug list.

# Input
You will receive a conversation containing user input, formatted as follows:
{sentences}

As well as a system-processed data set containing an analysis of the user's symptoms from the text and the system's confidence level regarding the disease/symptom:
{bert_output}

Also a response of drug system, the corresponding drugs and related information that were matched are as follows:
{pipeline_output}

# Task
Please generate a structured response to the user based on the above input. You must strictly adhere to the following rules:

1. **Data Desensitization and Value Concealment (Top Priority)**:
   - **Absolutely Prohibited**: Do not expose any underlying numerical information of labels in the response (e.g., match rate 0.95, weight 80%, ranking order, confidence level, etc.).
   - Convert system data into natural and fluent dialogue. Avoid saying "The system disease label triggered X," and instead phrase it as "Based on your description, it may be related to X."

2. **Transparent Explanation of Recommendation Basis**:
   - Clearly inform the user of the reasoning behind the recommended medication. Explain the alignment between `inferred_tags` (disease/symptom labels) and `drug_candidates` (drug indications).
   - Example: "We recommend [Drug A] because it directly alleviates [Symptom X] and the potential [Disease Y] inferred from your description."

3. **Tone and Wording Guidelines**:
   - Maintain empathy and objectivity to reassure the user.
   - **Avoid Overdiagnosis**: Use cautious phrasing such as "may be related to..." or "exhibits characteristics of..." rather than absolute statements like "you are diagnosed with..." or "will definitely cure."

5. **Mandatory Medical Disclaimer**:
   - The response must conclude with a standard disclaimer emphasizing that AI recommendations cannot replace a face-to-face consultation with a professional doctor.
   
   
# Output Format
Reply ONLY with a JSON array.
Do NOT output explanations, markdown, or text outside JSON.

Drug recommendation plan, including the following fields:
- `drug_name`: Recommended drugs/compound medications. The drug names must be strictly output as matched by the system, without any rewriting (e.g., synonym substitution, abbreviations, etc.). For compound drugs, use the compound drug name matched by the system, and do not split it into single-component drugs.
- `drug_preference`: The appropriateness of the drug in the current conversation context (e.g., highly recommended, recommended, optional, not recommended), to be evaluated based on the match between the system-inferred symptom/disease labels and the drug's indications.
- `recommendation_reasoning`:Detailed explanation of the recommendation rationale, which must include an explanation of the relationship between the system-inferred symptom/disease labels and the drug's indications.
```

Example:
[
    {{
        "drug_name": "Drug A",
        "drug_preference": "highly recommended",
        "recommendation_reasoning": "We recommend Drug A because it directly alleviates Symptom X and the potential Disease Y inferred from your description."
    }},
    {{
        "drug_name": "Drug B",
        "drug_preference": "recommended",
        "recommendation_reasoning": "Drug B is quite suitable for your current condition and is a common choice for treating Disease Y. It is an over-the-counter medication, so you can purchase it at a pharmacy without a doctor's prescription."
    }}
]
"""
        return prompt

    def send(self, input_object: object) -> None:
        """发送消息到 DeepSeek"""
        self.message_queue.put(input_object)

    def _deepseek_background_task(self):
        """DeepSeek 后台任务处理函数"""
        while True:
            try:
                if self.message_queue.empty():
                    continue
                input_object: object = self.message_queue.get(block=False, timeout=1)
                # logger.debug(f"正在处理用户输入: {input_message}")

                input_content: str = self._prompt_build(input_object)
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
                    # logger.info(f"DeepSeek:\n{_response_content}")
                    if _response_content is not None:
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
