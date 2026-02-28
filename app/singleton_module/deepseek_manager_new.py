import queue
import threading
import json
from pathlib import Path
from concurrent.futures import Future
from openai import OpenAI

from .singleton_meta import SingletonMeta
from static_module import (
    DEEPSEEK_API_KEY,
    RUNTIME_TIMESTAMP,
    CHAT_HISTORY_DIR,
)
from utility_module import logger


class DeepSeekManager(metaclass=SingletonMeta):

    def __init__(self, debug_mode=False):

        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )

        self._system_prompt = (
            "you are a helpful assistant for data analysis. Please answer the question using english."
        )

        self._history_lock = threading.Lock()
        self._history = [
            {"role": "system", "content": self._system_prompt}
        ]

        self.message_queue: queue.Queue = queue.Queue()

        self._prepare_history_file()

        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self.worker.start()

    # ========================
    # public API
    # ========================

    def chat(self, message: str, timeout=None) -> str:
        """
        同步调用（推荐）
        """

        future = Future()
        self.message_queue.put((message, future))

        return future.result(timeout=timeout)

    # ========================
    # worker
    # ========================

    def _worker_loop(self):

        while True:
            message, future = self.message_queue.get()

            try:
                with self._history_lock:
                    self._history.append(
                        {"role": "user", "content": message}
                    )

                    response = self.client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=self._history,
                    )

                    content = (
                        response.choices[0]
                        .message.content
                    )

                    self._history.append(
                        {"role": "assistant", "content": content}
                    )

                self._save_history()

                future.set_result(content)

            except Exception as e:
                logger.error(e, exc_info=True)
                future.set_exception(e)

    # ========================
    # history
    # ========================

    def _prepare_history_file(self):

        path = Path.cwd() / CHAT_HISTORY_DIR
        path.mkdir(exist_ok=True)

        self.history_file = (
            path /
            f"{RUNTIME_TIMESTAMP:%Y%m%d_%H%M%S}.json"
        )

    def _save_history(self):

        with self._history_lock:
            data = self._history

        with open(self.history_file, "w",
                  encoding="utf-8") as f:
            json.dump(data, f,
                      ensure_ascii=False,
                      indent=2)

deepseek_manager_new = DeepSeekManager()