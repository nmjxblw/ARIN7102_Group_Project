"""
静态参数配置模块

从.env文件加载静态参数，供项目其他部分使用。
"""

# 系统/第三方模块导入
import dotenv
import os
import datetime

RUNTIME_TIMESTAMP: datetime.datetime = datetime.datetime.now()
""" 程序运行时间戳 """
RUNTIME_TIMESTAMP_STR: str = RUNTIME_TIMESTAMP.strftime("%Y%m%d_%H%M%S")
""" 程序运行时间戳字符串格式 """

assert dotenv.load_dotenv(), "加载.env文件失败，请确保文件存在且格式正确。"

PROJECT_NAME: str = os.getenv("PROJECT_NAME", "UnknownProject")
""" 项目名称 """

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
""" 日志级别 """
THREAD_TIMEOUT: float = float(os.getenv("THREAD_TIMEOUT", "5.0"))
""" 线程超时时间（秒）,默认5秒 """

# region DeepSeek API 相关设定
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
""" DEEPSEEK API 密钥 """
assert (
    DEEPSEEK_API_KEY is not None and DEEPSEEK_API_KEY != ""
), "DEEPSEEK_API_KEY未在.env文件中设置"
CHAT_HISTORY_DIR: str = os.getenv(
    "CHAT_HISTORY_DIR", "remote_llm_module/chat_histories"
)
""" 对话历史记录文件路径 """
# endregion
