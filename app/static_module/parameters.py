"""
静态参数配置模块

从.env文件加载静态参数，供项目其他部分使用。
"""

import dotenv
import os


assert dotenv.load_dotenv(), "加载.env文件失败，请确保文件存在且格式正确。"

PROJECT_NAME = os.getenv("PROJECT_NAME", "UnknownProject")
""" 项目名称 """
API_KEY = os.getenv("API_KEY", "")
""" API 密钥 """
assert API_KEY is not None and API_KEY != "", "API_KEY未在.env文件中设置"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
