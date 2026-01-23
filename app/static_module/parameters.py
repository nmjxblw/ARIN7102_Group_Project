import dotenv
import os


assert dotenv.load_dotenv(), "Failed to load .env file"

PROJECT_NAME = os.getenv("PROJECT_NAME", "UnknownProject")
""" 项目名称 """
API_KEY = os.getenv("API_KEY", "")
""" API 密钥 """
