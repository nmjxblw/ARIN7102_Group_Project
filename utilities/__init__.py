__doc__ = """ 开发者工具模组 """
import dotenv

dotenv.load_dotenv()

from .logger import logger

__all__ = ["logger"]
