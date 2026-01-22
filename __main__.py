from utilities import logger
import os


logger.info("程序启动，当前工作目录: %s", os.getcwd())
logger.info("项目名称: %s", os.getenv("ProjectName", "未知项目"))
