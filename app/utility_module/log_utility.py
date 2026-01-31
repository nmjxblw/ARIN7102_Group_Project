"""
高级日志配置模块
支持按日期创建文件夹，按时间命名日志文件
"""

# 系统/第三方模块导入
import os
import sys
from io import TextIOWrapper
import logging
from pathlib import Path
from datetime import datetime
from types import FrameType, TracebackType
from typing import Optional, Tuple, Type

# 本地模块导入
from static_module import PROJECT_NAME, LOG_LEVEL

ProjectName: str = PROJECT_NAME
"""项目名称，用于日志记录器名称"""
LogLevel: int = int(getattr(logging, LOG_LEVEL))
"""日志级别"""


class _FileHandler(logging.Handler):
    """
    自定义日志处理器
    按日期创建目录，按启动时间创建日志文件
    """

    def __init__(
        self,
        base_dir: str = "logs",
        encoding: str = "utf-8",
        level: int = LogLevel,
    ):
        """
        初始化处理器

        参数:
            base_dir: 日志根目录
            encoding: 文件编码
            level: 日志级别
        """
        super().__init__(level)

        self.base_dir: Path = Path(base_dir)
        """日志根目录"""
        self.encoding: str = encoding
        """文件编码"""
        self.stream: Optional[TextIOWrapper] = None
        """日志文件流"""
        self.current_date: Optional[str] = None
        """当前日志日期"""
        self.log_file_path: Optional[Path] = None
        """当前日志文件路径"""
        # 在初始化时就创建日志文件
        self._setup_log_file()

    def _setup_log_file(self):
        """设置日志文件"""
        # 获取当前日期和时间
        app_timestamp: datetime = datetime.now()
        """应用启动时间"""
        date_str: str = app_timestamp.strftime(r"%Y%m%d")
        """当前日期字符串"""
        time_str: str = app_timestamp.strftime(r"%H%M%S_%f")[:-3]  # 毫秒保留3位
        """当前时间字符串"""
        # 创建日期文件夹
        date_dir: Path = self.base_dir / date_str
        """日期目录路径"""
        date_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志文件路径
        self.log_file_path = date_dir / f"{time_str}.log"
        self.current_date = date_str

        # 打开文件流
        self.stream: Optional[TextIOWrapper] = open(
            self.log_file_path, "a", encoding=self.encoding
        )

        print(f"✓ 日志文件已创建: {self.log_file_path}")

    def emit(self, record):
        """
        输出日志记录

        参数:
            record: 日志记录对象
        """
        try:
            # 检查日期是否改变（跨天处理）
            current_date = datetime.now().strftime("%Y%m%d")
            if current_date != self.current_date:
                # 关闭旧文件
                if self.stream:
                    self.stream.close()
                    self.stream = None
                # 创建新文件
                self._setup_log_file()

            # 确保 stream 已准备好（避免类型检查器认为可能为 None）
            if self.stream is None:
                self._setup_log_file()

            # 格式化并写入日志
            msg: str = self.format(record)
            if self.stream:
                self.stream.write(msg + "\n")
                self.stream.flush()

        except Exception:
            self.handleError(record)

    def close(self):
        """关闭处理器"""
        if self.stream:
            self.stream.close()
        super().close()


class LoggerConfig:
    """日志配置类"""

    def __init__(
        self,
        base_dir: str = "logs",
        log_format: str = r"[%(asctime)s.%(msecs)03d][%(pathname)s:%(lineno)d][%(levelname)s]"
        + os.linesep
        + r"%(message)s"
        + os.linesep,
        date_format: str = r"%Y-%m-%d %H:%M:%S",
        level: int = LogLevel,
        encoding: str = "utf-8",
        console_output: bool = True,
    ):
        """
        初始化日志配置

        参数:
            base_dir: 日志根目录
            log_format: 日志格式
            date_format: 时间格式
            level: 日志级别
            encoding: 文件编码
            console_output: 是否同时输出到控制台
        """
        self.base_dir = base_dir
        """日志根目录"""
        self.log_format = log_format
        """日志格式"""
        self.date_format = date_format
        """时间格式"""
        self.level = level
        """日志级别"""
        self.encoding = encoding
        """文件编码"""
        self.console_output = console_output
        """是否同时输出到控制台"""

        # 确保日志根目录存在
        Path(base_dir).mkdir(parents=True, exist_ok=True)

    def get_logger(self, name: str = ProjectName) -> logging.Logger:
        """
        获取配置好的日志记录器

        参数:
            name: 日志记录器名称

        返回:
            配置好的日志记录器
        """
        # 创建日志记录器
        logger = logging.getLogger(name)
        """日志记录器对象"""
        logger.setLevel(self.level)

        # 清除已有的处理器（避免重复）
        logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter(fmt=self.log_format, datefmt=self.date_format)
        """日志格式化器"""
        # 添加文件处理器
        file_handler = _FileHandler(
            base_dir=self.base_dir, encoding=self.encoding, level=self.level
        )
        """文件处理器"""
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 添加控制台处理器（可选）
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger


def setup_logger(
    name: str = "AppLogger",
    base_dir: str = "logs",
    level: int = LogLevel,
    console_output: bool = True,
) -> logging.Logger:
    """
    快速设置日志记录器的便捷函数

    参数:
        name: 日志记录器名称
        base_dir: 日志根目录
        level: 日志级别
        console_output: 是否同时输出到控制台

    返回:
        配置好的日志记录器

    示例:
        logger = setup_logger("MyApp", "logs", logging.INFO)
        logger.info("这是一条日志")
    """
    config = LoggerConfig(base_dir=base_dir, level=level, console_output=console_output)
    """日志配置对象"""
    return config.get_logger(name)


# 全局默认日志记录器
logger: logging.Logger = setup_logger(name=ProjectName, level=LogLevel)
"""全局默认日志记录器对象"""


def get_default_logger(default_name: str = ProjectName) -> logging.Logger:
    """
    获取全局默认日志记录器（单例模式）

    返回:
        全局日志记录器
    """
    global logger
    if logger is None:
        logger = setup_logger(name=default_name, level=LogLevel)
    return logger


# region 全局未捕获异常处理钩子

_exception_handler_initialized: bool = False
"""全局异常处理器初始化标志"""


def is_fatal_error(exc_type: Type[BaseException], exc_value: BaseException) -> bool:
    """判断异常是否为致命错误"""
    if issubclass(
        exc_type,
        (
            SystemExit,
            KeyboardInterrupt,
            MemoryError,
            SystemError,
            RecursionError,
            ImportError,
        ),
    ):
        return True

    fatal_keywords = ["fatal", "critical", "memory", "system", "recursion"]
    error_msg = str(exc_value).lower()
    if any(keyword in error_msg for keyword in fatal_keywords):
        return True

    return False


def _setup_custom_exception_hook():
    """设置自定义的全局异常钩子以记录未捕获的异常"""
    global _exception_handler_initialized
    if _exception_handler_initialized:
        return
    _exception_handler_initialized = True
    global logger

    def handle_exception(
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_traceback: Optional[TracebackType],
    ):
        """处理未捕获的异常"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 允许键盘中断正常退出
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        # 提取异常位置信息
        filename, lineno = _extract_exception_location(exc_traceback)

        # 记录异常信息[1](@ref)
        logger.error(
            "\n捕获异常:[%s: %s] 在 [%s:%s]",
            exc_type.__name__,
            str(exc_value),
            filename,
            lineno,
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        if is_fatal_error(exc_type, exc_value):
            logger.critical("检测到致命错误，程序将终止。")
            sys.exit(1)

    sys.excepthook = handle_exception
    logger.debug("全局未捕获异常处理钩子已设置。")


def _extract_exception_location(
    exc_traceback: Optional[TracebackType],
) -> Tuple[str, int]:
    """提取异常位置信息"""
    filename: str = "<unknown>"
    lineno: int = -1
    if exc_traceback is not None:
        last_tb: TracebackType = exc_traceback
        while last_tb.tb_next is not None:
            last_tb = last_tb.tb_next
        frame: FrameType = last_tb.tb_frame
        filename = os.path.basename(frame.f_code.co_filename)  # 只取文件名
        lineno = last_tb.tb_lineno
    return filename, lineno


_setup_custom_exception_hook()
# endregion
