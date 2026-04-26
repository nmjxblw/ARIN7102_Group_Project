"""
静态资源模块

在这里存放项目所需的静态资源，

如静态参数配置、静态类定义（JSON序列化）。
"""

from .parameters import *
from .classes import *
from .enums import *

__all__ = [
    # Parameters
    "RUNTIME_TIMESTAMP",
    "RUNTIME_TIMESTAMP_STR",
    "PROJECT_NAME",
    "DEEPSEEK_API_KEY",
    "LOG_LEVEL",
    "CHAT_HISTORY_DIR",
    "DEEPSEEK_MODEL",
    "THREAD_TIMEOUT",
    "KAGGLE_DATASET_DOWNLOAD_URLS_FILE",
    "DATABASE_FILE",
    "CLINICAL_BERT",
    "GIT_BERT_MODEL_URL",
    "DISEASES_SYMPTOM_DICT",
    "TORCH_DEVICE",
    "USE_CUDA",
    "BERT_TRAINING_DATASET_FOLDER",
    "DRUGS_TRAINING_DATASET_FOLDER",
    # Classes
    "AppAsyncTask",
    # Enums
    "TaskStatus",
]
