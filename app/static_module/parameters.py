"""
静态参数配置模块

从.env文件加载静态参数，供项目其他部分使用。
"""

# 系统/第三方模块导入
import dotenv
import os
import datetime
from types import SimpleNamespace

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
    "CHAT_HISTORY_DIR", r"remote_llm_module/chat_histories"
)
""" 对话历史记录文件路径 """
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
""" DeepSeek模型名称，默认为"deepseek-v4-flash" """


DEFAULT_PROMPT_FOLDER_PATH: str = os.getenv(
    "DEFAULT_PROMPT_FOLDER_PATH", r"remote_llm_module/prompts"
)
""" 默认提示词文件夹路径 """
# endregion

# region 数据集相关设定
KAGGLE_DATASET_DOWNLOAD_URLS_FILE: str = os.getenv(
    "KAGGLE_DATASET_DOWNLOAD_URLS_FILE",
    r"dataset_module/kaggle_dataset_download_urls.json",
)
""" Kaggle数据集下载URL列表文件路径 """

DATABASE_FILE: str = os.getenv("DATABASE_FILE", r"database_module/database.db")
""" 数据库文件路径 """

CLINICAL_BERT: str = os.getenv("CLINICAL_BERT", r"deployment_module/clinicalbert_local")
""" 本地临床BERT模型文件夹路径 """
DISEASES_SYMPTOM_DICT: str = os.getenv(
    "DISEASES_SYMPTOM_DICT",
    r"dataset_module/disease-symptom-description-dataset/disease_symptoms_dict.json",
)
""" 疾病症状字典文件路径 """
# endregion

# region BERT模型训练参数
try:
    import torch

    torch_version = getattr(torch, "version", SimpleNamespace(cuda=None))
    print(
        f"检测到torch库，版本: {torch.__version__}, CUDA支持: {torch_version.cuda is not None}"
    )
    TORCH_DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    """ Torch训练设备，优先使用GPU """
    USE_CUDA: bool = (
        TORCH_DEVICE.type == "cuda" and os.getenv("USE_CUDA", "true").lower() == "true"
    )
    """ 是否使用CUDA进行训练 """

except ImportError as e:
    print("未安装torch库。")
    raise e

BERT_TRAINING_DATASET_FOLDER: str = os.getenv(
    "BERT_TRAINING_DATASET_FOLDER", r"dataset_module/bert_training_dataset"
)
""" BERT训练数据集文件夹路径，包含训练数据和标签文件 """

# endregion


# region 药物训练数据集参数
DRUGS_TRAINING_DATASET_FOLDER: str = os.getenv(
    "DRUGS_TRAINING_DATASET_FOLDER", r"dataset_module/drugs_training_dataset"
)
""" 药物训练数据集文件夹路径，包含训练数据和标签文件 """
# endregion

# region 本地模型
GIT_BERT_MODEL_URL: str = os.getenv(
    "GIT_BERT_MODEL_URL", r"https://huggingface.co/medicalai/ClinicalBERT.git "
)
""" Git BERT模型仓库URL"""
DEPLOYMENT_FOLDER: str = os.getenv(
    "DEPLOYMENT_FOLDER", r"deployment_module/clinicalbert_local"
)
""" 本地 BERT 模型部署路径 """
TRAINED_BERT_SAVE_PATH: str = os.getenv(
    "TRAINED_BERT_SAVE_PATH", r"deployment_module/trained_bert"
)
""" 训练后 BERT 模型保存路径 """

# endregion
