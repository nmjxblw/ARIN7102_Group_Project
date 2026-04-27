from __future__ import annotations

import json
import math
import re
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    DistilBertConfig,
    DistilBertTokenizer,
    DistilBertModel,
    DistilBertPreTrainedModel,
    TrainingArguments,
    Trainer,
    PretrainedConfig,
    EvalPrediction,
    BatchEncoding,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils.generic import ModelOutput
from typing import Any, Optional, Tuple, cast
from pathlib import Path
import subprocess
import os

from singleton_module import SingletonMeta
from static_module import (
    DEPLOYMENT_FOLDER,
    TRAINED_BERT_SAVE_PATH,
    TORCH_DEVICE,
    USE_CUDA,
    BERT_TRAINING_DATASET_FOLDER,
    GIT_BERT_MODEL_URL,
)
from utility_module import logger

# ====================== 1. 配置与数据准备 ======================
INITIAL_BERT_MODEL_PATH: Path = Path.cwd() / DEPLOYMENT_FOLDER
""" 原始BERT模型路径，包含模型权重和配置文件。请确保该路径下有 transformers 兼容的模型文件（如 pytorch_model.bin 和 config.json） """
INITIAL_BERT_MODEL_PATH.mkdir(parents=True, exist_ok=True)  # 确保模型路径存在
SAVE_PATH: Path = Path.cwd() / TRAINED_BERT_SAVE_PATH
""" 训练后模型保存路径，训练完成后模型和标签编码器将保存在此路径下 """
MAX_LEN: int = 256
""" 模型输入的最大长度，超过部分将被截断。根据你的数据特点调整，过长可能导致训练效率降低，过短可能丢失信息。 """
BATCH_SIZE: int = 8
""" 训练批次大小，根据你的 GPU 内存调整，过大可能导致 OOM，过小可能导致训练不稳定。 """
EPOCHS: int = 10
""" 训练轮数，根据你的数据量和模型复杂度调整，过多可能导致过拟合，过少可能导致欠拟合。 """
LEARNING_RATE: float = 3e-5
""" 学习率，建议使用较小的学习率进行微调，过大可能导致训练不稳定，过小可能导致收敛过慢。 """
# ===== 疾病标签优化参数 =====
DISEASE_LOSS_WEIGHT: float = 2.5
""" 疾病任务损失权重，增加以提高疾病预测准确率（相对于其他任务）"""
SYMPTOM_LOSS_WEIGHT: float = 1.0
""" 症状任务损失权重 """
FIRST_AID_LOSS_WEIGHT: float = 0.5
""" 急救任务损失权重 """
DISEASE_CLASSIFIER_LR: float = 5e-4
""" 疾病分类头学习率，比其他头更高以加快收敛 """
SYMPTOM_CLASSIFIER_LR: float = 1e-3
""" 症状分类头学习率 """
FIRST_AID_CLASSIFIER_LR: float = 1e-3
""" 急救分类头学习率 """
DISEASE_POSITIVE_WEIGHT: float = 1.2
""" 疾病标签的正类权重，处理类别不平衡 """
# ==================
RAW_DATA_PATH: Path = (
    Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "generated_medical_dataset.json"
)
""" 原始数据路径 """
DISEASE_LABELS_PATH: Path = (
    Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "disease_labels.json"
)
""" 疾病标签路径 """
SYMPTOM_LABELS_PATH: Path = (
    Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "symptom_labels.json"
)
""" 症状标签路径 """
USE_FP16: bool = USE_CUDA
""" 是否使用混合精度训练 """
if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass
class MultitaskSequenceClassifierOutput(ModelOutput):
    """多任务分类模型的输出，包含损失、三个分类头的 logits，以及可选的隐藏状态和注意力权重。"""

    loss: Optional[torch.Tensor] = None
    logits: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


# 处理数据集
class MultitaskDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        indices,
        data: list[dict[str, Any]],
        tokenizer: DistilBertTokenizer,
        disease_label_binarizer: MultiLabelBinarizer,
        symptom_label_binarizer: MultiLabelBinarizer,
        max_len: int,
    ) -> None:
        """数据集类，负责将原始数据转换为模型输入格式，包括文本的分词和标签的编码。
        Args:
            data: 原始数据列表，每个元素是一个包含 "sentences", "disease", "symptoms", "need_first_aid" 等字段的字典。
            tokenizer: DistilBertTokenizer 实例，用于文本分词。
            max_len: 模型输入的最大长度，超过部分将被截断。
        """
        self.data: list[dict[str, Any]] = data
        self.tokenizer: DistilBertTokenizer = tokenizer
        self.max_len: int = max_len
        self.indices = indices
        self.disease_label_binarizer = disease_label_binarizer
        self.symptom_label_binarizer = symptom_label_binarizer

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, id: int) -> dict[str, torch.Tensor]:
        idx = self.indices[id]
        item: dict[str, Any] = self.data[idx]
        text: str = item.get("sentences", "")
        if not text:
            raise ValueError(f"数据项缺少 'sentences' 字段或其值为空: {item}")
        # 分词
        encoding: BatchEncoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # 编码标签
        labels_disease: torch.FloatTensor = torch.FloatTensor(
            self.disease_label_binarizer.transform([item["disease"]])[0]
        )
        labels_symptom: torch.FloatTensor = torch.FloatTensor(
            self.symptom_label_binarizer.transform([item["symptoms"]])[0]
        )
        labels_first_aid: torch.FloatTensor = torch.FloatTensor(
            [item["need_first_aid"]]
        )  # 二分类用float方便BCEWithLogitsLoss

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long).squeeze(
                0
            ),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ).squeeze(0),
            "labels_disease": labels_disease,
            "labels_symptom": labels_symptom,
            "labels_first_aid": labels_first_aid,
        }


# # ====================== 多任务模型 ======================
class DistilBertForMultitaskLearning(DistilBertPreTrainedModel):
    """自定义多任务学习模型，基于 DistilBERT，包含三个分类头：疾病预测、症状预测和是否需要急救的二分类。"""

    def __init__(
        self,
        config: PretrainedConfig,
        disease_label_binarizer: MultiLabelBinarizer,
        symptom_label_binarizer: MultiLabelBinarizer,
    ) -> None:
        super().__init__(config)
        self.distil_bert: DistilBertModel = DistilBertModel(config)

        # transformers 5.x 的 from_pretrained 会调用 all_tied_weights_keys.keys()，
        # 必须显式设为空 dict（本模型无权重共享）。
        self.all_tied_weights_keys: dict[str, str] = {}

        self.disease_label_binarizer: MultiLabelBinarizer = disease_label_binarizer
        self.symptom_label_binarizer: MultiLabelBinarizer = symptom_label_binarizer

        # 分类头的维度
        self.num_diseases: int = len(self.disease_label_binarizer.classes_)
        self.num_symptoms: int = len(self.symptom_label_binarizer.classes_)
        self.hidden_size: int = config.hidden_size  # 768

        # 三个独立的分类头
        self.classifier_disease: nn.Linear = nn.Linear(
            self.hidden_size, self.num_diseases
        )
        self.classifier_symptom: nn.Linear = nn.Linear(
            self.hidden_size, self.num_symptoms
        )
        self.classifier_first_aid: nn.Linear = nn.Linear(self.hidden_size, 1)  # 二分类

        # 损失函数
        self.bce_with_logits_loss: nn.BCEWithLogitsLoss = (
            nn.BCEWithLogitsLoss()
        )  # 用于多标签和二分类

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_disease=None,
        labels_symptom=None,
        labels_first_aid=None,
        **kwargs,
    ) -> MultitaskSequenceClassifierOutput:
        # DistilBERT 前向传播
        outputs: BaseModelOutput = self.distil_bert(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        # 取  token 的输出 (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        assert outputs.last_hidden_state is not None, "模型输出缺少 last_hidden_state"
        pooled_output: torch.Tensor = outputs.last_hidden_state[:, 0, :]

        # 三个头的输出
        logits_disease: torch.Tensor = self.classifier_disease(pooled_output)
        logits_symptom: torch.Tensor = self.classifier_symptom(pooled_output)
        logits_first_aid: torch.Tensor = self.classifier_first_aid(pooled_output)

        # 计算损失 - 使用加权的多任务学习
        loss: torch.Tensor | None = None  # 默认无损失
        if labels_disease is not None:
            loss_disease: torch.Tensor = self.bce_with_logits_loss(
                logits_disease, labels_disease
            )
            loss_symptom: torch.Tensor = self.bce_with_logits_loss(
                logits_symptom, labels_symptom
            )
            loss_first_aid: torch.Tensor = self.bce_with_logits_loss(
                logits_first_aid, labels_first_aid
            )
            # 对疾病正样本应用额外权重，提高疾病预测精度
            disease_weight = torch.where(
                labels_disease == 1,
                torch.full_like(labels_disease, DISEASE_POSITIVE_WEIGHT),
                torch.ones_like(labels_disease),
            )
            loss_disease = (loss_disease * disease_weight).mean()
            # 使用不同的权重组合三个损失
            loss = (
                DISEASE_LOSS_WEIGHT * loss_disease
                + SYMPTOM_LOSS_WEIGHT * loss_symptom
                + FIRST_AID_LOSS_WEIGHT * loss_first_aid
            )

        return MultitaskSequenceClassifierOutput(
            loss=loss,
            logits=(logits_disease, logits_symptom, logits_first_aid),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BERTManager(metaclass=SingletonMeta):
    """本地 BERT 模型管理器实例"""

    def __init__(self, *, debug_mode=False) -> None:
        self._debug_mode = debug_mode
        """ 调试模式标志，启用后会在推理过程中输出更多内部状态信息，帮助分析模型行为 """
        self._is_initialized: bool = False
        """ 是否已初始化标志，确保模型和相关组件只加载一次 """
        self._device: torch.device = TORCH_DEVICE
        """ 模型运行设备"""
        self._need_pretrain: bool = True
        """ 是否需要预训练，初始为 True，首次调用 predict 时如果模型文件缺失会自动下载预训练模型并设置为 False，避免重复下载 """
        self._raw_training_data: Optional[list[dict[str, Any]]] = None
        """ 原始训练数据实例"""
        self._train_dataset: Optional[MultitaskDataset] = None
        """ 训练数据集实例，首次调用 predict 时加载并缓存以加快后续推理速度 """
        self._val_dataset: Optional[MultitaskDataset] = None
        """ 验证数据集实例，首次调用 predict 时加载并缓存以加快后续推理速度 """
        self._bert_config: Optional[PretrainedConfig] = None
        """ BERT 模型配置实例，初始为 None """
        self._bert_model: Optional[DistilBertForMultitaskLearning] = None
        """ BERT 模型实例，初始为 None"""
        self._tokenizer: Optional[DistilBertTokenizer] = None
        """ BERT 分词器实例，初始为 None"""
        self._disease_label_binarizer: Optional[MultiLabelBinarizer] = None
        """ 疾病标签二值化器，适用于多标签分类，将疾病列表转换为多热编码 """
        self._symptom_label_binarizer: Optional[MultiLabelBinarizer] = None
        """ 症状标签二值化器，适用于多标签分类，将症状列表转换为多热编码 """
        self._train_dataset: Optional[MultitaskDataset] = None
        """ 训练数据集实例，首次调用 predict 时加载并缓存以加快后续推理速度 """
        self._val_dataset: Optional[MultitaskDataset] = None
        """ 验证数据集实例，首次调用 predict 时加载并缓存以加快后续推理速度 """
        self._medians: Optional[dict] = None
        """ 中位数实例，初始为 None，首次调用 predict 时加载并缓存以加快后续推理速度 """
        self._initialize_bert()

    def _initialize_bert(self):
        """初始化管理器，确保预训练模型文件存在并加载模型、分词器和标签编码器。该方法在实例化时调用，并且只会执行一次以避免重复加载。"""
        if self._is_initialized:
            return  # 已初始化，无需重复加载
        if self._debug_mode:
            logger.debug("BERTManager: 正在初始化 BERT 模型和相关组件...")
        if self._check_trained_bert_exists():
            self._need_pretrain = False
        elif not self._check_bert_download():
            if self._debug_mode:
                logger.debug("BERTManager: 本地预训练模型文件缺失，正在下载...")
            success = self._download_bert_from_git_url()
            if not success:
                raise RuntimeError("无法下载预训练模型，请检查网络连接或下载链接。")

        # TODO: 加载分词器、模型和标签编码器，并缓存到实例变量中以加快后续推理速度
        self._get_tokenizer()
        self._load_multilabel_binarizers()
        self._get_bert_config()
        if self._need_pretrain:
            self.train_bert()
        # 预加载模型和相关组件到内存，确保首次调用 predict 时能够快速响应
        self.preload()
        self._is_initialized = True
        if self._debug_mode:
            logger.debug("BERTManager: BERT 模型和相关组件已成功初始化。")

    def _check_bert_download(self) -> bool:
        """检查本地是否存在预训练模型文件，确保推理前模型已正确下载和解压。"""
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "vocab.txt",
        ]
        for file in required_files:
            if not (INITIAL_BERT_MODEL_PATH / file).exists():
                return False
        return True

    def _download_bert_from_git_url(self):
        """从 Hugging Face URL 下载预训练模型，并解压到指定目录。"""
        import shutil

        # 检查 git-xet 命令是否存在
        if shutil.which("git-xet"):
            if self._debug_mode:
                logger.info("检测到 Git-Xet 已安装，跳过下载。")
        else:
            if self._debug_mode:
                logger.info("未检测到 Git-Xet，准备安装...")
            winget_install_command = [
                "winget",
                "install",
                "-e",
                "--id",
                "HuggingFace.Git-Xet",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--silent",
            ]
            try:
                subprocess.run(winget_install_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                if e.returncode == 2316632107:
                    if self._debug_mode:
                        logger.info(
                            "检测到 Git-Xet 已经安装且是最新版本，跳过安装步骤。"
                        )
                else:
                    logger.error(f"安装 Git 失败: {e}")
                    return False
        git_clone_command = rf"git clone {GIT_BERT_MODEL_URL} {INITIAL_BERT_MODEL_PATH}"
        try:
            subprocess.run(git_clone_command, shell=True, check=True)
            if self._debug_mode:
                logger.info(
                    f"成功从 {GIT_BERT_MODEL_URL} 下载并解压模型到 {INITIAL_BERT_MODEL_PATH}"
                )
        except subprocess.CalledProcessError as e:
            logger.error(f"下载模型失败: {e}")
            return False

        return True

    def _check_trained_bert_exists(self) -> bool:
        """检查本地是否存在训练好的模型文件，确保推理前模型已正确训练和保存。"""
        required_files = [
            "config.json",
            "label_encoders.pkl",
            "model.safetensors",
            "tokenizer_config.json",
            "tokenizer.json",
        ]
        for file in required_files:
            if not (SAVE_PATH / file).exists():
                self._need_pretrain = False
                return False
        self._need_pretrain = True
        return True

    # 加载分词器
    @classmethod
    def _get_tokenizer(
        cls,
        model_path: str | Path = INITIAL_BERT_MODEL_PATH,
        local_files_only: bool = False,
    ) -> DistilBertTokenizer:
        """兼容不同 transformers 版本的 Mistral regex 修复参数。"""
        if cls._tokenizer is not None and isinstance(
            cls._tokenizer, DistilBertTokenizer
        ):
            return cls._tokenizer
        try:
            cls._tokenizer = DistilBertTokenizer.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                fix_mistral_regex=True,
            )

        except TypeError as exc:
            if "fix_mistral_regex" in str(exc) and "multiple values" in str(exc):
                cls._tokenizer = DistilBertTokenizer.from_pretrained(
                    model_path,
                    local_files_only=local_files_only,
                )
            raise
        if isinstance(cls._tokenizer, DistilBertTokenizer):
            return cls._tokenizer
        raise RuntimeError(
            f"加载分词器失败，预期类型 DistilBertTokenizer，但实际类型为 {type(cls._tokenizer)}"
        )

    @staticmethod
    def get_runtime_device_info() -> str:
        """获取当前运行设备的信息，帮助确认是否使用了 GPU 以及相关的 CUDA 版本和 GPU 型号。"""
        msg = f"当前设备: {TORCH_DEVICE}\n"
        if USE_CUDA:
            torch_version = getattr(torch, "version", SimpleNamespace(cuda=None))
            msg += f"CUDA 版本: {torch_version.cuda}\n"
            msg += f"GPU: {torch.cuda.get_device_name(0)}\n"
        else:
            msg += "未检测到可用 CUDA，当前将使用 CPU 训练。"

        return msg

    @classmethod
    def _load_multilabel_binarizers(
        cls,
    ) -> tuple[MultiLabelBinarizer, MultiLabelBinarizer]:
        """
        加载疾病和症状标签的 MultiLabelBinarizer 实例，并缓存到类变量中以加快后续训练和推理速度。
        该方法会检查是否已经加载过，如果已经加载过则直接返回，避免重复加载。
        """
        # 判断是否已经加载过，如果加载过直接 return
        if (
            cls._disease_label_binarizer is not None
            and cls._symptom_label_binarizer is not None
        ):
            return cls._disease_label_binarizer, cls._symptom_label_binarizer
        # 提取所有标签
        if not DISEASE_LABELS_PATH.exists():
            raise FileNotFoundError(f"疾病标签文件未找到: {DISEASE_LABELS_PATH}")
        with open(DISEASE_LABELS_PATH, "r", encoding="utf-8") as f:
            all_diseases: list[str] = json.load(f)

        if not SYMPTOM_LABELS_PATH.exists():
            raise FileNotFoundError(f"症状标签文件未找到: {SYMPTOM_LABELS_PATH}")
        with open(SYMPTOM_LABELS_PATH, "r", encoding="utf-8") as f:
            all_symptoms: list[str] = json.load(f)

        # 初始化标签编码器
        cls._disease_label_binarizer = MultiLabelBinarizer().fit([all_diseases])
        cls._symptom_label_binarizer = MultiLabelBinarizer().fit([all_symptoms])

        return cls._disease_label_binarizer, cls._symptom_label_binarizer

    def _load_training_data(self) -> list[dict[str, Any]]:
        """加载训练数据"""
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            raw_data: list[dict[str, Any]] = json.load(f)
        self._raw_training_data = raw_data
        data_length = len(raw_data)
        if data_length == 0:
            raise ValueError(
                f"训练数据文件已加载，但没有任何样本。请检查数据文件[{RAW_DATA_PATH}]是否正确。"
            )
        if self._debug_mode:
            logger.debug(f"已加载训练数据，共 {len(raw_data)} 条样本。")
        return self._raw_training_data

    @classmethod
    def _get_bert_config(cls) -> PretrainedConfig:
        if cls._bert_config is not None and isinstance(
            cls._bert_config, PretrainedConfig
        ):
            return cls._bert_config
        config: PretrainedConfig = AutoConfig.from_pretrained(
            INITIAL_BERT_MODEL_PATH, local_files_only=True
        )
        cls._bert_config = config
        return cls._bert_config

    def _get_bert_model(self) -> DistilBertForMultitaskLearning:
        if self._bert_model is not None and isinstance(
            self._bert_model, DistilBertForMultitaskLearning
        ):
            return self._bert_model
        if self._bert_config is None:
            self._bert_config = self._get_bert_config()
        if (
            self._disease_label_binarizer is None
            or self._symptom_label_binarizer is None
        ):
            self._disease_label_binarizer, self._symptom_label_binarizer = (
                self._load_multilabel_binarizers()
            )
        model = DistilBertForMultitaskLearning(
            self._bert_config,
            self._disease_label_binarizer,
            self._symptom_label_binarizer,
        )
        self._bert_model = model
        return self._bert_model

    def train_bert(self):
        """训练 BERT 模型"""
        self._load_training_data()
        assert self._raw_training_data is not None, "训练数据未加载，无法进行训练"
        # 创建 Dataset
        train_idx, test_idx = train_test_split(
            range(len(self._raw_training_data)), test_size=0.2, random_state=42
        )
        assert self._tokenizer is not None and isinstance(
            self._tokenizer, DistilBertTokenizer
        ), "分词器未加载，无法进行训练"
        assert self._disease_label_binarizer is not None and isinstance(
            self._disease_label_binarizer, MultiLabelBinarizer
        ), "疾病标签编码器未加载，无法进行训练"
        assert self._symptom_label_binarizer is not None and isinstance(
            self._symptom_label_binarizer, MultiLabelBinarizer
        ), "症状标签编码器未加载，无法进行训练"
        # 简单的训练/验证划分（实际请用 train_test_split）
        # logger.debug(f"训练集大小: {len(train_idx)}, 验证集大小: {len(test_idx)}")
        self._train_dataset = MultitaskDataset(
            train_idx,
            self._raw_training_data,
            self._tokenizer,
            self._disease_label_binarizer,
            self._symptom_label_binarizer,
            max_len=MAX_LEN,
        )
        self._val_dataset = MultitaskDataset(
            test_idx,
            self._raw_training_data,
            self._tokenizer,
            self._disease_label_binarizer,
            self._symptom_label_binarizer,
            max_len=MAX_LEN,
        )
        # 1. 准备 DataLoader
        train_loader: DataLoader = DataLoader(
            self._train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader: DataLoader = DataLoader(
            self._val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        assert self._bert_model is not None and isinstance(
            self._bert_model, DistilBertForMultitaskLearning
        ), "BERT 模型未加载，无法进行训练"
        # 2. 定义优化器 (使用原生 PyTorch AdamW)
        # 优化器配置优化：疾病分类头使用更高学习率以加快收敛
        optimizer = AdamW(
            [
                {"params": self._bert_model.distil_bert.parameters(), "lr": 2e-5},
                {
                    "params": self._bert_model.classifier_disease.parameters(),
                    "lr": DISEASE_CLASSIFIER_LR,
                },
                {
                    "params": self._bert_model.classifier_symptom.parameters(),
                    "lr": SYMPTOM_CLASSIFIER_LR,
                },
                {
                    "params": self._bert_model.classifier_first_aid.parameters(),
                    "lr": FIRST_AID_CLASSIFIER_LR,
                },
            ],
            weight_decay=0.01,
        )
        self.get_runtime_device_info()
        model = cast(DistilBertForMultitaskLearning, self._bert_model)
        nn.Module.to(model, self._device)
        best_f1: float = 0.0

        for epoch in range(EPOCHS):
            # --- 训练阶段 ---
            self._bert_model.train()
            total_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

            for batch in train_pbar:
                optimizer.zero_grad()

                # 将所有输入移至设备
                input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).to(
                    self._device
                )
                attention_mask = torch.tensor(
                    batch["attention_mask"], dtype=torch.long
                ).to(self._device)
                labels_dis = torch.tensor(
                    batch["labels_disease"], dtype=torch.float
                ).to(self._device)
                labels_sym = torch.tensor(
                    batch["labels_symptom"], dtype=torch.float
                ).to(self._device)
                labels_emg = torch.tensor(
                    batch["labels_first_aid"], dtype=torch.float
                ).to(self._device)

                # 前向传播 (模型内部已计算联合 Loss)
                outputs: MultitaskSequenceClassifierOutput = self._bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels_disease=labels_dis,
                    labels_symptom=labels_sym,
                    labels_first_aid=labels_emg,
                )

                loss = outputs.loss
                assert loss is not None, "模型前向传播未返回损失，无法进行反向传播"
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = total_loss / len(train_loader)

            # --- 验证阶段 ---
            self._bert_model.eval()
            all_dis_true, all_dis_pred = [], []
            all_sym_true, all_sym_pred = [], []
            all_emg_true, all_emg_pred = [], []
            total_val_loss = 0
            val_losses = {"disease": 0.0, "symptom": 0.0, "first_aid": 0.0}

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Eval]")
            with torch.no_grad():
                for batch in val_pbar:
                    input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).to(
                        TORCH_DEVICE
                    )
                    attention_mask = torch.tensor(
                        batch["attention_mask"], dtype=torch.long
                    ).to(TORCH_DEVICE)
                    labels_dis = torch.tensor(
                        batch["labels_disease"], dtype=torch.float
                    ).to(TORCH_DEVICE)
                    labels_sym = torch.tensor(
                        batch["labels_symptom"], dtype=torch.float
                    ).to(TORCH_DEVICE)
                    labels_emg = torch.tensor(
                        batch["labels_first_aid"], dtype=torch.float
                    ).to(TORCH_DEVICE)

                    outputs: MultitaskSequenceClassifierOutput = self._bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels_disease=labels_dis,
                        labels_symptom=labels_sym,
                        labels_first_aid=labels_emg,
                    )
                    assert (
                        outputs.logits is not None
                    ), "模型前向传播未返回 logits，无法计算指标"
                    logits_dis, logits_sym, logits_emg = outputs.logits

                    # 计算单个任务的loss用于分析
                    loss_dis_batch: torch.Tensor = (
                        self._bert_model.bce_with_logits_loss(logits_dis, labels_dis)
                    )
                    loss_sym_batch: torch.Tensor = (
                        self._bert_model.bce_with_logits_loss(logits_sym, labels_sym)
                    )
                    loss_emg_batch: torch.Tensor = (
                        self._bert_model.bce_with_logits_loss(logits_emg, labels_emg)
                    )

                    val_losses["disease"] += loss_dis_batch.item() * len(labels_dis)
                    val_losses["symptom"] += loss_sym_batch.item() * len(labels_sym)
                    val_losses["first_aid"] += loss_emg_batch.item() * len(labels_emg)
                    assert (
                        outputs.loss is not None
                    ), "模型前向传播未返回总损失，无法计算平均验证损失"
                    total_val_loss += outputs.loss.item()

                    # 转换预测值
                    all_dis_pred.append((torch.sigmoid(logits_dis) > 0.5).cpu().numpy())
                    all_sym_pred.append((torch.sigmoid(logits_sym) > 0.5).cpu().numpy())
                    all_emg_pred.append((torch.sigmoid(logits_emg) > 0.5).cpu().numpy())

                    # 收集真实值
                    all_dis_true.append(
                        torch.tensor(batch["labels_disease"], dtype=torch.float).numpy()
                    )
                    all_sym_true.append(
                        torch.tensor(batch["labels_symptom"], dtype=torch.float).numpy()
                    )
                    all_emg_true.append(
                        torch.tensor(
                            batch["labels_first_aid"], dtype=torch.float
                        ).numpy()
                    )

            # 计算指标
            f1_dis = f1_score(
                np.vstack(all_dis_true), np.vstack(all_dis_pred), average="micro"
            )
            f1_sym = f1_score(
                np.vstack(all_sym_true), np.vstack(all_sym_pred), average="micro"
            )
            acc_emg = accuracy_score(np.vstack(all_emg_true), np.vstack(all_emg_pred))

            # 额外的疾病标签评估指标
            try:
                disease_precision = precision_score(
                    np.vstack(all_dis_true),
                    np.vstack(all_dis_pred),
                    average="micro",
                    zero_division=0,
                )
                disease_recall = recall_score(
                    np.vstack(all_dis_true),
                    np.vstack(all_dis_pred),
                    average="micro",
                    zero_division=0,
                )
            except:
                disease_precision = 0.0
                disease_recall = 0.0

            msg = f"\nEpoch {epoch + 1} Summary:\n"
            msg += f"Train Loss: {avg_train_loss:.4f} | Val Loss: {total_val_loss/len(val_loader):.4f}\n"
            msg += f"Task Losses - Disease: {val_losses['disease']/len(test_idx):.4f}, "
            msg += f"Symptom: {val_losses['symptom']/len(test_idx):.4f}, "
            msg += f"FirstAid: {val_losses['first_aid']/len(test_idx):.4f}\n"
            msg += f"Disease - F1: {f1_dis:.4f} | Precision: {disease_precision:.4f} | Recall: {disease_recall:.4f}\n"
            msg += f"Symptom F1: {f1_sym:.4f} | Emergency Acc: {acc_emg:.4f}"
            logger.debug(f"{msg}")
            # 保存逻辑 - 优先考虑疾病预测准确率
            # 组合指标：疾病F1权重最高
            current_disease_metric = (
                f1_dis * 0.6 + disease_precision * 0.2 + disease_recall * 0.2
            )
            if current_disease_metric > best_f1:
                best_f1 = float(current_disease_metric)

                logger.info("模型表现提升，计算每个 label 正样本的中位数...")
                medians_dict = self.compute_label_medians()

                assert self._disease_label_binarizer is not None
                assert self._symptom_label_binarizer is not None

                self._bert_model.save_pretrained(SAVE_PATH)
                self._tokenizer.save_pretrained(SAVE_PATH)
                # 保存 label_encoders
                label_encoders = {
                    "disease": self._disease_label_binarizer,
                    "symptom": self._symptom_label_binarizer,
                    "medians": medians_dict,
                }
                with open(SAVE_PATH / "label_encoders.pkl", "wb") as f:
                    pickle.dump(label_encoders, f)
                logger.debug(
                    f"★ 模型表现提升 (Disease Metric: {current_disease_metric:.4f})，已保存至 {SAVE_PATH}"
                )

    @classmethod
    def preload(
        cls, model_path: str | Path = SAVE_PATH, device: torch.device = TORCH_DEVICE
    ):
        """模型预测前预加载，加载模型、分词器和标签编码器，并将模型移动到指定设备。"""
        cls._device = device
        cls._tokenizer = cls._get_tokenizer(model_path)
        cls._bert_model = cast(
            DistilBertForMultitaskLearning,
            DistilBertForMultitaskLearning.from_pretrained(
                model_path, local_files_only=True, ignore_mismatched_sizes=True
            ),
        )
        if isinstance(cls._device, torch.device) and cls._device.type == "cuda":
            torch.nn.Module.cuda(cls._bert_model)
        else:
            torch.nn.Module.cpu(cls._bert_model)
        encoders: Any = None
        with open(f"{model_path}/label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
            # 验证encoders的类型
        if isinstance(encoders, list):
            logger.debug("警告：encoders是列表类型，尝试转换为字典")
            # 如果保存时是列表格式，需要处理
            # 这里假设列表中第一个元素是disease，第二个是symptom
            encoders = {"disease": encoders[0], "symptom": encoders[1]}
        cls._disease_label_binarizer = encoders["disease"]
        cls._symptom_label_binarizer = encoders["symptom"]
        cls._medians = dict(encoders).get("medians", {})
        return (
            cls._device,
            cls._tokenizer,
            cls._bert_model,
            cls._disease_label_binarizer,
            cls._symptom_label_binarizer,
            cls._medians,
        )

    @staticmethod
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        """计算 F1 和准确率指标"""

        # logits 是一个 tuple: (disease_logits, symptom_logits, first_aid_logits)
        if not isinstance(eval_pred.predictions, tuple) or not isinstance(
            eval_pred.label_ids, tuple
        ):
            raise ValueError(
                f"预期 predictions 和 label_ids 都是包含三个元素的 tuple，但实际类型分别为 {type(eval_pred.predictions)} 和 {type(eval_pred.label_ids)}"
            )
        logits_tuple: tuple = eval_pred.predictions
        labels_tuple: tuple = eval_pred.label_ids
        disease_logits, symptom_logits, first_aid_logits = logits_tuple
        disease_labels, symptom_labels, first_aid_labels = labels_tuple

        # 预测
        disease_predicts = (
            torch.sigmoid(torch.tensor(disease_logits)).numpy() >= 0.5
        ).astype(int)
        symptom_predicts = (
            torch.sigmoid(torch.tensor(symptom_logits)).numpy() >= 0.5
        ).astype(int)
        first_aid_predicts = (
            torch.sigmoid(torch.tensor(first_aid_logits)).numpy() >= 0.5
        ).astype(int)

        return {
            "f1_disease": float(
                f1_score(disease_labels, disease_predicts, average="micro")
            ),
            "f1_symptom": float(
                f1_score(symptom_labels, symptom_predicts, average="micro")
            ),
            "acc_first_aid": float(
                accuracy_score(first_aid_labels, first_aid_predicts)
            ),
        }

    # ====================== 计算每个 label 正样本的中位数 ======================

    def compute_label_medians(self) -> dict:
        """在训练集的**正样本**上运行模型，收集每个 label 的 sigmoid 概率，
        并计算其中位数（仅针对 ground-truth 为正的样本）。
        返回结构：
        {
            "disease": {"疾病名1": 中位数, "疾病名2": 中位数, ...},
            "symptom": {"症状名1": 中位数, ...},
            "first_aid": 急救中位数 (float)
        }
        """
        assert self._bert_model is not None, "BERT 模型未加载，无法计算中位数"
        self._bert_model.eval()
        disease_probs_pos: list[list[float]] = [
            [] for _ in range(self._bert_model.num_diseases)
        ]
        symptom_probs_pos: list[list[float]] = [
            [] for _ in range(self._bert_model.num_symptoms)
        ]
        first_aid_probs_pos: list[float] = []
        assert self._train_dataset is not None, "训练数据集未加载，无法计算中位数"
        loader: DataLoader = DataLoader(
            self._train_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        with torch.no_grad():
            for batch in tqdm(loader, desc="计算每个 label 正样本中位数"):
                # 只传入模型需要的输入（不传入 labels，避免计算 loss）
                input_ids = torch.tensor(batch["input_ids"]).to(self._device).long()
                attention_mask = (
                    torch.tensor(batch["attention_mask"]).to(self._device).long()
                )

                outputs: MultitaskSequenceClassifierOutput = self._bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                assert outputs.logits is not None
                logits_d, logits_s, logits_f = outputs.logits

                # sigmoid 概率
                probs_d = torch.sigmoid(logits_d).cpu().numpy()  # (bs, num_diseases)
                probs_s = torch.sigmoid(logits_s).cpu().numpy()  # (bs, num_symptoms)
                probs_f = torch.sigmoid(logits_f).cpu().numpy()  # (bs, 1)

                # ground-truth
                labels_d = (
                    torch.tensor(batch["labels_disease"]).cpu().numpy()
                )  # (bs, num_diseases)
                labels_s = (
                    torch.tensor(batch["labels_symptom"]).cpu().numpy()
                )  # (bs, num_symptoms)
                labels_f = (
                    torch.tensor(batch["labels_first_aid"]).cpu().numpy()
                )  # (bs, 1)

                for i in range(len(probs_d)):  # 遍历 batch 内每个样本
                    # 疾病
                    for j in range(self._bert_model.num_diseases):
                        if labels_d[i, j] == 1.0:
                            disease_probs_pos[j].append(probs_d[i, j])
                    # 症状
                    for j in range(self._bert_model.num_symptoms):
                        if labels_s[i, j] == 1.0:
                            symptom_probs_pos[j].append(probs_s[i, j])
                    # 急救（二分类）
                    if labels_f[i, 0] == 1.0:
                        first_aid_probs_pos.append(probs_f[i, 0])
        assert (
            self._disease_label_binarizer is not None
        ), "疾病标签二值化器未加载，无法计算中位数"
        assert (
            self._symptom_label_binarizer is not None
        ), "症状标签二值化器未加载，无法计算中位数"
        # 计算中位数
        disease_medians: dict[str, float] = {}
        for j, prob_list in enumerate(disease_probs_pos):
            label_name = self._disease_label_binarizer.classes_[j]
            if prob_list:
                disease_medians[label_name] = float(np.median(prob_list))
            else:
                disease_medians[label_name] = 0.5

        symptom_medians: dict[str, float] = {}
        for j, prob_list in enumerate(symptom_probs_pos):
            label_name = self._symptom_label_binarizer.classes_[j]
            if prob_list:
                symptom_medians[label_name] = float(np.median(prob_list))
            else:
                symptom_medians[label_name] = 0.5

        first_aid_median = (
            float(np.median(first_aid_probs_pos)) if first_aid_probs_pos else 0.5
        )

        return {
            "disease": disease_medians,
            "symptom": symptom_medians,
            "first_aid": first_aid_median,
        }

    def predict_with_preload(
        self,
        text: str,
        *,
        threshold: float = 1 / (1 + math.pow(math.e, -(0))),
    ) -> dict[str, Any]:
        """使用预加载的模型和相关组件进行预测，返回带置信度的疾病和症状列表。"""
        assert self._tokenizer is not None and isinstance(
            self._tokenizer, DistilBertTokenizer
        ), "分词器未加载，无法进行预测"
        assert self._bert_model is not None and isinstance(
            self._bert_model, DistilBertForMultitaskLearning
        ), "BERT 模型未加载，无法进行预测"
        assert self._device is not None and isinstance(
            self._device, torch.device
        ), "推理设备未加载，无法进行预测"
        assert self._medians is not None, "中位数信息未加载，无法进行置信度计算"
        assert self._disease_label_binarizer is not None and isinstance(
            self._disease_label_binarizer, MultiLabelBinarizer
        ), "疾病标签编码器未加载，无法进行预测"
        assert self._symptom_label_binarizer is not None and isinstance(
            self._symptom_label_binarizer, MultiLabelBinarizer
        ), "症状标签编码器未加载，无法进行预测"
        # 预处理
        inputs: BatchEncoding = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        inputs_kwargs: dict[str, torch.Tensor] = {
            key: value.to(self._device) for key, value in inputs.items()
        }

        # 推理
        self._bert_model.eval()
        with torch.no_grad():
            outputs: MultitaskSequenceClassifierOutput = self._bert_model(
                **inputs_kwargs
            )
        assert outputs.logits is not None, "模型输出缺少 logits"
        disease_logits, symptoms_logits, first_aid_logits = outputs.logits
        disease_logits = disease_logits.detach().cpu()
        symptoms_logits = symptoms_logits.detach().cpu()
        first_aid_logits = first_aid_logits.detach().cpu()
        # logger.debug(f"disease_logits: {disease_logits}\n")
        # logger.debug(f"symptoms_logits: {symptoms_logits}\n")
        # logger.debug(f"first_aid_logits: {first_aid_logits}\n")

        disease_probs = (
            torch.sigmoid(disease_logits).cpu().numpy()[0]
        )  # (num_diseases,)
        symptom_probs = (
            torch.sigmoid(symptoms_logits).cpu().numpy()[0]
        )  # (num_symptoms,)
        first_aid_prob = torch.sigmoid(first_aid_logits).cpu().numpy()[0][0]  # scalar

        # 4. 决定正类（保持原有 > threshold 逻辑）
        disease_mask = disease_probs >= threshold
        symptom_mask = symptom_probs >= threshold

        # 5. 构造带置信信息的返回结果
        diseases_result_temp = []
        for idx in np.where(disease_mask)[0]:
            label_name = self._disease_label_binarizer.classes_[idx]
            prob = float(disease_probs[idx])
            median = self._medians.get("disease", {}).get(label_name, 0.5)
            # 置信度判断（可自行调整规则）
            if median <= 0.5:
                norm_conf = 1.0
            else:
                norm_conf = 1.0 if prob >= median else (prob - 0.5) / (median - 0.5)
            norm_conf = float(np.clip(norm_conf, 0.0, 1.0))
            diseases_result_temp.append(
                {
                    "name": label_name,
                    # "probability": round(prob, 4),
                    # "median_positive": round(median, 4),
                    "confidence": round(norm_conf, 2),  # 与中位数比较得出的置信等级
                    # "above_median": prob >= median
                }
            )
        # === others 特殊处理逻辑 ===
        others_name: str = "others"
        predicted_diseases = diseases_result_temp[:]  # 浅拷贝
        has_others = any(d.get("name") == others_name for d in predicted_diseases)
        num_pred = len(predicted_diseases)

        if has_others:
            if num_pred > 1:
                # 有 others + 其他疾病 → 丢弃 others
                predicted_diseases = [
                    d for d in predicted_diseases if d.get("name") != others_name
                ]
            # else: 只有 others → 保留
        else:
            if num_pred == 0:
                # 没有任何疾病标签 → 强制添加 others，置信度=1.0
                predicted_diseases = [{"name": others_name, "confidence": 1.0}]

        diseases_result = predicted_diseases

        symptoms_result = []
        for idx in np.where(symptom_mask)[0]:
            label_name = self._symptom_label_binarizer.classes_[idx]
            prob = float(symptom_probs[idx])
            median = self._medians.get("symptom", {}).get(label_name, 0.5)
            if median <= 0.5:
                norm_conf = 1.0
            else:
                norm_conf = 1.0 if prob >= median else (prob - 0.5) / (median - 0.5)
            norm_conf = float(np.clip(norm_conf, 0.0, 1.0))
            symptoms_result.append(
                {
                    "name": label_name,
                    # "probability": round(prob, 4),
                    # "median_positive": round(median, 4),
                    "confidence": norm_conf,
                    # "above_median": prob >= median
                }
            )

        # 急救（二分类）
        need_first_aid = int(first_aid_prob >= threshold)
        fa_median = self._medians.get("first_aid", 0.5)
        if fa_median <= 0.5:
            fa_norm_conf = 1.0 if first_aid_prob >= 0.5 else 0.0
        else:
            fa_norm_conf = (
                1.0
                if first_aid_prob >= fa_median
                else (first_aid_prob - 0.5) / (fa_median - 0.5)
            )
        fa_norm_conf = float(np.clip(fa_norm_conf, 0.0, 1.0))

        result = {
            "diseases": diseases_result,  # 每个正类疾病都带 probability + median + confidence
            "symptoms": symptoms_result,
            "need_first_aid": need_first_aid,
        }

        return result
