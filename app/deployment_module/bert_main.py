from __future__ import annotations

import json
import math
from sklearn.model_selection import train_test_split
from torch import FloatTensor
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

from static_module import (
    BERT_FOLDER,
    TRAINED_BERT_SAVE_PATH,
    TORCH_DEVICE,
    USE_CUDA,
    BERT_TRAINING_DATASET_FOLDER,
)
from utility_module import logger

# ====================== 1. 配置与数据准备 ======================
LOCAL_MODEL_PATH: Path = Path.cwd() / BERT_FOLDER
""" 本地BERT模型路径，包含模型权重和配置文件。请确保该路径下有 transformers 兼容的模型文件（如 pytorch_model.bin 和 config.json） """
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


def print_runtime_device_info() -> None:
    """打印当前运行设备的信息，帮助确认是否使用了 GPU 以及相关的 CUDA 版本和 GPU 型号。"""
    msg = f"当前设备: {TORCH_DEVICE}\n"
    if USE_CUDA:
        torch_version = getattr(torch, "version", SimpleNamespace(cuda=None))
        msg += f"CUDA 版本: {torch_version.cuda}\n"
        msg += f"GPU: {torch.cuda.get_device_name(0)}\n"
    else:
        msg += "未检测到可用 CUDA，当前将使用 CPU 训练。"

    logger.info(msg)


# 训练数据
with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
    raw_data: list[dict[str, Any]] = json.load(f)

# ====================== 2. 数据预处理与标签编码 ======================
# 提取所有标签
with open(DISEASE_LABELS_PATH, "r", encoding="utf-8") as f:
    all_diseases: list[str] = json.load(f)
with open(SYMPTOM_LABELS_PATH, "r", encoding="utf-8") as f:
    all_symptoms: list[str] = json.load(f)

# 初始化标签编码器
disease_label_binarizer: MultiLabelBinarizer = MultiLabelBinarizer().fit([all_diseases])
""" 疾病标签二值化器，适用于多标签分类，将疾病列表转换为多热编码 """
symptom_label_binarizer: MultiLabelBinarizer = MultiLabelBinarizer().fit([all_symptoms])
""" 症状标签二值化器，适用于多标签分类，将症状列表转换为多热编码 """


# 处理数据集
class MultitaskDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        indices,
        data: list[dict[str, Any]],
        tokenizer: DistilBertTokenizer,
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
            disease_label_binarizer.transform([item["disease"]])[0]
        )
        labels_symptom: torch.FloatTensor = torch.FloatTensor(
            symptom_label_binarizer.transform([item["symptoms"]])[0]
        )
        labels_first_aid: torch.FloatTensor = torch.FloatTensor(
            [item["need_first_aid"]]
        )  # 二分类用float方便BCEWithLogitsLoss

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels_disease": labels_disease,
            "labels_symptom": labels_symptom,
            "labels_first_aid": labels_first_aid,
        }


# 加载分词器
def load_tokenizer_compat(
    model_path: str | Path, local_files_only: bool = False
) -> DistilBertTokenizer:
    """兼容不同 transformers 版本的 Mistral regex 修复参数。"""
    try:
        return DistilBertTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            fix_mistral_regex=True,
        )
    except TypeError as exc:
        if "fix_mistral_regex" in str(exc) and "multiple values" in str(exc):
            return DistilBertTokenizer.from_pretrained(
                model_path,
                local_files_only=local_files_only,
            )
        raise


tokenizer = load_tokenizer_compat(LOCAL_MODEL_PATH, local_files_only=True)


# ====================== 3. 自定义多任务模型 ======================
class DistilBertForMultitaskLearning(DistilBertPreTrainedModel):
    """自定义多任务学习模型，基于 DistilBERT，包含三个分类头：疾病预测、症状预测和是否需要急救的二分类。"""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.distil_bert: DistilBertModel = DistilBertModel(config)

        # transformers 5.x 的 from_pretrained 会调用 all_tied_weights_keys.keys()，
        # 必须显式设为空 dict（本模型无权重共享）。
        self.all_tied_weights_keys: dict[str, str] = {}

        # 分类头的维度
        self.num_diseases: int = len(disease_label_binarizer.classes_)
        self.num_symptoms: int = len(symptom_label_binarizer.classes_)
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


# 初始化模型
config: PretrainedConfig = AutoConfig.from_pretrained(
    LOCAL_MODEL_PATH, local_files_only=True
)
model = DistilBertForMultitaskLearning(config)


# ====================== 4. 自定义 Trainer (可选，为了计算指标) ======================
def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """计算 F1 和准确率指标"""

    # logits 是一个 tuple: (disease_logits, symptom_logits, first_aid_logits)
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
        "acc_first_aid": float(accuracy_score(first_aid_labels, first_aid_predicts)),
    }

# ====================== 新增：计算每个 label 正样本的中位数 ======================
def compute_label_medians(
    model: DistilBertForMultitaskLearning,
    dataset: MultitaskDataset,
    device: torch.device,
) -> dict:
    """在训练集的**正样本**上运行模型，收集每个 label 的 sigmoid 概率，
    并计算其中位数（仅针对 ground-truth 为正的样本）。
    返回结构：
    {
        "disease": {"疾病名1": 中位数, "疾病名2": 中位数, ...},
        "symptom": {"症状名1": 中位数, ...},
        "first_aid": 急救中位数 (float)
    }
    """
    model.eval()
    disease_probs_pos: list[list[float]] = [[] for _ in range(model.num_diseases)]
    symptom_probs_pos: list[list[float]] = [[] for _ in range(model.num_symptoms)]
    first_aid_probs_pos: list[float] = []

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(loader, desc="计算每个 label 正样本中位数"):
            # 只传入模型需要的输入（不传入 labels，避免计算 loss）
            input_ids = batch["input_ids"].to(device).long()
            attention_mask = batch["attention_mask"].to(device).long()

            outputs: MultitaskSequenceClassifierOutput = model(
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
            labels_d = batch["labels_disease"].cpu().numpy()  # (bs, num_diseases)
            labels_s = batch["labels_symptom"].cpu().numpy()  # (bs, num_symptoms)
            labels_f = batch["labels_first_aid"].cpu().numpy()  # (bs, 1)

            for i in range(len(probs_d)):  # 遍历 batch 内每个样本
                # 疾病
                for j in range(model.num_diseases):
                    if labels_d[i, j] == 1.0:
                        disease_probs_pos[j].append(probs_d[i, j])
                # 症状
                for j in range(model.num_symptoms):
                    if labels_s[i, j] == 1.0:
                        symptom_probs_pos[j].append(probs_s[i, j])
                # 急救（二分类）
                if labels_f[i, 0] == 1.0:
                    first_aid_probs_pos.append(probs_f[i, 0])

    # 计算中位数
    disease_medians: dict[str, float] = {}
    for j, prob_list in enumerate(disease_probs_pos):
        label_name = disease_label_binarizer.classes_[j]
        if prob_list:
            disease_medians[label_name] = float(np.median(prob_list))
        else:
            disease_medians[label_name] = 0.5

    symptom_medians: dict[str, float] = {}
    for j, prob_list in enumerate(symptom_probs_pos):
        label_name = symptom_label_binarizer.classes_[j]
        if prob_list:
            symptom_medians[label_name] = float(np.median(prob_list))
        else:
            symptom_medians[label_name] = 0.5

    first_aid_median = float(np.median(first_aid_probs_pos)) if first_aid_probs_pos else 0.5

    return {
        "disease": disease_medians,
        "symptom": symptom_medians,
        "first_aid": first_aid_median,
    }

# ====================== 5. 开始训练 (手动循环) ======================


def train_bert():

    # 创建 Dataset
    train_idx, test_idx = train_test_split(
        range(len(raw_data)), test_size=0.2, random_state=42
    )
    # 简单的训练/验证划分（实际请用 train_test_split）
    logger.debug(f"训练集大小: {len(train_idx)}, 验证集大小: {len(test_idx)}")
    train_dataset = MultitaskDataset(train_idx, raw_data, tokenizer, max_len=MAX_LEN)
    val_dataset = MultitaskDataset(test_idx, raw_data, tokenizer, max_len=MAX_LEN)
    # 1. 准备 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 定义优化器 (使用原生 PyTorch AdamW)
    # 优化器配置优化：疾病分类头使用更高学习率以加快收敛
    optimizer = AdamW(
        [
            {"params": model.distil_bert.parameters(), "lr": 2e-5},
            {
                "params": model.classifier_disease.parameters(),
                "lr": DISEASE_CLASSIFIER_LR,
            },
            {
                "params": model.classifier_symptom.parameters(),
                "lr": SYMPTOM_CLASSIFIER_LR,
            },
            {
                "params": model.classifier_first_aid.parameters(),
                "lr": FIRST_AID_CLASSIFIER_LR,
            },
        ],
        weight_decay=0.01,
    )
    print_runtime_device_info()
    model.to(TORCH_DEVICE)
    best_f1: float = 0.0

    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

        for batch in train_pbar:
            optimizer.zero_grad()

            # 将所有输入移至设备
            input_ids = batch["input_ids"].to(TORCH_DEVICE).long()
            attention_mask = batch["attention_mask"].to(TORCH_DEVICE).long()
            labels_dis = batch["labels_disease"].to(TORCH_DEVICE)
            labels_sym = batch["labels_symptom"].to(TORCH_DEVICE)
            labels_emg = batch["labels_first_aid"].to(TORCH_DEVICE)

            # 前向传播 (模型内部已计算联合 Loss)
            outputs: MultitaskSequenceClassifierOutput = model(
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
        model.eval()
        all_dis_true, all_dis_pred = [], []
        all_sym_true, all_sym_pred = [], []
        all_emg_true, all_emg_pred = [], []
        total_val_loss = 0
        val_losses = {"disease": 0.0, "symptom": 0.0, "first_aid": 0.0}

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Eval]")
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(TORCH_DEVICE).long()
                attention_mask = batch["attention_mask"].to(TORCH_DEVICE).long()
                labels_dis = batch["labels_disease"].to(TORCH_DEVICE)
                labels_sym = batch["labels_symptom"].to(TORCH_DEVICE)
                labels_emg = batch["labels_first_aid"].to(TORCH_DEVICE)

                outputs: MultitaskSequenceClassifierOutput = model(
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
                loss_dis_batch: torch.Tensor = model.bce_with_logits_loss(
                    logits_dis, labels_dis
                )
                loss_sym_batch: torch.Tensor = model.bce_with_logits_loss(
                    logits_sym, labels_sym
                )
                loss_emg_batch: torch.Tensor = model.bce_with_logits_loss(
                    logits_emg, labels_emg
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
                all_dis_true.append(batch["labels_disease"].numpy())
                all_sym_true.append(batch["labels_symptom"].numpy())
                all_emg_true.append(batch["labels_first_aid"].numpy())

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
            medians_dict = compute_label_medians(model, val_dataset, TORCH_DEVICE)

            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            # 保存 label_encoders
            label_encoders = {
                "disease": disease_label_binarizer,
                "symptom": symptom_label_binarizer,
                "medians": medians_dict,
            }
            with open(SAVE_PATH / "label_encoders.pkl", "wb") as f:
                pickle.dump(label_encoders, f)
            logger.debug(
                f"★ 模型表现提升 (Disease Metric: {current_disease_metric:.4f})，已保存至 {SAVE_PATH}"
            )


# ====================== 6. 推理函数 ======================
def predict(
    text: str,
    model_path: str | Path = SAVE_PATH,
    threshold: float = 1 / (1 + math.pow(math.e, -(0))),
    device: torch.device = TORCH_DEVICE,
) -> dict[str, Any]:
    """给定输入文本，返回预测的疾病、症状和是否需要急救的结果"""
    # 加载
    inference_device = device
    tokenizer = load_tokenizer_compat(model_path)
    inference_model = cast(
        DistilBertForMultitaskLearning,
        DistilBertForMultitaskLearning.from_pretrained(
            model_path, local_files_only=True, ignore_mismatched_sizes=True
        ),
    )
    if isinstance(inference_device, torch.device) and inference_device.type == "cuda":
        torch.nn.Module.cuda(inference_model)
    else:
        torch.nn.Module.cpu(inference_model)
    with open(f"{model_path}/label_encoders.pkl", "rb") as f:
        encoders: Any = pickle.load(f)
        # 验证encoders的类型
    if isinstance(encoders, list):
        logger.debug("警告：encoders是列表类型，尝试转换为字典")
        # 如果保存时是列表格式，需要处理
        # 这里假设列表中第一个元素是disease，第二个是symptom
        encoders = {"disease": encoders[0], "symptom": encoders[1]}
    mlb_d: MultiLabelBinarizer = encoders["disease"]
    mlb_s: MultiLabelBinarizer = encoders["symptom"]
    medians: dict = encoders.get("medians", {})
    # 预处理
    inputs: BatchEncoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    inputs_kwargs: dict[str, torch.Tensor] = {
        key: value.to(inference_device) for key, value in inputs.items()
    }

    # 推理
    inference_model.eval()
    with torch.no_grad():
        outputs: MultitaskSequenceClassifierOutput = inference_model(**inputs_kwargs)
    assert outputs.logits is not None, "模型输出缺少 logits"
    disease_logits, symptoms_logits, first_aid_logits = outputs.logits
    disease_logits = disease_logits.detach().cpu()
    symptoms_logits = symptoms_logits.detach().cpu()
    first_aid_logits = first_aid_logits.detach().cpu()
    logger.debug(f"disease_logits: {disease_logits}\n")
    logger.debug(f"symptoms_logits: {symptoms_logits}\n")
    logger.debug(f"first_aid_logits: {first_aid_logits}\n")

    disease_probs = torch.sigmoid(disease_logits).cpu().numpy()[0]  # (num_diseases,)
    symptom_probs = torch.sigmoid(symptoms_logits).cpu().numpy()[0]  # (num_symptoms,)
    first_aid_prob = torch.sigmoid(first_aid_logits).cpu().numpy()[0][0]  # scalar

    # 4. 决定正类（保持原有 > threshold 逻辑）
    disease_mask = disease_probs >= threshold
    symptom_mask = symptom_probs >= threshold

    # 5. 构造带置信信息的返回结果
    diseases_result = []
    for idx in np.where(disease_mask)[0]:
        label_name = mlb_d.classes_[idx]
        prob = float(disease_probs[idx])
        median = medians.get("disease", {}).get(label_name, 0.5)
        # 置信度判断（可自行调整规则）
        if median <= 0.5:
            norm_conf = 1.0
        else:
            norm_conf = 1.0 if prob >= median else (prob - 0.5) / (median - 0.5)
        norm_conf = float(np.clip(norm_conf, 0.0, 1.0))
        diseases_result.append({
            "label": label_name,
            "probability": round(prob, 4),
            "median_positive": round(median, 4),
            "confidence": norm_conf,  # 与中位数比较得出的置信等级
            "above_median": prob >= median
        })

    symptoms_result = []
    for idx in np.where(symptom_mask)[0]:
        label_name = mlb_s.classes_[idx]
        prob = float(symptom_probs[idx])
        median = medians.get("symptom", {}).get(label_name, 0.5)
        if median <= 0.5:
            norm_conf = 1.0
        else:
            norm_conf = 1.0 if prob >= median else (prob - 0.5) / (median - 0.5)
        norm_conf = float(np.clip(norm_conf, 0.0, 1.0))
        symptoms_result.append({
            "label": label_name,
            "probability": round(prob, 4),
            "median_positive": round(median, 4),
            "confidence": norm_conf,
            "above_median": prob >= median
        })

    # 急救（二分类）
    need_first_aid = int(first_aid_prob >= threshold)
    fa_median = medians.get("first_aid", 0.5)
    if fa_median <= 0.5:
        fa_norm_conf = 1.0 if first_aid_prob >= 0.5 else 0.0
    else:
        fa_norm_conf = 1.0 if first_aid_prob >= fa_median else (first_aid_prob - 0.5) / (fa_median - 0.5)
    fa_norm_conf = float(np.clip(fa_norm_conf, 0.0, 1.0))

    result = {
        "diseases": diseases_result,  # 每个正类疾病都带 probability + median + confidence
        "symptoms": symptoms_result,
        "need_first_aid": {
            "value": need_first_aid,
            "probability": round(first_aid_prob, 4),
            "median_positive": round(fa_median, 4),
            "confidence": fa_norm_conf,
            "above_median": first_aid_prob >= fa_median if need_first_aid else None,
        },
    }

    return result

# 测试推理
if __name__ == "__main__":
    #train_bert()
    test_text = "I feel dizzy and nauseous"
    result = predict(test_text)
    msg = f"输入文本：{test_text}\n推理测试结果:"
    msg += json.dumps(result, indent=2)
    logger.info(msg)
