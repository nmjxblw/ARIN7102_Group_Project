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
from sklearn.metrics import f1_score, accuracy_score
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


# 模拟你的训练数据（实际使用时请替换为加载你的 JSON 文件列表）
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
        self.loss_fn_bce: nn.BCEWithLogitsLoss = (
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

        # 计算损失
        loss = None  # 默认无损失
        if labels_disease is not None:
            loss_disease: FloatTensor = self.loss_fn_bce(logits_disease, labels_disease)
            loss_symptom: FloatTensor = self.loss_fn_bce(logits_symptom, labels_symptom)
            loss_first_aid: FloatTensor = self.loss_fn_bce(
                logits_first_aid, labels_first_aid
            )
            loss = loss_disease + loss_symptom + loss_first_aid  # 联合损失

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
    logits_tuple, labels_tuple = eval_pred
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
    # 推荐对 BERT 层使用较小的学习率，对分类头使用较大的学习率
    optimizer = AdamW(
        [
            {"params": model.distil_bert.parameters(), "lr": 2e-5},
            {"params": model.classifier_disease.parameters(), "lr": 1e-3},
            {"params": model.classifier_symptom.parameters(), "lr": 1e-3},
            {"params": model.classifier_first_aid.parameters(), "lr": 1e-3},
        ],
        weight_decay=0.01,
    )
    print_runtime_device_info()
    model.to(TORCH_DEVICE)
    best_f1 = 0.0

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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_disease=labels_dis,
                labels_symptom=labels_sym,
                labels_first_aid=labels_emg,
            )

            loss = outputs.loss
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

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Eval]")
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(TORCH_DEVICE).long()
                attention_mask = batch["attention_mask"].to(TORCH_DEVICE).long()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_dis, logits_sym, logits_emg = outputs.logits

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

        msg = f"\nEpoch {epoch + 1} Summary:\n"
        msg += f"Train Loss: {avg_train_loss:.4f}\n"
        msg += f"Disease F1: {f1_dis:.4f} | Symptom F1: {f1_sym:.4f} | Emergency Acc: {acc_emg:.4f}"
        logger.debug(f"{msg}")
        # 保存逻辑
        current_combined_f1 = f1_dis + f1_sym
        if current_combined_f1 > best_f1:
            best_f1 = current_combined_f1
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            # 保存 label_encoders
            label_encoders = {
                "disease": disease_label_binarizer,
                "symptom": symptom_label_binarizer,
            }
            with open(SAVE_PATH / "label_encoders.pkl", "wb") as f:
                pickle.dump(label_encoders, f)
            logger.debug(f"★ 模型表现提升，已保存至 {SAVE_PATH}")


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
        DistilBertForMultitaskLearning.from_pretrained(model_path),
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
    # 解码
    diseases = mlb_d.inverse_transform(
        (torch.sigmoid(disease_logits).numpy() >= threshold).astype(int)
    )[0]
    symptoms = mlb_s.inverse_transform(
        (torch.sigmoid(symptoms_logits).numpy() >= threshold).astype(int)
    )[0]
    need_first_aid = (torch.sigmoid(first_aid_logits).numpy() >= threshold).astype(int)[
        0
    ][0]

    return {
        "disease": list(diseases),
        "symptoms": list(symptoms),
        "need_first_aid": int(need_first_aid),
    }


# 测试推理
if __name__ == "__main__":
    # train_bert()
    test_text = "I feel dizzy and nauseous"
    result = predict(test_text)
    msg = f"输入文本：{test_text}\n推理测试结果:"
    msg += json.dumps(result, indent=2)
    logger.info(msg)
