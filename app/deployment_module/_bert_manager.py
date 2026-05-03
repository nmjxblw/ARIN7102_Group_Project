from __future__ import annotations

import json
import math
import pickle
import threading
from pathlib import Path
from typing import Any, Optional, Tuple, cast
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace

from transformers import (
    AutoConfig,
    BatchEncoding,
    DistilBertModel,
    DistilBertPreTrainedModel,
    DistilBertTokenizerFast,
    EvalPrediction,
    PretrainedConfig,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils.generic import ModelOutput

from static_module import (
    DEPLOYMENT_FOLDER,
    TRAINED_BERT_SAVE_PATH,
    TORCH_DEVICE,
    USE_CUDA,
    BERT_TRAINING_DATASET_FOLDER,
)
from utility_module import logger

# ====================== 1. 配置与参数 ======================
LOCAL_MODEL_PATH: Path = Path.cwd() / DEPLOYMENT_FOLDER
SAVE_PATH: Path = Path.cwd() / TRAINED_BERT_SAVE_PATH
MAX_LEN: int = 256
BATCH_SIZE: int = 64  # 已增大 Batch Size 以发挥 GPU 性能
EPOCHS: int = 10
LEARNING_RATE: float = 3e-5

# ===== 疾病标签优化参数 =====
DISEASE_LOSS_WEIGHT: float = 2.5
SYMPTOM_LOSS_WEIGHT: float = 1.0
FIRST_AID_LOSS_WEIGHT: float = 0.5
DISEASE_CLASSIFIER_LR: float = 5e-4
SYMPTOM_CLASSIFIER_LR: float = 1e-3
FIRST_AID_CLASSIFIER_LR: float = 1e-3
DISEASE_POSITIVE_WEIGHT: float = 1.2

# ===== 路径配置 =====
RAW_DATA_PATH: Path = Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "generated_medical_dataset.json"
DISEASE_LABELS_PATH: Path = Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "disease_labels.json"
SYMPTOM_LABELS_PATH: Path = Path.cwd() / BERT_TRAINING_DATASET_FOLDER / "symptom_labels.json"

USE_FP16: bool = USE_CUDA

if USE_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ====================== 2. 数据结构 ======================
@dataclass
class MultitaskSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class MultitaskDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """解耦后的数据集类，不再依赖外部全局变量"""
    def __init__(
        self,
        indices: list[int],
        data: list[dict[str, Any]],
        tokenizer: DistilBertTokenizerFast,
        max_len: int,
        mlb_disease: MultiLabelBinarizer,
        mlb_symptom: MultiLabelBinarizer,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.indices = indices
        self.mlb_disease = mlb_disease
        self.mlb_symptom = mlb_symptom

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, id: int) -> dict[str, torch.Tensor]:
        idx = self.indices[id]
        item = self.data[idx]
        text = item.get("sentences", "")
        if not text:
            raise ValueError(f"数据项缺少 'sentences' 字段: {item}")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels_disease = torch.FloatTensor(self.mlb_disease.transform([item["disease"]])[0])
        labels_symptom = torch.FloatTensor(self.mlb_symptom.transform([item["symptoms"]])[0])
        labels_first_aid = torch.FloatTensor([item.get("need_first_aid", 0)])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels_disease": labels_disease,
            "labels_symptom": labels_symptom,
            "labels_first_aid": labels_first_aid,
        }


class DistilBertForMultitaskLearning(DistilBertPreTrainedModel):
    """解耦后的多任务学习模型，维度通过参数传入"""
    def __init__(
        self, 
        config: PretrainedConfig, 
        num_diseases: Optional[int] = None, 
        num_symptoms: Optional[int] = None
    ) -> None:
        super().__init__(config)
        self.distil_bert = DistilBertModel(config)
        self.all_tied_weights_keys = {}

        # 允许从传入参数或 config 动态获取维度
        self.num_diseases = num_diseases if num_diseases is not None else getattr(config, "num_diseases", 1)
        self.num_symptoms = num_symptoms if num_symptoms is not None else getattr(config, "num_symptoms", 1)
        self.hidden_size = config.hidden_size

        self.classifier_disease = nn.Linear(self.hidden_size, self.num_diseases)
        self.classifier_symptom = nn.Linear(self.hidden_size, self.num_symptoms)
        self.classifier_first_aid = nn.Linear(self.hidden_size, 1)

        # 【优化点】直接将疾病权重固化到损失函数中，极大地加快 GPU 计算速度
        pos_weight = torch.tensor([DISEASE_POSITIVE_WEIGHT] * self.num_diseases)
        self.bce_disease = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_symptom = nn.BCEWithLogitsLoss()
        self.bce_first_aid = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_disease=None,
        labels_symptom=None,
        labels_first_aid=None,
        **kwargs,
    ) -> MultitaskSequenceClassifierOutput:
        
        outputs = self.distil_bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        logits_disease = self.classifier_disease(pooled_output)
        logits_symptom = self.classifier_symptom(pooled_output)
        logits_first_aid = self.classifier_first_aid(pooled_output)

        loss = None
        if labels_disease is not None:
            loss_disease = self.bce_disease(logits_disease, labels_disease)
            loss_symptom = self.bce_symptom(logits_symptom, labels_symptom)
            loss_first_aid = self.bce_first_aid(logits_first_aid, labels_first_aid)

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


# ====================== 3. 单例管理器 ======================
class BERTManager:
    """BERT 多任务模型管理器 (线程安全的单例模式)"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 避免在单例获取时重复初始化
        if getattr(self, "_is_initialized", False):
            return
            
        self.device = TORCH_DEVICE
        self._is_initialized = False
        
        self.tokenizer: Optional[DistilBertTokenizerFast] = None
        self.model: Optional[DistilBertForMultitaskLearning] = None
        self.mlb_d: Optional[MultiLabelBinarizer] = None
        self.mlb_s: Optional[MultiLabelBinarizer] = None
        self.medians: dict = {}
        
        self._is_initialized = True

    def _print_runtime_device_info(self) -> None:
        msg = f"当前设备: {self.device}\n"
        if USE_CUDA:
            torch_version = getattr(torch, "version", SimpleNamespace(cuda=None))
            msg += f"CUDA 版本: {torch_version.cuda}\n"
            msg += f"GPU: {torch.cuda.get_device_name(0)}\n"
        else:
            msg += "未检测到可用 CUDA，当前将使用 CPU 训练。"
        logger.info(msg)

    def load_tokenizer_compat(self, model_path: str | Path) -> DistilBertTokenizerFast:
        try:
            return DistilBertTokenizerFast.from_pretrained(
                model_path, local_files_only=True, fix_mistral_regex=True
            )
        except TypeError as exc:
            if "fix_mistral_regex" in str(exc):
                return DistilBertTokenizerFast.from_pretrained(
                    model_path, local_files_only=True
                )
            raise

    def preload(self, model_path: str | Path = SAVE_PATH) -> None:
        """加载已训练好的模型和配置到内存中"""
        if self.model is not None:
            return  # 已经加载过了

        logger.info("正在将模型预加载到内存中...")
        self.tokenizer = self.load_tokenizer_compat(model_path)

        with open(f"{model_path}/label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
            
        if isinstance(encoders, list):
            encoders = {"disease": encoders[0], "symptom": encoders[1]}
            
        self.mlb_d = encoders["disease"]
        self.mlb_s = encoders["symptom"]
        self.medians = encoders.get("medians", {})

        self.model = DistilBertForMultitaskLearning.from_pretrained(
            model_path, 
            local_files_only=True, 
            ignore_mismatched_sizes=True,
            num_diseases=len(self.mlb_d.classes_),
            num_symptoms=len(self.mlb_s.classes_)
        )
        
        if self.device.type == "cuda":
            self.model.cuda()
        else:
            self.model.cpu()
            
        self.model.eval()
        logger.info("预加载完成。")

    def _compute_label_medians(self, dataset: MultitaskDataset) -> dict:
        self.model.eval()
        disease_probs_pos = [[] for _ in range(self.model.num_diseases)]
        symptom_probs_pos = [[] for _ in range(self.model.num_symptoms)]
        first_aid_probs_pos = []

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(loader, desc="计算中位数"):
                input_ids = batch["input_ids"].to(self.device).long()
                attention_mask = batch["attention_mask"].to(self.device).long()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits_d, logits_s, logits_f = outputs.logits

                probs_d = torch.sigmoid(logits_d).cpu().numpy()
                probs_s = torch.sigmoid(logits_s).cpu().numpy()
                probs_f = torch.sigmoid(logits_f).cpu().numpy()

                labels_d = batch["labels_disease"].cpu().numpy()
                labels_s = batch["labels_symptom"].cpu().numpy()
                labels_f = batch["labels_first_aid"].cpu().numpy()

                for i in range(len(probs_d)):
                    for j in range(self.model.num_diseases):
                        if labels_d[i, j] == 1.0:
                            disease_probs_pos[j].append(probs_d[i, j])
                    for j in range(self.model.num_symptoms):
                        if labels_s[i, j] == 1.0:
                            symptom_probs_pos[j].append(probs_s[i, j])
                    if labels_f[i, 0] == 1.0:
                        first_aid_probs_pos.append(probs_f[i, 0])

        disease_medians = {
            self.mlb_d.classes_[j]: float(np.median(p)) if p else 0.5 
            for j, p in enumerate(disease_probs_pos)
        }
        symptom_medians = {
            self.mlb_s.classes_[j]: float(np.median(p)) if p else 0.5 
            for j, p in enumerate(symptom_probs_pos)
        }
        first_aid_median = float(np.median(first_aid_probs_pos)) if first_aid_probs_pos else 0.5

        return {
            "disease": disease_medians,
            "symptom": symptom_medians,
            "first_aid": first_aid_median,
        }

    def train_bert(self):
        """执行训练逻辑，包含了内部的数据加载，避免全局污染"""
        self._print_runtime_device_info()

        # 加载数据和标签
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        with open(DISEASE_LABELS_PATH, "r", encoding="utf-8") as f:
            all_diseases = json.load(f)
        with open(SYMPTOM_LABELS_PATH, "r", encoding="utf-8") as f:
            all_symptoms = json.load(f)

        self.tokenizer = self.load_tokenizer_compat(LOCAL_MODEL_PATH)
        self.mlb_d = MultiLabelBinarizer().fit([all_diseases])
        self.mlb_s = MultiLabelBinarizer().fit([all_symptoms])

        train_idx, test_idx = train_test_split(range(len(raw_data)), test_size=0.2, random_state=42)
        
        train_dataset = MultitaskDataset(train_idx, raw_data, self.tokenizer, MAX_LEN, self.mlb_d, self.mlb_s)
        val_dataset = MultitaskDataset(test_idx, raw_data, self.tokenizer, MAX_LEN, self.mlb_d, self.mlb_s)
        
        # 优化 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=USE_CUDA)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=USE_CUDA)

        config = AutoConfig.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        self.model = DistilBertForMultitaskLearning(
            config, 
            num_diseases=len(self.mlb_d.classes_), 
            num_symptoms=len(self.mlb_s.classes_)
        )
        self.model.to(self.device)

        optimizer = AdamW(
            [
                {"params": self.model.distil_bert.parameters(), "lr": LEARNING_RATE},
                {"params": self.model.classifier_disease.parameters(), "lr": DISEASE_CLASSIFIER_LR},
                {"params": self.model.classifier_symptom.parameters(), "lr": SYMPTOM_CLASSIFIER_LR},
                {"params": self.model.classifier_first_aid.parameters(), "lr": FIRST_AID_CLASSIFIER_LR},
            ],
            weight_decay=0.01,
        )

        # 开启自动混合精度加速
        scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)
        best_f1 = 0.0

        for epoch in range(EPOCHS):
            self.model.train()
            total_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

            for batch in train_pbar:
                optimizer.zero_grad(set_to_none=True)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels_dis = batch["labels_disease"].to(self.device, non_blocking=True)
                labels_sym = batch["labels_symptom"].to(self.device, non_blocking=True)
                labels_emg = batch["labels_first_aid"].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels_disease=labels_dis,
                        labels_symptom=labels_sym,
                        labels_first_aid=labels_emg,
                    )
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = total_loss / len(train_loader)

            # --- Eval 阶段 ---
            self.model.eval()
            all_dis_true, all_dis_pred = [], []
            total_val_loss = 0

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Eval]")
            with torch.no_grad():
                for batch in val_pbar:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels_dis = batch["labels_disease"].to(self.device)
                    labels_sym = batch["labels_symptom"].to(self.device)
                    labels_emg = batch["labels_first_aid"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels_disease=labels_dis,
                        labels_symptom=labels_sym,
                        labels_first_aid=labels_emg,
                    )
                    
                    total_val_loss += outputs.loss.item()
                    logits_dis = outputs.logits[0]
                    
                    all_dis_pred.append((torch.sigmoid(logits_dis) > 0.5).cpu().numpy())
                    all_dis_true.append(batch["labels_disease"].numpy())

            f1_dis = f1_score(np.vstack(all_dis_true), np.vstack(all_dis_pred), average="micro")
            logger.debug(f"\nEpoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {total_val_loss/len(val_loader):.4f} | Disease F1: {f1_dis:.4f}")

            if f1_dis > best_f1:
                best_f1 = f1_dis
                logger.info("模型表现提升，计算每个 label 正样本的中位数并保存...")
                
                # 将维度信息更新到 config，便于后续 load
                self.model.config.num_diseases = len(self.mlb_d.classes_)
                self.model.config.num_symptoms = len(self.mlb_s.classes_)
                
                self.medians = self._compute_label_medians(val_dataset)
                self.model.save_pretrained(SAVE_PATH, max_shard_size="50MB")
                self.tokenizer.save_pretrained(SAVE_PATH)
                
                with open(SAVE_PATH / "label_encoders.pkl", "wb") as f:
                    pickle.dump({
                        "disease": self.mlb_d,
                        "symptom": self.mlb_s,
                        "medians": self.medians,
                    }, f)
                logger.debug(f"★ 已保存至 {SAVE_PATH}")


    def predict(self, text_input: str, threshold: float = 0.5) -> dict[str, Any]:
        """单例级别的预测调用"""
        if self.model is None:
            self.preload()

        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        inputs_kwargs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs_kwargs)

        logits_d, logits_s, logits_f = outputs.logits
        disease_probs = torch.sigmoid(logits_d).cpu().numpy()[0]
        symptom_probs = torch.sigmoid(logits_s).cpu().numpy()[0]
        first_aid_prob = torch.sigmoid(logits_f).cpu().numpy()[0][0]

        # 计算置信度与结果逻辑...
        diseases_result_temp = []
        for idx in np.where(disease_probs >= threshold)[0]:
            name = self.mlb_d.classes_[idx]
            prob = float(disease_probs[idx])
            med = self.medians.get("disease", {}).get(name, 0.5)
            conf = 1.0 if med <= 0.5 or prob >= med else (prob - 0.5) / (med - 0.5)
            diseases_result_temp.append({"name": name, "confidence": round(float(np.clip(conf, 0.0, 1.0)), 2)})

        # others 处理
        has_others = any(d["name"] == "others" for d in diseases_result_temp)
        if has_others and len(diseases_result_temp) > 1:
            diseases_result = [d for d in diseases_result_temp if d["name"] != "others"]
        elif not diseases_result_temp:
            diseases_result = [{"name": "others", "confidence": 1.0}]
        else:
            diseases_result = diseases_result_temp

        symptoms_result = []
        for idx in np.where(symptom_probs >= threshold)[0]:
            name = self.mlb_s.classes_[idx]
            prob = float(symptom_probs[idx])
            med = self.medians.get("symptom", {}).get(name, 0.5)
            conf = 1.0 if med <= 0.5 or prob >= med else (prob - 0.5) / (med - 0.5)
            symptoms_result.append({"name": name, "confidence": round(float(np.clip(conf, 0.0, 1.0)), 2)})

        need_fa = int(first_aid_prob >= threshold)

        return {
            "diseases": diseases_result,
            "symptoms": symptoms_result,
            "need_first_aid": need_fa,
        }

# ====================== 4. 快速调用接口 ======================
# 对外暴露出实例化友好的接口，其他模块 import manager 即可
manager = BERTManager()

if __name__ == "__main__":
    # 取消注释以执行训练
    # manager.train_bert()
    
    test_text = "I feel dizzy and nauseous"
    result = manager.predict(test_text)
    logger.info(f"输入文本：{test_text}\n推理测试结果:\n{json.dumps(result, indent=2)}")