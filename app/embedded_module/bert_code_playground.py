import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DistilBertForMaskedLM,
)
from datasets import Dataset
import evaluate
import numpy as np

# ====================== 1. 配置与数据加载 ======================
MODEL_NAME = "medicalai/ClinicalBERT"
MAX_LEN = 256  # 症状文本通常不长，256足够
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 3e-5

# 假设你有一个 CSV 文件，列名为 'text' 和 'labels'（labels 用逗号分隔）
# 示例数据（你需要替换为真实数据）
data = [
    {"text": "发热38.5℃，咳嗽，咳黄痰，胸痛", "labels": "肺炎,上呼吸道感染"},
    {"text": "右上腹绞痛，恶心呕吐，厌油腻", "labels": "胆囊炎,胆结石"},
    {"text": "鼻塞、流涕、咽痛、全身乏力", "labels": "感冒,流感"},
    # ... 更多数据
]
df = pd.DataFrame(data)

# 处理标签：多标签编码
mlb = MultiLabelBinarizer()
df["labels"] = df["labels"].apply(lambda x: x.split(","))
labels = mlb.fit_transform(df["labels"])
num_labels = len(mlb.classes_)  # 病症类别总数

# 划分训练/验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 转换为 Hugging Face Dataset 格式
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ====================== 2. 分词与预处理 ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN
    )


# 分词并添加标签
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# 格式转换（适配 PyTorch）
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# 注意：这里需要手动把 labels 转成 tensor（因为 Dataset 里的 labels 还是 list）
def add_labels_tensor(example):
    # 找到当前 example 在原始 df 中的索引，获取对应的 labels
    idx = example["__index_level_0__"]
    example["labels"] = torch.tensor(labels[idx], dtype=torch.float)
    return example


tokenized_train = tokenized_train.map(add_labels_tensor)
tokenized_val = tokenized_val.map(add_labels_tensor)

# ====================== 3. 模型定义 ======================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification",  # 关键：指定多标签任务
)

# ====================== 4. 训练配置 ======================
# 定义评估指标（多标签常用 F1-score）
metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (predictions >= 0.5).astype(int)  # 阈值0.5，可调整
    return metric.compute(predictions=predictions, references=labels, average="micro")


training_args = TrainingArguments(
    output_dir="./clinicalbert-diagnosis",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # 早停防止过拟合
)

# ====================== 5. 开始训练 ======================
trainer.train()


# ====================== 6. 推理函数 ======================
def predict_diseases(symptom_text, threshold=0.5):
    # 分词
    inputs = tokenizer(
        symptom_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).numpy()[0]

    # 解码标签：只输出概率 > threshold 的病症
    predicted_labels = [
        mlb.classes_[i] for i, prob in enumerate(probabilities) if prob > threshold
    ]
    return predicted_labels, probabilities


# ====================== 7. 测试推理 ======================
sample_symptom = "患者发热39℃，伴剧烈咳嗽、脓痰，呼吸急促"
predicted_diseases, probs = predict_diseases(sample_symptom)
print(f"输入症状: {sample_symptom}")
print(f"预测病症: {predicted_diseases}")
