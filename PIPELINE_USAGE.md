# Drug Recommendation Pipeline - 使用文档

## 系统架构

```
用户输入 (症状文本)
       │
       ├─────────────────────────────────────────────┐
       │                                             │
       ▼                                             ▼
┌─────────────────┐                      ┌─────────────────────┐
│  BERT 多任务模型  │  (use_bert=true)     │   语义向量编码        │
│  DistilBERT      │                      │   PubMedBERT (768d) │
│  - 疾病分类(44)   │                      └──────────┬──────────┘
│  - 症状分类(132)  │                                 │
│  - 急救判断(1)    │                                 ▼
└────────┬─────────┘                      ┌─────────────────────┐
         │                                │  语义召回 Top-300     │
         │ diseases + symptoms            │  (余弦相似度)         │
         │ (带置信度)                      └──────────┬──────────┘
         ▼                                           │
┌─────────────────────┐                              │
│  标签召回 Top-300     │                              │
│  (置信度加权匹配)     │                              │
└────────┬────────────┘                              │
         │                                           │
         └───────────────┬───────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │  双路融合             │
              │  score = α·semantic  │
              │       + β·label      │
              │  → Top-300           │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  CrossEncoder 重排    │
              │  ms-marco-MiniLM     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  业务因子调整          │
              │  评分·评论数·价格     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  最终排名 Top-K       │
              │  final = 0.35·recall │
              │  + 0.50·cross_enc    │
              │  + 0.15·business     │
              └─────────────────────┘
```

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

确保以下文件存在：
- `match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv` — 药物数据表
- `drug_comprehensive_embeddings.npy` — 预计算的药物向量 (5595×768)
- `app/deployment_module/trained_bert/` — 训练好的 BERT 模型权重

### 2. 启动服务

```bash
cd app
python -m uvicorn fastapi_module.app_main:app --host 0.0.0.0 --port 8000
```

### 3. 健康检查

```bash
GET http://localhost:8000/v1/recommendation/health
```

## API 接口

### POST /v1/recommendation/drugs

药物推荐主接口。支持两种模式：

---

### 模式 A：BERT 自动预测（推荐）

只需提供症状文本，BERT 自动识别疾病和症状标签：

```json
{
    "symptom_text": "I have a severe headache, blurry vision, and I feel dizzy all the time",
    "use_bert_prediction": true,
    "top_k": 10
}
```

BERT 模型会自动：
- 从文本中识别疾病标签（如 hypertension, diabetes）及其置信度
- 识别症状标签（如 headache, blurred_vision, dizziness）及其置信度
- 将预测结果传入标签召回通道

---

### 模式 B：手动指定标签

手动提供疾病/症状标签及置信度：

```json
{
    "symptom_text": "I have a severe headache and blurry vision",
    "diseases": [
        {"name": "hypertension", "confidence": 0.83},
        {"name": "diabetes", "confidence": 0.44}
    ],
    "symptoms": [
        {"name": "headache", "confidence": 1.0},
        {"name": "blurred_and_distorted_vision", "confidence": 0.88}
    ],
    "top_k": 10
}
```

---

### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `symptom_text` | string | (必填) | 用户自然语言症状描述 |
| `diseases` | list | [] | 疾病标签列表，格式 `{"name": "...", "confidence": 0.0~1.0}` |
| `symptoms` | list | [] | 症状标签列表，格式同上 |
| `use_bert_prediction` | bool | false | 是否启用 BERT 自动预测（覆盖 diseases/symptoms） |
| `top_k` | int | 10 | 最终返回的药物数量 |
| `recall_top_k_each` | int | 300 | 每个召回通道保留的候选数 |
| `fused_top_k` | int | 300 | 融合后保留的候选数 |
| `recall_weight_semantic` | float | 0.5 | 语义召回权重 |
| `recall_weight_label` | float | 0.5 | 标签召回权重 |

### 响应格式

```json
{
    "count": 10,
    "results": [
        {
            "drug_name": "lisinopril",
            "final_score": 0.87,
            "recall_fused_score": 0.72,
            "cross_encoder_score": 0.95,
            "business_score": 0.68,
            "semantic_score": 0.81,
            "label_score": 0.63,
            "disease_conf_overlap": 0.83,
            "symptom_conf_overlap": 1.0,
            "avg_rating": 4.2,
            "total_reviews": 156,
            "extra": {}
        }
    ]
}
```

### 评分字段说明

| 字段 | 说明 |
|------|------|
| `final_score` | 最终综合评分 = 0.35×recall + 0.50×cross_encoder + 0.15×business |
| `recall_fused_score` | 双路召回融合分 |
| `cross_encoder_score` | CrossEncoder 重排分（语义精排） |
| `business_score` | 业务因子分（评分 0.55 + 评论数 0.30 + 价格 0.15） |
| `semantic_score` | 语义召回通道原始得分 |
| `label_score` | 标签召回通道原始得分 |
| `disease_conf_overlap` | 疾病标签置信度重叠 |
| `symptom_conf_overlap` | 症状标签置信度重叠 |

## 核心文件结构

```
app/
├── fastapi_module/
│   ├── router.py          # API 路由入口
│   ├── schemas.py         # 请求/响应数据模型
│   └── service.py         # 服务层：BERT + 召回管道编排
├── deployment_module/
│   ├── bert_main.py       # BERT 多任务模型（训练 + 推理）
│   └── trained_bert/      # 模型权重、tokenizer、label_encoders
├── embedded_module/
│   ├── dual_recall_pipeline.py    # 双通道召回主管道
│   ├── drug_embedding_engine.py   # PubMedBERT 向量编码
│   ├── cross_encoder_reranker.py  # CrossEncoder 精排
│   ├── recommendation_pipeline.py # 工具函数
│   └── drug_knn_retriever.py      # Faiss KNN（备用）
└── static_module/
    └── parameters.py      # 全局参数配置
```

## 模型清单

| 模型 | 用途 | 大小 | 设备 |
|------|------|------|------|
| DistilBERT (multi-task) | 疾病/症状标签预测 | ~66M | CPU / CUDA |
| PubMedBERT | 药物语义向量编码 | ~110M | CPU / CUDA |
| ms-marco-MiniLM-L-6-v2 | CrossEncoder 重排 | ~22M | CPU / CUDA |

所有模型均支持 CPU 推理，无需 GPU。首次请求时懒加载。

---

## 一键脚本

### 单条推荐（run_recommend.py）

在项目根目录直接运行，无需 cd 到 app：

```bash
# 使用默认 query
python run_recommend.py

# 自定义 query
python run_recommend.py --query "I have a persistent cough with chest pain and shortness of breath"

# 启用 BERT 自动预测标签
python run_recommend.py --query "..." --use-bert

# 手动指定标签
python run_recommend.py --query "headache and nausea" --diseases "migraine:0.9,hypertension:0.5" --symptoms "headache:1.0,nausea:0.8"
```

### 批量评估（run_evaluate.py）

```bash
# 跑全部评估集
python run_evaluate.py

# 只跑前 N 条
python run_evaluate.py --limit 20

# 指定输出
python run_evaluate.py --limit 50 --output eval_results.json
```

---

## 环境变量（可选）

在 `.env` 中配置：

```env
DRUG_TABLE_PATH=match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv
DRUG_EMBEDDING_PATH=drug_comprehensive_embeddings.npy
TRAINED_BERT_SAVE_PATH=deployment_module/trained_bert
CROSS_ENCODER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
MEDBERT_MODEL_NAME=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
AUTO_BUILD_EMBEDDINGS=false
```

## 管道权重配置

### 召回阶段
- 语义召回 vs 标签召回：各 0.5（可通过 API 参数调整）
- 标签召回内部：疾病权重 0.55，症状权重 0.45

### 重排阶段（固定）
- 召回融合分：0.35
- CrossEncoder 分：0.50
- 业务因子分：0.15

### 业务因子（固定）
- 平均评分：0.55
- 评论数 log：0.30
- 价格（越低越好）：0.15

---

## 生产链路（推荐完整流程）

### 端到端调用链

```
HTTP POST /v1/recommendation/drugs
       │
       ▼
┌─── FastAPI Router ───┐
│  解析 RecommendRequest │
└──────────┬───────────┘
           ▼
┌─── DrugRecommendationService.recommend() ───┐
│                                               │
│  [可选] BERT 自动预测                          │
│  if use_bert_prediction:                      │
│    symptom_text → DistilBERT                  │
│    → diseases: [{name, confidence}]           │
│    → symptoms: [{name, confidence}]           │
│                                               │
└──────────────────┬────────────────────────────┘
                   ▼
┌─── DualRecallDrugRecommender.recommend() ───┐
│                                               │
│  ① semantic_recall()                          │
│     encode_query(symptom_text) → 768d vec     │
│     cosine_similarity(query, 5595 drugs)      │
│     → Top-300 candidates                      │
│                                               │
│  ② label_recall()                             │
│     parse diseases/symptoms → confidence map  │
│     match drug.disease_key_list ∩ query        │
│     match drug.symptom_list ∩ query            │
│     weighted_score = 0.55·disease + 0.45·sym  │
│     → Top-300 candidates                      │
│                                               │
│  ③ fuse_recalls()                             │
│     left_join两路 → fill 0                    │
│     fused = α·semantic + β·label              │
│     → Top-300 candidates                      │
│                                               │
│  ④ rerank()                                   │
│     build_rerank_query()                      │
│     CrossEncoder.score_pairs(query, 300 docs) │
│     business_score(rating, reviews, price)    │
│     final = 0.35·recall + 0.50·CE + 0.15·biz │
│     → Top-K results                           │
│                                               │
└──────────────────┬────────────────────────────┘
                   ▼
┌─── 响应组装 ───┐
│  DataFrame → CandidateScore[]  │
│  + PipelineTrace (if enabled)  │
└────────────────────────────────┘
```

### 关键数据资产

| 资产 | 路径 | 说明 |
|------|------|------|
| 药物数据表 | `match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv` | 5595 条药物，含疾病/症状映射 |
| 药物向量 | `drug_comprehensive_embeddings.npy` | 预计算的 (5595, 768) float32 |
| BERT 权重 | `app/deployment_module/trained_bert/` | DistilBERT 多任务模型 |
| 标签编码器 | `app/deployment_module/trained_bert/label_encoders.pkl` | 44 疾病 + 132 症状 + medians |

### 单条药物的语义文本构成

每条药物通过 `build_semantic_text()` 将以下字段拼接为一段文本后编码为向量：

```
Drug name: {drug_name}.
Generic ingredient: {generic_name}.
Drug classes: {drug_classes}.
Related diseases: {disease_key_list 逗号分隔}.
Target symptoms: {symptom_list_normalized 前20个}.
Medical condition context: {medical_condition_description}.
Disease description: {disease_description}.
Known side effects: {side_effects}.
```

PubMedBERT max_length=256 tokens，超出截断。

---

## 评估流程

### 评估数据集

| 文件 | 条数 | 说明 |
|------|------|------|
| `match_data_preprocessing/data/eval_dataset_candidates.json` | 300 | 初版，规则生成 |
| `match_data_preprocessing/data/eval_dataset_verified.json` | ~200 | 经 DDI 安全审查的最终版 |

每条记录格式：
```json
{
    "query_id": "eval_0001",
    "symptom_text": "...",
    "diseases": [{"name": "...", "confidence": 0.8}],
    "symptoms": [{"name": "...", "confidence": 0.9}],
    "relevant_drugs": ["drug_a", "drug_b"],
    "relevance_scores": {"drug_a": 3, "drug_b": 2}
}
```

- `relevance_scores`: 3 = 高度相关，2 = 一般相关
- `ddi_flags`（verified 版本独有）：药物相互作用安全性标记

### 运行评估

#### 方式一：命令行一键跑评估集（支持限制条数）

```bash
# 从项目根目录运行
python run_evaluate.py

# 只跑前 10 条（快速验证）
python run_evaluate.py --limit 10

# 指定评估集 + K 值 + 输出文件
python run_evaluate.py --eval-dataset match_data_preprocessing/data/eval_dataset_verified.json --k-values 5 10 20 --limit 50 --output eval_results.json
```

#### 方式二：从 app 目录使用原始模块

```bash
cd app
python -m evaluation.run_evaluation --eval-dataset ../match_data_preprocessing/data/eval_dataset_verified.json --k-values 5 10 20 --output ../eval_results.json
```

#### 方式三：Python 代码调用

```python
from run_recommend import init_service

service = init_service()
result_df, trace = service.recommend(
    symptom_text="I have a headache and nausea",
    diseases=[{"name": "migraine", "confidence": 0.9}],
    symptoms=[{"name": "headache", "confidence": 1.0}],
    top_k=10,
    enable_trace=True,
)
print(result_df[["drug_name", "final_score"]].to_string())
```

### 评估指标

| 指标 | 含义 |
|------|------|
| `Precision@K` | Top-K 中正确药物的比例 |
| `Recall@K` | ground truth 中被找回的比例 |
| `Hit@K` | Top-K 中至少命中一个的概率 |
| `MRR` | 第一个正确药物的排名倒数 |
| `NDCG@K` | 考虑相关性等级的排序质量 |

### 评估数据集生成逻辑

生成脚本：`app/evaluation/build_test_set.py`

1. 读取 `enhanced_drug_table_v1_structured.csv`，过滤无效药物
2. 从 `eval_dataset_llm_v2.json` 抽取 query（symptoms/diseases）
3. 按 disease 数量分桶：150 single + 100 double + 50 triple+
4. 为每个 query 从对应 disease 药池中选 relevant drugs（症状需有交集，按评分排序）
5. 每 disease 最多取 3 个药，总数 3~9 个

---

## 数据埋点（Pipeline Trace）

### 启用方式

请求中设置 `"enable_trace": true`：

```json
{
    "symptom_text": "I have a severe headache and nausea",
    "use_bert_prediction": true,
    "enable_trace": true,
    "top_k": 10
}
```

### 返回的 trace 字段

响应中会多一个 `trace` 对象：

```json
{
    "count": 10,
    "results": [...],
    "trace": {
        "time_semantic_recall_ms": 23.45,
        "time_label_recall_ms": 8.12,
        "time_fuse_ms": 1.03,
        "time_rerank_ms": 1520.67,
        "time_total_ms": 1553.27,
        "count_semantic_candidates": 300,
        "count_label_candidates": 187,
        "count_fused_candidates": 300,
        "count_final": 10,
        "score_semantic_mean": 0.4523,
        "score_label_mean": 0.3812,
        "score_cross_encoder_mean": 0.6234,
        "score_business_mean": 0.5102,
        "score_final_mean": 0.5567,
        "score_final_max": 0.8213,
        "score_final_min": 0.3456,
        "query_length": 42,
        "num_disease_labels": 2,
        "num_symptom_labels": 3
    }
}
```

### 埋点指标说明

#### 耗时指标
| 字段 | 说明 |
|------|------|
| `time_semantic_recall_ms` | 语义召回耗时（含 query 编码 + 余弦计算） |
| `time_label_recall_ms` | 标签召回耗时（置信度匹配） |
| `time_fuse_ms` | 双路融合耗时 |
| `time_rerank_ms` | CrossEncoder 重排耗时（主要瓶颈） |
| `time_total_ms` | 管道总耗时 |

#### 候选数量
| 字段 | 说明 |
|------|------|
| `count_semantic_candidates` | 语义召回返回的候选数 |
| `count_label_candidates` | 标签召回返回的候选数（可能 < top_k） |
| `count_fused_candidates` | 融合后保留的候选数 |
| `count_final` | 最终返回的药物数 |

#### 分数分布
| 字段 | 说明 |
|------|------|
| `score_*_mean` | Top-K 结果中各通道分数的均值 |
| `score_final_max/min` | 最终分数的极值（观察区分度） |

#### 输入元数据
| 字段 | 说明 |
|------|------|
| `query_length` | 症状文本字符长度 |
| `num_disease_labels` | 输入的疾病标签数量 |
| `num_symptom_labels` | 输入的症状标签数量 |

### 埋点用途

1. **性能分析**：定位慢查询瓶颈（通常是 CrossEncoder）
2. **召回诊断**：label_candidates 数为 0 说明标签匹配失败
3. **分数健康度**：final_max - final_min 过小说明区分度不够
4. **A/B 实验**：对比不同参数配置下的分数分布变化
5. **批量评估**：结合评估数据集收集 300 条 trace，统计整体耗时和分数分布
