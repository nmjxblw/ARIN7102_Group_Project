# Embedding 召回模块交接文档

## 1. 项目概述

**项目名称**：ARIN7102 Group Project — 药物推荐系统

**核心功能**：用户输入症状描述（口语文本），系统推荐最相关的药物。

**技术栈**：Python, FastAPI, PyTorch, Transformers, NumPy, Sentence-Transformers

**数据规模**：
- 药物库：约 2,600 种药物（来自 `enhanced_drug_table_v1_structured.csv`）
- 评估数据集：`eval_dataset_llm_v2.json`，63,260 条记录，覆盖 42 种疾病，LLM 生成的口语化症状查询

---

## 2. 系统架构

```
用户口语输入 (symptom_text)
       │
       ▼
┌──────────────────────────────┐
│  BERT 分类器 (ClinicalBERT)  │  → 提取 disease_labels, symptom_labels
│  app/deployment_module/      │     及其 confidence 权重
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  _build_combined_query()     │  → enriched_query = "口语 Diseases: d1,d2. Symptoms: s1,s2."
│  (可选) LLM 术语扩展          │  → 追加 "Medical terms: term1, term2, ..."
└──────────┬───────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
 semantic       label
 _recall()     _recall()
 (向量检索)     (精确标签匹配)
     │            │
     └─────┬──────┘
           ▼
┌──────────────────────────────┐
│  fuse_recalls()              │  语义 0.5 + 标签 0.5 加权融合
│  → top-200 候选              │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  CrossEncoder 精排            │  ms-marco-MiniLM-L-6-v2
│  final = 0.35×recall         │
│        + 0.50×cross_encoder  │
│        + 0.15×business       │
└──────────┬───────────────────┘
           ▼
       top-10 药物推荐
```

---

## 3. 当前调整模块

### 模块位置

| 文件 | 职责 |
|------|------|
| `app/embedded_module/dual_recall_pipeline.py` | 双路召回管道核心（semantic + label → fuse → rerank） |
| `app/embedded_module/drug_embedding_engine.py` | bi-encoder 模型加载、向量编码 |
| `app/embedded_module/drug_knn_retriever.py` | embedding 加载、cosine 相似度计算 |
| `app/embedded_module/cross_encoder_reranker.py` | CrossEncoder 精排 |
| `app/embedded_module/query_expander.py` | LLM 医学术语扩展 |
| `pipeline_config.py` / `pipeline_config.env` | 集中参数配置 |
| `app/evaluation/metrics.py` | 评估指标计算（R@K, Precision, MRR, NDCG） |
| `_test_recall_compare.py` | 多通道召回对比测试脚本 |

### 关键模型

| 模型 | 用途 | 维度 |
|------|------|------|
| `pritamdeka/S-PubMedBert-MS-MARCO` | bi-encoder 语义编码 | 768 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder 精排 | - |
| `deepseek-chat` (DeepSeek API) | LLM 术语扩展 | - |

### Embedding 存储

- 文件：`drug_comprehensive_embeddings.npy`，shape `(N, 768)`
- 仅保留 View 1（disease_description 视角），View 0 已废弃
- 药物文本格式：`"Drug: X. Ingredient: Y. Class: Z. Diseases: ... Symptoms: ... Description: ..."`

---

## 4. 已完成的优化

| 序号 | 优化内容 | 效果 | 状态 |
|------|---------|------|------|
| 1 | 模型替换 PubMedBERT → S-PubMedBert-MS-MARCO + mean pooling | 检索能力质的飞跃 | ✅ |
| 2 | 双向量 → 单向量（仅 View 1） | R@200 +8.4pp，存储减半 | ✅ |
| 3 | 语义召回查询动态拼接 | R@200 25% → 66.7% | ✅ |
| 4 | LLM 医学术语扩展（DeepSeek） | R@200 66.7% → 75.0%（+8.3pp） | ✅ |

---

## 5. 当前性能基线（召回阶段，36 疾病，top-200）

| 召回组合 | 平均排名 | R@50 | R@100 | R@200 | 召回率 |
|---------|---------|------|-------|-------|--------|
| 纯语义(oral) | 84 | 5.6% | 13.9% | 22.2% | 22% |
| 纯语义(enriched) | 67 | 30.6% | 50.0% | 66.7% | 67% |
| LLM扩展语义 | 60 | 44.4% | 58.3% | 75.0% | 75% |
| 纯标签 | 28 | 69.4% | 86.1% | 88.9% | 89% |
| **语义+标签（生产配置）** | **53** | **61.1%** | **69.4%** | **88.9%** | **92%** |

**核心发现**：
- 标签通道是召回率的主要贡献者（89%），语义通道提供额外 +3pp
- 语义通道单独 R@200 仅 66.7%（enriched）/ 75%（+LLM），有较大提升空间
- 纯口语语义 R@200 仅 22.2%，存在严重的词汇鸿沟问题

---

## 6. 调整方向与目标

### 核心目标

**提升语义召回通道的独立召回率**，减少对标签通道的强依赖，使语义+标签组合的 R@200 达到 95%+。

### 优先方向

#### 方向 1：微调 bi-encoder（高优先级，高收益）

用 `eval_dataset_llm_v2.json` 构造正负样本对，微调 `S-PubMedBert-MS-MARCO`：

```python
# 正样本: (口语查询, 目标药物文本)
positive_pairs = [("I have painful bumps", "Drug: Doxycycline... Description: Acne vulgaris...")]
# 训练方式: MultipleNegativesRankingLoss / contrastive learning
```

- 预期提升：语义 R@200 +20~30pp
- 需要：GPU 训练环境、构造训练数据管道
- 微调后需重建 `drug_comprehensive_embeddings.npy`

#### 方向 2：换用指令型 embedding 模型（中优先级）

替换为支持 query/passage 前缀的模型（如 `intfloat/e5-base-v2`、`BAAI/bge-base-en-v1.5`）：

```python
query   = "query: I have painful bumps on my face"
passage = "passage: Drug: Doxycycline. Description: Acne vulgaris..."
```

- 专为非对称检索（短 query ↔ 长 passage）优化
- 预期提升：R@200 +10~20pp
- 需重建 embedding

#### 方向 3：精排阶段优化（中优先级）

- 当前 CrossEncoder 使用通用模型 `ms-marco-MiniLM-L-6-v2`，可替换为医学领域微调版本
- 调整 `final_weight` 三项权重比例
- 探索 LLM reranking（用大模型对 top-K 候选重排序）

#### 方向 4：Description 前置（低优先级）

将药物文本中的 description 从末尾移到开头，避免长文本被 512 token 截断。当前 View 1 截断率仅 0.1%，收益极低。

---

## 7. 配置管理

所有参数集中在 `pipeline_config.env`，无需改代码即可调参：

```env
# 召回权重
RECALL_WEIGHT_SEMANTIC=0.5
RECALL_WEIGHT_LABEL=0.5

# 精排权重
FINAL_WEIGHT_RECALL=0.35
FINAL_WEIGHT_CROSS_ENCODER=0.50
FINAL_WEIGHT_BUSINESS=0.15

# 模型
MEDBERT_MODEL_NAME=pritamdeka/S-PubMedBert-MS-MARCO
EMBEDDING_POOLING=mean
EMBEDDING_MAX_LENGTH=512

# LLM 扩展
ENABLE_LLM_QUERY_EXPANSION=true
```

LLM API 凭证在根目录 `.env`：
```env
OPENAI_API_KEY=<your-deepseek-key>
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

---

## 8. 运行指南

### 召回对比测试

```bash
cd ARIN7102_Group_Project
python _test_recall_compare.py
```

输出 5 种召回组合的 R@K、平均排名等指标对比。

### 启动服务

```bash
python -m app
```

### 重建 embedding（模型变更后）

```python
from app.embedded_module.drug_embedding_engine import DrugEmbeddingEngine
engine = DrugEmbeddingEngine()
engine.build_embeddings(drug_table)  # 输出 drug_comprehensive_embeddings.npy
```

---

## 9. 已知问题与注意事项

1. **`parse_weighted_labels` 兼容性**：已修复，同时支持 `"name"` 和 `"label"` 键。修改此函数时需确保两种格式都能正确解析。

2. **LLM 扩展延迟**：每次查询增加约 3 秒 API 调用延迟，超时 10 秒自动 fallback 到原始文本。

3. **评估数据集局限性**：`eval_dataset_llm_v2.json` 的 ground truth 药物列表并非完整（每个疾病通常只有 1-3 种目标药物），R@K 指标可能低估实际效果。

4. **CrossEncoder 瓶颈**：CPU 上每 100 条候选约需 6 秒精排，`FUSED_TOP_K` 设置过大会显著增加延迟。

---

## 10. 参考文档

- 优化详细记录：`yinan-optimize/EMBEDDING_OPTIMIZATION.md`
- 管道设计文档：`docs/pipeline_design.md`
- 评估数据集格式：`docs/evaluation_dataset_format.md`
- LLM 评估配置：`docs/llm_eval_dataset_config.md`
