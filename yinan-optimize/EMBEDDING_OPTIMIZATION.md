# Embedding 召回模块优化记录

## 当前语义召回架构

```
symptom_text + diseases + symptoms
       │ _build_combined_query() 动态拼接
       ▼
┌─────────────────────────────┐
│  LLM Query Expansion        │  DeepSeek API
│  expand_query_with_llm()    │  追加 Medical terms: ...
└──────────┬──────────────────┘
           │ enriched_query
           ▼
┌─────────────────────────────┐
│  DrugEmbeddingEngine        │  模型: S-PubMedBert-MS-MARCO
│  (bi-encoder, mean pooling) │  max_length: 512
│  encode_query(text) → (1,768)│
└──────────┬──────────────────┘
           │ query_vec
           ▼
┌─────────────────────────────────────────────────────┐
│  drug_comprehensive_embeddings.npy                   │
│  shape: (N, 768) — 单向量 (disease_description)     │
│                                                      │
│  文本: 基本信息 + disease_description               │
│        (医学/临床视角: "comedones, papules...")       │
└──────────┬──────────────────────────────────────────┘
           │ cosine similarity
           ▼
 semantic_recall top-K ──┐
                          ├─► 双路融合 ──► CrossEncoder 精排 ──► top-10
 label_recall (标签)   ──┘
```

### 药物侧 embedding 文本格式

```
Drug: {drug_name}. Ingredient: {generic_name}. Class: {drug_classes}.
Diseases: {disease_keys}. Symptoms: {symptoms[:15]}.
Description: {disease_description}
```

- `disease_description` 为医学/临床视角描述（如 "Acne vulgaris is the formation of comedones, papules..."）
- 约 51% 的药物有此字段，无此字段时退化为纯基本信息

### 查询侧（`_build_combined_query` 动态拼接）

| 输入情况 | 拼接结果 |
|---------|----------|
| 三个字段都有 | `"{symptom_text} Diseases: d1, d2. Symptoms: s1, s2."` |
| 只有口语 | `"{symptom_text}"` |
| 只有标签 | `"Diseases: d1, d2. Symptoms: s1, s2."` |
| 都为空 | `"[MASK]"` |

---

## 已实施优化

### 优化1：模型替换 + 文本精简

- embedding 模型从 `PubMedBERT`（MLM 预训练，[CLS] 不适合检索）→ `pritamdeka/S-PubMedBert-MS-MARCO`（MS MARCO 微调的 bi-encoder）
- Pooling: CLS → mean pooling
- max_length: 256 → 512
- 去掉冗长的 side_effects，只保留核心检索字段

### 优化2：双向量 → 单向量（仅保留 View 1）

**原方案**：每药两个向量 (N, 2, 768)，View 0 = medical_condition_description（口语），View 1 = disease_description（医学），取 max score。

**问题**：实测 View 0 拖累了召回效果。原因：
1. View 0 有 **22.2%** 的文本被 512 token 截断（description 在末尾被截掉）
2. max 操作不是提升目标药物分数，而是提升了无关药物的 View 0 分数，导致目标药物排名下降

**View 对比实测数据（36 疾病）：**

| 查询类型 | 评分模式 | 平均排名 | R@100 | R@200 | R@500 |
|---------|---------|---------|-------|-------|-------|
| 纯口语 | View0 only | 1401 | 2.8% | 11.1% | 33.3% |
| 纯口语 | **View1 only** | **716** | **16.7%** | **25.0%** | **44.4%** |
| 纯口语 | max(V0,V1) | 798 | 11.1% | 25.0% | 47.2% |
| 口+全标签 | View0 only | 930 | 22.2% | 33.3% | 44.4% |
| 口+全标签 | **View1 only** | **196** | **47.2%** | **66.7%** | **91.7%** |
| 口+全标签 | max(V0,V1) | 207 | 47.2% | 58.3% | 86.1% |

**结论**：改为仅使用 View 1 后：
- 口+标签 R@200: 58.3% → **66.7%**（+8.4%）
- 口+标签 R@500: 86.1% → **91.7%**（+5.6%）
- 存储和计算量减半：(N, 2, 768) → (N, 768)

### 优化3：语义召回查询动态拼接

- semantic_recall 输入从纯 `symptom_text` 改为 `symptom_text + diseases + symptoms` 动态拼接
- `_build_combined_query()` 统一用于 semantic_recall 和 CrossEncoder 精排
- 效果：R@200 从 25%（纯口语）→ **66.7%**（口+标签 + View 1 only）

### 当前效果汇总（双路召回，36 疾病，召回阶段 top-200）

| 召回组合 | 平均排名 | R@50 | R@100 | R@200 | 召回率 |
|---------|---------|------|-------|-------|--------|
| 纯语义(oral) | 84 | 5.6% | 13.9% | 22.2% | 22% |
| 纯语义(enriched) | 67 | 30.6% | 50.0% | 66.7% | 67% |
| LLM扩展语义 | 60 | 44.4% | 58.3% | 75.0% | 75% |
| 纯标签 | 28 | 69.4% | 86.1% | 88.9% | 89% |
| **语义+标签（生产配置）** | **53** | **61.1%** | **69.4%** | **88.9%** | **92%** |

> 生产链路 = enriched query（口语+标签拼接）的 semantic_recall + label_recall 双通道加权融合。
> 标签通道是当前召回率的主要贡献者；语义通道扩大覆盖面，提供额外 +3pp 召回。

---

## 纯口语召回低的根因分析

纯口语 R@200 仅 25%，核心原因：

| 原因 | 说明 | 示例 |
|------|------|------|
| **词汇鸿沟** | 患者口语与医学术语无词汇重叠 | "painful bumps" ↔ "papules, nodules" |
| **实体缺失** | 口语中无药物名/成分名/疾病键 | 口语无 "doxycycline"、"acne" |
| **长度不对称** | 口语 ~20 tokens vs 药物文本 ~150 tokens，mean pooling 信号稀释 | 短查询向量与长文档向量空间偏移 |

---

## 待选优化方案

### 优化4：Description 前置（低难度，低-中收益）

将 description 从文本末尾移到开头，避免被 512 token 截断：

```
当前: "Drug: X. Ingredient: Y. ... Description: {长文本}"
改为: "Description: {长文本} Drug: X. Ingredient: Y. ..."
```

- 当前 View 1 仅 0.1% 被截断，收益有限
- 若未来 description 变长或换用更短 max_length 时有价值

### 优化5：LLM 医学术语扩展（已实施）

semantic_recall 前用 LLM（DeepSeek）将口语翻译为医学术语，追加到查询末尾：

```
原始: "I have big painful bumps on my face and some blackheads"
扩展: "... Medical terms: acne vulgaris, papules, comedones, cystic acne"
```

- 模块: `app/embedded_module/query_expander.py`
- 开关: `pipeline_config.env` → `ENABLE_LLM_QUERY_EXPANSION=true`
- API 凭证: 根目录 `.env` 的 `OPENAI_*` 配置
- 降级: LLM 调用失败时返回原始文本，不影响链路

**LLM 扩展实测数据（36 疾病）：**

| 查询方式 | 平均排名 | 中位排名 | R@100 | R@200 | R@500 |
|---------|---------|---------|-------|-------|-------|
| 纯口语 (baseline) | 716 | 596 | 16.7% | 25.0% | 44.4% |
| 口+全标签 (baseline) | 196 | 144 | 47.2% | 66.7% | 91.7% |
| 纯标签 (baseline) | 144 | 76 | 58.3% | 69.4% | 97.2% |
| A: 口语+LLM | 426 | 185 | 41.7% | 50.0% | 75.0% |
| **C: 口+标签+LLM** | **160** | **68** | **55.6%** | **72.2%** | **91.7%** |
| D: 纯LLM术语 | 323 | 180 | 41.7% | 52.8% | 86.1% |

**结论**：
- **C（口+标签+LLM）效果最好**，R@200: 66.7% → **72.2%**（+5.5pp），中位排名: 144 → **68**
- A（口语+LLM）相比纯口语大幅提升（R@200: 25%→50%），但不如加标签
- D（纯LLM）R@500 高但精度不够，说明 LLM 术语扩大覆盖面但不够聚焦
- 生产链路采用 C 方案
- 代价：每次查询增加 ~3s LLM 调用延迟

### 优化6：微调 bi-encoder（高难度，高收益）

用 eval_dataset_llm_v2.json（63260 条）构造训练对，微调 S-PubMedBert-MS-MARCO：

```python
positive_pairs = [("I have painful bumps", "Drug: Doxycycline... Description: Acne vulgaris...")]
# MultipleNegativesRankingLoss 训练
```

- 最根本的解决方案，但需 GPU + 调参

### 优化7：换用指令型 embedding 模型（中等难度，中-高收益）

换 `intfloat/e5-base-v2` 或 `BAAI/bge-base-en-v1.5`，支持 query/passage 前缀区分：

```python
query   = "query: I have painful bumps on my face"
passage = "passage: Drug: Doxycycline. Description: Acne vulgaris..."
```

- 专为非对称检索（短 query ↔ 长 passage）优化
- 需重建 embedding

### 优先级排序

| 优先级 | 方案 | 预期 R@200 提升 | 实现成本 |
|--------|------|----------------|----------|
| ~~1~~ | ~~优化5: LLM 术语扩展~~ | ~~已实施: R@200 +8.3pp~~ | ~~已完成~~ |
| 2 | 优化6: 微调 bi-encoder | +20~30% | 需构造训练数据 + GPU |
| 3 | 优化7: 换指令型模型 | +10~20% | 重建 embedding |
| 4 | 优化4: Description 前置 | +1~3% | View 1 截断率仅 0.1%，收益极低 |
