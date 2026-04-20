# 双路召回链路问题分析

基于 `eval_0011`（acne, nodal_skin_eruptions, scurring）单条诊断数据的分析结果。

---

## 1. 语义召回 — 完全失效

### 现象

| 指标 | 值 |
|------|-----|
| Ground truth 药物是否在 semantic top-300 | **全部不在** |
| 语义 top-5 召回的药物 | nasal relief, propafenone, afrin, echinacea, coricidin |
| semantic_score_raw 区间 | 0.922~0.923（极窄，几乎无区分度） |
| 最终 top-10 中 semantic_score | **全为 0** |

### 根因：Query-Drug 语义空间不对齐

```
用户 Query（口语化）: "I have big, painful bumps on my skin that leave scars"
             ↓ PubMedBERT encode
        query 向量 (768d)
             ↓ cosine_similarity
Drug 向量（结构化拼接）: "Drug name: doxycycline. Related diseases: acne.
                          Target symptoms: nodal skin eruptions, scurring..."
```

- **PubMedBERT 是在医学摘要上预训练的**，理解术语间关系（acne ↔ doxycycline），但不擅长口语化描述到专业术语的跨域匹配
- 所有 5595 条药物与 query 的余弦相似度集中在 0.92 附近，模型无法区分哪个更相关
- 这是一个 **asymmetric retrieval** 问题：query 是口语，document 是结构化文本

### 数据佐证

```
两通道交集仅 1 条药物（tetracycline）
Semantic top-300 与 Label top-300 几乎完全不重叠（overlap=1/300）
```

### 可能的改进方向

1. **Query 增强**：将 symptom_text + BERT 预测的标签拼接后再编码  
   `"painful bumps on skin. Diseases: acne. Symptoms: nodal skin eruptions, scurring"`
2. **对比学习微调**：在 (query, positive_drug, negative_drug) 三元组上微调 PubMedBERT
3. **换用双塔模型**：用 sentence-transformers 训练专用的 query-drug 双塔，而非共享编码器
4. **语义文本重构**：在药物侧加入更多口语化描述（如 "used for painful bumps and acne scars"）

---

## 2. 标签召回 — 召回正确但区分度低

### 现象

| 指标 | 值 |
|------|-----|
| Ground truth 药物是否在 label top-300 | **全部命中（label_score=1.0）** |
| label top-300 总数 | 300（满额） |
| label_score 分布 | 最高组=1.0（disease+2symptoms），最低组=0.55（仅disease） |
| 同分候选数 | 大量药物得分相同 |

### 根因：匹配粒度太粗 + 无负向信号

1. **只做 set overlap**：只要药物标注了 "acne" 就有分，不管它是否真正治疗 nodal skin eruptions
2. **区分度低**：所有匹配 disease + 2 symptoms 的药物得分一样（1.0），CrossEncoder 需要在大量同分候选中选择
3. **没有负向信号**：洗面奶、避孕药只要标注了 acne 也会被召回
4. **置信度未充分利用**：所有 ground truth confidence 都是 1.0，无法区分主次

### label_score 计算公式

```
label_score_raw = 0.55 × disease_overlap + 0.45 × symptom_overlap

其中:
  disease_overlap = Σ drug.disease_key_list 中每个 disease 在 query.diseases 中对应的 confidence
  symptom_overlap = Σ drug.symptom_list 中每个 symptom 在 query.symptoms 中对应的 confidence
```

### 可能的改进方向

1. **引入 IDF 加权**：对高频 disease（如 acne 出现在 300+ 药物中）降权
2. **症状覆盖率**：`symptom_overlap / len(drug.symptom_list)` 避免通用药占优
3. **负向惩罚**：药物标注的 disease 中有与 query 不匹配的，适当减分
4. **TF-IDF 或 BM25 替代精确匹配**

---

## 3. CrossEncoder 重排 — 排序偏差

### 现象

| 指标 | 值 |
|------|-----|
| CrossEncoder 平均分 | 0.89（很高） |
| 最终 top-10 命中 ground truth | **0 / 3** |
| top-10 实际推荐 | femhrt, finacea, loestrin 等非典型 acne 药 |

### 根因

- ms-marco-MiniLM 是**通用搜索排序模型**，不理解医学逻辑
- 它给 "painful bumps on skin" vs "finacea (azelaic acid for acne)" 高分是基于文本相关性，而非治疗适用性
- doxycycline 的文本描述可能不够突出其与 "painful bumps" 的关联

### 可能的改进方向

1. **在医学 QA 对上微调 CrossEncoder**
2. **保留标签召回分的排序优先级**：label_score=1.0 的候选在 rerank 中给予 boost
3. **调整权重**：降低 cross_encoder 权重（当前 0.50），提高 recall_fused_score 权重（当前 0.35）

---

## 4. 总结

```
整体链路:  语义召回(失效) + 标签召回(正确召回) → 融合(语义无贡献) → CrossEncoder(排错) → Recall@10=0

根本问题:
├── 语义通道: PubMedBERT 无法将口语 query 映射到正确药物附近
├── 标签通道: 召回了正确答案，但同分候选太多
└── 精排阶段: 通用 CrossEncoder 没有医学领域知识，排序不可靠
```

### 优先级建议

| 优先级 | 方向 | 预期收益 |
|--------|------|----------|
| P0 | 调整 rerank 权重（加大 label 权重、减小 CE 权重） | 快速止血，让已召回的正确药物排进 top-10 |
| P1 | Query 增强（拼接标签后再做语义编码） | 改善语义通道的召回率 |
| P2 | 标签召回引入 IDF/覆盖率 | 提高标签通道的区分度 |
| P3 | 微调 CrossEncoder / 换用医学领域重排模型 | 长期提升精排质量 |
