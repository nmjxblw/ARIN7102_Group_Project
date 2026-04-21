# 药物推荐系统 - 双召回链路设计说明

## 系统架构概览

```
用户输入 → 双召回（语义 + 标签）→ 融合 → Cross-Encoder精排 → 业务打分 → 推荐结果
```

---

## 链路A：语义召回（MedBERT Embedding）

### 输入
- `symptom_text: str` - 用户自然语言症状描述
  - 示例：`"I have severe headache with fever and sore throat"`

### 处理流程
1. 使用 S-PubMedBert-MS-MARCO (bi-encoder, mean pooling) 将 `symptom_text` 编码为 768 维向量
2. 与药物库的预计算向量（`drug_embeddings`）计算余弦相似度
3. 按相似度降序排序，取 Top-K（默认 200）

### 输出
- DataFrame，包含字段：
  - `semantic_score_raw`: 原始余弦相似度分数
  - `semantic_score`: Min-Max 归一化后的分数 [0, 1]
  - 药物基础信息（drug_name, generic_name 等）

### 评估方式
- **Recall@K**: 在 Top-K 结果中，正确药物被召回的比例
- **MRR (Mean Reciprocal Rank)**: 第一个正确药物的排名倒数的平均值
- **Coverage**: 有结果返回的查询占比

---

## 链路B：标签召回（置信度加权匹配）

### 输入
- `diseases: List[Dict]` - 疾病标签及置信度
  - 格式：`[{"common_cold": 0.82}, {"influenza": 0.44}]`
  - 或：`[{"name": "common_cold", "confidence": 0.82}]`
- `symptoms: List[Dict]` - 症状标签及置信度
  - 格式：`[{"fever": 0.91}, {"headache": 0.78}]`

### 处理流程
1. 解析标签和置信度，构建 `{label_name: confidence}` 映射
2. 对每个药物计算：
   - `disease_conf_overlap = Σ(药物疾病标签 ∩ 输入疾病标签的置信度)`
   - `symptom_conf_overlap = Σ(药物症状标签 ∩ 输入症状标签的置信度)`
3. 计算加权分数：
   - `label_score_raw = 0.55 × disease_conf_overlap + 0.45 × symptom_conf_overlap`
4. 过滤：保留 `disease_conf_overlap > 0 OR symptom_conf_overlap > 0` 的药物
5. 归一化并取 Top-K（默认 200）

### 输出
- DataFrame，包含字段：
  - `disease_conf_overlap`: 疾病置信度重叠分数
  - `symptom_conf_overlap`: 症状置信度重叠分数
  - `label_score_raw`: 原始加权分数
  - `label_score`: 归一化分数 [0, 1]

### 评估方式
- **Precision@K**: Top-K 中正确药物的比例
- **Hit@K**: Top-K 中是否至少包含一个正确药物
- **Disease/Symptom Match Rate**: 标签匹配的准确性

---

## 融合阶段

### 输入
- 链路A的候选集（semantic_candidates）
- 链路B的候选集（label_candidates）
- 融合权重：`recall_weight_semantic`（默认 0.5）、`recall_weight_label`（默认 0.5）

### 处理流程
1. 按药物索引对齐两路结果
2. 缺失分数补零
3. 计算融合分数：
   ```
   recall_fused_score = w_semantic × semantic_score + w_label × label_score
   ```
4. 按融合分数降序排序，取 Top-K（默认 200）

### 输出
- DataFrame，包含字段：
  - `recall_fused_score`: 融合后的召回分数
  - 保留两路的所有子分数

### 评估方式
- **ΔRecall@K**: 相比单路召回的提升
- **ΔnDCG@K**: 排序质量的提升

---

## 精排阶段（Cross-Encoder）

### 输入
- 融合后的候选集
- 重排查询文本（由 `symptom_text + disease/symptom labels` 组合）
- 候选药物的 `semantic_text`

### 处理流程
1. 构建查询文本：
   ```
   "User symptom description: {symptom_text}. Related diseases: {diseases}. Target symptoms: {symptoms}."
   ```
2. 对每个候选药物，使用 Cross-Encoder 对 `(query, drug_semantic_text)` 打分
3. 批量推理（batch_size=32）
4. 归一化分数到 [0, 1]

### 输出
- DataFrame，新增字段：
  - `cross_encoder_score`: 精排相关性分数

### 评估方式
- **nDCG@K**: 归一化折损累积增益
- **Precision@K**: 精排后的准确率
- **Reranking Gain**: 相比融合阶段的排序改进

---

## 业务打分阶段

### 输入
- 精排后的候选集
- 药物业务特征：`avg_rating`, `total_reviews`, `price`（可选）

### 处理流程
1. 计算业务分数：
   ```
   business_score = 0.55 × rating_score + 0.30 × reviews_score + 0.15 × price_score
   ```
   - `rating_score`: avg_rating 归一化
   - `reviews_score`: log1p(total_reviews) 归一化
   - `price_score`: 1 - price 归一化（价格越低越好）

2. 计算最终分数：
   ```
   final_score = 0.35 × recall_fused_score
               + 0.50 × cross_encoder_score
               + 0.15 × business_score
   ```

3. 按 `final_score` 降序排序，取 Top-K（默认 10-20）

### 输出
- DataFrame，包含字段：
  - `final_score`: 最终推荐分数
  - `business_score`: 业务因素分数
  - 所有前序阶段的分数

### 评估方式
- **Overall Precision@K**: 最终推荐的准确率
- **User Satisfaction Score**: 综合评分、评论数的满意度
- **Safety Rate**: 无 DDI 风险、无禁忌症的比例

---

## 端到端评估指标

### 召回层指标
- Recall@10, Recall@50, Recall@100
- Coverage（有结果的查询占比）

### 排序层指标
- nDCG@5, nDCG@10, nDCG@20
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)

### 准确率指标
- Precision@5, Precision@10
- Hit@5, Hit@10（至少命中一个正确药物）

### 业务指标
- 平均推荐药物评分
- 平均价格
- DDI 风险率（需外接知识库）

---

## 权重配置

### 当前默认配置
```python
# 召回融合权重
recall_weight_semantic = 0.5
recall_weight_label = 0.5

# 标签召回内部权重
disease_weight = 0.55
symptom_weight = 0.45

# 最终打分权重
final_weight_recall = 0.35
final_weight_cross_encoder = 0.50
final_weight_business = 0.15

# 业务打分内部权重
rating_weight = 0.55
reviews_weight = 0.30
price_weight = 0.15
```

### 调优建议
- 若标签抽取置信度高：提高 `recall_weight_label`
- 若标签抽取置信度低：提高 `recall_weight_semantic`
- 若需要更强医学约束：提高 `disease_weight`
- 若需要更好用户体验：提高 `final_weight_business`
