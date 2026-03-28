# 评估数据集格式说明

## 数据集结构

评估数据集用于量化推荐系统的准确率，每条记录包含：

### 输入部分
- `query_id`: 查询唯一标识符
- `symptom_text`: 用户自然语言症状描述
- `diseases`: 疾病标签列表（带置信度）
- `symptoms`: 症状标签列表（带置信度）

### 输出部分（Ground Truth）
- `relevant_drugs`: 正确/相关的药物列表
- `relevance_scores`: 每个药物的相关性分数（可选，用于 nDCG 计算）

---

## JSON 格式示例

```json
{
  "query_id": "eval_001",
  "symptom_text": "I have severe headache with high fever and body aches",
  "diseases": [
    {"name": "influenza", "confidence": 0.85},
    {"name": "common_cold", "confidence": 0.45}
  ],
  "symptoms": [
    {"name": "fever", "confidence": 0.92},
    {"name": "headache", "confidence": 0.88},
    {"name": "body_aches", "confidence": 0.76}
  ],
  "relevant_drugs": [
    "acetaminophen",
    "ibuprofen",
    "aspirin"
  ],
  "relevance_scores": {
    "acetaminophen": 3,
    "ibuprofen": 3,
    "aspirin": 2
  }
}
```

---

## 相关性分数定义

- **3**: 高度相关（首选药物）
- **2**: 相关（可接受的替代药物）
- **1**: 弱相关（边缘情况可用）
- **0**: 不相关

---

## 数据集生成策略

### 方法1：基于药物反向生成（推荐）
从药物表中选择药物，根据其适应症、症状字段生成对应的查询。

**优点**：
- 自动化程度高
- 覆盖面广
- 可快速生成大量样本

**缺点**：
- 可能不够自然
- 需要人工验证质量

### 方法2：人工标注
由医学专家或标注人员根据真实场景编写查询并标注正确药物。

**优点**：
- 质量高
- 更接近真实用户查询

**缺点**：
- 成本高
- 速度慢

### 方法3：混合方式（推荐用于本项目）
1. 使用方法1自动生成初始数据集
2. 人工抽样验证和修正（10-20%）
3. 保留高质量样本作为评估集

---

## 评估指标计算方式

### Precision@K
```
Precision@K = (推荐的K个药物中正确的数量) / K
```

### Recall@K
```
Recall@K = (推荐的K个药物中正确的数量) / (总的正确药物数量)
```

### Hit@K
```
Hit@K = 1 if 推荐的K个药物中至少有1个正确, else 0
```

### nDCG@K
```
DCG@K = Σ(rel_i / log2(i+1)) for i in [1, K]
IDCG@K = 理想排序下的 DCG@K
nDCG@K = DCG@K / IDCG@K
```

### MRR (Mean Reciprocal Rank)
```
RR = 1 / (第一个正确药物的排名)
MRR = 所有查询的 RR 平均值
```
