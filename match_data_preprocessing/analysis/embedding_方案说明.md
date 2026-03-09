# 药名 Embedding 聚类验证与维度分析方案

## 数据基础

- 数据集：`data/drug_ingredient_pairs.json`
- 记录数：1,624 条（商品名 → 有效成分配对）
- 有效成分种类：503 种（每种 ≥2 个品牌名）

---

## 目标 1：验证聚类是否准确

### 思路

> 对 1,624 个 `drug_name` 做 embedding → 聚类 → 用 `active_ingredient` 作为 ground truth 标签，计算聚类评估指标。

### 推荐方案：KMeans + ARI/NMI

**步骤：**

1. 用 SapBERT 对所有 `drug_name` 生成 768 维 embedding
2. 用 KMeans 聚类（K 设为成分种类数 503，或用 Silhouette Score 搜索最优 K）
3. 用 `active_ingredient` 的编码作为真实标签，计算评估指标

**评估指标：**

| 指标 | 范围 | 说明 |
|------|------|------|
| **ARI**（Adjusted Rand Index） | [-1, 1] | 校正随机效应后的聚类一致性，>0.5 为好 |
| **NMI**（Normalized Mutual Information） | [0, 1] | 聚类与真实标签的互信息，越接近 1 越好 |

**核心代码：**

```python
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import torch, json, numpy as np

# 1. 加载数据
with open("data/drug_ingredient_pairs.json") as f:
    data = json.load(f)
drug_names = [d["drug_name"] for d in data]
ingredients = [d["active_ingredient"] for d in data]

# 2. Embedding
tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model.eval()

batch_size = 64
all_embeddings = []
for i in range(0, len(drug_names), batch_size):
    batch = drug_names[i:i+batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
    all_embeddings.append(cls_emb)
embeddings = np.concatenate(all_embeddings, axis=0)  # (1624, 768)

# 3. 聚类
le = LabelEncoder()
true_labels = le.fit_transform(ingredients)
K = len(set(true_labels))  # 503

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
pred_labels = kmeans.fit_predict(embeddings)

# 4. 评估
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)
print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}")
```

**补充说明：**
- 若 ARI 接近 0，说明 embedding 仅靠药名文本无法有效区分有效成分（药名和成分命名差异大时预期如此）
- 可额外尝试 DBSCAN 或层次聚类作为对比

---

## 目标 2：找出 embedding 中哪些维度与有效成分最相关

### 思路

> 将 768 维 embedding 的每一维视为特征，`active_ingredient` 作为分类标签，通过 ANOVA F-test 找出判别力最强的维度。

### 推荐方案：ANOVA F-test

**原理：** 对每个维度做单因素方差分析——如果某个维度在不同有效成分之间的均值差异显著大于组内方差，说明该维度对区分有效成分有贡献。

**步骤：**

1. 输入 embedding 矩阵 (1624, 768) 和有效成分标签 (1624,)
2. 对 768 个维度分别计算 F-score 和 p-value
3. 按 F-score 降序排列，选出 Top-K 维度
4. 用 Top-K 维度做 t-SNE 可视化，验证这些维度是否确实有区分力

**核心代码：**

```python
from sklearn.feature_selection import f_classif
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 1. ANOVA F-test
F_scores, p_values = f_classif(embeddings, true_labels)

# 2. 排序，找 Top 维度
top_k = 50
top_dims = np.argsort(F_scores)[::-1][:top_k]
print(f"Top {top_k} 维度索引: {top_dims}")
print(f"对应 F-score: {F_scores[top_dims]}")
print(f"显著维度数 (p<0.05): {np.sum(p_values < 0.05)}")
print(f"显著维度数 (p<0.01): {np.sum(p_values < 0.01)}")

# 3. 可视化对比：全部维度 vs Top-K 维度
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 全部 768 维
tsne_all = TSNE(n_components=2, random_state=42, perplexity=30)
coords_all = tsne_all.fit_transform(embeddings)
axes[0].scatter(coords_all[:, 0], coords_all[:, 1], c=true_labels, cmap="tab20", s=5, alpha=0.6)
axes[0].set_title("t-SNE: All 768 dims")

# Top-K 维度
embeddings_top = embeddings[:, top_dims]
tsne_top = TSNE(n_components=2, random_state=42, perplexity=30)
coords_top = tsne_top.fit_transform(embeddings_top)
axes[1].scatter(coords_top[:, 0], coords_top[:, 1], c=true_labels, cmap="tab20", s=5, alpha=0.6)
axes[1].set_title(f"t-SNE: Top {top_k} dims (by F-score)")

plt.tight_layout()
plt.savefig("embedding_tsne_comparison.png", dpi=150)
plt.show()
```

**结果解读：**
- F-score 高的维度 = 对区分有效成分贡献大的维度
- 如果 Top-K 维度的 t-SNE 图比全维度图聚类更清晰，说明筛选有效
- p-value < 0.01 的维度数量反映了 embedding 中有多少维度编码了"有效成分"信息

---

## 方案对比总结

| | 目标 1：聚类验证 | 目标 2：维度分析 |
|--|------------------|------------------|
| **方法** | KMeans + ARI/NMI | ANOVA F-test |
| **优点** | 简单直接，指标成熟 | 无需训练，统计上严谨 |
| **计算量** | 中等（聚类 1624 × 768） | 低（768 次 F-test） |
| **依赖** | sklearn | sklearn |
| **输出** | ARI/NMI 数值 | Top-K 维度索引 + F-score |
