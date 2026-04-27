# `app/interactions/evaluation_delivery.ipynb` 解释

这份说明用于配合 [`evaluation_delivery.ipynb`](/Users/jayden/Desktop/7012%20datamining%20and%20text/project_march/ARIN7102_Group_Project/app/interactions/evaluation_delivery.ipynb) 一起看，帮助解释这个 notebook 在做什么、为什么这样设计，以及结果应该如何解读。

## 1. 这个 notebook 是做什么的

这个 notebook 的目标是回答下面这个问题：

> 根据 `app/dataset_module/bert_training_dataset/medical_questions_dataset.json` 的疾病和症状标签，统计药物召回准确度，并评估我们向量库的匹配准确性。

所以它不是在做真实临床诊断，也不是在判断某个药物在现实医学上是否一定正确，而是在做一类 **label-based retrieval evaluation**：

- 输入：`disease labels + symptom labels`
- 输出：药物排序结果
- 评估：返回的药物是否和这些标签匹配

## 2. 为什么要先写这句边界说明

notebook 开头有一句固定说明：

`This evaluation reports label-based retrieval accuracy, not real clinical prescription accuracy.`

意思是：

- 这里的准确率，是“标签匹配准确率”
- 不是“真实临床处方准确率”

原因很简单：

- benchmark 数据里有 `question / disease / symptoms`
- 但没有每条样本对应的 gold drug answer

所以我们能评估的是：

- 系统有没有把标签更匹配的药排到前面

而不能直接评估：

- 系统是不是给出了真实医学上最正确的处方药

## 3. notebook 里比较了哪三种方法

### `baseline`

纯标签匹配方法。

做法是：

1. 看药物的 `matched_disease_keys` 和 query 的 disease 有没有重合
2. 看药物的 `matched_symptoms` 和 query 的 symptoms 有没有重合
3. 根据 overlap 和评分排序

特点：

- 不用 embedding
- 不用向量库
- 更像结构化检索 / rule-based retrieval

### `pure_knn`

纯向量检索方法。

做法是：

1. 把 query 的 `disease + symptoms` 拼成一段短文本
2. 用 PubMedBERT 把这段 query 文本编码成向量
3. 用这个 query 向量去全药库 embedding 上做相似度搜索

特点：

- 不做 baseline 粗筛
- 直接在全库检索
- 最能体现“向量库本身的匹配准确性”

### `hybrid`

混合方法。

做法是：

1. 先用 baseline 筛出 disease / symptom 相关候选
2. 再在候选集内用 embedding similarity 做 rerank

特点：

- 兼顾规则匹配和语义排序
- 更像“候选召回 + 精排”
- 通常比 pure KNN 更稳

## 4. embedding 到底用在了哪里

只有 `pure_knn` 和 `hybrid` 用到了 embedding。

这里的 embedding 不是对单个词编码，而是对一段文本编码。

### query 侧

query 侧输入不是原始病人自然语言全文，而是由标签拼出来的一段短文本，例如：

```text
Related diseases: common_cold. Target symptoms: fever, headache, cough.
```

### drug 侧

drug 侧 embedding 用的也不是单个词，而是药物的 `semantic_text`。

这个 `semantic_text` 是由多个字段拼出来的，例如：

- `drug_name`
- `generic_name`
- `drug_classes`
- `matched_disease_keys`
- `matched_symptoms`
- `medical_condition_description`
- `disease_description`
- `side_effects`

### 当前默认 embedding

当前 notebook 默认使用的是：

- model: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- pooling: `[CLS]`
- embedding file: [`interaction/drug_comprehensive_embeddings.npy`](/Users/jayden/Desktop/7012%20datamining%20and%20text/project_march/ARIN7102_Group_Project/interaction/drug_comprehensive_embeddings.npy)

所以一句话总结：

- `baseline` 不用 embedding
- `pure_knn` 和 `hybrid` 都用 semantic embedding
- embedding 的输入不是一个词，而是一段 semantic text

## 5. 各个指标是什么意思

notebook 里重点解释了这些指标：

### `hit@5_strict`

Top-5 结果中，是否至少有一个药物同时满足：

- 命中正确 disease
- 命中至少一个正确 symptom

如果有，当前 query 的这个指标记为 1；否则记为 0。最后对所有 query 取平均。

### `precision@5_strict`

Top-5 结果中，有多少比例的药物满足 strict 条件。

比如 Top-5 里有 3 个 strict relevant drug，那这一条 query 的 `precision@5_strict = 3/5 = 0.6`。

### `mrr_strict`

看第一个 strict relevant drug 出现在第几名。

- 如果第 1 名就是 strict relevant，分数是 1
- 如果第 2 名才出现，分数是 1/2
- 如果第 5 名才出现，分数是 1/5

最后对所有 query 取平均。

### `ndcg@5`

衡量 Top-5 结果整体排序质量。

不是只看“有没有相关药”，还看“更相关的药是不是排得更靠前”。

### `ndcg@10`

和 `ndcg@5` 一样，只是观察窗口变成 Top-10。

## 6. 为什么 notebook 默认不是全量测试

在 notebook 里，默认写的是：

```python
limit=100
```

这表示默认先跑前 100 条 query，而不是直接跑完整个数据集。

原因是：

- `baseline` 跑得快
- `pure_knn` 和 `hybrid` 需要加载模型并逐条做 query embedding
- 全量跑 8600 条会比较慢

所以 notebook 默认是一个 **delivery / demo version**，方便先产出结果和截图。

如果要做正式结果，可以把 `limit=100` 改成：

- `limit=None`
- 或者删掉 `limit` 参数

## 7. 如何解读目前已经跑出来的结果

如果结果大致是：

- `baseline` 很高甚至接近满分
- `pure_knn` 明显低一些
- `hybrid` 又接近 `baseline`

那更合理的解读是：

1. `baseline` 很强，说明标签匹配本身已经能解决大部分问题
2. `pure_knn` 能体现向量库本身的独立匹配能力
3. `hybrid` 很强，说明向量库更适合做 reranking，而不是独立做全库召回

也就是说，当前向量库的更合理定位通常是：

- 不是单独主召回器
- 而是 baseline candidate set 上的 semantic reranker

## 8. 为什么 `baseline` 不一定永远 100%

虽然 `baseline` 看起来像“直接查标签”，但它并不是严格意义上的 exact lookup。

原因是它允许：

- disease 命中
或
- symptom 命中

就进入候选集。

所以：

- 检索逻辑可能是 `disease OR symptom`
- 评估 strict 指标却要求 `disease AND symptom`

两边定义不完全一致时，`baseline` 在更大样本上也可能低于 100%。

这不一定是代码错了，更可能是 retrieval rule 和 evaluation rule 本来就不是同一个标准。

## 9. 这份 notebook 最适合怎么交付给对方

建议把它当成一个简洁的 delivery notebook：

1. 说明 benchmark 数据来源
2. 说明三种方法
3. 给出 summary table
4. 给出关键指标解释
5. 最后给一句结论

一句结论模板可以写成：

> On the label-based benchmark built from disease and symptom annotations, pure KNN reflects the standalone matching ability of the vector store, while hybrid reflects the reranking gain on top of baseline candidate filtering.

## 10. 一句话总结

这份 notebook 的价值不在于证明“真实临床推荐准确率”，而在于用统一 benchmark 对 `baseline / pure_knn / hybrid` 三种方法做可解释、可展示、可比较的药物召回评估，并回答“向量库本身匹配得怎么样”这个问题。
