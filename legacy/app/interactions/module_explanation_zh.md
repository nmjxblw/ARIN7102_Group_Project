# 模块说明

这份说明用于对外介绍当前这部分工作的代码结构，包括：

- 现在几个 `.py` 文件分别负责什么
- notebook 主要负责什么
- 在 notebook 中预期可以看到什么结果

## 一、当前整体结构

目前这一部分已经拆成两层：

1. **`.py` 文件负责可复用的核心逻辑**
2. **`ipynb` 负责调用这些逻辑并展示实验结果**

这样做的目的，是避免把所有逻辑都堆在 notebook 里，提升复用性、可维护性和展示清晰度。

## 二、各个 `.py` 文件分别做什么

### 1. `app/embedded_module/drug_embedding_engine.py`

这个文件负责 **药物语义向量化**。

它的主要功能是：

- 加载 PubMedBERT 模型
- 将药物的 `semantic_text` 编码成向量
- 将用户查询文本也编码成向量
- 支持将向量保存为 `.npy`
- 支持从本地 `.npy` 直接加载已有向量

一句话理解：
它是整个推荐流程里的 embedding 引擎。

### 2. `app/embedded_module/drug_knn_retriever.py`

这个文件负责 **基于 Faiss 的 KNN 检索**。

它的主要功能是：

- 把药物 embedding 建成 Faiss 索引
- 根据 query embedding 做近邻搜索
- 返回相似度最高的候选药物列表

一句话理解：
它是整个推荐流程里的语义召回模块。

### 3. `app/embedded_module/member_c_pipeline.py`

这个文件负责 **推荐流程中的公共逻辑**。

目前主要包含：

- 数据清洗辅助函数
- 症状名标准化
- 药物 `semantic_text` 构造
- 药物表结构化预处理
- baseline disease/symptom overlap 过滤
- 安全的 cosine similarity 计算
- hybrid retrieval 主流程

一句话理解：
它是把“数据准备 + baseline + hybrid 逻辑”统一封装起来的主流程模块。

## 三、notebook 现在主要负责什么

notebook 不再承担全部业务逻辑，而是主要负责：

1. 读取数据
2. 调用 `.py` 模块中的函数
3. 展示 baseline、pure KNN、hybrid 的结果
4. 做对比实验和可视化
5. 保留 RF 和 DDI 作为后续扩展实验

也就是说，notebook 现在更像是：

- 实验展示入口
- 结果对比面板
- 答辩与汇报用工作台

而不是“所有代码都写在里面”的草稿本。

## 四、在 notebook 中预期可以看到什么结果

### 1. 数据预处理部分

预期结果：

- 药物表被成功读取
- 新增结构化字段，例如：
  - `disease_key_list`
  - `symptom_list_normalized`
  - `semantic_text`

这说明药物数据已经被处理成适合后续推荐的格式。

### 2. embedding 部分

预期结果：

- 成功加载或生成 `drug_embeddings`
- 可以看到类似 `(5595, 768)` 这样的向量矩阵维度

如果本地已有 `.npy`，则 notebook 会直接读取，不会每次都重新编码。

### 3. baseline 部分

预期结果：

- 根据输入的 `disease_labels + symptoms`
- 返回一批在结构化字段上更直接匹配的候选药物

这个部分的特点是：

- 可解释性强
- 结果较稳
- 适合作为基线方法

### 4. pure KNN 部分

预期结果：

- 返回语义相似度最高的候选药物

但实验上可能会出现：

- 返回药物在 disease/symptom 上不够贴合
- 说明“全库纯 KNN”不够稳定

这个结果本身就是实验发现的一部分。

### 5. hybrid 部分

预期结果：

- 先经过 baseline 粗筛
- 再用 KNN 做精排
- 最终返回比 pure KNN 更稳、更合理的推荐结果

这是当前 notebook 主线里最核心的方法，也是目前最适合展示的结果。

### 6. 可视化部分

预期结果：

- 展示 pure KNN 与 hybrid 在分数分布上的差异
- 支持说明为什么 hybrid 比 pure KNN 更可靠

这部分主要用于帮助解释实验结论。

## 五、如何理解当前成果

目前这部分工作的核心不是单纯“做了一个 embedding”，而是已经形成了一个较完整的推荐实验主线：

1. 药物数据结构化
2. 语义文本构造
3. embedding 生成
4. baseline 检索
5. pure KNN 检索
6. hybrid 检索
7. 结果对比与可视化

在这个基础上，Random Forest 和 DDI 则作为进一步扩展功能继续保留。

## 六、一句话总结

现在这些 `.py` 文件负责真正可复用的推荐逻辑，notebook 负责调用这些模块并展示实验结果；在 notebook 中，最核心、最值得展示的结果是 baseline、pure KNN 与 hybrid 三者的对比，以及 hybrid 作为当前主方法的实验结论。
