# 实验药物召回 Aside Plan

## 目标

这份计划是主计划之外的旁路验证任务，目的只有三个：

1. 解释当前 `candidate_union` 指标提升到底主要来自哪里
2. 判断 `bm25` 在当前方案里是增益还是噪声
3. 判断当前 `dense/semantic` 路径到底是否可用，以及问题更像是 view 选择问题还是更上游的 embedding 资产问题

这份 aside 不负责推进主线集成，也不替换现有 `EXP_DRUG_RECALL_PLAN.md`。

## 为什么这是 Aside

当前已有结论是：

- `label_idf_only` 相比 `yinan` 线 label-only 只有小幅提升
- `candidate_union` 相比 `yinan` 线 fusion 提升明显
- `semantic-only` 当前结果接近或等于不可用

但现在还没有把“为什么会提升”拆干净。继续推进主计划会把后续工作建立在一个解释不充分的结果上，所以需要先补一轮解释性实验。

## 范围边界

这轮只允许改实验面，不改生产默认路径：

- 可以改：
  - `app/embedded_module/experimental_recall_pipeline.py`
  - `app/evaluation/run_exp_drug_recall.py`
  - 新增 `app/evaluation/compare_exp_vs_yinan.py`
  - 新增 `app/evaluation/diagnose_dense_recall.py`
- 不可以改：
  - FastAPI 默认 service
  - 生产默认 recommender
  - `EXP_DRUG_RECALL_PLAN.md`
  - `docs/架构.md`
  - `docs/新数据集.md`

## 工作项

### 1. 补三组剥离式 ablation

在实验 pipeline 中新增 3 个 mode：

- `candidate_union_no_prior`
- `candidate_union_no_bm25`
- `candidate_union_no_prior_no_bm25`

定义必须是 stage 级别剥离，不允许重新发明新的打分逻辑：

- `candidate_union_no_prior`
  - 候选来源保留：`disease + strict + symptom + bm25 + dense`
  - 完全移除 `prior`
- `candidate_union_no_bm25`
  - 候选来源保留：`disease + strict + symptom + dense + prior`
  - 完全移除 `bm25`
- `candidate_union_no_prior_no_bm25`
  - 候选来源保留：`disease + strict + symptom + dense`
  - 同时移除 `prior` 和 `bm25`

实现要求：

- 被移除的 stage 不能进入候选 union
- 被移除的 stage 对应 feature 必须强制为 0
- 继续复用现有 deterministic scorer，不改权重体系
- `local_ranker` 保留入口，但不纳入这轮默认批次

### 2. 更新实验 runner 默认批次

`run_exp_drug_recall.py` 的默认 mode 顺序改为：

1. `label_idf_only`
2. `label_bm25`
3. `candidate_union_no_prior`
4. `candidate_union_no_bm25`
5. `candidate_union_no_prior_no_bm25`
6. `candidate_union`

这轮 aside 不把 `local_ranker` 放进默认批次。

### 3. 生成 exp vs yinan 对比总结

新增脚本：`app/evaluation/compare_exp_vs_yinan.py`

输入固定读取：

- `data/eval_results_label.json`
- `data/eval_results_fusion.json`
- `data/eval_results_semantic.json`
- `artifacts/exp_drug_recall/metrics.json`
- `artifacts/exp_drug_recall/per_query_results.csv`

输出固定写到：

- `artifacts/exp_drug_recall/comparison_summary.md`
- `artifacts/exp_drug_recall/comparison_summary.json`
- `artifacts/exp_drug_recall/comparison_per_query.csv`

`comparison_summary.md` 必须至少包含以下对比：

- `yinan label` vs `exp label_idf_only`
- `yinan fusion` vs `exp candidate_union`
- `candidate_union` vs `candidate_union_no_prior`
- `candidate_union` vs `candidate_union_no_bm25`
- `candidate_union` vs `candidate_union_no_prior_no_bm25`

每行指标必须包含：

- `hit@20`
- `recall@20`
- `mrr`
- `precision@20`
- 每个指标对应的 delta

结论生成规则固定：

- 如果去掉 `prior` 的跌幅最大，结论写：`prior 是当前主要增益来源`
- 如果去掉 `bm25` 后结果更好或基本不变，结论写：`BM25 当前更像噪声`
- 如果 `semantic-only` 为 0 或 dense stage hit 为 0，结论写：`dense 当前不可用，不应被计入正向解释`

### 4. 单独做 dense 诊断

新增脚本：`app/evaluation/diagnose_dense_recall.py`

要求：

- 不重建 embedding
- 不下载模型
- 不改生产路径

固定输入：

- `match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv`
- `drug_comprehensive_embeddings.npy`
- `data/eval_dataset_verified.json`

对当前 3D embedding 固定评估三种投影：

- `view0 = emb[:, 0, :]`
- `view1 = emb[:, 1, :]`
- `mean = (emb[:, 0, :] + emb[:, 1, :]) / 2`

每个 variant 都要跑 semantic-only top-20，并输出：

- `hit@20`
- `recall@20`
- `mrr`
- 命中 query 数
- 失败 query 的样例及其 top semantic candidates

输出固定写到：

- `artifacts/exp_drug_recall/dense_diagnosis.json`
- `artifacts/exp_drug_recall/dense_diagnosis.md`

诊断结论规则固定：

- 如果三个 variant 都接近 0，结论写：`问题不只是 view 选择，更可能在 embedding 生成或对齐链路`
- 如果某个 variant 明显更好，结论写：`优先修 view / projection 策略`

## 产物

这轮 aside 完成时，必须同时具备：

- 根目录计划文档：`EXP_DRUG_RECALL_ASIDE_PLAN.md`
- 更新后的 `artifacts/exp_drug_recall/metrics.json`
- 更新后的 `artifacts/exp_drug_recall/per_query_results.csv`
- 更新后的 `artifacts/exp_drug_recall/stage_trace.jsonl`
- `artifacts/exp_drug_recall/comparison_summary.md`
- `artifacts/exp_drug_recall/comparison_summary.json`
- `artifacts/exp_drug_recall/comparison_per_query.csv`
- `artifacts/exp_drug_recall/dense_diagnosis.json`
- `artifacts/exp_drug_recall/dense_diagnosis.md`

## 成功标准

这轮 aside 必须能直接回答三件事，不允许还要靠人工猜：

1. 当前提升是否主要来自 `prior`
2. `bm25` 当前是在帮忙还是在制造噪声
3. `dense` 当前是否真的可用

如果做完后仍然回答不了这三件事，这轮 aside 视为没有完成。

## 约束

- 只在内层 repo 工作：
  - `/Users/jayden/Desktop/7012 datamining and text/project_march/ARIN7102_Group_Project`
- 统一基线使用当前 `data/eval_dataset_verified.json` 的 191 条 query
- 不动生产默认行为
- 不重算 `.npy`
- 不碰 `docs/架构.md` 和 `docs/新数据集.md`
- 这轮结束条件是“产出结论和 artifact”，不是“恢复主计划”

## 交接说明

交给别的 agent 时，按下面顺序做：

1. 先补 3 个剥离式 ablation mode
2. 跑完整 experiment batch，刷新 `artifacts/exp_drug_recall/`
3. 生成 `exp vs yinan` 对比总结
4. 做 dense 三视图诊断
5. 最后只汇报结论，不要顺手推进主计划

这轮做完以后，下一步是 review 这些 artifact 是否真的支持结论，再决定是否恢复主计划。
