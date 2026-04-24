# 实验药物召回 Phase 1 记录

## 目标

Phase 1 的目标不是继续推进主线集成，而是先把当前实验结果解释清楚，回答三个问题：

1. `candidate_union` 的提升主要来自哪里
2. `bm25` 和 `prior` 在当前方案里到底是增益还是噪声
3. `dense/semantic` 当前是否可用

## Phase 1 做了什么

### 1. 补了 aside 计划要求的三组 ablation

在实验 pipeline 中新增并跑完以下 mode：

- `candidate_union_no_prior`
- `candidate_union_no_bm25`
- `candidate_union_no_prior_no_bm25`

相关代码：

- [app/embedded_module/experimental_recall_pipeline.py](app/embedded_module/experimental_recall_pipeline.py)
- [app/evaluation/run_exp_drug_recall.py](app/evaluation/run_exp_drug_recall.py)

### 2. 生成了 exp vs yinan 对比产物

新增并修正了对比脚本：

- [app/evaluation/compare_exp_vs_yinan.py](app/evaluation/compare_exp_vs_yinan.py)

产物：

- [artifacts/exp_drug_recall/comparison_summary.md](artifacts/exp_drug_recall/comparison_summary.md)
- [artifacts/exp_drug_recall/comparison_summary.json](artifacts/exp_drug_recall/comparison_summary.json)
- [artifacts/exp_drug_recall/comparison_per_query.csv](artifacts/exp_drug_recall/comparison_per_query.csv)

### 3. 修了 ablation trace/metrics 对齐问题

修正了 no-prior / no-bm25 模式下 `stage_hit_*` 仍被错误记为命中的问题，使 trace 与 ablation 定义一致。

### 4. 补了 dense 三视图诊断脚本

新增：

- [app/evaluation/diagnose_dense_recall.py](app/evaluation/diagnose_dense_recall.py)

这版脚本已经改成：

- 只读 root `drug_comprehensive_embeddings.npy`
- 固定评估 `view0 / view1 / mean`
- 使用 191 条 verified queries
- 强制离线、本地缓存 only，不允许自动下载模型

### 5. 额外补了 Task A sanity check

为了解释 `no_bm25` 和 `no_prior_no_bm25` 为什么提升异常大，新增：

- [app/evaluation/check_ablation_sanity.py](app/evaluation/check_ablation_sanity.py)

产物：

- [artifacts/exp_drug_recall/ablation_sanity_check.md](artifacts/exp_drug_recall/ablation_sanity_check.md)
- [artifacts/exp_drug_recall/ablation_sanity_check.json](artifacts/exp_drug_recall/ablation_sanity_check.json)

## Phase 1 关键结果

### 主指标

| Mode | hit@20 | recall@20 | mrr | precision@20 |
|---|---:|---:|---:|---:|
| `label_idf_only` | 0.6859 | 0.5603 | 0.3199 | 0.0775 |
| `candidate_union` | 0.7801 | 0.6832 | 0.4030 | 0.0958 |
| `candidate_union_no_prior` | 0.7749 | 0.6878 | 0.3847 | 0.0953 |
| `candidate_union_no_bm25` | 0.9372 | 0.8809 | 0.6135 | 0.1152 |
| `candidate_union_no_prior_no_bm25` | 0.9843 | 0.9475 | 0.6313 | 0.1204 |

## Phase 1 结论

### 结论 1：`label-only` 和 `yinan label` 基本同一水平

`exp label_idf_only` 与 `yinan label-only` 基本持平，只是有小幅改良，不是路线级变化。

### 结论 2：当前 `BM25` 不应进入主线

从对比产物可以直接看到：

- 去掉 `bm25` 后，`hit@20` 和 `recall@20` 都显著提升
- `label_bm25` 也明显弱于 `label_idf_only`

当前结论应视为：

- `BM25` 当前更像噪声或负贡献
- 不应被作为主线正向能力计入

### 结论 3：`prior` 不是当前主增益来源

去掉 `prior` 后，指标没有出现大幅回落，说明当前 `candidate_union` 的提升并不是主要由 `prior expansion` 驱动。

### 结论 4：`candidate_union_no_prior_no_bm25` 的超大提升不是来自新增召回源

Task A 证明了：

- `label_idf_only` 和 `candidate_union_no_prior_no_bm25` 在 `191/191` 个 query 上候选集完全相同

因此：

- `0.6859 -> 0.9843` 这段提升，不是“召回到更多候选”
- 而是同一批 `label core` 候选被 deterministic scorer 重新排序后，排序质量显著提升

### 结论 5：当前 `candidate_union` 的异常主要来自 `bm25 + 1000 cap`

Task A 还证明了：

- `candidate_union` 在 `191/191` 个 query 上都触发了 `1000` cap
- `bm25` 参与后，在 `158/191` 个 query 上把 `label core` 候选挤出了最终候选池
- pre-cap union 平均大小约为 `1556.39`
- `bm25` 原始分数尺度明显大于 strict label 分数：
  - `bm25` mean per-query max = `57.3521`
  - `strict` mean per-query max = `13.0218`

因此当前实现中，`_rows_for_mode()` 在 cap 前直接相加各 stage 原始分数，会对 `bm25` 产生数值偏置。

### 结论 6：`dense` 已完成三视图诊断，但暂不进入主线

当前结论应更新为：

- `dense` 不是完全不可用；三视图诊断已经跑出非零结果。
- `view0` hit@20 = 0.2461
- `view1` hit@20 = 0.2984
- `vmean` hit@20 = 0.3246，是当前三种投影里最好的一种。
- 但这个结果仍明显弱于 `label core + deterministic rerank`。
- 因此 dense 不能作为 Phase 1 主增益解释，也不应直接进入 Phase 2 主线。

当前 dense 状态应记为：

`diagnosed, excluded from mainline by default`

也就是：

- dense 已经不是 `deferred`
- 但 dense 当前不作为 Phase 2 主线能力
- 如果后续还有时间，再把 projection/view 当作可选探索项处理

## Phase 1 是否完成

如果只看 aside / 验证目标，Phase 1 状态可以记为：

`Phase 1 done, with dense excluded from the mainline`

也就是：

- `BM25 / prior / label core` 已验证完
- `Task A` 已完成并解释清楚异常提升来源
- dense 已完成三视图诊断，但默认不纳入 Phase 2 主线

## 对 Phase 2 的直接影响

基于 Phase 1 结果，下一阶段建议是：

1. 主线先收敛到 `label core`
   - `disease`
   - `strict`
   - `symptom`
2. 暂不把 `BM25` 放进主线
3. `prior` 不作为当前主增益来源推进
4. `dense` 不纳入主线；如需继续，只作为可选探索
5. `local_ranker` 后置，在候选池稳定后再推进

## 本阶段核心产物

- [EXP_DRUG_RECALL_ASIDE_PLAN.md](EXP_DRUG_RECALL_ASIDE_PLAN.md)
- [artifacts/exp_drug_recall/metrics.json](artifacts/exp_drug_recall/metrics.json)
- [artifacts/exp_drug_recall/per_query_results.csv](artifacts/exp_drug_recall/per_query_results.csv)
- [artifacts/exp_drug_recall/stage_trace.jsonl](artifacts/exp_drug_recall/stage_trace.jsonl)
- [artifacts/exp_drug_recall/comparison_summary.md](artifacts/exp_drug_recall/comparison_summary.md)
- [artifacts/exp_drug_recall/ablation_sanity_check.md](artifacts/exp_drug_recall/ablation_sanity_check.md)
