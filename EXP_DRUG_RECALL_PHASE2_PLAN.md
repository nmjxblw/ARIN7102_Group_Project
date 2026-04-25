# Experimental Drug Recall Phase II Plan

## Final Update

Phase II 已经进入收尾状态。实验结果表明：

1. `label_core_rerank` 已经成为干净主线。
2. 旧的 half row-level 评估口径不合理，因为它把同一 disease 下的其他正确药误算为 false negative。
3. half-data 的有效 Phase II 口径改为：
   - 先合并 `drug_data_half_1.json` 和 `drug_data_half_2.json`
   - 再按 `disease` 聚合成 query-level relevant drug pool
   - 可选再按 `disease_symptom` 做更细粒度检查
4. 新增 verified 742 条结果已经跑完，用来确认 191 条 verified 高分不是偶然。

Phase II 最终状态：

| Evaluation | Queries | hit@20 | precision@20 | recall@20 | ndcg@20 | mrr |
|---|---:|---:|---:|---:|---:|---:|
| old verified | 191 | 0.9843 | 0.1204 | 0.9475 | 0.6716 | 0.6313 |
| new verified | 742 | 0.9879 | 0.1172 | 0.9638 | 0.6925 | 0.6654 |
| half disease-level | 31 | 0.7419 | 0.2806 | 0.1477 | 0.3453 | 0.5110 |

Final Phase II conclusion:

```text
Verified-set recall is saturated and stable, while clean half-derived
disease-level evaluation still exposes coverage gaps.
```

## Summary

Phase II 的目标是把 Phase I 的结论落成一个干净主线，并把 half-data
变成两条可比较的辅助线：

1. 主线：实现 `label_core_rerank`
   - 召回只保留 `disease + strict + symptom`
   - fixed-weight deterministic rerank 保留，但去掉 `BM25 / dense / prior stage`
2. Half Line A：把 half-data 接成 clean-data sanity check
   - 不使用 half confidence
   - 不再用 row-level 单药口径做主结论
   - 合并 half1 + half2 后按 disease 聚合 relevant drug pool
   - 主要看 `hit@k` / `precision@k` / `recall@k` / `mrr` / `ndcg@k`
3. Half Line B：做 confidence 分桶分析
   - 统计原始 half confidence
   - 画 density / ECDF
   - 按 `p33 / p67` 分成 `low / mid / high`
   - 对比 `label_idf_only` 和 `label_core_rerank` 在各 bucket 的效果

本阶段只动实验链路，不动生产链路，不做 `local_ranker`。

## Step 0: Boundaries

只允许修改这些文件：

- `app/embedded_module/experimental_recall_pipeline.py`
- `app/evaluation/run_exp_drug_recall.py`
- 新增 `app/evaluation/half_data_adapter.py`
- 新增 `app/evaluation/analyze_half_confidence.py`
- 新增 `EXP_DRUG_RECALL_PHASE2.md`

明确不改：

- `app/fastapi_module/service.py`
- `app/embedded_module/dual_recall_pipeline.py`
- `docs/架构.md`
- `docs/新数据集.md`

运行环境统一优先使用：

```bash
conda run -n 7606 python ...
```

## Phase II-A: Mainline Mode

### Goal

在 `app/embedded_module/experimental_recall_pipeline.py` 中新增一个真正干净的
`label_core_rerank`。

### Implementation Requirements

1. 新增 mode：
   - `label_core_rerank`

2. 抽出统一常量：
   - `MODE_STAGE_MAP`
   - 不再在 `_rows_for_mode()` 和 `_build_trace()` 中各写一份 mode-stage 对应关系

3. `label_core_rerank` 的 candidate stage 固定为：
   - `disease`
   - `strict`
   - `symptom`

4. `label_core_rerank` 的 rerank 因子固定保留：
   - `label_idf_score`
   - `symptom_coverage`
   - `quality_prior`
   - `stage_strict`
   - `stage_disease`
   - `others_penalty`

5. `label_core_rerank` 中强制关闭：
   - `bm25_score = 0`
   - `dense_score = 0`
   - `stage_prior = 0`

6. `label_core_rerank` 的 `final_score` 固定为：

```text
final_score =
    0.50 * label_idf_score
  + 0.20 * symptom_coverage
  + 0.10 * quality_prior
  + 0.06 * stage_strict
  + 0.03 * stage_disease
  - others_penalty
```

7. 保留现有其他 mode，不改变其语义；本轮只新增主线 mode 和公共常量重构。

### Acceptance Criteria

- CLI 能识别 `label_core_rerank`
- `label_core_rerank` 的 trace 中：
  - `stage_hit_bm25 == 0.0`
  - `stage_hit_dense == 0.0`
  - `stage_hit_prior == 0.0`
- `label_core_rerank` 与 `label_idf_only` 使用相同 candidate stage，只改变排序，
  不改变主线候选来源

## Phase II-B: Half Line A (Clean Half Sanity Check Without Confidence)

### Goal

把 half-data 接进现有 evaluator，但不把 half confidence 注入当前主线。

Final update: row-level half evaluation is deprecated for main conclusions. The
corrected Line A merges half splits first, then groups by disease.

### Files

- 新增 `app/evaluation/half_data_adapter.py`
- 修改 `app/evaluation/run_exp_drug_recall.py`

### Data Semantics

half 原始 JSON 中的数值是 drug-side association confidence，不是 query-side
prediction confidence。因此在 Phase II-A/B 中：

- 不把它写进当前 query-side `confidence`
- 不让它参与当前 `label_core_rerank` 或 `label_idf_only` 的 recall/rerank
- 它只留给 Phase II-C 的分桶分析使用

### Adapter Output Schema

Adapter 支持三种 grouping：

1. `row`
   - legacy / deprecated
   - 每行只保留一个 `drug_name`
   - 只用于追溯旧实验，不作为 Phase II 主结论

2. `disease`
   - final Phase II main half view
   - 合并 half1 + half2
   - 对每个 disease 合并所有 half drug names

3. `disease_symptom`
   - optional follow-up view
   - 对每个 disease + symptom pair 合并 drug names
   - 用于检查 symptom 粒度是否过严或过松

`disease` grouping 输出示例：

```json
{
  "query_id": "half_all_disease_acne_0001",
  "symptom_text": "Drugs for acne.",
  "diseases": [{"name": "acne", "confidence": 0.95}],
  "symptoms": [],
  "relevant_drugs": ["aczone", "accutane", "tretinoin"]
}
```

规则：

1. `query_id`
   - row mode: `half1_000001`, `half2_000001`, ...
   - merged disease mode: `half_all_disease_<disease>_<idx>`

2. `diseases`
   - 只保留 label name，不显式写 query confidence
   - 例如 `{"acne": 0.95}` -> `"acne"`

3. `symptoms`
   - 同上，只保留 label name

4. `symptom_text`
   - 固定为空字符串 `""`

5. `relevant_drugs`
   - row mode: `[drug_name]`
   - disease mode: all canonicalized half drugs for that disease
   - disease-symptom mode: all canonicalized half drugs for that pair

6. `relevance_scores`
   - use flat score `3` for all half-derived relevant drugs when NDCG is needed
   - interpretation: ungraded relevant pool, not clinical gold relevance

7. Drug-name canonicalization
   - map half spellings such as `abatuss_dmx` to table names such as
     `abatuss dmx` where possible
   - otherwise exact-match metrics undercount valid hits

### Runner Changes

1. 在 `app/evaluation/run_exp_drug_recall.py` 中新增：
   - `--eval-kind {verified,half}`
   - `--half-grouping {row,disease,disease_symptom}`
   - `--half-extra-dataset`
   - 默认 `verified`

2. half 模式下：
   - 先用 `half_data_adapter.py` 转换 / 聚合
   - 再进入统一 evaluation 流程

3. 把：

```python
query["symptom_text"]
```

改成：

```python
query.get("symptom_text", "")
```

4. half 模式只允许：
   - `label_idf_only`
   - `label_core_rerank`

### Line A Run Commands

Deprecated row-level commands are retained only for historical comparison:

```bash
conda run -n 7606 python app/evaluation/run_exp_drug_recall.py \
  --eval-kind half \
  --half-grouping row \
  --eval-dataset app/dataset_module/drugs_training_dataset/drug_data_half_1.json \
  --modes label_idf_only label_core_rerank \
  --artifact-dir artifacts/exp_drug_recall/phase2_half1
```

```bash
conda run -n 7606 python app/evaluation/run_exp_drug_recall.py \
  --eval-kind half \
  --half-grouping row \
  --eval-dataset app/dataset_module/drugs_training_dataset/drug_data_half_2.json \
  --modes label_idf_only label_core_rerank \
  --artifact-dir artifacts/exp_drug_recall/phase2_half2
```

Final disease-level command:

```bash
conda run -n 7606 env PYTHONPATH=app python app/evaluation/run_exp_drug_recall.py \
  --eval-kind half \
  --half-grouping disease \
  --eval-dataset app/dataset_module/drugs_training_dataset/drug_data_half_1.json \
  --half-extra-dataset app/dataset_module/drugs_training_dataset/drug_data_half_2.json \
  --modes label_core_rerank \
  --k-values 5 10 20 \
  --embeddings /tmp/nonexistent_drug_embeddings.npy \
  --artifact-dir artifacts/exp_drug_recall/phase2_half_all_disease
```

### Line A Outputs

每个 artifact 目录至少包含：

- `metrics.json`
- `per_query_results.csv`
- `stage_trace.jsonl`
- `asset_manifest.json`

Line A final analysis should prioritize the corrected disease-level output:

- `hit@5`, `hit@10`, `hit@20`
- `precision@5`, `precision@10`, `precision@20`
- `recall@5`, `recall@10`, `recall@20`
- `ndcg@5`, `ndcg@10`, `ndcg@20`
- `mrr`

Interpretation rule:

- row-level half metrics are deprecated
- disease-level half metrics are clean-data sanity checks, not formal verified
  benchmark metrics

## Phase II-C: Half Line B (Confidence Stratification)

### Goal

利用 half 原始 confidence 做描述性和分层效果分析，不改变当前主线打分。

### File

新增：

- `app/evaluation/analyze_half_confidence.py`

### Inputs

Line B 不直接 rerun pipeline。它读取：

1. 原始 half 数据：
   - `app/dataset_module/drugs_training_dataset/drug_data_half_1.json`
   - `app/dataset_module/drugs_training_dataset/drug_data_half_2.json`

2. Line A 结果：
   - `artifacts/exp_drug_recall/phase2_half1/per_query_results.csv`
   - `artifacts/exp_drug_recall/phase2_half2/per_query_results.csv`

### `row_conf` Definition

对每条 half row 计算：

```text
if symptom list is non-empty:
    row_conf = 0.7 * max(disease_conf) + 0.3 * mean(symptom_conf)
else:
    row_conf = max(disease_conf)
```

固定规则：

- `disease_conf` 取该 row 所有 disease confidence 的最大值
- `symptom_conf` 取该 row 所有 symptom confidence 的均值
- 若无 symptom，直接用 `max(disease_conf)`
- 不对该值再做额外归一化

### Distribution and Buckets

1. 合并 half1 + half2 共 `2931` 行，形成一个统一分布
2. 在合并后的 `row_conf` 上计算：
   - density
   - ECDF
   - `p33`
   - `p67`
3. 分桶规则固定：
   - `low`: `row_conf <= p33`
   - `mid`: `p33 < row_conf <= p67`
   - `high`: `row_conf > p67`

统一使用全局阈值，不要 half1、half2 各算各的阈值。

### Required Artifacts

输出目录：

- `artifacts/exp_drug_recall/phase2_half_confidence`

至少生成：

- `half_confidence_summary.json`
- `half_confidence_rows.csv`
- `half_confidence_bucket_metrics.json`
- `half_confidence_bucket_metrics.md`
- `half_confidence_density.png`
- `half_confidence_ecdf.png`

### Line B Metrics

按 bucket 分别统计以下指标，按 mode 对比：

- `count`
- `hit@5`
- `hit@10`
- `hit@20`
- `recall@5`
- `recall@10`
- `recall@20`
- `mrr`

必须输出下面两种视角：

1. `label_idf_only` 在 `low / mid / high` 的表现
2. `label_core_rerank` 在 `low / mid / high` 的表现
3. 额外输出 `delta = label_core_rerank - label_idf_only` 的 bucket 对比表

### Questions Line B Must Answer

最终文档里必须明确回答：

1. high bucket 是否明显比 low bucket 更容易命中
2. `label_core_rerank` 的提升是否集中在 high bucket
3. 如果 bucket 差异很小，说明 half confidence 只适合作描述统计，不适合进入后续训练设计
4. 如果 high bucket 明显更强，说明 Phase III 可以考虑把 `row_conf` 用作 `local_ranker` 的 sample weight

### Plotting Constraints

- 统一用 `conda run -n 7606 python`
- 不额外安装依赖
- 若绘图失败但数据统计成功，仍需产出：
  - `half_confidence_rows.csv`
  - `half_confidence_bucket_metrics.json`
  - `half_confidence_bucket_metrics.md`
- 并在最终文档中明确标记 `plot generation failed, numeric bucket analysis completed`

## Phase II-D: Verified Evaluation and Report

### Verified Runs

先 smoke，再 full。

#### Smoke

```bash
conda run -n 7606 python app/evaluation/run_exp_drug_recall.py \
  --modes label_idf_only label_core_rerank candidate_union_no_prior_no_bm25 \
  --limit 10 \
  --artifact-dir artifacts/exp_drug_recall/phase2_verified_smoke
```

#### Full

```bash
conda run -n 7606 python app/evaluation/run_exp_drug_recall.py \
  --modes label_idf_only label_core_rerank candidate_union_no_prior_no_bm25 \
  --artifact-dir artifacts/exp_drug_recall/phase2_verified
```

#### Expanded Verified

```bash
conda run -n 7606 env PYTHONPATH=app python app/evaluation/run_exp_drug_recall.py \
  --eval-kind verified \
  --eval-dataset data/eval_dataset_verified_1000_deepseek_v4_flash.json \
  --modes label_core_rerank \
  --k-values 5 10 20 \
  --embeddings /tmp/nonexistent_drug_embeddings.npy \
  --artifact-dir artifacts/exp_drug_recall/phase2_verified_1000
```

Purpose: verify that the 191-query result is not a small-sample artifact.

### Report

新增：

- `EXP_DRUG_RECALL_PHASE2.md`

### Report Must Include

1. 主线定义
   - `label_core_rerank` 是什么
   - 保留哪些召回 stage
   - 保留哪些 rerank 因子
   - 明确关闭哪些项

2. verified 正式结论
   - 比较：
     - `label_idf_only`
     - `label_core_rerank`
     - `candidate_union_no_prior_no_bm25`
   - 额外加入 new verified 742 条稳定性检查
   - 至少报告：
     - `hit@20`
     - `precision@20`
     - `recall@20`
     - `ndcg@20`
     - `mrr`
   - 明确 `label_core_rerank` 的 stage leak 检查结果

3. Half Line A
   - 明确 old row-level half 口径 deprecated
   - 报告 merged half disease-level 结果
   - 至少写 `hit@20`, `precision@20`, `recall@20`, `ndcg@20`, `mrr`
   - 不写成 verified benchmark，只写成 clean-data sanity check

4. Half Line B
   - density / ECDF
   - `p33 / p67`
   - `low / mid / high` 分桶结果
   - `label_core_rerank` 相比 `label_idf_only` 的 bucket delta
   - 标注 row-level confidence bucket 结果仅保留为历史分析，不能作为主结论

5. Phase II 最终结论
   - verified 口径已经饱和且稳定
   - half disease-level 暴露 clean-data coverage gap
   - half confidence 不能直接作为 query confidence 或 ranker weight
   - Phase III 先修 coverage gap，再考虑 ranker

## Acceptance Criteria

### Must Pass

1. `label_core_rerank` 在 verified 上优于 `label_idf_only`
2. `label_core_rerank` 的：
   - `stage_hit_bm25 == 0.0`
   - `stage_hit_dense == 0.0`
   - `stage_hit_prior == 0.0`
3. half1 / half2 都能直接跑通，不依赖 `symptom_text`
4. Line B 能输出全局 `p33 / p67` 和 bucket metrics
5. merged half disease-level 能跑通，并输出 `phase2_half_all_disease`
6. `EXP_DRUG_RECALL_PHASE2.md` 明确区分：
   - verified benchmark
   - deprecated row-level half
   - corrected disease-level half sanity check
   - confidence stratification

### Allowed With Explanation

1. 如果 `label_core_rerank` 比 `candidate_union_no_prior_no_bm25` 略低，但差异不超过：
   - `hit@20` 下降 `0.01`
   - `recall@20` 下降 `0.01`
   则可以接受，解释为“换来更干净、无 dense leakage 的主线”

2. 如果 Line B 绘图失败但分桶统计成功，可以继续收尾，但文档里必须写清楚

### Failure Conditions

以下任一成立则 Phase II 不算完成：

- `label_core_rerank` 仍有 BM25/dense/prior leakage
- half adapter 把原始 half confidence 当成 query confidence 注入主线
- 文档仍把 row-level half 指标写成主结论
- 文档把 half 结果写成正式 pipeline benchmark
- verified 上 `label_core_rerank` 没有超过 `label_idf_only`

## Deliverables

### Code

- `app/embedded_module/experimental_recall_pipeline.py`
- `app/evaluation/run_exp_drug_recall.py`
- `app/evaluation/half_data_adapter.py`
- `app/evaluation/analyze_half_confidence.py`

### Artifacts

- `artifacts/exp_drug_recall/phase2_verified*`
- `artifacts/exp_drug_recall/phase2_verified_1000`
- `artifacts/exp_drug_recall/phase2_half_all_disease`
- `artifacts/exp_drug_recall/phase2_half1` and `phase2_half2` only as deprecated historical row-level traces
- `artifacts/exp_drug_recall/phase2_half_confidence`

### Documentation

- `EXP_DRUG_RECALL_PHASE2.md`

## State After Phase II

Phase II 完成后，项目状态应当是：

- 有一个干净主线：`label_core_rerank`
- 有 old verified 和 new verified 稳定性结果
- 有 corrected half disease-level clean-data sanity check
- 有 half confidence 分桶分析，但仅作为历史 row-level 分析保留
- 下一阶段建议：
  - `Phase III-A = repair label coverage gaps exposed by half disease-level`
  - `Phase III-B = disease_symptom half check`
  - `Phase III-C = local_ranker only after coverage gaps are understood`
  - `Phase IV = dense/BM25 optional revisit`
