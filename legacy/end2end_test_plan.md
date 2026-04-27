套餐



Main 端到端 Verified-Set 评测计划
Summary
新增一个独立的 main 端到端评测脚本，用 data/eval_dataset_verified.json 作为测试集，只走这条口径：

symptom_text -> BERT 预测 diseases/symptoms -> 主推荐链路 -> recommended drugs -> 与 relevant_drugs 算指标

不改现有 run_evaluation.py / run_evaluation_stages.py 的语义，避免把“gold-label 驱动评测”和“纯文本端到端评测”混在一起。

Key Changes
新增独立 E2E evaluator
新增 app/evaluation/run_main_e2e_verified.py，职责固定为 verified-set 的纯端到端评测。

脚本流程固定为：

读取 data/eval_dataset_verified.json
service.ensure_ready()
对每条 query 调 service.predict_labels(query["symptom_text"])
将预测出的 diseases/symptoms 再传给 service.recommend(...)，且 use_bert_prediction=False
取返回结果中的 drug_name 列表，与 relevant_drugs / relevance_scores 用现有 evaluation.metrics 计算指标
这里显式拆成两步，而不是直接 use_bert_prediction=True，原因是要把 BERT 预测结果落到 per-query artifact 里，方便分析错误来源，同时保持与主链路一致的推荐输入。

CLI 和输出约定
脚本提供这些 CLI：

--eval-dataset，默认 data/eval_dataset_verified.json
--artifact-dir，默认 artifacts/main_e2e_verified
--k-values，默认 5 10 20
--limit，可选，仅跑前 N 条做 smoke test
--top-k，默认 max(k_values)
推荐阶段参数默认直接沿用 pipeline_config.cfg 当前值，不在脚本里重新硬编码；最终把解析后的有效参数一并写入 metrics artifact，保证结果可复现。

输出物固定为：

metrics.json：聚合指标、query 数、k 值、运行参数
per_query_results.csv：每条 query 的预测标签、推荐列表、gold drugs、单条指标
predictions.jsonl：每条 query 的原始 BERT 预测和最终推荐，便于后续排错
数据和比较口径
verified-set 直接复用现有字段：

输入只使用 symptom_text
gold 只使用 relevant_drugs / relevance_scores
不直接使用 verified-set 里的 diseases/symptoms 参与推荐
evaluation.metrics 保持原样复用，不增加名字归一化或额外容错，确保和现有 repo 指标口径一致。

兼容与边界
不改：

app/__main__.py
run_evaluation.py
run_evaluation_stages.py
这个脚本是“新增端到端评测视角”，不是替换现有 gold-label evaluator。

Test Plan
Smoke test
用 --limit 3 跑通脚本，确认会产出三类 artifact，且每条记录都包含：
query_id、symptom_text、predicted_diseases、predicted_symptoms、recommended、relevant

Full verified-set run
对 191 条 verified queries 完整跑一次，确认：

metrics.json 生成成功
per_query_results.csv 行数等于成功评测条数
predictions.jsonl 条数一致
聚合指标至少包含 precision@k、recall@k、hit@k、mrr，以及有 relevance_scores 时的 ndcg@k
Failure-path checks
缺失 trained_bert 权重、缺失 embedding、缺失表文件时，脚本应直接报清晰错误，不静默跳过初始化失败。

Non-regression
运行前后不改变现有 evaluator 的调用方式和指标口径；run_evaluation.py 与 run_evaluation_stages.py 无行为变化。

Public Interfaces
新增一个 CLI 入口：

python -m app.evaluation.run_main_e2e_verified
不新增服务层公共 API；复用现有：

get_recommendation_service()
service.predict_labels(...)
service.recommend(...)
evaluation.metrics.evaluate_batch(...)
Assumptions
app/deployment_module/trained_bert/model.safetensors 已存在且可加载
运行环境沿用当前可用的 Conda 环境与 .env 约束，不把密钥或 .env 内容写入 repo
verified 集的 relevant_drugs 继续按现有“严格药名匹配”口径计分
这次只做“纯 E2E only”，不在同一脚本里顺手输出 gold-label baseline 对照