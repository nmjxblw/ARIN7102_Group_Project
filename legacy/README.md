# Legacy 归档

已废弃的 yaoja123 产出文件，供审计和回溯参考。

---

## 文件清单

### EXP_DRUG_RECALL_ASIDE_PLAN.md

| 属性 | 内容 |
|---|---|
| 原路径 | EXP_DRUG_RECALL_ASIDE_PLAN.md |
| 创建日期 | 2026-04-24 |
| 最后更新 | 2026-04-24 |
| 原作者 | yaoja123 |
| 原用途 | 实验药物召回旁路验证计划，记录 label_adapter、candidate_union 等独立验证任务 |
| 归档原因 | 功能已集成至 experimental_recall_pipeline.py |

### EXP_DRUG_RECALL_PHASE1.md

| 属性 | 内容 |
|---|---|
| 原路径 | EXP_DRUG_RECALL_PHASE1.md |
| 创建日期 | 2026-04-24 |
| 最后更新 | 2026-04-24 |
| 原作者 | yaoja123 |
| 原用途 | 实验药物召回 Phase 1 记录，回答 Baseline→KNN→Hybrid→RF→DDI 的实验结论 |
| 归档原因 | Phase 1 实验记录，功能已迁移至 experimental_recall_pipeline.py |

### EXP_DRUG_RECALL_PHASE2.md

| 属性 | 内容 |
|---|---|
| 原路径 | EXP_DRUG_RECALL_PHASE2.md |
| 创建日期 | 2026-04-25 |
| 最后更新 | 2026-04-25 |
| 原作者 | yaoja123 |
| 原用途 | 实验药物召回 Phase 2 报告，记录 half_data 使用方式和 recall 稳定性验证 |
| 归档原因 | 实验记录，已被 run_exp_drug_recall.py 的 ablation 系统替代 |

### EXP_DRUG_RECALL_PHASE2_PLAN.md

| 属性 | 内容 |
|---|---|
| 原路径 | EXP_DRUG_RECALL_PHASE2_PLAN.md |
| 创建日期 | 2026-04-25 |
| 最后更新 | 2026-04-25 |
| 原作者 | yaoja123 |
| 原用途 | Phase 2 实验计划，含 8 阶段实现 checklist 和 7 种 ablation 模式设计 |
| 归档原因 | 实验计划，功能已实现并归档于 HERMES.md |

### EXP_DRUG_RECALL_PLAN.md

| 属性 | 内容 |
|---|---|
| 原路径 | EXP_DRUG_RECALL_PLAN.md |
| 创建日期 | 2026-04-25 |
| 最后更新 | 2026-04-25 |
| 原作者 | yaoja123 |
| 原用途 | 实验药物召回完整计划，含实现 checklist 和评估 ablations |
| 归档原因 | 功能已实现并归档于 HERMES.md |

### EXP_DRUG_RECALL_PLAN_图解版.md

| 属性 | 内容 |
|---|---|
| 原路径 | EXP_DRUG_RECALL_PLAN_图解版.md |
| 创建日期 | 2026-04-24 |
| 最后更新 | 2026-04-24 |
| 原作者 | yaoja123 |
| 原用途 | EXP_DRUG_RECALL_PLAN.md 的中文图解版 |
| 归档原因 | 同上，架构图已更新至 HERMES.md |

### PHASE2_FINAL_RECOMMENDATION_PIPELINE_PLAN.md

| 属性 | 内容 |
|---|---|
| 原路径 | PHASE2_FINAL_RECOMMENDATION_PIPELINE_PLAN.md |
| 创建日期 | 2026-04-25 |
| 最后更新 | 2026-04-25 |
| 原作者 | yaoja123 |
| 原用途 | Phase 2 Final Recommendation Pipeline 计划，含 BERT→Phase2→half_pool 两阶段设计 |
| 归档原因 | 功能已实现于 phase2_final_recommender_exp.py，计划归档于 HERMES.md |

### PHASE2_WHOLE_PIPELINE.md

| 属性 | 内容 |
|---|---|
| 原路径 | PHASE2_WHOLE_PIPELINE.md |
| 创建日期 | 2026-04-25 |
| 最后更新 | 2026-04-25 |
| 原作者 | yaoja123 |
| 原用途 | Phase 2 完整 pipeline 文档，含数据流、各阶段输出格式 |
| 归档原因 | 同上，已被 HERMES.md 替代 |

### end2end_test_plan.md

| 属性 | 内容 |
|---|---|
| 原路径 | end2end_test_plan.md |
| 创建日期 | 2026-04-25 |
| 最后更新 | 2026-04-25 |
| 原作者 | yaoja123 |
| 原用途 | 端到端测试计划文档 |
| 归档原因 | 测试计划文件，当前项目测试流程已由 HERMES.md 记录 |

### modify_notebook.py

| 属性 | 内容 |
|---|---|
| 原路径 | modify_notebook.py |
| 创建日期 | 2026-04-21 |
| 最后更新 | 2026-04-21 |
| 原作者 | yaoja123 |
| 原用途 | Jupyter notebook 修改工具脚本 |
| 归档原因 | 工具脚本，当前工作流未使用 |

### app/评估使用_mac.md

| 属性 | 内容 |
|---|---|
| 原路径 | app/评估使用_mac.md |
| 创建日期 | 2026-04-21 |
| 最后更新 | 2026-04-21 |
| 原作者 | yaoja123 |
| 原用途 | macOS 版评估使用操作指南 |
| 归档原因 | 操作指南文档，当前已由 HERMES.md 替代 |

### app/embedded_module/drug_embedding_engine.py

| 属性 | 内容 |
|---|---|
| 原路径 | app/embedded_module/drug_embedding_engine.py |
| 创建日期 | 2026-04-24 |
| 最后更新 | 2026-04-24 |
| 原作者 | yaoja123 |
| 原用途 | PubMedBERT 药物向量编码引擎，用于 dense 召回 |
| 归档原因 | dense 召回未在主链路使用，仅被消融实验和诊断脚本引用 |

### 3_28_Sat_下午EDA.md

| 属性 | 内容 |
|---|---|
| 原路径 | app/interactions/3_28_Sat_下午EDA.md |
| 创建日期 | 2026-04-21 |
| 最后更新 | 2026-04-21 |
| 原作者 | yaoja123 |
| 原用途 | 2026-03-28 周六下午 EDA 分析记录 |
| 归档原因 | 实验分析文档，与当前主链路无关 |

### app_interactions_evaluation_delivery.ipynb_解释.md

| 属性 | 内容 |
|---|---|
| 原路径 | app/interactions/app_interactions_evaluation_delivery.ipynb_解释.md |
| 创建日期 | 2026-04-21 |
| 最后更新 | 2026-04-21 |
| 原作者 | yaoja123 |
| 原用途 | evaluation_delivery.ipynb 的解释文档 |
| 归档原因 | 实验 notebook 配套文档，与当前主链路无关 |

### comment_eda_and_coverage.ipynb

| 属性 | 内容 |
|---|---|
| 原路径 | app/interactions/comment_eda_and_coverage.ipynb |
| 创建日期 | 2026-04-21 |
| 最后更新 | 2026-04-21 |
| 原作者 | yaoja123 |
| 原用途 | 评论数据 EDA 与覆盖率分析 notebook |
| 归档原因 | 实验数据分析 notebook，与当前主链路无关 |

### drug_recommendation_experiment.ipynb

| 属性 | 内容 |
|---|---|
| 原路径 | app/interactions/drug_recommendation_experiment.ipynb |
| 创建日期 | 2026-04-21 |
| 最后更新 | 2026-04-21 |
| 原作者 | yaoja123 |
| 原用途 | 药物推荐实验记录 notebook |
| 归档原因 | 实验 notebook，与当前主链路无关 |

### evaluation_delivery.ipynb

| 属性 | 内容 |
|---|---|
| 原路径 | app/interactions/evaluation_delivery.ipynb |
| 创建日期 | 2026-04-21 |
| 最后更新 | 2026-04-21 |
| 原作者 | yaoja123 |
| 原用途 | 评估结果交付 notebook |
| 归档原因 | 实验 notebook，与当前主链路无关 |

### module_explanation_zh.md

| 属性 | 内容 |
|---|---|
| 原路径 | app/interactions/module_explanation_zh.md |
| 创建日期 | 2026-03-19 |
| 最后更新 | 2026-03-19 |
| 原作者 | yaoja123 |
| 原用途 | 早期模块中文解释文档 |
| 归档原因 | 实验性文档，与当前主链路无关 |

### naive_bayes_baseline_comparison.ipynb

| 属性 | 内容 |
|---|---|
| 原路径 | app/interactions/naive_bayes_baseline_comparison.ipynb |
| 创建日期 | 2026-04-21 |
| 最后更新 | 2026-04-21 |
| 原作者 | yaoja123 |
| 原用途 | 朴素贝叶斯 baseline 对比实验 notebook |
| 归档原因 | 实验 notebook，与当前主链路无关 |

### simple_explnation.md

| 属性 | 内容 |
|---|---|
| 原路径 | app/interactions/simple_explnation.md |
| 创建日期 | 2026-03-19 |
| 最后更新 | 2026-03-19 |
| 原作者 | yaoja123 |
| 原用途 | 早期模块简单解释文档 |
| 归档原因 | 实验性文档，与当前主链路无关 |


