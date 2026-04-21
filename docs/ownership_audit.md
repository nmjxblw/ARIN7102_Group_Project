# Ownership Audit

## Audit Basis

- Identity treated as yours: `yaoja123 <yaoja123@gmail.com>`
- Ownership rule: file primary author by commit count from `git log --follow --format=%an/%ae`
- Current uncommitted changes default to your current responsibility
- This audit describes repository evidence, not real-world authorship certainty

## Summary

- Total files in audit: `170`
- Tracked files: `139`
- Untracked local files: `31`
- Historically yours: `5`
- Historically others: `125`
- New tracked local with no history: `0`
- Currently modified by you: `9`
- Local untracked treated as yours: `31`

## Yours

- `app/embedded_module/drug_embedding_engine.py`
- `app/embedded_module/drug_knn_retriever.py`
- `app/embedded_module/recommendation_pipeline.py`
- `app/interactions/module_explanation_zh.md`
- `app/interactions/simple_explnation.md`

## Others

- `.claude/settings.local.json` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `.env.example` (primary author: `YangZhenyu23 <905929921@qq.com>`)
- `.vscode/launch.json` (primary author: `nmjxblw <915253492@qq.com>`)
- `.vscode/settings.json` (primary author: `z c <915253492@qq.com>`)
- `README.md` (primary author: `nmjxblw <915253492@qq.com>`)
- `References/ARIN 7102 Project Instruction and Guideline - 2026 Spring.docx` (primary author: `z c <915253492@qq.com>`)
- `app/.env.example` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/__init__.py` (primary author: `z c <915253492@qq.com>`)
- `app/__main__.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/database_module/__init__.py` (primary author: `z c <915253492@qq.com>`)
- `app/database_module/database_main.py` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/__init__.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/bert_training_dataset/disease_labels.json` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/bert_training_dataset/extracted_unlabeled_diseases.json` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/bert_training_dataset/generated_medical_dataset.json` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/bert_training_dataset/symptom_labels.json` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/csv_to_json.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/data_process.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/disease-symptom-description-dataset/Symptom-severity.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/disease-symptom-description-dataset/Symptom-severity_cleaned.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/disease-symptom-description-dataset/dataset.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/disease-symptom-description-dataset/dataset_cleaned.csv` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/disease-symptom-description-dataset/disease_symptoms_dict.json` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/disease-symptom-description-dataset/symptom_Description.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/disease-symptom-description-dataset/symptom_Description_cleaned.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/disease-symptom-description-dataset/symptom_precaution.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/disease-symptom-description-dataset/symptom_precaution_cleaned.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/disease-symptoms-and-patient-profile-dataset/Disease_symptom_and_patient_profile_dataset.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/disease-symptoms-and-patient-profile-dataset/Disease_symptom_and_patient_profile_dataset_cleaned.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/disease_data_process.py` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/drug-prescription-to-disease-dataset/final.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/drug-prescription-to-disease-dataset/final_cleaned.csv` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/drug_data_process.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/drugs-related-to-common-treatments/drugs_for_common_treatments.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/drugs-related-to-common-treatments/drugs_for_common_treatments_cleaned.csv` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/drugs-side-effects-and-medical-condition/drugs_side_effects_drugs_com.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/drugs-side-effects-and-medical-condition/drugs_side_effects_drugs_com_cleaned.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/drugs_training_dataset/.gitignore` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/drugs_training_dataset/drug_disease_mapping.json` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/drugs_training_dataset/enhanced_drug_table_v1.json` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/drugs_training_dataset/enhanced_drug_table_v1_structured.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/drugs_training_dataset/eval_dataset_llm.json` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/drugs_training_dataset/eval_dataset_llm_v2.json` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/drugs_training_dataset/temp_tool.py` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/extract_unlabeled_diseases.py` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/first-aid-QA/firstaidqa_v1.json` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/first-aid-QA/labeled_firstaidqa.json` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/generate_eval_datasets_via_bert.py` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/generated-miss-data/generated_medical_dataset_missing_keys.json` (primary author: `PeaHaWf <1161428420@qq.com>`)
- `app/dataset_module/json_to_dataframe.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/kaggle_dataset_download_urls.json` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/kaggle_download.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/kuc-hackathon-winter-2018/drugsComTest_raw.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/kuc-hackathon-winter-2018/drugsComTest_raw_cleaned.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/kuc-hackathon-winter-2018/drugsComTrain_raw.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/kuc-hackathon-winter-2018/drugsComTrain_raw_cleaned.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/labels_extractor.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/raw_data_merge.py` (primary author: `z c <915253492@qq.com>`)
- `app/dataset_module/symptom2disease/Symptom2Disease.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/dataset_module/symptom2disease/Symptom2Disease_cleaned.csv` (primary author: `z c <915253492@qq.com>`)
- `app/deployment_module/__init__.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/deployment_module/bert_main.py` (primary author: `PeaHaWf <1161428420@qq.com>`)
- `app/deployment_module/clinicalbert_local/ClinicalBERT` (primary author: `PeaHaWf <1161428420@qq.com>`)
- `app/deployment_module/clinicalbert_local/README.md` (primary author: `z c <915253492@qq.com>`)
- `app/deployment_module/clinicalbert_local/config.json` (primary author: `z c <915253492@qq.com>`)
- `app/deployment_module/clinicalbert_local/medicalaiClinicalBERT_git_clone.png` (primary author: `z c <915253492@qq.com>`)
- `app/deployment_module/clinicalbert_local/special_tokens_map.json` (primary author: `z c <915253492@qq.com>`)
- `app/deployment_module/clinicalbert_local/tokenizer_config.json` (primary author: `z c <915253492@qq.com>`)
- `app/deployment_module/clinicalbert_local/vocab.txt` (primary author: `z c <915253492@qq.com>`)
- `app/deployment_module/trained_bert/config.json` (primary author: `PeaHaWf <1161428420@qq.com>`)
- `app/deployment_module/trained_bert/label_encoders.pkl` (primary author: `z c <915253492@qq.com>`)
- `app/deployment_module/trained_bert/tokenizer.json` (primary author: `z c <915253492@qq.com>`)
- `app/deployment_module/trained_bert/tokenizer_config.json` (primary author: `z c <915253492@qq.com>`)
- `app/embedded_module/cross_encoder_reranker.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/embedded_module/dual_recall_pipeline.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/evaluation/generate_eval_dataset.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/evaluation/metrics.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/evaluation/run_evaluation.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/fastapi_module/__init__.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/fastapi_module/app_main.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/fastapi_module/offline_assets_builder.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/fastapi_module/router.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/fastapi_module/schemas.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/fastapi_module/service.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/launcher_module/__init__.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/launcher_module/launcher_main.py` (primary author: `z c <915253492@qq.com>`)
- `app/launcher_module/main_thread_task_manager.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/remote_llm_module/__init__.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/remote_llm_module/deepseek_manager.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/remote_llm_module/query_balance.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/singleton_module/__init__.py` (primary author: `z c <915253492@qq.com>`)
- `app/singleton_module/singleton_meta.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/static_module/__init__.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/static_module/classes.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/static_module/enums.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/static_module/parameters.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/ui_module/__init__.py` (primary author: `z c <915253492@qq.com>`)
- `app/ui_module/ui_main.py` (primary author: `z c <915253492@qq.com>`)
- `app/utility_module/__init__.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `app/utility_module/log_utility.py` (primary author: `nmjxblw <915253492@qq.com>`)
- `disease_keys.json` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `docs/evaluation_dataset_format.md` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `docs/llm_eval_dataset_config.md` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `docs/pipeline_design.md` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `drug_comprehensive_embeddings.npy` (primary author: `nmjxblw <915253492@qq.com>`)
- `match_data_preprocessing/analysis/drug_ingredient_pairs_数据描述.md` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/analysis/embedding_方案说明.md` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/analysis/增强数据的使用.md` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/analysis/数据使用说明.md` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/analysis/数据清洗说明.md` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/analysis/链路实现函数级说明.md` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/analysis/链路输入输出说明.md` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/data/db_drug_interactions.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `match_data_preprocessing/data/drug_ingredient_pairs.csv` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/data/enhanced_drug_table.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `match_data_preprocessing/data/enhanced_drug_table_v1.csv` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/data/enhanced_drug_table_v1.json` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv` (primary author: `nmjxblw <915253492@qq.com>`)
- `match_data_preprocessing/disease_keys.json` (primary author: `z c <915253492@qq.com>`)
- `match_data_preprocessing/scripts/analyze_and_build_dataset.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/scripts/backfill_others_symptoms.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/scripts/build_enhanced_drug_table.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `match_data_preprocessing/scripts/generate_others_symptoms.py` (primary author: `yinan <zhangyinan197@microconnect.com>`)
- `requirements.txt` (primary author: `z c <915253492@qq.com>`)
- `评估使用.md` (primary author: `Sparks <smart030518@126.com>`)

## Yours Currently Modified

- `.gitignore` (historical primary owner: `others`, primary author: `nmjxblw <915253492@qq.com>`)
- `app/embedded_module/__init__.py` (historical primary owner: `others`, primary author: `yinan <zhangyinan197@microconnect.com>`)
- `app/embedded_module/evaluation.py` (historical primary owner: `new_tracked_local`, primary author: `new_tracked_local`)
- `app/evaluation/__init__.py` (historical primary owner: `others`, primary author: `nmjxblw <915253492@qq.com>`)
- `app/evaluation/generate_eval_dataset_llm.py` (historical primary owner: `others`, primary author: `nmjxblw <915253492@qq.com>`)
- `app/evaluation/llm_client.py` (historical primary owner: `others`, primary author: `nmjxblw <915253492@qq.com>`)
- `app/interactions/comment_eda_and_coverage.ipynb` (historical primary owner: `new_tracked_local`, primary author: `new_tracked_local`)
- `app/interactions/drug_recommendation_experiment.ipynb` (historical primary owner: `yours`, primary author: `yaoja123 <yaoja123@gmail.com>`)
- `app/interactions/evaluation_delivery.ipynb` (historical primary owner: `new_tracked_local`, primary author: `new_tracked_local`)

## Yours Local Untracked

- `app/interactions/3_28_Sat_下午EDA.md`
- `app/interactions/app_interactions_evaluation_delivery.ipynb_解释.md`
- `app/interactions/data_collection_eda copy.ipynb`
- `app/interactions/data_collection_eda.ipynb`
- `app/interactions/data_collection_eda_notes.md`
- `app/interactions/data_collection_eda_解释.md`
- `app/interactions/naive_bayes_baseline_comparison.ipynb`
- `app/评估使用.md`
- `app/评估使用_mac.md`
- `data/eval_dataset_llm.json`
- `data/eval_results.json`
- `data/ownership_audit.csv`
- `data/sync_backup_20260414_1/branch.txt`
- `data/sync_backup_20260414_1/diff_staged.patch`
- `data/sync_backup_20260414_1/diff_unstaged.patch`
- `data/sync_backup_20260414_1/status_before_sync.txt`
- `data/sync_backup_20260414_1/untracked_files.txt`
- `docs/ownership_audit.md`
- `docs/架构.md`
- `eda/ds1_prescription_to_disease_eda.ipynb`
- `eda/ds2_drug_side_effects_eda.ipynb`
- `eda/ds3_common_treatment_drugs_eda.ipynb`
- `eda/ds4_drug_reviews_eda.ipynb`
- `eda/ds5_disease_symptom_mapping_eda.ipynb`
- `eda/ds6_symptom_description_eda.ipynb`
- `eda/ds7_patient_profile_eda.ipynb`
- `eda/ds8_symptom2disease_nl_eda.ipynb`
- `eda/generate_eda_notebooks.py`
- `eda/hf_firstaidqa_eda.ipynb`
- `modify_notebook.py`
- `requirements_mac.txt`

## New Tracked Local

- None

## Verification Sample

- `app/deployment_module/bert_main.py`: primary author = `PeaHaWf` <`1161428420@qq.com`>, bucket = `others`
- `app/embedded_module/dual_recall_pipeline.py`: primary author = `yinan` <`zhangyinan197@microconnect.com`>, bucket = `others`
- `app/fastapi_module/service.py`: primary author = `yinan` <`zhangyinan197@microconnect.com`>, bucket = `others`
- `match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv`: primary author = `nmjxblw` <`915253492@qq.com`>, bucket = `others`
- `app/deployment_module/trained_bert/label_encoders.pkl`: primary author = `z c` <`915253492@qq.com`>, bucket = `others`

## Limitations

- Primary-author ownership does not equal line-level authorship
- Mixed-contribution files are still forced into a single primary owner
- Shared email identities can collapse multiple people into one bucket
- Uncommitted local responsibility is a working rule, not proof of authorship
