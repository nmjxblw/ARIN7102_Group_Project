# Experimental Drug Recall Pipeline Plan

## Goal

Build a stronger local drug recall pipeline for the current ARIN7102 drug
recommendation project. The pipeline should improve exact drug-name retrieval on
the verified evaluation set while still fitting the project guideline: data
selection, preprocessing, statistics/text mining, AI modeling, result display,
saved outputs, exception handling, and clear modular code.

Primary target:

- Evaluation set: `data/eval_dataset_verified.json`
- Main metric: `hit@20`
- Secondary metrics: `recall@20`, `mrr`, `precision@5`, `ndcg@20`
- Baseline to beat: current label recall, not broken semantic recall
  - `hit@20 = 0.6859`
  - `recall@20 = 0.5534`
  - `mrr = 0.2983`

## Data Sources

Main recall table:

- `match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv`
- Key fields:
  - `drug_name`
  - `generic_name`
  - `drug_classes`
  - `brand_names`
  - `related_drugs`
  - `avg_rating`
  - `total_reviews`
  - `matched_disease_keys`
  - `matched_symptoms`
  - `symptom_severity`
  - `disease_description`
  - `medical_condition_description`
  - `semantic_text`
  - `semantic_text_mcd`
  - `semantic_text_dd`

Auxiliary data:

- `app/dataset_module/drugs_training_dataset/eval_dataset_llm_v2.json`
  - BERT-generated query data.
  - Use for weak training and stress testing.
- `app/dataset_module/drugs_training_dataset/drug_data_half_1.json`
  - Weak structured drug-label mapping data, 1465 rows.
  - Use for Phase 2 label-core coverage and regression testing.
  - Do not treat as a verified gold benchmark.
- `app/dataset_module/drugs_training_dataset/drug_data_half_2.json`
  - Weak structured drug-label mapping data, 1466 rows.
  - Use together with `drug_data_half_1.json` for large-scale smoke tests.
  - Do not use for final reported metrics.
- `data/eval_dataset_verified.json`
  - Gold evaluation only.
  - Do not train on this file.
- `app/dataset_module/drugs_training_dataset/drug_disease_mapping.json`
  - Optional consistency check and expansion source.
  - It covers fewer drugs than the structured CSV, so it is not the main table.
- `drug_comprehensive_embeddings.npy` and
  `interaction/drug_comprehensive_embeddings.npy`
  - Must be validated by shape and manifest before use.
  - Canonical target should be a 2D `(n_drugs, dim)` embedding aligned with the
    structured CSV.

## Input Contract

The upstream BERT layer provides:

```json
{
  "sentence": "I have big, painful bumps on my skin that leave scars.",
  "diseases": [{"name": "acne", "confidence": 1.0}],
  "symptoms": [
    {"name": "nodal_skin_eruptions", "confidence": 1.0},
    {"name": "scurring", "confidence": 1.0}
  ],
  "need_first_aid": 0
}
```

The new adapter must accept all currently observed label formats:

- `{"name": "acne", "confidence": 1.0}`
- `{"label": "acne", "confidence": 1.0}`
- `{"acne": 1.0}`

Normalization rules:

- Convert symptom labels to one canonical form for matching.
- Support both underscore and space forms.
- Drop `others` when another disease label exists.
- Keep `others` only when it is the only disease label.
- Clip confidence into `[0, 1]`.

## Pipeline Design

### Stage 1: Build Local Recall Index

Build an in-memory `DrugRecallIndex` from the structured CSV:

- `disease -> drug row ids`
- `symptom -> drug row ids`
- `generic_name -> drug row ids`
- `drug_class -> drug row ids`
- `related_drug -> drug row ids`
- BM25 text corpus from drug name, generic name, class, disease/symptom labels,
  disease description, medical condition description, and semantic text.
- Disease IDF and symptom IDF from the full drug table.
- Drug quality priors from `avg_rating` and `log1p(total_reviews)`.

The index should validate:

- drug table row count
- embedding row count
- embedding dimension
- asset path used
- whether 3D legacy embedding is being projected to View 1

### Stage 2: Mechanical High-Recall Candidate Generation

The goal is to ensure likely correct drugs enter the candidate pool.

Candidate sources:

1. Disease candidates
   - Select rows where `matched_disease_keys` overlaps BERT diseases.
   - Strongest base signal.

2. Strict disease-symptom candidates
   - Select rows matching disease plus at least one high-confidence symptom.
   - Rank above loose disease-only matches.

3. Symptom fallback candidates
   - Select rows matching multiple high-confidence symptoms even if disease does
     not match.
   - Protects against BERT disease errors.

4. BM25 lexical candidates
   - Query text is `sentence + disease labels + symptom labels`.
   - Search the local text corpus built from the structured CSV fields.
   - This is local text mining, not external LLM.

5. Dense embedding candidates
   - Query text is also enriched with BERT labels.
   - Use only if embedding asset passes validation.
   - Dense recall is a supplement, not the main decision maker.

6. Prior expansion candidates
   - From already recalled drugs, add same generic name, related drugs, same
     disease high-review drugs, and high-rating drugs in the same disease group.

Candidate pool policy:

- Union candidates from all stages.
- Preserve stage provenance for each candidate.
- Cap final pool at 1000 before ranking.
- Never allow dense retrieval alone to remove strong label candidates.

### Stage 3: Mechanical Recall Scoring

For each candidate, compute explainable features:

- disease confidence overlap
- symptom confidence overlap
- symptom IDF sum
- symptom coverage ratio
- disease specificity
- BM25 score
- dense cosine score
- same generic/class/related-drug flags
- rating prior
- review prior
- `others` penalty
- stage hit flags

Fallback deterministic score:

```text
score =
  0.40 * label_idf_score
+ 0.20 * symptom_coverage
+ 0.15 * bm25_score
+ 0.15 * dense_score
+ 0.10 * quality_prior
- others_penalty
```

If dense retrieval is unavailable or harmful in ablation, set dense weight to
`0.05` and redistribute the remaining weight to label and BM25 features.

### Stage 4: Local Learning-to-Rank

Use a local, reproducible model rather than an online LLM.

Recommended model:

- `sklearn.ensemble.HistGradientBoostingClassifier`

Training data:

- Use `eval_dataset_llm_v2.json` as weak training queries.
- Generate positives from matching disease/drug associations in the structured
  table and `drug_disease_mapping.json`.
- Generate negatives from same-query candidate pools not matching the target
  disease/symptom pattern.
- Do not train on `data/eval_dataset_verified.json`.

Prediction:

- Rank candidates by model probability.
- If model artifact is missing, fall back to deterministic score.

Why this is useful:

- The first stages maximize recall.
- The ranker learns which combinations of disease, symptoms, BM25, dense score,
  and drug quality actually predict exact relevant drugs.
- Features remain explainable for the report and demo.

### Stage 5: Output and Trace

Recommendation output should include:

- `drug_name`
- `final_score`
- `ranker_score`
- `label_idf_score`
- `bm25_score`
- `dense_score`
- `quality_prior`
- `stage_hits`
- `matched_diseases`
- `matched_symptoms`
- `avg_rating`
- `total_reviews`
- short evidence string

Trace output should include:

- input labels after normalization
- candidate count from each stage
- final union size
- embedding asset path and shape
- whether fallback mode was used
- top missed relevant drugs during evaluation

## Evaluation Plan

Run ablations on `data/eval_dataset_verified.json`:

1. `label_idf_only`
2. `bm25_only`
3. `dense_only`
4. `label_idf + bm25`
5. `label_idf + bm25 + dense`
6. `candidate_union + deterministic_score`
7. `candidate_union + local_ranker`

Primary evaluation remains `data/eval_dataset_verified.json`. The two
`drug_data_half_*.json` files are secondary weak structured evaluation data for
Phase 2 regression only. They should be used to check label adapter coverage,
label-core recall, large-scale pipeline stability, and whether the original
`drug_name` appears in top-k results. They must not replace verified metrics in
the report.

For half-data evaluation, add an adapter that accepts the source schema:

```json
{
  "drug_name": "aczone",
  "diseases": [{"acne": 0.95}],
  "symptoms": [{"skin_rash": 0.85}]
}
```

The adapter should convert it to evaluation-style rows with:

- `query_id`: `half1_000001` or `half2_000001`
- `diseases`: `{"name": "acne", "confidence": 0.95}`
- `symptoms`: `{"name": "skin_rash", "confidence": 0.85}`
- `relevant_drugs`: `["aczone"]`
- `symptom_text`: optional; pure label-core tests should allow it to be absent
  or empty

Run half-data checks first with `label_idf_only`, then with the Phase 2
`label_core_rerank` mode once it exists. If BM25 or dense modes are evaluated on
half data, the run must either construct synthetic query text from labels or
explicitly document that the test is label-text based rather than natural
language based.

Required report metrics:

- `precision@5`, `precision@10`, `precision@20`
- `recall@5`, `recall@10`, `recall@20`
- `hit@5`, `hit@10`, `hit@20`
- `mrr`
- `ndcg@5`, `ndcg@10`, `ndcg@20`
- per-stage candidate hit rate before final ranking

Acceptance criteria:

- Final `hit@20` must exceed `0.6859`.
- Final `recall@20` must exceed `0.5534`.
- Final `mrr` should be at least `0.2983`.
- If the ranker cannot beat deterministic scoring, ship deterministic scoring
  and document the ranker as an experiment.

## Implementation Modules

Suggested module layout:

- `app/embedded_module/label_adapter.py`
  - Normalize BERT disease/symptom inputs.
- `app/embedded_module/drug_recall_index.py`
  - Build inverted indexes, IDF stats, BM25 corpus, and asset manifest.
- `app/embedded_module/experimental_recall_pipeline.py`
  - Candidate generation, feature extraction, scoring, ranking, trace output.
- `app/embedded_module/drug_ranker.py`
  - Train/load/predict local ranker.
- `app/evaluation/run_exp_drug_recall.py`
  - Run full ablation and save metrics.

Artifacts:

- `artifacts/exp_drug_recall/metrics.json`
- `artifacts/exp_drug_recall/per_query_results.csv`
- `artifacts/exp_drug_recall/stage_trace.jsonl`
- `artifacts/exp_drug_recall/ranker.joblib`
- `artifacts/exp_drug_recall/asset_manifest.json`

## Risks and Controls

- Risk: exact drug-name labels are incomplete.
  - Control: still optimize exact metrics, but also report disease/symptom
    relevance sanity metrics.
- Risk: dense embedding asset mismatch.
  - Control: validate shape, path, model name, and row count before use.
- Risk: model overfits weak training labels.
  - Control: verified set is evaluation only, and deterministic fallback remains
    available.
- Risk: disease candidates are too broad.
  - Control: IDF, symptom coverage, BM25, and ranker features handle ordering.
- Risk: external LLM hurts reproducibility.
  - Control: no online LLM in the default pipeline.

## First Experiment Checklist

1. Build label adapter and verify it parses current BERT and historical formats.
2. Build recall index from the structured CSV.
3. Implement label-IDF candidate generation and evaluate `label_idf_only`.
4. Add BM25 and evaluate union recall.
5. Add embedding validation and dense recall as optional supplement.
6. Add deterministic scorer and compare against current label baseline.
7. Train local ranker from weak data and evaluate against deterministic scorer.
8. Save metrics and traces for report and presentation.

## Phase 2 TODO

1. Add a weak half-data evaluation adapter for `drug_data_half_1.json` and
   `drug_data_half_2.json`.
2. Support half-data rows with `drug_name`, `diseases`, and `symptoms`; do not
   require `symptom_text` for pure label-core tests.
3. Convert each source `drug_name` into `relevant_drugs: [drug_name]` and assign
   deterministic query ids such as `half1_000001` and `half2_000001`.
4. Use half-data evaluation only for weak structured regression, label adapter
   coverage, and top-k containment checks.
5. Keep final Phase 2 metrics on `data/eval_dataset_verified.json`.
6. Limit the first half-data runs to `label_idf_only`; later rerun with
   `label_core_rerank` after that mode is implemented.
