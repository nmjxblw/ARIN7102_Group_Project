# Phase II Final Recommendation Pipeline Plan

## Goal

Build a standalone Phase II final drug recommendation pipeline for submission.
Do not change the FastAPI service in this step.

The final behavior is:

```text
1 disease  -> output Top3 drugs
2 diseases -> output Top6 drugs
3 diseases -> output Top9 drugs
```

Each disease is handled independently:

```text
disease + shared symptoms
-> Phase II label_core_rerank Top20 recall
-> filter Top20 by the same disease's merged half-json drug pool
-> output that disease's Top3
```

There is no second-stage weighted `selection_score`.
There is no cross-disease global reranking.

## Current Repo Entrypoints To Reuse

- Phase II recall:
  - Use `ExperimentalDrugRecallPipeline` from `app/embedded_module/experimental_recall_pipeline.py`.
  - Use `mode="label_core_rerank"`.
  - Use `top_k=20` for internal recall.
  - The existing evaluation script `app/evaluation/run_exp_drug_recall.py` shows how to build:
    - `DrugRecallIndex`
    - `ExperimentalDrugRecallPipeline`
    - table path defaults

- Drug table:
  - Default table:
    `match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv`

- Half data:
  - Use both:
    - `data/drug_data_half_1.json`
    - `data/drug_data_half_2.json`
  - Half data must be merged before grouping by disease.
  - Reuse or extend `app/evaluation/half_data_adapter.py`.
  - Do not evaluate/select half data row by row for this final pipeline.

## Input Format

The input format follows the disease/symptom classifier output.
Symptoms are shared at the whole-query level; they are not nested under each
disease.

Example input:

```json
{
  "diseases": [
    {
      "label": "gastroenteritis",
      "confidence": 0.59
    }
  ],
  "symptoms": [
    {
      "label": "diarrhoea",
      "confidence": 0.34187650582049783
    },
    {
      "label": "mild_fever",
      "confidence": 0.6203438056388872
    }
  ],
  "need_first_aid": 0,
  "sentence": "A bit of fever with diarrhea and feeling sick. Is this common?"
}
```

Batch input should also be supported:

```json
[
  {
    "diseases": [
      {"label": "gastroenteritis", "confidence": 0.59},
      {"label": "common_cold", "confidence": 0.31}
    ],
    "symptoms": [
      {"label": "diarrhoea", "confidence": 0.34},
      {"label": "mild_fever", "confidence": 0.62}
    ],
    "need_first_aid": 0,
    "sentence": "A bit of fever with diarrhea and feeling sick. Is this common?"
  }
]
```

Processing rules:

- `diseases[].label`: each disease gets its own recommendation run.
- `diseases[].confidence`: preserve in output metadata; do not use for ranking.
- `symptoms[].label`: shared symptoms passed to every disease's Phase II recall.
- `symptoms[].confidence`: preserve in output metadata if convenient; do not use for ranking.
- `sentence`: preserve in output metadata.
- `need_first_aid`: ignore in this pipeline.

## Recommendation Logic

For each input query:

1. Parse all shared symptoms once.
2. For each disease in `diseases`:
   - Build a Phase II query with that one disease plus all shared symptoms.
   - Call `ExperimentalDrugRecallPipeline.recommend(...)` with:
     - `mode="label_core_rerank"`
     - `top_k=20`
     - a reasonable existing `pool_size` default from the experimental recall script.
   - Keep the returned Phase II rank order exactly as-is.
3. Build the merged half disease pool:
   - Merge `drug_data_half_1.json` and `drug_data_half_2.json`.
   - Group by normalized disease label.
   - Canonicalize drug names to the enhanced drug table names.
   - For each disease, the pool is the set of all canonical drugs associated
     with that disease in merged half data.
4. Select final Top3 for that disease:
   - First pass: take drugs from the Phase II Top20 that are also in that
     disease's half pool.
   - Keep their original Phase II order.
   - Mark these rows as `selection_source = "half_confirmed"`.
   - Stop when 3 drugs are selected.
5. Fallback if fewer than 3 half-confirmed drugs exist:
   - Continue scanning the same Phase II Top20 in original rank order.
   - Add drugs not already selected.
   - Mark these rows as `selection_source = "phase2_fallback"`.
   - Stop at 3 total drugs.
6. Combine disease-level results for display:
   - Do not rerank across diseases.
   - Concatenate results in the same order as input `diseases`.
   - `global_display_rank` is only a display index, not a model score.

Important:

- Do not add `selection_score`.
- Do not combine all diseases into one global Top20.
- Do not choose only 3 drugs total for a multi-disease query.
- Do not use half confidence to reorder the final Top3.

## Suggested Files

Add a small standalone implementation. Suggested paths:

- `app/embedded_module/phase2_final_recommender.py`
  - Owns the final per-disease recommendation logic.
  - Builds or receives:
    - `DrugRecallIndex`
    - `ExperimentalDrugRecallPipeline`
    - merged half disease pool
  - Exposes a method such as:
    `recommend_query(query: dict, top_k_recall: int = 20, top_k_per_disease: int = 3) -> dict`

- `app/evaluation/run_phase2_final_recommendation.py`
  - CLI wrapper for manual runs and artifact generation.
  - Supports `--input-json` for classifier-style input.
  - Supports simple debug flags:
    - `--diseases gastroenteritis common_cold`
    - `--symptoms diarrhoea mild_fever`
  - Supports `--output-json`.

Keep this standalone. Do not wire it into `app/fastapi_module/service.py` yet.

## Output Format

Return both grouped and flat views.

Grouped view:

```json
{
  "queries": [
    {
      "query_index": 0,
      "sentence": "A bit of fever with diarrhea and feeling sick. Is this common?",
      "shared_symptoms": [
        {"label": "diarrhoea", "confidence": 0.34187650582049783},
        {"label": "mild_fever", "confidence": 0.6203438056388872}
      ],
      "disease_results": [
        {
          "disease": "gastroenteritis",
          "disease_confidence": 0.59,
          "phase2_top20_count": 20,
          "half_confirmed_in_top20": 3,
          "final_top3": [
            {
              "drug_name": "example_drug",
              "disease_rank": 1,
              "phase2_rank": 4,
              "phase2_score": 0.73,
              "half_disease_confidence": 0.9,
              "matched_diseases": ["gastroenteritis"],
              "matched_symptoms": ["diarrhoea"],
              "selection_source": "half_confirmed"
            }
          ]
        }
      ]
    }
  ]
}
```

Flat view:

```json
{
  "recommendations": [
    {
      "query_index": 0,
      "disease": "gastroenteritis",
      "disease_confidence": 0.59,
      "drug_name": "example_drug",
      "disease_rank": 1,
      "global_display_rank": 1,
      "phase2_rank": 4,
      "selection_source": "half_confirmed"
    }
  ]
}
```

Output requirements:

- `disease_rank`: rank inside that disease's Top3.
- `global_display_rank`: display order after concatenating disease results.
- `phase2_rank`: original rank in the disease-specific Phase II Top20.
- `phase2_score`: original Phase II score, if available.
- `half_disease_confidence`: evidence metadata from half data if available;
  use `0` or `null` for fallback rows.
- `selection_source`: either `half_confirmed` or `phase2_fallback`.

## CLI Examples

Run on an input JSON:

```bash
python -m app.evaluation.run_phase2_final_recommendation \
  --input-json data/sample_phase2_final_queries.json \
  --top-k-recall 20 \
  --top-k-per-disease 3 \
  --output-json artifacts/phase2_final_recommendations/results.json
```

Run a quick debug query:

```bash
python -m app.evaluation.run_phase2_final_recommendation \
  --diseases gastroenteritis common_cold \
  --symptoms diarrhoea mild_fever \
  --top-k-recall 20 \
  --top-k-per-disease 3
```

## Test Plan

Static check:

```bash
python -m py_compile \
  app/embedded_module/phase2_final_recommender.py \
  app/evaluation/run_phase2_final_recommendation.py
```

Smoke test:

```bash
python -m app.evaluation.run_phase2_final_recommendation \
  --diseases acne common_cold \
  --symptoms cough mild_fever \
  --output-json artifacts/phase2_final_recommendations/smoke.json
```

Acceptance criteria:

- One disease returns at most 3 final recommendations.
- Two diseases return at most 6 final recommendations.
- Each disease has its own internal Phase II Top20 recall.
- Final Top3 uses original Phase II rank order.
- Half-confirmed drugs are preferred over fallback drugs.
- Fallback drugs are clearly marked as `phase2_fallback`.
- No `selection_score` exists in output.
- Multi-disease output preserves disease grouping and does not global-rerank.
- The CLI can read classifier-style JSON where symptoms are shared across diseases.

## Assumptions

- This is the Phase II final submission pipeline.
- FastAPI integration is out of scope for this step.
- Half JSON is a post-selection reference, not a replacement for Phase II recall.
- Symptom confidence and disease confidence are explanation metadata only.
- `need_first_aid` is not part of drug recommendation selection.
