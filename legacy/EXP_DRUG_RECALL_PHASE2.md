# Experimental Drug Recall Phase II Report

## 1. Scope

Phase II focused on stabilizing the experimental recall mainline and clarifying
how the filtered half data should be used.

The production FastAPI path was not changed. Phase II only updated the
experimental evaluation path around:

- `app/embedded_module/experimental_recall_pipeline.py`
- `app/evaluation/run_exp_drug_recall.py`
- `app/evaluation/half_data_adapter.py`
- `app/evaluation/analyze_half_confidence.py`

The final mainline mode is:

```text
label_core_rerank
```

This mode keeps only label-core candidate stages:

- `disease`
- `strict`
- `symptom`

It explicitly disables:

- BM25 score
- dense score
- prior expansion stage

The intent is to measure label-driven drug recall without BM25, dense, or prior
stage leakage.

## 2. Mainline Definition

`label_core_rerank` uses the same core candidate sources as `label_idf_only`,
then reranks those candidates with deterministic label and quality features.

Final score:

```text
final_score =
    0.50 * label_idf_score
  + 0.20 * symptom_coverage
  + 0.10 * quality_prior
  + 0.06 * stage_strict
  + 0.03 * stage_disease
  - others_penalty
```

Leakage check for the final verified runs:

| Stage | Expected | Observed |
|---|---:|---:|
| `stage_hit_bm25` | 0.0 | 0.0 |
| `stage_hit_dense` | 0.0 | 0.0 |
| `stage_hit_prior` | 0.0 | 0.0 |

## 3. Verified Results

The original verified benchmark has 191 queries. A newer larger verified file
adds a 742-query check:

- `data/eval_dataset_verified.json`
- `data/eval_dataset_verified_1000_deepseek_v4_flash.json`

Both were evaluated with `label_core_rerank`.

| Dataset | Queries | hit@20 | precision@20 | recall@20 | ndcg@20 | mrr |
|---|---:|---:|---:|---:|---:|---:|
| old verified | 191 | 0.9843 | 0.1204 | 0.9475 | 0.6716 | 0.6313 |
| new verified | 742 | 0.9879 | 0.1172 | 0.9638 | 0.6925 | 0.6654 |

The larger verified run confirms that the strong verified-set result is stable,
not an accident from the 191-query sample.

Important interpretation:

- Verified queries have small relevant-drug pools.
- The old verified set averages 2.61 relevant drugs per query.
- The new verified set averages 2.44 relevant drugs per query.
- Therefore, `recall@20` can be very high when the label-core path finds those
  small answer sets.

## 4. Half Data Reassessment

The original Phase II half evaluation converted every half row into one query
with exactly one relevant drug. That row-level interpretation is now deprecated.

Reason:

```text
half row:
  drug_name = aczone
  diseases = acne

old row-level query:
  disease = acne
  relevant_drugs = [aczone]
```

This incorrectly treats other acne drugs as false negatives. For example,
`accutane`, `tretinoin`, and `adapalene / benzoyl peroxide` can all be valid
acne drugs, but the old row-level metric marks them wrong unless the exact row's
single `drug_name` appears.

The corrected half evaluation merges `drug_data_half_1.json` and
`drug_data_half_2.json` before grouping:

- `--half-grouping disease`
- optional future view: `--half-grouping disease_symptom`

For disease-level grouping:

```text
query = acne
relevant_drugs = all half1 + half2 drugs labelled acne
```

The adapter also canonicalizes half spellings such as underscore-separated drug
names to the display names in the recall table when possible.

## 5. Half Disease-Level Results

The corrected half disease-level evaluation uses 31 disease queries from the
merged half data.

| Dataset | Queries | hit@20 | precision@20 | recall@20 | ndcg@20 | mrr |
|---|---:|---:|---:|---:|---:|---:|
| half disease-level | 31 | 0.7419 | 0.2806 | 0.1477 | 0.3453 | 0.5110 |

The half disease-level relevant pools are much larger than verified:

- Average relevant pool size: 79.29 drugs.
- Median relevant pool size: 47 drugs.
- Maximum relevant pool size: 340 drugs.

This explains the metric shape:

- `hit@20` is moderate-high because 23 of 31 diseases have at least one hit.
- `precision@20` means the average Top-20 contains about 5.61 half-matching
  disease drugs.
- `recall@20` is low because Top-20 cannot cover a large disease pool.

Examples:

| Disease | Relevant pool | Top-20 hits | hit@20 | precision@20 | recall@20 |
|---|---:|---:|---:|---:|---:|
| acne | 171 | 15 | 1.0 | 0.75 | 0.0877 |
| common_cold | 305 | 5 | 1.0 | 0.25 | 0.0164 |

Half disease-level misses at Top-20:

- `alcoholic_hepatitis`
- `cervical_spondylosis`
- `dimorphic_hemorrhoids_piles`
- `drug_reaction`
- `fungal_infection`
- `peptic_ulcer_disease`
- `tuberculosis`
- `typhoid`

These misses are the best next coverage targets if we continue improving the
label-core path.

## 6. Half Confidence Analysis

The original half confidence values remain useful only for descriptive analysis,
not as query-side model confidence.

Observed distribution:

- Total rows: 2931
- `p33`: 0.8239
- `p67`: 0.8917

The previous row-level confidence bucket run showed that the high-confidence
bucket was not easier to hit. Because row-level half metrics are now deprecated,
these bucket results should not be used as model-quality evidence. The safe
conclusion remains:

```text
Do not use raw half confidence as query confidence or direct ranker weight.
```

If Phase III uses half confidence, it should first rebuild the analysis on the
corrected disease-level or disease-symptom-level grouping.

## 7. Final Conclusion

Phase II experiments are complete.

Conclusions:

1. `label_core_rerank` is the clean experimental mainline.
2. Verified benchmark performance is stable after expanding from 191 to 742
   verified queries.
3. The old half row-level evaluation is invalid for main conclusions and should
   be treated as deprecated.
4. The corrected half disease-level evaluation is a cleaner sanity check. It
   shows that the system covers most diseases, but still has non-trivial
   coverage and ranking gaps.
5. Phase II should not claim that drug recall is fully solved. The right claim
   is:

```text
Verified-set recall is saturated and stable, while clean half-derived
disease-level evaluation still exposes coverage gaps.
```

## 8. Artifacts

Verified:

- `artifacts/exp_drug_recall/phase2_verified/metrics.json`
- `artifacts/exp_drug_recall/phase2_verified_1000/metrics.json`

Half:

- `artifacts/exp_drug_recall/phase2_half_all_disease/metrics.json`
- `artifacts/exp_drug_recall/phase2_half_all_disease/per_query_results.csv`

Deprecated half row-level artifacts:

- `artifacts/exp_drug_recall/phase2_half1/metrics.json`
- `artifacts/exp_drug_recall/phase2_half2/metrics.json`

These row-level artifacts are retained for traceability but should not be used
as the main Phase II conclusion.

## 9. Next Step

Recommended Phase III direction:

1. First repair label coverage for the eight missed half disease groups.
2. Then run a `disease_symptom` half view to check whether symptom-level
   filtering is too strict or too loose.
3. Only after coverage gaps are understood, revisit `local_ranker`.

This order is safer than training a ranker on top of known label coverage gaps.
