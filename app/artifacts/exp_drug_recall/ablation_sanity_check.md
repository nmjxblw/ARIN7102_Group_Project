# Ablation Sanity Check

- Eval dataset: `/Users/jayden/Desktop/7012 datamining and text/project_march/ARIN7102_Group_Project/data/eval_dataset_verified.json`
- Queries: `191`
- Checked pool size: `1000`

## Baseline Metrics

| Mode | hit@20 | recall@20 | mrr | precision@20 |
|---|---|---|---|---|
| `label_idf_only` | 0.6859 | 0.5603 | 0.3199 | 0.0775 |
| `candidate_union` | 0.7801 | 0.6832 | 0.4030 | 0.0958 |
| `candidate_union_no_prior` | 0.7749 | 0.6878 | 0.3847 | 0.0953 |
| `candidate_union_no_bm25` | 0.9372 | 0.8809 | 0.6135 | 0.1152 |
| `candidate_union_no_prior_no_bm25` | 0.9843 | 0.9475 | 0.6313 | 0.1204 |

## Candidate-Set Checks

- `label_idf_only` and `candidate_union_no_prior_no_bm25` have identical candidate sets on `191/191` queries.
- Mean label-core candidate count: `758.7`
- Mean `candidate_union` selected count: `1000.0`
- Mean pre-cap union size: `1556.39`
- `candidate_union` exceeds the 1000-row cap on `191/191` queries.
- BM25 participation excludes label-core rows on `158/191` queries.
- Total label-core rows dropped by `candidate_union`: `52375`
- Queries where dropped rows contain relevant drugs: `0`

## Raw Stage Score Scale

| Stage | queries | mean max | mean p95 | mean median |
|---|---|---|---|---|
| `disease` | 191 | 4.6875 | 4.6106 | 3.8434 |
| `strict` | 191 | 13.0218 | 12.1705 | 9.7364 |
| `symptom` | 191 | 8.3343 | 6.9482 | 4.2038 |
| `bm25` | 191 | 57.3521 | 41.8703 | 21.0044 |
| `prior` | 191 | 0.8517 | 0.8500 | 0.8500 |

## Candidate Union Pool Sweep

| pool_size | hit@20 | recall@20 | mrr | precision@20 |
|---|---|---|---|---|
| `1000` | 0.7801 | 0.6832 | 0.4030 | 0.0958 |
| `1500` | 0.8010 | 0.7018 | 0.3919 | 0.0971 |
| `2000` | 0.8115 | 0.7170 | 0.3937 | 0.0992 |
| `3000` | 0.8115 | 0.7183 | 0.4046 | 0.0984 |
| `5595` | 0.8168 | 0.7216 | 0.4123 | 0.0987 |

## Conclusions

- label_idf_only and candidate_union_no_prior_no_bm25 use the same candidate rows on all queries; their performance gap comes from deterministic reranking, not extra recall sources.
- candidate_union overflows the 1000-row cap on every query, and BM25 participation causes label-core rows to be dropped before final scoring.
- candidate_union improves from hit@20=0.7801 / recall@20=0.6832 at pool_size=1000 to hit@20=0.8168 / recall@20=0.7216 at pool_size=5595; the current 1000-row cap is interacting badly with raw stage-score selection.
- BM25 raw scores operate on a much larger scale than strict label scores (mean per-query max 57.3521 vs 13.0218), so summing raw stage scores before cap selection is numerically biased toward BM25.
