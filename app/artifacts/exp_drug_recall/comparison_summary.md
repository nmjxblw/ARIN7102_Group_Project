# Exp vs Yinan Comparison Summary

Generated from: `metrics.json` + `per_query_results.csv`

## Per-Comparison Metrics

| Comparison | mode_A hit@20 | mode_B hit@20 | Δ hit@20 | mode_A recall@20 | mode_B recall@20 | Δ recall@20 | mode_A mrr | mode_B mrr | Δ mrr | mode_A P@20 | mode_B P@20 | Δ P@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yinan_label vs exp_label_idf_only | 0.6859 | 0.6859 | +0.0000 | 0.5534 | 0.5603 | +0.0069 | 0.2983 | 0.3199 | +0.0216 | 0.0793 | 0.0775 | -0.0018 |
| yinan_fusion vs exp_candidate_union | 0.6492 | 0.7801 | +0.1309 | 0.496 | 0.6832 | +0.1872 | 0.2585 | 0.403 | +0.1445 | 0.0660 | 0.0958 | +0.0298 |
| candidate_union vs candidate_union_no_prior | 0.7801 | 0.7749 | -0.0052 | 0.6832 | 0.6878 | +0.0046 | 0.403 | 0.3847 | -0.0182 | 0.0958 | 0.0953 | -0.0005 |
| candidate_union vs candidate_union_no_bm25 | 0.7801 | 0.9372 | +0.1571 | 0.6832 | 0.8809 | +0.1977 | 0.403 | 0.6135 | +0.2106 | 0.0958 | 0.1152 | +0.0194 |
| candidate_union vs candidate_union_no_prior_no_bm25 | 0.7801 | 0.9843 | +0.2042 | 0.6832 | 0.9475 | +0.2643 | 0.403 | 0.6313 | +0.2284 | 0.0958 | 0.1204 | +0.0246 |

## Conclusions (derived by fixed rules)
### Rule: `dense_unavailable`
semantic-only hit@20 = 0 — dense/semantic path is not functional. Should NOT be credited as a positive contributor in any explanation.
```json
{
  "semantic_hit@20": 0.0
}
```
### Rule: `bm25_is_noise_or_harmful`
Removing BM25 improves hit@20 by +0.1571 and improves recall@20 by +0.1977. BM25 is currently noisy or harmful, not a positive contributor.
```json
{
  "union_hit@20": 0.7801,
  "no_bm25_hit@20": 0.9372,
  "delta_hit@20": 0.1571,
  "union_recall@20": 0.6832,
  "no_bm25_recall@20": 0.8809,
  "bm25_recall_delta": 0.1977,
  "prior_recall_delta": 0.0046
}
```
