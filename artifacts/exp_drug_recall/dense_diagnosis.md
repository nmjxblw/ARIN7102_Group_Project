# Dense Recall Diagnosis

**Embedding**: `/Users/jayden/Desktop/7012 datamining and text/project_march/ARIN7102_Group_Project/drug_comprehensive_embeddings.npy`  shape=(5595, 2, 768)
**Query count**: 191  (all of `eval_dataset_verified.json`)

## Per-View Results

| View | hit@20 | recall@20 | mrr | hit queries | miss queries |
|---|---|---|---|---|---|
| `view0`  | 0.2461 | 0.1240 | 0.0787 | 47 | 144 |
| `view1`  ◄ CURRENT | 0.2984 | 0.1646 | 0.1134 | 57 | 134 |
| `vmean`  ◄ BEST | 0.3246 | 0.1610 | 0.1106 | 62 | 129 |

### Notes
- **view0**: emb[:, 0, :]  — drug_recall_index.py path
- **view1**: emb[:, 1, :]  — dual_recall_pipeline.py path (CURRENT)
- **vmean**: (emb[:,0,:]+emb[:,1,:])/2  — mean pooling

## Conclusions (by plan rules)
### Rule: `view_selection_matters`
View vmean performs best (hit@20=0.3246). Consider fixing the view/projection strategy first.

Evidence:
```json
{
  "view0": 0.2461,
  "view1": 0.2984,
  "vmean": 0.3246
}
```

### Rule: `view0_vs_view1`
view1 outperforms the other by 0.0524 hit@20. Verify which projection is used by the production pipeline.

Evidence:
```json
{
  "view0_hit@20": 0.2461,
  "view1_hit@20": 0.2984
}
```

## Sample Failure Queries (per view)

### view0 failures (144 queries)
- `eval_0011`  top1=`sebulex`  score=13.8386  relevant=3  text="I have big, painful bumps on my skin that leave sc"
- `eval_0006`  top1=`uceris`  score=14.1256  relevant=1  text="Every time I eat, I feel sick and my stomach hurts"
- `eval_0009`  top1=`aerobid`  score=14.3628  relevant=1  text="I'm struggling to breathe, coughing with phlegm, a"
- `eval_0016`  top1=`sotret`  score=13.9617  relevant=3  text="My friend's skin has a rash with some scarring fro"
- `eval_0019`  top1=`pryflex`  score=14.0883  relevant=2  text="I can't walk because of the pain in my knees and t"

### view1 failures (134 queries)
- `eval_0011`  top1=`salicylic acid / sulfur`  score=13.9142  relevant=3  text="I have big, painful bumps on my skin that leave sc"
- `eval_0006`  top1=`phosphorated carbohydrate solution`  score=14.1229  relevant=1  text="Every time I eat, I feel sick and my stomach hurts"
- `eval_0009`  top1=`pulmicort respules`  score=14.4128  relevant=1  text="I'm struggling to breathe, coughing with phlegm, a"
- `eval_0016`  top1=`finacea`  score=13.9591  relevant=3  text="My friend's skin has a rash with some scarring fro"
- `eval_0019`  top1=`orudis kt`  score=14.1649  relevant=2  text="I can't walk because of the pain in my knees and t"

### vmean failures (129 queries)
- `eval_0011`  top1=`sebulex`  score=13.9669  relevant=3  text="I have big, painful bumps on my skin that leave sc"
- `eval_0006`  top1=`uceris`  score=14.1544  relevant=1  text="Every time I eat, I feel sick and my stomach hurts"
- `eval_0009`  top1=`pulmicort respules`  score=14.4479  relevant=1  text="I'm struggling to breathe, coughing with phlegm, a"
- `eval_0016`  top1=`junel fe 1 / 20`  score=14.0573  relevant=3  text="My friend's skin has a rash with some scarring fro"
- `eval_0019`  top1=`relamine`  score=14.1507  relevant=2  text="I can't walk because of the pain in my knees and t"
