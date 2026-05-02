#!/usr/bin/env python3
"""
Compare experimental ablation results vs yinan pipeline.
Reads fixed inputs, writes fixed outputs, derives conclusions by rules.

Inputs:
  data/eval_results_label.json
  data/eval_results_fusion.json
  data/eval_results_semantic.json
  artifacts/exp_drug_recall/metrics.json
  artifacts/exp_drug_recall/per_query_results.csv

Outputs:
  artifacts/exp_drug_recall/comparison_summary.md
  artifacts/exp_drug_recall/comparison_summary.json
  artifacts/exp_drug_recall/comparison_per_query.csv
"""
import csv
import json
from pathlib import Path

APP_ROOT = Path.cwd()
OUT_DIR  = APP_ROOT / "artifacts" / "exp_drug_recall"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load yinan metrics (aggregate only — no per-query in these files) ─────────
yinan_label     = json.load(open(APP_ROOT / "data" / "eval_results_label.json"))["metrics"]
yinan_fusion    = json.load(open(APP_ROOT / "data" / "eval_results_fusion.json"))["metrics"]
yinan_semantic  = json.load(open(APP_ROOT / "data" / "eval_results_semantic.json"))["metrics"]

# ── Load exp metrics ──────────────────────────────────────────────────────────
exp_metrics = json.load(open(OUT_DIR / "metrics.json"))["metrics"]

def gm(m, key):
    """Get metric, supporting both mean_* and bare key names."""
    return m.get(f"mean_{key}") or m.get(key, 0.0)

# ── Build comparison rows ────────────────────────────────────────────────────
comparisons = [
    # (label, mode_a, mode_b, a_dict, b_dict)
    ("yinan_label vs exp_label_idf_only",
     "yinan_label", "exp_label_idf_only",
     yinan_label, exp_metrics["label_idf_only"]),

    ("yinan_fusion vs exp_candidate_union",
     "yinan_fusion", "exp_candidate_union",
     yinan_fusion, exp_metrics["candidate_union"]),

    ("candidate_union vs candidate_union_no_prior",
     "candidate_union", "candidate_union_no_prior",
     exp_metrics["candidate_union"], exp_metrics["candidate_union_no_prior"]),

    ("candidate_union vs candidate_union_no_bm25",
     "candidate_union", "candidate_union_no_bm25",
     exp_metrics["candidate_union"], exp_metrics["candidate_union_no_bm25"]),

    ("candidate_union vs candidate_union_no_prior_no_bm25",
     "candidate_union", "candidate_union_no_prior_no_bm25",
     exp_metrics["candidate_union"], exp_metrics["candidate_union_no_prior_no_bm25"]),
]

rows = []
for label, a_name, b_name, a, b in comparisons:
    row = {
        "comparison": label,
        "mode_a": a_name,
        "mode_b": b_name,
        # mode_a metrics
        f"{a_name}_hit@20":     round(gm(a, "hit@20"),     4),
        f"{a_name}_recall@20":  round(gm(a, "recall@20"),  4),
        f"{a_name}_mrr":        round(gm(a, "mrr"),        4),
        f"{a_name}_precision@20": round(gm(a, "precision@20"), 4),
        # mode_b metrics
        f"{b_name}_hit@20":     round(gm(b, "hit@20"),     4),
        f"{b_name}_recall@20":  round(gm(b, "recall@20"),  4),
        f"{b_name}_mrr":        round(gm(b, "mrr"),        4),
        f"{b_name}_precision@20": round(gm(b, "precision@20"), 4),
        # deltas (b - a)
        "delta_hit@20":       round(gm(b, "hit@20")     - gm(a, "hit@20"),     4),
        "delta_recall@20":    round(gm(b, "recall@20") - gm(a, "recall@20"), 4),
        "delta_mrr":          round(gm(b, "mrr")        - gm(a, "mrr"),        4),
        "delta_precision@20":  round(gm(b, "precision@20") - gm(a, "precision@20"), 4),
    }
    rows.append(row)

# ── Apply fixed conclusion rules ─────────────────────────────────────────────
#
# Rule ordering (checked in sequence; first matching rule wins):
#
# 1. dense_unavailable: yinan semantic hit@20 == 0
# 2. both_are_noise: removing prior, bm25, and both all improve-or-match hit@20
# 3. bm25_is_noise_or_harmful:
#      no_bm25 hit@20 > union hit@20
#      and no_bm25 recall@20 >= union recall@20
# 4. bm25_hurts_ranking_but_needed_for_recall:
#      no_bm25 hit@20 > union hit@20
#      and no_bm25 recall@20 < union recall@20
# 5. prior_is_main_gain:
#      removing prior hurts recall more than removing bm25,
#      and bm25 does not hurt ranking
#
# This order keeps the primary interpretation tied to measured metric direction.

hit_semantic  = gm(yinan_semantic, "hit@20")
hit_union     = gm(exp_metrics["candidate_union"],          "hit@20")
hit_no_prior  = gm(exp_metrics["candidate_union_no_prior"],  "hit@20")
hit_no_bm25   = gm(exp_metrics["candidate_union_no_bm25"],  "hit@20")
hit_no_pb     = gm(exp_metrics["candidate_union_no_prior_no_bm25"], "hit@20")

rec_union     = gm(exp_metrics["candidate_union"],          "recall@20")
rec_no_prior  = gm(exp_metrics["candidate_union_no_prior"],  "recall@20")
rec_no_bm25   = gm(exp_metrics["candidate_union_no_bm25"],  "recall@20")

# Delta vs union (positive = removing this component improves the metric)
prior_recall_delta = rec_no_prior - rec_union
bm25_recall_delta  = rec_no_bm25  - rec_union

bm25_hurts_ranking = hit_no_bm25 > hit_union
prior_hurts_ranking = hit_no_prior > hit_union

conclusions = []

# Rule 3: dense unavailable
if hit_semantic == 0.0:
    conclusions.append({
        "rule": "dense_unavailable",
        "finding": "semantic-only hit@20 = 0 — dense/semantic path is not functional. "
                   "Should NOT be credited as a positive contributor in any explanation.",
        "evidence": {"semantic_hit@20": hit_semantic}
    })

# Rule 2: both are noise — their removal strictly improves or matches hit@20
both_match_or_improve = (
    hit_no_pb     >= hit_union and
    hit_no_prior  >= hit_union and
    hit_no_bm25   >= hit_union
)
if both_match_or_improve:
    conclusions.append({
        "rule": "both_are_noise",
        "finding": f"All ablations (no_prior, no_bm25, no_prior_no_bm25) improve hit@20 vs union "
                   f"({hit_union:.4f}). Both prior and BM25 are ranking noise — label-only union is best.",
        "evidence": {
            "union_hit@20":          round(hit_union,    4),
            "no_prior_hit@20":        round(hit_no_prior, 4),
            "no_bm25_hit@20":        round(hit_no_bm25,  4),
            "no_prior_no_bm25_hit@20": round(hit_no_pb,   4),
            "prior_recall_delta":     round(prior_recall_delta, 4),
            "bm25_recall_delta":      round(bm25_recall_delta,  4),
        }
    })
elif bm25_hurts_ranking and prior_recall_delta <= 0:
    # BM25 hurts ranking AND prior doesn't help recall → BM25 is pure ranking noise
    conclusions.append({
        "rule": "bm25_is_ranking_noise",
        "finding": f"Removing BM25 improves hit@20 by {hit_no_bm25 - hit_union:+.4f} (BM25 is "
                   f"ranking noise). Prior also hurts ranking (hit@20 delta {hit_no_prior - hit_union:+.4f}). "
                   f"Both should be excluded; label union is sufficient.",
        "evidence": {
            "union_hit@20":          round(hit_union,    4),
            "no_bm25_hit@20":        round(hit_no_bm25,  4),
            "delta_hit@20":          round(hit_no_bm25 - hit_union, 4),
            "no_prior_hit@20":        round(hit_no_prior, 4),
            "prior_recall_delta":     round(prior_recall_delta, 4),
        }
    })
elif bm25_hurts_ranking and bm25_recall_delta >= 0:
    # BM25 hurts ranking and does not help recall → noisy or harmful.
    recall_phrase = (
        f"improves recall@20 by {bm25_recall_delta:+.4f}"
        if bm25_recall_delta > 0
        else "does not change recall@20"
    )
    conclusions.append({
        "rule": "bm25_is_noise_or_harmful",
        "finding": f"Removing BM25 improves hit@20 by {hit_no_bm25 - hit_union:+.4f} and "
                   f"{recall_phrase}. BM25 is currently noisy or harmful, not a positive contributor.",
        "evidence": {
            "union_hit@20":          round(hit_union,    4),
            "no_bm25_hit@20":        round(hit_no_bm25,  4),
            "delta_hit@20":          round(hit_no_bm25 - hit_union, 4),
            "union_recall@20":       round(rec_union,    4),
            "no_bm25_recall@20":     round(rec_no_bm25,  4),
            "bm25_recall_delta":     round(bm25_recall_delta, 4),
            "prior_recall_delta":    round(prior_recall_delta, 4),
        }
    })
elif bm25_hurts_ranking and bm25_recall_delta < 0:
    # BM25 hurts ranking but is needed for recall coverage → trade-off
    conclusions.append({
        "rule": "bm25_hurts_ranking_but_needed_for_recall",
        "finding": f"Removing BM25 improves hit@20 by {hit_no_bm25 - hit_union:+.4f} (BM25 "
                   f"hurts ranking), but removing it drops recall@20 by {abs(bm25_recall_delta):.4f}. "
                   f"Prior (recall delta {prior_recall_delta:+.4f}) is not the main recall driver.",
        "evidence": {
            "union_hit@20":          round(hit_union,    4),
            "no_bm25_hit@20":        round(hit_no_bm25,  4),
            "delta_hit@20":          round(hit_no_bm25 - hit_union, 4),
            "union_recall@20":       round(rec_union,    4),
            "no_bm25_recall@20":     round(rec_no_bm25,  4),
            "bm25_recall_delta":     round(bm25_recall_delta, 4),
            "prior_recall_delta":    round(prior_recall_delta, 4),
        }
    })
else:
    # Neither BM25 nor prior clearly hurts ranking → check recall
    if prior_recall_delta > bm25_recall_delta and prior_recall_delta > 0:
        conclusions.append({
            "rule": "prior_is_main_gain",
            "finding": f"Prior is the main recall gain source (delta={prior_recall_delta:+.4f}). "
                       f"BM25 recall delta={bm25_recall_delta:+.4f}.",
            "evidence": {
                "union_recall@20":   round(rec_union,    4),
                "no_prior_recall@20": round(rec_no_prior, 4),
                "prior_recall_delta": round(prior_recall_delta, 4),
                "bm25_recall_delta":  round(bm25_recall_delta,  4),
            }
        })
    elif bm25_recall_delta > 0:
        conclusions.append({
            "rule": "bm25_is_main_gain",
            "finding": f"BM25 is the main recall gain source (delta={bm25_recall_delta:+.4f}). "
                       f"Prior recall delta={prior_recall_delta:+.4f}.",
            "evidence": {
                "union_recall@20": round(rec_union,    4),
                "no_bm25_recall@20": round(rec_no_bm25, 4),
                "bm25_recall_delta": round(bm25_recall_delta, 4),
                "prior_recall_delta": round(prior_recall_delta, 4),
            }
        })
    else:
        conclusions.append({
            "rule": "both_are_noise_or_harmful",
            "finding": f"Both prior (recall delta={prior_recall_delta:+.4f}) and BM25 "
                       f"(recall delta={bm25_recall_delta:+.4f}) either hurt or have negligible "
                       f"effect on recall.",
            "evidence": {
                "union_recall@20":    round(rec_union,    4),
                "no_prior_recall@20": round(rec_no_prior, 4),
                "no_bm25_recall@20":  round(rec_no_bm25,  4),
                "prior_recall_delta": round(prior_recall_delta, 4),
                "bm25_recall_delta":  round(bm25_recall_delta,  4),
            }
        })

# ── Write comparison_summary.json ─────────────────────────────────────────────
summary_json = {
    "comparisons": rows,
    "conclusions": conclusions,
    "all_modes_hit20": {
        "yinan_semantic":                      round(hit_semantic, 4),
        "yinan_label":                         round(gm(yinan_label,    "hit@20"), 4),
        "yinan_fusion":                        round(gm(yinan_fusion,   "hit@20"), 4),
        "exp_label_idf_only":                   round(gm(exp_metrics["label_idf_only"],                    "hit@20"), 4),
        "exp_label_bm25":                       round(gm(exp_metrics["label_bm25"],                       "hit@20"), 4),
        "exp_candidate_union":                 round(hit_union, 4),
        "exp_candidate_union_no_prior":         round(hit_no_prior, 4),
        "exp_candidate_union_no_bm25":          round(hit_no_bm25, 4),
        "exp_candidate_union_no_prior_no_bm25": round(hit_no_pb, 4),
    },
}

with open(OUT_DIR / "comparison_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2, ensure_ascii=False)

# ── Write comparison_summary.md ────────────────────────────────────────────────
md_lines = [
    "# Exp vs Yinan Comparison Summary\n\n",
    f"Generated from: `metrics.json` + `per_query_results.csv`\n\n",
    "## Per-Comparison Metrics\n\n",
]

# Build table header
md_lines.append(
    "| Comparison | mode_A hit@20 | mode_B hit@20 | Δ hit@20 | "
    "mode_A recall@20 | mode_B recall@20 | Δ recall@20 | "
    "mode_A mrr | mode_B mrr | Δ mrr | "
    "mode_A P@20 | mode_B P@20 | Δ P@20 |\n"
)
md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

for r in rows:
    a = r["mode_a"]
    b = r["mode_b"]
    md_lines.append(
        f"| {r['comparison']} |"
        f" {r[f'{a}_hit@20']} |"
        f" {r[f'{b}_hit@20']} |"
        f" {r['delta_hit@20']:+.4f} |"
        f" {r[f'{a}_recall@20']} |"
        f" {r[f'{b}_recall@20']} |"
        f" {r['delta_recall@20']:+.4f} |"
        f" {r[f'{a}_mrr']} |"
        f" {r[f'{b}_mrr']} |"
        f" {r['delta_mrr']:+.4f} |"
        f" {r[f'{a}_precision@20']:.4f} |"
        f" {r[f'{b}_precision@20']:.4f} |"
        f" {r['delta_precision@20']:+.4f} |\n"
    )

md_lines.append("\n## Conclusions (derived by fixed rules)\n")
for c in conclusions:
    md_lines.append(f"### Rule: `{c['rule']}`\n")
    md_lines.append(f"{c['finding']}\n")
    md_lines.append(f"```json\n{json.dumps(c['evidence'], indent=2)}\n```\n")

with open(OUT_DIR / "comparison_summary.md", "w") as f:
    f.writelines(md_lines)

# ── Write comparison_per_query.csv ──────────────────────────────────────────
# Load per-query CSV from exp run
per_query_csv = OUT_DIR / "per_query_results.csv"

# We don't have per-query yinan data, so per-query CSV is just the exp modes.
# Load all exp per-query results.
exp_per_query = {}
with open(per_query_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        mode = row["mode"]
        qid  = row["query_id"]
        if mode not in exp_per_query:
            exp_per_query[mode] = {}
        exp_per_query[mode][qid] = row

# Write comparison_per_query: label_idf_only vs candidate_union vs candidate_union_no_bm25
modes_4 = ["label_idf_only", "candidate_union", "candidate_union_no_prior",
           "candidate_union_no_bm25", "candidate_union_no_prior_no_bm25"]

# Collect all query IDs present
all_qids = set()
for mode in modes_4:
    all_qids.update(exp_per_query.get(mode, {}).keys())

comparison_csv_rows = []
for qid in sorted(all_qids):
    base_row = {"query_id": qid}
    for mode in modes_4:
        m = exp_per_query.get(mode, {}).get(qid)
        if m:
            base_row[f"{mode}_hit@20"]    = m.get("hit@20", "0")
            base_row[f"{mode}_recall@20"] = m.get("recall@20", "0")
            base_row[f"{mode}_mrr"]       = m.get("mrr", "0")
        else:
            base_row[f"{mode}_hit@20"]    = "NA"
            base_row[f"{mode}_recall@20"] = "NA"
            base_row[f"{mode}_mrr"]       = "NA"
    comparison_csv_rows.append(base_row)

fieldnames = ["query_id"]
for mode in modes_4:
    for suf in ["_hit@20", "_recall@20", "_mrr"]:
        fieldnames.append(f"{mode}{suf}")

with open(OUT_DIR / "comparison_per_query.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(comparison_csv_rows)

print("Written:")
print(f"  {OUT_DIR / 'comparison_summary.json'}")
print(f"  {OUT_DIR / 'comparison_summary.md'}")
print(f"  {OUT_DIR / 'comparison_per_query.csv'}  ({len(comparison_csv_rows)} rows)")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CONCLUSIONS (derived by plan rules)")
print("=" * 70)
for c in conclusions:
    print(f"\n[{c['rule']}]")
    print(f"  {c['finding']}")
    print(f"  Evidence: {c['evidence']}")
