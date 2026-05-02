import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path.cwd()

def calculate_row_conf(row: dict) -> float:
    diseases = row.get("diseases", [])
    symptoms = row.get("symptoms", [])
    
    disease_confs = []
    for d in diseases:
        if isinstance(d, dict):
            # The value is the confidence, e.g. {"acne": 0.95}
            disease_confs.append(float(next(iter(d.values()))))
        elif isinstance(d, (float, int)):
            disease_confs.append(float(d))
            
    symptom_confs = []
    for s in symptoms:
        if isinstance(s, dict):
            symptom_confs.append(float(next(iter(s.values()))))
        elif isinstance(s, (float, int)):
            symptom_confs.append(float(s))
            
    max_disease_conf = max(disease_confs) if disease_confs else 0.0
    
    if symptom_confs:
        mean_symptom_conf = sum(symptom_confs) / len(symptom_confs)
        return 0.7 * max_disease_conf + 0.3 * mean_symptom_conf
    else:
        return max_disease_conf

def load_half_data(json_path: Path, prefix: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    
    confs = {}
    for i, row in enumerate(data, start=1):
        q_id = f"{prefix}_{i:06d}"
        confs[q_id] = calculate_row_conf(row)
    return confs

def main():
    half1_json = REPO_ROOT / "dataset_module" / "drugs_training_dataset" / "drug_data_half_1.json"
    half2_json = REPO_ROOT / "dataset_module" / "drugs_training_dataset" / "drug_data_half_2.json"
    
    half1_csv = REPO_ROOT / "artifacts" / "exp_drug_recall" / "phase2_half1" / "per_query_results.csv"
    half2_csv = REPO_ROOT / "artifacts" / "exp_drug_recall" / "phase2_half2" / "per_query_results.csv"
    
    out_dir = REPO_ROOT / "artifacts" / "exp_drug_recall" / "phase2_half_confidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files exist
    if not half1_csv.exists() or not half2_csv.exists():
        print("Missing per_query_results.csv from Phase II-B. Please run them first.")
        return

    conf1 = load_half_data(half1_json, "half1")
    conf2 = load_half_data(half2_json, "half2")
    all_confs = {**conf1, **conf2}
    
    df1 = pd.read_csv(half1_csv)
    df2 = pd.read_csv(half2_csv)
    df = pd.concat([df1, df2], ignore_index=True)
    
    df["row_conf"] = df["query_id"].map(all_confs)
    
    # Drop rows without matching row_conf just in case
    df = df.dropna(subset=["row_conf"])
    
    query_conf = df[["query_id", "row_conf"]].drop_duplicates()["row_conf"]
    p33 = np.percentile(query_conf, 33.33)
    p67 = np.percentile(query_conf, 66.67)
    
    def get_bucket(val):
        if val <= p33:
            return "low"
        elif val <= p67:
            return "mid"
        else:
            return "high"
            
    df["bucket"] = df["row_conf"].apply(get_bucket)
    
    # Save half_confidence_rows.csv
    df.to_csv(out_dir / "half_confidence_rows.csv", index=False)
    
    # Group by mode and bucket
    metrics = ["hit@5", "hit@10", "hit@20", "recall@5", "recall@10", "recall@20", "mrr"]
    
    bucket_metrics = {}
    modes = df["mode"].unique()
    buckets = ["low", "mid", "high"]
    
    for mode in modes:
        mode_df = df[df["mode"] == mode]
        bucket_metrics[mode] = {}
        for b in buckets:
            b_df = mode_df[mode_df["bucket"] == b]
            b_res = {"count": len(b_df)}
            for m in metrics:
                if m in b_df.columns:
                    b_res[m] = b_df[m].mean()
            bucket_metrics[mode][b] = b_res
            
    # Calculate delta for label_core_rerank vs label_idf_only
    delta_metrics = {}
    if "label_core_rerank" in bucket_metrics and "label_idf_only" in bucket_metrics:
        for b in buckets:
            delta_metrics[b] = {}
            for m in metrics:
                if m in bucket_metrics["label_core_rerank"][b] and m in bucket_metrics["label_idf_only"][b]:
                    delta_metrics[b][m] = bucket_metrics["label_core_rerank"][b][m] - bucket_metrics["label_idf_only"][b][m]
    
    # Save json metrics
    with open(out_dir / "half_confidence_bucket_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"bucket_metrics": bucket_metrics, "delta_label_core_rerank_vs_label_idf_only": delta_metrics}, f, indent=2)
        
    with open(out_dir / "half_confidence_summary.json", "w", encoding="utf-8") as f:
        json.dump({"total_queries": len(all_confs), "p33": p33, "p67": p67}, f, indent=2)
        
    # Generate MD report
    md_lines = ["# Half Confidence Bucket Metrics\n"]
    md_lines.append(f"**p33**: {p33:.4f}  \n**p67**: {p67:.4f}\n")
    
    for mode in modes:
        md_lines.append(f"## {mode}")
        header = "| Bucket | Count | " + " | ".join(metrics) + " |"
        md_lines.append(header)
        md_lines.append("|---|---|---|" + "|".join(["---" for _ in range(len(metrics)-1)]))
        for b in buckets:
            row = f"| {b} | {bucket_metrics[mode][b]['count']} | "
            vals = []
            for m in metrics:
                val = bucket_metrics[mode][b].get(m, 0.0)
                vals.append(f"{val:.4f}")
            row += " | ".join(vals) + " |"
            md_lines.append(row)
        md_lines.append("")
        
    if delta_metrics:
        md_lines.append("## Delta: label_core_rerank - label_idf_only")
        header = "| Bucket | " + " | ".join(metrics) + " |"
        md_lines.append(header)
        md_lines.append("|---|---|" + "|".join(["---" for _ in range(len(metrics)-1)]))
        for b in buckets:
            row = f"| {b} | "
            vals = []
            for m in metrics:
                val = delta_metrics[b].get(m, 0.0)
                vals.append(f"{val:.4f}")
            row += " | ".join(vals) + " |"
            md_lines.append(row)
        md_lines.append("")
        
    with open(out_dir / "half_confidence_bucket_metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    # Plotting
    plot_failed = False
    try:
        conf_array = np.array(list(all_confs.values()))
        
        plt.figure(figsize=(10, 6))
        sns.histplot(conf_array, kde=True)
        plt.axvline(p33, color='r', linestyle='--', label=f'p33: {p33:.4f}')
        plt.axvline(p67, color='g', linestyle='--', label=f'p67: {p67:.4f}')
        plt.title('Distribution of row_conf')
        plt.xlabel('row_conf')
        plt.ylabel('Density/Count')
        plt.legend()
        plt.savefig(out_dir / "half_confidence_density.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(conf_array)
        plt.axhline(0.3333, color='r', linestyle='--', label='33.33%')
        plt.axhline(0.6667, color='g', linestyle='--', label='66.67%')
        plt.axvline(p33, color='r', linestyle=':', label=f'p33: {p33:.4f}')
        plt.axvline(p67, color='g', linestyle=':', label=f'p67: {p67:.4f}')
        plt.title('ECDF of row_conf')
        plt.xlabel('row_conf')
        plt.ylabel('Proportion')
        plt.legend()
        plt.savefig(out_dir / "half_confidence_ecdf.png")
        plt.close()
        
    except Exception as e:
        print(f"Plotting failed: {e}")
        plot_failed = True
        
    if plot_failed:
        print("plot generation failed, numeric bucket analysis completed")

if __name__ == "__main__":
    main()
