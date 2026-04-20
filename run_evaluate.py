"""
Drug Recommendation - 一键评估脚本
在项目根目录直接运行: python run_evaluate.py

用法:
    python run_evaluate.py                       # 跑全部评估集
    python run_evaluate.py --limit 10            # 只跑前 10 条
    python run_evaluate.py --limit 50 --output eval_results.json
    python run_evaluate.py --eval-dataset path/to/dataset.json --k-values 5 10 20
"""
import sys
import os
import argparse
import json
import time
from pathlib import Path

# ─── 环境初始化 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
os.chdir(APP_ROOT)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / "pipeline_config.env", override=False)
load_dotenv(APP_ROOT / ".env", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)

if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "dummy_not_needed_for_pipeline"

from pipeline_config import cfg


def main():
    parser = argparse.ArgumentParser(description="Drug Recommendation Pipeline - 一键评估")
    parser.add_argument(
        "--eval-dataset", type=str,
        default=str(PROJECT_ROOT / "match_data_preprocessing" / "data" / "eval_dataset_verified.json"),
        help="评估数据集路径",
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20], help="K 值列表")
    parser.add_argument("--limit", type=int, default=None, help="限制评估条数（用于快速验证）")
    parser.add_argument("--output", type=str, default=None, help="结果输出文件路径")
    args = parser.parse_args()
    eval_path = Path(args.eval_dataset)
    if not eval_path.exists():
        print(f"ERROR: 评估数据集不存在: {eval_path}")
        sys.exit(1)

    print("=" * 60)
    print("Drug Recommendation Pipeline - Evaluation")
    print("=" * 60)

    # 加载评估集
    with open(eval_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    total = len(eval_data)
    if args.limit:
        eval_data = eval_data[: args.limit]
    print(f"Dataset:  {eval_path.name} ({total} total, running {len(eval_data)})")
    print(f"K values: {args.k_values}")
    print()

    # 初始化服务
    print("Loading models...")
    from fastapi_module.service import get_recommendation_service
    service = get_recommendation_service()
    service.ensure_ready()
    print("Service ready.\n")

    # 逐条评估
    from evaluation.metrics import evaluate_batch

    results = []
    start_time = time.time()

    for i, query in enumerate(eval_data):
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(eval_data) - i - 1) if i > 0 else 0
        print(f"\r[{i+1}/{len(eval_data)}] {query['query_id']}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", end="", flush=True)

        try:
            result_df = service.recommend(
                symptom_text=query["symptom_text"],
                diseases=query["diseases"],
                symptoms=query["symptoms"],
                top_k=max(args.k_values),
            )

            recommended = result_df["drug_name"].tolist()
            results.append({
                "query_id": query["query_id"],
                "recommended": recommended,
                "relevant": query["relevant_drugs"],
                "relevance_scores": query.get("relevance_scores", {}),
            })
        except Exception as e:
            print(f"\n  ERROR on {query['query_id']}: {e}")
            continue

    total_time = time.time() - start_time
    print(f"\n\nDone! {len(results)} queries evaluated in {total_time:.1f}s "
          f"({total_time/max(len(results),1):.1f}s/query)\n")

    # 计算指标
    metrics = evaluate_batch(results, k_values=args.k_values)

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name:25s}: {value:.4f}")

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": metrics,
                "num_queries": len(results),
                "k_values": args.k_values,
                "total_time_sec": total_time,
                "avg_time_per_query_sec": total_time / max(len(results), 1),
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
