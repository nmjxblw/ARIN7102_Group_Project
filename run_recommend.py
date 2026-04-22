"""
Drug Recommendation - 一键推荐脚本
在项目根目录直接运行: python run_recommend.py

用法:
    python run_recommend.py
    python run_recommend.py --query "I have a headache and feel dizzy"
    python run_recommend.py --query "..." --use-bert
    python run_recommend.py --query "..." --diseases "acne:1.0" --symptoms "nodal_skin_eruptions:1.0,scurring:1.0"
"""
import sys
import os
import argparse
import json
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


# ─── 工具函数 ─────────────────────────────────────────────────
def parse_label_string(label_str: str) -> list[dict]:
    """解析 'name1:conf1,name2:conf2' 格式的标签字符串"""
    if not label_str:
        return []
    labels = []
    for item in label_str.split(","):
        item = item.strip()
        if ":" in item:
            name, conf = item.rsplit(":", 1)
            labels.append({"name": name.strip(), "confidence": float(conf)})
        else:
            labels.append({"name": item, "confidence": 1.0})
    return labels


def init_service():
    """初始化推荐服务（可供外部 import 使用）"""
    from fastapi_module.service import get_recommendation_service
    service = get_recommendation_service()
    service.ensure_ready()
    return service


# ─── 主函数 ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Drug Recommendation Pipeline - 一键推荐")
    parser.add_argument(
        "--query", type=str,
        default="I have big, painful bumps on my skin that leave scars. How can I treat this?",
        help="症状描述文本"
    )
    parser.add_argument("--diseases", type=str, default="", help="疾病标签 (格式: name:conf,name:conf)")
    parser.add_argument("--symptoms", type=str, default="", help="症状标签 (格式: name:conf,name:conf)")
    parser.add_argument("--use-bert", action="store_true", help="启用 BERT 自动预测标签")
    parser.add_argument("--top-k", type=int, default=10, help="返回药物数量")
    parser.add_argument("--no-trace", action="store_true", help="禁用 pipeline trace")
    args = parser.parse_args()

    diseases = parse_label_string(args.diseases)
    symptoms = parse_label_string(args.symptoms)
    enable_trace = not args.no_trace

    print("=" * 60)
    print("Drug Recommendation Pipeline")
    print("=" * 60)
    print(f"Query:    {args.query}")
    if args.use_bert:
        print(f"Mode:     BERT 自动预测标签")
    elif diseases or symptoms:
        print(f"Diseases: {diseases}")
        print(f"Symptoms: {symptoms}")
    else:
        print(f"Mode:     无标签（仅语义召回）")
    print()

    # 初始化
    print("Loading models...")
    service = init_service()
    print("Service ready.\n")

    # 推荐
    print("Running recommendation...")
    result = service.recommend(
        symptom_text=args.query,
        diseases=diseases,
        symptoms=symptoms,
        top_k=args.top_k,
        use_bert_prediction=args.use_bert,
        enable_trace=enable_trace,
    )

    if enable_trace:
        result_df, trace = result
    else:
        result_df = result
        trace = None

    # 输出结果
    print("\n" + "=" * 60)
    print(f"TOP-{args.top_k} RECOMMENDED DRUGS")
    print("=" * 60)

    display_cols = ["drug_name", "final_score", "semantic_score", "label_score", "cross_encoder_score", "business_score"]
    available_cols = [c for c in display_cols if c in result_df.columns]
    print(result_df[available_cols].to_string(index=False))

    if trace:
        print("\n" + "=" * 60)
        print("PIPELINE TRACE")
        print("=" * 60)
        trace_dict = trace.to_dict()
        # 按类别分组打印
        print("\n[Timing]")
        for k, v in trace_dict.items():
            if k.startswith("time_"):
                print(f"  {k}: {v:.1f} ms")
        print("\n[Candidates]")
        for k, v in trace_dict.items():
            if k.startswith("count_"):
                print(f"  {k}: {v}")
        print("\n[Scores]")
        for k, v in trace_dict.items():
            if k.startswith("score_"):
                print(f"  {k}: {v:.4f}")
        print("\n[Input]")
        for k, v in trace_dict.items():
            if k.startswith("query_") or k.startswith("num_"):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
