"""
为 others 类别的 condition 使用 LLM 批量生成症状列表

功能:
    1. 从 enhanced_drug_table.csv 中收集所有 matched_disease_keys 为 ["others"] 的 condition
    2. 对每个 unique condition, 调用 LLM 生成常见症状列表 (JSON 格式)
    3. 将结果保存为 others_condition_symptoms.json 映射文件
    4. 支持断点续跑: 如果映射文件已存在, 只处理尚未生成的 condition
    5. 支持 --retry-empty: 重新请求映射为空的 condition

使用方式:
    # 首次运行 (推荐单条 + 并发)
    python generate_others_symptoms.py --api-key YOUR_KEY --batch-size 1 --workers 5

    # 重跑空映射
    python generate_others_symptoms.py --api-key YOUR_KEY --retry-empty --batch-size 1 --workers 5

    # 可选参数:
    --base-url     API地址 (默认 https://api.deepseek.com)
    --model        模型名称 (默认 deepseek-chat)
    --batch-size   每次发送的condition数量 (默认 1, 单条更快)
    --workers      并发线程数 (默认 5, 仅 batch-size=1 时生效)
    --max-retries  最大重试次数 (默认 3)
    --retry-empty  重新请求映射为空的 condition
    --output       输出文件路径

输出:
    match_data_preprocessing/data/others_condition_symptoms.json
"""

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 尝试导入 openai ----------
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] 缺少 openai 库, 请先安装: pip install openai")
    sys.exit(1)

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PREPROCESS_DIR = PROJECT_ROOT / "match_data_preprocessing"
DATA_DIR = PREPROCESS_DIR / "data"
ENHANCED_TABLE = DATA_DIR / "enhanced_drug_table.csv"
DEFAULT_OUTPUT = DATA_DIR / "others_condition_symptoms.json"

# ============================================================
# Prompt 模板
# ============================================================
SYSTEM_PROMPT = """You are a medical knowledge expert. Your task is to generate common symptoms for given medical conditions.

Rules:
1. Return ONLY valid JSON, no extra text.
2. For each condition, provide 5-15 common symptoms.
3. Use standardized, lowercase English symptom names with underscores (e.g. "chest_pain", "high_fever", "skin_rash").
4. If a condition is not a real medical condition (e.g. "Not Listed / Othe", "users found this comment helpful"), return an empty list [].
5. Focus on symptoms that a patient would describe, not clinical signs.
"""

# batch 模式 prompt (多个 condition)
USER_PROMPT_BATCH = """For each of the following medical conditions, provide a JSON object mapping condition name to a list of common symptoms.

Conditions:
{conditions}

Return format (JSON only, no markdown fences):
{{
  "condition_name_1": ["symptom_1", "symptom_2", ...],
  "condition_name_2": ["symptom_1", "symptom_2", ...],
  ...
}}"""

# 单条模式 prompt (更简短, LLM 响应更快)
USER_PROMPT_SINGLE = """Medical condition: "{condition}"

Return a JSON object with the condition name as key and a list of common symptoms as value.
JSON only, no markdown fences:
{{"{condition}": ["symptom_1", "symptom_2", ...]}}"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 LLM 为 others 类别的 condition 批量生成症状"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API Key (也可通过环境变量 LLM_API_KEY 设置)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.deepseek.com",
        help="API Base URL (默认: https://api.deepseek.com)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="模型名称 (默认: deepseek-chat)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="每次 API 调用处理的 condition 数量 (默认: 1, 单条更快)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="并发线程数 (默认: 5, 仅 batch-size=1 时生效)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="API 调用失败的最大重试次数 (默认: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"输出文件路径 (默认: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="每次 API 调用之间的等待秒数 (默认: 0.3)",
    )
    parser.add_argument(
        "--retry-empty",
        action="store_true",
        help="重新请求映射为空列表 [] 的 condition",
    )
    return parser.parse_args()


def load_others_conditions() -> list[str]:
    """从 enhanced_drug_table.csv 中提取所有 others 药物的 unique condition"""
    import pandas as pd

    df = pd.read_csv(ENHANCED_TABLE)
    others = df[df["matched_disease_keys"] == '["others"]']
    all_conds = set()
    for c in others["original_conditions"].dropna():
        try:
            conds = json.loads(c)
            for cond in conds:
                cond_clean = cond.strip()
                if cond_clean and cond_clean.lower() != "nan":
                    all_conds.add(cond_clean)
        except (json.JSONDecodeError, TypeError):
            pass

    # 过滤掉明显不是疾病的条目
    filtered = []
    for c in sorted(all_conds):
        if "users found this comment helpful" in c.lower():
            continue
        if len(c) < 2:
            continue
        filtered.append(c)

    return filtered


def load_existing_mapping(output_path: str) -> dict:
    """加载已存在的映射文件 (用于断点续跑)"""
    p = Path(output_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[INFO] 加载已有映射文件: {len(data)} 条记录")
            return data
        except (json.JSONDecodeError, Exception) as e:
            print(f"[WARN] 映射文件读取失败 ({e}), 将从头开始")
    return {}


# 文件写入锁 (多线程安全)
_save_lock = threading.Lock()


def save_mapping(mapping: dict, output_path: str):
    """保存映射文件 (线程安全)"""
    with _save_lock:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)


def call_llm_single(
    client: OpenAI,
    model: str,
    condition: str,
    max_retries: int = 3,
) -> Optional[list]:
    """
    调用 LLM 为单个 condition 生成症状 (单条模式, 更快)。
    返回 [symptoms] 或 None。
    """
    user_prompt = USER_PROMPT_SINGLE.format(condition=condition)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            content = response.choices[0].message.content.strip()

            # 清理 markdown 围栏
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            result = json.loads(content)

            if isinstance(result, dict):
                # 取第一个 (也是唯一一个) value
                for k, v in result.items():
                    if isinstance(v, list):
                        return [str(s).strip() for s in v if str(s).strip()]
                    return []
            elif isinstance(result, list):
                return [str(s).strip() for s in result if str(s).strip()]

        except json.JSONDecodeError as e:
            if attempt == max_retries:
                print(f"    [WARN] {condition}: JSON 解析失败 ({e})")
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"    [WARN] {condition}: API 失败 ({e})")

    return None


def call_llm_batch(
    client: OpenAI,
    model: str,
    conditions: list[str],
    max_retries: int = 3,
) -> Optional[dict]:
    """
    调用 LLM 为一批 condition 生成症状 (批量模式)。
    返回 {condition: [symptoms]} 或 None。
    """
    conditions_text = "\n".join(f"- {c}" for c in conditions)
    user_prompt = USER_PROMPT_BATCH.format(conditions=conditions_text)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            content = response.choices[0].message.content.strip()

            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            result = json.loads(content)

            if isinstance(result, dict):
                cleaned = {}
                for k, v in result.items():
                    if isinstance(v, list):
                        cleaned[k] = [str(s).strip() for s in v if str(s).strip()]
                    else:
                        cleaned[k] = []
                return cleaned
            else:
                print(f"  [WARN] 返回非 dict 类型, 重试 ({attempt}/{max_retries})")

        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON 解析失败: {e}, 重试 ({attempt}/{max_retries})")
            if attempt == max_retries:
                print(f"  [ERROR] 原始返回内容:\n{content[:500]}")
        except Exception as e:
            print(f"  [WARN] API 调用失败: {e}, 重试 ({attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    return None


def run_single_mode(client, model, todo, mapping, args):
    """单条 + 多线程并发模式"""
    success_count = 0
    fail_count = 0
    total = len(todo)
    done = 0
    _counter_lock = threading.Lock()

    def process_one(condition: str):
        nonlocal success_count, fail_count, done
        symptoms = call_llm_single(client, model, condition, max_retries=args.max_retries)

        with _counter_lock:
            done += 1
            if symptoms is not None and len(symptoms) > 0:
                mapping[condition] = symptoms
                success_count += 1
                preview = symptoms[:3]
                print(f"  [{done}/{total}] [OK] {condition}: {preview}{'...' if len(symptoms) > 3 else ''}")
            elif symptoms is not None:
                # LLM 认为这不是疾病, 返回空列表
                mapping[condition] = []
                success_count += 1
                print(f"  [{done}/{total}] [OK] {condition}: [] (非疾病条目)")
            else:
                # API 彻底失败, 不写入映射 (下次可重试)
                fail_count += 1
                print(f"  [{done}/{total}] [FAIL] {condition}")

            # 每 10 条保存一次
            if done % 10 == 0:
                save_mapping(mapping, args.output)

    workers = min(args.workers, len(todo))
    print(f"  并发线程数: {workers}")
    print()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for cond in todo:
            f = executor.submit(process_one, cond)
            futures[f] = cond
            time.sleep(args.delay)  # 提交间隔, 避免瞬间并发太多

        # 等待全部完成
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                cond = futures[f]
                print(f"  [ERROR] {cond}: 线程异常 ({e})")

    # 最终保存
    save_mapping(mapping, args.output)
    return success_count, fail_count


def run_batch_mode(client, model, todo, mapping, args):
    """批量模式 (batch_size > 1)"""
    success_count = 0
    fail_count = 0

    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(todo) - 1) // args.batch_size + 1

        print(f"\n  --- Batch {batch_num}/{total_batches} ({len(batch)} conditions) ---")
        for c in batch:
            print(f"    - {c}")

        result = call_llm_batch(client, model, batch, max_retries=args.max_retries)

        if result:
            for cond in batch:
                matched_value = result.get(cond, None)
                if matched_value is None:
                    for k, v in result.items():
                        if k.lower().strip() == cond.lower().strip():
                            matched_value = v
                            break
                if matched_value is not None:
                    mapping[cond] = matched_value
                    success_count += 1
                    preview = matched_value[:5] if len(matched_value) > 5 else matched_value
                    print(f"    [OK] {cond}: {preview}{'...' if len(matched_value) > 5 else ''}")
                else:
                    mapping[cond] = []
                    fail_count += 1
                    print(f"    [--] {cond}: LLM 未返回, 设为空列表")

            save_mapping(mapping, args.output)
            print(f"  [SAVE] 已保存 {len(mapping)} 条映射")
        else:
            fail_count += len(batch)
            # 批量失败时不写入, 让下次可以重试
            save_mapping(mapping, args.output)
            print(f"  [FAIL] 本批次失败")

        if i + args.batch_size < len(todo):
            time.sleep(args.delay)

    return success_count, fail_count


def main():
    args = parse_args()

    # 解析 API Key
    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        print("=" * 60)
        print("[ERROR] 未提供 API Key!")
        print()
        print("请通过以下方式之一提供:")
        print("  1. 命令行参数: python generate_others_symptoms.py --api-key YOUR_KEY")
        print("  2. 环境变量:   set LLM_API_KEY=YOUR_KEY")
        print("=" * 60)
        sys.exit(1)

    mode_str = "单条并发" if args.batch_size == 1 else f"批量(batch={args.batch_size})"
    print("=" * 60)
    print("LLM 批量生成 others 条件的症状")
    print("=" * 60)
    print(f"  API Base URL : {args.base_url}")
    print(f"  Model        : {args.model}")
    print(f"  模式         : {mode_str}")
    if args.batch_size == 1:
        print(f"  并发线程     : {args.workers}")
    else:
        print(f"  Batch Size   : {args.batch_size}")
    print(f"  Max Retries  : {args.max_retries}")
    print(f"  Delay        : {args.delay}s")
    print(f"  Retry Empty  : {args.retry_empty}")
    print(f"  Output       : {args.output}")
    print()

    # 初始化 LLM 客户端
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # Step 1: 收集 others conditions
    print("[Step 1] 收集 others conditions ...")
    all_conditions = load_others_conditions()
    print(f"  共 {len(all_conditions)} 个 unique condition")

    # Step 2: 加载已有映射 (断点续跑)
    print("[Step 2] 检查已有映射文件 ...")
    existing = load_existing_mapping(args.output)

    # 过滤出需要处理的 condition
    if args.retry_empty:
        # 重试模式: 处理不在映射中的 + 映射为空列表的
        empty_keys = {k for k, v in existing.items() if not v}
        todo = [c for c in all_conditions if c not in existing or c in empty_keys]
        # 把空映射从 existing 中移除, 让它们重新被处理
        for k in empty_keys:
            existing.pop(k, None)
        print(f"  已有非空映射: {len(existing)}")
        print(f"  需重试(空映射): {len(empty_keys)}")
        print(f"  新增待处理:    {len(todo) - len(empty_keys)}")
        print(f"  总待处理:      {len(todo)}")
    else:
        todo = [c for c in all_conditions if c not in existing]
        print(f"  已完成: {len(existing)}, 待处理: {len(todo)}")

    if not todo:
        print("\n[DONE] 所有 condition 已生成完毕, 无需再次运行!")
        if args.retry_empty:
            empty_in_file = sum(1 for v in existing.values() if not v)
            print(f"  (其中空映射: {empty_in_file} 条, 可能是非疾病条目)")
        print(f"  映射文件: {args.output}")
        return

    # Step 3: 调用 LLM
    mapping = dict(existing)  # 复制已有非空结果
    total_todo = len(todo)

    if args.batch_size == 1:
        print(f"\n[Step 3] 单条并发模式 (共 {total_todo} 条, {args.workers} 线程) ...")
        success_count, fail_count = run_single_mode(client, args.model, todo, mapping, args)
    else:
        total_batches = (total_todo - 1) // args.batch_size + 1
        print(f"\n[Step 3] 批量模式 (共 {total_todo} 条, 分 {total_batches} 批) ...")
        success_count, fail_count = run_batch_mode(client, args.model, todo, mapping, args)

    # Step 4: 最终统计
    print("\n" + "=" * 60)
    print("[DONE] 生成完毕!")
    print(f"  总 condition 数:  {len(all_conditions)}")
    print(f"  本次处理:        {total_todo}")
    print(f"  成功:            {success_count}")
    print(f"  失败:            {fail_count}")
    print(f"  映射文件:        {args.output}")
    print()

    # 统计症状覆盖率
    has_symptoms = sum(1 for v in mapping.values() if v)
    empty_count = sum(1 for v in mapping.values() if not v)
    not_in_map = len(all_conditions) - len(mapping)
    print(f"  映射总条数:     {len(mapping)}")
    print(f"    有症状的:     {has_symptoms} ({has_symptoms/max(len(mapping),1)*100:.1f}%)")
    print(f"    空列表:       {empty_count}")
    print(f"  未在映射中:     {not_in_map}")

    total_symptoms = sum(len(v) for v in mapping.values())
    if has_symptoms > 0:
        print(f"  总症状条目:     {total_symptoms}")
        print(f"  平均每条:       {total_symptoms/has_symptoms:.1f} 个症状")

    if fail_count > 0:
        print()
        print(f"  [TIP] 有 {fail_count} 条失败, 可加 --retry-empty 重跑:")
        print(f"    python generate_others_symptoms.py --api-key YOUR_KEY --retry-empty")

    print()
    print("下一步: 运行 backfill_others_symptoms.py 将症状回填到 enhanced_drug_table.csv")


if __name__ == "__main__":
    main()
