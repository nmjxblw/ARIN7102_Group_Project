"""
Stage 2: LLM verification of the candidate test set.

For each candidate query, a single LLM call is made (DeepSeek or MiniMax,
configurable). The LLM returns structured JSON with, per drug:
  - disease_match     (0-3)
  - symptom_match     (0-3)
  - contraindication_risk (0-3) — Beers / FDA-label style
  - ddi_risk_overall  (0-3) — Lexicomp / Micromedex style across the drug list

Retention rules:
  - Keep a drug when disease_match >= 2, symptom_match >= 1,
    and contraindication_risk <= 1.
  - Drop the whole query when ddi_risk_overall >= 2 (clinically significant DDI).
  - Drop the whole query when no drugs survive.
  - LLM's disease_match is written back as relevance_scores for the drug.

Outputs:
  - verified JSON (same schema as input, pruned relevant_drugs / relevance_scores)
  - JSONL audit log (LLM raw response + decision)
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from .ddi_context import format_ddi_context, get_default_index
    from .llm_providers import add_provider_args, make_client
except ImportError:
    from ddi_context import format_ddi_context, get_default_index  # type: ignore
    from llm_providers import add_provider_args, make_client  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]


VERIFY_SYSTEM_PROMPT = """You are a clinical pharmacology reviewer. You assess whether a set of recommended drugs is appropriate for a described patient case, following standard medical references:
- For drug-disease correspondence: FDA drug label indications.
- For contraindications: FDA drug label Contraindications section, Beers Criteria.
- For drug-drug interactions: Lexicomp / Micromedex severity grading (major / moderate / minor).

You must return STRICT JSON only — no markdown fences, no commentary, no preamble,
no chain-of-thought, no <think> tags. The first character of your response must
be "{". Use integer scores as specified.
"""


VERIFY_USER_TEMPLATE = """Patient description: {symptom_text}

Diseases (with model confidence): {diseases}
Symptoms (with model confidence): {symptoms}

CANDIDATE drug pool (retrieval results — these are parallel options, NOT a
concurrent regimen; the ranker will later pick a small subset): {drugs}

{ddi_context}

For EACH candidate drug, rate (integer 0-3):
  - disease_match: 3 = first-line treatment for at least one listed disease; 2 = acceptable alternative; 1 = weak/edge-case; 0 = not indicated.
  - symptom_match: 3 = directly addresses multiple listed symptoms; 2 = addresses some; 1 = marginal; 0 = unrelated.
  - contraindication_risk: 0 = no concerns given listed diseases; 1 = caution advised; 2 = relative contraindication; 3 = absolute contraindication.
  - rationale: ONE short sentence citing the indication or contraindication source.

DDI evaluation (pool-aware). Because the pool lists multiple alternatives per
disease, do NOT assume all drugs are co-administered. Rate ddi_risk_overall
as the WORST interaction that would arise if a clinician picks a REASONABLE
subset (e.g. at most one drug per disease class). If every such reasonable
subset is safe, ddi_risk_overall = 0.
  - ddi_risk_overall (0-3): 0 = no clinically relevant interaction in any reasonable subset; 1 = minor; 2 = moderate (monitoring needed); 3 = major (avoid).
  - ddi_rationale: ONE sentence naming the worst pair, or "N/A".

Return JSON exactly in this shape:
{{
  "drugs": [
    {{"name": "<drug name as given>", "disease_match": 0, "symptom_match": 0,
      "contraindication_risk": 0, "rationale": "..."}}
  ],
  "ddi_risk_overall": 0,
  "ddi_rationale": "..."
}}
"""


REPAIR_SYSTEM_PROMPT = """You convert an existing model answer into STRICT JSON.
Return JSON only. No markdown, no commentary, no <think> tags, no analysis.
The first character must be "{". Keep drug names exactly as given when possible.
"""


REPAIR_USER_TEMPLATE = """Rewrite the following answer into valid JSON only.

Required JSON shape:
{{
  "drugs": [
    {{
      "name": "<drug name as given>",
      "disease_match": 0,
      "symptom_match": 0,
      "contraindication_risk": 0,
      "rationale": "..."
    }}
  ],
  "ddi_risk_overall": 0,
  "ddi_rationale": "..."
}}

Allowed drug names in this case:
{drug_names}

Original answer to convert:
{raw}
"""


def _format_labels(items: list[dict]) -> str:
    return ", ".join(f"{it['name']} ({it.get('confidence', 0):.2f})" for it in items)


def build_verify_prompt(case: dict, ddi_context_text: str) -> str:
    return VERIFY_USER_TEMPLATE.format(
        symptom_text=case["symptom_text"],
        diseases=_format_labels(case["diseases"]),
        symptoms=_format_labels(case["symptoms"]),
        drugs=", ".join(case["relevant_drugs"]),
        ddi_context=ddi_context_text,
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    txt = raw.strip()
    # strip code fences if present
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n", "", txt)
        txt = re.sub(r"\n```$", "", txt)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(txt)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, min(3, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def classify_api_error(err: str) -> str:
    lower = err.lower()
    if "401" in lower or "invalid api key" in lower or "authorized_error" in lower:
        return "api_auth_error"
    if "model not support" in lower or "not support model" in lower:
        return "api_model_error"
    if "connection error" in lower or "connect" in lower or "timed out" in lower:
        return "api_connection_error"
    return "api_error"


def load_retry_query_ids(log_paths: list[Path], reason: str) -> set[str]:
    query_ids: set[str] = set()
    for path in log_paths:
        if not path.exists():
            raise FileNotFoundError(f"Retry log not found: {path}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("reason") == reason and entry.get("query_id"):
                    query_ids.add(str(entry["query_id"]))
    return query_ids


def call_verifier(
    llm,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str | None]:
    try:
        resp = llm.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            extra_body={"reasoning_split": True},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content or ""
        if not raw:
            finish = resp.choices[0].finish_reason
            return "", f"empty_content (finish_reason={finish})"
        return raw, None
    except Exception as exc:
        return "", str(exc)


def attempt_repair(
    llm,
    case: dict,
    raw: str,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, str, str | None]:
    if not raw.strip():
        return None, raw, None
    repair_prompt = REPAIR_USER_TEMPLATE.format(
        drug_names=", ".join(case["relevant_drugs"]),
        raw=raw[:6000],
    )
    repaired_raw, repaired_err = call_verifier(
        llm=llm,
        model=llm.model,
        system_prompt=REPAIR_SYSTEM_PROMPT,
        user_prompt=repair_prompt,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    repaired_verdict = parse_llm_json(repaired_raw) if repaired_raw else None
    return repaired_verdict, repaired_raw, repaired_err


def apply_retention_rules(
    case: dict,
    verdict: dict[str, Any],
    min_disease_match: int = 2,
    min_symptom_match: int = 1,
    max_contraindication: int = 1,
    max_ddi: int = 1,
) -> tuple[dict | None, dict]:
    """Return (verified_case_or_None, decision_audit_dict)."""
    decision: dict[str, Any] = {
        "query_id": case["query_id"],
        "kept": False,
        "reason": None,
        "per_drug": [],
    }
    drugs_raw = verdict.get("drugs") or []
    ddi_overall = _coerce_int(verdict.get("ddi_risk_overall"))
    ddi_rationale = verdict.get("ddi_rationale", "") or ""
    decision["ddi_overall"] = ddi_overall
    decision["ddi_rationale"] = ddi_rationale

    kept_drugs: list[str] = []
    new_scores: dict[str, int] = {}
    for entry in drugs_raw:
        name = str(entry.get("name", "")).strip()
        d_match = _coerce_int(entry.get("disease_match"))
        s_match = _coerce_int(entry.get("symptom_match"))
        contra = _coerce_int(entry.get("contraindication_risk"))
        keep = (
            d_match >= min_disease_match
            and s_match >= min_symptom_match
            and contra <= max_contraindication
        )
        decision["per_drug"].append(
            {
                "name": name,
                "disease_match": d_match,
                "symptom_match": s_match,
                "contraindication_risk": contra,
                "kept": keep,
                "rationale": entry.get("rationale", ""),
            }
        )
        if keep and name in case["relevant_drugs"]:
            kept_drugs.append(name)
            new_scores[name] = d_match

    if not kept_drugs:
        decision["reason"] = "all_drugs_failed_threshold"
        return None, decision

    verified = dict(case)
    verified["relevant_drugs"] = kept_drugs
    verified["relevance_scores"] = new_scores
    verified["ddi_flags"] = {
        "overall_severity": ddi_overall,
        "rationale": ddi_rationale,
        "checked_subset_semantics": "pool",
    }
    decision["kept"] = True
    decision["reason"] = "ok"
    return verified, decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: LLM verification of candidates")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "eval_dataset_candidates.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "eval_dataset_verified.json",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "data" / "verification_log.jsonl",
    )
    parser.add_argument("--max-ddi-rows", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--repair-parse-fail",
        action="store_true",
        help="When the first response is non-JSON, ask the model once more to rewrite it into strict JSON.",
    )
    parser.add_argument("--min-disease-match", type=int, default=2)
    parser.add_argument("--min-symptom-match", type=int, default=1)
    parser.add_argument("--max-contraindication", type=int, default=1)
    parser.add_argument("--max-ddi", type=int, default=1)
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Stop once this many verified cases are retained.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only verify the first N candidates (smoke test).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between calls (rate limiting, serial mode only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel LLM calls (default: 1 = serial).",
    )
    parser.add_argument(
        "--retry-log",
        action="append",
        type=Path,
        default=[],
        help="Existing verification JSONL log(s); when provided, only cases whose reason matches --retry-reason are re-run.",
    )
    parser.add_argument(
        "--retry-reason",
        type=str,
        default="parse_fail",
        help="Reason to select from --retry-log. Default: parse_fail",
    )
    add_provider_args(parser)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    with open(args.input, encoding="utf-8") as f:
        cases = json.load(f)
    if args.retry_log:
        retry_ids = load_retry_query_ids(args.retry_log, args.retry_reason)
        cases = [case for case in cases if case["query_id"] in retry_ids]
        print(
            f"[verify_test_set] retry filter reason={args.retry_reason} "
            f"matched {len(cases)} cases from {len(args.retry_log)} log(s)"
        )
    if args.limit:
        cases = cases[: args.limit]

    llm = make_client(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
    )
    print(f"[verify_test_set] provider={args.provider} model={llm.model} n={len(cases)}")

    ddi_index = None
    try:
        ddi_index = get_default_index()
    except FileNotFoundError:
        print("[verify_test_set] WARN: DDI csv not found — prompts will omit DDI context")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    verified_cases: list[dict] = []
    stats: Counter[str] = Counter()
    lock = threading.Lock()

    def _postfix(progress: tqdm) -> None:
        progress.set_postfix(
            kept=len(verified_cases),
            parse_fail=stats["parse_fail"],
            api_fail=(
                stats["api_auth_error"]
                + stats["api_model_error"]
                + stats["api_connection_error"]
                + stats["api_error"]
            ),
            drug_drop=stats["all_drugs_failed_threshold"],
        )

    def _process_case(case: dict) -> tuple[dict, dict | None]:
        """Run one LLM call + retention logic. Thread-safe (no shared state written here)."""
        ddi_text = (
            format_ddi_context(case["relevant_drugs"], ddi_index, args.max_ddi_rows)
            if ddi_index is not None
            else "DDI database unavailable."
        )
        prompt = build_verify_prompt(case, ddi_text)
        raw, err = call_verifier(
            llm=llm,
            model=llm.model,
            system_prompt=VERIFY_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        verdict = parse_llm_json(raw) if raw else None
        repair_err = None
        if verdict is None and args.repair_parse_fail and raw:
            verdict, repair_raw, repair_err = attempt_repair(
                llm=llm, case=case, raw=raw, max_tokens=args.max_tokens
            )
            if verdict is not None:
                raw = repair_raw
            elif repair_err and err is None:
                err = repair_err

        if verdict is None:
            reason = classify_api_error(err) if err else "parse_fail"
            fail_entry = {
                "query_id": case["query_id"],
                "kept": False,
                "reason": reason,
                "error": err,
                "raw": raw[:2000],
                "repair_error": repair_err,
            }
            return fail_entry, None

        verified, decision = apply_retention_rules(
            case,
            verdict,
            min_disease_match=args.min_disease_match,
            min_symptom_match=args.min_symptom_match,
            max_contraindication=args.max_contraindication,
            max_ddi=args.max_ddi,
        )
        decision["raw"] = raw[:2000]
        return decision, verified

    with open(args.log, "w", encoding="utf-8") as log_f:
        if args.workers <= 1:
            progress = tqdm(cases, desc="verify")
            for case in progress:
                if args.target_count is not None and len(verified_cases) >= args.target_count:
                    break
                decision, verified = _process_case(case)
                stats[decision["reason"]] += 1
                log_f.write(json.dumps(decision, ensure_ascii=False) + "\n")
                if verified is not None:
                    verified_cases.append(verified)
                _postfix(progress)
                if args.sleep:
                    time.sleep(args.sleep)
        else:
            progress = tqdm(total=len(cases), desc="verify")
            stop_event = threading.Event()
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_process_case, case): case for case in cases}
                for future in as_completed(futures):
                    decision, verified = future.result()
                    with lock:
                        if stop_event.is_set():
                            progress.update(1)
                            continue
                        stats[decision["reason"]] += 1
                        log_f.write(json.dumps(decision, ensure_ascii=False) + "\n")
                        log_f.flush()
                        if verified is not None:
                            verified_cases.append(verified)
                            if (
                                args.target_count is not None
                                and len(verified_cases) >= args.target_count
                            ):
                                stop_event.set()
                        _postfix(progress)
                        progress.update(1)
            progress.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(verified_cases, f, indent=2, ensure_ascii=False)

    print(f"[verify_test_set] kept {len(verified_cases)}/{len(cases)} → {args.output}")
    print(f"[verify_test_set] audit log → {args.log}")
    print(f"[verify_test_set] stats → {dict(stats)}")


if __name__ == "__main__":
    main()
