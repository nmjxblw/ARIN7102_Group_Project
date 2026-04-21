"""
LLM Query Expander — 医学术语扩展模块

在 semantic_recall 前将用户口语描述翻译为医学术语，弥补词汇鸿沟。
使用 OpenAI 兼容 API（DeepSeek / GPT 等）。
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from pipeline_config import cfg

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a medical terminology expert. "
    "Given a patient's symptom description in plain language, "
    "extract and list the most relevant medical/clinical terms "
    "(disease names, symptom names, anatomical terms). "
    "Output ONLY a comma-separated list of English medical terms, nothing else. "
    "If the input already contains medical terms, include them and add related terms. "
    "Limit to 10-15 terms maximum."
)

_client = None


def _get_client():
    """Lazy-init OpenAI client."""
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
        )
    return _client


def expand_query_with_llm(symptom_text: str, timeout: float = 10.0) -> str:
    """将口语症状描述扩展为包含医学术语的富文本。

    Args:
        symptom_text: 用户原始口语描述
        timeout: API 超时（秒）

    Returns:
        扩展后的文本: "{原始文本} Medical terms: term1, term2, ..."
        如果 LLM 调用失败，返回原始文本（降级）
    """
    symptom_text = str(symptom_text or "").strip()
    if not symptom_text or symptom_text == "[MASK]":
        return symptom_text

    try:
        t0 = time.perf_counter()
        client = _get_client()
        response = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": symptom_text},
            ],
            max_tokens=150,
            temperature=0.0,
            timeout=timeout,
        )
        medical_terms = response.choices[0].message.content.strip()
        elapsed = time.perf_counter() - t0
        logger.info(f"[QueryExpander] {elapsed:.2f}s | terms: {medical_terms[:80]}")

        if medical_terms:
            return f"{symptom_text} Medical terms: {medical_terms}"
        return symptom_text

    except Exception as e:
        logger.warning(f"[QueryExpander] LLM call failed, fallback to raw query: {e}")
        return symptom_text
