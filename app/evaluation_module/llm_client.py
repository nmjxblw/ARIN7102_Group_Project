"""
LLM client for generating natural language queries.
Supports OpenAI-compatible APIs.
"""

from __future__ import annotations

import os
import re
from typing import Any

import openai


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks emitted inline by reasoning models
    (e.g. MiniMax-M2.5). If the closing tag is missing (truncation), drop
    everything up to and including the last `</think>` if present, else
    return as-is."""
    if "<think>" not in text:
        return text.strip()
    cleaned = _THINK_RE.sub("", text)
    # Handle unclosed <think> (truncated): keep portion after last </think>.
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned.strip()


class LLMClient:
    """OpenAI-compatible LLM client."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-3.5-turbo",
        extra_body: dict | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.extra_body = extra_body or {}

        if not self.api_key:
            raise ValueError(
                "API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 200,
    ) -> str:
        """Generate text from prompt."""
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content: Any = msg.content or getattr(msg, "reasoning_content", "") or ""
        return _strip_reasoning(str(content))
