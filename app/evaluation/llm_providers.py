"""
LLM provider factory.

Supports OpenAI-compatible providers: DeepSeek, MiniMax, OpenAI.
All providers share the existing `LLMClient` interface from `llm_client.py` —
only base_url / default model / env var names differ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Note: LLMClient is imported lazily inside make_client() to avoid requiring
# the `openai` package when callers only need add_provider_args() / PROVIDERS.


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str | None
    default_model: str
    env_key: str


PROVIDERS: dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-v4-flash",
        env_key="DEEPSEEK_API_KEY",
    ),
    "minimax": ProviderConfig(
        name="minimax",
        base_url="https://api.minimaxi.com/v1",
        default_model="MiniMax-M2.5",
        env_key="MINIMAX_API_KEY",
    ),
    "openai": ProviderConfig(
        name="openai",
        base_url=None,
        default_model="gpt-4o-mini",
        env_key="OPENAI_API_KEY",
    ),
}


def make_client(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
):
    """Create an LLMClient bound to the requested provider.

    Resolution order for each field:
      - explicit argument
      - provider-specific env var
      - provider default (for model/base_url)
    """
    key = provider.lower().strip()
    if key not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {sorted(PROVIDERS)}"
        )
    cfg = PROVIDERS[key]

    resolved_key = api_key or os.getenv(cfg.env_key)
    if not resolved_key and key == "minimax":
        resolved_key = os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError(
            f"API key for provider '{key}' not found. "
            f"Set {cfg.env_key} or pass --api-key."
        )
    resolved_model = model or os.getenv(f"{cfg.env_key.replace('_API_KEY', '')}_MODEL") or cfg.default_model
    resolved_base_url = base_url or cfg.base_url

    try:
        from .llm_client import LLMClient
    except ImportError:
        from llm_client import LLMClient  # type: ignore[no-redef]

    return LLMClient(
        api_key=resolved_key,
        base_url=resolved_base_url,
        model=resolved_model,
    )


def add_provider_args(parser) -> None:
    """Shared argparse flags for any CLI that needs an LLM."""
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="minimax",
        help="LLM provider (default: minimax).",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None, help="Override default model.")
    parser.add_argument("--base-url", default=None, help="Override API base URL.")
