"""Configuration module and LLM factory."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel

from paper_qa_lang.config.settings import Settings


def get_chat_model(
    settings: Settings | None = None,
    **overrides: Any,
) -> BaseChatModel:
    """Build a LangChain chat model from settings.

    Args:
        settings: Settings instance (uses defaults if None).
        **overrides: Override any ``LLMSettings`` field (e.g. ``model_name="..."``).

    Returns:
        A configured ``BaseChatModel`` instance.
    """
    s = (settings or Settings()).llm

    # Apply overrides
    provider = overrides.pop("provider", s.provider)
    model_name = overrides.pop("model_name", s.model_name)
    api_key = overrides.pop("api_key", s.api_key) or os.environ.get(
        _env_key_for(provider), None
    )
    base_url = overrides.pop("base_url", s.base_url)
    temperature = overrides.pop("temperature", s.temperature)
    max_tokens = overrides.pop("max_tokens", s.max_tokens)

    kw: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        **overrides,
    }
    if api_key:
        kw["api_key"] = api_key
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    if base_url:
        kw["base_url"] = base_url

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**kw)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**kw)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**kw)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _env_key_for(provider: str) -> str:
    mapping = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    return mapping.get(provider, f"{provider.upper()}_API_KEY")
