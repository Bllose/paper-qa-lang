"""Shared utilities."""
from __future__ import annotations

from typing import Any

from paper_qa_lang.utils.hash import md5sum  # noqa: F401


def llm_content_text(content: Any) -> str:
    """Extract text from an LLM response content, handling both str and list formats.

    LangChain's ``ChatAnthropic`` may return ``content`` as a plain string or as a
    list of content blocks (e.g. ``[{"type": "text", "text": "..."}, {"type": "thinking", ...}]``
    for models with extended thinking).

    Args:
        content: ``response.content`` from a LangChain chat model.

    Returns:
        Extracted text content (empty string if none found).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return ""
