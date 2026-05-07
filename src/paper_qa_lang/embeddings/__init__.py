"""Embedding model wrappers.

Usage::

    from paper_qa_lang.embeddings import get_embedding_model

    # Auto-detect from settings
    embedder = get_embedding_model()

    # Or specify model path
    embedder = get_embedding_model(
        model_name="bge-base-zh-v1.5",
        model_path="D:/workplace/models/BAAI/bge-base-zh-v1.5",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.embeddings import Embeddings

from paper_qa_lang.config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, type[Embeddings]] = {}


def register(name: str, cls: type[Embeddings]) -> None:
    """Register an embedding class for the given model name keyword."""
    _MODEL_REGISTRY[name.lower()] = cls


def get_embedding_model(
    model_name: str | None = None,
    model_path: str | None = None,
    settings: EmbeddingSettings | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Unified entry point for loading an embedding model.

    Resolution order:
        1. ``settings.provider`` — dispatch via provider name
        2. ``model_name`` — match against registered model keywords
        3. Default → ``HuggingFaceEmbeddings`` (sentence-transformers)

    Args:
        model_name: Model name / keyword (e.g. ``"bge-base-zh-v1.5"``).
        model_path: Local filesystem path to the model.
        settings: ``EmbeddingSettings`` object (falls back to defaults).
        **kwargs: Extra args forwarded to the embedding constructor.

    Returns:
        A LangChain ``Embeddings`` instance.
    """
    resolved = settings or EmbeddingSettings()
    name = (model_name or resolved.model_name).lower()
    path = model_path or resolved.model_path or kwargs.pop("model_path", None)

    # 1. By provider name
    provider = resolved.provider.lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=name, api_key=resolved.api_key or kwargs.pop("api_key", None)
        )

    # 2. By registered model keyword
    for keyword, cls in _MODEL_REGISTRY.items():
        if keyword in name:
            init_kwargs: dict[str, Any] = {}
            if path:
                init_kwargs["model_path"] = path
            init_kwargs.update(kwargs)
            return cls(**init_kwargs)

    # 3. Fallback: sentence-transformers
    from langchain_huggingface import HuggingFaceEmbeddings

    init_kw = {"model_name": name}
    if path:
        init_kw["model_name"] = path
    init_kw.update(kwargs)
    return HuggingFaceEmbeddings(**init_kw)


# ---- register known models ----

def _auto_register() -> None:
    try:
        from paper_qa_lang.embeddings.qwen_embedding import BgeEmbedding

        register("bge", BgeEmbedding)
    except ImportError:
        pass


_auto_register()
