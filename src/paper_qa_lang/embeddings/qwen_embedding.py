"""BAAI BGE embedding model wrapped as a LangChain Embeddings interface.

Loads the model directly via ``transformers`` + ``torch`` (no ``sentence-transformers``
dependency), matching the approach used in ``tests/bllose_person_test/person_test.py``.
Uses mean pooling + L2 normalization (BGE convention) and applies a query prefix to
distinguish queries from documents.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "D:/workplace/models/BAAI/bge-base-zh-v1.5"

# BGE中文指令前缀：区分查询和文档
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BgeEmbedding(Embeddings):
    """LangChain Embeddings wrapper around BAAI/bge-base-zh-v1.5.

    Uses ``AutoTokenizer`` + ``AutoModel`` with ``local_files_only=True``.
    Embeddings are computed by mean-pooling the last hidden state followed by
    L2 normalization, which is the standard BGE post-processing.
    The ``embed_query`` method prepends a retrieval instruction prefix.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model_path = model_path
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            **(model_kwargs or {}),
        }

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self._model = AutoModel.from_pretrained(model_path, **kwargs).to(
            self._device
        )
        self._model.eval()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        inputs = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # L2 normalization (BGE convention)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """BGE document embedding — no instruction prefix."""
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """BGE query embedding — prepends retrieval instruction prefix."""
        return self._embed([BGE_QUERY_INSTRUCTION + text])[0]
