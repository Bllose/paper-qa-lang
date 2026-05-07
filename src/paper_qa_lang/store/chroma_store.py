"""Chroma vector store wrapper for paper chunks.

Encapsulates LangChain's Chroma and exposes add/search/mmr_search
working with our PaperChunk model.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from paper_qa_lang.helper.proxy_detector import ProxyDetector
from paper_qa_lang.models.types import PaperChunk

logger = logging.getLogger(__name__)


def _chunk_to_document(chunk: PaperChunk) -> Document:
    """Convert PaperChunk → LangChain Document for Chroma storage.

    Note: ``media`` is intentionally excluded — Chroma metadata only supports
    simple types (str, int, float, bool). Media stays attached to PaperChunk
    in memory during the add flow and is used during answer generation.
    """
    meta = {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "page_num": chunk.page_num,
    }
    if chunk.page_range:
        meta["page_start"] = chunk.page_range[0]
        meta["page_end"] = chunk.page_range[1]
    meta.update(chunk.metadata)
    return Document(
        page_content=chunk.text,
        id=chunk.chunk_id,
        metadata=meta,
    )


def _document_to_chunk(doc: Document) -> PaperChunk:
    """Convert LangChain Document back → PaperChunk."""
    meta = doc.metadata or {}
    skip_keys = {"chunk_id", "paper_id", "page_num", "page_start", "page_end"}
    page_range: tuple[int, int] | None = None
    if meta.get("page_start") is not None and meta.get("page_end") is not None:
        page_range = (meta["page_start"], meta["page_end"])
    return PaperChunk(
        chunk_id=meta.get("chunk_id", doc.id or ""),
        text=doc.page_content,
        paper_id=meta.get("paper_id", ""),
        page_num=meta.get("page_num"),
        page_range=page_range,
        metadata={k: v for k, v in meta.items() if k not in skip_keys},
    )


class ChromaStore:
    """Wrapper around LangChain Chroma for paper chunk storage and retrieval."""

    def __init__(
        self,
        collection_name: str = "papers",
        persist_directory: str = ".chroma",
        embedding_fn: Embeddings | None = None,
    ) -> None:
        if embedding_fn is None:
            from paper_qa_lang.embeddings import get_embedding_model

            embedding_fn = get_embedding_model()

        self._store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_fn,
        )
        self._embedding_fn = embedding_fn

    # ---- write ----

    def add_chunks(
        self, chunks: list[PaperChunk], use_enrichment: bool = True
    ) -> list[str]:
        """Add chunks to the vector store.

        When chunks have enriched media descriptions and ``use_enrichment``
        is set, the embedding is computed from text + enrichment (for better
        retrieval recall), while only the original text is stored.

        Args:
            chunks: Chunks to add.
            use_enrichment: Whether to include enriched media descriptions
                in the embedding computation.

        Returns:
            List of stored chunk IDs.
        """
        has_enrichment = use_enrichment and any(
            any(m.enriched_description for m in c.media) for c in chunks
        )

        if has_enrichment:
            # Pre-compute embeddings on enriched text,
            # but store only the original text and metadata
            texts = [c.text for c in chunks]
            embed_texts = [c.get_embeddable_text(with_enrichment=True) for c in chunks]
            embeddings = self._embedding_fn.embed_documents(embed_texts)
            metadatas = [_chunk_to_document(c).metadata for c in chunks]
            ids = [c.chunk_id for c in chunks]
            stored_ids = self._store.add_embeddings(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(
                f"Added {len(stored_ids)} chunks with enriched embeddings"
            )
            return stored_ids

        # Standard path: embed page_content directly
        docs = [_chunk_to_document(c) for c in chunks]
        ids = self._store.add_documents(docs)
        logger.info(f"Added {len(ids)} chunks to Chroma store")
        return ids

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID."""
        self._store.delete(chunk_ids)
        logger.info(f"Deleted {len(chunk_ids)} chunks from Chroma store")

    # ---- search ----

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[PaperChunk]:
        """Plain vector similarity search."""
        docs = self._store.similarity_search(query, k=k, filter=filter)
        return [_document_to_chunk(d) for d in docs]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[PaperChunk]:
        """Search by raw embedding vector."""
        docs = self._store.similarity_search_by_vector(embedding, k=k, filter=filter)
        return [_document_to_chunk(d) for d in docs]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 10,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[PaperChunk]:
        """MMR search — balances relevance & diversity.

        ``lambda_mult=0.5`` is the default from LangChain.
        Lower values = more diversity, higher = more relevance.
        """
        docs = self._store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )
        return [_document_to_chunk(d) for d in docs]

    def count(self) -> int:
        """Total number of documents in the collection."""
        return self._store._collection.count()  # type: ignore[attr-defined]
