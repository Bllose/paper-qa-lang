"""PDF ingestion pipeline built with LangGraph.

Takes a Paper object → parse → (optionally enrich) → chunk → store.

Usage::

    from paper_qa_lang.graph import ingest_paper
    from paper_qa_lang.store.paper_library import PaperLibrary
    from paper_qa_lang.config import get_chat_model
    from paper_qa_lang.models.types import Paper

    lib = PaperLibrary()
    llm = get_chat_model()  # needed only if you want multi-modal enrichment
    paper = Paper(doc_id="my_doc", file_location="path/to/paper.pdf")
    count = ingest_paper(paper=paper, library=lib, llm=llm)
    print(f"Stored {count} chunks")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langchain_core.language_models import BaseChatModel
from typing_extensions import TypedDict

from paper_qa_lang.models.types import Paper, PaperChunk, ParsedMedia
from paper_qa_lang.store.paper_library import PaperLibrary

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────


class IngestionState(TypedDict, total=False):
    """State that flows through the ingestion graph.

    ``path`` and ``doc_id`` are inputs; everything else is populated
    by the nodes as the pipeline executes.
    """
    path: str                          # Input: local PDF path
    doc_id: str                        # Input: document identifier

    # Pipeline state (populated by nodes)
    paper: Paper | None                # Paper(…, doc_id=doc_id, …)
    pages: dict | None                 # parse_pdf_to_pages() result
    has_media: bool                    # Whether parsed pages contain images/tables
    chunks: list[PaperChunk] | None    # chunk_pdf_pages() result
    chunk_count: int | None            # How many chunks were stored
    error: str | None                  # Error message if a node failed


# ── Graph builder ─────────────────────────────────────────────────────────


def build_ingestion_graph(
    library: PaperLibrary,
    llm: Any = None,
    *,
    chunk_chars: int = 3000,
    overlap: int = 200,
    parse_media: bool = True,
    enrichment_page_radius: int = 1,
    use_enrichment: bool = True,
    image_cluster_tolerance: float | tuple[float, float] = 25,
) -> Any:
    """Build a compiled LangGraph that ingests a PDF into the paper library.

    The graph has four stages:

    ::

        [parse_pdf] ──→ (has media + llm?) ──yes──→ [enrich_media] ──→ [chunk_pages] ──→ [store_paper]
               │                                      │
               └──────── no ──────────────────────────┘

    Parameters
    ----------
    library:
        ``PaperLibrary`` instance that holds the vector store + SQLite.
    llm:
        A LangChain ``BaseChatModel`` with multimodal support.  If ``None``
        the enrichment stage is skipped.
    chunk_chars:
        Target chunk size in characters (passed to ``chunk_pdf_pages``).
    overlap:
        Overlap between chunks in characters.
    parse_media:
        Whether to extract images and tables from the PDF.
    enrichment_page_radius:
        Number of surrounding pages to use as text context during enrichment.
    use_enrichment:
        Whether to include enriched media descriptions in embedding computation.
    image_cluster_tolerance:
        Tolerance (pts) for ``Page.cluster_drawings``.

    Returns
    -------
    A compiled ``langgraph.graph.CompiledStateGraph``.
    """

    # ── Node implementations (closures capture config) ────────────────────

    def _parse_pdf(state: IngestionState) -> dict:
        """Node 1: parse PDF → text + media per page."""
        path = state["path"]
        logger.info("Parsing PDF: %s", path)

        # Lazy imports so optional dependencies aren't required at import time
        from paper_qa_lang.parsing.pdf_parser import parse_pdf_to_pages

        try:
            pages = parse_pdf_to_pages(
                path,
                parse_media=parse_media,
                image_cluster_tolerance=image_cluster_tolerance,
            )
        except Exception as exc:
            return {"error": f"PDF parse failed: {exc}"}

        has_media = _pages_have_media(pages)
        logger.info("Parsed %d pages (media: %s)", len(pages), has_media)

        return {"pages": pages, "has_media": has_media}

    def _router(state: IngestionState) -> Literal["enrich_media", "chunk_pages"]:
        """Decide whether to run multimodal enrichment or skip straight to chunking."""
        if (
            llm is not None
            and state.get("has_media", False)
            and state.get("error") is None
        ):
            return "enrich_media"
        return "chunk_pages"

    def _enrich_media(state: IngestionState) -> dict:
        """Node 2 (optional): enrich media items with LLM descriptions."""
        from paper_qa_lang.parsing.enrichment import enrich_pages_media

        logger.info("Enriching media with LLM (page_radius=%d)", enrichment_page_radius)

        try:
            enriched = enrich_pages_media(
                state["pages"],
                llm=llm,
                page_radius=enrichment_page_radius,
            )
        except Exception as exc:
            logger.warning("Enrichment failed, proceeding without: %s", exc)
            return {}  # non-fatal — keep original pages

        return {"pages": enriched}

    def _chunk_pages(state: IngestionState) -> dict:
        """Node 3: page-aware chunking."""
        from paper_qa_lang.parsing.chunking import chunk_pdf_pages

        paper = state["paper"]
        pages = state["pages"]

        logger.info("Chunking %d pages", len(pages))
        try:
            chunks = chunk_pdf_pages(
                pages,
                paper=paper,
                chunk_chars=chunk_chars,
                overlap=overlap,
            )
        except Exception as exc:
            return {"error": f"Chunking failed: {exc}"}

        logger.info("Produced %d chunks", len(chunks))
        return {"chunks": chunks}

    def _store_paper(state: IngestionState) -> dict:
        """Node 4: store chunks in vector store + metadata in SQLite."""
        paper = state["paper"]
        chunks = state["chunks"]

        if not chunks:
            return {"error": "No chunks to store", "chunk_count": 0}

        logger.info(
            "Storing %d chunks for paper %s", len(chunks), paper.doc_id
        )
        try:
            count = library.add_chunks(paper, chunks, use_enrichment=use_enrichment)
        except Exception as exc:
            return {"error": f"Store failed: {exc}", "chunk_count": 0}

        return {"chunk_count": count}

    # ── Assemble graph ────────────────────────────────────────────────────

    graph = StateGraph(IngestionState)

    graph.add_node("parse_pdf", _parse_pdf)
    graph.add_node("enrich_media", _enrich_media)
    graph.add_node("chunk_pages", _chunk_pages)
    graph.add_node("store_paper", _store_paper)

    graph.set_entry_point("parse_pdf")
    graph.add_conditional_edges(
        "parse_pdf",
        _router,
        {
            "enrich_media": "enrich_media",
            "chunk_pages": "chunk_pages",
        },
    )
    graph.add_edge("enrich_media", "chunk_pages")
    graph.add_edge("chunk_pages", "store_paper")
    graph.add_edge("store_paper", END)

    return graph.compile()


# ── Convenience wrapper ───────────────────────────────────────────────────

def ingest_paper(
    paper: Paper,
    library: PaperLibrary,
    enrich_llm: BaseChatModel = None,
    **kwargs: Any,
) -> int:
    """Parse, (optionally enrich), chunk, and store a PDF — all in one call.

    This is a convenience wrapper around ``build_ingestion_graph().invoke()``.
    Use it when you just want to ingest a paper without interacting with the
    graph directly.

    Parameters
    ----------
    paper:
        ``Paper`` object with ``file_location`` and ``doc_id`` set.
    library:
        ``PaperLibrary`` instance for storage.
    enrich_llm:
        Multimodal chat model for media enrichment.  ``None`` = skip enrichment.
    **kwargs:
        Passed through to ``build_ingestion_graph()`` (e.g. ``chunk_chars``,
        ``overlap``, ``parse_media``, ``enrichment_page_radius``).

    Returns
    -------
    Number of chunks stored in the vector store.

    Raises
    ------
    RuntimeError
        If any pipeline stage fails (check ``error`` in the returned state).
    FileNotFoundError
        If the PDF does not exist.
    """
    path_str = paper.file_location
    if not path_str or not os.path.isfile(path_str):
        raise FileNotFoundError(f"PDF not found: {path_str}")

    graph = build_ingestion_graph(library, enrich_llm, **kwargs)

    initial_state: IngestionState = {
        "path": path_str,
        "doc_id": paper.doc_id,
        "paper": paper,
    }

    result: IngestionState = graph.invoke(initial_state)

    if result.get("error"):
        raise RuntimeError(
            f"Ingestion failed at node: {result['error']}"
        )

    return result.get("chunk_count", 0)


# ── Helpers ───────────────────────────────────────────────────────────────

def _pages_have_media(
    pages: dict[str, tuple[str, list[ParsedMedia]]],
) -> bool:
    """Check if any page has media items with image data."""
    return any(
        bool(m.data)
        for _, (_, media_list) in pages.items()
        for m in media_list
    )
