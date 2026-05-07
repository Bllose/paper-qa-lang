"""PaperLibrary — unified paper library combining Chroma vector search + SQLite metadata.

Replaces the original ``Docs`` (721 lines) + ``NumpyVectorStore`` (585 lines).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paper_qa_lang.config.settings import Settings
from paper_qa_lang.models.types import Paper, PaperChunk, ParsedMedia
from paper_qa_lang.parsing.context_builder import ContextBuilder, ScoredContext
from paper_qa_lang.store.chroma_store import ChromaStore
from paper_qa_lang.utils import md5sum

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from :meth:`PaperLibrary.query`."""

    answer: str = ""
    contexts: list[ScoredContext] = field(default_factory=list)
    question: str = ""


class PaperLibrary:
    """Unified paper library: Chroma vector store + SQLite metadata.

    Usage::

        lib = PaperLibrary()
        lib.add_paper("path/to/paper.pdf", Paper(doc_id="abc123", ...))
        results = lib.search("machine learning in biology")
        meta = lib.query_metadata("some_doc_id")
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_fn: Embeddings | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._vector_store = ChromaStore(
            collection_name=self._settings.store.collection_name,
            persist_directory=self._settings.store.persist_directory,
            embedding_fn=embedding_fn,
        )
        self._db_path = self._settings.store.metadata_db_path
        self._init_db()

    # ---- metadata storage (SQLite) ----

    def _init_db(self) -> None:
        """Ensure the SQLite metadata table exists."""
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    doc_id          TEXT PRIMARY KEY,
                    doi            TEXT,
                    title          TEXT,
                    authors        TEXT,
                    year           INTEGER,
                    journal        TEXT,
                    citation_count INTEGER,
                    pdf_url        TEXT,
                    file_location  TEXT,
                    source_quality INTEGER,
                    is_retracted   INTEGER,
                    abstract       TEXT,
                    bibtex         TEXT,
                    publication_date TEXT,
                    content_hash   TEXT
                );
            """)
            # Add content_hash column if upgrading an existing DB
            try:
                conn.execute("ALTER TABLE papers ADD COLUMN content_hash TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

    def _save_paper(self, paper: Paper) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO papers
                   (doc_id, doi, title, authors, year, journal,
                    citation_count, pdf_url, file_location,
                    source_quality, is_retracted, abstract, bibtex,
                    publication_date, content_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    paper.doc_id,
                    paper.doi,
                    paper.title,
                    json.dumps(paper.authors, ensure_ascii=False),
                    paper.year,
                    paper.journal,
                    paper.citation_count,
                    paper.pdf_url,
                    paper.file_location,
                    paper.source_quality,
                    int(paper.is_retracted) if paper.is_retracted is not None else None,
                    paper.abstract,
                    paper.bibtex,
                    paper.publication_date,
                    paper.content_hash,
                ),
            )

    def _load_paper(self, doc_id: str) -> Paper | None:
        _COLS = (
            "doc_id doi title authors year journal citation_count "
            "pdf_url file_location source_quality is_retracted abstract "
            "bibtex publication_date content_hash"
        ).split()
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"SELECT {','.join(_COLS)} FROM papers WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            return None
        data = dict(zip(_COLS, row))
        if data.get("authors"):
            data["authors"] = json.loads(data["authors"])
        if data.get("is_retracted") is not None:
            data["is_retracted"] = bool(data["is_retracted"])
        return Paper(**data)

    def _delete_paper_meta(self, doc_id: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM papers WHERE doc_id = ?", (doc_id,))

    def content_hash_exists(self, content_hash: str) -> bool:
        """Check if a content_hash already exists in the library (dedup)."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            ).fetchone()
        return row is not None

    def list_papers(self) -> list[Paper]:
        """Return all papers in the metadata store."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT doc_id, doi, title, authors, year, journal FROM papers ORDER BY journal"
            ).fetchall()
        papers: list[Paper] = []
        for row in rows:
            data = {
                "doc_id": row[0],
                "doi": row[1],
                "title": row[2],
                "authors": json.loads(row[3]) if row[3] else [],
                "year": row[4],
                "journal": row[5],
            }
            papers.append(Paper(**data))
        return papers

    # ---- chunk storage (Chroma) ----

    def add_chunks(
        self,
        paper: Paper,
        chunks: list[PaperChunk],
        use_enrichment: bool = True,
    ) -> int:
        """Add a paper's chunks to the vector store and metadata to SQLite.

        Args:
            paper: Paper metadata.
            chunks: Pre-parsed and chunked text fragments.
            use_enrichment: When True (default), if any chunk has enriched
                media descriptions, the embedding will be computed from
                ``text + enrichment`` for better retrieval recall.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            raise ValueError("No chunks provided — nothing to store.")
        for i, c in enumerate(chunks):
            if not c.chunk_id:
                c.chunk_id = f"{paper.doc_id}__chunk_{i:04d}"
            c.paper_id = paper.doc_id
        self._save_paper(paper)
        ids = self._vector_store.add_chunks(chunks, use_enrichment=use_enrichment)
        logger.info(f"Stored paper {paper.doc_id} ({paper.title}) with {len(ids)} chunks")
        return len(ids)

    # ---- add paper (PDF / text) ----

    def add_paper(self, path: str | os.PathLike, paper: Paper, **kwargs: Any) -> int:
        """Parse a PDF and add it with page-aware chunking.

        Args:
            path: Path to a PDF file.
            paper: Paper metadata.
            **kwargs: Passed through to the PDF parser.
                Key options:
                - parse_media (bool): Extract images/tables (default True).
                - page_range (tuple[int,int]|None): Page range to parse.
                - chunk_chars (int): Target chunk size (default 3000).
                - overlap (int): Chunk overlap (default 200).
                - enrichment_llm (BaseChatModel|None):
                  If set, run multimodal enrichment after PDF parsing (before
                  chunking). Each image is sent to the LLM for description and
                  relevance filtering. Requires ``parse_media=True``.
                - enrichment_page_radius (int): Page radius for enrichment
                  context text (default 1). -1 = all pages.
                - use_enrichment (bool): Include enrichment descriptions in
                  embedding computation (default True).
                - image_cluster_tolerance (float|tuple): Drawing clustering tolerance.

        Returns:
            Number of chunks stored.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is already indexed (dedup) or content is invalid.
        """
        path_str = str(path)
        if not os.path.isfile(path_str):
            raise FileNotFoundError(f"File not found: {path}")

        # Compute content hash for deduplication
        content_hash = md5sum(path_str)

        if self.content_hash_exists(content_hash):
            raise ValueError(
                f"File already indexed (content_hash={content_hash}): {path}"
            )
        paper.content_hash = content_hash

        chunk_chars = kwargs.pop("chunk_chars", 3000)
        overlap = kwargs.pop("overlap", 200)
        enrichment_llm: BaseChatModel | None = kwargs.pop("enrichment_llm", None)
        enrichment_page_radius: int = kwargs.pop("enrichment_page_radius", 1)
        use_enrichment: bool = kwargs.pop("use_enrichment", True)

        if path_str.endswith(".pdf"):
            chunks = self._ingest_pdf(
                path_str, paper, chunk_chars, overlap, kwargs,
                enrichment_llm, enrichment_page_radius,
            )
            # Validate that parsed content looks like text
            combined_text = "".join(c.text for c in chunks)
            if not combined_text.strip():
                raise ValueError(
                    f"No text was extracted from the PDF at {path}. "
                    "The file may be corrupt or image-only."
                )
            if not _looks_like_text(combined_text):
                raise ValueError(
                    f"The extracted content from {path} does not look like "
                    "a text document. It may be empty, corrupt, or a scanned image PDF."
                )
        else:
            chunks = self._ingest_text(path_str, paper, chunk_chars, overlap)
            if not chunks or not chunks[0].text.strip():
                raise ValueError(f"No text was extracted from {path}")
            if not _looks_like_text(chunks[0].text):
                raise ValueError(f"The file {path} does not appear to be a text document.")

        return self.add_chunks(paper, chunks, use_enrichment=use_enrichment)

    def _ingest_pdf(
        self,
        path: str,
        paper: Paper,
        chunk_chars: int,
        overlap: int,
        parser_kwargs: dict[str, Any],
        enrichment_llm: BaseChatModel | None = None,
        enrichment_page_radius: int = 1,
    ) -> list[PaperChunk]:
        """Parse PDF → (optional enrichment) → page-aware chunking."""
        from paper_qa_lang.parsing.chunking import chunk_pdf_pages
        from paper_qa_lang.parsing.pdf_parser import parse_pdf_to_pages

        parse_media = parser_kwargs.pop("parse_media", True)
        pages = parse_pdf_to_pages(
            path,
            parse_media=parse_media,
            **parser_kwargs,
        )

        # Run multimodal enrichment before chunking (if LLM provided)
        if enrichment_llm and parse_media and _has_media(pages):
            from paper_qa_lang.parsing.enrichment import enrich_pages_media
            logger.info(
                "Running multimodal enrichment for %s with page_radius=%d",
                paper.doc_id, enrichment_page_radius,
            )
            pages = enrich_pages_media(
                pages,
                llm=enrichment_llm,
                page_radius=enrichment_page_radius,
            )

        chunks = chunk_pdf_pages(pages, paper, chunk_chars=chunk_chars, overlap=overlap)
        paper.file_location = path
        return chunks

    def _ingest_text(
        self,
        path: str,
        paper: Paper,
        chunk_chars: int,
        overlap: int,
    ) -> list[PaperChunk]:
        """Read a text file → plain chunking."""
        from paper_qa_lang.parsing.chunking import chunk_plain_text

        text = Path(path).read_text(encoding="utf-8", errors="replace")
        chunks = chunk_plain_text(text, paper, chunk_size=chunk_chars, chunk_overlap=overlap)
        paper.file_location = path
        return chunks


    def add_paper_with_chunks(
        self,
        paper: Paper,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> int:
        """Split raw text into chunks and store everything.

        Convenience wrapper that runs a LangChain ``RecursiveCharacterTextSplitter``
        on the plain text before calling :meth:`add_chunks`.

        .. deprecated::
            Prefer :meth:`add_paper` which handles PDF parsing natively.
        """
        cs = chunk_size or self._settings.chunk.size
        co = chunk_overlap or self._settings.chunk.overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs,
            chunk_overlap=co,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_docs = splitter.create_documents([text])
        chunks = [
            PaperChunk(
                chunk_id=f"{paper.doc_id}__chunk_{i:04d}",
                text=d.page_content,
                paper_id=paper.doc_id,
            )
            for i, d in enumerate(split_docs)
        ]
        return self.add_chunks(paper, chunks)

    # ---- unified ingestion entry point ----

    async def ingest(
        self,
        path: str | os.PathLike | None = None,
        doi: str | None = None,
        paper: Paper | None = None,
        *,
        download_dir: str = ".downloads",
        **kwargs: Any,
    ) -> int:
        """Unified ingestion entry point: download, identify, parse, and store.

        Supports three modes:

        **Mode 1 — DOI**: download the PDF via paper-download-mcp, then
        query metadata via paper-metadata-mcp, and ingest::

            count = await lib.ingest(doi="10.1038/s41586-023-06236-9")

        **Mode 2 — file path**: identify metadata from the PDF, then ingest::

            count = await lib.ingest(path="paper.pdf")

        **Mode 3 — file path + pre-identified Paper**: skip identification::

            paper = Paper(doc_id="my_doc", title="My Paper")
            count = await lib.ingest(path="paper.pdf", paper=paper)

        Args:
            path: Path to a PDF file.
            doi: DOI of a paper to download and ingest.
            paper: Pre-identified ``Paper`` object.  If provided with
                ``path``, the identification step is skipped.
            download_dir: Directory to save downloaded PDFs (default ``.downloads``).
            **kwargs: Passed through to :meth:`add_paper` — see its docs for
                options like ``enrichment_llm``, ``chunk_chars``, ``overlap``,
                ``parse_media``, etc.

        Returns:
            Number of chunks stored.

        Raises:
            ValueError: If none of ``path``, ``doi``, or ``paper`` is provided.
            FileNotFoundError: If ``path`` does not exist.
            RuntimeError: If paper-download MCP is required but not configured.
        """
        resolved_path: str | None = None

        if doi:
            # Fast path: query metadata by DOI, then download + parse
            paper = await self._identify_by_doi(doi)
            resolved_path = await self._download_by_doi(doi, download_dir)
            paper.file_location = resolved_path
        elif path is not None and paper is None:
            resolved_path = str(path)
            from paper_qa_lang.ingestion.identify import paper_from_pdf

            paper = await paper_from_pdf(resolved_path)
        elif path is not None and paper is not None:
            resolved_path = str(path)
            paper.file_location = resolved_path
        else:
            raise ValueError(
                "Provide at least one of: path (to ingest a local file), "
                "doi (to download and ingest), or "
                "path+paper (to skip identification)."
            )

        return self.add_paper(resolved_path, paper, **kwargs)

    async def _identify_by_doi(self, doi: str) -> Paper:
        """Query paper-metadata-mcp and build a Paper from the result.

        Falls back to a minimal Paper (doc_id from DOI) if the MCP server
        is unavailable.
        """
        doc_id = re.sub(r"[^\w]", "_", doi)

        try:
            from paper_qa_lang.ingestion.identify import _load_mcp_tools

            mcp_tools = await _load_mcp_tools("paper-metadata", self._settings)
            if not mcp_tools:
                logger.info(
                    "paper-metadata MCP not available, using minimal Paper for DOI %s",
                    doi,
                )
                return Paper(doc_id=doc_id, doi=doi)

            query_tool = next(
                (t for t in mcp_tools if t.name == "query_by_doi"), None
            )
            bibtex_tool = next(
                (t for t in mcp_tools if t.name == "get_bibtex"), None
            )

            metadata_text: str | None = None
            bibtex_str: str | None = None

            if query_tool:
                raw = await query_tool.ainvoke({"doi": doi})
                if isinstance(raw, str):
                    metadata_text = raw

            if bibtex_tool:
                raw = await bibtex_tool.ainvoke({"doi": doi})
                if isinstance(raw, str) and not raw.startswith("No BibTeX"):
                    bibtex_str = raw

            return self._build_paper_from_metadata(
                doi, doc_id, metadata_text, bibtex_str,
            )
        except Exception as exc:
            logger.warning(
                "Metadata lookup failed for DOI %s: %s", doi, exc,
            )
            return Paper(doc_id=doc_id, doi=doi)

    def _build_paper_from_metadata(
        self,
        doi: str,
        doc_id: str,
        metadata_text: str | None,
        bibtex_str: str | None,
    ) -> Paper:
        """Build a Paper from the MCP query_by_doi response text and BibTeX."""
        data: dict[str, Any] = {"doc_id": doc_id, "doi": doi}

        # Parse BibTeX first (more structured)
        if bibtex_str:
            bib_data = self._parse_bibtex(bibtex_str)
            data.setdefault("title", bib_data.get("title"))
            data.setdefault("authors", bib_data.get("authors", []))
            data.setdefault("year", bib_data.get("year"))
            data.setdefault("journal", bib_data.get("journal"))
            data["bibtex"] = bibtex_str

        # Parse text metadata form (fills gaps BibTeX might miss)
        if metadata_text:
            for line in metadata_text.split("\n"):
                line = line.strip()
                if line.startswith("Title:") and "title" not in data:
                    data["title"] = line[len("Title:"):].strip()
                elif line.startswith("DOI:") and not data.get("doi"):
                    data["doi"] = line[len("DOI:"):].strip()
                elif line.startswith("Journal:") and "journal" not in data:
                    data["journal"] = line[len("Journal:"):].strip()
                elif line.startswith("Citation Count:"):
                    try:
                        data["citation_count"] = int(
                            line[len("Citation Count:"):].strip()
                        )
                    except ValueError:
                        pass
                elif line.startswith("OA PDF URL:") and "pdf_url" not in data:
                    data["pdf_url"] = line[len("OA PDF URL:"):].strip()
                elif line.startswith("Abstract:"):
                    abstract = line[len("Abstract:"):].strip()
                    if abstract:
                        data["abstract"] = abstract

        # Ensure minimum fields
        data.setdefault("title", f"Paper {doi}")
        data.setdefault("authors", [])

        return Paper(**data)

    @staticmethod
    def _parse_bibtex(bibtex_str: str) -> dict[str, Any]:
        """Minimal BibTeX parser — extracts title, author, year, journal."""
        result: dict[str, Any] = {}

        # Extract title
        m = re.search(
            r"title\s*=\s*\{([^}]+)\}", bibtex_str, re.IGNORECASE
        )
        if m:
            result["title"] = m.group(1).strip()

        # Extract author(s)
        m = re.search(
            r"author\s*=\s*\{([^}]+)\}", bibtex_str, re.IGNORECASE
        )
        if m:
            authors_str = m.group(1)
            authors = [a.strip() for a in authors_str.split(" and ")]
            result["authors"] = authors

        # Extract year
        m = re.search(
            r"year\s*=\s*\{([^}]+)\}", bibtex_str, re.IGNORECASE
        )
        if m:
            try:
                result["year"] = int(m.group(1).strip())
            except ValueError:
                pass

        # Extract journal
        m = re.search(
            r"(?:journal|booktitle)\s*=\s*\{([^}]+)\}",
            bibtex_str, re.IGNORECASE,
        )
        if m:
            result["journal"] = m.group(1).strip()

        return result

    async def _download_by_doi(self, doi: str, download_dir: str = ".downloads") -> str:
        """Download a PDF by DOI using the paper-download MCP server."""
        from paper_qa_lang.ingestion.identify import _load_mcp_tools

        mcp_tools = await _load_mcp_tools("paper-download", self._settings)
        if not mcp_tools:
            raise RuntimeError(
                "paper-download MCP server is not configured. "
                "Add it to .claude/settings.local.json under 'mcpServers'."
            )

        download_tool = next(
            (t for t in mcp_tools if t.name == "download_pdf"), None
        )
        if not download_tool:
            raise RuntimeError(
                "No 'download_pdf' tool found in paper-download MCP server."
            )

        # Ensure the download directory exists
        dest_dir = Path(download_dir).expanduser().resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        result = await download_tool.ainvoke(
            {"doi": doi, "output_dir": str(dest_dir)}
        )

        # The MCP tool returns a text response like:
        #   "Downloaded: /path/to/file.pdf\nSize: ...\nDOI: ..."
        if isinstance(result, str):
            result_text = result.strip()
        elif isinstance(result, list):
            key = result[0]['type']
            result_text = result[0][key]
        
        if result_text.startswith("Downloaded: "):
            return result_text.split("\n")[0].replace("Downloaded: ", "").strip()
        raise RuntimeError(f"Download failed for DOI {doi}: {result}")

        # raise RuntimeError(
        #     f"Unexpected result type from download_pdf tool: {type(result).__name__}"
        # )

    # ---- query ----

    def search(
        self,
        query: str,
        k: int = 10,
        mmr: bool = True,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[PaperChunk]:
        """Vector search — plain or MMR.

        Args:
            query: Natural-language query.
            k: Number of results.
            mmr: Use MMR (True) or plain similarity (False).
            lambda_mult: MMR diversity knob (0=diverse, 1=relevant).
            filter: Chroma metadata filter dict.

        Returns:
            Ranked list of PaperChunks.
        """
        if mmr:
            return self._vector_store.max_marginal_relevance_search(
                query, k=k, lambda_mult=lambda_mult, filter=filter
            )
        return self._vector_store.similarity_search(query, k=k, filter=filter)

    async def query(
        self,
        question: str,
        k: int = 10,
        mmr: bool = True,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        llm: BaseChatModel | None = None,
        summary_length: int = 150,
        answer_length: int = 500,
        max_contexts: int = 5,
        min_score: int = 1,
    ) -> QueryResult:
        """End-to-end question-answering pipeline.

        Steps:

        1. **Vector search** — retrieve the top *k* chunks (MMR by default).
        2. **Metadata lookup** — load ``Paper`` objects for all retrieved chunks.
        3. **Evidence scoring** — concurrently score each chunk for relevance
           (async LLM calls with concurrency limit).
        4. **Filter & sort** — discard low-scoring chunks, keep the *N* best.
        5. **Answer generation** — produce a final answer from the top contexts.

        Args:
            question: Natural-language question.
            k: Number of chunks to retrieve.
            mmr: Use MMR (True) or plain similarity (False).
            lambda_mult: MMR diversity knob (0=diverse, 1=relevant).
            filter: Chroma metadata filter dict.
            llm: Chat model for scoring and answer generation.
                Falls back to ``self._settings.llm.get_llm()`` if ``None``.
            summary_length: Max words per evidence summary.
            answer_length: Max words for the final answer.
            max_contexts: Max evidence contexts to include in the answer.
            min_score: Minimum relevance score to keep a context (1 = default).

        Returns:
            A ``QueryResult`` with the answer text, scored contexts,
            and original question.
        """
        # 1. Vector search
        chunks = self.search(
            query=question, k=k, mmr=mmr,
            lambda_mult=lambda_mult, filter=filter,
        )
        if not chunks:
            logger.info("No chunks found for question: %s", question[:80])
            return QueryResult(
                answer="No relevant documents found in the library.",
                question=question,
            )

        # 2. Look up paper metadata
        paper_ids = list({c.paper_id for c in chunks})
        papers: dict[str, Paper] = {}
        for pid in paper_ids:
            p = self._load_paper(pid)
            if p:
                papers[pid] = p
        logger.info(
            "Found %d chunks from %d papers",
            len(chunks), len(papers),
        )

        # 3. Evidence scoring (async concurrent)
        resolved_llm = llm or self._get_default_llm()
        cb = ContextBuilder(
            llm=resolved_llm,
            summary_length=summary_length,
            answer_length=answer_length,
        )
        contexts = await cb.score_chunks_async(
            chunks, question, papers=papers,
        )

        # 4. Filter by minimum score
        contexts = [c for c in contexts if c.relevance_score >= min_score]

        # 5. Truncate to max_contexts (already sorted by score desc)
        contexts = contexts[:max_contexts]

        # 6. Generate answer
        result = cb.generate_answer(question, contexts)

        return QueryResult(
            answer=result.answer,
            contexts=contexts,
            question=question,
        )

    def _get_default_llm(self) -> BaseChatModel:
        """Build an LLM from the current settings."""
        return self._settings.llm.get_llm()

    def query_metadata(self, doc_id: str) -> Paper | None:
        """Retrieve a paper's metadata by its doc_id."""
        return self._load_paper(doc_id)

    def delete_paper(self, doc_id: str) -> None:
        """Remove a paper and all its chunks."""
        # Gather chunk IDs from Chroma by paper_id filter
        try:
            matching = self._vector_store.similarity_search(
                query="", k=10000, filter={"paper_id": doc_id}
            )
            chunk_ids = [c.chunk_id for c in matching if c.chunk_id]
            if chunk_ids:
                self._vector_store.delete_chunks(chunk_ids)
                logger.info(f"Deleted {len(chunk_ids)} chunks for paper {doc_id}")
        except Exception as exc:
            logger.warning(f"Failed to delete chunks for {doc_id}: {exc}")

        self._delete_paper_meta(doc_id)
        logger.info(f"Deleted paper {doc_id} from metadata store")

    @property
    def chunk_count(self) -> int:
        """Total indexed chunks."""
        return self._vector_store.count()

    @property
    def paper_count(self) -> int:
        """Total papers in metadata store."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
        return row[0] if row else 0


def _has_media(pages: dict[str, tuple[str, list[ParsedMedia]]]) -> bool:
    """Check if any page has media items with image data."""
    return any(
        bool(m.data)
        for _, (_, media_list) in pages.items()
        for m in media_list
    )


def _looks_like_text(text: str, threshold: float = 2.5) -> bool:
    """Check if text appears to be readable content using entropy-based analysis."""
    if not text.strip():
        return False
    total = len(text)
    if total < 20:
        return False
    counts = Counter(text)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    return threshold <= entropy <= 8.0
