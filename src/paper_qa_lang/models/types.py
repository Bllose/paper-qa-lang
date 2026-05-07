"""Core data models — lightweight replacement for original DocDetails/Text (1384 lines)."""

from __future__ import annotations

import hashlib
import logging
from random import Random
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Paper(BaseModel):
    """论文元数据 (精简版 DocDetails)."""

    doc_id: str = Field(description="Unique document identifier")
    doi: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    citation_count: int | None = None
    pdf_url: str | None = None
    file_location: str | None = None
    source_quality: int | None = None
    is_retracted: bool | None = None
    abstract: str | None = None
    bibtex: str | None = None
    publication_date: str | None = None
    content_hash: str | None = Field(
        default=None,
        description="MD5 hex digest of the source file contents, used for deduplication.",
    )


class ParsedMedia(BaseModel):
    """An image or table extracted from a document page."""

    index: int = Field(description="Index within the page (0 if single image).")
    data: bytes = Field(default=b"", description="Raw image bytes (PNG preferred).")
    text: str | None = Field(default=None, description="Optional text content.")
    media_type: str = Field(default="drawing", description="Type: drawing, table, screenshot.")
    page_num: int | None = Field(default=None, description="Page number (1-indexed).")
    bbox: tuple[float, float, float, float] | None = Field(default=None, description="Bounding box.")
    enriched_description: str | None = Field(default=None, description="LLM-generated description.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_id(self) -> UUID:
        to_hash: bytes = self.data if not self.text else self.data + self.text.encode()
        seed = int(hashlib.sha256(to_hash).hexdigest(), 16) % (2**32)
        uuid_int = Random(seed).getrandbits(128)
        uuid_int &= ~(0xF << 76)
        uuid_int |= 0x4 << 76
        uuid_int &= ~(0x3 << 62)
        uuid_int |= 0x2 << 62
        return UUID(int=uuid_int)


class PaperChunk(BaseModel):
    """A chunk of text from a paper, optionally linked to images/tables."""

    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk text content")
    paper_id: str = Field(description="Associated Paper.doc_id")
    page_num: int | None = Field(default=None, description="Starting page number.")
    page_range: tuple[int, int] | None = Field(default=None, description="Page range.")
    media: list[ParsedMedia] = Field(default_factory=list, description="Associated media.")
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_embeddable_text(self, with_enrichment: bool = True) -> str:
        if not with_enrichment or not self.media:
            return self.text
        descriptions = [m.enriched_description for m in self.media if m.enriched_description]
        if not descriptions:
            return self.text
        return "\n\n".join([self.text] + [f"[Media description] {d}" for d in descriptions])


# ── Re-export paper identification from the ingestion module ─────────────────

from paper_qa_lang.ingestion.identify import paper_from_pdf  # noqa: F401