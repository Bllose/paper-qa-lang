"""Data models for paper-download-mcp."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PdfSource(BaseModel):
    """Result from finding a PDF URL."""

    doi: str
    url: str
    source: str = Field(description="Which provider found the URL (arxiv/unpaywall/semantic_scholar)")
    is_open_access: bool = False


class DownloadResult(BaseModel):
    """Result from downloading a PDF."""

    doi: str | None = None
    url: str | None = None
    file_path: str
    file_size_bytes: int
