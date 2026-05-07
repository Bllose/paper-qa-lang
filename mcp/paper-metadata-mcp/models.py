"""Data models and metadata merging logic for paper-metadata-mcp."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Normalized academic paper metadata aggregated from multiple sources."""

    doi: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_date: str | None = None
    year: int | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    publisher: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    url: str | None = None
    pdf_url: str | None = None
    bibtex: str | None = None
    issn: str | None = None
    is_open_access: bool | None = None
    publication_types: list[str] = Field(default_factory=list)
    abstract: str | None = None
    reference_count: int | None = None
    retraction_status: str | None = None
    source_quality: str | None = None
    sources: list[str] = Field(default_factory=list, description="Which providers contributed")


# Priority order for conflict resolution: earlier = more trusted
PROVIDER_PRIORITY = ["crossref", "semantic_scholar", "openalex", "unpaywall"]


def merge_metadata(results: list[PaperMetadata | None]) -> PaperMetadata | None:
    """Merge metadata from multiple providers into a single result.

    Strategy: earlier providers in PROVIDER_PRIORITY fill in missing fields.
    For citation counts, take the maximum.
    """
    valid = [r for r in results if r is not None and r.doi is not None]
    if not valid:
        return None

    merged = PaperMetadata()
    seen_sources: set[str] = set()
    priority_map = {name: i for i, name in enumerate(PROVIDER_PRIORITY)}

    def higher_priority(source: str | None, current_best: str | None) -> bool:
        if source is None:
            return False
        if current_best is None:
            return True
        return priority_map.get(source, 99) < priority_map.get(current_best, 99)

    for r in sorted(valid, key=lambda x: x.doi or ""):
        for src in r.sources:
            seen_sources.add(src)
        # Prefer non-None, higher-priority values for scalar fields
        for field_name in (
            "doi", "title", "publication_date", "journal",
            "volume", "issue", "pages", "publisher", "url", "pdf_url",
            "bibtex", "issn", "abstract", "retraction_status", "source_quality",
        ):
            new_val = getattr(r, field_name, None)
            curr_val = getattr(merged, field_name, None)
            if new_val is not None and curr_val is None:
                setattr(merged, field_name, new_val)

        # Year: prefer non-None
        if r.year is not None and merged.year is None:
            merged.year = r.year

        # Citation counts: take maximum
        if r.citation_count is not None:
            merged.citation_count = max(
                (merged.citation_count or 0), r.citation_count
            ) or r.citation_count
        if r.influential_citation_count is not None:
            merged.influential_citation_count = max(
                (merged.influential_citation_count or 0), r.influential_citation_count
            ) or r.influential_citation_count
        if r.reference_count is not None:
            merged.reference_count = max(
                (merged.reference_count or 0), r.reference_count
            ) or r.reference_count

        # Authors: merge uniquely
        for a in r.authors:
            if a and a not in merged.authors:
                merged.authors.append(a)

        # Publication types: merge uniquely
        for t in r.publication_types:
            if t and t not in merged.publication_types:
                merged.publication_types.append(t)

        # Boolean: prefer True
        if r.is_open_access is True:
            merged.is_open_access = True

    merged.sources = sorted(seen_sources)
    return merged


# --------------- Cache ---------------


@dataclass
class CacheEntry:
    data: Any
    expires_at: float


class TTLCache:
    """Simple in-memory TTL cache to avoid hitting API rate limits."""

    def __init__(self, default_ttl: float = 3600.0):
        self._store: dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        return entry.data

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._store[key] = CacheEntry(
            data=value,
            expires_at=time.monotonic() + (ttl or self.default_ttl),
        )

    def make_key(self, provider: str, query_type: str, value: str) -> str:
        return f"{provider}:{query_type}:{value}"
