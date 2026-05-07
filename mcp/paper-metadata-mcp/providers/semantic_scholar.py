"""Semantic Scholar API provider for paper metadata."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from models import PaperMetadata

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "title,authors,citationCount,influentialCitationCount,"
    "openAccessPdf,publicationDate,journal,venue,abstract,"
    "externalIds,publicationTypes,referenceCount"
)
S2_TIMEOUT = 10.0


def _parse_s2_paper(data: dict[str, Any]) -> PaperMetadata | None:
    """Parse Semantic Scholar paper response into PaperMetadata."""
    doi = None
    ext_ids = data.get("externalIds") or {}
    doi = ext_ids.get("DOI")

    if not doi and not data.get("title"):
        return None

    title = data.get("title")

    # Authors
    authors: list[str] = []
    for a in data.get("authors") or []:
        name = a.get("name")
        if name:
            authors.append(name)

    # Date
    pub_date = data.get("publicationDate")
    year = int(pub_date[:4]) if pub_date and len(pub_date) >= 4 else None

    # Journal
    journal = None
    journal_info = data.get("journal")
    if journal_info:
        journal = journal_info.get("name") or None
    if not journal:
        journal = data.get("venue")

    # PDF URL
    pdf_url = None
    oa_pdf = data.get("openAccessPdf")
    if oa_pdf:
        pdf_url = oa_pdf.get("url")

    # Citation counts
    citation_count = data.get("citationCount")
    influential_count = data.get("influentialCitationCount")

    # Abstract
    abstract = data.get("abstract")

    # Publication types
    pub_types: list[str] = data.get("publicationTypes") or []

    # Reference count
    reference_count = data.get("referenceCount")

    # URL
    url = f"https://www.semanticscholar.org/paper/{data.get('paperId', '')}" if data.get("paperId") else None

    return PaperMetadata(
        doi=doi,
        title=title,
        authors=authors,
        publication_date=pub_date,
        year=year,
        journal=journal,
        citation_count=citation_count,
        influential_citation_count=influential_count,
        pdf_url=pdf_url,
        abstract=abstract,
        url=url,
        publication_types=pub_types,
        reference_count=reference_count,
        sources=["semantic_scholar"],
    )


async def query_by_doi(
    doi: str, client: httpx.AsyncClient
) -> PaperMetadata | None:
    """Query Semantic Scholar by DOI."""
    url = f"{S2_BASE}/paper/DOI:{doi}"
    params = {"fields": S2_FIELDS}
    try:
        resp = await client.get(url, params=params, timeout=S2_TIMEOUT)
        resp.raise_for_status()
        return _parse_s2_paper(resp.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (404, 429):
            return None
        logger.warning(f"Semantic Scholar HTTP error for {doi}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Semantic Scholar query failed for {doi}: {e}")
        return None


async def query_by_title(
    title: str,
    client: httpx.AsyncClient,
    authors: list[str] | None = None,
) -> PaperMetadata | None:
    """Query Semantic Scholar by title via search endpoint."""
    params = {
        "query": title,
        "limit": 3,
        "fields": S2_FIELDS,
    }
    try:
        resp = await client.get(
            f"{S2_BASE}/paper/search",
            params=params,
            timeout=S2_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data") or []
        if not items:
            return None
        return _parse_s2_paper(items[0])
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.debug("Semantic Scholar rate limited")
            return None
        logger.warning(f"Semantic Scholar search error: {e}")
        return None
    except Exception as e:
        logger.debug(f"Semantic Scholar search failed: {e}")
        return None
