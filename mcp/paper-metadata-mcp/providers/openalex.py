"""OpenAlex API provider for paper metadata."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from models import PaperMetadata

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_TIMEOUT = 15.0

# Polite pool: https://docs.openalex.org/how-to-use-the-api/rate-limits
# OpenAlex has a fast pool for API key holders and a polite pool for everyone else.
# The polite pool allows 10 requests per minute with Retry-After header.


def _inverted_index_to_text(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Convert OpenAlex's inverted index abstract to plain text."""
    if not inverted_index:
        return None
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


def _parse_openalex_work(data: dict[str, Any]) -> PaperMetadata | None:
    """Parse OpenAlex Work response into PaperMetadata."""
    doi = data.get("doi")
    if doi:
        # OpenAlex returns DOIs as full URLs
        doi = doi.replace("https://doi.org/", "")

    title = data.get("title")
    if not doi and not title:
        return None

    # Authors
    authors: list[str] = []
    for authorship in data.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)

    # Date
    pub_date = data.get("publication_date")

    # Biblio
    biblio = data.get("biblio") or {}
    volume = biblio.get("volume")
    issue = biblio.get("issue")
    pages = biblio.get("pages")

    # Journal / source
    primary_location = data.get("primary_location") or {}
    source = (primary_location.get("source")) or {}
    journal = source.get("display_name") or source.get("name")
    issn_list = source.get("issn") or []
    issn = issn_list[0] if issn_list else None
    is_oa = source.get("is_oa")

    # Publisher
    publisher = None
    if source:
        publisher = source.get("host_organization_name")

    # PDF URL
    pdf_url = None
    if primary_location:
        pdf_url = primary_location.get("pdf_url")
        if not pdf_url:
            oa_url = primary_location.get("landing_page_url")

    # Abstract
    abstract = _inverted_index_to_text(data.get("abstract_inverted_index"))

    # Citation count
    citation_count = data.get("cited_by_count")

    # Publication type
    pub_types = []
    if data.get("type"):
        pub_types.append(data["type"])

    # Year
    year = data.get("publication_year")

    # Reference count
    reference_count = data.get("referenced_works_count")

    # URL
    url = data.get("id")
    if url:
        url = f"https://openalex.org/{url.split('/')[-1]}"

    return PaperMetadata(
        doi=doi,
        title=title,
        authors=authors,
        publication_date=pub_date,
        year=year,
        journal=journal,
        volume=volume,
        issue=issue,
        pages=pages,
        publisher=publisher,
        citation_count=citation_count,
        pdf_url=pdf_url,
        url=url,
        issn=issn,
        is_open_access=is_oa,
        abstract=abstract,
        publication_types=pub_types,
        reference_count=reference_count,
        sources=["openalex"],
    )


async def query_by_doi(
    doi: str, client: httpx.AsyncClient
) -> PaperMetadata | None:
    """Query OpenAlex by DOI."""
    url = f"{OPENALEX_BASE}/works/doi:{doi}"
    try:
        resp = await client.get(url, timeout=OPENALEX_TIMEOUT)
        resp.raise_for_status()
        return _parse_openalex_work(resp.json())
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        logger.warning(f"OpenAlex HTTP error for {doi}: {e}")
        return None
    except Exception as e:
        logger.debug(f"OpenAlex query failed for {doi}: {e}")
        return None


async def query_by_title(
    title: str,
    client: httpx.AsyncClient,
    authors: list[str] | None = None,
) -> PaperMetadata | None:
    """Query OpenAlex by title."""
    params: dict[str, str | int] = {
        "search": title,
        "per-page": 3,
        "sort": "relevance_score:desc",
    }
    try:
        resp = await client.get(
            f"{OPENALEX_BASE}/works",
            params=params,
            timeout=OPENALEX_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        if not results:
            return None
        return _parse_openalex_work(results[0])
    except Exception as e:
        logger.debug(f"OpenAlex search failed: {e}")
        return None
