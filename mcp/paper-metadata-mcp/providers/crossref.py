"""Crossref API provider for paper metadata."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from models import PaperMetadata

logger = logging.getLogger(__name__)

CROSSREF_BASE = "https://api.crossref.org"
CROSSREF_TIMEOUT = 15.0


def _parse_crossref_item(item: dict[str, Any]) -> PaperMetadata:
    """Parse a Crossref work item into PaperMetadata."""
    # Title
    title = (item.get("title") or [None])[0]

    # Authors
    authors: list[str] = []
    for author in item.get("author") or []:
        given = author.get("given", "")
        family = author.get("family", "")
        if given and family:
            authors.append(f"{family}, {given}")
        elif family:
            authors.append(family)

    # Date
    date_parts = None
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (item.get(key) or {}).get("date-parts")
        if parts:
            date_parts = parts[0]
            break
    year = date_parts[0] if date_parts else None
    pub_date = "-".join(str(p) for p in date_parts) if date_parts else None

    # Journal info
    container = (item.get("container-title") or [None])[0]

    # ISBN/ISSN
    issn_list = item.get("ISSN") or item.get("issn") or []
    issn = issn_list[0] if issn_list else None

    # URL
    url = item.get("URL") or None

    # Abstract
    abstract = item.get("abstract")

    # Publication type
    pub_types = [item["type"]] if item.get("type") else []

    # References
    reference_count = item.get("reference-count")

    # Publisher
    publisher = item.get("publisher")

    # Page info
    volume = item.get("volume")
    issue = item.get("issue")
    pages = item.get("page")

    return PaperMetadata(
        doi=item.get("DOI"),
        title=title,
        authors=authors,
        publication_date=pub_date,
        year=year,
        journal=container,
        volume=volume,
        issue=issue,
        pages=pages,
        publisher=publisher,
        url=url,
        issn=issn,
        abstract=abstract,
        publication_types=pub_types,
        reference_count=reference_count,
        sources=["crossref"],
    )


async def query_by_doi(
    doi: str, client: httpx.AsyncClient
) -> PaperMetadata | None:
    """Query Crossref by DOI."""
    url = f"{CROSSREF_BASE}/works/{doi}"
    try:
        resp = await client.get(url, timeout=CROSSREF_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        item = data.get("message", {})
        metadata = _parse_crossref_item(item)

        # Try to get BibTeX separately
        try:
            bib_resp = await client.get(
                f"{CROSSREF_BASE}/works/{doi}/transform/application/x-bibtex",
                timeout=CROSSREF_TIMEOUT,
            )
            if bib_resp.status_code == 200:
                metadata.bibtex = bib_resp.text.strip()
        except Exception as e:
            logger.debug(f"BibTeX fetch failed for {doi}: {e}")

        return metadata
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.debug(f"Crossref: DOI {doi} not found")
            return None
        logger.warning(f"Crossref HTTP error for {doi}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Crossref query failed for {doi}: {e}")
        return None


async def query_by_title(
    title: str,
    client: httpx.AsyncClient,
    authors: list[str] | None = None,
    rows: int = 3,
) -> PaperMetadata | None:
    """Query Crossref by title (and optionally authors)."""
    params: dict[str, str | int] = {
        "query.title": title,
        "rows": rows,
    }
    if authors:
        params["query.author"] = " ".join(authors)

    try:
        resp = await client.get(
            f"{CROSSREF_BASE}/works",
            params=params,
            timeout=CROSSREF_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        items = (data.get("message") or {}).get("items", [])
        if not items:
            return None

        # Return the best matching item
        best = _parse_crossref_item(items[0])
        return best
    except Exception as e:
        logger.warning(f"Crossref title query failed for {title!r}: {e}")
        return None
