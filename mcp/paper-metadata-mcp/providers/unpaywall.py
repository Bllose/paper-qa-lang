"""Unpaywall API provider for open access information."""

from __future__ import annotations

import logging
import os

import httpx

from models import PaperMetadata

logger = logging.getLogger(__name__)

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
UNPAYWALL_TIMEOUT = 10.0


async def query_by_doi(
    doi: str,
    client: httpx.AsyncClient,
    email: str | None = None,
) -> PaperMetadata | None:
    """Query Unpaywall by DOI for open access info.

    Note: Unpaywall requires an email parameter. Provide it via
    the UNPAYWALL_EMAIL environment variable or directly.
    """
    email = email or os.environ.get("UNPAYWALL_EMAIL")
    if not email:
        logger.debug("Unpaywall: no email configured, skipping")
        return None

    url = f"{UNPAYWALL_BASE}/{doi}"
    params = {"email": email}
    try:
        resp = await client.get(url, params=params, timeout=UNPAYWALL_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        # PDF URL from best OA location
        pdf_url = None
        best_loc = data.get("best_oa_location") or {}
        if best_loc.get("url_for_pdf"):
            pdf_url = best_loc["url_for_pdf"]

        # Journal info
        journal = data.get("journal_name")

        # Date
        pub_date = data.get("published_date")
        year = None
        if data.get("year"):
            year = int(data["year"])

        authors_raw = data.get("authors", [])
        authors = []
        for a in authors_raw:
            if isinstance(a, dict):
                name = a.get("name") or a.get("family_name", "")
                if a.get("given_name"):
                    name = f"{a['family_name']}, {a['given_name']}"
                if name:
                    authors.append(name)

        return PaperMetadata(
            doi=data.get("doi", doi),
            title=data.get("title"),
            authors=authors,
            publication_date=pub_date,
            year=year,
            journal=journal,
            pdf_url=pdf_url,
            is_open_access=data.get("is_oa"),
            url=data.get("doi_url"),
            sources=["unpaywall"],
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (404, 403):
            return None
        logger.warning(f"Unpaywall HTTP error for {doi}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unpaywall query failed for {doi}: {e}")
        return None
