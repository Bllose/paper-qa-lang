"""Semantic Scholar provider for finding OA PDF URLs."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_TIMEOUT = 10.0


async def find_pdf_url(
    doi: str,
    client: httpx.AsyncClient,
) -> str | None:
    """Look up PDF URL from Semantic Scholar's openAccessPdf field.

    Returns None if no OA PDF or paper not found.
    """
    url = f"{S2_BASE}/paper/DOI:{doi}"
    params = {"fields": "openAccessPdf"}
    try:
        resp = await client.get(url, params=params, timeout=S2_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        oa_pdf = data.get("openAccessPdf")
        if oa_pdf and isinstance(oa_pdf, dict):
            pdf_url = oa_pdf.get("url")
            if pdf_url:
                logger.debug(f"Semantic Scholar found PDF URL for {doi}")
                return pdf_url

        logger.debug(f"Semantic Scholar: no OA PDF for {doi}")
        return None

    except httpx.HTTPStatusError as e:
        if e.response.status_code in (404, 429):
            return None
        logger.warning(f"Semantic Scholar HTTP error for {doi}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Semantic Scholar query failed for {doi}: {e}")
        return None
