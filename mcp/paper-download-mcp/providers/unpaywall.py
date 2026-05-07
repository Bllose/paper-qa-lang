"""Unpaywall provider for finding OA PDF URLs."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
UNPAYWALL_TIMEOUT = 10.0


async def find_pdf_url(
    doi: str,
    client: httpx.AsyncClient,
    email: str | None = None,
) -> str | None:
    """Look up PDF URL from Unpaywall's best OA location.

    Requires UNPAYWALL_EMAIL env var or explicit email parameter.
    Returns None if no OA location or email not configured.
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

        best_loc = data.get("best_oa_location") or {}
        pdf_url = best_loc.get("url_for_pdf")
        if pdf_url:
            logger.debug(f"Unpaywall found PDF URL for {doi}")
            return pdf_url

        logger.debug(f"Unpaywall: no OA PDF location for {doi}")
        return None

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.debug(f"Unpaywall: DOI {doi} not found")
            return None
        logger.warning(f"Unpaywall HTTP error for {doi}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unpaywall query failed for {doi}: {e}")
        return None
