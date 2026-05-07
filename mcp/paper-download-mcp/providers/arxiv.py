"""arXiv PDF URL provider — no external API needed, derives URL from DOI pattern."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

ARXIV_DOI_PATTERN = re.compile(r"^10\.48550/arXiv\.(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)
ARXIV_OLD_PATTERN = re.compile(r"^10\.48550/arXiv\.([a-z\-]+(?:\.\w+)?/\d{7})(v\d+)?$", re.IGNORECASE)


def find_pdf_url(doi: str) -> str | None:
    """Derive arXiv PDF URL from DOI pattern.

    arXiv DOIs follow the pattern 10.48550/arXiv.XXXXX
    PDFs are available at https://arxiv.org/pdf/XXXXX.pdf

    Returns None if the DOI doesn't match arXiv pattern.
    """
    # Try new-style arXiv ID (e.g., 10.48550/arXiv.2301.12345)
    m = ARXIV_DOI_PATTERN.match(doi)
    if m:
        arxiv_id = m.group(1)
        version = m.group(2) or ""
        url = f"https://arxiv.org/pdf/{arxiv_id}{version}.pdf"
        logger.debug(f"Derived arXiv PDF URL: {url}")
        return url

    # Try old-style arXiv ID (e.g., 10.48550/arXiv.math/0612345)
    m = ARXIV_OLD_PATTERN.match(doi)
    if m:
        arxiv_id = m.group(1)
        version = m.group(2) or ""
        url = f"https://arxiv.org/pdf/{arxiv_id}{version}.pdf"
        logger.debug(f"Derived arXiv PDF URL: {url}")
        return url

    return None
