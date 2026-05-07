"""Multimodal media enrichment — uses an LLM to describe and filter images/tables.

After PDF parsing (text + media extraction) and before chunking, each media
item is sent to a multimodal LLM with co-located page text. The LLM:
1. Labels it RELEVANT or IRRELEVANT (scientific content vs decorative)
2. Produces a text description of the scientific content

The enriched description is then used in two ways:
- **Embedding enrichment**: appended to chunk text before vector embedding
  (improves retrieval recall for image-containing chunks)
- **Context creation**: sent as images in multimodal messages during QA
  (allows the LLM to "see" the figures when answering)
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from paper_qa_lang.models.types import ParsedMedia
from paper_qa_lang.utils import llm_content_text
from paper_qa_lang.prompts.templates import MEDIA_ENRICHMENT_PROMPT

logger = logging.getLogger(__name__)


def parse_enrichment_response(response: str) -> tuple[bool, str]:
    """Parse an LLM enrichment response into (is_irrelevant, description).

    The LLM is instructed to start with ``RELEVANT:`` or ``IRRELEVANT:``.
    """
    text = response.strip()
    if text.startswith("IRRELEVANT:"):
        description = text.removeprefix("IRRELEVANT:").strip()
        return True, description
    if text.startswith("RELEVANT:"):
        description = text.removeprefix("RELEVANT:").strip()
        return False, description
    # No label found — default to relevant, keep the full response
    logger.warning("Enrichment response missing RELEVANT/IRRELEVANT label: %.80s", text)
    return False, text


def enrich_pages_media(
    pages: dict[str, tuple[str, list[ParsedMedia]]],
    llm: BaseChatModel,
    page_radius: int = 1,
    max_image_size: int = 5 * 1024 * 1024,  # 5 MB (Anthropic limit)
) -> dict[str, tuple[str, list[ParsedMedia]]]:
    """Enrich all media in parsed pages with LLM descriptions.

    For each ``ParsedMedia`` with non-empty ``data``:
    1. Collect co-located text from surrounding pages (configurable radius).
    2. Call the multimodal LLM with the image + context text.
    3. Parse the response into an ``enriched_description`` and ``is_irrelevant`` flag.
    4. Filter out irrelevant media in-place.

    Args:
        pages: Output of ``parse_pdf_to_pages()``:
            ``{page_num: (page_text, [ParsedMedia])}``
        llm: A LangChain ``BaseChatModel`` that supports multimodal messages
            (e.g. ``ChatOpenAI``, ``ChatAnthropic``, ``ChatGoogleGenerativeAI``).
        page_radius: Number of surrounding pages to include as text context.
            ``-1`` = all pages, ``0`` = current page only, ``1`` = ±1 page.
        max_image_size: Maximum image bytes to send to LLM. Images larger
            than this are skipped with a warning. Default 5 MB (Anthropic limit).

    Returns:
        New pages dict with irrelevant media filtered out and
        ``enriched_description`` set on remaining media.
    """
    total_enriched = 0
    total_filtered = 0

    # Collect all media items with page context
    items_to_enrich: list[tuple[str, ParsedMedia]] = []
    for page_num, (_, media_list) in pages.items():
        for m in media_list:
            if m.data and not m.enriched_description:
                items_to_enrich.append((page_num, m))

    if not items_to_enrich:
        logger.info("No media items to enrich.")
        return pages

    logger.info("Enriching %d media items with LLM...", len(items_to_enrich))

    for page_num, media in items_to_enrich:
        if len(media.data) > max_image_size:
            logger.warning(
                "Skipping media index %d on page %s: %d bytes exceeds max %d",
                media.index, page_num, len(media.data), max_image_size,
            )
            continue

        # Collect surrounding text context
        context_text = _collect_context(pages, int(page_num), page_radius)

        # Build multimodal prompt
        prompt = MEDIA_ENRICHMENT_PROMPT.format(
            context_text=(
                f"Here is the co-located text:\n\n{context_text}\n\n"
                if context_text
                else ""
            )
        )

        try:
            # Build multimodal message with image embedded as data URL
            b64_data = base64.b64encode(media.data).decode("utf-8")
            image_url = f"data:image/png;base64,{b64_data}"

            msg = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )
            response = llm.invoke([msg])
            raw_text = llm_content_text(response.content)

            is_irrelevant, description = parse_enrichment_response(raw_text)

            # Store enrichment on the media item
            media.enriched_description = description
            media.metadata["is_irrelevant"] = is_irrelevant

            if is_irrelevant:
                total_filtered += 1
            else:
                total_enriched += 1

        except Exception as exc:
            logger.warning(
                "Enrichment failed for media index %d on page %s: %s",
                media.index, page_num, exc,
            )
            # Keep the media as-is on failure
            continue

    # Filter out irrelevant media from the result
    filtered_pages: dict[str, tuple[str, list[ParsedMedia]]] = {}
    for page_num, (text, media_list) in pages.items():
        kept = [m for m in media_list if not m.metadata.get("is_irrelevant", False)]
        filtered_pages[page_num] = (text, kept)
        total_filtered += len(media_list) - len(kept)

    logger.info(
        "Enrichment done: %d enriched, %d filtered",
        total_enriched, total_filtered,
    )
    return filtered_pages


def _collect_context(
    pages: dict[str, tuple[str, list[ParsedMedia]]],
    page_num: int,
    radius: int,
) -> str:
    """Collect text from surrounding pages as context for enrichment."""
    sorted_keys = sorted(pages.keys(), key=int)
    if radius == -1:
        relevant = sorted_keys
    else:
        idx = sorted_keys.index(str(page_num)) if str(page_num) in sorted_keys else -1
        if idx < 0:
            return ""
        start = max(0, idx - radius)
        end = min(len(sorted_keys), idx + radius + 1)
        relevant = sorted_keys[start:end]

    texts: list[str] = []
    for k in relevant:
        content = pages[k][0]
        if content:
            texts.append(f"[Page {k}]\n{content}")
    return "\n\n".join(texts)
