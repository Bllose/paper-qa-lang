"""Page-aware chunking — accumulates pages until size limit, preserves media.

Based on the original ``paper-qa.readers.chunk_pdf()`` but adapted for
our data models and with clearer separation of concerns.
"""

from __future__ import annotations

from typing import Any

from paper_qa_lang.models.types import Paper, PaperChunk, ParsedMedia


def chunk_pdf_pages(
    pages: dict[str, tuple[str, list[ParsedMedia]]],
    paper: Paper,
    chunk_chars: int = 3000,
    overlap: int = 200,
) -> list[PaperChunk]:
    """Chunk a page-keyed text+media dict into ``PaperChunk``\\ s.

    Pages are accumulated sequentially. When the accumulated text exceeds
    ``chunk_chars``, a chunk is emitted for the accumulated page range.
    Media from all pages in that range are attached to the chunk.

    Args:
        pages: Output of ``parse_pdf_to_pages()``:
            ``{page_num_str: (page_text, [ParsedMedia])}``
        paper: The paper these chunks belong to.
        chunk_chars: Target chunk size in characters.
        overlap: Overlap between chunks in characters.

    Returns:
        List of ``PaperChunk``\\ s with ``page_range`` and ``media`` populated.
    """
    if not pages:
        return []

    chunks: list[PaperChunk] = []
    accumulated_text = ""
    accumulated_page_nums: list[str] = []
    chunk_index = 0

    # Sort pages by numeric order
    sorted_pages = sorted(pages.items(), key=lambda x: int(x[0]))

    for page_num_str, (page_text, media) in sorted_pages:
        accumulated_text += page_text
        accumulated_page_nums.append(page_num_str)

        while len(accumulated_text) > chunk_chars:
            chunk_text = accumulated_text[:chunk_chars]
            chunks.append(
                _build_chunk(
                    chunk_text, paper, accumulated_page_nums, pages, chunk_index
                )
            )
            chunk_index += 1
            accumulated_text = accumulated_text[chunk_chars - overlap :]
            accumulated_page_nums = [page_num_str]

    # Final chunk (if there's meaningful content left or no chunks were created)
    if len(accumulated_text) > overlap or not chunks:
        chunks.append(
            _build_chunk(
                accumulated_text[:chunk_chars], paper, accumulated_page_nums, pages, chunk_index
            )
        )

    return chunks


def _build_chunk(
    text: str,
    paper: Paper,
    page_nums: list[str],
    all_pages: dict[str, tuple[str, list[ParsedMedia]]],
    chunk_index: int = 0,
) -> PaperChunk:
    """Build a single PaperChunk, collecting media from its page range."""
    first_page = int(page_nums[0])
    last_page = int(page_nums[-1])

    # Collect media from all pages in this chunk's range
    media: list[ParsedMedia] = []
    seen_ids: set[str] = set()
    for pn in range(first_page, last_page + 1):
        pn_str = str(pn)
        if pn_str in all_pages:
            for m in all_pages[pn_str][1]:
                mid = str(m.to_id())
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    media.append(m)

    chunk_id = f"{paper.doc_id}__p{first_page:04d}-p{last_page:04d}__i{chunk_index:04d}"

    return PaperChunk(
        chunk_id=chunk_id,
        text=text,
        paper_id=paper.doc_id,
        page_num=first_page,
        page_range=(first_page, last_page),
        media=media,
    )


def chunk_plain_text(
    text: str,
    paper: Paper,
    chunk_size: int = 3000,
    chunk_overlap: int = 200,
) -> list[PaperChunk]:
    """Chunk plain text (no page structure, no media).

    Falls back to a simple character-based sliding-window for text-like
    content (e.g. plain text files, HTML).
    """
    chunks: list[PaperChunk] = []
    start = 0
    index = 0

    while start < len(text) or (not chunks and start == 0):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        if not chunk_text:
            break

        chunks.append(
            PaperChunk(
                chunk_id=f"{paper.doc_id}__chunk_{index:04d}",
                text=chunk_text,
                paper_id=paper.doc_id,
            )
        )
        index += 1
        start += chunk_size - chunk_overlap
        if end == len(text):
            break

    return chunks
