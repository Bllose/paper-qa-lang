"""PDF parser using PyMuPDF — extracts text, figures, and tables per page.

Based on the original ``paper-qa-pymupdf`` reader (276 lines) but adapted
to use our own data models (``ParsedMedia`` instead of the original).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import pymupdf

from paper_qa_lang.models.types import ParsedMedia

logger = logging.getLogger(__name__)

# Pixmap attributes preserved as media metadata
_PIXMAP_ATTRS = {
    "alpha", "height", "width", "x", "y", "xres", "yres",
    "size", "stride", "n", "irect", "is_monochrome", "is_unicolor",
}


def is_likely_formula(text: str) -> bool:
    """Detect if a text line likely contains a mathematical formula."""
    t = text.strip()
    if not t or len(t) < 3:
        return False

    # Math symbols (Unicode)
    if any(c in '≤≥∑∫∂∞≈≠∈⊂⊃→⇒⇔√Δδαβγθλμπσφω±×÷' for c in t):
        return True

    # Formula number at end: (1), (2), ...
    if re.search(r'\([0-9]+\)\s*$', t):
        return True

    # Fraction: letter/digit )/( or )/{
    if re.search(r'[a-zA-Z0-9)]\s*/\s*[({]', t):
        return True

    # Subscript: T_c, P_D, x_{i}
    if re.search(r'\b[A-Za-z]_[a-zA-Z0-9]\b', t) or re.search(r'_[{][^}]*[}]', t):
        return True

    # Superscript: g^2, x^{n}
    if re.search(r'\^[0-9]', t) or re.search(r'\^[{][^}]*[}]', t):
        return True

    # Equal sign in short line (< 80 chars, likely an equation, not a sentence)
    if '=' in t and len(t) < 80:
        return True

    return False


def _formula_number(text: str) -> str | None:
    """Extract the formula number like ``(1)`` from text, or None."""
    m = re.search(r'\(([0-9]+)\)', text)
    return m.group(0) if m else None


class PDFParseError(Exception):
    """Raised when PDF parsing fails."""


def parse_pdf_to_pages(
    path: str | os.PathLike,
    page_size_limit: int | None = None,
    page_range: tuple[int, int] | None = None,
    parse_media: bool = True,
    image_cluster_tolerance: float | tuple[float, float] = 25,
    dpi: float | None = None,
    **kwargs: Any,
) -> dict[str, tuple[str, list[ParsedMedia]]]:
    """Parse a PDF into page-keyed text + media items.

    Args:
        path: Path to the PDF file.
        page_size_limit: Max chars per page (raises if exceeded).
        page_range: Inclusive (start_page, end_page), 1-indexed. ``None`` = all pages.
        parse_media: Extract figures and tables as images.
        image_cluster_tolerance: Tolerance (pts) for ``Page.cluster_drawings``.
            Single value or ``(x_tol, y_tol)`` tuple.
        dpi: Image rendering DPI (``None`` = PyMuPDF default).
        **kwargs: Ignored (for compatibility).

    Returns:
        ``{page_num_str: (page_text, [ParsedMedia, ...]), ...}``
    """
    x_tol, y_tol = (
        image_cluster_tolerance
        if isinstance(image_cluster_tolerance, tuple)
        else (image_cluster_tolerance, image_cluster_tolerance)
    )

    result: dict[str, tuple[str, list[ParsedMedia]]] = {}

    with pymupdf.open(path) as doc:
        page_count = doc.page_count
        pages = _resolve_page_range(page_range, page_count)

        for i in pages:
            page = doc.load_page(i)
            page_num = i + 1

            # Enhanced text extraction with formula detection
            text, formula_media = _extract_text_and_formulas(page, page_num)

            if page_size_limit and len(text) > page_size_limit:
                raise PDFParseError(
                    f"Page {i + 1} text ({len(text)} chars) exceeds "
                    f"page_size_limit={page_size_limit} in {path}."
                )

            media: list[ParsedMedia] = []
            if parse_media:
                media = _extract_page_media(page, i, x_tol, y_tol, dpi)

            result[str(page_num)] = (text, formula_media + media)

    return result


def _extract_text_and_formulas(
    page: pymupdf.Page,
    page_num: int,
) -> tuple[str, list[ParsedMedia]]:
    """Extract page text using ``dict`` mode, detecting and marking formulas.

    Returns ``(page_text_with_markers, formula_media_list)``.
    Formula markers ``[FORMULA_START]...[FORMULA_END]`` are inserted inline.
    """
    blocks = page.get_text("dict", sort=True)["blocks"]

    lines_out: list[str] = []
    formulas: list[ParsedMedia] = []
    formula_idx = 0

    for block in blocks:
        if block["type"] != 0:  # skip image blocks
            continue

        block_lines: list[str] = []
        line_bboxes: list[tuple[float, float, float, float]] = []
        for line in block["lines"]:
            line_text = "".join(span["text"] for span in line["spans"])
            block_lines.append(line_text)
            line_bboxes.append(line["bbox"])

        if not block_lines:
            continue

        non_empty = [l for l in block_lines if l.strip()]
        if not non_empty:
            continue

        all_formula = all(is_likely_formula(l) for l in non_empty)

        if all_formula and len(non_empty) <= 2:
            # Compact formula block → single formula
            combined = " ".join(non_empty)
            formula_idx += 1
            num = _formula_number(combined)
            lines_out.append(f"[FORMULA_START]{combined}[FORMULA_END]")
            formulas.append(ParsedMedia(
                index=formula_idx,
                data=b"",
                text=combined,
                media_type="formula",
                page_num=page_num,
                bbox=block["bbox"],
                metadata={"number": num, "content": combined, "page_num": page_num},
            ))

        else:
            # Mixed or multi-line → handle per-line
            for line_i, line in enumerate(block_lines):
                if line.strip() and is_likely_formula(line):
                    formula_idx += 1
                    num = _formula_number(line)
                    lines_out.append(f"[FORMULA_START]{line}[FORMULA_END]")
                    formulas.append(ParsedMedia(
                        index=formula_idx,
                        data=b"",
                        text=line,
                        media_type="formula",
                        page_num=page_num,
                        bbox=line_bboxes[line_i],
                        metadata={"number": num, "content": line, "page_num": page_num},
                    ))
                else:
                    lines_out.append(line)

    return "\n".join(lines_out), formulas


def _extract_page_media(
    page: pymupdf.Page,
    page_index: int,
    x_tol: float,
    y_tol: float,
    dpi: float | None,
) -> list[ParsedMedia]:
    """Extract figures (clustered drawings + embedded images) and tables from a single page."""
    media: list[ParsedMedia] = []
    page_num = page_index + 1

    page_rect = page.rect
    logger.info(
        "Page %s — dimensions: %.1f x %.1f pts (%.1f x %.1f in @ 72 DPI)",
        page_num, page_rect.width, page_rect.height,
        page_rect.width / 72, page_rect.height / 72,
    )

    # --- Figures (clustered drawings) ---
    for box_i, box in enumerate(
        page.cluster_drawings(
            drawings=page.get_drawings(),
            x_tolerance=x_tol,
            y_tolerance=y_tol,
        )
    ):
        pix = page.get_pixmap(clip=box, dpi=dpi)
        meta: dict[str, Any] = {"type": "drawing", "bbox": tuple(box)}
        meta.update({a: getattr(pix, a) for a in _PIXMAP_ATTRS})
        meta["info_hashable"] = json.dumps(meta, sort_keys=True)
        meta["page_num"] = page_num
        logger.info(
            "  [drawing %s] bbox=(%.1f, %.1f, %.1f, %.1f)  size=%.0fx%.0f pts",
            box_i, box.x0, box.y0, box.x1, box.y1,
            box.width, box.height,
        )
        media.append(
            ParsedMedia(
                index=box_i,
                data=pix.tobytes(),
                media_type="drawing",
                page_num=page_num,
                bbox=tuple(box) if box else None,
                metadata=meta,
            )
        )

    # --- Embedded images (Image XObjects) ---
    existing_bboxes = {m.bbox for m in media if m.bbox}
    img_index_offset = len(media)
    min_size = 50  # pts — filter out icons / decorations
    for img_i, img_info in enumerate(page.get_images()):
        xref = img_info[0]
        rects = page.get_image_rects(xref)
        for rect in rects:
            if rect.width < min_size and rect.height < min_size:
                logger.debug(
                    "  [image xref=%s] skipped — too small: %.0fx%.0f pts",
                    xref, rect.width, rect.height,
                )
                continue
            # Deduplicate against already-extracted bboxes
            rect_tuple = (rect.x0, rect.y0, rect.x1, rect.y1)
            if any(_bbox_overlaps(rect_tuple, eb, 5) for eb in existing_bboxes):
                continue
            pix = page.get_pixmap(clip=rect, dpi=dpi)
            meta: dict[str, Any] = {
                "type": "image_xobject",
                "xref": xref,
                "bbox": rect_tuple,
            }
            meta.update({a: getattr(pix, a) for a in _PIXMAP_ATTRS})
            meta["info_hashable"] = json.dumps(meta, sort_keys=True)
            meta["page_num"] = page_num
            logger.info(
                "  [image xref=%s] bbox=(%.1f, %.1f, %.1f, %.1f)  size=%.0fx%.0f pts",
                xref, rect.x0, rect.y0, rect.x1, rect.y1,
                rect.width, rect.height,
            )
            idx = img_index_offset + img_i
            media.append(
                ParsedMedia(
                    index=idx,
                    data=pix.tobytes(),
                    media_type="image_xobject",
                    page_num=page_num,
                    bbox=rect_tuple,
                    metadata=meta,
                )
            )
            existing_bboxes.add(rect_tuple)

    # --- Tables ---
    for table_i, table in enumerate(page.find_tables()):
        pix = page.get_pixmap(clip=table.bbox, dpi=dpi)
        meta = {"type": "table", "bbox": tuple(table.bbox)}
        meta.update({a: getattr(pix, a) for a in _PIXMAP_ATTRS})
        meta["info_hashable"] = json.dumps(meta, sort_keys=True)
        meta["page_num"] = page_num
        # Clean the markdown output — original paper-qa had issues with
        # orphaned surrogates and control characters in table markdown
        table_text = table.to_markdown().strip()
        media.append(
            ParsedMedia(
                index=table_i,
                data=pix.tobytes(),
                text=table_text if table_text else None,
                media_type="table",
                page_num=page_num,
                bbox=tuple(table.bbox) if table.bbox else None,
                metadata=meta,
            )
        )

    return media


def _bbox_overlaps(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    tolerance: float = 0,
) -> bool:
    """Check if two bboxes ``(x0, y0, x1, y1)`` overlap, with optional tolerance."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ax1 <= bx0 - tolerance or bx1 <= ax0 - tolerance:
        return False
    if ay1 <= by0 - tolerance or by1 <= ay0 - tolerance:
        return False
    return True


def _resolve_page_range(
    page_range: tuple[int, int] | None, page_count: int
) -> range:
    """Convert 1-indexed (start, end) → 0-indexed range."""
    if page_range is None:
        return range(page_count)
    start = max(0, page_range[0] - 1)
    end = min(page_range[1], page_count)
    return range(start, end)
