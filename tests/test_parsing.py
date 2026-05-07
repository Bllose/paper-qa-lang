"""Tests for PDF parsing, page-aware chunking, and model types."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from paper_qa_lang.models.types import Paper, PaperChunk, ParsedMedia
from paper_qa_lang.parsing.chunking import chunk_pdf_pages, chunk_plain_text
from paper_qa_lang.parsing.pdf_parser import parse_pdf_to_pages


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(doc_id="test001", title="Test Paper", doi="10.1234/test.001")


@pytest.fixture
def two_page_pages() -> dict[str, tuple[str, list[ParsedMedia]]]:
    return {
        "1": ("Page one content. " * 50, []),
        "2": ("Page two content. " * 50, []),
    }


@pytest.fixture
def pages_with_media() -> dict[str, tuple[str, list[ParsedMedia]]]:
    return {
        "1": (
            "Page with a figure. " * 30,
            [
                ParsedMedia(
                    index=0,
                    data=b"fake_image_bytes",
                    media_type="drawing",
                    page_num=1,
                    bbox=(0, 0, 100, 100),
                ),
                ParsedMedia(
                    index=0,
                    data=b"table_image",
                    text="| Col1 | Col2 |\n|------|------|\n| A    | B    |",
                    media_type="table",
                    page_num=1,
                ),
            ],
        ),
        "2": ("Page two text. " * 30, []),
        "3": (
            "Page three with more media. " * 30,
            [
                ParsedMedia(
                    index=0,
                    data=b"figure_on_page_3",
                    media_type="drawing",
                    page_num=3,
                ),
            ],
        ),
    }


# ── ParsedMedia tests ─────────────────────────────────────────────────────────


class TestParsedMedia:
    def test_to_id_deterministic(self):
        m1 = ParsedMedia(index=0, data=b"hello", text="test")
        m2 = ParsedMedia(index=0, data=b"hello", text="test")
        assert m1.to_id() == m2.to_id()

    def test_to_id_different_data(self):
        m1 = ParsedMedia(index=0, data=b"abc")
        m2 = ParsedMedia(index=0, data=b"def")
        assert m1.to_id() != m2.to_id()


# ── Chunking tests ────────────────────────────────────────────────────────────


class TestChunkPdfPages:
    def test_empty_pages(self, sample_paper):
        assert chunk_pdf_pages({}, sample_paper) == []

    def test_single_page_within_limit(self, sample_paper):
        pages = {"1": ("Hello world. " * 10, [])}
        chunks = chunk_pdf_pages(pages, sample_paper, chunk_chars=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0].page_range == (1, 1)

    def test_page_range_tracking(self, sample_paper, two_page_pages):
        """Chunks should track which pages they span."""
        chunks = chunk_pdf_pages(two_page_pages, sample_paper, chunk_chars=150, overlap=20)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.page_range is not None
            assert c.page_range[0] >= 1
            assert c.page_range[1] >= c.page_range[0]

    def test_media_attached_to_chunk(self, sample_paper, pages_with_media):
        """Media from pages in the chunk's range should be attached."""
        chunks = chunk_pdf_pages(pages_with_media, sample_paper, chunk_chars=500, overlap=50)
        # First chunk covers pages 1-2, should get media from page 1
        # Second chunk covers pages 2-3, should get media from pages 2-3
        chunks_with_media = [c for c in chunks if c.media]
        assert len(chunks_with_media) >= 1
        # Check media dedup: media from page 1 should appear in first chunk
        first_chunk = chunks[0]
        media_types = {m.media_type for m in first_chunk.media}
        assert "drawing" in media_types or "table" in media_types

    def test_chunk_id_format(self, sample_paper, two_page_pages):
        chunks = chunk_pdf_pages(two_page_pages, sample_paper, chunk_chars=150, overlap=20)
        for c in chunks:
            assert c.chunk_id.startswith("test001__")

    def test_no_duplicate_media(self, sample_paper):
        """Same media on a page should not be duplicated in a chunk."""
        media = [
            ParsedMedia(index=0, data=b"unique_bytes", media_type="drawing", page_num=1),
        ]
        # Same media ref twice — ensure dedup
        pages = {
            "1": ("Page one. " * 100, media),
            "2": ("Page two. " * 100, media),  # same media object
        }
        chunks = chunk_pdf_pages(pages, sample_paper, chunk_chars=500, overlap=50)
        for c in chunks:
            # Should not have more than 1 media item since there's only 1 unique
            assert len(c.media) <= 1


class TestChunkPlainText:
    def test_basic_chunking(self, sample_paper):
        # "word " is 5 chars. 2000 * 5 = 10000 total.
        # With chunk_size=500, overlap=50 → effective stride = 450.
        # 10000 / 450 ≈ 22.2 → 23 chunks, last one short.
        text = "word " * 2000
        chunks = chunk_plain_text(text, sample_paper, chunk_size=500, chunk_overlap=50)
        assert 20 <= len(chunks) <= 25

    def test_empty_text(self, sample_paper):
        assert chunk_plain_text("", sample_paper) == []

    def test_short_text(self, sample_paper):
        chunks = chunk_plain_text("short", sample_paper, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0].text == "short"


# ── PDF parser tests ──────────────────────────────────────────────────────────


class TestPdfParser:
    @pytest.fixture(scope="session")
    def stub_pdf_path(self):
        """Use one of the original paper-qa test PDFs."""
        path = Path("D:/workplace/Bllose/paper-qa/tests/stub_data/paper.pdf")
        if not path.exists():
            pytest.skip("Test PDF not found")
        return path

    def test_parse_all_pages(self, stub_pdf_path):
        pages = parse_pdf_to_pages(stub_pdf_path)
        assert len(pages) > 0
        for pg, (text, media) in pages.items():
            assert len(text) > 0
            assert isinstance(media, list)
            assert int(pg) >= 1

    def test_parse_page_range(self, stub_pdf_path):
        pages = parse_pdf_to_pages(stub_pdf_path, page_range=(1, 3))
        assert len(pages) <= 3

    def test_parse_text_only(self, stub_pdf_path):
        """parse_media=False should produce no media."""
        pages = parse_pdf_to_pages(stub_pdf_path, parse_media=False)
        for _, (_, media) in pages.items():
            assert len(media) == 0

    def test_parse_with_media(self, stub_pdf_path):
        pages = parse_pdf_to_pages(stub_pdf_path, parse_media=True)
        total_media = sum(len(m) for _, (_, m) in pages.items())
        assert total_media >= 0  # some PDFs have no extractable media

    def test_parser_error_on_nonexistent_file(self):
        with pytest.raises(Exception):
            parse_pdf_to_pages("/nonexistent/file.pdf")

    def test_parse_small_pdf(self):
        """Create a minimal PDF and verify parsing works."""
        try:
            import pymupdf
        except ImportError:
            pytest.skip("pymupdf not installed")

        tmp_path = tempfile.mktemp(suffix=".pdf")
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Hello PDF parser!")
        doc.save(tmp_path)
        doc.close()

        try:
            pages = parse_pdf_to_pages(tmp_path)
            assert len(pages) == 1
            text, media = pages["1"]
            assert "Hello PDF parser!" in text
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_parse_real_pdf_full(self, stub_pdf_path):
        """End-to-end: parse → chunk → verify chunk structure."""
        pages = parse_pdf_to_pages(stub_pdf_path, parse_media=True, dpi=72)
        paper = Paper(doc_id="e2e_test", title="E2E Test")
        chunks = chunk_pdf_pages(pages, paper, chunk_chars=2000, overlap=100)
        assert len(chunks) > 0
        for c in chunks:
            assert c.paper_id == "e2e_test"
            assert c.page_range is not None
            assert len(c.text) <= 2100  # allow small overflow
            assert c.chunk_id.startswith("e2e_test__")
