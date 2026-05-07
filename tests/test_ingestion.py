"""Tests for ingestion pipeline: identify, dedup, ingest, and query."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from paper_qa_lang.config.settings import Settings
from paper_qa_lang.models.types import Paper, PaperChunk
from paper_qa_lang.store.paper_library import PaperLibrary, QueryResult
from paper_qa_lang.ingestion.identify import (
    _parse_llm_metadata_response,
    _extract_title_fallback,
)


@pytest.fixture
def tmp_dir() -> str:
    d = tempfile.mkdtemp()
    yield d
    try:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def settings(tmp_dir: str) -> Settings:
    return Settings(
        store={
            "persist_directory": os.path.join(tmp_dir, ".chroma"),
            "metadata_db_path": os.path.join(tmp_dir, ".papers.db"),
        },
    )


@pytest.fixture
def lib(settings: Settings) -> PaperLibrary:
    return PaperLibrary(settings=settings)


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(doc_id="test001", title="Test Paper", doi="10.1234/test.001")


# ── Content deduplication ────────────────────────────────────────────────────


class TestContentDedup:
    def test_add_same_file_twice_raises(self, lib, tmp_dir):
        """Adding the same file twice should raise ValueError due to dedup."""
        text_path = os.path.join(tmp_dir, "duplicate.txt")
        with open(text_path, "w") as f:
            f.write("Some unique content for dedup testing. " * 50)

        paper1 = Paper(doc_id="dup1", title="First add")
        lib.add_paper(text_path, paper1, chunk_chars=500, overlap=50)

        paper2 = Paper(doc_id="dup2", title="Second add")
        with pytest.raises(ValueError, match="already indexed"):
            lib.add_paper(text_path, paper2, chunk_chars=500, overlap=50)

    def test_different_files_no_dedup(self, lib, tmp_dir):
        """Different files should both be added successfully."""
        path1 = os.path.join(tmp_dir, "file1.txt")
        path2 = os.path.join(tmp_dir, "file2.txt")
        with open(path1, "w") as f:
            f.write("Content of file one. " * 50)
        with open(path2, "w") as f:
            f.write("Content of file two, different. " * 50)

        lib.add_paper(path1, Paper(doc_id="f1", title="File 1"), chunk_chars=500, overlap=50)
        lib.add_paper(path2, Paper(doc_id="f2", title="File 2"), chunk_chars=500, overlap=50)
        assert lib.paper_count == 2


# ── Document validation (_looks_like_text) ────────────────────────────────────


class TestLooksLikeText:
    """Indirectly tested via add_paper's internal validation."""

    def test_empty_text_file_raises(self, lib, tmp_dir):
        """Empty file should raise ValueError."""
        path = os.path.join(tmp_dir, "empty.txt")
        Path(path).write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No text was extracted"):
            lib.add_paper(path, Paper(doc_id="empty", title="Empty"), chunk_chars=500, overlap=50)

    def test_binary_file_raises(self, lib, tmp_dir):
        """Binary/gibberish file should fail _looks_like_text check."""
        path = os.path.join(tmp_dir, "binary.bin")
        with open(path, "wb") as f:
            f.write(b"\x00\x01\x02\x03\xff\xfe\xfd\xfc" * 100)
        with pytest.raises(ValueError, match="does not appear"):
            lib.add_paper(path, Paper(doc_id="binary", title="Binary"), chunk_chars=500, overlap=50)

    def test_valid_text_succeeds(self, lib, tmp_dir):
        """Valid text file should be ingested successfully."""
        path = os.path.join(tmp_dir, "valid.txt")
        with open(path, "w") as f:
            f.write("This is valid English text for testing purposes. " * 100)
        n = lib.add_paper(path, Paper(doc_id="valid", title="Valid"), chunk_chars=500, overlap=50)
        assert n > 0
        assert lib.paper_count == 1


# ── ingest() method (sync parts only — MCP/LLM require external services) ────


class TestIngestMethod:
    def test_ingest_no_args_raises(self, lib):
        """ingest() with no arguments should raise ValueError."""
        import asyncio
        with pytest.raises(ValueError, match="Provide at least"):
            asyncio.run(lib.ingest())

    def test_ingest_with_path_and_paper(self, lib, tmp_dir):
        """ingest(path=..., paper=...) should work like add_paper."""
        path = os.path.join(tmp_dir, "ingest_test.txt")
        with open(path, "w") as f:
            f.write("Ingest method test content. " * 100)

        import asyncio
        n = asyncio.run(
            lib.ingest(path=path, paper=Paper(doc_id="ingest1", title="Ingest Test"))
        )
        assert n > 0
        assert lib.paper_count == 1

        # Verify content was stored
        results = lib.search("Ingest method", k=3)
        assert len(results) > 0


# ── QueryResult dataclass ────────────────────────────────────────────────────


class TestQueryResult:
    def test_query_result_defaults(self):
        result = QueryResult()
        assert result.answer == ""
        assert result.contexts == []
        assert result.question == ""

    def test_query_result_with_values(self):
        result = QueryResult(
            answer="Some answer text.",
            question="What is the answer?",
        )
        assert result.answer == "Some answer text."
        assert result.question == "What is the answer?"
        assert result.contexts == []


# ── Identity helpers ─────────────────────────────────────────────────────────


class TestParseLLMMetadataResponse:
    def test_plain_json(self):
        data = _parse_llm_metadata_response(
            MockResponse('{"title": "Test", "doi": "10.1234/test"}')
        )
        assert data["title"] == "Test"
        assert data["doi"] == "10.1234/test"

    def test_markdown_fenced_json(self):
        data = _parse_llm_metadata_response(
            MockResponse('```json\n{"title": "Fenced", "year": 2024}\n```')
        )
        assert data["title"] == "Fenced"
        assert data["year"] == 2024

    def test_invalid_json_returns_empty(self):
        data = _parse_llm_metadata_response(
            MockResponse("This is not JSON at all")
        )
        assert data == {}

    def test_none_response(self):
        data = _parse_llm_metadata_response(None)
        assert data == {}


class MockResponse:
    """Minimal AIMessage-like object for testing."""
    def __init__(self, text: str):
        self.text = text


class TestExtractTitleFallback:
    def test_extracts_title_from_first_lines(self):
        text = "123 - Machine Learning in Biology\nJohn Doe\nAbstract\n"
        title = _extract_title_fallback(text)
        assert "Machine Learning" in title

    def test_skips_copyright_lines(self):
        text = "Copyright 2024 Some Publisher\nDOI: 10.1234/test\nThe Real Title Is Here\n"
        title = _extract_title_fallback(text)
        assert title == "The Real Title Is Here"

    def test_short_lines_skipped(self):
        text = "Short\n" * 10
        title = _extract_title_fallback(text)
        assert title == "Unknown Paper"

    def test_empty_text(self):
        assert _extract_title_fallback("") == "Unknown Paper"
