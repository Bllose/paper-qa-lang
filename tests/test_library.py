"""Tests for PaperLibrary — chunk storage, metadata, and PDF ingestion."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from paper_qa_lang.config.settings import Settings
from paper_qa_lang.models.types import Paper, PaperChunk, ParsedMedia
from paper_qa_lang.store.paper_library import PaperLibrary


@pytest.fixture
def tmp_dir() -> str:
    """Create a temp directory for Chroma + SQLite data."""
    d = tempfile.mkdtemp()
    yield d
    # Cleanup — Chroma holds file locks on Windows, best-effort
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


class TestPaperLibrary:
    def test_empty_library(self, lib):
        assert lib.paper_count == 0
        assert lib.chunk_count == 0
        assert lib.list_papers() == []

    def test_add_chunks_no_chunks_raises(self, lib, sample_paper):
        with pytest.raises(ValueError, match="No chunks"):
            lib.add_chunks(sample_paper, [])

    def test_add_and_retrieve_chunks(self, lib, sample_paper):
        chunks = [
            PaperChunk(chunk_id="c1", text="Machine learning is great.", paper_id="test001"),
            PaperChunk(chunk_id="c2", text="Deep learning is also great.", paper_id="test001"),
        ]
        n = lib.add_chunks(sample_paper, chunks)
        assert n == 2
        assert lib.chunk_count == 2

    def test_search_returns_results(self, lib, sample_paper):
        chunks = [
            PaperChunk(chunk_id="c1", text="Machine learning for biology.", paper_id="test001"),
            PaperChunk(chunk_id="c2", text="Quantum physics is interesting.", paper_id="test001"),
            PaperChunk(chunk_id="c3", text="Biology and machine learning intersect.", paper_id="test001"),
        ]
        lib.add_chunks(sample_paper, chunks)
        results = lib.search("biology", k=2)
        assert len(results) == 2
        # Results should be PaperChunk objects
        for r in results:
            assert isinstance(r, PaperChunk)
            assert r.paper_id == "test001"

    def test_query_metadata(self, lib, sample_paper):
        chunks = [PaperChunk(chunk_id="c1", text="test", paper_id="test001")]
        lib.add_chunks(sample_paper, chunks)
        retrieved = lib.query_metadata("test001")
        assert retrieved is not None
        assert retrieved.title == "Test Paper"
        assert retrieved.doi == "10.1234/test.001"

    def test_query_metadata_not_found(self, lib):
        assert lib.query_metadata("nonexistent") is None

    def test_list_papers(self, lib):
        p1 = Paper(doc_id="p1", title="Paper 1")
        p2 = Paper(doc_id="p2", title="Paper 2")
        lib.add_chunks(p1, [PaperChunk(chunk_id="a", text="t1", paper_id="p1")])
        lib.add_chunks(p2, [PaperChunk(chunk_id="b", text="t2", paper_id="p2")])
        papers = lib.list_papers()
        assert len(papers) == 2

    def test_delete_paper(self, lib, sample_paper):
        chunks = [
            PaperChunk(chunk_id="c1", text="Text content.", paper_id="test001"),
        ]
        lib.add_chunks(sample_paper, chunks)
        assert lib.paper_count == 1
        lib.delete_paper("test001")
        assert lib.paper_count == 0
        assert lib.query_metadata("test001") is None

    def test_add_paper_with_chunks(self, lib, sample_paper):
        n = lib.add_paper_with_chunks(sample_paper, "Hello world. " * 500, chunk_size=200, chunk_overlap=20)
        assert n > 1
        assert lib.chunk_count == n
        assert lib.paper_count == 1

    def test_multiple_papers_search(self, lib):
        p1 = Paper(doc_id="bio", title="Biology paper")
        p2 = Paper(doc_id="physics", title="Physics paper")
        lib.add_chunks(p1, [
            PaperChunk(chunk_id="b1", text="DNA and RNA are molecules.", paper_id="bio"),
            PaperChunk(chunk_id="b2", text="Proteins are made of amino acids.", paper_id="bio"),
        ])
        lib.add_chunks(p2, [
            PaperChunk(chunk_id="p1", text="Quantum mechanics is weird.", paper_id="physics"),
            PaperChunk(chunk_id="p2", text="E=mc^2 is famous.", paper_id="physics"),
        ])
        bio_results = lib.search("DNA RNA", k=5)
        assert any("DNA" in r.text for r in bio_results)


class TestPaperLibraryPDFIngestion:
    """Integration tests for paper ingestion from real PDFs."""

    @pytest.fixture(scope="class")
    def stub_pdf_path(self):
        path = Path("D:/workplace/Bllose/paper-qa/tests/stub_data/paper.pdf")
        if not path.exists():
            pytest.skip("Test PDF not found")
        return str(path)

    def test_ingest_pdf(self, lib, stub_pdf_path):
        paper = Paper(doc_id="pdf_test", title="PDF Ingest Test")
        n = lib.add_paper(
            stub_pdf_path,
            paper,
            parse_media=True,
            chunk_chars=3000,
            overlap=200,
            dpi=72,
        )
        assert n > 1
        assert lib.paper_count == 1
        assert lib.chunk_count == n

        # Verify metadata stored
        retrieved = lib.query_metadata("pdf_test")
        assert retrieved is not None
        assert retrieved.title == "PDF Ingest Test"
        assert retrieved.file_location == stub_pdf_path

    def test_ingest_pdf_searchable(self, lib, stub_pdf_path):
        """After PDF ingestion, search should find relevant content."""
        paper = Paper(doc_id="pdf_search", title="Searchable PDF")
        lib.add_paper(
            stub_pdf_path,
            paper,
            parse_media=True,
            chunk_chars=2000,
            overlap=100,
            dpi=72,
        )

        # Search for something likely in the PDF
        results = lib.search("model", k=5)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, PaperChunk)
            assert r.paper_id == "pdf_search"

    def test_ingest_pdf_text_only(self, lib, stub_pdf_path):
        """parse_media=False should still produce valid chunks."""
        paper = Paper(doc_id="pdf_text", title="Text Only")
        n = lib.add_paper(
            stub_pdf_path,
            paper,
            parse_media=False,
            chunk_chars=3000,
            overlap=200,
        )
        assert n > 1

    def test_ingest_pdf_then_delete(self, lib, stub_pdf_path):
        paper = Paper(doc_id="pdf_del", title="To be deleted")
        lib.add_paper(stub_pdf_path, paper, parse_media=False, chunk_chars=5000, overlap=100)
        assert lib.paper_count == 1
        lib.delete_paper("pdf_del")
        assert lib.paper_count == 0

    def test_ingest_missing_file(self, lib):
        paper = Paper(doc_id="missing", title="Missing")
        with pytest.raises(FileNotFoundError):
            lib.add_paper("/nonexistent/file.pdf", paper)

    def test_ingest_text_file(self, lib, tmp_dir):
        """Non-PDF files should use text ingestion."""
        text_path = os.path.join(tmp_dir, "sample.txt")
        with open(text_path, "w") as f:
            f.write("This is a sample text file. " * 200)

        paper = Paper(doc_id="txt_test", title="Text File")
        n = lib.add_paper(text_path, paper, chunk_chars=500, overlap=50)
        assert n > 1
        assert lib.paper_count == 1

        # Verify search works
        results = lib.search("sample", k=3)
        assert len(results) > 0
