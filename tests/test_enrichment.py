"""Tests for media enrichment and context builder — real LLM calls."""
from __future__ import annotations

import base64
import os

import pytest

from paper_qa_lang.config import Settings, get_chat_model
from paper_qa_lang.models.types import Paper, PaperChunk, ParsedMedia
from paper_qa_lang.parsing.context_builder import ContextBuilder, ScoredContext
from paper_qa_lang.parsing.enrichment import (
    enrich_pages_media,
    parse_enrichment_response,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _tiny_png() -> bytes:
    """A minimal valid 1×1 red PNG (bare minimum for multimodal tests)."""
    # PNG: IHDR (1×1 8-bit RGB), IDAT (raw filter byte + RGB), IEND
    import struct
    import zlib

    # Build IDAT: filter byte 0x00 then RGB(255,0,0)
    raw = b"\x00" + bytes([255, 0, 0])
    compressed = zlib.compress(raw)
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13) + b"IHDR" + ihdr_data + ihdr_crc
        + struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc
        + struct.pack(">I", 0) + b"IEND"
        + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    )


# ── Enrichment parser tests (no LLM needed) ────────────────────────────────────


class TestParseEnrichmentResponse:
    def test_relevant_label(self):
        is_irr, desc = parse_enrichment_response(
            "RELEVANT: This figure shows a bar chart of expression levels."
        )
        assert is_irr is False
        assert "bar chart" in desc

    def test_irrelevant_label(self):
        is_irr, desc = parse_enrichment_response(
            "IRRELEVANT: This is a journal logo."
        )
        assert is_irr is True
        assert "logo" in desc

    def test_missing_label(self):
        is_irr, desc = parse_enrichment_response(
            "This is a description without a label."
        )
        assert is_irr is False  # default to relevant
        assert desc == "This is a description without a label."

    def test_label_with_surrounding_whitespace(self):
        is_irr, desc = parse_enrichment_response(
            "  IRRELEVANT:  A decorative margin element.  "
        )
        assert is_irr is True
        assert "decorative" in desc


# ── EnrichPagesMedia tests (real LLM) ──────────────────────────────────────────


@pytest.fixture(scope="session")
def llm():
    """Real LangChain chat model (DeepSeek V4 Flash via Anthropic-compatible API)."""
    if not _has_api_key():
        pytest.skip("ANTHROPIC_API_KEY not set — skipping real LLM tests")
    return get_chat_model(
        model_name="deepseek-v4-flash",
        base_url="https://api.deepseek.com/anthropic",
    )


class TestEnrichPagesMedia:
    def test_skip_media_without_data(self):
        """Media with empty data should be skipped."""
        pages = {
            "1": ("Text", [ParsedMedia(index=0, data=b"", media_type="drawing")]),
        }
        result = enrich_pages_media(pages, llm=None)
        assert result == pages

    def test_skip_media_with_existing_enrichment(self):
        """Already-enriched media should not be re-enriched."""
        media = ParsedMedia(
            index=0, data=b"image", media_type="drawing",
            enriched_description="Already described.",
        )
        pages = {"1": ("Text", [media])}
        result = enrich_pages_media(pages, llm=None)
        assert result["1"][1][0].enriched_description == "Already described."

    def test_enrich_valid_image(self, llm):
        """LLM should enrich a valid image with a description."""
        png = _tiny_png()
        media = ParsedMedia(index=0, data=png, media_type="drawing", page_num=1)
        pages = {"1": ("Figure 1 shows experimental results.", [media])}
        result = enrich_pages_media(pages, llm, page_radius=0)
        enriched = result["1"][1][0]
        assert enriched.enriched_description is not None
        assert len(enriched.enriched_description) > 0
        # Should be labeled as relevant (it's a real figure)
        assert enriched.metadata.get("is_irrelevant") is False

    def test_page_context_included_in_enrichment(self, llm):
        """Surrounding page text should influence the enrichment description."""
        png = _tiny_png()
        pages = {
            "1": (
                "We analyzed gene expression in cancer samples.",
                [ParsedMedia(index=0, data=png, media_type="drawing", page_num=1)],
            ),
            "2": ("The results show significant upregulation of TP53.", []),
        }
        result = enrich_pages_media(pages, llm, page_radius=1)
        enriched = result["1"][1][0]
        assert enriched.enriched_description is not None
        # Context should mention something biology-related
        assert any(term in enriched.enriched_description.lower()
                   for term in ["gene", "expression", "cancer", "tp53", "figure", "chart"])


# ── ContextBuilder tests (real LLM) ────────────────────────────────────────────


class TestContextBuilder:
    @pytest.fixture
    def builder(self, llm):
        return ContextBuilder(llm=llm)

    @pytest.fixture
    def sample_chunk(self):
        return PaperChunk(
            chunk_id="chunk_001",
            text="Machine learning is transforming biology research.",
            paper_id="paper_001",
            page_num=1,
            page_range=(1, 2),
        )

    def test_score_chunk_basic(self, builder, sample_chunk):
        ctx = builder.score_chunk(sample_chunk, "How does ML help biology?")
        assert isinstance(ctx, ScoredContext)
        assert 0 <= ctx.relevance_score <= 10
        assert len(ctx.summary) > 0
        assert ctx.chunk is sample_chunk

    def test_score_chunks_filters_low_scores(self, builder):
        chunks = [
            PaperChunk(chunk_id="c1", text="Quantum physics and particle acceleration.", paper_id="p1"),
            PaperChunk(chunk_id="c2", text="CRISPR-Cas9 gene editing in zebrafish.", paper_id="p1"),
        ]
        results = builder.score_chunks(chunks, "What is CRISPR used for?")
        assert len(results) >= 1  # at least one should match
        for ctx in results:
            assert isinstance(ctx, ScoredContext)
            assert 0 <= ctx.relevance_score <= 10

    def test_generate_answer_no_contexts(self, builder):
        result = builder.generate_answer("test question", [])
        assert "cannot answer" in result.answer.lower()

    def test_generate_answer_with_contexts(self, builder, sample_chunk):
        contexts = [
            ScoredContext(
                summary="ML helps analyze biological data by finding patterns.",
                relevance_score=8,
                chunk=sample_chunk,
            )
        ]
        result = builder.generate_answer("How does ML help biology?", contexts)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert len(result.contexts) == 1

    def test_generate_answer_multiple_contexts(self, builder):
        chunks = [
            PaperChunk(chunk_id="c1", text="Deep learning predicts protein folding.", paper_id="p1"),
            PaperChunk(chunk_id="c2", text="GNNs model molecular interactions.", paper_id="p1"),
        ]
        contexts = builder.score_chunks(chunks, "How do neural networks help biology?")
        result = builder.generate_answer("How do neural networks help biology?", contexts)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        # The answer should reference neural networks or biology
        assert any(term in result.answer.lower()
                   for term in ["neural", "network", "biology", "protein", "molecular"])

    # ── Unit tests for _parse_score_response (no LLM needed) ─────────────────

    def test_parse_score_response(self, builder):
        summary, score = builder._parse_score_response(
            '{"summary": "Key finding.", "relevance_score": 9}'
        )
        assert summary == "Key finding."
        assert score == 9

    def test_parse_score_response_markdown(self, builder):
        summary, score = builder._parse_score_response(
            '```json\n{"summary": "Data shows X.", "relevance_score": 7}\n```'
        )
        assert summary == "Data shows X."
        assert score == 7

    def test_parse_score_fallback(self, builder):
        summary, score = builder._parse_score_response(
            "Some text without JSON.\nRelevance: 6/10"
        )
        assert score >= 0
