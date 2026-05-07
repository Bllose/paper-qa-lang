"""Context builder — multimodal evidence scoring and answer assembly.

After vector retrieval, each ``PaperChunk`` (potentially with ``ParsedMedia``) is
sent to an LLM for relevance scoring and summarization. Media images are sent
as multimodal messages. Tables are interleaved as markdown text.

This mirrors paper-qa's ``_map_fxn_summary()`` but adapted for LangChain's
``BaseChatModel`` interface and our data models.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from paper_qa_lang.models.types import Paper, PaperChunk
from paper_qa_lang.utils import llm_content_text
from paper_qa_lang.prompts.templates import (
    CONTEXT_SUMMARY_MULTIMODAL_SYSTEM_PROMPT,
    CONTEXT_SUMMARY_SYSTEM_PROMPT,
    CONTEXT_SUMMARY_USER_PROMPT,
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT,
    TABLE_INTERLEAVE_TEMPLATE,
)

logger = logging.getLogger(__name__)


@dataclass
class ScoredContext:
    """A scored evidence context, mirroring paper-qa's ``Context``."""

    summary: str
    relevance_score: int  # 0-10
    chunk: PaperChunk
    paper: Paper | None = None
    used_images: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    """Result from the answer generation step."""

    answer: str
    contexts: list[ScoredContext]
    token_counts: dict[str, list[int]] = field(default_factory=dict)


class ContextBuilder:
    """Build scored contexts from retrieved chunks, then generate answers.

    Two-phase pipeline:

    1. **Evidence scoring**: each chunk (with media) → LLM → ``ScoredContext``
       (summary + relevance_score). Images sent as multimodal messages.

    2. **Answer generation**: scored contexts → prompt → final answer.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        summary_length: int = 150,
        answer_length: int = 500,
    ) -> None:
        self._llm = llm
        self._summary_length = summary_length
        self._answer_length = answer_length

    # ── Phase 1: Evidence scoring ─────────────────────────────────────────

    def _prepare_score_messages(
        self,
        chunk: PaperChunk,
        question: str,
        paper: Paper | None = None,
    ) -> tuple[list, bool]:
        """Build the message list for scoring a chunk.

        Returns (messages, used_images).
        Shared between sync ``score_chunk`` and async ``score_chunks_async``.
        """
        # Deduplicate media by deterministic ID
        seen_ids = set()
        deduped_media = []
        for m in chunk.media:
            mid = str(m.to_id())
            if mid not in seen_ids:
                seen_ids.add(mid)
                deduped_media.append(m)
        unique_media = deduped_media

        # Build text with table interleaving
        table_texts = [
            m.text for m in unique_media if m.media_type == "table" and m.text
        ]
        display_text = chunk.text
        if table_texts:
            display_text = TABLE_INTERLEAVE_TEMPLATE.format(
                text=display_text,
                tables="\n\n".join(table_texts),
            )

        citation = f"{chunk.chunk_id}: {paper.title if paper else 'Unknown'}"
        user_content = CONTEXT_SUMMARY_USER_PROMPT.format(
            citation=citation,
            text=display_text,
            question=question,
        )

        has_images = any(
            m.data and m.media_type != "table" for m in unique_media
        )

        if has_images and self._supports_images():
            system_prompt = CONTEXT_SUMMARY_MULTIMODAL_SYSTEM_PROMPT.format(
                summary_length=self._summary_length,
            )
            messages = self._build_multimodal_messages(
                system_prompt, user_content, unique_media
            )
            used_images = True
        else:
            system_prompt = CONTEXT_SUMMARY_SYSTEM_PROMPT.format(
                summary_length=self._summary_length,
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
            used_images = False

        return messages, used_images

    def score_chunk(
        self,
        chunk: PaperChunk,
        question: str,
        paper: Paper | None = None,
    ) -> ScoredContext:
        """Score a single chunk for relevance to a question.

        If the chunk has media (images/tables), they are sent as multimodal
        content. Tables are interleaved as markdown text.
        """
        messages, used_images = self._prepare_score_messages(
            chunk, question, paper
        )
        response = self._llm.invoke(messages)
        raw = llm_content_text(response.content)
        summary, score = self._parse_score_response(raw)

        return ScoredContext(
            summary=summary,
            relevance_score=score,
            chunk=chunk,
            paper=paper,
            used_images=used_images,
        )

    def score_chunks(
        self,
        chunks: list[PaperChunk],
        question: str,
        papers: dict[str, Paper] | None = None,
    ) -> list[ScoredContext]:
        """Score multiple chunks sequentially, filtering out irrelevant (score <= 0)."""
        contexts: list[ScoredContext] = []
        for chunk in chunks:
            paper = (papers or {}).get(chunk.paper_id)
            ctx = self.score_chunk(chunk, question, paper=paper)
            if ctx.relevance_score > 0:
                contexts.append(ctx)
        contexts.sort(key=lambda c: c.relevance_score, reverse=True)
        logger.info(
            "Scored %d chunks, kept %d relevant",
            len(chunks), len(contexts),
        )
        return contexts

    async def score_chunks_async(
        self,
        chunks: list[PaperChunk],
        question: str,
        papers: dict[str, Paper] | None = None,
        max_concurrency: int = 5,
    ) -> list[ScoredContext]:
        """Score multiple chunks concurrently using ``ainvoke``.

        Args:
            chunks: Chunks to score.
            question: The user's question.
            papers: Lookup dict ``{paper_id: Paper}``.
            max_concurrency: Max concurrent LLM calls (default 5).

        Returns:
            Scored contexts sorted by relevance (highest first), with
            irrelevant (score <= 0) entries filtered out.
        """
        import asyncio
        from functools import partial

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _score_one(chunk: PaperChunk) -> ScoredContext:
            paper = (papers or {}).get(chunk.paper_id)
            messages, used_images = self._prepare_score_messages(
                chunk, question, paper
            )
            async with semaphore:
                response = await self._llm.ainvoke(messages)
            raw = llm_content_text(response.content)
            summary, score = self._parse_score_response(raw)
            return ScoredContext(
                summary=summary,
                relevance_score=score,
                chunk=chunk,
                paper=paper,
                used_images=used_images,
            )

        results = await asyncio.gather(
            *[_score_one(c) for c in chunks]
        )
        contexts = [r for r in results if r.relevance_score > 0]
        contexts.sort(key=lambda c: c.relevance_score, reverse=True)
        logger.info(
            "Scored %d chunks async, kept %d relevant",
            len(chunks), len(contexts),
        )
        return contexts

    # ── Phase 2: Answer generation ────────────────────────────────────────

    def generate_answer(
        self,
        question: str,
        contexts: list[ScoredContext],
    ) -> AnswerResult:
        """Generate a final answer from scored contexts."""
        if not contexts:
            return AnswerResult(
                answer="I cannot answer this question due to insufficient information.",
                contexts=[],
            )

        context_str = "\n\n".join(
            f"{c.chunk.chunk_id}: {c.summary}"
            + (f"\nFrom: {c.paper.title}" if c.paper and c.paper.title else "")
            for c in contexts
        )

        user_prompt = QA_USER_PROMPT.format(
            context=context_str,
            question=question,
            answer_length=self._answer_length,
        )

        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self._llm.invoke(messages)
        answer = response.content if isinstance(response.content, str) else ""

        return AnswerResult(answer=answer, contexts=contexts)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _supports_images(self) -> bool:
        """Check if the LLM supports multimodal (image) messages.

        LangChain's BaseChatModel doesn't have a standard way to check this,
        so we try-catch during invocation. For now, assume modern models
        support it and fall back on invocation error.
        """
        # We'll just try and catch — most modern LLM providers support images
        return True

    def _build_multimodal_messages(
        self,
        system_prompt: str,
        user_text: str,
        media_list: list,
    ) -> list:
        """Build messages with inline images for multimodal LLMs."""
        content: list[dict[str, Any]] = []

        # Add non-table images
        for m in media_list:
            if m.data and m.media_type != "table":
                b64 = base64.b64encode(m.data).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        # Add text
        content.append({"type": "text", "text": user_text})

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

    def _parse_score_response(self, response: str) -> tuple[str, int]:
        """Parse LLM response into (summary, relevance_score).

        Tries JSON first (```json {summary, relevance_score}), falls back
        to extracting score from the last line.
        """
        text = response.strip()

        # Try JSON extraction
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
        )
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                summary = data.get("summary", text)
                score = int(data.get("relevance_score", 5))
                return summary, max(0, min(10, score))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Try bare JSON (no markdown fence)
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                data = json.loads(text[brace_start : brace_end + 1])
                summary = data.get("summary", text)
                score = int(data.get("relevance_score", 5))
                return summary, max(0, min(10, score))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: last line contains score
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        last_line = lines[-1] if lines else ""
        score_match = re.search(r"(\d+)\s*/?\s*10", last_line)
        if score_match:
            score = int(score_match.group(1))
            summary = "\n".join(lines[:-1])
            return summary, max(0, min(10, score))

        # Final fallback
        return text, 5
