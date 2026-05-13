"""ChatEngine — streaming chat with Search-then-Evaluate RAG and question classification."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage

from paper_qa_lang.chat.classifier import QuestionClassifier
from paper_qa_lang.models.types import PaperChunk
from paper_qa_lang.prompts.templates import CHAT_SYSTEM_PROMPT
from paper_qa_lang.store.paper_library import PaperLibrary

logger = logging.getLogger(__name__)

GREETING_SYSTEM_PROMPT = "你是一个友好、专业的论文助手，简洁地回答用户的问题。"


def _extract_content(chunk: Any) -> str:
    """Extract text content from a langchain stream chunk."""
    if hasattr(chunk, "content"):
        content = chunk.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
    return ""


def _format_chunks(chunks: list[PaperChunk], max_chars: int = 600) -> str:
    """Format search results for inclusion in the system prompt."""
    parts = []
    for c in chunks:
        text = c.text[:max_chars]
        parts.append(f"[{c.chunk_id}] {text}")
    return "\n\n".join(parts)


class ChatEngine:
    """Streaming chat engine with automatic knowledge base retrieval and classification.

    Flow for each message:
      1. Classify the question (greetings / AI-tech / other)
      2. Route:
         - 打招呼/自我介绍 → small model directly (no search, no token tracking)
         - AI技术/论文      → search knowledge base + main model (full RAG)
         - 其他             → main model directly (no search, token tracking)
    """

    def __init__(
        self,
        paper_library: PaperLibrary,
        llm: BaseChatModel | None = None,
        k: int = 5,
        system_prompt: str | None = None,
        max_history: int = 20,
        classifier: QuestionClassifier | None = None,
        small_llm: BaseChatModel | None = None,
    ) -> None:
        self.paper_lib = paper_library
        self.llm = llm or paper_library._get_default_llm()
        self.k = k
        self.system_prompt = system_prompt or CHAT_SYSTEM_PROMPT
        self.max_history = max_history
        self.score_threshold: float | None = 0.3
        self.messages: list[BaseMessage] = []
        self.last_usage: dict[str, int] | None = None

        # Classification & routing
        self.classifier = classifier
        self.small_llm = small_llm

    async def astream_chat(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Stream a chat response, yielding structured events.

        Event types:
            {"type": "input_tokens", "count": N}   — estimated input tokens
            {"type": "token", "content": "..."}     — text content chunk
            {"type": "usage", "input_tokens": N, "output_tokens": M}  — actual usage
        """
        # 0. Classify the question
        category = self._classify(message)

        if category == "打招呼/自我介绍":
            async for event in self._stream_direct(
                message, model=self.small_llm or self.llm,
                system_prompt=GREETING_SYSTEM_PROMPT, track_usage=False,
            ):
                yield event
        elif category == "AI技术/论文":
            async for event in self._stream_rag(message):
                yield event
        else:
            async for event in self._stream_direct(
                message, model=self.llm,
                system_prompt=self.system_prompt, track_usage=True,
            ):
                yield event

    # ── classification ──────────────────────────────────────────────────────

    def _classify(self, message: str) -> str:
        """Classify a message. Falls back to 'AI技术/论文' if no classifier."""
        if self.classifier is None:
            return "AI技术/论文"
        category, score, hit = self.classifier.classify(message)
        logger.debug(
            "Classified as '%s' (score=%.3f, hit='%s')", category, score, hit,
        )
        return category

    # ── RAG stream (type 2: AI papers) ──────────────────────────────────────

    async def _stream_rag(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """Original flow: search knowledge base → build system prompt → stream answer."""
        # 1. Search knowledge base (with relevance threshold)
        chunks = self.paper_lib.search(
            query=message, k=self.k, score_threshold=self.score_threshold
        )
        if chunks:
            logger.debug("Retrieved %d chunks for: %.60s", len(chunks), message)

        # 2. Build message list
        system = SystemMessage(content=self._build_system(chunks))
        user = HumanMessage(content=message)
        history = self.messages[-(self.max_history * 2):]
        messages = [system, *history, user]

        # 3. Pre-compute estimated input tokens
        try:
            estimated_input = self.llm.get_num_tokens_from_messages(messages)
            yield {"type": "input_tokens", "count": estimated_input}
        except Exception:
            logger.debug("Failed to estimate input tokens, skipping")
            yield {"type": "input_tokens", "count": 0}

        # 4. Stream response
        full_response = ""
        last_chunk: AIMessageChunk | None = None
        try:
            async for chunk in self.llm.astream(messages):
                last_chunk = chunk
                content = _extract_content(chunk)
                if content:
                    full_response += content
                    yield {"type": "token", "content": content}
        except Exception as e:
            logger.error("Stream error: %s", e)
            yield {"type": "token", "content": f"\n[Error: {e}]"}
            return

        # 5. Emit actual usage from stream metadata
        usage = None
        if last_chunk is not None and hasattr(last_chunk, "usage_metadata"):
            usage = last_chunk.usage_metadata
        if usage:
            self.last_usage = dict(usage)
            yield {
                "type": "usage",
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }

        # 6. Save conversation history
        self.messages.append(user)
        self.messages.append(AIMessage(content=full_response))

    # ── Direct stream (type 1: greetings / type 3: other) ───────────────────

    async def _stream_direct(
        self, message: str, *,
        model: BaseChatModel,
        system_prompt: str,
        track_usage: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a direct answer from a model without searching the knowledge base."""
        system = SystemMessage(content=system_prompt)
        user = HumanMessage(content=message)
        history = self.messages[-(self.max_history * 2):]
        messages = [system, *history, user]

        # Estimate input tokens (only when tracking)
        if track_usage:
            try:
                estimated_input = model.get_num_tokens_from_messages(messages)
                yield {"type": "input_tokens", "count": estimated_input}
            except Exception:
                logger.debug("Failed to estimate input tokens, skipping")
                yield {"type": "input_tokens", "count": 0}

        # Stream response
        full_response = ""
        last_chunk: AIMessageChunk | None = None
        try:
            async for chunk in model.astream(messages):
                last_chunk = chunk
                content = _extract_content(chunk)
                if content:
                    full_response += content
                    yield {"type": "token", "content": content}
        except Exception as e:
            logger.error("Stream error: %s", e)
            yield {"type": "token", "content": f"\n[Error: {e}]"}
            return

        # Emit actual usage (only when tracking)
        if track_usage:
            usage = None
            if last_chunk is not None and hasattr(last_chunk, "usage_metadata"):
                usage = last_chunk.usage_metadata
            if usage:
                self.last_usage = dict(usage)
                yield {
                    "type": "usage",
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                }

        # Save conversation history
        self.messages.append(user)
        self.messages.append(AIMessage(content=full_response))

    # ── system prompt builder ───────────────────────────────────────────────

    def _build_system(self, chunks: list[PaperChunk]) -> str:
        """Build the system prompt, optionally appending search results."""
        prompt = self.system_prompt
        if chunks:
            context = _format_chunks(chunks)
            prompt += f"\n\n## 参考资料\n{context}"
        return prompt
