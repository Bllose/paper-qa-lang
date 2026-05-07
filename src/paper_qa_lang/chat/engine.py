"""ChatEngine — streaming chat with Search-then-Evaluate RAG."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from paper_qa_lang.models.types import PaperChunk
from paper_qa_lang.prompts.templates import CHAT_SYSTEM_PROMPT
from paper_qa_lang.store.paper_library import PaperLibrary

logger = logging.getLogger(__name__)


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
    """Streaming chat engine with automatic knowledge base retrieval.

    Flow for each message:
      1. Search the paper knowledge base (always)
      2. Build a prompt with system + history + search results + user message
      3. Stream the LLM response token by token
      4. Save the conversation history

    The LLM decides autonomously whether to use the search results based
    on their relevance to the question — no separate classification step.
    """

    def __init__(
        self,
        paper_library: PaperLibrary,
        llm: BaseChatModel | None = None,
        k: int = 5,
        system_prompt: str | None = None,
        max_history: int = 20,
    ) -> None:
        self.paper_lib = paper_library
        self.llm = llm or paper_library._get_default_llm()
        self.k = k
        self.system_prompt = system_prompt or CHAT_SYSTEM_PROMPT
        self.max_history = max_history
        self.messages: list[BaseMessage] = []

    async def astream_chat(self, message: str) -> AsyncIterator[str]:
        """Stream a chat response, yielding text tokens.

        The knowledge base is always searched and results are included in
        the prompt context. The LLM decides whether to use them.
        """
        # 1. Search knowledge base
        chunks = self.paper_lib.search(query=message, k=self.k)
        if chunks:
            logger.debug("Retrieved %d chunks for: %.60s", len(chunks), message)

        # 2. Build message list
        system = SystemMessage(content=self._build_system(chunks))
        user = HumanMessage(content=message)
        # Keep recent history (each round adds 2 messages: user + ai)
        history = self.messages[-(self.max_history * 2) :]
        messages = [system, *history, user]

        # 3. Stream response
        full_response = ""
        try:
            async for chunk in self.llm.astream(messages):
                content = _extract_content(chunk)
                if content:
                    full_response += content
                    yield content
        except Exception as e:
            logger.error("Stream error: %s", e)
            yield f"\n[Error: {e}]"
            return

        # 4. Save conversation history
        self.messages.append(user)
        self.messages.append(AIMessage(content=full_response))

    def _build_system(self, chunks: list[PaperChunk]) -> str:
        """Build the system prompt, optionally appending search results."""
        prompt = self.system_prompt
        if chunks:
            context = _format_chunks(chunks)
            prompt += f"\n\n## 参考资料\n{context}"
        return prompt
