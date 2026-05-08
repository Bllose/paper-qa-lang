"""Paper identification — extract metadata from a PDF via LLM + MCP.

Moved from ``models/types.py`` into its own module during the ``ingest()``
unified-entry-point refactor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from random import Random
from typing import Any
from uuid import UUID

from langchain_core.messages import HumanMessage

from paper_qa_lang.config.settings import Settings

logger = logging.getLogger(__name__)


async def _load_mcp_tools(server_name: str, settings: Settings) -> list[Any]:
    """Load tools from a named MCP server via langchain-mcp-adapters."""
    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError:
        logger.warning(
            "langchain-mcp-adapters not installed. "
            "Install with: pip install langchain-mcp-adapters"
        )
        return []

    server_cfg = settings.mcp.get_server(server_name)
    if not server_cfg:
        logger.warning("MCP server '%s' not found in settings", server_name)
        return []

    connection: dict[str, Any] = {
        "transport": "stdio",
        "command": server_cfg.command,
        "args": server_cfg.args,
    }
    if server_cfg.cwd:
        connection["cwd"] = server_cfg.cwd
    if server_cfg.env:
        connection["env"] = server_cfg.env

    try:
        tools = await load_mcp_tools(None, connection=connection)
        logger.info(
            "Loaded %d tools from MCP server '%s': %s",
            len(tools), server_name, [t.name for t in tools],
        )
        return tools
    except Exception as e:
        logger.warning("Failed to load MCP server '%s': %s", server_name, e)
        return []


async def paper_from_pdf(path: str | os.PathLike) -> "Paper":
    """Identify a paper from a PDF file and return a Paper object.

    Flow:
    1. Read first page text from the PDF
    2. Use a ReAct Agent Graph to let the LLM identify the title and
       query paper-metadata-mcp tools for metadata
    3. Assemble and return a ``Paper`` object
    """
    # Avoid circular import: Paper is defined in models.types
    from paper_qa_lang.models.types import Paper

    path = str(path)

    # Step 1: Parse first page
    from paper_qa_lang.parsing.pdf_parser import parse_pdf_to_pages

    pages = parse_pdf_to_pages(path, page_range=(1, 1), parse_media=False)
    page_text = next(iter(pages.values()), ("", []))[0] if pages else ""

    # Step 2: Setup LLM and MCP tools
    settings = Settings()
    llm = settings.llm.get_llm()
    mcp_tools = await _load_mcp_tools("paper-metadata", settings)

    from paper_qa_lang.prompts.templates import PAPER_IDENTIFY_PROMPT
    from paper_qa_lang.graph.react import build_react_graph

    # Step 3: Build ReAct graph and run tool-calling loop
    bound_llm = llm.bind_tools(mcp_tools) if mcp_tools else llm
    graph = build_react_graph(bound_llm, mcp_tools)

    result_state = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=PAPER_IDENTIFY_PROMPT.format(
                        page_text=page_text[:4000]
                    )
                )
            ]
        }
    )

    if result_state.get("error"):
        logger.warning("ReAct graph finished with error: %s", result_state["error"])

    # Step 4: Parse final answer → direct dict, then build Paper
    doc_id = re.sub(r"[^\w]", "_", os.path.basename(path).rsplit(".", 1)[0])
    final_msg = (
        result_state["messages"][-1] if result_state.get("messages") else None
    )
    data = _parse_llm_metadata_response(final_msg) if final_msg else {}

    if not data.get("title") or data["title"] == "Unknown Paper":
        data["title"] = _extract_title_fallback(page_text)

    return Paper(**data, doc_id=doc_id, file_location=path)


async def paper_from_doi(doi: str) -> "Paper":
    """Identify a paper from a DOI and return a Paper object.

    Flow:
    1. Use a ReAct Agent Graph to let the LLM query paper-metadata-mcp tools
    2. Parse the LLM's final JSON response
    3. Assemble and return a ``Paper`` object
    """
    from paper_qa_lang.models.types import Paper

    # Step 1: Setup LLM and MCP tools
    settings = Settings()
    llm = settings.llm.get_llm()
    mcp_tools = await _load_mcp_tools("paper-metadata", settings)

    from paper_qa_lang.prompts.templates import PAPER_IDENTIFY_BY_DOI_PROMPT
    from paper_qa_lang.graph.react import build_react_graph

    # Step 2: Build ReAct graph and run tool-calling loop
    bound_llm = llm.bind_tools(mcp_tools) if mcp_tools else llm
    graph = build_react_graph(bound_llm, mcp_tools)

    result_state = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=PAPER_IDENTIFY_BY_DOI_PROMPT.format(doi=doi)
                )
            ]
        }
    )

    if result_state.get("error"):
        logger.warning("ReAct graph finished with error: %s", result_state["error"])

    # Step 3: Parse final answer → dict, then build Paper
    doc_id = re.sub(r"[^\w]", "_", doi)
    final_msg = (
        result_state["messages"][-1] if result_state.get("messages") else None
    )
    data = _parse_llm_metadata_response(final_msg) if final_msg else {}

    if not data.get("title"):
        data["title"] = f"Paper {doi}"

    return Paper(**data, doc_id=doc_id)


def _parse_llm_metadata_response(response: Any) -> dict[str, Any]:
    """Extract paper metadata JSON from an AIMessage response."""
    raw = response.text if hasattr(response, "text") else ""
    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse metadata JSON from: %s", raw[:200])
        return {}


def _extract_title_fallback(page_text: str) -> str:
    """Fallback: extract title from first page using heuristics."""
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]

    for line in lines[:15]:
        if len(line) > 15 and not any(
            kw in line.lower()
            for kw in ["copyright", "doi", "journal", "abstract", "identifier"]
        ):
            line = re.sub(r"^\d+\s*[-–.]\s*", "", line)
            return line

    return "Unknown Paper"
