#!/usr/bin/env python3
"""
paper-metadata-mcp — MCP server for academic paper metadata.

Provides tools to query metadata from multiple sources:
  Crossref, Semantic Scholar, OpenAlex, Unpaywall.

Usage:
    uv run --directory mcp/paper-metadata-mcp python server.py

Or add to Claude Desktop config:
    {
        "mcpServers": {
            "paper-metadata": {
                "command": "uv",
                "args": ["run", "--directory", "path/to/mcp/paper-metadata-mcp", "server.py"]
            }
        }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure this file's directory is on sys.path so imports work from any CWD
_self_dir = str(Path(__file__).resolve().parent)
if _self_dir not in sys.path:
    sys.path.insert(0, _self_dir)

import anyio
import httpx
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from models import PaperMetadata, TTLCache, merge_metadata
from providers import crossref, semantic_scholar, openalex, unpaywall

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "WARNING"),
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("paper-metadata-mcp")

# ── Cache ────────────────────────────────────────────────
cache = TTLCache(default_ttl=3600.0)  # 1 hour default
http_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=10),
            headers={
                "User-Agent": (
                    "paper-metadata-mcp/0.1.0"
                    " (mailto:user@example.org)"
                )
            },
        )
    return http_client


# ── MCP Server ───────────────────────────────────────────
server = Server("paper-metadata-mcp")


def _metadata_to_dict(m: PaperMetadata) -> dict[str, Any]:
    """Serialize PaperMetadata to dict, dropping empty values for clean output."""
    return {
        k: v for k, v in m.model_dump().items()
        if v is not None and v != [] and v != {}
    }


def _fmt_result(m: PaperMetadata | None, query_desc: str) -> str:
    """Format PaperMetadata as human-readable text."""
    if m is None:
        return f"No results found for {query_desc}."

    lines = []
    if m.title:
        lines.append(f"Title: {m.title}")
    if m.doi:
        lines.append(f"DOI:   {m.doi}")
    if m.authors:
        alist = m.authors[:8]
        astr = ", ".join(alist)
        if len(m.authors) > 8:
            astr += " et al."
        lines.append(f"Authors: {astr}")
    if m.journal:
        parts = [m.journal]
        if m.volume:
            parts.append(f"vol.{m.volume}")
        if m.issue:
            parts.append(f"no.{m.issue}")
        if m.pages:
            parts.append(f"pp.{m.pages}")
        if m.year:
            parts.append(f"({m.year})")
        lines.append(f"Journal: {' '.join(parts)}")
    elif m.year:
        lines.append(f"Year: {m.year}")
    if m.citation_count is not None:
        lines.append(f"Citations: {m.citation_count}")
    if m.influential_citation_count is not None:
        lines.append(f"Influential citations: {m.influential_citation_count}")
    if m.pdf_url:
        lines.append(f"Open Access PDF: {m.pdf_url}")
    if m.is_open_access is True:
        lines.append("Open Access: Yes")
    if m.abstract:
        ab = m.abstract[:500]
        if len(m.abstract) > 500:
            ab += "..."
        lines.append(f"\nAbstract:\n{ab}")
    if m.sources:
        lines.append(f"\n(Derived from: {', '.join(m.sources)})")
    return "\n".join(lines)


# ── Query orchestration ─────────────────────────────────

async def _query_by_doi(doi: str) -> PaperMetadata | None:
    """Query all providers in parallel by DOI and merge results."""
    cache_key = cache.make_key("aggregate", "doi", doi)
    cached = cache.get(cache_key)
    if cached is not None:
        return PaperMetadata(**cached) if isinstance(cached, dict) else cached

    client = get_client()
    results = await asyncio.gather(
        crossref.query_by_doi(doi, client),
        semantic_scholar.query_by_doi(doi, client),
        openalex.query_by_doi(doi, client),
        unpaywall.query_by_doi(doi, client),
        return_exceptions=True,
    )
    # Filter to valid results only
    valid = [r for r in results if isinstance(r, PaperMetadata)]
    merged = merge_metadata(valid)
    if merged is not None:
        cache.set(cache_key, merged.model_dump())
    return merged


async def _query_by_title(
    title: str, authors: list[str] | None = None
) -> PaperMetadata | None:
    """Query title-searchable providers in parallel and merge results."""
    authors_str = ",".join(authors) if authors else ""
    cache_key = cache.make_key("aggregate", "title", f"{title}|{authors_str}")
    cached = cache.get(cache_key)
    if cached is not None:
        return PaperMetadata(**cached) if isinstance(cached, dict) else cached

    client = get_client()
    results = await asyncio.gather(
        crossref.query_by_title(title, client, authors),
        openalex.query_by_title(title, client, authors),
        return_exceptions=True,
    )
    valid = [r for r in results if isinstance(r, PaperMetadata)]
    merged = merge_metadata(valid)
    if merged is not None:
        cache.set(cache_key, merged.model_dump())
    return merged


# ── Tool handlers ───────────────────────────────────────

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_by_doi",
            description=(
                "Query paper metadata by DOI. Returns title, authors, journal, "
                "citation count, abstract, open access PDF link, and more. "
                "Aggregates data from Crossref, Semantic Scholar, OpenAlex, and Unpaywall."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": "DOI of the paper, e.g. '10.1038/s41586-023-06236-9'",
                    }
                },
                "required": ["doi"],
            },
        ),
        types.Tool(
            name="query_by_title",
            description=(
                "Query paper metadata by title (and optionally authors). "
                "Searches Crossref and OpenAlex for the best match. "
                "Returns the same rich metadata as query_by_doi."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Paper title (can be partial)",
                    },
                    "authors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional author name(s) to help disambiguate",
                    },
                },
                "required": ["title"],
            },
        ),
        types.Tool(
            name="get_bibtex",
            description="Get BibTeX citation entry for a paper by DOI. Uses Crossref.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": "DOI of the paper",
                    }
                },
                "required": ["doi"],
            },
        ),
        types.Tool(
            name="get_citation_count",
            description="Get citation count for a paper by DOI. Uses Semantic Scholar.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": "DOI of the paper",
                    }
                },
                "required": ["doi"],
            },
        ),
        types.Tool(
            name="get_open_access_url",
            description=(
                "Get open access PDF URL for a paper by DOI. "
                "Checks Unpaywall and OpenAlex."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "doi": {
                        "type": "string",
                        "description": "DOI of the paper",
                    }
                },
                "required": ["doi"],
            },
        ),
        types.Tool(
            name="bulk_query",
            description=(
                "Query metadata for multiple papers at once. "
                "Each query item should have either 'doi' or 'title'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doi": {
                                    "type": "string",
                                    "description": "DOI to query",
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Title to query",
                                },
                                "authors": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional authors for title query",
                                },
                            },
                        },
                        "description": "List of query items",
                    }
                },
                "required": ["queries"],
            },
        ),
    ]


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    # Dynamic resources are served via read_resource, not listed statically
    return []


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    if uri.startswith("metadata://paper/"):
        doi = uri.removeprefix("metadata://paper/")
        result = await _query_by_doi(doi)
        if result is None:
            raise ValueError(f"Paper not found: {doi}")
        return json.dumps(_metadata_to_dict(result), indent=2, ensure_ascii=False)
    raise ValueError(f"Unknown resource URI: {uri}")


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if arguments is None:
        arguments = {}

    if name == "query_by_doi":
        doi = arguments.get("doi", "").strip()
        if not doi:
            raise ValueError("doi is required")
        result = await _query_by_doi(doi)
        return [types.TextContent(type="text", text=_fmt_result(result, f"DOI {doi}"))]

    if name == "query_by_title":
        title = arguments.get("title", "").strip()
        if not title:
            raise ValueError("title is required")
        authors = arguments.get("authors")
        result = await _query_by_title(title, authors)
        return [
            types.TextContent(type="text", text=_fmt_result(result, f"title {title!r}"))
        ]

    if name == "get_bibtex":
        doi = arguments.get("doi", "").strip()
        if not doi:
            raise ValueError("doi is required")
        client = get_client()
        result = await crossref.query_by_doi(doi, client)
        if result is not None and result.bibtex:
            return [types.TextContent(type="text", text=result.bibtex)]
        return [
            types.TextContent(type="text", text=f"No BibTeX found for DOI {doi}.")
        ]

    if name == "get_citation_count":
        doi = arguments.get("doi", "").strip()
        if not doi:
            raise ValueError("doi is required")
        client = get_client()
        result = await semantic_scholar.query_by_doi(doi, client)
        if result is not None and result.citation_count is not None:
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"Citation count for {doi}: {result.citation_count}"
                        + (
                            f"\nInfluential citations: {result.influential_citation_count}"
                            if result.influential_citation_count is not None
                            else ""
                        )
                    ),
                )
            ]
        return [
            types.TextContent(type="text", text=f"No citation data found for {doi}.")
        ]

    if name == "get_open_access_url":
        doi = arguments.get("doi", "").strip()
        if not doi:
            raise ValueError("doi is required")
        client = get_client()
        results = await asyncio.gather(
            unpaywall.query_by_doi(doi, client),
            openalex.query_by_doi(doi, client),
            return_exceptions=True,
        )
        valid = [r for r in results if isinstance(r, PaperMetadata)]
        merged = merge_metadata(valid)
        if merged is not None and merged.pdf_url:
            return [
                types.TextContent(
                    type="text",
                    text=f"Open Access PDF URL: {merged.pdf_url}",
                )
            ]
        return [
            types.TextContent(type="text", text=f"No open access PDF found for {doi}.")
        ]

    if name == "bulk_query":
        queries = arguments.get("queries", [])
        if not queries:
            raise ValueError("queries list is required")
        results: list[dict] = []
        for q in queries:
            if doi := q.get("doi"):
                r = await _query_by_doi(doi)
            elif title := q.get("title"):
                r = await _query_by_title(title, q.get("authors"))
            else:
                continue
            results.append(
                _metadata_to_dict(r) if r is not None else {"error": "not found"}
            )
        return [
            types.TextContent(
                type="text",
                text=json.dumps(results, indent=2, ensure_ascii=False),
            )
        ]

    raise ValueError(f"Unknown tool: {name}")


# ── Main ─────────────────────────────────────────────────


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("paper-metadata-mcp server started")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="paper-metadata-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    anyio.run(main)
