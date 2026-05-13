"""FastAPI web service for PaperQA Lang streaming chat.

Usage:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from paper_qa_lang.chat.classifier import QuestionClassifier
from paper_qa_lang.chat.engine import ChatEngine
from paper_qa_lang.config.settings import Settings
from paper_qa_lang.embeddings.qwen_embedding import BgeEmbedding
from paper_qa_lang.models.types import Paper
from paper_qa_lang.store.paper_library import PaperLibrary

logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="PaperQA Chat", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons — rebuild on first request
_settings: Settings | None = None
_library: PaperLibrary | None = None
_engine: ChatEngine | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _get_library() -> PaperLibrary:
    global _library
    if _library is None:
        _library = PaperLibrary(settings=_get_settings())
    return _library


def _get_engine() -> ChatEngine:
    global _engine
    if _engine is None:
        settings = _get_settings()

        emb_model_path = settings.embedding.model_path or "D:/workplace/models/BAAI/bge-base-zh-v1.5"
        embedding_model = BgeEmbedding(model_path=emb_model_path)
        classifier = QuestionClassifier(
            embedding_model=embedding_model,
            threshold=settings.classifier.threshold,
            margin=settings.classifier.margin,
            top_k=settings.classifier.top_k,
        )

        small_llm = settings.small_chat.getSmallChatModel()

        _engine = ChatEngine(
            paper_library=_get_library(),
            classifier=classifier,
            small_llm=small_llm,
        )
    return _engine


def _paper_to_dict(p: Paper) -> dict:
    """Convert a Paper to a JSON-safe dict."""
    return {
        "doc_id": p.doc_id,
        "doi": p.doi,
        "title": p.title,
        "authors": p.authors or [],
        "year": p.year,
        "journal": p.journal,
        "citation_count": p.citation_count,
        "pdf_url": p.pdf_url,
        "file_location": p.file_location,
        "source_quality": p.source_quality,
        "is_retracted": p.is_retracted,
        "abstract": p.abstract,
        "bibtex": p.bibtex,
        "publication_date": p.publication_date,
    }


# ── Request/response models ────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str


class DoiIngestRequest(BaseModel):
    doi: str


# ── Endpoints ──────────────────────────────────────────────────────────────────


async def _event_stream(message: str) -> AsyncIterator[bytes]:
    """SSE event stream: yield structured events (input_tokens, token, usage)."""
    engine = _get_engine()
    try:
        async for event in engine.astream_chat(message):
            data = json.dumps(event, ensure_ascii=False)
            yield f"data: {data}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
    except Exception as e:
        logger.error("Chat stream error: %s", e)
        data = json.dumps({"error": str(e)}, ensure_ascii=False)
        yield f"data: {data}\n\n".encode("utf-8")


@app.post("/v1/chat")
async def chat(request: ChatRequest):
    """Streaming chat endpoint.

    Returns SSE events:
        data: {"token": "..."}
        data: [DONE]
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    return StreamingResponse(
        _event_stream(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Paper endpoints ──────────────────────────────────────────────────────────


@app.get("/v1/papers")
async def list_papers():
    """List all papers in the library."""
    lib = _get_library()
    papers = lib.list_papers()
    return [_paper_to_dict(p) for p in papers]


@app.get("/v1/papers/{doc_id}")
async def get_paper(doc_id: str):
    """Get full metadata for a single paper."""
    lib = _get_library()
    paper = lib.query_metadata(doc_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {doc_id}")
    return _paper_to_dict(paper)


@app.post("/v1/papers/ingest/doi")
async def ingest_by_doi(request: DoiIngestRequest):
    """Ingest a paper by DOI — download, identify, parse, and store."""
    if not request.doi.strip():
        raise HTTPException(status_code=400, detail="doi is required")
    lib = _get_library()
    try:
        count = await lib.ingest(doi=request.doi.strip())
        return {"status": "ok", "chunks": count}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error("Ingest DOI error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/papers/ingest/upload")
async def ingest_upload(file: UploadFile):
    """Ingest a paper by uploading a PDF file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    # Save uploaded file to a temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="paperqa_upload_"))
    tmp_path = tmp_dir / (file.filename or "upload.pdf")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    lib = _get_library()
    try:
        count = await lib.ingest(path=str(tmp_path))
        return {"status": "ok", "chunks": count, "path": str(tmp_path)}
    except ValueError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error("Ingest upload error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/papers/{doc_id}")
async def delete_paper(doc_id: str):
    """Delete a paper and all its chunks from the library."""
    lib = _get_library()
    try:
        lib.delete_paper(doc_id)
        return {"status": "ok"}
    except Exception as e:
        logger.error("Delete paper error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Frontend static files ────────────────────────────────────────────────

_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount(
        "/",
        StaticFiles(directory=str(_frontend_dir), html=True),
        name="frontend",
    )
