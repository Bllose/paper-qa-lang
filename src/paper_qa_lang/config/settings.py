"""Configuration for paper-qa-lang."""

from __future__ import annotations

import os
from typing import Any

import dotenv
dotenv.load_dotenv()

from pydantic import BaseModel, Field


class EmbeddingSettings(BaseModel):
    """Embedding model configuration."""

    provider: str = Field(
        default="huggingface",
        description="Embedding provider: huggingface, openai, or custom",
    )
    model_name: str = Field(
        default="bge-base-zh-v1.5",
        description="Model name for the embedding provider",
    )
    model_path: str | None = Field(
        default="D:/workplace/models/BAAI/bge-base-zh-v1.5",
        description="Local filesystem path to the model (if not on Hub)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (for providers that require one)",
    )


class ChunkSettings(BaseModel):
    """Text chunking configuration."""

    size: int = Field(default=1000, ge=100, description="Chunk size in characters")
    overlap: int = Field(default=200, ge=0, description="Chunk overlap in characters")


class StoreSettings(BaseModel):
    """Vector store / paper library settings."""

    collection_name: str = Field(default="papers")
    persist_directory: str = Field(default=".chroma", description="Chroma DB path")
    metadata_db_path: str = Field(default=".papers.db", description="SQLite DB path")


class LLMSettings(BaseModel):
    """LLM configuration for enrichment, scoring, and answer generation.

    Resolves credentials in order:
      1. Explicit value set on the field
      2. Provider-specific env var (e.g. ``ANTHROPIC_API_KEY``)
    """

    provider: str = Field(
        default="anthropic",
        description="LLM provider: anthropic, openai, google, or custom",
    )
    model_name: str = Field(
        default="deepseek-v4-flash",
        description="Model name for the LLM provider",
        env="MODEL_NAME",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (reads from env var if not set, e.g. ANTHROPIC_API_KEY)",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom base URL (e.g. https://api.deepseek.com/anthropic)",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=4096, ge=1)

    def get_llm(self) -> Any:
        """Build and return an LLM client based on the provider."""
        api_key = self.api_key
        base_url = self.base_url
        if not api_key:
            api_key = _resolve_api_key(self.provider)
        if not base_url:
            base_url = _resolve_base_url(self.provider)

        if self.provider == "anthropic":
            from langchain.chat_models import init_chat_model
            return init_chat_model(self.model_name,
                                   model_provider="anthropic",
                                   api_key=api_key,
                                   base_url=base_url,
                                   temperature=self.temperature,
                                   max_tokens=self.max_tokens)
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            kw = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if api_key:
                kw["api_key"] = api_key
            if self.base_url:
                kw["base_url"] = self.base_url
            return ChatOpenAI(**kw)
        if self.provider == "google":
            from langchain_google_genai import ChatGoogleGenAI
            kw = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if api_key:
                kw["api_key"] = api_key
            return ChatGoogleGenAI(**kw)
        raise ValueError(f"Unknown LLM provider: {self.provider}")


def _resolve_api_key(provider: str, prefix: str = "") -> str | None:
    """Resolve an API key from environment variables.

    Checks in order:
      1. ``{PREFIX}{PROVIDER}_API_KEY`` (e.g. ``SMALL_ANTHROPIC_API_KEY``)
      2. ``{PROVIDER}_API_KEY`` (standard fallback, only if prefix is set)
    """
    import os

    upper = provider.upper()
    if prefix:
        value = os.environ.get(f"{prefix}{upper}_API_KEY")
        if value:
            return value
    return os.environ.get(f"{upper}_API_KEY")


def _resolve_base_url(provider: str, prefix: str = "") -> str | None:
    """Resolve a base URL from environment variables.

    Checks in order:
      1. ``{PREFIX}{PROVIDER}_BASE_URL`` (e.g. ``SMALL_ANTHROPIC_BASE_URL``)
      2. ``{PROVIDER}_BASE_URL`` (standard fallback, only if prefix is set)
    """
    import os

    upper = provider.upper()
    if prefix:
        value = os.environ.get(f"{prefix}{upper}_BASE_URL")
        if value:
            return value
    return os.environ.get(f"{upper}_BASE_URL")


class SmallChatSettings(BaseModel):
    """Small / cheap chat model configuration for simple responses (greetings etc.)."""

    provider: str = Field(
        default="anthropic",
        description="LLM provider: anthropic, openai, google, or custom",
    )
    model_name: str = Field(
        default="deepseek-v4-flash",
        description="Model name for the small chat model",
        env="SMALL_MODEL_NAME",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (reads from env var if not set)",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom base URL",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1024, ge=1)

    def getSmallChatModel(self) -> Any:
        """Build and return a small/cheap LLM client.

        Resolves credentials in order:
          1. Explicit value set on the field
          2. ``SMALL_{PROVIDER}_API_KEY`` / ``SMALL_{PROVIDER}_BASE_URL``
          3. ``{PROVIDER}_API_KEY`` / ``{PROVIDER}_BASE_URL`` (shared fallback)
        """
        api_key = self.api_key
        base_url = self.base_url
        if not api_key:
            api_key = _resolve_api_key(self.provider, prefix="SMALL_")
        if not base_url:
            base_url = _resolve_base_url(self.provider, prefix="SMALL_")

        if self.provider == "anthropic":
            from langchain.chat_models import init_chat_model
            return init_chat_model(self.model_name,
                                   model_provider="anthropic",
                                   api_key=api_key,
                                   base_url=base_url,
                                   temperature=self.temperature,
                                   max_tokens=self.max_tokens)
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            kw = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if api_key:
                kw["api_key"] = api_key
            if self.base_url:
                kw["base_url"] = self.base_url
            return ChatOpenAI(**kw)
        if self.provider == "google":
            from langchain_google_genai import ChatGoogleGenAI
            kw = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if api_key:
                kw["api_key"] = api_key
            return ChatGoogleGenAI(**kw)
        raise ValueError(f"Unknown LLM provider: {self.provider}")


class ClassifierSettings(BaseModel):
    """Question classification settings."""

    threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum top-K average similarity to accept a category",
    )
    margin: float = Field(
        default=0.06,
        ge=0.0,
        le=1.0,
        description="Best category must beat runner-up by this margin, else → 其他",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top exemplars to average per category",
    )


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    command: str = Field(description="Command to run the MCP server")
    args: list[str] = Field(default_factory=list, description="Arguments for the server")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    cwd: str | None = Field(default=None, description="Working directory")


class MCPSettings(BaseModel):
    """MCP servers configuration — loaded from mcp_servers.json or env."""

    servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="Named MCP servers, e.g. {'paper-metadata': {command: 'python', args: ['-m', 'server']}}",
    )

    @classmethod
    def from_json_file(cls, path: str | os.PathLike) -> MCPSettings:
        """Load MCP settings from a JSON config file (e.g. mcp_servers.json or settings.local.json)."""
        import json
        with open(path) as f:
            data = json.load(f)
        servers = {}
        # Support both "mcpServers" (Claude Desktop format) and "mcp_servers" keys
        mcp_data = data.get("mcpServers", {}) or data.get("mcp_servers", {})
        for name, cfg in mcp_data.items():
            servers[name] = MCPServerConfig(
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                cwd=cfg.get("cwd"),
            )
        return cls(servers=servers)

    @classmethod
    def load_default(cls) -> MCPSettings:
        """Load MCP settings from the project's .claude/settings.local.json."""
        import os
        # __file__ is at src/paper_qa_lang/config/settings.py → project root is 4 levels up
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        default_path = os.path.join(root, ".claude", "settings.local.json")
        if os.path.exists(default_path):
            mcp = cls.from_json_file(default_path)
            # Set default cwd for servers without one so relative paths resolve correctly
            for svr in mcp.servers.values():
                if svr.cwd is None:
                    svr.cwd = root
            return mcp
        return cls(servers={})

    def get_server(self, name: str) -> MCPServerConfig | None:
        return self.servers.get(name)


class Settings(BaseModel):
    """Top-level settings aggregating all sub-configs."""

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    small_chat: SmallChatSettings = Field(default_factory=SmallChatSettings)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    mcp: MCPSettings = Field(default_factory=lambda: MCPSettings.load_default())
