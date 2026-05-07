# paper-qa-lang

PaperQA 的 LangChain + LangGraph + MCP 重构版本 — 一个面向学术论文的 RAG 问答系统。

## 架构

``` text
┌─────────────────────────────────────────────────┐
│                   前端 (frontend/)                │
│              纯 HTML/CSS/JS 单页应用               │
└─────────────────────┬───────────────────────────┘
                      │ SSE streaming
┌─────────────────────▼───────────────────────────┐
│                 API 层 (api/)                     │
│          FastAPI — 文件上传 / 聊天接口             │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              核心库 (src/paper_qa_lang/)          │
│  ┌──────────┬──────────┬──────────┬──────────┐  │
│  │ parsing  │  store   │  graph   │   chat   │  │
│  │ PDF解析  │ChromaDB  │LangGraph │  对话引擎 │  │
│  │ 分块增强  │ SQLite   │  工作流   │  CLI/API │  │
│  └──────────┴──────────┴──────────┴──────────┘  │
└─────────────────────┬───────────────────────────┘
                      │ MCP protocol
┌─────────────────────▼───────────────────────────┐
│              MCP 服务层 (mcp/)                     │
│  ┌──────────────────────┬─────────────────────┐ │
│  │ paper-metadata-mcp   │ paper-download-mcp  │ │
│  │ Crossref / OpenAlex  │ ArXiv / Unpaywall   │ │
│  │ Semantic Scholar     │ Semantic Scholar    │ │
│  └──────────────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## 项目结构

``` text
paper-qa-lang/
├── src/paper_qa_lang/       # 核心库
│   ├── chat/                # 对话引擎 + CLI
│   ├── config/              # 配置系统 (Pydantic Settings)
│   ├── embeddings/          # Embedding 模型 (Qwen)
│   ├── graph/               # LangGraph 工作流 (ingestion / ReAct)
│   ├── helper/              # 工具 (proxy 检测等)
│   ├── ingestion/           # 论文识别与导入
│   ├── models/              # 数据模型 (Paper, Chunk 等)
│   ├── parsing/             # PDF 解析 / 分块 / 上下文增强
│   ├── prompts/             # Prompt 模板
│   ├── store/               # ChromaDB 向量存储 + PaperLibrary
│   └── utils/               # 工具函数
├── mcp/
│   ├── paper-metadata-mcp/  # MCP: 元数据查询 (6 tools)
│   └── paper-download-mcp/  # MCP: 论文下载 (4 tools)
├── api/                     # FastAPI Web 服务
├── frontend/                # Web 前端
└── tests/                   # 测试
```

## 快速开始

### 1. 环境准备

Python >= 3.11

### 2. 安装

```bash
# 核心依赖
pip install -e .

# 完整安装 (含所有功能)
pip install -e ".[all]"

# 按需安装
pip install -e ".[parsing,store]"   # PDF 解析 + 向量存储
pip install -e ".[chat]"            # Web API + CLI
pip install -e ".[llm]"            # LLM 集成
```

### 3. 配置

```bash
cp .env.example .env
```

按需编辑 `.env`，至少配置一个 LLM provider 的 API Key：

```env
ANTHROPIC_API_KEY=sk-your-api-key-here
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
```

支持三个 LLM provider：
| Provider | 环境变量 |
|----------|----------|
| Anthropic (含 DeepSeek 兼容) | `ANTHROPIC_API_KEY` + `ANTHROPIC_BASE_URL` |
| OpenAI (含 Qwen 兼容) | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| Google Gemini | `GOOGLE_API_KEY` |

### 4. 运行

**CLI 对话：**

```bash
python -m paper_qa_lang.chat.cli
```

**Web API：**

```bash
uvicorn api.main:app --reload --port 8000
# 打开 http://localhost:8000
```

**MCP 服务：**

```bash
cd mcp/paper-metadata-mcp
python server.py
```

## MCP 工具

### paper-metadata-mcp (元数据查询)

| 工具 | 功能 | 数据源 |
|------|------|--------|
| `query_by_doi` | DOI 查完整元数据 | 聚合所有源 |
| `query_by_title` | 标题查元数据 | Crossref + OpenAlex |
| `get_bibtex` | BibTeX 引用格式 | Crossref |
| `get_citation_count` | 引用数 | Semantic Scholar |
| `get_open_access_url` | 开放获取 PDF 链接 | Unpaywall + OpenAlex |
| `bulk_query` | 批量查询 | 聚合所有源 |

### paper-download-mcp (论文下载)

| 工具 | 功能 | 数据源 |
|------|------|--------|
| `download_by_doi` | DOI 下载论文 PDF | ArXiv + Unpaywall |
| `download_by_title` | 标题搜索并下载 | ArXiv + Semantic Scholar |
| `check_availability` | 检查可下载来源 | 所有源 |
| `get_download_status` | 查询下载进度 | 本地缓存 |

### 配置到 Claude Desktop

```json
{
  "mcpServers": {
    "paper-metadata": {
      "command": "python",
      "args": ["mcp/paper-metadata-mcp/server.py"],
      "env": {
        "UNPAYWALL_EMAIL": "your@email.com"
      }
    }
  }
}
```

## 模块概览

| 模块 | 文件 | 职责 |
|------|------|------|
| `parsing` | `pdf_parser.py` / `chunking.py` / `enrichment.py` | PDF 文本提取、智能分块、元数据增强 |
| `store` | `chroma_store.py` / `paper_library.py` | ChromaDB 向量存储、SQLite 论文库管理 |
| `graph` | `ingestion.py` / `react.py` | LangGraph 工作流：论文摄入、ReAct 问答 |
| `chat` | `engine.py` / `cli.py` | 对话引擎、CLI 交互界面 |
| `config` | `settings.py` | 统一配置 (Pydantic Settings + .env) |

## 测试

```bash
pip install -e ".[all]"
pytest tests/ -v
```

## 技术栈

- **LLM 框架** — LangChain 1.2 / LangGraph 1.1
- **向量数据库** — ChromaDB
- **PDF 解析** — PyMuPDF
- **Web** — FastAPI + SSE streaming
- **协议** — MCP (Model Context Protocol)
- **配置** — Pydantic Settings + python-dotenv
- **Embedding** — HuggingFace Transformers (BGE / Qwen)
