# paper-qa-lang

PaperQA 的 LangChain + LangGraph + MCP 重构版本 — 一个面向学术论文的 RAG 问答系统。

![web演示](./resources/演示web.gif)

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

## 问题分类与路由

ChatEngine 在收到用户消息后，先通过嵌入式模型进行零样本分类，再根据类别走不同的回答路线。

### 分类架构

```text
用户问题
  → BGE Embedding 编码（与 exemplar 使用同一编码方法，无检索前缀）
  → 与预计算的 exemplar 向量逐一计算余弦相似度
  → 按类别取 Top-K 均值 + 边距判断
  → 三类输出:
      ├─ 打招呼/自我介绍  → 小模型直接回答（不搜知识库，不统计 token）
      ├─ AI技术/论文      → 知识库搜索 + 主模型 RAG 回答
      └─ 其他             → 主模型直接回答（不搜知识库）
```

### 术语表

分类依赖预定义的 exemplar 列表，存储在 `src/paper_qa_lang/data/` 下：

| 文件 | 类别 | 例句数 |
|------|------|--------|
| `greetings.json` | 打招呼/自我介绍 | 28 |
| `ai_tech.json` | AI技术/论文 | 40 |

Agent 启动时加载所有 exemplar，通过 `BgeEmbedding.embed_documents()` 批量预计算向量。运行时对外部输入仅做一次 `embed_documents()` 推理，开销恒定。

### 算法：Top-K 均值 + 边距

单条 exemplar 匹配容易被噪音误导（BGE 空间中短中文对话句天然聚集）。改为类别级判断：

1. **Top-K 均值** — 与每类别相似度最高的 K 条 exemplar 取平均，作为该类别得分
2. **绝对阈值** — 最高类别得分 ≥ `threshold` 才有效
3. **边距检查** — 最佳类别必须领先 runner-up ≥ `margin`，否则归"其他"

```
"今天星期几"  →  greetings Top5均值=0.62  AI_tech Top5均值=0.58
              →  边距 0.04 < margin(0.06)  →  "其他" ✓
```

### 配置参数

```python
# settings.py → ClassifierSettings
threshold: float = 0.55   # 最低 Top-K 均值
margin: float = 0.06      # 必须领先 runner-up 的边距
top_k: int = 5            # 每类别取前 K 条均值
```

### 小模型配置

打招呼/自我介绍走 `SmallChatSettings` 独立配置，凭据解析优先级：

```
显式赋值 → SMALL_{PROVIDER}_API_KEY → {PROVIDER}_API_KEY（回退共享）
```

```bash
# .env 示例
SMALL_MODEL_NAME=gpt-4o-mini
SMALL_OPENAI_API_KEY=sk-xxx      # 不设则回退到 OPENAI_API_KEY
SMALL_OPENAI_BASE_URL=https://api.openai.com/v1
```

### 新增文件

| 文件 | 职责 |
|------|------|
| `chat/classifier.py` | `QuestionClassifier` — 零样本分类，Top-K 均值 + 边距 |
| `data/greetings.json` | 打招呼/自我介绍术语例句 |
| `data/ai_tech.json` | AI 技术/论文术语例句 |
| `config/settings.py` | 新增 `SmallChatSettings`、`ClassifierSettings` |

## 上下文宽度监控

### 数据流

```text
用户输入
  → engine.py: 构建 messages = [system(+检索chunks), ...history, user]
  → llm.get_num_tokens_from_messages(messages)      ← 预计算 input_tokens
  → SSE → 前端即时显示「输入: N tokens」
  → llm.astream(messages)                           ← 流式调用
  → chunk.usage_metadata 捕获最后一个 chunk 的用量    ← 实际 output + input
  → SSE → 前端累加 output，修正 input，显示累计总消耗
```

### 上下文的三部分构成

每条 LLM 调用的 `input_tokens` 由以下三部分拼接而成：

| 来源 | 内容 | 变化规律 |
| ---- | ---- | -------- |
| System Prompt | 基础指令 + 论文检索结果（chunks 需通过余弦相似度阈值 `score_threshold=0.3` 才会纳入） | 每次检索后重新构建 |
| 对话历史 | 历史 user / assistant 消息（受 `max_history` 截断） | 逐轮累积增长 |
| 当前消息 | 用户最新输入 | 每条不同 |

`messages = [SystemMessage(含检索结果), *历史, HumanMessage(当前)]`

### 预计算与修正

`get_num_tokens_from_messages()` 是 LLM provider 的客户端/服务端 tokenizer 估算值。流式结束后，从最后一个 `AIMessageChunk` 的 `usage_metadata` 中提取 LLM 真实计费数据，与预估值对比并自动修正。

- **Anthropic** (`ChatAnthropic`): 调用官方 `messages.count_tokens` API，服务端精确计数，`stream_usage=True` 默认开启
- **OpenAI** (`ChatOpenAI`): 使用 `tiktoken` 客户端估算，`stream_usage` 默认开启

两个关键参数：

| 参数 | 位置 | 用途 |
| ---- | ---- | ---- |
| `stream_usage=True` | LLM 构造参数 | 流式最后一个 chunk 携带 `usage_metadata` |
| `score_threshold=0.3` | `ChatEngine` | 低于此值的 chunks 不纳入上下文 |

### SSE 事件协议

后端 `astream_chat` 产出的结构化事件经 `api/main.py` 透传为 SSE：

| 事件 type | JSON 结构 | 前端行为 |
| --------- | --------- | -------- |
| `input_tokens` | `{"type":"input_tokens", "count": N}` | 刷新 `#input-tokens`，累加入累计输入 |
| `token` | `{"type":"token", "content":"..."}` | 流式拼接 render markdown |
| `usage` | `{"type":"usage", "input_tokens":N, "output_tokens":M}` | 累加 output，修正 input 差值 |

### 页面展示

stats 栏同一行显示：

```html
输入: <strong id="input-tokens">0</strong>  <!-- 最新一次上下文宽度 -->
累计消耗: <strong id="total-tokens">0</strong>  <!-- 会话总消耗 = Σ input + Σ output -->
```

前端维护的状态由 [app.js](frontend/app.js) 的 `ChatApp` 管理：

```text
#lastInputTokens    → 最新一次输入宽度 (每次发送消息时刷新)
#totalInputTokens   → 历史累计输入
#totalOutputTokens  → 历史累计输出
totalConsumedTokens → getter: input + output
```

每次发送新消息时，`input-tokens` 立刻刷新为当次上下文宽度，`total-tokens` 持续累加——用户可同时看到单次开销和会话总计。

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
