# paper-qa-lang

PaperQA 的 LangChain + LangGraph + MCP 重构版本。

## 项目布局

```
paper-qa-lang/
├── mcp/paper-metadata-mcp/   # MCP 服务：论文元数据采集
│   ├── server.py             # MCP 服务入口
│   ├── models.py             # 数据模型 + 缓存 + 合并逻辑
│   └── providers/            # 元数据提供者
│       ├── crossref.py       # Crossref API
│       ├── semantic_scholar.py  # Semantic Scholar API
│       ├── openalex.py       # OpenAlex API
│       └── unpaywall.py      # Unpaywall API
└── src/paper_qa_lang/        # (后续) 核心 RAG 逻辑
```

## paper-metadata-mcp

### 快速启动

```bash
cd mcp/paper-metadata-mcp
python server.py
```

MCP stdio 模式，等待客户端连接。

### 可用工具 (6个)

| 工具 | 功能 | 数据源 |
|------|------|--------|
| `query_by_doi` | DOI 查完整元数据 | 聚合所有源 |
| `query_by_title` | 标题查元数据 | Crossref + OpenAlex |
| `get_bibtex` | BibTeX 引用格式 | Crossref |
| `get_citation_count` | 引用数 | Semantic Scholar |
| `get_open_access_url` | 开放获取 PDF 链接 | Unpaywall + OpenAlex |
| `bulk_query` | 批量查询 | 聚合所有源 |

### 环境变量

| 变量 | 用途 | 必需 |
|------|------|------|
| `LOG_LEVEL` | 日志级别 (默认 WARNING) | 否 |
| `UNPAYWALL_EMAIL` | Unpaywall API 邮箱 | 仅 Unpaywall 需要 |

### 配置到 Claude Desktop

```json
{
  "mcpServers": {
    "paper-metadata": {
      "command": "python",
      "args": ["-m", "server"],
      "env": {
        "UNPAYWALL_EMAIL": "your@email.com"
      }
    }
  }
}
```

### 测试验证

```bash
cd mcp/paper-metadata-mcp
python -c "
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

async def t():
    p = StdioServerParameters(command='python', args=['server.py'])
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            r = await s.call_tool('query_by_doi', {'doi': '10.1038/s41586-023-06236-9'})
            print(r.content[0].text[:300])
asyncio.run(t())
"
```

### 元数据提供者说明

| 提供者 | 需要 API Key | Rate Limit | 最佳用途 |
|--------|-------------|------------|----------|
| Crossref | 可选 (更快) | 无公开限制 (有建议) | BibTeX、基本元数据 |
| Semantic Scholar | 可选 (更快) | 1 req/s (无 key) | 引用数、影响力 |
| OpenAlex | 否 | 10 req/s (公开池) | 摘要、详细元数据 |
| Unpaywall | 否 (需要邮箱) | 100k/天 | OA PDF 链接 |

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **paper-qa-lang** (1096 symbols, 1587 relationships, 24 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/paper-qa-lang/context` | Codebase overview, check index freshness |
| `gitnexus://repo/paper-qa-lang/clusters` | All functional areas |
| `gitnexus://repo/paper-qa-lang/processes` | All execution flows |
| `gitnexus://repo/paper-qa-lang/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
