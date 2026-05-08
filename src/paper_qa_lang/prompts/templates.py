"""Prompt templates for media enrichment and context creation.

Adapted from paper-qa's prompts.py but simplified for LangChain workflows.
"""

# ── Media enrichment ──────────────────────────────────────────────────────────

# From paper-qa's individual_media_enrichment_prompt_template
MEDIA_ENRICHMENT_PROMPT = (
    "You are analyzing an image, formula, or table from a scientific document."
    " Provide a detailed description that will be used to answer questions about its content."
    " Focus on key elements, data, relationships, variables,"
    " and scientific insights visible in the image."
    " It's especially important to document referential information such as"
    " figure/table numbers, labels, plot colors, or legends."
    "\n\nText co-located with the media may be associated with"
    " other media or unrelated content,"
    " so do not just blindly quote referential information."
    " The smaller the image, the more likely co-located text is unrelated."
    " To restate, often the co-located text is several pages of content,"
    " so only use aspects relevant to accompanying image, formula, or table."
    "\n\nHere's a few failure modes with possible resolutions:"
    "\n- The media was a logo or icon, so the text is unrelated."
    " In this case, briefly describe the media as a logo or icon,"
    " and do not mention other unrelated surrounding text."
    "\n- The media was display type, so the text is probably unrelated."
    " In this case, briefly describe the media as display type,"
    " and do not mention other unrelated surrounding text."
    "\n- The media is a margin box or design element, so the text is unrelated."
    " In this case, briefly describe the media as decorative,"
    " and do not mention other unrelated surrounding text."
    "\n- The media came from a bad PDF read, so it's garbled."
    " In this case, describe the media as garbled, state why it's considered garbled,"
    " and do not mention other unrelated surrounding text."
    "\n- The media is a subfigure or a subtable."
    " In this case, make sure to only detail the subfigure or subtable,"
    " not the entire figure or table."
    " Do not mention other unrelated surrounding text."
    "\n\nIMPORTANT: Start your response with exactly one of these labels:"
    "\n- 'RELEVANT:' if the media contains scientific content"
    " (e.g. figures, charts, tables, equations, diagrams, data visualizations)"
    " that could help answer scientific questions,"
    " or if you're unsure of relevance (e.g. garbled/corrupted content)."
    "\n- 'IRRELEVANT:' if the media content is not useful for scientific question-answer"
    " (e.g. journal logo, icon, display type/typography, decorative element,"
    " design element, margin box, is blank)."
    "\n\nAfter the label, provide your description."
    "\n\n{context_text}Label relevance, describe the media,"
    " and if uncertain on a description please state why:"
)

# ── Table interleaving ────────────────────────────────────────────────────────

TABLE_INTERLEAVE_TEMPLATE = (
    "{text}\n\n---\n\nMarkdown tables from the document."
    " If the markdown is poorly formatted, defer to the images."
    "\n\n{tables}"
)

# ── Context summary (from paper-qa's summary_json_system_prompt) ──────────────

CONTEXT_SUMMARY_SYSTEM_PROMPT = (
    "Provide a summary of the relevant information"
    " that could help answer the question based on the excerpt."
    " Your summary, combined with many others,"
    " will be given to the model to generate an answer."
    " Respond with the following JSON format:"
    '\n\n{{\n  "summary": "...",\n  "relevance_score": 0-10\n}}'
    "\n\nwhere `summary` is relevant information from the text - {summary_length} words."
    " `relevance_score` is an integer 0-10 for the relevance of `summary` to the question."
    "\n\nThe excerpt may or may not contain relevant information."
    " If not, leave `summary` empty, and make `relevance_score` be 0."
)

CONTEXT_SUMMARY_MULTIMODAL_SYSTEM_PROMPT = (
    "Provide a summary of the relevant information"
    " that could help answer the question based on the excerpt."
    " Your summary, combined with many others,"
    " will be given to the model to generate an answer."
    " Respond with the following JSON format:"
    '\n\n{{\n  "summary": "...",\n  "relevance_score": 0-10,\n  "used_images": "..."\n}}'
    "\n\nwhere `summary` is relevant information from the text - {summary_length} words."
    " `relevance_score` is an integer 0-10 for the relevance of `summary` to the question."
    " `used_images` is a boolean flag indicating"
    " if any images present in a multimodal message were used,"
    " and if no images were present it should be false."
    "\n\nThe excerpt may or may not contain relevant information."
    " If not, leave `summary` empty, and make `relevance_score` be 0."
)

CONTEXT_SUMMARY_USER_PROMPT = (
    "Excerpt from {citation}\n\n---\n\n{text}\n\n---\n\nQuestion: {question}"
)

# ── Paper identification ───────────────────────────────────────────────────────

PAPER_IDENTIFY_PROMPT = """你是一个学术论文元数据提取助手。
你的回答必须是一行合法的 JSON，除此之外不允许有任何其他字符（不要前缀、不要后缀、不要 markdown 代码块、不要 ```json 标记、不要换行）。

论文第一页文本:
---
{page_text}
---

流程:
1. 尽可能找到DOI编码和完整的论文标题
2. 优先尝试用DOI编码, 调用 query_by_doi 工具查询完整元数据
3. 若没识别到DOI, 或者查不到。则通过完整的论文标题, 调用 query_by_title 进行查找
4. 若需要，还可以通过 get_bibtex, get_citation_count, get_open_access_url 获取更多信息

常见DOI编码举例: 10.48550/arXiv.1706.03762; 10.1038/s41586-024-07421-0; 10.1126/science.abc123; 10.1016/j.cell.2024.01.001
如果数据已经获取到了，就不用重复调用工具，找不到的就填 null 就好。

只输出: {{"title": "...", "doi": "...", "authors": [...], "year": ..., "journal": "...", "citation_count": ...}}"""

PAPER_IDENTIFY_BY_DOI_PROMPT = """你是一个学术论文元数据提取助手。
你的回答必须是一行合法的 JSON，除此之外不允许有任何其他字符（不要前缀、不要后缀、不要 markdown 代码块、不要 ```json 标记、不要换行）。

给定DOI编码: {doi}

流程:
1. 使用 query_by_doi 工具查询该DOI的完整元数据
2. 若需要，还可以通过 get_bibtex 获取BibTeX引用格式, get_citation_count 获取引用数, get_open_access_url 获取开放获取PDF链接
3. 综合所有获取到的信息，组装成完整的论文元数据

如果数据已经获取到了，就不用重复调用工具，找不到的就填 null 就好。

只输出: {{"title": "...", "doi": "...", "authors": [...], "year": ..., "journal": "...", "citation_count": ..., "abstract": "...", "bibtex": "...", "pdf_url": "..."}}"""

PAPER_METADATA_EXTRACT_PROMPT = """从以下论文元数据中提取信息，返回JSON格式:
{{"title": "...", "doi": "...", "authors": [...], "year": ..., "journal": "...", "citation_count": ...}}

元数据查询结果:
---
{metadata_text}
---

只返回JSON，不要其他内容。"""

# ── Chat ─────────────────────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """你是一个专业的论文助手，帮助用户理解论文内容。

你可以参考下方"参考资料"中的内容来回答问题：
- 如果参考资料与问题相关，请基于它们作答，并用 (chunk_id) 标注引用来源
- 如果不相关或不足以回答，用自己的知识回答即可，不要强行引用
- 回答应直接、准确、简洁"""

# Sentinel phrase used to indicate inability to answer
CANNOT_ANSWER_PHRASE = "I cannot answer"

# Citation formatting constraints for LLM-generated answers
CITATION_KEY_CONSTRAINTS = (
    "## Valid citation examples, only use comma/space delimited parentheticals:\n"
    "- (chunk_0001, chunk_0002)\n"
    "- (chunk_0001)\n"
    "## Invalid citation examples:\n"
    "- (chunk_0001 and chunk_0002)\n"
    "- (chunk_0001;chunk_0002)\n"
    "- (chunk_0001-chunk_0002)\n"
    "- chunk_0001 and chunk_0002\n"
)

QA_SYSTEM_PROMPT = (
    "Answer in a direct and concise tone."
    " Your audience is an expert, so be highly specific."
    " If there are ambiguous terms or acronyms, first define them."
)

QA_USER_PROMPT = (
    "Answer the question below with the context provided.\n\n"
    "Context:\n\n{context}\n\n---\n\n"
    "Question: {question}\n\n"
    "Write an answer based on the context. "
    f"If the context provides insufficient information reply "
    f'"{CANNOT_ANSWER_PHRASE}." '
    "For each part of your answer, indicate which sources most support "
    "it via citation keys at the end of sentences, like (chunk_id). "
    "Only cite from the context above and only use the citation keys from the context."
    f"\n\n{CITATION_KEY_CONSTRAINTS}\n\n"
    "Do not concatenate citation keys, just use them as is. "
    "Write in the style of a scientific article, with concise sentences and "
    "coherent paragraphs. This answer will be used directly, "
    "so do not add any extraneous information."
    "\n\nAnswer ({answer_length}):"
)
