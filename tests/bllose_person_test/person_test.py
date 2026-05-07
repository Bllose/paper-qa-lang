import logging
import os
import dotenv
import asyncio

from paper_qa_lang.graph.ingestion import ingest_paper
from paper_qa_lang.store.paper_library import PaperLibrary
from paper_qa_lang.models.types import Paper, paper_from_pdf
from langchain.chat_models import init_chat_model
from paper_qa_lang.store.paper_library import PaperLibrary

dotenv.load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(message)s")

path=r"D:/workplace/Bllose/Papers/AI/Ungrouped/2602.21548/2602.21548v2.pdf"

# thePaper = None
#
# async def main():
#     thePaper = await paper_from_pdf(path)
#     print(thePaper)
#
# asyncio.run(main())

curPaper = Paper(
    doc_id='2602_21548v2',
    doi='10.48550/arXiv.2602.21548',
    title='DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference',
    authors=['Yongtong Wu', 'Shaoyuan Chen', 'Yinmin Zhong', 'Rilin Huang', 'Yixuan Tan', 'Wentao Zhang', 'Liyue Zhang', 'Shangyan Zhou', 'Yuxuan Liu', 'Shunfeng Zhou', 'Mingxing Zhang', 'Xin Jin', 'Panpan Huang'],
    year=2026,
    journal='arXiv preprint cs.DC',
    citation_count=None,
    pdf_url=None,
    file_location='D:/workplace/Bllose/Papers/AI/Ungrouped/2602.21548/2602.21548v2/2602.21548v2_1.pdf',
)

paperLib = PaperLibrary()

# llm = init_chat_model(
#     os.getenv("MODEL_ID"),
#     model_provider="anthropic",
#     api_key=os.getenv("ANTHROPIC_API_KEY"),
#     base_url=os.getenv("ANTHROPIC_BASE_URL"),
#     temperature=0,
#     max_tokens=4096,
# )
async def main():
    chunks_num = await paperLib.ingest(path='D:/workplace/Bllose/Papers/AI/Ungrouped/2602.21548/2602.21548v2.pdf')
    print(f"Stored {chunks_num} chunks")

async def query():
    # query = "在多轮、长上下文的智能体工作负载下，流行的预填充-解码（P-D）分离架构中，预填充引擎的存储网卡带宽被占满，而解码引擎的存储网卡却处于空闲状态，这种问题如何解决"
    query = "今天星期几"
    results = await paperLib.query(query, k=5, max_contexts=2)
    for i, context in enumerate(results.contexts, 1):
        print(f"Result {i}:")
        print(f"Chunk ID: {context.chunk.chunk_id}")
        print(f"Summary: {context.summary}")
        print(f"Metadata: {context.metadata}")
        print("-" * 20)

# asyncio.run(main())
asyncio.run(query())