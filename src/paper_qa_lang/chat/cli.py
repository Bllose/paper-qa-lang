"""CLI chat entry point for PaperQA Lang.

Usage:
    python -m paper_qa_lang.chat.cli
    python -m paper_qa_lang.chat.cli --no-color
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from paper_qa_lang.chat.classifier import QuestionClassifier
from paper_qa_lang.chat.engine import ChatEngine
from paper_qa_lang.config.settings import Settings
from paper_qa_lang.embeddings.qwen_embedding import BgeEmbedding
from paper_qa_lang.store.paper_library import PaperLibrary

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.prompt import Prompt

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaperQA Lang CLI Chat")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable rich formatting (plain text only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


async def _chat_loop(engine: ChatEngine, use_rich: bool) -> None:
    """Interactive chat loop."""
    if use_rich and HAS_RICH:
        console = Console()
        console.print(
            "[bold green]PaperQA Chat[/bold green] — "
            "Type your questions. Press Ctrl+C to exit.\n"
        )
        while True:
            try:
                message = Prompt.ask("[bold blue]You[/bold blue]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Bye![/yellow]")
                break

            if not message.strip():
                continue

            # Short "thinking" notice — replaced by first token
            console.print("[dim]Thinking...[/dim]")
            full = ""
            async for event in engine.astream_chat(message):
                if event["type"] == "input_tokens":
                    if event["count"]:
                        console.print(f"\x1b[1A\x1b[2K[dim]上下文 ~{event['count']} tokens[/dim]")
                elif event["type"] == "token":
                    text = event["content"]
                    if not full:
                        console.print(
                            "\x1b[1A\x1b[2K[bold green]Assistant[/bold green] ", end=""
                        )
                    full += text
                    console.print(text, end="")
                elif event["type"] == "usage":
                    console.print(
                        f"\n[dim]本次: input={event['input_tokens']}, "
                        f"output={event['output_tokens']}[/dim]"
                    )
            console.print()  # trailing newline after response
    else:
        # Plain fallback without rich
        print("PaperQA Chat — Type your questions. Press Ctrl+C to exit.\n")
        while True:
            try:
                message = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not message:
                continue

            print("Assistant: ", end="", flush=True)
            async for event in engine.astream_chat(message):
                if event["type"] == "input_tokens":
                    if event["count"]:
                        print(f"\n[上下文 ~{event['count']} tokens]")
                elif event["type"] == "token":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "usage":
                    print(
                        f"\n[本次: input={event['input_tokens']}, "
                        f"output={event['output_tokens']}]"
                    )
            print()


async def _amain(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(message)s",
    )

    settings = Settings()
    lib = PaperLibrary(settings=settings)

    # Set up classifier with BGE embeddings
    emb_model_path = settings.embedding.model_path or "D:/workplace/models/BAAI/bge-base-zh-v1.5"
    embedding_model = BgeEmbedding(model_path=emb_model_path)
    classifier = QuestionClassifier(
        embedding_model=embedding_model,
        threshold=settings.classifier.threshold,
        margin=settings.classifier.margin,
        top_k=settings.classifier.top_k,
    )

    # Set up small chat model for greetings
    small_llm = settings.small_chat.getSmallChatModel()

    engine = ChatEngine(
        paper_library=lib,
        classifier=classifier,
        small_llm=small_llm,
    )

    use_rich = not args.no_color and HAS_RICH
    await _chat_loop(engine, use_rich)


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(_amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
