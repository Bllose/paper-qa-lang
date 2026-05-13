"""Zero-shot question classification using BGE embeddings + terminology exemplars.

Uses category-level Top-K averaging with a required margin between the best
category and the runner-up.  This avoids the problem where a single short
Chinese sentence falsely matches a single exemplar at high cosine similarity
simply because BGE models cluster short conversational text together.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CATEGORY_OTHER = "其他"


class QuestionClassifier:
    """Classify questions by comparing embeddings against pre-computed exemplars.

    Categories:
      - 打招呼/自我介绍  → route to small model
      - AI技术/论文      → route to knowledge-base RAG
      - 其他             → route to main model directly (no search)

    Algorithm:
      1. Compute cosine similarity to every exemplar.
      2. Per category, take the **top-K** similarities and average them.
      3. The best-scoring category must:
         a. Beat ``threshold`` (absolute quality)
         b. Beat the runner-up category by at least ``margin`` (distinctiveness)
      4. If either check fails → ``"其他"``.
    """

    def __init__(
        self,
        embedding_model: Any,
        threshold: float = 0.55,
        margin: float = 0.06,
        top_k: int = 5,
        data_dir: str | None = None,
    ) -> None:
        self._embedding = embedding_model
        self._threshold = threshold
        self._margin = margin
        self._top_k = top_k

        self._data_dir = data_dir or str(
            Path(__file__).resolve().parents[1] / "data"
        )
        # {category: [vector, ...]}
        self._exemplars: dict[str, list[np.ndarray]] = defaultdict(list)
        # {category: [text, ...]}
        self._examples: dict[str, list[str]] = defaultdict(list)
        self._load_exemplars()

    # ── loading ───────────────────────────────────────────────────────────

    def _load_exemplars(self) -> None:
        """Load terminology files and pre-compute embeddings (no query prefix)."""
        for filename, category in [
            ("greetings.json", "打招呼/自我介绍"),
            ("ai_tech.json", "AI技术/论文"),
        ]:
            filepath = Path(self._data_dir) / filename
            if not filepath.exists():
                logger.warning("Terminology file not found: %s", filepath)
                continue
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            examples = data.get("examples", [])
            if not examples:
                continue
            # Batch-encode all exemplars for this category at once
            vecs = self._embedding.embed_documents(examples)
            for text, vec in zip(examples, vecs):
                arr = np.array(vec)
                arr = arr / np.linalg.norm(arr)  # ensure unit vector
                self._exemplars[category].append(arr)
                self._examples[category].append(text)
            logger.info(
                "Loaded %d exemplars for category '%s' from %s",
                len(examples), category, filename,
            )

    # ── classification ────────────────────────────────────────────────────

    def classify(self, text: str) -> tuple[str, float, str]:
        """Classify a question.

        Returns:
            (category, avg_top_k_similarity, top_matched_example)
        """
        # Encode query — same method as exemplars (no query prefix)
        query_vec = np.array(self._embedding.embed_documents([text])[0])
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Per-category: compute similarity to every exemplar, keep top-K
        cat_scores: dict[str, float] = {}
        cat_best_example: dict[str, str] = {}

        for cat in self._exemplars:
            sims = [
                float(np.dot(query_vec, v))
                for v in self._exemplars[cat]
            ]
            sims.sort(reverse=True)
            top = sims[: self._top_k]
            cat_scores[cat] = sum(top) / len(top)

            # Track the single best-matching exemplar for debugging
            best_idx = int(np.argmax(sims))
            cat_best_example[cat] = self._examples[cat][best_idx]

        # Find best and runner-up categories
        ranked = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        best_cat, best_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # Must beat absolute threshold
        if best_score < self._threshold:
            return CATEGORY_OTHER, best_score, cat_best_example[best_cat]

        # Must beat runner-up by margin
        if best_score - runner_up_score < self._margin:
            return CATEGORY_OTHER, best_score, cat_best_example[best_cat]

        return best_cat, best_score, cat_best_example[best_cat]
