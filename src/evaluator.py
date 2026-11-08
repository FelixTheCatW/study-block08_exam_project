"""Retrieval evaluation module (блок 8).

Сравнивает настройки поиска на контрольных вопросах:
- Top-1 accuracy — нужный документ оказался первым;
- Hit@K — нужный документ попал в первые K разных источников.

В большом корпусе один и тот же ответ может находиться в нескольких
документах (конспект, исходный Word, глоссарий). Поэтому `expected`
может быть строкой или списком допустимых источников.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.retriever import TfidfRetriever


class SearchEvaluator:
    """Класс оценки поиска по вопросам с известными источниками."""

    @staticmethod
    def check(
        retriever: TfidfRetriever,
        questions: list[tuple[str, str | Sequence[str]]],
        top_k: int = 3,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Посчитать детализацию и сводные метрики Top-1 / Hit@K."""
        details: list[dict[str, object]] = []
        for question, expected in questions:
            expected_set = (
                {expected} if isinstance(expected, str) else set(expected)
            )
            hits = retriever.search(question, top_k=top_k)
            docs = [str(hit["doc_id"]) for hit in hits]
            details.append(
                {
                    "вопрос": question,
                    "верный документ": ", ".join(sorted(expected_set)),
                    "первый найденный": docs[0] if docs else "нет",
                    "top1": int(bool(docs) and docs[0] in expected_set),
                    "hit_at_k": int(any(doc in expected_set for doc in docs)),
                }
            )

        detail_df = pd.DataFrame(details)
        summary = {
            "top1_accuracy": float(detail_df["top1"].mean()),
            "hit_at_k": float(detail_df["hit_at_k"].mean()),
        }
        return detail_df, summary
