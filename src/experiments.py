"""Experiment runner module (блок 8).

Перебирает настройки чанков и поиска, строит отдельный индекс для каждого
варианта и сравнивает Top-1 / Hit@K на контрольных вопросах.
"""
from __future__ import annotations

import pandas as pd

from src.chunker import build_chunks
from src.config import EXPERIMENTS
from src.evaluator import SearchEvaluator
from src.retriever import TfidfRetriever


class ExperimentRunner:
    """Класс запуска серии поисковых экспериментов."""

    def __init__(
        self,
        documents: list[dict[str, str]],
        check_questions: list[tuple[str, str]],
        experiments: list[dict[str, object]] | None = None,
    ):
        self.documents = documents
        self.check_questions = check_questions
        self.experiments = experiments or EXPERIMENTS

    def run(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Запустить эксперименты и вернуть сводку и детализацию."""
        rows: list[dict[str, object]] = []
        details: dict[str, pd.DataFrame] = {}

        for experiment in self.experiments:
            records = build_chunks(
                self.documents,
                chunk_size=int(experiment["chunk_size"]),
                overlap=int(experiment["overlap"]),
            )
            retriever = TfidfRetriever(str(experiment["analyzer"])).build(records)
            detail_df, stats = SearchEvaluator.check(
                retriever,
                self.check_questions,
                top_k=int(experiment["top_k"]),
            )

            rows.append(
                {
                    "опыт": experiment["name"],
                    "чанк": int(experiment["chunk_size"]),
                    "overlap": int(experiment["overlap"]),
                    "поиск": experiment["analyzer"],
                    "top_k": int(experiment["top_k"]),
                    "чанков": len(records),
                    "top1": round(stats["top1_accuracy"], 3),
                    "hit_at_k": round(stats["hit_at_k"], 3),
                }
            )
            details[str(experiment["name"])] = detail_df

        return pd.DataFrame(rows), details
