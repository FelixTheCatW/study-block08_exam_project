"""RAG assistant module (блок 8).

Высокоуровневый помощник: строит TF-IDF индекс по базе знаний и отвечает
на вопрос цитатами с указанием источников.
"""
from __future__ import annotations

from src.answerer import EvidenceAnswerer
from src.chunker import build_chunks
from src.config import CHUNK_SIZE, OVERLAP, SEARCH_METHOD, TOP_K
from src.retriever import TfidfRetriever


class RagAssistant:
    """Консольный помощник по документам (поиск + подтверждённые цитаты)."""

    def __init__(
        self,
        documents: list[dict[str, str]],
        chunk_size: int = CHUNK_SIZE,
        overlap: int = OVERLAP,
        search_method: str = SEARCH_METHOD,
        top_k: int = TOP_K,
    ):
        self.documents = documents
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.records = build_chunks(documents, chunk_size, overlap)
        self.retriever = TfidfRetriever(search_method).build(self.records)
        self.answerer = EvidenceAnswerer()

    def search(self, question: str, top_k: int | None = None) -> list[dict[str, object]]:
        """Найти источники по вопросу."""
        return self.retriever.search(question, top_k=top_k or self.top_k)

    def ask(self, question: str, min_similarity: float | None = None) -> dict[str, object]:
        """Найти источники и вернуть ответ с цитатами."""
        found = self.search(question)
        answer = self.answerer.answer(
            question,
            found,
            self.retriever,
            min_similarity=min_similarity,
        )
        return {
            "question": question,
            "found": found,
            "answer": answer,
            "best_score": float(found[0]["score"]) if found else 0.0,
        }

    def format_result(self, result: dict[str, object]) -> str:
        """Красиво отформатировать результат для консоли."""
        lines = [f"Вопрос: {result['question']}"]
        found = result["found"]
        if found:
            lines.append("Найденные источники:")
            for number, item in enumerate(found, 1):
                lines.append(
                    f"  {number}. {item['doc_id']} — {item['title']} "
                    f"(сходство {float(item['score']):.3f})"
                )
        lines.append("")
        lines.append(str(result["answer"]))
        return "\n".join(lines)
