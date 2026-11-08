"""Evidence answer module (блок 8).

Собирает ответ из точных предложений найденных источников. Это не
генерация нейросетью: смысл цитат не меняется, а рядом указывается
документ. Слабые совпадения отбрасываются, чтобы не выдумывать ответ.
"""
from __future__ import annotations

import re

import numpy as np

from src.config import MAX_SENTENCES, MIN_SIMILARITY, RELATIVE_SIMILARITY, SENTENCE_MIN_SIMILARITY
from src.retriever import TfidfRetriever

NO_ANSWER = "Не нашёл подтверждённого ответа в учебных документах."


class EvidenceAnswerer:
    """Класс ответа подтверждёнными выдержками с ссылками на источники."""

    def __init__(
        self,
        min_similarity: float = MIN_SIMILARITY,
        sentence_min_similarity: float = SENTENCE_MIN_SIMILARITY,
        relative_similarity: float = RELATIVE_SIMILARITY,
        max_sentences: int = MAX_SENTENCES,
    ):
        self.min_similarity = min_similarity
        self.sentence_min_similarity = sentence_min_similarity
        self.relative_similarity = relative_similarity
        self.max_sentences = max_sentences

    def answer(
        self,
        question: str,
        found: list[dict[str, object]],
        retriever: TfidfRetriever,
        min_similarity: float | None = None,
    ) -> str:
        """Вернуть цитаты с источниками или честное сообщение об отказе."""
        threshold = self.min_similarity if min_similarity is None else min_similarity
        if not found or float(found[0]["score"]) < threshold:
            return NO_ANSWER

        candidates: list[tuple[float, str, str, str]] = []
        for result in found:
            text = str(result["text"])
            text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
            sentences = [
                re.sub(r"^[#>*\-\s]+", "", sentence).strip()
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
            ]
            sentences = [sentence for sentence in sentences if len(sentence) > 25]
            if not sentences:
                sentences = [text]

            similarities = retriever.score_sentences(question, sentences)
            best_index = int(np.argmax(similarities))
            candidates.append(
                (
                    float(similarities[best_index]),
                    sentences[best_index],
                    str(result["doc_id"]),
                    str(result["title"]),
                )
            )

        candidates.sort(key=lambda item: -item[0])
        if not candidates:
            return NO_ANSWER

        strongest = candidates[0][0]
        lines: list[str] = []
        for score, sentence, doc_id, title in candidates[: self.max_sentences]:
            if (
                score > self.sentence_min_similarity
                and score >= self.relative_similarity * strongest
            ):
                lines.append(f"• {sentence} [Источник: {doc_id}, «{title}»]")

        return "\n".join(lines) if lines else NO_ANSWER
